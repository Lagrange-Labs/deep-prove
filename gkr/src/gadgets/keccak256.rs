#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]

use crate::{
    error::GKRError,
    structs::{Circuit, CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval},
};
use ark_std::rand::{
    Rng, RngCore, SeedableRng,
    rngs::{OsRng, StdRng},
};
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::DenseMultilinearExtension, virtual_poly::ArcMultilinearExtension,
};
use p3_field::FieldAlgebra;
use simple_frontend::structs::CircuitBuilder;
use std::iter;
use sumcheck::util::ceil_log2;
use transcript::{BasicTranscript, Transcript};

const THETA: [(usize, [usize; 5], [usize; 5]); 25] = [
    // format: (
    //    x + y*5,
    //    [(x+4, 0), (x+4, 1),... (x+4, 4)], // input
    //    [(x+1, 0), (x+1, 1), ..., (x+1, 4)] // rotated input
    // )
    (0, [4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
    (1, [0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
    (2, [1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
    (3, [2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
    (4, [3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
    (5, [4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
    (6, [0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
    (7, [1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
    (8, [2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
    (9, [3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
    (10, [4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
    (11, [0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
    (12, [1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
    (13, [2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
    (14, [3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
    (15, [4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
    (16, [0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
    (17, [1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
    (18, [2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
    (19, [3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
    (20, [4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
    (21, [0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
    (22, [1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
    (23, [2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
    (24, [3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
];

const RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

const ROUNDS: usize = 24;

const RC: [u64; ROUNDS] = [
    1u64,
    0x8082u64,
    0x800000000000808au64,
    0x8000000080008000u64,
    0x808bu64,
    0x80000001u64,
    0x8000000080008081u64,
    0x8000000000008009u64,
    0x8au64,
    0x88u64,
    0x80008009u64,
    0x8000000au64,
    0x8000808bu64,
    0x800000000000008bu64,
    0x8000000000008089u64,
    0x8000000000008003u64,
    0x8000000000008002u64,
    0x8000000000000080u64,
    0x800au64,
    0x800000008000000au64,
    0x8000000080008081u64,
    0x8000000000008080u64,
    0x80000001u64,
    0x8000000080008008u64,
];

/// Bits of a word in big-endianness
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Word([usize; 64]);

impl Default for Word {
    fn default() -> Self {
        Self([0; 64])
    }
}

impl Word {
    fn new<E: ExtensionField>(cb: &mut CircuitBuilder<E>) -> Self {
        Self(cb.create_cells(64).try_into().unwrap())
    }

    fn rotate_left(&self, mid: usize) -> Self {
        let mut word = *self;
        word.0.rotate_left(mid);
        word
    }
}

// out = lhs ^ rhs = lhs + rhs - 2 * lhs * rhs
fn xor<E: ExtensionField>(cb: &mut CircuitBuilder<E>, lhs: &Word, rhs: &Word) -> Word {
    let out = Word::new(cb);
    izip!(&out.0, &lhs.0, &rhs.0).for_each(|(out, lhs, rhs)| {
        cb.add(*out, *lhs, E::BaseField::ONE);
        cb.add(*out, *rhs, E::BaseField::ONE);
        cb.mul2(*out, *lhs, *rhs, -E::BaseField::ONE.double());
    });
    out
}

// out = lhs ^ rhs = lhs + rhs - 2 * lhs * rhs
fn relay<E: ExtensionField>(cb: &mut CircuitBuilder<E>, lhs: &Word) -> Word {
    let out = Word::new(cb);
    izip!(&out.0, &lhs.0).for_each(|(out, lhs)| {
        cb.add(*out, *lhs, E::BaseField::ONE);
    });
    out
}

// out = !lhs & rhs
#[allow(dead_code)]
fn not_lhs_and_rhs<E: ExtensionField>(cb: &mut CircuitBuilder<E>, lhs: &Word, rhs: &Word) -> Word {
    let out = Word::new(cb);
    izip!(&out.0, &lhs.0, &rhs.0).for_each(|(out, lhs, rhs)| {
        cb.add(*out, *rhs, E::BaseField::ONE);
        cb.mul2(*out, *lhs, *rhs, -E::BaseField::ONE);
    });
    out
}

// (x0 + x1 + x2) - 2x0x2 - 2x1x2 - 2x0x1 + 4x0x1x2
fn xor3<E: ExtensionField>(cb: &mut CircuitBuilder<E>, words: &[Word; 3]) -> Word {
    let out = Word::new(cb);
    izip!(&out.0, &words[0].0, &words[1].0, &words[2].0).for_each(
        |(out, wire_0, wire_1, wire_2)| {
            // (x0 + x1 + x2)
            cb.add(*out, *wire_0, E::BaseField::ONE);
            cb.add(*out, *wire_1, E::BaseField::ONE);
            cb.add(*out, *wire_2, E::BaseField::ONE);
            // - 2x0x2 - 2x1x2 - 2x0x1
            cb.mul2(*out, *wire_0, *wire_1, -E::BaseField::ONE.double());
            cb.mul2(*out, *wire_0, *wire_2, -E::BaseField::ONE.double());
            cb.mul2(*out, *wire_1, *wire_2, -E::BaseField::ONE.double());
            // 4x0x1x2
            cb.mul3(
                *out,
                *wire_0,
                *wire_1,
                *wire_2,
                E::BaseField::ONE.double().double(),
            );
        },
    );
    out
}

// chi truth table
// | x0 | x1 | x2 | x0 ^ ((not x1) & x2) |
// |----|----|----|----------------------|
// | 0  | 0  | 0  | 0                    |
// | 0  | 0  | 1  | 1                    |
// | 0  | 1  | 0  | 0                    |
// | 0  | 1  | 1  | 0                    |
// | 1  | 0  | 0  | 1                    |
// | 1  | 0  | 1  | 0                    |
// | 1  | 1  | 0  | 1                    |
// | 1  | 1  | 1  | 1                    |
// (1-x0)*(1-x1)*(x2) + x0(1-x1)(1-x2) + x0x1(1-x2) + x0x1x2
// = x2 - x0x2 - x1x2 + x0x1x2 + x0 - x0x1 - x0x2 + x0x1x2 + x0x1 - x0x1x2 + x0x1x2
// = (x0 + x2) - 2x0x2 - x1x2 + 2x0x1x2
fn chi<E: ExtensionField>(cb: &mut CircuitBuilder<E>, words: &[Word; 3]) -> Word {
    let out = Word::new(cb);
    izip!(&out.0, &words[0].0, &words[1].0, &words[2].0).for_each(
        |(out, wire_0, wire_1, wire_2)| {
            // (x0 + x2)
            cb.add(*out, *wire_0, E::BaseField::ONE);
            cb.add(*out, *wire_2, E::BaseField::ONE);
            // - 2x0x2 - x1x2
            cb.mul2(*out, *wire_0, *wire_2, -E::BaseField::ONE.double());
            cb.mul2(*out, *wire_1, *wire_2, -E::BaseField::ONE);
            // 2x0x1x2
            cb.mul3(*out, *wire_0, *wire_1, *wire_2, E::BaseField::ONE.double());
        },
    );
    out
}

// chi_output xor constant
// = chi_output + constant - 2*chi_output*constant
// = c + (x0 + x2) - 2x0x2 - x1x2 + 2x0x1x2 - 2(c*x0 + c*x2 - 2c*x0*x2 - c*x1*x2 + 2*c*x0*x1*x2)
// = x0 + x2 + c - 2*x0*x2 - x1*x2 + 2*x0*x1*x2 - 2*c*x0 - 2*c*x2 + 4*c*x0*x2 + 2*c*x1*x2 -
// 4*c*x0*x1*x2 = x0*(1-2c) + x2*(1-2c) + c + x0*x2*(-2 + 4c) + x1*x2(-1 + 2c) + x0*x1*x2(2 - 4c)
fn chi_and_xor_constant<E: ExtensionField>(
    cb: &mut CircuitBuilder<E>,
    words: &[Word; 3],
    constant: u64,
) -> Word {
    let out = Word::new(cb);
    izip!(
        &out.0,
        &words[0].0,
        &words[1].0,
        &words[2].0,
        iter::successors(Some(constant.reverse_bits()), |constant| {
            Some(constant >> 1)
        })
    )
    .for_each(|(out, wire_0, wire_1, wire_2, constant)| {
        let const_bit = constant & 1;
        // x0*(1-2c) + x2*(1-2c) + c
        if const_bit & 1 == 1 {
            // -x0
            cb.add(*out, *wire_0, -E::BaseField::ONE);
        } else {
            // x0
            cb.add(*out, *wire_0, E::BaseField::ONE);
        };
        if const_bit & 1 == 1 {
            // -x2
            cb.add(*out, *wire_2, -E::BaseField::ONE);
        } else {
            // x2
            cb.add(*out, *wire_2, E::BaseField::ONE);
        };
        cb.add_const(
            *out,
            if const_bit & 1 == 1 {
                E::BaseField::ONE
            } else {
                E::BaseField::ZERO
            },
        );

        // x0*x2*(-2 + 4c) + x1*x2(-1 + 2c)
        if const_bit & 1 == 1 {
            // 2*x0*x2
            cb.mul2(*out, *wire_0, *wire_2, E::BaseField::ONE.double());
        } else {
            // -2*x0*x2
            cb.mul2(*out, *wire_0, *wire_2, -E::BaseField::ONE.double());
        };
        if const_bit & 1 == 1 {
            // x1*x2
            cb.mul2(*out, *wire_1, *wire_2, E::BaseField::ONE);
        } else {
            // -x1*x2
            cb.mul2(*out, *wire_1, *wire_2, -E::BaseField::ONE);
        };

        // x0*x1*x2(2 - 4c)
        if const_bit & 1 == 1 {
            // -2*x0*x1*x2
            cb.mul3(*out, *wire_0, *wire_1, *wire_2, -E::BaseField::ONE.double());
        } else {
            // 2*x0*x1*x2
            cb.mul3(*out, *wire_0, *wire_1, *wire_2, E::BaseField::ONE.double());
        }
    });
    out
}

#[allow(dead_code)]
fn xor2_constant<E: ExtensionField>(
    cb: &mut CircuitBuilder<E>,
    words: &[Word; 2],
    constant: u64,
) -> Word {
    let out = Word::new(cb);

    izip!(
        &out.0,
        &words[0].0,
        &words[1].0,
        iter::successors(Some(constant.reverse_bits()), |constant| {
            Some(constant >> 1)
        })
    )
    .for_each(|(out, wire_0, wire_1, constant)| {
        let const_bit = constant & 1;
        // (x0 + x1 + x2)
        cb.add(*out, *wire_0, E::BaseField::ONE);
        cb.add(*out, *wire_1, E::BaseField::ONE);
        cb.add_const(
            *out,
            if const_bit & 1 == 1 {
                E::BaseField::ONE
            } else {
                E::BaseField::ZERO
            },
        );
        // - 2x0x2 - 2x1x2 - 2x0x1
        if const_bit == 1 {
            cb.add(*out, *wire_0, -E::BaseField::ONE.double());
            cb.add(*out, *wire_1, -E::BaseField::ONE.double());
        }
        cb.mul2(*out, *wire_0, *wire_1, -E::BaseField::ONE.double());

        // 4x0x1x2
        if const_bit == 1 {
            cb.mul2(*out, *wire_0, *wire_1, E::BaseField::ONE.double().double());
        }
    });
    out
}

// TODO: Optimization:
//       - Theta use lookup
//       - Use mul3 to make Chi less layers
pub fn keccak256_circuit<E: ExtensionField>() -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();

    let [mut state, input] = [25 * 64, 17 * 64].map(|n| {
        cb.create_witness_in(n)
            .1
            .chunks(64)
            .map(|word| Word(word.to_vec().try_into().unwrap()))
            .collect_vec()
    });

    // Absorption
    state = izip!(
        state.iter(),
        input.into_iter().map(Some).chain(iter::repeat(None))
    )
    .map(|(state, input)| {
        if let Some(input) = input {
            xor(cb, state, &input)
        } else {
            relay(cb, state)
        }
    })
    .collect_vec();

    // Permutation
    for i in 0..ROUNDS {
        let mut array = [Word::default(); 5];

        // Theta step
        // state[x, y] = state[x, y] XOR state[x+4, 0] XOR state[x+4, 1] XOR state[x+4, 2] XOR
        // state[x+4, 3] XOR state[x+4, 4] XOR state[x+1, 0] XOR state[x+1, 1] XOR
        // state[x+1, 2] XOR state[x+1, 3] XOR state[x+1, 4]
        state = THETA
            .map(|(index, inputs, rotated_input)| {
                let input = state[index];
                let input_words = inputs.map(|index| state[index]);
                let rotated_input_words = rotated_input.map(|index| state[index].rotate_left(1));
                let xor_inputs = iter::once(input)
                    .chain(input_words)
                    .chain(rotated_input_words)
                    .collect::<Vec<_>>();
                assert!(xor_inputs.len() == 11);

                // first layer => reduce size from 11 to 4
                let xor_inputs = xor_inputs
                    .chunks(3)
                    .map(|chunk| {
                        let chunked_inputs = chunk.to_vec();
                        match chunked_inputs.len() {
                            3 => xor3(cb, &chunked_inputs.try_into().unwrap()),
                            2 => xor(cb, &chunked_inputs[0], &chunked_inputs[1]),
                            _ => unreachable!(),
                        }
                    })
                    .collect::<Vec<_>>();
                assert!(xor_inputs.len() == 4);

                // second layer => reduce size from 4 to 2
                let xor_inputs = xor_inputs
                    .chunks(2)
                    .map(|chunk| {
                        let chunked_inputs = chunk.to_vec();
                        assert!(chunked_inputs.len() == 2);
                        xor(cb, &chunked_inputs[0], &chunked_inputs[1])
                    })
                    .collect::<Vec<_>>();
                assert!(xor_inputs.len() == 2);

                // third layer => reduce size from 2 to 1
                let xor_inputs = xor_inputs
                    .chunks(2)
                    .map(|chunk| {
                        let chunked_inputs = chunk.to_vec();
                        assert!(chunked_inputs.len() == 2);
                        xor(cb, &chunked_inputs[0], &chunked_inputs[1])
                    })
                    .collect::<Vec<_>>();

                assert!(xor_inputs.len() == 1);
                xor_inputs[0]
            })
            .to_vec();

        assert!(state.len() == 25);

        // Rho and pi
        let mut last = state[1];
        for x in 0..24 {
            array[0] = state[PI[x]];
            state[PI[x]] = last.rotate_left(RHO[x] as usize);
            last = array[0];
        }

        // Chi + Iota
        for y_step in 0..5 {
            let y = y_step * 5;
            for x in 0..5 {
                array[x] = state[y + x];
            }
            for x in 0..5 {
                if x == 0 && y == 0 {
                    // Chi + Iota
                    state[0] = chi_and_xor_constant(cb, &[array[0], array[1], array[2]], RC[i]);
                } else {
                    // Chi
                    state[y + x] = chi(cb, &[array[x], array[(x + 1) % 5], array[(x + 2) % 5]]);
                }
            }
        }
    }

    // FIXME: If we use the `create_wire_out_from_cells`, the ordering of these cells in wire_out
    //        will be different, so it's duplicating cells to avoid that as a temporary solution.
    // cb.create_wire_out_from_cells(&state.iter().flat_map(|word| word.0).collect_vec());

    let (_, out) = cb.create_witness_out(256);
    izip!(&out, state.iter().flat_map(|word| &word.0))
        .for_each(|(out, state)| cb.add(*out, *state, E::BaseField::ONE));

    cb.configure();
    Circuit::new(cb)
}

pub fn prove_keccak256<E: ExtensionField>(
    instance_num_vars: usize,
    circuit: &Circuit<E>,
    max_thread_id: usize,
) -> Option<(IOPProof<E>, CircuitWitness<E>)> {
    assert!(
        ceil_log2(max_thread_id) <= instance_num_vars,
        "ceil_log2(N) {} > instance_num_vars {}",
        ceil_log2(max_thread_id),
        instance_num_vars
    );
    // Sanity-check
    #[cfg(test)]
    {
        use crate::structs::CircuitWitness;
        use multilinear_extensions::mle::IntoMLE;
        let all_zero: Vec<DenseMultilinearExtension<E>> = vec![
            vec![E::BaseField::ZERO; 25 * 64],
            vec![E::BaseField::ZERO; 17 * 64],
        ]
        .into_iter()
        .map(|wit_in| wit_in.into_mle())
        .collect();
        let all_one = vec![
            vec![E::BaseField::ONE; 25 * 64],
            vec![E::BaseField::ZERO; 17 * 64],
        ]
        .into_iter()
        .map(|wit_in| wit_in.into_mle())
        .collect();
        let mut witness = CircuitWitness::new(circuit, Vec::new());
        witness.add_instance(circuit, all_zero);
        witness.add_instance(circuit, all_one);

        izip!(
            witness.witness_out_ref()[0]
                .get_base_field_vec()
                .chunks(256),
            [[0; 25], [u64::MAX; 25]]
        )
        .for_each(|(wire_out, state)| {
            let output = wire_out[..256]
                .chunks_exact(64)
                .map(|bits| {
                    bits.iter().fold(0, |acc, bit| {
                        (acc << 1) + (*bit == E::BaseField::ONE) as u64
                    })
                })
                .collect_vec();
            let expected = {
                let mut state = state;
                tiny_keccak::keccakf(&mut state);
                state[0..4].to_vec()
            };
            assert_eq!(output, expected)
        });
    }

    let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
    let mut witness = CircuitWitness::new(circuit, Vec::new());
    for _ in 0..1 << instance_num_vars {
        let [rand_state, rand_input] = [25 * 64, 17 * 64].map(|n| {
            let mut data = vec![E::BaseField::ZERO; 1 << ceil_log2(n)];
            data.iter_mut()
                .take(n)
                .for_each(|d| *d = E::BaseField::from_canonical_u64(rng.gen_bool(0.5) as u64));
            data
        });
        witness.add_instance(
            circuit,
            vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    ceil_log2(rand_state.len()),
                    rand_state,
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    ceil_log2(rand_input.len()),
                    rand_input,
                ),
            ],
        );
    }

    let output_mle = &witness.witness_out_ref()[0];

    let mut prover_transcript = BasicTranscript::<E>::new(b"test");
    let output_point = iter::repeat_with(|| {
        prover_transcript
            .get_and_append_challenge(b"output point")
            .elements
    })
    .take(output_mle.num_vars())
    .collect_vec();
    let output_eval = output_mle.evaluate(&output_point);

    let start = std::time::Instant::now();
    let (proof, _) = IOPProverState::prove_parallel(
        circuit,
        &witness,
        vec![],
        vec![PointAndEval::new(output_point, output_eval)],
        max_thread_id,
        &mut prover_transcript,
    );
    println!("{}: {:?}", 1 << instance_num_vars, start.elapsed());
    Some((proof, witness))
}

pub fn verify_keccak256<E: ExtensionField>(
    instance_num_vars: usize,
    output_mle: &ArcMultilinearExtension<E>,
    proof: IOPProof<E>,
    circuit: &Circuit<E>,
) -> Result<GKRInputClaims<E>, GKRError> {
    let mut verifier_transcript = BasicTranscript::<E>::new(b"test");
    let output_point = iter::repeat_with(|| {
        verifier_transcript
            .get_and_append_challenge(b"output point")
            .elements
    })
    .take(output_mle.num_vars())
    .collect_vec();
    let output_eval = output_mle.evaluate(&output_point);
    crate::structs::IOPVerifierState::verify_parallel(
        circuit,
        &[],
        vec![],
        vec![PointAndEval::new(output_point, output_eval)],
        proof,
        instance_num_vars,
        &mut verifier_transcript,
    )
}
