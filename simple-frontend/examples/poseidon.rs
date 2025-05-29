//! Poseidon hash function. This is modified from https://github.com/iden3/circomlib/blob/master/circuits/poseidon.circom.

use ff_ext::GoldilocksExt2;
use mock_constant::{poseidon_c, poseidon_m, poseidon_p, poseidon_s};
use p3_field::FieldAlgebra;
use p3_goldilocks::Goldilocks;
use simple_frontend::structs::{CellId, CircuitBuilder};

// round constant
const N_ROUNDS_P: [usize; 16] = [
    56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68,
];

// template Sigma() {
//     signal input in;
//     signal output out;

//     signal in2;
//     signal in4;

//     in2 <== in*in;
//     in4 <== in2*in2;

//     out <== in4*in;
// }

fn sigma(circuit_builder: &mut CircuitBuilder<GoldilocksExt2>, in_: CellId) -> CellId {
    let in2 = circuit_builder.create_cell();
    let in4 = circuit_builder.create_cell();

    let out = circuit_builder.create_cell();

    let one = Goldilocks::ONE;
    circuit_builder.mul2(in2, in_, in_, one);
    circuit_builder.mul2(in4, in2, in2, one);
    circuit_builder.mul2(out, in4, in_, one);

    out
}

// template Ark(t, C, r) {
//     signal input in[t];
//     signal output out[t];

//     for (var i=0; i<t; i++) {
//         out[i] <== in[i] + C[i + r];
//     }
// }

fn ark(
    circuit_builder: &mut CircuitBuilder<GoldilocksExt2>,
    in_: &[CellId],
    c: &[Goldilocks],
    r: usize,
) -> Vec<CellId> {
    let out = circuit_builder.create_cells(in_.len());

    let one = Goldilocks::ONE;

    for i in 0..in_.len() {
        circuit_builder.add(out[i], in_[i], one);
        circuit_builder.add_const(out[i], c[i + r]);
    }

    out
}

// template Mix(t, M) {
//     signal input in[t];
//     signal output out[t];

//     var lc;
//     for (var i=0; i<t; i++) {
//         lc = 0;
//         for (var j=0; j<t; j++) {
//             lc += M[j][i]*in[j];
//         }
//         out[i] <== lc;
//     }
// }

fn mix(
    circuit_builder: &mut CircuitBuilder<GoldilocksExt2>,
    in_: &[CellId],
    m: &[&[Goldilocks]],
) -> Vec<CellId> {
    let out = circuit_builder.create_cells(in_.len());

    for (i, out_elem) in out.iter().enumerate() {
        for (j, in_elem) in in_.iter().enumerate() {
            circuit_builder.add(*out_elem, *in_elem, m[j][i]);
        }
    }

    out
}

// template MixLast(t, M, s) {
//     signal input in[t];
//     signal output out;

//     var lc = 0;
//     for (var j=0; j<t; j++) {
//         lc += M[j][s]*in[j];
//     }
//     out <== lc;
// }

fn mix_last(
    circuit_builder: &mut CircuitBuilder<GoldilocksExt2>,
    in_: &[CellId],
    m: &[&[Goldilocks]],
    s: usize,
) -> CellId {
    let out = circuit_builder.create_cell();

    for j in 0..in_.len() {
        circuit_builder.add(out, in_[j], m[j][s]);
    }

    out
}

// template MixS(t, S, r) {
//     signal input in[t];
//     signal output out[t];

//     var lc = 0;
//     for (var i=0; i<t; i++) {
//         lc += S[(t*2-1)*r+i]*in[i];
//     }
//     out[0] <== lc;
//     for (var i=1; i<t; i++) {
//         out[i] <== in[i] +  in[0] * S[(t*2-1)*r + t + i -1];
//     }
// }

fn mix_s(
    circuit_builder: &mut CircuitBuilder<GoldilocksExt2>,
    in_: &[CellId],
    s: &[Goldilocks],
    r: usize,
) -> Vec<CellId> {
    let t = in_.len();
    let out = circuit_builder.create_cells(t);

    let one = Goldilocks::ONE;

    for i in 0..in_.len() {
        circuit_builder.add(out[0], in_[i], s[(t * 2 - 1) * r + i]);
    }

    for i in 1..t {
        circuit_builder.add(out[i], in_[0], s[(t * 2 - 1) * r + t + i - 1]);
        circuit_builder.add(out[i], in_[i], one);
    }

    out
}

fn poseidon_ex(
    circuit_builder: &mut CircuitBuilder<GoldilocksExt2>,
    n_outs: usize,
    inputs: &[CellId],
    initial_state: CellId,
) -> Vec<CellId> {
    //     signal input inputs[nInputs];
    //     signal input initialState;
    //     signal output out[nOuts];

    //     var N_ROUNDS_P[16] = [56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68];
    //     var t = nInputs + 1;
    //     var nRoundsF = 8;
    //     var nRoundsP = N_ROUNDS_P[t - 2];
    //     var C[t*nRoundsF + nRoundsP] = POSEIDON_C(t);
    //     var S[  N_ROUNDS_P[t-2]  *  (t*2-1)  ]  = POSEIDON_S(t);
    //     var M[t][t] = POSEIDON_M(t);
    //     var P[t][t] = POSEIDON_P(t);

    let n_inputs = inputs.len();

    let t = n_inputs + 1;
    let n_rounds_f = 8;
    let n_rounds_p = N_ROUNDS_P[t - 2];
    let c = poseidon_c(t);
    let s = poseidon_s(t);
    let m = poseidon_m(t);
    let m_slices = m
        .iter()
        .map(|row| row.as_slice())
        .collect::<Vec<&[Goldilocks]>>();
    let p = poseidon_p(t);
    let p_slices = p
        .iter()
        .map(|row| row.as_slice())
        .collect::<Vec<&[Goldilocks]>>();

    //     component ark[nRoundsF];
    //     component sigmaF[nRoundsF][t];
    //     component sigmaP[nRoundsP];
    //     component mix[nRoundsF-1];
    //     component mixS[nRoundsP];
    //     component mixLast[nOuts];

    let mut ark_in = vec![vec![]; n_rounds_f];
    let mut ark_out = vec![vec![]; n_rounds_f];
    let mut sigma_f_in = vec![vec![0usize; t]; n_rounds_f];
    let mut sigma_f_out = vec![vec![0usize; t]; n_rounds_f];
    let mut sigma_p_in = vec![0usize; n_rounds_p];
    let mut sigma_p_out = vec![0usize; n_rounds_p];
    let mut mix_in = vec![vec![]; n_rounds_f - 1];
    let mut mix_out = vec![vec![]; n_rounds_f - 1];
    let mut mix_s_in = vec![vec![]; n_rounds_p];
    let mut mix_s_out = vec![vec![]; n_rounds_p];
    let mut mix_last_in = vec![vec![]; n_outs];
    let mut mix_last_out = vec![0usize; n_outs];

    //     ark[0] = Ark(t, C, 0);
    //     for (var j=0; j<t; j++) {
    //         if (j>0) {
    //             ark[0].in[j] <== inputs[j-1];
    //         } else {
    //             ark[0].in[j] <== initialState;
    //         }
    //     }

    ark_in[0] = {
        let mut ark_in_0 = Vec::with_capacity(t);
        ark_in_0.push(initial_state);
        ark_in_0.extend_from_slice(&inputs[0..t - 1]);
        ark_in_0
    };

    ark_out[0] = ark(circuit_builder, &ark_in[0], &c, 0);

    //     for (var r = 0; r < nRoundsF\2-1; r++) {
    //         for (var j=0; j<t; j++) {
    //             sigmaF[r][j] = Sigma();
    //             if(r==0) {
    //                 sigmaF[r][j].in <== ark[0].out[j];
    //             } else {
    //                 sigmaF[r][j].in <== mix[r-1].out[j];
    //             }
    //         }

    //         ark[r+1] = Ark(t, C, (r+1)*t);
    //         for (var j=0; j<t; j++) {
    //             ark[r+1].in[j] <== sigmaF[r][j].out;
    //         }

    //         mix[r] = Mix(t,M);
    //         for (var j=0; j<t; j++) {
    //             mix[r].in[j] <== ark[r+1].out[j];
    //         }

    //     }

    for r in 0..(n_rounds_f / 2 - 1) {
        for j in 0..t {
            sigma_f_in[r][j] = if r == 0 {
                ark_out[0][j]
            } else {
                mix_out[r - 1][j]
            };
            sigma_f_out[r][j] = sigma(circuit_builder, sigma_f_in[r][j]);
        }

        for j in 0..t {
            ark_in[r + 1].push(sigma_f_out[r][j]);
        }
        ark_out[r + 1] = ark(circuit_builder, &sigma_f_in[r], &c, (r + 1) * t);

        mix_in[r] = std::mem::take(&mut ark_out[r + 1]);
        mix_out[r] = mix(circuit_builder, &mix_in[r], &m_slices);
    }

    //     for (var j=0; j<t; j++) {
    //         sigmaF[nRoundsF\2-1][j] = Sigma();
    //         sigmaF[nRoundsF\2-1][j].in <== mix[nRoundsF\2-2].out[j];
    //     }

    for j in 0..t {
        sigma_f_in[n_rounds_f / 2 - 1][j] = mix_out[n_rounds_f / 2 - 2][j];
        sigma_f_out[n_rounds_f / 2 - 1][j] =
            sigma(circuit_builder, sigma_f_in[n_rounds_f / 2 - 1][j]);
    }

    //     ark[nRoundsF\2] = Ark(t, C, (nRoundsF\2)*t );
    //     for (var j=0; j<t; j++) {
    //         ark[nRoundsF\2].in[j] <== sigmaF[nRoundsF\2-1][j].out;
    //     }

    for j in 0..t {
        ark_in[n_rounds_f / 2].push(sigma_f_out[n_rounds_f / 2 - 1][j]);
    }
    ark_out[n_rounds_f / 2] = ark(
        circuit_builder,
        &ark_in[n_rounds_f / 2],
        &c,
        n_rounds_f / 2 * t,
    );

    //     mix[nRoundsF\2-1] = Mix(t,P);
    //     for (var j=0; j<t; j++) {
    //         mix[nRoundsF\2-1].in[j] <== ark[nRoundsF\2].out[j];
    //     }

    mix_in[n_rounds_f / 2 - 1] = std::mem::take(&mut ark_out[n_rounds_f / 2]);
    mix_out[n_rounds_f / 2 - 1] = mix(circuit_builder, &mix_in[n_rounds_f / 2 - 1], &p_slices);

    //     for (var r = 0; r < nRoundsP; r++) {
    //         sigmaP[r] = Sigma();
    //         if (r==0) {
    //             sigmaP[r].in <== mix[nRoundsF\2-1].out[0];
    //         } else {
    //             sigmaP[r].in <== mixS[r-1].out[0];
    //         }

    //         mixS[r] = MixS(t, S, r);
    //         for (var j=0; j<t; j++) {
    //             if (j==0) {
    //                 mixS[r].in[j] <== sigmaP[r].out + C[(nRoundsF\2+1)*t + r];
    //             } else {
    //                 if (r==0) {
    //                     mixS[r].in[j] <== mix[nRoundsF\2-1].out[j];
    //                 } else {
    //                     mixS[r].in[j] <== mixS[r-1].out[j];
    //                 }
    //             }
    //         }
    //     }

    let one = Goldilocks::ONE;
    for r in 0..n_rounds_p {
        sigma_p_in[r] = if r == 0 {
            mix_out[n_rounds_f / 2 - 1][0]
        } else {
            mix_s_out[r - 1][0]
        };
        sigma_p_out[r] = sigma(circuit_builder, sigma_p_in[r]);

        for j in 0..t {
            mix_s_in[r].push(if j == 0 {
                let cell = circuit_builder.create_cell();
                circuit_builder.add(cell, sigma_p_out[r], one);
                circuit_builder.add_const(cell, c[(n_rounds_f / 2 + 1) * t + r]);
                cell
            } else if r == 0 {
                mix_out[n_rounds_f / 2 - 1][j]
            } else {
                mix_s_out[r - 1][j]
            });
        }
        mix_s_out[r] = mix_s(circuit_builder, &mix_s_in[r], &s, r);
    }

    //     for (var r = 0; r < nRoundsF\2-1; r++) {
    //         for (var j=0; j<t; j++) {
    //             sigmaF[nRoundsF\2 + r][j] = Sigma();
    //             if (r==0) {
    //                 sigmaF[nRoundsF\2 + r][j].in <== mixS[nRoundsP-1].out[j];
    //             } else {
    //                 sigmaF[nRoundsF\2 + r][j].in <== mix[nRoundsF\2+r-1].out[j];
    //             }
    //         }

    //         ark[ nRoundsF\2 + r + 1] = Ark(t, C,  (nRoundsF\2+1)*t + nRoundsP + r*t );
    //         for (var j=0; j<t; j++) {
    //             ark[nRoundsF\2 + r + 1].in[j] <== sigmaF[nRoundsF\2 + r][j].out;
    //         }

    //         mix[nRoundsF\2 + r] = Mix(t,M);
    //         for (var j=0; j<t; j++) {
    //             mix[nRoundsF\2 + r].in[j] <== ark[nRoundsF\2 + r + 1].out[j];
    //         }

    //     }

    for r in 0..(n_rounds_f / 2 - 1) {
        for j in 0..t {
            sigma_f_in[n_rounds_f / 2 + r][j] = if r == 0 {
                mix_s_out[n_rounds_p - 1][j]
            } else {
                mix_out[n_rounds_f / 2 + r - 1][j]
            };
            sigma_f_out[n_rounds_f / 2 + r][j] =
                sigma(circuit_builder, sigma_f_in[n_rounds_f / 2 + r][j]);
        }

        for j in 0..t {
            ark_in[n_rounds_f / 2 + r + 1].push(sigma_f_out[n_rounds_f / 2 + r][j]);
        }
        ark_out[n_rounds_f / 2 + r + 1] = ark(
            circuit_builder,
            &ark_in[n_rounds_f / 2 + r + 1],
            &c,
            (n_rounds_f / 2 + 1) * t + n_rounds_p + r * t,
        );

        mix_in[n_rounds_f / 2 + r] = std::mem::take(&mut ark_out[n_rounds_f / 2 + r + 1]);
        mix_out[n_rounds_f / 2 + r] = mix(circuit_builder, &mix_in[n_rounds_f / 2 + r], &m_slices);
    }

    //     for (var j=0; j<t; j++) {
    //         sigmaF[nRoundsF-1][j] = Sigma();
    //         sigmaF[nRoundsF-1][j].in <== mix[nRoundsF-2].out[j];
    //     }

    for j in 0..t {
        sigma_f_in[n_rounds_f - 1][j] = mix_out[n_rounds_f - 2][j];
        sigma_f_out[n_rounds_f - 1][j] = sigma(circuit_builder, sigma_f_in[n_rounds_f - 1][j]);
    }

    //     for (var i=0; i<nOuts; i++) {
    //         mixLast[i] = MixLast(t,M,i);
    //         for (var j=0; j<t; j++) {
    //             mixLast[i].in[j] <== sigmaF[nRoundsF-1][j].out;
    //         }
    //         out[i] <== mixLast[i].out;
    //     }
    for i in 0..n_outs {
        for j in 0..t {
            mix_last_in[i].push(sigma_f_out[n_rounds_f - 1][j]);
        }
        mix_last_out[i] = mix_last(circuit_builder, &mix_last_in[i], &m_slices, i);
    }

    mix_last_out
}
fn main() {
    let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
    let n_inputs = 4;
    let (_, poseidon_ex_initial_state) = circuit_builder.create_witness_in(1);
    let (_, poseidon_ex_inputs) = circuit_builder.create_witness_in(n_inputs);
    let poseidon_ex_out = poseidon_ex(
        &mut circuit_builder,
        1,
        poseidon_ex_inputs.as_slice(),
        poseidon_ex_initial_state[0],
    );
    println!("The output is located at cell {:?}", poseidon_ex_out[0]);
    circuit_builder.configure();
    #[cfg(debug_assertions)]
    circuit_builder.print_info();
}

mod mock_constant {
    use p3_field::FieldAlgebra;
    use p3_goldilocks::Goldilocks;

    use crate::N_ROUNDS_P;

    pub(crate) fn poseidon_c(t: usize) -> Vec<Goldilocks> {
        let n = t * 8 + N_ROUNDS_P[t - 2];
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            c.push(Goldilocks::from_canonical_u64(i as u64));
        }
        c
    }

    pub(crate) fn poseidon_s(t: usize) -> Vec<Goldilocks> {
        let n = N_ROUNDS_P[t - 2] * (t * 2 - 1);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            s.push(Goldilocks::from_canonical_u64(i as u64));
        }
        s
    }

    pub(crate) fn poseidon_m(t: usize) -> Vec<Vec<Goldilocks>> {
        let mut m = Vec::with_capacity(t);
        for i in 0..t {
            let mut row = Vec::with_capacity(t);
            for j in 0..t {
                row.push(Goldilocks::from_canonical_u64((i * t + j) as u64));
            }
            m.push(row);
        }
        m
    }

    pub(crate) fn poseidon_p(t: usize) -> Vec<Vec<Goldilocks>> {
        let mut p = Vec::with_capacity(t);
        for i in 0..t {
            let mut row = Vec::with_capacity(t);
            for j in 0..t {
                row.push(Goldilocks::from_canonical_u64((i * t + j) as u64));
            }
            p.push(row);
        }
        p
    }
}
