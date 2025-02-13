//! Contains code for LogUp proving using GKR see: https://eprint.iacr.org/2023/1284.pdf for more.

use ff::Field;
use ff_ext::ExtensionField;
use gkr::{
    error::GKRError,
    structs::{Circuit, CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval},
};

use multilinear_extensions::mle::{DenseMultilinearExtension, FieldType, MultilinearExtension};
use simple_frontend::structs::CircuitBuilder;

use transcript::Transcript;

/// Function that builds the LogUp GKR circuit, the input `num_vars` is the number of variables the `p` and `q` MLEs will have.
pub fn logup_circuit<E: ExtensionField>(num_vars: usize) -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();

    let [(_, mut p), (_, mut q)] = [0; 2].map(|_| cb.create_witness_in(1 << num_vars));

    // at each level take adjacent pairs (p0, q0) and (p1, q1) and compute the next levels p and q as
    // p_next = p0 * q1 + p1 * q0
    // q_next = q0 * q1
    // We do this because we don't want to perform costly field inversions at each step and this emulates
    // p0/q0 + p1/q1 which is equal to (p0q1 + p1q0)/q0q1
    while p.len() > 1 && q.len() > 1 {
        (p, q) = p
            .chunks(2)
            .zip(q.chunks(2))
            .map(|(p_is, q_is)| {
                let p_out = cb.create_cell();
                cb.mul2(p_out, p_is[0], q_is[1], E::BaseField::ONE);

                cb.mul2(p_out, p_is[1], q_is[0], E::BaseField::ONE);

                let q_out = cb.create_cell();
                cb.mul2(q_out, q_is[0], q_is[1], E::BaseField::ONE);
                (p_out, q_out)
            })
            .unzip();
    }
    // Once the loop has finished we should be left with only one p and q,
    // the value stored in p should be the numerator of Sum p_original/q_original and should be equal to 0
    // the value stored in q is the product of all the evaluations of the input q mle and should be enforced to be non-zero by the verifier.
    assert_eq!(p.len(), 1);
    assert_eq!(q.len(), 1);
    let _ = cb.create_witness_out_from_cells(&p);
    let _ = cb.create_witness_out_from_cells(&q);

    cb.configure();

    Circuit::new(cb)
}

/// Given MLEs `p(X)` and `q(X)` this function produces an [`IOPProof`] that
/// `SUM P(X)/Q(X) == 0`.
pub fn prove_logup<E: ExtensionField, T: Transcript<E>>(
    p: DenseMultilinearExtension<E>,
    q: DenseMultilinearExtension<E>,
    circuit: &Circuit<E>,
    transcript: &mut T,
) -> Option<IOPProof<E>> {
    assert_eq!(p.num_vars(), q.num_vars());

    let mut witness = CircuitWitness::new(circuit, Vec::new());
    // We calculate the product of the evaluations of q here as it is an output of the GKR circuit
    let q_prod = match q.evaluations() {
        FieldType::Base(evals) => evals.iter().product::<E::BaseField>(),
        _ => unreachable!(),
    };
    witness.add_instance(circuit, vec![p, q]);

    let (proof, _) = IOPProverState::prove_parallel(
        circuit,
        &witness,
        vec![],
        vec![
            PointAndEval::new(vec![], E::BaseField::ZERO.into()),
            PointAndEval::new(vec![], q_prod.into()),
        ],
        1,
        transcript,
    );

    Some(proof)
}

/// Verifies a GKR proof that `SUM P(X)/Q(X) == 0` when provided with `PROD Q(X)`.
/// It also errors if `PROD Q(X) == 0`
pub fn verify_logup<E: ExtensionField, T: Transcript<E>>(
    q_prod: E::BaseField,
    proof: IOPProof<E>,
    circuit: &Circuit<E>,
    transcript: &mut T,
) -> Result<GKRInputClaims<E>, GKRError> {
    if q_prod.is_zero().into() {
        return Err(GKRError::VerifyError(
            "The provided product of the Q polynomial evaluations was zero",
        ));
    }
    gkr::structs::IOPVerifierState::verify_parallel(
        circuit,
        &[],
        vec![],
        vec![
            PointAndEval::new(vec![], E::BaseField::ZERO.into()),
            PointAndEval::new(vec![], q_prod.into()),
        ],
        proof,
        0,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        Rng, RngCore, SeedableRng,
        rngs::{OsRng, StdRng},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use transcript::BasicTranscript;

    use super::*;

    #[test]
    fn test_logup_gkr() {
        for n in 2..20 {
            let circuit = logup_circuit::<GoldilocksExt2>(n);

            let mut rng = StdRng::seed_from_u64(OsRng.next_u64());

            let p_evals = [vec![Goldilocks::ONE; 1 << (n - 1)], vec![
                -Goldilocks::ONE;
                1 << (n - 1)
            ]]
            .concat();
            let q_evals_1 = (0..1 << (n - 1))
                .map(|_| {
                    let random: u64 = rng.gen::<u64>() + 1u64;
                    Goldilocks::from(random)
                })
                .collect::<Vec<_>>();
            let mut q_evals_2 = q_evals_1.clone();
            q_evals_2.reverse();

            let q_evals = [q_evals_1, q_evals_2].concat();

            let q_prod = q_evals.iter().fold(Goldilocks::ONE, |acc, x| acc * x);

            let p = DenseMultilinearExtension::from_evaluations_vec(n, p_evals);
            let q = DenseMultilinearExtension::from_evaluations_vec(n, q_evals);

            let mut prover_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");

            let proof =
                prove_logup(p.clone(), q.clone(), &circuit, &mut prover_transcript).unwrap();
            let mut verifier_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            let claim = verify_logup(q_prod, proof, &circuit, &mut verifier_transcript);
            assert!(claim.is_ok());

            let input_claims = claim.unwrap();

            for (input_poly, point_and_eval) in
                [p, q].iter().zip(input_claims.point_and_evals.iter())
            {
                let actual_eval = input_poly.evaluate(&point_and_eval.point);

                assert_eq!(actual_eval, point_and_eval.eval);
            }
        }
    }
}
