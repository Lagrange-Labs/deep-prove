//! Contains code for building a GKR to obtain the output of a fractional sumcheck for lookup columns when using the LogUp protocol.

use ff::Field;
use ff_ext::ExtensionField;
use gkr::structs::Circuit;

use simple_frontend::structs::{CellId, CircuitBuilder, ExtCellId};

use super::add_fractions;

pub fn lookup_wire_fractional_sumcheck<E: ExtensionField>(
    no_lookup_columns: usize,
    num_vars: usize,
) -> Circuit<E> {
    let cb = &mut CircuitBuilder::default();

    // For each column of the table and lookup wire we have an input witness
    let lookup_columns = (0..no_lookup_columns)
        .map(|_| cb.create_witness_in(1 << num_vars).1)
        .collect::<Vec<Vec<CellId>>>();
    let lookups = cb.create_ext_cells(1 << num_vars);

    lookups.iter().enumerate().for_each(|(i, lookup_cell)| {
        let lookup_row = lookup_columns
            .iter()
            .map(|col| col[i])
            .collect::<Vec<CellId>>();
        cb.combine_columns(lookup_cell, &lookup_row, 0, 1);
    });

    let (mut lookup_nums, mut lookup_denoms): (Vec<ExtCellId<E>>, Vec<ExtCellId<E>>) = lookups
        .chunks(2)
        .map(|denoms| {
            let num_out = cb.create_ext_cell();
            cb.add_ext(&num_out, &denoms[0], -E::BaseField::ONE);
            cb.add_ext(&num_out, &denoms[1], -E::BaseField::ONE);

            let denom_out = cb.create_ext_cell();
            cb.mul2_ext(&denom_out, &denoms[0], &denoms[1], E::BaseField::ONE);
            (num_out, denom_out)
        })
        .unzip();

    while lookup_nums.len() > 1 && lookup_denoms.len() > 1 {
        (lookup_nums, lookup_denoms) = add_fractions(cb, &lookup_nums, &lookup_denoms);
    }

    lookup_nums
        .into_iter()
        .zip(lookup_denoms.into_iter())
        .for_each(|(num, denom)| {
            cb.create_witness_out_from_exts(&[num]);
            cb.create_witness_out_from_exts(&[denom]);
        });

    cb.configure();

    Circuit::new(cb)
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        RngCore, SeedableRng,
        rngs::{OsRng, StdRng},
    };
    use gkr::{
        error::GKRError,
        structs::{CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
    use transcript::{BasicTranscript, Transcript};

    use super::*;

    #[test]
    fn test_fractional_sumcheck() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());

        let challenge = GoldilocksExt2::random(&mut rng);
        let const_challenge = GoldilocksExt2::random(&mut rng);
        let challenges = vec![challenge, const_challenge];
        for n in 4..8 {
            println!("Testing with {n} number of variables");
            // Make the circuit for n variables with one columns
            let circuit = lookup_wire_fractional_sumcheck::<GoldilocksExt2>(1, n);

            // First make two random columns that should pass
            let lookup_column = (0..1 << n)
                .map(|_| Goldilocks::random(&mut rng))
                .collect::<Vec<Goldilocks>>();

            let piop_denoms = lookup_column
                .iter()
                .map(|val| const_challenge + challenge * GoldilocksExt2::from(*val))
                .collect::<Vec<GoldilocksExt2>>();
            let first_denom = piop_denoms[0];
            let (expected_num, expected_denom) = piop_denoms.iter().skip(1).fold(
                (-GoldilocksExt2::ONE, first_denom),
                |(acc_num, acc_denom), &denom_i| {
                    (acc_num * denom_i - acc_denom, acc_denom * denom_i)
                },
            );

            let lookup_mle = DenseMultilinearExtension::from_evaluations_slice(n, &lookup_column);

            // First make two random columns that should pass
            let lookup_column_2 = (0..1 << n)
                .map(|_| Goldilocks::random(&mut rng))
                .collect::<Vec<Goldilocks>>();

            let piop_denoms_2 = lookup_column_2
                .iter()
                .map(|val| const_challenge + challenge * GoldilocksExt2::from(*val))
                .collect::<Vec<GoldilocksExt2>>();
            let first_denom_2 = piop_denoms_2[0];
            let (expected_num_2, expected_denom_2) = piop_denoms_2.iter().skip(1).fold(
                (-GoldilocksExt2::ONE, first_denom_2),
                |(acc_num, acc_denom), &denom_i| {
                    (acc_num * denom_i - acc_denom, acc_denom * denom_i)
                },
            );

            let lookup_mle_2 =
                DenseMultilinearExtension::from_evaluations_slice(n, &lookup_column_2);
            // Initiate a new transcript for the prover
            let mut prover_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            // Make the proof and the claimed product of the denominator polynomial
            let now = std::time::Instant::now();
            let (proof, numerators, denominators) = prove_fractional_sumcheck(
                1,
                &[lookup_mle.clone(), lookup_mle_2.clone()],
                &circuit,
                challenges.clone(),
                &mut prover_transcript,
            )
            .unwrap();
            println!("Total time to run prove function: {:?}", now.elapsed());

            for (&output_num, &expected_numerator) in
                numerators.iter().zip([expected_num, expected_num_2].iter())
            {
                assert_eq!(output_num, expected_numerator);
            }
            for (&output_denom, &expected_denominator) in denominators
                .iter()
                .zip([expected_denom, expected_denom_2].iter())
            {
                assert_eq!(output_denom, expected_denominator);
            }

            // Initiate a new transcript for the verifier
            let mut verifier_transcript = BasicTranscript::<GoldilocksExt2>::new(b"test");
            let claims = verify_fractional_sumcheck(
                1,
                proof,
                &numerators,
                &denominators,
                &challenges,
                &circuit,
                &mut verifier_transcript,
            )
            .unwrap();

            let p_and_e = claims.point_and_evals[0].clone();
            let mut merged_lookup = lookup_mle.clone();
            merged_lookup.merge(lookup_mle_2.clone());

            assert_eq!(p_and_e.eval, merged_lookup.evaluate(&p_and_e.point));

            let trunc_point = &p_and_e.point[..n];
            let final_var = p_and_e.point[n];

            let l1 = lookup_mle.evaluate(trunc_point);
            let l2 = lookup_mle_2.evaluate(trunc_point);

            let final_eval = (GoldilocksExt2::ONE - final_var) * l1 + final_var * l2;
            assert_eq!(p_and_e.eval, final_eval);
        }
    }

    fn prove_fractional_sumcheck<E: ExtensionField, T: Transcript<E>>(
        num_instance_vars: usize,
        lookups: &[DenseMultilinearExtension<E>],
        circuit: &Circuit<E>,
        challenges: Vec<E>,
        transcript: &mut T,
    ) -> Option<(IOPProof<E>, Vec<E>, Vec<E>)> {
        assert_eq!(1 << num_instance_vars, lookups.len());
        let mut witness = CircuitWitness::new(circuit, challenges);
        for l in lookups.iter() {
            witness.add_instance(circuit, vec![l.clone()]);
        }

        // Squeeze a challenge to be used in evaluating the output mle (this is denom_prod flattened ot basefield elements)
        let output_point = std::iter::repeat_with(|| {
            transcript
                .get_and_append_challenge(b"output_challenge")
                .elements
        })
        .take(1 << num_instance_vars)
        .collect::<Vec<E>>();
        // Compute the output eval
        let witness_out_evals = witness
            .witness_out_ref()
            .iter()
            .map(|mle| PointAndEval::<E>::new(output_point.clone(), mle.evaluate(&output_point)))
            .collect::<Vec<PointAndEval<E>>>();

        let (proof, _) = IOPProverState::prove_parallel(
            circuit,
            &witness,
            vec![],
            witness_out_evals,
            1 << num_instance_vars,
            transcript,
        );
        let numerators = witness.witness_out_ref()[0]
            .get_base_field_vec()
            .chunks(2)
            .map(|chunk| E::from_bases(chunk))
            .collect::<Vec<E>>();
        let denominators = witness.witness_out_ref()[1]
            .get_base_field_vec()
            .chunks(2)
            .map(|chunk| E::from_bases(chunk))
            .collect::<Vec<E>>();

        Some((proof, numerators, denominators))
    }

    pub fn verify_fractional_sumcheck<E: ExtensionField, T: Transcript<E>>(
        instance_num_vars: usize,
        proof: IOPProof<E>,
        numerators: &[E],
        denominators: &[E],
        challenges: &[E],
        circuit: &Circuit<E>,
        transcript: &mut T,
    ) -> Result<GKRInputClaims<E>, GKRError> {
        let output_point = std::iter::repeat_with(|| {
            transcript
                .get_and_append_challenge(b"output_challenge")
                .elements
        })
        .take(1 << instance_num_vars)
        .collect::<Vec<E>>();

        let numerator_mle = DenseMultilinearExtension::from_evaluations_vec(
            1 << instance_num_vars,
            numerators
                .iter()
                .flat_map(|val| val.as_bases().to_vec())
                .collect::<Vec<E::BaseField>>(),
        );
        let denominator_mle = DenseMultilinearExtension::from_evaluations_vec(
            1 << instance_num_vars,
            denominators
                .iter()
                .flat_map(|val| val.as_bases().to_vec())
                .collect::<Vec<E::BaseField>>(),
        );
        let witness_out_evals = vec![
            PointAndEval::<E>::new(output_point.clone(), numerator_mle.evaluate(&output_point)),
            PointAndEval::<E>::new(
                output_point.clone(),
                denominator_mle.evaluate(&output_point),
            ),
        ];

        gkr::structs::IOPVerifierState::verify_parallel(
            circuit,
            challenges,
            vec![],
            witness_out_evals,
            proof,
            instance_num_vars,
            transcript,
        )
    }
}
