//! Contains code for verifying a LogUpProof

use ff_ext::ExtensionField;
use multilinear_extensions::virtual_poly::VPAuxInfo;
use p3_field::FieldAlgebra;
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use crate::commit::identity_eval;

use super::{
    error::LogUpError,
    structs::{LogUpProof, LogUpVerifierClaim, ProofType},
};

pub fn verify_logup_proof<E: ExtensionField, T: Transcript<E>>(
    proof: &LogUpProof<E>,
    num_instances: usize,
    constant_challenge: E,
    column_separation_challenge: E,
    transcript: &mut T,
) -> Result<LogUpVerifierClaim<E>, LogUpError> {
    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda
    transcript.append_field_element(&E::BaseField::from_canonical_u64(num_instances as u64));
    proof.append_to_transcript(transcript);

    let (numerators, denominators): (Vec<E>, Vec<E>) = proof.fractional_outputs();

    let batching_challenge = transcript
        .get_and_append_challenge(b"initial_batching")
        .elements;
    let mut alpha = transcript
        .get_and_append_challenge(b"initial_alpha")
        .elements;
    let mut lambda = transcript
        .get_and_append_challenge(b"initial_lambda")
        .elements;

    let (mut current_claim, _) =
        proof
            .circuit_outputs()
            .iter()
            .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                // we have four evals and we batch them as alpha * (batching_challenge * (e[1] - e[0]) + e[0] + lambda * (batching_challenge * (e[3] - e[2]) + e[2]) )
                (
                    acc + alpha_comb
                        * (batching_challenge * (e[1] - e[0])
                            + e[0]
                            + lambda * (batching_challenge * (e[3] - e[2]) + e[2])),
                    alpha_comb * alpha,
                )
            });
    // The initial sumcheck point is just the batching challenge
    let mut sumcheck_point: Vec<E> = vec![batching_challenge];

    for (i, (sumcheck_proof, round_evaluations)) in proof.proofs_and_evals().enumerate() {
        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        // Calculate the eq_poly evaluation for this round
        let eq_eval = identity_eval(&sumcheck_point, &sumcheck_proof.point);

        // Run this rounds sumcheck verification
        let current_num_vars = i + 1;
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![current_num_vars; 3]]);
        let sumcheck_subclaim =
            IOPVerifierState::<E>::verify(current_claim, sumcheck_proof, &aux_info, transcript);

        // Squeeze the challenges to combine everything into a single sumcheck
        let batching_challenge = transcript
            .get_and_append_challenge(b"logup_batching")
            .elements;
        let next_alpha = transcript.get_and_append_challenge(b"logup_alpha").elements;
        let next_lambda = transcript
            .get_and_append_challenge(b"logup_lambda")
            .elements;

        // Now we take the round evals and check their consistency with the sumcheck claim
        let evals_per_instance = round_evaluations.len() / num_instances;

        current_claim = if evals_per_instance == 4 {
            let (next_claim, _, sumcheck_claim, _) = round_evaluations.chunks(4).fold(
                (E::ZERO, E::ONE, E::ZERO, E::ONE),
                |(acc_next_claim, next_alpha_comb, acc_sumcheck_claim, prev_alpha), e| {
                    let next_claim_term = acc_next_claim
                        + next_alpha_comb
                            * (batching_challenge * (e[2] - e[0])
                                + e[0]
                                + next_lambda * (batching_challenge * (e[1] - e[3]) + e[3]));

                    let sumcheck_claim_term = acc_sumcheck_claim
                        + prev_alpha
                            * (eq_eval * (e[0] * e[1] + e[2] * e[3] + lambda * e[3] * e[1]));
                    (
                        next_claim_term,
                        next_alpha_comb * next_alpha,
                        sumcheck_claim_term,
                        prev_alpha * alpha,
                    )
                },
            );
            if sumcheck_claim != sumcheck_subclaim.expected_evaluation {
                return Err(LogUpError::VerifierError(format!(
                    "Calculated sumcheck claim: {:?} does not equal this rounds sumcheck output claim: {:?} at round: {}",
                    sumcheck_claim, sumcheck_subclaim.expected_evaluation, i
                )));
            }
            next_claim
        } else {
            let (next_claim, _, sumcheck_claim, _) = round_evaluations.chunks(2).fold(
                (E::ZERO, E::ONE, E::ZERO, E::ONE),
                |(acc_next_claim, alpha_comb, acc_sumcheck_claim, prev_alpha), e| {
                    let next_claim_term =
                        acc_next_claim + alpha_comb * (batching_challenge * (e[0] - e[1]) + e[1]);
                    let sumcheck_claim_term = acc_sumcheck_claim
                        + prev_alpha * eq_eval * (-e[1] - e[0] + lambda * e[0] * e[1]);
                    (
                        next_claim_term,
                        alpha_comb * next_alpha,
                        sumcheck_claim_term,
                        prev_alpha * alpha,
                    )
                },
            );
            if sumcheck_claim != sumcheck_subclaim.expected_evaluation {
                return Err(LogUpError::VerifierError(format!(
                    "Calculated sumcheck claim: {:?} does not equal this rounds sumcheck output claim: {:?} at round: {}",
                    sumcheck_claim, sumcheck_subclaim.expected_evaluation, i
                )));
            }
            next_claim
        };

        alpha = next_alpha;
        lambda = next_lambda;

        sumcheck_point = sumcheck_subclaim
            .point
            .iter()
            .map(|chal| chal.elements)
            .collect::<Vec<E>>();
        sumcheck_point.push(batching_challenge);
    }

    let calculated_eval = calculate_final_eval(
        proof,
        constant_challenge,
        column_separation_challenge,
        alpha,
        lambda,
        num_instances,
    );

    if calculated_eval != current_claim {
        return Err(LogUpError::VerifierError(format!(
            "Calculated final value: {calculated_eval:?} does not match final sumcheck output: {current_claim:?}"
        )));
    }

    Ok(LogUpVerifierClaim::<E>::new(
        proof.output_claims().to_vec(),
        numerators,
        denominators,
    ))
}

fn calculate_final_eval<E: ExtensionField>(
    proof: &LogUpProof<E>,
    constant_challenge: E,
    column_separation_challenge: E,
    alpha: E,
    lambda: E,
    num_instances: usize,
) -> E {
    match proof.proof_type() {
        ProofType::Lookup => {
            let claims_per_instance = proof.output_claims().len() / num_instances;

            proof
                .output_claims()
                .chunks(claims_per_instance)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), chunk| {
                    let chunk_eval = chunk
                        .iter()
                        .fold((constant_challenge, E::ONE), |(acc, csc_comb), cl| {
                            (
                                acc + cl.eval * csc_comb,
                                csc_comb * column_separation_challenge,
                            )
                        })
                        .0;
                    (acc + chunk_eval * alpha_comb, alpha_comb * alpha)
                })
                .0
        }
        ProofType::Table => {
            // The first output claim is the multiplicity poly which is the numerator
            let columns_eval = proof.output_claims()[1..]
                .iter()
                .fold((constant_challenge, E::ONE), |(acc, csc_comb), cl| {
                    (
                        acc + cl.eval * csc_comb,
                        csc_comb * column_separation_challenge,
                    )
                })
                .0;

            proof.output_claims()[0].eval + lambda * columns_eval
        }
    }
}
