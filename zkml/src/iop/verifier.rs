use crate::{
    Claim, VectorTranscript,
    commit::{self, precommit},
    iop::ChallengeStorage,
    layers::{LayerCtx, LayerProof},
    lookup::{context::TableType, logup_gkr::verifier::verify_logup_proof},
    tensor::Tensor,
};
use anyhow::{anyhow, bail, ensure};
use ff_ext::ExtensionField;

use itertools::Itertools;
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

use serde::{Serialize, de::DeserializeOwned};
use transcript::Transcript;

use super::{Context, Proof, TableProof};

/// What the verifier must have besides the proof
pub struct IO<E> {
    /// Input of the inference given to the model
    input: Tensor<E>,
    /// Output of the inference
    output: Tensor<E>,
}

impl<E> IO<E> {
    pub fn new(input: Tensor<E>, output: Tensor<E>) -> Self {
        Self { input, output }
    }
}

pub(crate) struct Verifier<'a, E: ExtensionField, T: Transcript<E>> {
    pub(crate) commit_verifier: precommit::CommitVerifier<E>,
    pub(crate) witness_verifier: precommit::CommitVerifier<E>,
    pub(crate) transcript: &'a mut T,
}

impl<'a, E: ExtensionField, T: Transcript<E>> Verifier<'a, E, T>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub(crate) fn new(transcript: &'a mut T) -> Self {
        Self {
            commit_verifier: precommit::CommitVerifier::new(),
            witness_verifier: precommit::CommitVerifier::new(),
            transcript,
        }
    }

    pub(crate) fn verify(
        mut self,
        ctx: Context<E>,
        proof: Proof<E>,
        io: IO<E>,
    ) -> anyhow::Result<()> {
        // Ordering of proofs.
        println!(
            "VERIFIER: Proof Order: {:?}",
            proof.steps.iter().map(|p| p.variant_name()).collect_vec()
        );
        // 1. Instatiate everything and append relevant info to the transcript
        let mut numerators = Vec::<E>::new();
        let mut denominators = Vec::<E>::new();

        ctx.write_to_transcript(self.transcript)?;

        // Here we generate and store all lookup related challenges
        // TODO: make this part of verifier struct
        let challenge_storage = if let Some((_, witness_context)) = proof.witness {
            witness_context.write_to_transcript(self.transcript)?;
            ChallengeStorage::<E>::initialise(&ctx, self.transcript)
        } else {
            ChallengeStorage::default()
        };

        proof.steps.iter().rev().for_each(|proof| {
            if let Some((num, denom)) = proof.get_lookup_data() {
                numerators.extend(num.into_iter());
                denominators.extend(denom.into_iter());
            }
        });

        proof.table_proofs.iter().for_each(|proof| {
            let (nums, denoms) = proof.lookup.fractional_outputs();
            numerators.extend(nums.into_iter());
            denominators.extend(denoms.into_iter());
        });

        // 2. Derive the first randomness
        let first_randomness = self
            .transcript
            .read_challenges(io.output.get_data().len().ilog2() as usize);
        // 3. For the output, we manually evaluate the MLE and check if it's the same as what prover
        //    gave. Note prover could ellude that but it's simpler to avoid that special check right
        //    now.
        let output_mle = io.output.get_data().to_vec().into_mle();
        let computed_sum = output_mle.evaluate(&first_randomness);

        let mut output_claim = Claim {
            point: first_randomness,
            eval: computed_sum,
        };

        // 4. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
        // We have two `HashSet`s, one for the type of table used and one for the lookup challenges used
        for proof_and_step in proof.steps.iter().zip(ctx.steps_info.iter()) {
            output_claim = match proof_and_step {
                (LayerProof::<E>::Activation(proof), LayerCtx::Activation(info)) => {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(&TableType::Relu.name())
                        .ok_or(anyhow!(
                            "Couldn't get challenges at Step: {}, LookupType was: {}",
                            proof_and_step.1.variant_name(),
                            TableType::Relu.name()
                        ))?;
                    info.verify_activation(
                        &mut self,
                        output_claim,
                        proof,
                        constant_challenge,
                        column_separation_challenge,
                    )?
                }
                (LayerProof::<E>::Dense(proof), LayerCtx::Dense(info)) => {
                    info.verify_dense(&mut self, output_claim, &proof)?
                }
                (LayerProof::<E>::Requant(proof), LayerCtx::Requant(info)) => {
                    info.verify_full_requant(&mut self, output_claim, proof)?
                }
                (LayerProof::Pooling(proof), LayerCtx::Pooling(info)) => {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(&TableType::Range.name())
                        .ok_or(anyhow!(
                            "Couldn't get challenges at Step: {}, LookupType was: {}",
                            proof_and_step.1.variant_name(),
                            TableType::Range.name()
                        ))?;
                    info.verify_pooling(
                        &mut self,
                        output_claim,
                        &proof,
                        constant_challenge,
                        column_separation_challenge,
                    )?
                }
                (LayerProof::<E>::Convolution(proof), LayerCtx::<E>::Convolution(info)) => {
                    info.verify_convolution(&mut self, output_claim, &proof)?
                }
                _ => bail!(
                    "Step proof: {} and step info: {} did not match",
                    proof_and_step.0.variant_name(),
                    proof_and_step.1.variant_name()
                ),
            }
        }

        // 5. Verify the lookup table proofs
        let mut table_poly_id = proof.steps.len();
        proof
            .table_proofs
            .iter()
            .zip(ctx.lookup.iter())
            .try_for_each(|(table_proof, table_type)| {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(anyhow!(
                        "No challenges found for table of type: {:?} during verification",
                        table_type.name()
                    ))?;

                verify_table::<_, _>(
                    table_proof,
                    *table_type,
                    table_poly_id,
                    &mut self.witness_verifier,
                    self.transcript,
                    constant_challenge,
                    column_separation_challenge,
                )?;
                table_poly_id += 1;

                Result::<(), anyhow::Error>::Ok(())
            })?;

        // 6. input verification: evaluating the input at the random evaluation point from the sumcheck
        let input_mle = io.input.get_data().to_vec().into_mle();
        let computed_randomized_input = input_mle.evaluate(&output_claim.point);
        let given_randomized_input = output_claim.eval;
        ensure!(
            computed_randomized_input == given_randomized_input,
            "input not valid from proof"
        );
        // 7. verify the opening of the accumulation of claims
        self.commit_verifier
            .verify(&ctx.weights, proof.commit, self.transcript)?;

        // 8. verify that the accumulated numerator is zero and accumulated denominator is non-zero
        let (final_num, final_denom) = numerators.into_iter().zip(denominators.into_iter()).fold(
            (E::ZERO, E::ONE),
            |(acc_num, acc_denom), (num, denom)| {
                (acc_num * denom + num * acc_denom, acc_denom * denom)
            },
        );

        ensure!(
            final_num == E::ZERO,
            "Final numerator was non-zero, got: {:?}",
            final_num
        );
        ensure!(
            final_denom != E::ZERO,
            "Final denominator was zero, lookup arguments are invalid"
        );

        Ok(())
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<E: ExtensionField, T: Transcript<E>>(
    ctx: Context<E>,
    proof: Proof<E>,
    io: IO<E>,
    transcript: &mut T,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    let verifier = Verifier::new(transcript);
    verifier.verify(ctx, proof, io)
}

fn verify_table<E: ExtensionField, T: Transcript<E>>(
    proof: &TableProof<E>,
    table_type: TableType,
    poly_id: usize,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
    constant_challenge: E,
    column_separation_challenge: E,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let verifier_claims = verify_logup_proof(
        &proof.lookup,
        1,
        constant_challenge,
        column_separation_challenge,
        t,
    )?;

    // 2. Accumulate the multiplicity poly claim into the witness commitment protocol
    let poly_claims = verifier_claims.claims();
    witness_verifier.add_claim(
        poly_id,
        poly_claims
            .first()
            .ok_or(anyhow!("Claims was empty in table verification!"))?
            .clone(),
    )?;
    // Hard indexing is okay here because we checked above that at least one claim exists
    let expected_claim_evals = table_type.evaluate_table_columns::<E>(&poly_claims[0].point)?;

    ensure!(
        expected_claim_evals.len() == (poly_claims.len() - 1),
        "Expected {} table column evaluation claims, got {}",
        expected_claim_evals.len(),
        poly_claims.len() - 1
    );
    for (poly_claim, expected) in poly_claims[1..].iter().zip(expected_claim_evals.iter()) {
        ensure!(
            poly_claim.eval == *expected,
            "Claimed table eval was wrong, claimed: {:?}, expected: {:?}",
            poly_claim.eval,
            expected
        );
    }
    Ok(())
}
