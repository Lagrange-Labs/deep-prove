use crate::{
    Claim, VectorTranscript,
    commit::{self, context},
    iop::ChallengeStorage,
    layers::{LayerCtx, LayerProof},
    lookup::{context::TableType, logup_gkr::verifier::verify_logup_proof},
    tensor::Tensor,
};
use anyhow::{anyhow, bail, ensure};
use ff_ext::ExtensionField;

use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

use serde::{Serialize, de::DeserializeOwned};
use tracing::info;
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

pub(crate) struct Verifier<'a, E: ExtensionField, T: Transcript<E>, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) ctx: Context<E, PCS>,
    pub(crate) commit_verifier: context::CommitmentVerifier<E, PCS>,
    pub(crate) transcript: &'a mut T,
}

impl<'a, E: ExtensionField, T: Transcript<E>, PCS> Verifier<'a, E, T, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) fn new(ctx: Context<E, PCS>, transcript: &'a mut T) -> Self {
        let commit_verifier = context::CommitmentVerifier::<E, PCS>::new(&ctx.weights);
        Self {
            ctx,
            commit_verifier,
            transcript,
        }
    }

    pub(crate) fn verify(
        mut self,
        ctx: Context<E, PCS>,
        proof: Proof<E, PCS>,
        io: IO<E>,
    ) -> anyhow::Result<()> {
        // Ordering of proofs.
        info!(
            "VERIFIER: Proof Order: {:?}",
            proof.steps.iter().map(|p| p.variant_name()).collect_vec()
        );
        // 1. Instatiate everything and append relevant info to the transcript
        let mut numerators = Vec::<E>::new();
        let mut denominators = Vec::<E>::new();

        ctx.write_to_transcript(self.transcript)?;
        proof.write_to_transcript(self.transcript)?;
        // Here we generate and store all lookup related challenges
        // TODO: make this part of verifier struct
        let challenge_storage = ChallengeStorage::<E>::initialise(&ctx, self.transcript);

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

        let shape_steps = ctx
            .steps_info
            .iter()
            .rev()
            .scan(None, |last_shape, step| {
                *last_shape = if let Some(shape_step) = last_shape {
                    Some(step.next_shape_step(&shape_step))
                } else {
                    Some(step.shape_step(&ctx.unpadded_input_shape, &io.input.get_shape()))
                };
                Some(last_shape.clone())
                // ?? rev() doesn't work with scan
            })
            .collect_vec()
            .into_iter()
            .rev()
            .map(|s| s.unwrap())
            .collect_vec();

        // 4. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
        // We have two `HashSet`s, one for the type of table used and one for the lookup challenges used
        for ((proof, step), shape_step) in proof
            .steps
            .iter()
            .zip(ctx.steps_info.iter())
            .zip(shape_steps)
        {
            output_claim = match (proof, step) {
                (LayerProof::<E, PCS>::Activation(proof), LayerCtx::Activation(info)) => {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(&TableType::Relu.name())
                        .ok_or(anyhow!(
                            "Couldn't get challenges at Step: {}, LookupType was: {}",
                            step.variant_name(),
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
                (LayerProof::<E, PCS>::Dense(proof), LayerCtx::Dense(info)) => {
                    info.verify_dense(&mut self, output_claim, &proof)?
                }
                (LayerProof::<E, PCS>::Requant(proof), LayerCtx::Requant(info)) => {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(
                            &TableType::Clamping(info.requant.clamping_size()).name(),
                        )
                        .ok_or(anyhow!(
                            "Couldn't get challenges at Step: {}, LookupType was: {}",
                            step.variant_name(),
                            TableType::Range.name()
                        ))?;
                    info.verify_requant(
                        &mut self,
                        output_claim,
                        &proof,
                        constant_challenge,
                        column_separation_challenge,
                    )?
                }
                (LayerProof::Pooling(proof), LayerCtx::Pooling(info)) => {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(&TableType::Range.name())
                        .ok_or(anyhow!(
                            "Couldn't get challenges at Step: {}, LookupType was: {}",
                            step.variant_name(),
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
                (LayerProof::<E, PCS>::Convolution(proof), LayerCtx::<E>::Convolution(info)) => {
                    info.verify_convolution(&mut self, output_claim, &proof, &shape_step)?
                }
                (LayerProof::<E, PCS>::Reshape, LayerCtx::Reshape) => {
                    // reshape doesn't change anything apart the shape but we dont "prove" the shape really
                    output_claim
                }
                _ => bail!(
                    "Step proof: {} and step info: {} did not match",
                    proof.variant_name(),
                    step.variant_name()
                ),
            }
        }

        // 5. Verify the lookup table proofs

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

                verify_table::<_, _, _>(
                    table_proof,
                    *table_type,
                    &mut self.commit_verifier,
                    self.transcript,
                    constant_challenge,
                    column_separation_challenge,
                )?;

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
            .verify(&self.ctx.weights, &proof.opening_proof, self.transcript)?;

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
pub fn verify<E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    ctx: Context<E, PCS>,
    proof: Proof<E, PCS>,
    io: IO<E>,
    transcript: &mut T,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    let verifier = Verifier::new(ctx.clone(), transcript);
    verifier.verify(ctx, proof, io)
}

fn verify_table<E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    proof: &TableProof<E, PCS>,
    table_type: TableType,
    witness_verifier: &mut commit::context::CommitmentVerifier<E, PCS>,
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

    witness_verifier.add_witness_claim(
        proof.get_commitment().clone(),
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
