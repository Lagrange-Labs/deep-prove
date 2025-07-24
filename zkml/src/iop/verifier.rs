use std::collections::HashMap;

use crate::{
    Claim,
    commit::context::{CommitmentVerifier, PolyId},
    iop::{ChallengeStorage, context::ShapeStep},
    layers::{
        LayerCtx, LayerProof,
        provable::{NodeCtx, NodeId, OpInfo, VerifiableCtx, compute_model_output_claims},
    },
    lookup::{context::TableType, logup_gkr::verifier::verify_logup_proof},
    model::ToIterator,
    tensor::Tensor,
    try_unzip,
};
use anyhow::{anyhow, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use tracing::trace;

use serde::{Serialize, de::DeserializeOwned};
use transcript::Transcript;

use super::{Context, Proof, TableProof};

/// What the verifier must have besides the proof
pub struct IO<E> {
    /// Input of the inference given to the model
    pub(crate) input: Vec<Tensor<E>>,
    /// Output of the inference
    pub(crate) output: Vec<Tensor<E>>,
}

impl<E> IO<E> {
    pub fn new(input: Vec<Tensor<E>>, output: Vec<Tensor<E>>) -> Self {
        Self { input, output }
    }
    pub fn inputs(&self) -> &[Tensor<E>] {
        &self.input
    }
}

pub struct Verifier<'a, E: ExtensionField, T: Transcript<E>, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) io: IO<E>,
    pub(crate) commit_verifier: CommitmentVerifier<E, PCS>,
    pub(crate) transcript: &'a mut T,
    pub(crate) challenge_storage: ChallengeStorage<E>,
}

impl<'a, E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>
    Verifier<'a, E, T, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub(crate) fn new(ctx: &Context<E, PCS>, transcript: &'a mut T, io: IO<E>) -> Self {
        let commit_verifier = CommitmentVerifier::<E, PCS>::new(&ctx.commitment_ctx);
        Self {
            io,
            commit_verifier,
            transcript,
            challenge_storage: ChallengeStorage::<E>::default(),
        }
    }

    pub(crate) fn verify(
        mut self,
        ctx: Context<E, PCS>,
        proof: Proof<E, PCS>,
    ) -> anyhow::Result<()> {
        // 1. Instantiate everything and append relevant info to the transcript
        let mut numerators = Vec::<E>::new();
        let mut denominators = Vec::<E>::new();

        ctx.write_to_transcript(self.transcript)?;

        // Here we generate and store all lookup related challenges
        // TODO: make this part of verifier struct
        self.challenge_storage = if ctx.lookup.is_empty() {
            ChallengeStorage::<E>::default()
        } else {
            ChallengeStorage::<E>::initialise(&ctx, self.transcript)
        };

        // iterate over the step proofs in inference order
        for (node_id, node) in ctx.steps_info.to_forward_iterator() {
            if !node.ctx.has_proof() {
                // if the current node is not provable, there is no proof, so we can skip it
                continue;
            }
            let node_proof = proof
                .steps
                .get(&node_id)
                .ok_or(anyhow!("Proof for node {} not found", node_id))?;
            if let Some((num, denom)) = node_proof.get_lookup_data() {
                numerators.extend(num.into_iter());
                denominators.extend(denom.into_iter());
            }
        }

        proof.table_proofs.iter().for_each(|proof| {
            let (nums, denoms) = proof.lookup.fractional_outputs();
            numerators.extend(nums);
            denominators.extend(denoms);
        });
        // 2. Derive output claims
        // first, we bind each output to the node that computes it, so that we know whether we
        // need to compute the output claim or not
        let num_outputs = self.io.output.len();
        let out_nodes =
            NodeCtx::bind_outputs_to_node(ctx.steps_info.to_backward_iterator(), num_outputs)?;
        let mut out_claims = vec![Claim::default(); num_outputs];
        out_nodes.into_iter().try_for_each(|(node_id, out_indexes)| {
            let node = ctx.steps_info.nodes.get(&node_id).ok_or(
                anyhow!("Node {node_id} not found in verifier context")
            )?;
            let output_values = out_indexes.iter().map(|index|
                &self.io.output[*index]
            ).collect_vec();
            let claims = compute_model_output_claims::<_, PCS, _, _>(
                &node.ctx,
                self.transcript,
                &output_values,
            );
            ensure!(
                claims.len() == out_indexes.len(),
                "Number of output claims ({}) does not match number of output indexes ({}) for node {node_id}",
                out_claims.len(),
                out_indexes.len(),
            );
            out_indexes.into_iter().zip(claims).for_each(|(index, claim)| {
                out_claims[index] = claim;
            });
            Ok(())
        })?;

        let mut shape_steps: HashMap<NodeId, ShapeStep> = HashMap::new();
        for (node_id, node_ctx) in ctx.steps_info.to_forward_iterator() {
            let (unpadded_input_shapes, padded_input_shapes): (Vec<_>, Vec<_>) =
                try_unzip(node_ctx.inputs.iter().map(|edge| {
                    if let Some(n) = edge.node {
                        let step = shape_steps
                            .get(&n)
                            .ok_or(anyhow!("Shapes for node {n} not found"))?;
                        ensure!(
                            edge.index < step.unpadded_output_shape.len(),
                            "Required input {} for node {n}, but there are only {} inputs shapes",
                            edge.index,
                            step.unpadded_output_shape.len(),
                        );
                        Ok((
                            step.unpadded_output_shape[edge.index].clone(),
                            step.padded_output_shape[edge.index].clone(),
                        ))
                    } else {
                        ensure!(
                            edge.index < ctx.unpadded_input_shapes.len(),
                            "Required input {} of model, but there are only {} inputs shapes",
                            edge.index,
                            ctx.unpadded_input_shapes.len(),
                        );
                        Ok((
                            ctx.unpadded_input_shapes[edge.index].clone(),
                            self.io.input[edge.index].get_shape(),
                        ))
                    }
                }))?;
            let shape_step = node_ctx
                .ctx
                .shape_step(&unpadded_input_shapes, &padded_input_shapes);
            shape_steps.insert(node_id, shape_step);
        }

        // 4. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
        // We have two `HashSet`s, one for the type of table used and one for the lookup challenges used
        let mut claims_by_layer: HashMap<NodeId, Vec<Claim<E>>> = HashMap::new();
        for (node_id, step) in ctx.steps_info.to_backward_iterator() {
            let node_proof = if step.ctx.has_proof() {
                proof
                    .steps
                    .get(&node_id)
                    .ok_or(anyhow!("Proof for node {} not found", node_id))?
            } else {
                &LayerProof::Dummy
            };
            let shape_step = shape_steps
                .get(&node_id)
                .ok_or(anyhow!("Shape for node {node_id} not found"))?;
            trace!(
                "Verifying proof {} for node {node_id}",
                node_proof.variant_name(),
            );
            let claims_for_verify = step.claims_for_node(&claims_by_layer, &out_claims)?;
            let claims = {
                if step.ctx.is_provable() {
                    // we verify the proof
                    step.ctx
                        .verify(node_proof, &claims_for_verify, &mut self, shape_step)?
                } else {
                    // we only propagate the claims, without changing them, as a non-provable layer
                    // shouldn't change the input values
                    claims_for_verify.into_iter().cloned().collect()
                }
            };
            claims_by_layer.insert(node_id, claims);
        }

        // 5. Verify the lookup table proofs
        proof
            .table_proofs
            .iter()
            .zip(ctx.lookup.iter())
            .try_for_each(|(table_proof, table_type)| {
                let (constant_challenge, column_separation_challenge) = self
                    .challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(anyhow!(
                        "No challenges found for table of type: {:?} during verification",
                        table_type.name()
                    ))?;

                verify_table::<_, _, _>(
                    table_proof,
                    table_type.clone(),
                    &mut self.commit_verifier,
                    self.transcript,
                    constant_challenge,
                    column_separation_challenge,
                )?;

                Result::<(), anyhow::Error>::Ok(())
            })?;

        // inputs are assigned at inference time using the forward iterator so we need to use the same ordering here.
        let input_claims =
            NodeCtx::input_claims(ctx.steps_info.to_forward_iterator(), &claims_by_layer)?;
        // 6. input verification: evaluating the input at the random evaluation point from the sumcheck
        let num_inputs = self.io.input.len();
        for (node_id, claims) in input_claims.into_iter() {
            // we assume the inputs are given in the same order as the claims, "flattened"
            let (inputs, claims): (Vec<_>, Vec<_>) = try_unzip(claims.into_iter()
                .map(|(index, claim)| {
                    ensure!(index < num_inputs,
                        "Processing claim associated to input {index}, but there are only {num_inputs} inputs",
                    );
                    Ok((
                        &self.io.input[index],
                        claim,
                    ))
                }))?;
            let node_ctx = ctx
                .steps_info
                .nodes
                .get(&node_id)
                .ok_or(anyhow!("Node {node_id} not found"))?;
            <LayerCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(
                &node_ctx.ctx,
                inputs.as_slice(),
                &claims,
            )?;
        }

        // 7. verify the opening of the accumulation of claims
        self.commit_verifier
            .verify(&ctx.commitment_ctx, &proof.commit, self.transcript)?;

        let num_len = numerators.len();
        // 8. verify that the accumulated numerator is zero and accumulated denominator is non-zero
        let (final_num, final_denom) = numerators.into_iter().zip(denominators).fold(
            (E::ZERO, E::ONE),
            |(acc_num, acc_denom), (num, denom)| {
                (acc_num * denom + num * acc_denom, acc_denom * denom)
            },
        );

        ensure!(
            final_num == E::ZERO,
            "Final numerator was non-zero, got: {:?} - numerator.len(): {}",
            final_num,
            num_len
        );
        ensure!(
            final_denom != E::ZERO,
            "Final denominator was zero, lookup arguments are invalid"
        );

        Ok(())
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeId,
        claims: HashMap<PolyId, Claim<E>>,
    ) -> anyhow::Result<()> {
        self.commit_verifier.add_common_claims(node_id, claims)
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    ctx: Context<E, PCS>,
    proof: Proof<E, PCS>,
    io: IO<E>,
    transcript: &mut T,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    let verifier = Verifier::new(&ctx, transcript, io);
    verifier.verify(ctx, proof)
}

fn verify_table<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    proof: &TableProof<E, PCS>,
    table_type: TableType,
    witness_verifier: &mut CommitmentVerifier<E, PCS>,
    t: &mut T,
    constant_challenge: E,
    column_separation_challenge: E,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
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
    // Add any table poly claims to the commitment verifier
    let table_poly_claims = table_type.table_claims(poly_claims);

    if !table_poly_claims.is_empty() {
        // If the table poly claims aren't empty there should only be 1
        ensure!(
            table_poly_claims.len() == 1,
            "If table poly claims isn't empty we should only have 1, got: {}",
            table_poly_claims.len()
        );
        witness_verifier.add_table_claim(table_type.clone(), table_poly_claims[0].clone())?;
    }

    // Hard indexing is okay here because we checked above that at least one claim exists
    let expected_claim_evals = table_type.evaluate_table_columns::<E>(&poly_claims[0].point)?;

    ensure!(
        expected_claim_evals.len() == (poly_claims.len() - table_poly_claims.len() - 1),
        "Expected {} table column evaluation claims, got {}, for table type: {}",
        poly_claims.len() - table_poly_claims.len() - 1,
        expected_claim_evals.len(),
        table_type.name(),
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
