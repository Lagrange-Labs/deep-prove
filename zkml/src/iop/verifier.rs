use std::collections::{HashMap, HashSet};

use crate::{
    Claim, VectorTranscript,
    commit::{self, precommit, same_poly},
    iop::{StepProof, context::StepInfo, precommit::PolyID},
    lookup::{self, LookupProtocol, LookupType},
};
use anyhow::{Context as CC, anyhow, bail, ensure};
use ff_ext::ExtensionField;
use itertools::{Itertools, multiunzip};
use log::debug;
use mpcs::BasefoldCommitment;
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};
use poseidon::poseidon_hash::hash_n_to_hash_no_pad;
use serde::{Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use super::{
    ActivationProof, Context, DenseProof, Proof, RequantProof, TableProof,
    context::{ActivationInfo, DenseInfo, RequantInfo, TableInfo},
};

/// What the verifier must have besides the proof
pub struct IO<E> {
    /// Input of the inference given to the model
    input: Vec<E>,
    /// Output of the inference
    output: Vec<E>,
}

impl<E> IO<E> {
    pub fn new(input: Vec<E>, output: Vec<E>) -> Self {
        Self { input, output }
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    ctx: Context<E>,
    proof: Proof<E>,
    io: IO<E>,
    transcript: &mut T,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // Ordering of proofs and work out how many table proofs there are.
    let mut num_tables = 0;
    let names = proof
        .steps
        .iter()
        .map(|p| {
            if let StepProof::Table(..) = p {
                num_tables += 1;
            };
            p.variant_name()
        })
        .collect_vec();
    println!("VERIFIER: Proof Order: {:?}", names);

    let total_steps = proof.steps.len();
    let non_table_proofs = total_steps - num_tables;

    // 1. Generate the challenges used in the lookups and extract the claimed numerators and denominators
    let lookup_proof_data = (0..total_steps)
        .filter_map(|i| {
            if i < non_table_proofs {
                proof.steps[total_steps - 1 - num_tables - i].get_lookup_data()
            } else {
                proof.steps[i].get_lookup_data()
            }
        })
        .collect::<Vec<(Vec<E::BaseField>, Vec<E>, Vec<E>)>>();

    let (lookup_commits, numerators, denominators): (
        Vec<Vec<E::BaseField>>,
        Vec<Vec<E>>,
        Vec<Vec<E>>,
    ) = multiunzip(lookup_proof_data);

    let lookup_commits = lookup_commits
        .into_iter()
        .flatten()
        .collect::<Vec<E::BaseField>>();
    let numerators = numerators.into_iter().flatten().collect::<Vec<E>>();
    let denominators = denominators.into_iter().flatten().collect::<Vec<E>>();

    let constant_challenge = E::from_bases(&hash_n_to_hash_no_pad(&lookup_commits).0[..2]);

    let mut lookup_challenges = HashMap::<String, Vec<E>>::new();

    (0..proof.steps.len()).for_each(|i| {
        let res = ctx.lookup.get_circuit_and_type(i);
        if let Ok((LookupType::RequantTable(bit_size), _)) = res {
            let lookup_type = LookupType::RequantTable(*bit_size);
            let bit_size = *bit_size as u8;
            let requant_extra = lookup_type
                .get_dom_sep()
                .iter()
                .chain(std::iter::once(&bit_size))
                .map(|byte| E::BaseField::from(*byte as u64))
                .collect::<Vec<E::BaseField>>();
            let input = [lookup_commits.clone(), requant_extra].concat();
            let requant_challenge = E::from_bases(&hash_n_to_hash_no_pad(&input).0[..2]);
            println!("inserting {}", lookup_type.name());
            lookup_challenges.insert(lookup_type.name(), vec![
                requant_challenge,
                constant_challenge,
            ]);
        } else if let Ok((LookupType::ReluTable, _)) = res {
            let lookup_type = LookupType::ReluTable;
            let relu_extra = lookup_type
                .get_dom_sep()
                .iter()
                .map(|byte| E::BaseField::from(*byte as u64))
                .collect::<Vec<E::BaseField>>();
            let input = [lookup_commits.clone(), relu_extra].concat();
            let relu_challenge = E::from_bases(&hash_n_to_hash_no_pad(&input).0[..2]);
            println!("inserting {}", lookup_type.name());
            lookup_challenges.insert(lookup_type.name(), vec![relu_challenge, constant_challenge]);
        }
    });

    let mut commit_verifier = precommit::CommitVerifier::new();
    let mut witness_verifier = precommit::CommitVerifier::new();

    ctx.write_to_transcript(transcript)?;
    // 1. Derive the first randomness
    let first_randomness = transcript.read_challenges(io.output.len().ilog2() as usize);
    // 2. For the output, we manually evaluate the MLE and check if it's the same as what prover
    //    gave. Note prover could ellude that but it's simpler to avoid that special check right
    //    now.
    let output_mle = io.output.into_mle();
    let computed_sum = output_mle.evaluate(&first_randomness);
    let mut output_claim = Claim {
        point: first_randomness,
        eval: computed_sum,
    };
    // NOTE: if we only had m2v then we need to do the following check manually to make sure the output is correct.
    // For other cases, for example if we have RELU at last, then we _always_ accumulate output claims into the
    // _witness_prover_ part,  so that claim will be verified nonetheless.
    // TODO: optimization to avoid proving the accumulation if last layer is RELU since verifier can do it himself.
    match proof.steps.first().expect("At least one proof") {
        StepProof::Dense(dproof) => {
            // checks that the last g(0) + g(1) is really equal to the output that the verifier's
            // expecting (random evaluation of the output)
            let claimed_sum = dproof.sumcheck.extract_sum();
            ensure!(
                computed_sum == claimed_sum,
                "output vector evaluation is incorrect"
            );
        }
        _ => {}
    }

    // 3. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
    // We have two `HashSet`s, one for the type of table used and one for the lookup challenges used
    for (i, proof) in proof.steps.iter().enumerate() {
        output_claim = match proof {
            StepProof::<E>::Activation(proof) => {
                let info = if let StepInfo::Activation(info) = &ctx.steps_info[i] {
                    info
                } else {
                    return Err(anyhow!("Step info does not line up at activation step"));
                };
                let step = total_steps - 1 - num_tables - i;
                let (lookup_type, _) = ctx.lookup.get_circuit_and_type(step)?;
                let challenges = lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                    "Couldn't get challenges at Activation verification, LookupType was: {:?}",
                    lookup_type
                ))?;
                verify_activation::<_, _, L>(
                    output_claim,
                    &proof,
                    info,
                    &mut witness_verifier,
                    &ctx.lookup,
                    transcript,
                    challenges,
                    step,
                )?
            }
            StepProof::<E>::Dense(proof) => {
                let info = if let StepInfo::Dense(info) = &ctx.steps_info[i] {
                    info
                } else {
                    return Err(anyhow!("Step info does not line up at Dense step"));
                };
                verify_dense(output_claim, &proof, info, &mut commit_verifier, transcript)?
            }
            StepProof::<E>::Requant(proof) => {
                let info = if let StepInfo::Requant(info) = &ctx.steps_info[i] {
                    info
                } else {
                    return Err(anyhow!("Step info does not line up at requant step"));
                };
                let step = total_steps - 1 - num_tables - i;
                let (lookup_type, _) = ctx.lookup.get_circuit_and_type(step)?;
                let challenges = lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                    "Couldn't get challenges at Requant verification, LookupType was: {:?}",
                    lookup_type
                ))?;
                verify_requant::<_, _, L>(
                    output_claim,
                    &proof,
                    info,
                    &mut witness_verifier,
                    &ctx.lookup,
                    transcript,
                    challenges,
                    step,
                )?
            }
            StepProof::<E>::Table(table_proof) => {
                let info = if let StepInfo::Table(info) = &ctx.steps_info[i] {
                    info
                } else {
                    return Err(anyhow!("Step info does not line up at table step"));
                };

                let (lookup_type, _) = ctx.lookup.get_circuit_and_type(i)?;
                let challenges = lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                    "Couldn't get challenges at Table verification, LookupType was: {:?}",
                    lookup_type
                ))?;
                verify_table::<_, _, L>(
                    &table_proof,
                    info,
                    &mut witness_verifier,
                    &ctx.lookup,
                    transcript,
                    challenges,
                    i,
                )?;
                output_claim
            }
            _ => bail!(
                "proof type {} at step {} shouldn't exist ",
                proof.variant_name(),
                i,
            ),
        }
    }
    // 3. input verification: evaluating the input at the random evaluation point from the sumcheck
    let input_mle = io.input.into_mle();
    let computed_randomized_input = input_mle.evaluate(&output_claim.point);
    let given_randomized_input = output_claim.eval;
    ensure!(
        computed_randomized_input == given_randomized_input,
        "input not valid from proof"
    );
    // 5. verify the opening of the accumulation of claims
    commit_verifier.verify(&ctx.weights, proof.commit, transcript)?;

    // 6. verify that the accumulated numerator is zero and accumulated denominator is non-zero
    let (final_num, final_denom) = numerators
        .into_iter()
        .zip(denominators.into_iter())
        .fold((E::ZERO, E::ONE), |(acc_num, acc_denom), (num, denom)| {
            (acc_num * denom + num * acc_denom, acc_denom * denom)
        });

    if final_num != E::ZERO {
        return Err(anyhow!(
            "Final numerator was non-zero, got: {:?}",
            final_num
        ));
    }

    if final_denom == E::ZERO {
        return Err(anyhow!(
            "Final denominator was zero, lookup arguments are invalid"
        ));
    }
    Ok(())
}

fn verify_activation<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    last_claim: Claim<E>,
    proof: &ActivationProof<E>,
    info: &ActivationInfo,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    lookup_ctx: &lookup::Context<E>,
    t: &mut T,
    challenges: &[E],
    step: usize,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let verifier_claims = L::verify(lookup_ctx, challenges, step, proof.lookup.clone(), t)?;

    // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
    sp_verifier.add_claim(last_claim)?;
    verifier_claims.claims()[1..]
        .iter()
        .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

    let new_output_claim = sp_verifier.verify(&proof.io_accumulation, t)?;
    // 3. Accumulate the new claim into the witness commitment protocol
    witness_verifier.add_claim(info.poly_id, new_output_claim)?;

    // 4. return the input claim for to be proven at subsequent step
    Ok(verifier_claims.claims()[0].clone())
}

fn verify_requant<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    last_claim: Claim<E>,
    proof: &RequantProof<E>,
    info: &RequantInfo,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    lookup_ctx: &lookup::Context<E>,
    t: &mut T,
    challenges: &[E],
    step: usize,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let verifier_claims = L::verify(lookup_ctx, challenges, step, proof.lookup.clone(), t)?;

    // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
    sp_verifier.add_claim(last_claim)?;

    let (lookup_type, _) = lookup_ctx.get_circuit_and_type(step)?;
    let num_actual_claims = lookup_type.number_of_columns();

    let total_claims = verifier_claims.claims().len();
    // The actual claims that we care about
    let actual_claims = &verifier_claims.claims()[total_claims - num_actual_claims..];

    let point = actual_claims[0].point.clone();

    // Need to work out the constant values to add/subtract for this step
    let max_bit = info.requant.range << 1;
    let max_bit = max_bit as u64;
    let subtract = max_bit >> info.requant.right_shift;

    let first_claim = actual_claims
        .first()
        .ok_or(anyhow::anyhow!("No claims found"))?;

    sp_verifier.add_claim(first_claim.clone())?;

    let new_output_claim = sp_verifier.verify(&proof.io_accumulation, t)?;
    // 3. Accumulate the new claim into the witness commitment protocol
    witness_verifier.add_claim(info.poly_id, new_output_claim)?;

    // Here we recombine all of the none dummy polynomials to get the actual claim that should be passed to the next layer
    let tmp_eval = E::from(1 << info.requant.right_shift as u64)
        * (first_claim.eval + E::from(subtract))
        + actual_claims
            .iter()
            .skip(1)
            .rev()
            .enumerate()
            .fold(E::ZERO, |acc, (i, claim)| {
                acc + E::from((info.requant.after_range.pow(i as u32)) as u64)
                    * (claim.eval + E::from(128u64))
            });
    let eval = tmp_eval - E::from(max_bit);
    // 4. return the input claim for to be proven at subsequent step
    Ok(Claim { point, eval })
}

fn verify_dense<E: ExtensionField, T: Transcript<E>>(
    last_claim: Claim<E>,
    proof: &DenseProof<E>,
    info: &DenseInfo<E>,
    commit_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    debug!("VERIFIER: claim {:?}", last_claim);
    // TODO: currently that API can panic - should remove panic for error
    let subclaim =
        IOPVerifierState::<E>::verify(last_claim.eval, &proof.sumcheck, &info.poly_aux, t);

    // MATRIX OPENING PART
    // pcs_eval means this evaluation should come from a PCS opening proof
    let pcs_eval_input = subclaim
        .point_flat()
        .iter()
        .chain(last_claim.point.iter())
        .cloned()
        .collect_vec();
    // 0 because Matrix comes first in Matrix x Vector
    // Note we don't care about verifying that for the vector since it's verified at the next
    // step.
    let pcs_eval_output = proof.individual_claims[0];
    commit_verifier.add_claim(info.poly_id, Claim::new(pcs_eval_input, pcs_eval_output))?;

    // SUMCHECK verification part
    // Instead of computing the polynomial at the random point requested like this
    // let computed_point = vp.evaluate(
    //     subclaim
    //         .point
    //         .iter()
    //         .map(|c| c.elements)
    //         .collect_vec()
    //         .as_ref(),
    //
    // We compute the evaluation directly from the individual final evaluations of each polynomial
    // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
    ensure!(
        proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
        "sumcheck claim failed",
    );

    // the output claim for this step that is going to be verified at next step
    Ok(Claim {
        // the new randomness to fix at next layer is the randomness from the sumcheck !
        point: subclaim.point_flat(),
        // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
        // random point. 1 because vector is secondary.
        eval: proof.individual_claims[1],
    })
}

fn verify_table<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    proof: &TableProof<E>,
    info: &TableInfo<E>,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    lookup_ctx: &lookup::Context<E>,
    t: &mut T,
    challenges: &[E],
    step: usize,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // Get the lookup type for this and check its a table
    let (lookup_type, _) = lookup_ctx.get_circuit_and_type(step)?;

    if !lookup_type.is_table() {
        return Err(anyhow!(
            "Verifying a table when the lookup type was not a table type"
        ));
    }

    // 1. Verify the lookup proof
    let verifier_claims = L::verify(lookup_ctx, challenges, step, proof.lookup.clone(), t)?;

    // 2. Accumulate the multiplicity poly claim into the witness commitment protocol
    witness_verifier.add_claim(
        info.poly_id,
        verifier_claims.claims().last().unwrap().clone(),
    )?;

    Ok(())
}
