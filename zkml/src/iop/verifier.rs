use crate::{
    Claim, VectorTranscript,
    commit::{self, identity_eval, precommit, same_poly},
    iop::{ChallengeStorage, StepProof, context::StepInfo},
    lookup::{context::TableType, logup_gkr::verifier::verify_logup_proof},
    quantization,
    tensor::{Tensor, get_root_of_unity},
};
use anyhow::{anyhow, bail, ensure};
use ff_ext::ExtensionField;

use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    util::ceil_log2,
    virtual_poly::VPAuxInfo,
};
use tracing::debug;

use serde::{Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use super::{
    ActivationProof, Context, ConvProof, DenseProof, PoolingProof, Proof, RequantProof, TableProof,
    context::{ActivationInfo, ConvInfo, DenseInfo, PoolingInfo, RequantInfo},
};

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
    // Ordering of proofs.
    println!(
        "VERIFIER: Proof Order: {:?}",
        proof.steps.iter().map(|p| p.variant_name()).collect_vec()
    );

    // 1. Instatiate everything and append relevant info to the transcript
    let mut commit_verifier = precommit::CommitVerifier::new();
    let mut witness_verifier = precommit::CommitVerifier::new();

    let mut numerators = Vec::<E>::new();
    let mut denominators = Vec::<E>::new();

    ctx.write_to_transcript(transcript)?;

    // Here we generate and store all lookup related challenges
    let challenge_storage = if let Some((_, witness_context)) = proof.witness {
        witness_context.write_to_transcript(transcript)?;
        ChallengeStorage::<E>::initialise(&ctx, transcript)
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
    let first_randomness = transcript.read_challenges(io.output.get_data().len().ilog2() as usize);
    // 3. For the output, we manually evaluate the MLE and check if it's the same as what prover
    //    gave. Note prover could ellude that but it's simpler to avoid that special check right
    //    now.
    let output_mle = io.output.get_data().to_vec().into_mle();
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

    // 4. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
    // We have two `HashSet`s, one for the type of table used and one for the lookup challenges used
    for proof_and_step in proof.steps.iter().zip(ctx.steps_info.iter()) {
        output_claim = match proof_and_step {
            (StepProof::<E>::Activation(proof), StepInfo::Activation(info)) => {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&TableType::Relu.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges at Step: {}, LookupType was: {}",
                        proof_and_step.1.variant_name(),
                        TableType::Relu.name()
                    ))?;
                verify_activation::<_, _>(
                    output_claim,
                    &proof,
                    info,
                    &mut witness_verifier,
                    transcript,
                    constant_challenge,
                    column_separation_challenge,
                )?
            }
            (StepProof::<E>::Dense(proof), StepInfo::Dense(info)) => {
                verify_dense(output_claim, &proof, info, &mut commit_verifier, transcript)?
            }
            (StepProof::<E>::Requant(proof), StepInfo::Requant(info)) => {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&TableType::Range.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges at Step: {}, LookupType was: {}",
                        proof_and_step.1.variant_name(),
                        TableType::Range.name()
                    ))?;
                verify_requant::<_, _>(
                    output_claim,
                    &proof,
                    info,
                    &mut witness_verifier,
                    transcript,
                    constant_challenge,
                    column_separation_challenge,
                )?
            }
            (StepProof::Pooling(proof), StepInfo::Pooling(info)) => {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&TableType::Range.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges at Step: {}, LookupType was: {}",
                        proof_and_step.1.variant_name(),
                        TableType::Range.name()
                    ))?;

                verify_pooling::<_, _>(
                    output_claim,
                    proof,
                    info,
                    &mut witness_verifier,
                    transcript,
                    constant_challenge,
                    column_separation_challenge,
                )?
            }
            (StepProof::<E>::Convolution(proof), StepInfo::<E>::Convolution(info)) => {
                verify_convolution(output_claim, &proof, info, &mut commit_verifier, transcript)?
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
                &mut witness_verifier,
                &mut commit_verifier,
                transcript,
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
    commit_verifier.verify(&ctx.weights, proof.commit, transcript)?;

    // 8. verify that the accumulated numerator is zero and accumulated denominator is non-zero
    let (final_num, final_denom) = numerators
        .into_iter()
        .zip(denominators.into_iter())
        .fold((E::ZERO, E::ONE), |(acc_num, acc_denom), (num, denom)| {
            (acc_num * denom + num * acc_denom, acc_denom * denom)
        });

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

fn verify_pooling<E: ExtensionField, T: Transcript<E>>(
    last_claim: Claim<E>,
    proof: &PoolingProof<E>,
    info: &PoolingInfo,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
    constant_challenge: E,
    column_separation_challenge: E,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let verifier_claims = verify_logup_proof(
        &proof.lookup,
        4,
        constant_challenge,
        column_separation_challenge,
        t,
    )?;

    // 2. Verify the sumcheck proof
    let poly_aux = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![info.num_vars; 5]]);
    let batching_challenge = t.get_and_append_challenge(b"batch_pooling").elements;
    let initial_value = verifier_claims
        .claims()
        .iter()
        .fold((E::ZERO, batching_challenge), |(acc, comb), claim| {
            (acc + claim.eval * comb, comb * batching_challenge)
        })
        .0;
    let subclaim = IOPVerifierState::<E>::verify(initial_value, &proof.sumcheck, &poly_aux, t);

    // Run the same poly verifier for the output claims
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);

    sp_verifier.add_claim(last_claim)?;

    let output_claims = &proof.output_claims;
    output_claims
        .iter()
        .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

    let output_proof = &proof.io_accumulation;

    let commit_claim = sp_verifier.verify(output_proof, t)?;

    // Add the result of the same poly verifier to the commitment verifier.
    witness_verifier.add_claim(info.poly_id, commit_claim)?;

    // Challenegs used to batch input poly claims together and link them with zerocheck and lookup verification output
    let [r1, r2] = [t.get_and_append_challenge(b"input_batching").elements; 2];
    let one_minus_r1 = E::ONE - r1;
    let one_minus_r2 = E::ONE - r2;

    let eval_multiplicands = [
        one_minus_r1 * one_minus_r2,
        one_minus_r1 * r2,
        r1 * one_minus_r2,
        r1 * r2,
    ];
    let zc_point = subclaim
        .point
        .iter()
        .map(|chal| chal.elements)
        .collect::<Vec<E>>();
    let zerocheck_point = [
        &[r1],
        &zc_point[..proof.variable_gap],
        &[r2],
        &zc_point[proof.variable_gap..],
    ]
    .concat();

    let zerocheck_input_eval = izip!(proof.zerocheck_evals.iter(), eval_multiplicands.iter())
        .fold(E::ZERO, |zerocheck_acc, (&ze, &me)| {
            zerocheck_acc + (output_claims[0].eval - ze) * me
        });

    let out_claim = Claim {
        point: zerocheck_point,
        eval: zerocheck_input_eval,
    };

    // Now we check consistency between the lookup/sumcheck proof claims and the claims passed to the same poly verifiers.
    let beta_eval = identity_eval(&output_claims[0].point, &verifier_claims.claims()[0].point);

    let computed_zerocheck_claim = proof
        .zerocheck_evals
        .iter()
        .chain(std::iter::once(&beta_eval))
        .product::<E>()
        + proof
            .zerocheck_evals
            .iter()
            .fold((E::ZERO, batching_challenge), |(acc, comb), v| {
                (acc + *v * beta_eval * comb, comb * batching_challenge)
            })
            .0;

    ensure!(
        computed_zerocheck_claim == subclaim.expected_evaluation,
        "Computed zerocheck claim did not line up with output of sumcheck verification"
    );

    Ok(out_claim)
}

fn verify_activation<E: ExtensionField, T: Transcript<E>>(
    last_claim: Claim<E>,
    proof: &ActivationProof<E>,
    info: &ActivationInfo,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
    constant_challenge: E,
    column_separation_challenge: E,
) -> anyhow::Result<Claim<E>>
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

fn verify_requant<E: ExtensionField, T: Transcript<E>>(
    last_claim: Claim<E>,
    proof: &RequantProof<E>,
    info: &RequantInfo,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
    constant_challenge: E,
    column_separation_challenge: E,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let num_instances = (info.requant.right_shift - 1) / ceil_log2(info.requant.after_range) + 2;
    let verifier_claims = verify_logup_proof(
        &proof.lookup,
        num_instances,
        constant_challenge,
        column_separation_challenge,
        t,
    )?;

    // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
    sp_verifier.add_claim(last_claim)?;

    let first_claim = verifier_claims
        .claims()
        .first()
        .ok_or(anyhow::anyhow!("No claims found"))?;
    let point = first_claim.point.clone();
    // The first claim needs to be shifted down as we add a value to make sure that all its evals are in the range 0..1 << BIT_LEn
    let corrected_claim = Claim::<E>::new(
        point.clone(),
        first_claim.eval - E::from(1 << (*quantization::BIT_LEN - 1)),
    );
    sp_verifier.add_claim(corrected_claim)?;

    let new_output_claim = sp_verifier.verify(&proof.io_accumulation, t)?;
    // 3. Accumulate the new claim into the witness commitment protocol
    witness_verifier.add_claim(info.poly_id, new_output_claim)?;

    // Here we recombine all of the none dummy polynomials to get the actual claim that should be passed to the next layer
    let eval_claims = verifier_claims
        .claims()
        .iter()
        .map(|claim| claim.eval)
        .collect::<Vec<E>>();
    let eval = info.requant.recombine_claims(&eval_claims);
    // 4. return the input claim for to be proven at subsequent step
    Ok(Claim { point, eval })
}

pub fn phi_eval<E: ExtensionField>(
    r: Vec<E>,
    rand1: E,
    rand2: E,
    exponents: Vec<E>,
    first_iter: bool,
) -> E {
    let mut eval = E::ONE;
    for i in 0..r.len() {
        eval *= E::ONE - r[i] + r[i] * exponents[exponents.len() - r.len() + i];
    }

    if first_iter {
        eval = (E::ONE - rand2) * (E::ONE - rand1 + rand1 * eval);
    } else {
        eval = E::ONE - rand1 + (E::ONE - E::from(2) * rand2) * rand1 * eval;
    }

    return eval;
}

pub fn pow_two_omegas<E: ExtensionField>(n: usize, is_fft: bool) -> Vec<E> {
    let mut pows = vec![E::ZERO; n - 1];
    let mut rou: E = get_root_of_unity(n);
    if is_fft {
        rou = rou.invert().unwrap();
    }
    pows[0] = rou;
    for i in 1..(n - 1) {
        pows[i] = pows[i - 1] * pows[i - 1];
    }
    return pows;
}

fn verify_convolution<E: ExtensionField, T: Transcript<E>>(
    last_claim: Claim<E>,
    proof: &ConvProof<E>,
    info: &ConvInfo<E>,
    commit_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
) -> anyhow::Result<Claim<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    let conv_claim = last_claim.eval - proof.bias_claim;

    IOPVerifierState::<E>::verify(conv_claim, &proof.ifft_proof, &info.ifft_aux, t);
    assert_eq!(
        info.delegation_ifft.len(),
        proof.ifft_delegation_proof.len(),
        "Inconsistency in iFFT delegation proofs/aux size"
    );

    let mut iter = proof.ifft_delegation_proof.len();
    let mut claim = proof.ifft_claims[1];
    let mut exponents = pow_two_omegas(iter + 1, true);
    let mut prev_r = proof.ifft_proof.point.clone();
    for i in 0..iter {
        IOPVerifierState::<E>::verify(
            claim,
            &proof.ifft_delegation_proof[i],
            &info.delegation_ifft[i],
            t,
        );
        assert_eq!(
            identity_eval(
                proof.ifft_delegation_proof[i].point.clone().as_slice(),
                prev_r.clone().as_slice()
            ),
            proof.ifft_delegation_claims[i][0],
            "Error in identity evaluation ifft delegation iter : {}",
            i
        );
        assert_eq!(
            phi_eval(
                proof.ifft_delegation_proof[i].point.clone(),
                E::ONE - last_claim.point[i],
                prev_r[prev_r.len() - 1],
                exponents.clone(),
                false
            ),
            proof.ifft_delegation_claims[i][1],
            "Error in phi computation ifft delegation iter : {}",
            i
        );

        prev_r = proof.ifft_delegation_proof[i].point.clone();
        claim = proof.ifft_delegation_claims[i][2];
    }
    let scale = E::from(1 << (iter + 1)).invert().unwrap();

    assert_eq!(
        claim,
        scale * (E::ONE) * prev_r[0] + scale * (E::ONE - prev_r[0]),
        "Error in final iFFT delegation step"
    );

    IOPVerifierState::<E>::verify(
        proof.ifft_claims[0],
        &proof.hadamard_proof,
        &info.hadamard,
        t,
    );
    assert_eq!(
        proof.hadamard_clams[2],
        identity_eval(&proof.ifft_proof.point, &proof.hadamard_proof.point),
        "Error in Beta evaluation"
    );

    commit_verifier.add_claim(
        info.poly_id,
        Claim::new(
            [
                proof.hadamard_proof.point.clone(),
                last_claim.point[((info.filter_size).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            proof.hadamard_clams[0],
        ),
    )?;

    commit_verifier.add_claim(
        info.bias_poly_id,
        Claim::new(
            last_claim.point[(proof.ifft_delegation_proof.len())..].to_vec(),
            proof.bias_claim,
        ),
    )?;

    // >>>>>> TODO : 1) Dont forget beta evaluation 2) verification of the last step of delegation <<<<<<<
    // Verify fft sumcheck
    IOPVerifierState::<E>::verify(proof.hadamard_clams[1], &proof.fft_proof, &info.fft_aux, t);
    claim = proof.fft_claims[1];

    assert_eq!(
        info.delegation_fft.len(),
        proof.fft_delegation_proof.len(),
        "Inconsistency in FFT delegation proofs/aux size"
    );
    iter = proof.fft_delegation_proof.len();
    // Verify delegation protocol of W iFFT matrix
    exponents = pow_two_omegas(iter + 1, false);
    prev_r = proof.fft_proof.point.clone();
    for i in 0..iter {
        IOPVerifierState::<E>::verify(
            claim,
            &proof.fft_delegation_proof[i],
            &info.delegation_fft[i],
            t,
        );

        assert_eq!(
            identity_eval(
                proof.fft_delegation_proof[i].point.clone().as_slice(),
                prev_r.clone().as_slice()
            ),
            proof.fft_delegation_claims[i][0],
            "Error in identity evaluation fft delegation iter : {}",
            i
        );

        assert_eq!(
            phi_eval(
                proof.fft_delegation_proof[i].point.clone(),
                proof.hadamard_proof.point[i],
                prev_r[prev_r.len() - 1],
                exponents.clone(),
                i == 0
            ),
            proof.fft_delegation_claims[i][1],
            "Error in phi computation fft delegation iter : {}",
            i
        );

        claim = proof.fft_delegation_claims[i][2];
        prev_r = proof.fft_delegation_proof[i].point.clone();
    }
    assert_eq!(
        claim,
        (E::ONE - E::from(2) * proof.hadamard_proof.point[iter]) * prev_r[0] + E::ONE - prev_r[0],
        "Error in final FFT delegation step"
    );
    let mut input_point = proof.fft_proof.point.clone();
    let mut v = input_point.pop().unwrap();
    v = (E::ONE - v).invert().unwrap();
    for i in 0..input_point.len() {
        input_point[i] = E::ONE - input_point[i];
    }
    // the output claim for this step that is going to be verified at next step
    Ok(Claim {
        // the new randomness to fix at next layer is the randomness from the sumcheck !
        point: [
            input_point.clone(),
            proof.hadamard_proof.point[((info.filter_size * 2).ilog2() as usize)..].to_vec(),
        ]
        .concat(),
        // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
        // random point. 1 because vector is secondary.
        eval: proof.fft_claims[0] * v,
    })
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
    // Subtract the bias evaluation from the previous claim to remove the bias
    let eval_no_bias = last_claim.eval - proof.bias_eval;
    debug!("VERIFIER: claim {:?}", last_claim);
    // TODO: currently that API can panic - should remove panic for error
    let subclaim =
        IOPVerifierState::<E>::verify(eval_no_bias, &proof.sumcheck, &info.matrix_poly_aux, t);

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
    commit_verifier.add_claim(
        info.matrix_poly_id,
        Claim::new(pcs_eval_input, pcs_eval_output),
    )?;
    commit_verifier.add_claim(
        info.bias_poly_id,
        Claim::new(last_claim.point, proof.bias_eval),
    )?;

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

fn verify_table<E: ExtensionField, T: Transcript<E>>(
    proof: &TableProof<E>,
    table_type: TableType,
    poly_id: usize,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    commit_verifier: &mut commit::precommit::CommitVerifier<E>,
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
    let expected_claim_evals =
        table_type.evaluate_table_columns::<E>(&poly_claims[0].point, commit_verifier)?;

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
