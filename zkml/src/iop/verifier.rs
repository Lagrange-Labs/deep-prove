use std::collections::HashMap;

use crate::{
    Claim, VectorTranscript,
    commit::{self, identity_eval, precommit, same_poly},
    iop::{StepProof, context::StepInfo},
    lookup::{self, LookupProtocol, TableInfo},
    tensor::{Tensor, get_root_of_unity},
};
use anyhow::{anyhow, bail, ensure};
use ff_ext::ExtensionField;

use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
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
    // Ordering of proofs.
    println!(
        "VERIFIER: Proof Order: {:?}",
        proof.steps.iter().map(|p| p.variant_name()).collect_vec()
    );

    let total_steps = proof.steps.len();

    // 1. Instatiate everything and append relevant info to the transcript
    let mut commit_verifier = precommit::CommitVerifier::new();
    let mut witness_verifier = precommit::CommitVerifier::new();

    let mut numerators = Vec::<E>::new();
    let mut denominators = Vec::<E>::new();

    ctx.write_to_transcript(transcript)?;
    proof.steps.iter().rev().for_each(|proof| {
        if let Some((commit, num, denom)) = proof.get_lookup_data() {
            transcript.append_field_elements(&commit);
            numerators.extend(num.into_iter());
            denominators.extend(denom.into_iter());
        }
    });

    let constant_challenge = transcript
        .get_and_append_challenge(b"table_constant_challenge")
        .elements;
    let mut lookup_challenges = HashMap::<String, Vec<E>>::new();
    proof
        .table_proofs
        .iter()
        .zip(ctx.lookup.get_table_circuits().iter())
        .for_each(|(table_proof, table_info)| {
            let table_type = table_info.lookup_type;

            transcript.append_field_elements(table_proof.lookup.get_digest().0.as_slice());

            let actual_challenge = transcript
                .get_and_append_challenge(b"table_challenge")
                .elements;
            lookup_challenges.insert(table_type.name(), vec![
                actual_challenge,
                constant_challenge,
            ]);
            numerators.extend(table_proof.lookup.numerators().into_iter());
            denominators.extend(table_proof.lookup.denominators().into_iter());
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
    for (i, proof_and_step) in proof.steps.iter().zip(ctx.steps_info.iter()).enumerate() {
        output_claim = match proof_and_step {
            (StepProof::<E>::Activation(proof), StepInfo::Activation(info)) => {
                let step = total_steps - 1 - i;
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
            (StepProof::<E>::Dense(proof), StepInfo::Dense(info)) => {
                verify_dense(output_claim, &proof, info, &mut commit_verifier, transcript)?
            }
            (StepProof::<E>::Requant(proof), StepInfo::Requant(info)) => {
                let step = total_steps - 1 - i;
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
            (StepProof::Pooling(proof), StepInfo::Pooling(info)) => {
                let step = total_steps - 1 - i;
                let (lookup_type, _) = ctx.lookup.get_circuit_and_type(step)?;
                let challenges = lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                    "Couldn't get challenges at Requant verification, LookupType was: {:?}",
                    lookup_type
                ))?;

                verify_pooling::<_, _, L>(
                    output_claim,
                    proof,
                    info,
                    &mut witness_verifier,
                    &ctx.lookup,
                    transcript,
                    challenges,
                    step,
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
    proof
        .table_proofs
        .iter()
        .zip(ctx.lookup.get_table_circuits())
        .try_for_each(|(table_proof, table_info)| {
            let challenges = lookup_challenges
                .get(&table_info.lookup_type.name())
                .ok_or(anyhow!(
                    "No challenges found for table of type: {:?} during verification",
                    table_info.lookup_type
                ))?;
            verify_table::<_, _, L>(
                table_proof,
                table_info,
                &mut witness_verifier,
                transcript,
                challenges,
            )
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

fn verify_pooling<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    last_claim: Claim<E>,
    proof: &PoolingProof<E>,
    info: &PoolingInfo,
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

    // 2. Verify the sumcheck proof
    // Squeeze some randomness from the transcript to
    let challenge_point = (0..info.num_vars)
        .map(|_| t.get_and_append_challenge(b"zerocheck_challenge").elements)
        .collect::<Vec<E>>();
    let poly_aux = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![info.num_vars; 5]]);
    let subclaim = IOPVerifierState::<E>::verify(E::ZERO, &proof.sumcheck, &poly_aux, t);

    // Run the same poly verifier for the output claims
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);

    sp_verifier.add_claim(last_claim)?;

    let output_claims = &proof.output_claims;
    output_claims
        .iter()
        .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

    let [input_proof, output_proof] = &proof.io_accumulation;

    let commit_claim = sp_verifier.verify(output_proof, t)?;

    // Add the result of the same poly verifier to the commitment verifier.
    witness_verifier.add_claim(info.poly_id, commit_claim)?;

    // Now we do the same poly verifiaction claims for the input poly
    let sp_ctx = same_poly::Context::<E>::new(info.num_vars + 2);
    let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);

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

    let mut zerocheck_point = proof.input_claims[0].point.clone();
    zerocheck_point[0] = r1;
    zerocheck_point[1 + proof.variable_gap] = r2;

    let mut lookup_point = proof.input_claims[1].point.clone();
    lookup_point[0] = r1;
    lookup_point[1 + proof.variable_gap] = r2;

    let (zerocheck_input_eval_claims, lookup_input_eval_claims): (Vec<E>, Vec<E>) = proof
        .input_claims
        .chunks(2)
        .map(|chunk| (chunk[0].eval, chunk[1].eval))
        .unzip();

    let (zerocheck_eval, lookup_eval) = izip!(
        zerocheck_input_eval_claims.iter(),
        lookup_input_eval_claims.iter(),
        eval_multiplicands.iter()
    )
    .fold(
        (E::ZERO, E::ZERO),
        |(zerocheck_acc, lookup_acc), (&ze, &le, &me)| {
            (zerocheck_acc + ze * me, lookup_acc + le * me)
        },
    );

    sp_verifier.add_claim(Claim {
        point: zerocheck_point,
        eval: zerocheck_eval,
    })?;
    sp_verifier.add_claim(Claim {
        point: lookup_point,
        eval: lookup_eval,
    })?;

    // Run the same poly verifier, the output claim from this is what we pass to the next step of verification.
    let out_claim = sp_verifier.verify(input_proof, t)?;

    // Now we check consistency between the lookup/sumcheck proof claims and the claims passed to the same poly verifiers.
    let input_claims = &proof.input_claims;
    let zerocheck_claim_no_beta = input_claims
        .iter()
        .step_by(2)
        .map(|claim| output_claims[0].eval - claim.eval)
        .product::<E>();
    let beta_eval = identity_eval(&output_claims[0].point, &challenge_point);

    let computed_zerocheck_claim = beta_eval * zerocheck_claim_no_beta;

    ensure!(
        computed_zerocheck_claim == subclaim.expected_evaluation,
        "Computed zerocheck claim did not line up with output of sumcheck verification"
    );

    let lookup_gkr_claim = verifier_claims.gkr_claim();
    let gkr_point = &lookup_gkr_claim.point_and_evals[0].point;
    let gkr_point_len = gkr_point.len();
    let gkr1 = gkr_point[gkr_point_len - 2];
    let gkr2 = gkr_point[gkr_point_len - 1];
    let extra_multiplcands = [
        (E::ONE - gkr1) * (E::ONE - gkr2),
        (E::ONE - gkr1) * gkr2,
        gkr1 * (E::ONE - gkr2),
        gkr1 * gkr2,
    ];
    let lookup_claim = input_claims
        .iter()
        .skip(1)
        .step_by(2)
        .zip(extra_multiplcands)
        .map(|(claim, m)| (output_claims[1].eval - claim.eval) * m)
        .sum::<E>();

    ensure!(
        lookup_claim == lookup_gkr_claim.point_and_evals[0].eval,
        "Lookup claim did not line up got {:?} expected {:?}",
        lookup_claim,
        lookup_gkr_claim.point_and_evals[0].eval
    );
    Ok(out_claim)
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

    let first_claim = verifier_claims
        .claims()
        .first()
        .ok_or(anyhow::anyhow!("No claims found"))?;
    let point = first_claim.point.clone();
    sp_verifier.add_claim(first_claim.clone())?;

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
    IOPVerifierState::<E>::verify(last_claim.eval, &proof.ifft_proof, &info.ifft_aux, t);
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
    IOPVerifierState::<E>::verify(
        proof.ifft_claims[0],
        &proof.hadamard_proof,
        &info.hadamard,
        t,
    );
    commit_verifier.add_claim(
        info.poly_id,
        Claim::new(
            [
                proof.hadamard_proof.point.clone(),
                proof.ifft_proof.point[((2 * info.filter_size).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            proof.hadamard_clams[0],
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
    println!("Successful Verification of Convolution");
    let mut input_point = proof.fft_proof.point.clone();
    let mut v = input_point.pop().unwrap();
    v = (E::ONE - v).invert().unwrap();
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

fn verify_table<E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>(
    proof: &TableProof<E>,
    info: &TableInfo<E>,
    witness_verifier: &mut commit::precommit::CommitVerifier<E>,
    t: &mut T,
    challenges: &[E],
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let verifier_claims = L::verify_table(challenges, &info.circuit, proof.lookup.clone(), t)?;

    // 2. Accumulate the multiplicity poly claim into the witness commitment protocol
    witness_verifier.add_claim(
        info.poly_id,
        verifier_claims
            .claims()
            .last()
            .ok_or(anyhow!("Claims was empty in table verification!"))?
            .clone(),
    )?;

    Ok(())
}
