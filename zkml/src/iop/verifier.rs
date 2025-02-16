use crate::{
    commit, iop::{precommit, StepProof}, vector_to_mle, Claim, VectorTranscript
};
use crate::iop::precommit::PolyID;
use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::debug;
use multilinear_extensions::{mle::{IntoMLE, MultilinearExtension}, virtual_poly::VPAuxInfo};
use serde::{Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use super::{Context, Matrix2VecProof, Proof};

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
    let mut commit_verifier = precommit::CommitVerifier::new();
    ctx.write_to_transcript(transcript)?;
    // 0. Derive the first randomness
    let mut randomness_to_fix = transcript.read_challenges(io.output.len().ilog2() as usize);
    // 1. For the output, we manually evaluate the MLE and check if it's the same as what prover
    //    gave. Note prover could ellude that but it's simpler to avoid that special check right
    //    now.
    let output_mle = io.output.into_mle();
    let computed_sum = output_mle.evaluate(&randomness_to_fix);
    let mut output_claim = Claim {
        point: randomness_to_fix,
        eval: computed_sum,
    };
    // NOTE: if we only had m2v then we would need to do the following check manually.
    // However, now that we have m2v + relu, we _always_ accumulate output claims into the _witness_prover_
    // part. So we don't need to
    // let mut claimed_sum = proof
    //    .steps
    //    .first()
    //    .expect("at least one layer")
    //    .proof
    //    // checks that the last g(0) + g(1) is really equal to the output that the verifier's
    //    // expecting (random evaluation of the output)
    //    .extract_sum();

    // ensure!(
    //    computed_sum == claimed_sum,
    //    "output vector evaluation is incorrect"
    //);
    // let layers = ctx.model.layers().collect_vec();
    // let nlayers = layers.len();

    // 2. Verify each proof sequentially
    // TODO generalize according to the type of proof - and find a way to link polys to only m2v proofs
    let mut poly_aux_idx = 0;
    for (i, proof) in proof.steps.iter().enumerate() {
        match proof {
            StepProof::Activation(_) => unimplemented!(),
            StepProof::M2V(proof) => {
                // fetch corresponding poly info for that step
                let (id, aux) = ctx.polys_aux[poly_aux_idx].clone();
                // increase the index for next time we see a m2v proof - i.e. the verifier is enforcing the VPAux sequence to 
                // match the sequence of m2v proofs the prover is giving.
                poly_aux_idx += 1;

                output_claim = verify_m2v(output_claim, proof, (id,aux), &mut commit_verifier, transcript)?;
            }
        }
    }
    // 3. input verification: evaluating the input at the random evaluation point from the sumcheck
    let input_mle = vector_to_mle(io.input);
    let computed_randomized_input = input_mle.evaluate(&output_claim.point);
    let given_randomized_input = output_claim.eval;
    ensure!(
        computed_randomized_input == given_randomized_input,
        "input not valid from proof"
    );
    // 4. verify the opening of the accumulation of claims
    commit_verifier.verify(&ctx.weights, proof.commit, transcript)?;
    Ok(())
}

fn verify_m2v<E: ExtensionField,T: Transcript<E>>(
        last_claim: Claim<E>, 
        proof: &Matrix2VecProof<E>,
        (poly_id, aux): (PolyID, VPAuxInfo<E>),
        commit_verifier: &mut commit::precommit::CommitVerifier<E>,
        t: &mut T) -> anyhow::Result<Claim<E>> 
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
         // TODO: currently that API can panic - should remove panic for error
        let subclaim =
            IOPVerifierState::<E>::verify(last_claim.eval, &proof.sumcheck, &aux, t);

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
        commit_verifier.add_claim(poly_id, Claim::from(pcs_eval_input, pcs_eval_output))?;

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
