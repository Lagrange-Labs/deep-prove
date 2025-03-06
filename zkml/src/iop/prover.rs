use super::{
    Context, Proof, RequantProof, StepProof, TableProof,
    context::{DenseInfo, PoolingInfo, StepInfo},
};
use crate::{
    Claim, Element, VectorTranscript,
    activation::Activation,
    commit::{compute_betas_eval, identity_eval, precommit, same_poly},
    iop::{ActivationProof, DenseProof, PoolingProof},
    lookup::{self, LookupProtocol},
    model::{InferenceStep, InferenceTrace, Layer},
    tensor::Tensor,
};
use anyhow::{Context as CC, anyhow, bail};
use ff_ext::ExtensionField;

use log::{debug, warn};
use multilinear_extensions::{
    mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{marker::PhantomData, sync::Arc};
use sumcheck::structs::{IOPProverState, IOPVerifierState};
use transcript::Transcript;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>, L: LookupProtocol<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx: &'a Context<E>,
    // proofs for each layer being filled
    proofs: Vec<StepProof<E>>,
    table_proofs: Vec<TableProof<E>>,
    transcript: &'a mut T,
    commit_prover: precommit::CommitProver<E>,
    /// the context of the witness part (IO of lookups, linked with matrix2vec for example)
    /// is generated during proving time. It is first generated and then the fiat shamir starts.
    /// The verifier doesn't know about the individual polys (otherwise it beats the purpose) so
    /// that's why it is generated at proof time.
    witness_ctx: Option<precommit::Context<E>>,
    /// The prover related to proving multiple claims about different witness polyy (io of lookups etc)
    witness_prover: precommit::CommitProver<E>,
    /// The context for the lookups
    lookup_witness: lookup::WitnessContext<'a, E>,
    _phantom: PhantomData<L>,
}

impl<'a, E, T, L> Prover<'a, E, T, L>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    L: LookupProtocol<E>,
{
    pub fn new(ctx: &'a Context<E>, transcript: &'a mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            table_proofs: Vec::default(),
            commit_prover: precommit::CommitProver::new(),
            // at this step, we can't build the ctx since we don't know the individual polys
            witness_ctx: None,
            witness_prover: precommit::CommitProver::new(),
            lookup_witness: lookup::WitnessContext::default(),
            _phantom: PhantomData,
        }
    }
    fn prove_step<'b>(
        &mut self,
        last_claim: Claim<E>,
        input: &Tensor<E>,
        step: &InferenceStep<'b, E>,
        info: &StepInfo<E>,
    ) -> anyhow::Result<Claim<E>> {
        println!("PROVER: proving layer {}", step.layer.to_string());
        let claim = match (step.layer, info) {
            (Layer::Dense(matrix), StepInfo::Dense(info)) => {
                // NOTE: here we treat the ID of the step AS the ID of the polynomial. THat's okay because we only care
                // about these IDs being unique, so as long as the mapping between poly <-> id is correct, all good.
                // This is the case here since we treat each matrix as a different poly
                self.prove_dense_step(last_claim, input, &step.output, info, matrix)
            }
            (Layer::Activation(Activation::Relu(..)), StepInfo::Activation(..))
            | (Layer::Requant(..), StepInfo::Requant(..)) => {
                self.prove_lookup(&last_claim, &step.output.get_data(), info)
            }
            (Layer::Pooling(..), StepInfo::Pooling(info)) => {
                self.prove_pooling(last_claim, input, &step.output, info)
            }
            _ => bail!(
                "inconsistent proof step {} and info step {} from ctx",
                step.layer.describe(),
                info.variant_name()
            ),
        };

        claim
    }

    fn prove_lookup(
        &mut self,
        last_claim: &Claim<E>,
        output: &[E],
        step: &StepInfo<E>,
    ) -> anyhow::Result<Claim<E>> {
        // First we check that the step requires lookup
        if !step.requires_lookup() {
            return Err(anyhow!(
                "A step of type: {} does not require a lookup proof",
                step.variant_name()
            ));
        }
        let prover_info = self
            .lookup_witness
            .next()
            .ok_or(anyhow!("No more lookup witness!"))?;
        // Run the lookup protocol and return the lookup proof
        let lookup_proof = L::prove(&self.ctx.lookup, &prover_info, self.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;

        match step {
            StepInfo::Activation(info) => {
                // Activation proofs have two columns, input and output

                let input_claim = lookup_proof.claims()[0].clone();
                let output_claim = lookup_proof.claims()[1].clone();

                same_poly_prover.add_claim(output_claim)?;
                let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;
                // order is (output,mult)
                self.witness_prover
                    .add_claim(info.poly_id, claim_acc_proof.extract_claim())?;

                // Add the proof in
                self.proofs.push(StepProof::Activation(ActivationProof {
                    io_accumulation: claim_acc_proof,
                    lookup: lookup_proof,
                }));
                Ok(input_claim)
            }
            StepInfo::Requant(requant_info) => {
                // For requant layers we have to extract the correct "chunk" from the list of claims
                let eval_claims = lookup_proof
                    .claims()
                    .iter()
                    .map(|claim| claim.eval)
                    .collect::<Vec<E>>();

                let combined_eval = requant_info.requant.recombine_claims(&eval_claims);

                // Pass the eval associated with the poly used in the activation step to the same poly prover
                let first_claim = lookup_proof
                    .claims()
                    .first()
                    .ok_or(anyhow!("No claims found"))?;
                let point = first_claim.point.clone();

                // Add the claim used in the activation function
                same_poly_prover.add_claim(first_claim.clone())?;
                let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;

                self.witness_prover
                    .add_claim(requant_info.poly_id, claim_acc_proof.extract_claim())?;

                self.proofs.push(StepProof::Requant(RequantProof {
                    io_accumulation: claim_acc_proof,
                    lookup: lookup_proof,
                }));

                Ok(Claim {
                    point,
                    eval: combined_eval,
                })
            }
            _ => Err(anyhow!(
                "Should not be in prove_lookup function for step: {}",
                step.variant_name()
            )),
        }
    }

    fn prove_tables(&mut self) -> anyhow::Result<()> {
        self.lookup_witness
            .get_table_witnesses()
            .iter()
            .zip(self.ctx.lookup.get_table_circuits().iter())
            .try_for_each(|(table_witness, table_info)| {
                let poly_id = table_info.poly_id;
                println!("PROVING table of type: {:?}", table_info.lookup_type);
                // Make the proof for the table
                let table_proof =
                    L::prove_table(&table_info.circuit, &table_witness, self.transcript)?;

                // Add the multiplicity poly claim
                self.witness_prover
                    .add_claim(poly_id, table_proof.claims().last().unwrap().clone())?;

                self.table_proofs.push(TableProof {
                    lookup: table_proof,
                });
                Ok(())
            })
    }

    fn prove_pooling(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &Tensor<E>,
        // output of dense layer evaluation
        output: &Tensor<E>,
        info: &PoolingInfo,
    ) -> anyhow::Result<Claim<E>> {
        // Create the range check proof for the diff
        let prover_info = self
            .lookup_witness
            .next()
            .ok_or(anyhow!("No more lookup witness!"))?;
        // Run the lookup protocol and return the lookup proof
        let lookup_proof = L::prove(&self.ctx.lookup, &prover_info, self.transcript)?;

        let max_pool_polys = info.poolinfo.compute_polys_field::<E>(input, output);
        // These are the polys that get passed to the zero check make sure their product is zero at every evaluation point
        let diff_polys = max_pool_polys[1..]
            .iter()
            .map(|fixed_input| {
                DenseMultilinearExtension::<E>::from_evaluations_vec(
                    info.num_vars,
                    max_pool_polys[0]
                        .iter()
                        .zip(fixed_input.iter())
                        .map(|(output, input)| *output - *input)
                        .collect::<Vec<E::BaseField>>(),
                )
                .into()
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        // Run the Zerocheck that checks enforces that output does contain the maximum value for the kernel
        let mut vp = VirtualPolynomial::<E>::new(info.num_vars);

        // Squeeze some randomness from the transcript to
        let challenge_point = (0..info.num_vars)
            .map(|_| {
                self.transcript
                    .get_and_append_challenge(b"zerocheck_challenge")
                    .elements
            })
            .collect::<Vec<E>>();

        // Comput the identity poly
        let beta_eval = compute_betas_eval(&challenge_point);
        let beta_poly: ArcDenseMultilinearExtension<E> =
            DenseMultilinearExtension::<E>::from_evaluations_ext_vec(info.num_vars, beta_eval)
                .into();

        vp.add_mle_list(diff_polys.clone(), E::ONE);
        vp.mul_by_mle(beta_poly.clone(), E::BaseField::from(1));

        #[allow(deprecated)]
        let (proof, _) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        // We need to prove that the output of this step is the input to following activation function
        let mles = max_pool_polys
            .iter()
            .map(|evals| {
                DenseMultilinearExtension::<E>::from_evaluations_slice(info.num_vars, evals)
            })
            .collect::<Vec<DenseMultilinearExtension<E>>>();
        let mut same_poly_prover = same_poly::Prover::<E>::new(mles[0].clone());

        let zerocheck_point = &proof.point;
        let output_zerocheck_eval = mles[0].evaluate(zerocheck_point);

        let lookup_point = &lookup_proof.claims()[0].point;
        let output_lookup_eval = mles[0].evaluate(lookup_point);

        // Accumulate claims about the output polynomial in each of the protocols we ran together with the final claim from the previous proof.
        let mut output_claims = Vec::<Claim<E>>::new();
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;

        let zerocheck_claim = Claim {
            point: zerocheck_point.clone(),
            eval: output_zerocheck_eval,
        };
        same_poly_prover.add_claim(zerocheck_claim.clone())?;
        output_claims.push(zerocheck_claim);

        let lookup_claim = Claim {
            point: lookup_point.clone(),
            eval: output_lookup_eval,
        };

        same_poly_prover.add_claim(lookup_claim.clone())?;

        output_claims.push(lookup_claim);

        // This is the proof for the output poly
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;

        let output_claim = claim_acc_proof.extract_claim();

        self.commit_prover
            .add_claim(info.poly_id, output_claim)
            .context("unable to add claim")?;
        // Now we must do the samething accumulating evals for the input poly as we fix variables on the input poly.
        // The point length is 2 longer because for now we only support MaxPool2D.
        let mut input_claims = Vec::<Claim<E>>::new();
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len() + 2);
        let input_mle = DenseMultilinearExtension::<E>::from_evaluations_ext_slice(
            last_claim.point.len() + 2,
            input.get_data(),
        );
        let mut same_poly_prover = same_poly::Prover::<E>::new(input_mle.clone());
        let padded_input_shape = input.dims();
        [[E::ZERO, E::ZERO], [E::ZERO, E::ONE], [E::ONE, E::ZERO], [
            E::ONE,
            E::ONE,
        ]]
        .iter()
        .try_for_each(|pair| {
            let point_1 = [
                &[pair[0]],
                &zerocheck_point[..padded_input_shape[3] >> 2],
                &[pair[1]],
                &zerocheck_point[padded_input_shape[3] >> 2..],
            ]
            .concat();
            let eval = input_mle.evaluate(&point_1);

            let zerocheck_claim = Claim {
                point: point_1,
                eval,
            };

            same_poly_prover.add_claim(zerocheck_claim.clone())?;
            input_claims.push(zerocheck_claim);
            let point_2 = [
                &[pair[0]],
                &lookup_point[..padded_input_shape[3] >> 2],
                &[pair[1]],
                &lookup_point[padded_input_shape[3] >> 2..],
            ]
            .concat();
            let eval = input_mle.evaluate(&point_2);

            let lookup_claim = Claim {
                point: point_2,
                eval,
            };

            input_claims.push(lookup_claim.clone());
            same_poly_prover.add_claim(lookup_claim)
        })?;

        // This is the proof for the input_poly
        let input_claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;

        let next_claim = input_claim_acc_proof.extract_claim();

        // Push the step proof to the list
        self.proofs.push(StepProof::Pooling(PoolingProof {
            sumcheck: proof,
            lookup: lookup_proof,
            io_accumulation: [input_claim_acc_proof, claim_acc_proof],
            output_claims,
            input_claims,
        }));
        Ok(next_claim)
    }

    fn prove_dense_step(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &Tensor<E>,
        // output of dense layer evaluation
        output: &Tensor<E>,
        info: &DenseInfo<E>,
        matrix: &Tensor<Element>,
    ) -> anyhow::Result<Claim<E>> {
        // println!("PROVER: claim {:?}", last_claim);
        let (nrows, ncols) = (matrix.nrows_2d(), matrix.ncols_2d());
        assert_eq!(
            nrows,
            output.get_data().len(),
            "dense proving: nrows {} vs output {}",
            nrows,
            output.get_data().len()
        );
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(
            ncols,
            input.get_data().len(),
            "something's wrong with the input"
        );
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle_2d();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = input.get_data().to_vec().into_mle();

        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let tmp_transcript = self.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        debug_assert!({
            let mut t = tmp_transcript;
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(num_vars);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = output.get_data().to_vec().into_mle();
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            debug_assert_eq!(claimed_sum, last_claim.eval, "sumcheck eval weird");
            debug_assert_eq!(claimed_sum, proof.extract_sum(), "sumcheck output weird");

            debug!("prover: claimed sum: {:?}", claimed_sum);
            let subclaim = IOPVerifierState::<E>::verify(claimed_sum, &proof, &vp.aux_info, &mut t);
            // now assert that the polynomial evaluated at the random point of the sumcheck proof
            // is equal to last small poly sent by prover (`subclaim.expected_evaluation`). This
            // step can be done via PCS opening proofs for all steps but first (output of
            // inference) and last (input of inference)
            let computed_point = vp.evaluate(subclaim.point_flat().as_ref());

            let final_prover_point = state
                .get_mle_final_evaluations()
                .into_iter()
                .fold(E::ONE, |acc, eval| acc * eval);
            assert_eq!(computed_point, final_prover_point);

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // is done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let point = [proof.point.as_slice(), last_claim.point.as_slice()].concat();
        let eval = state.get_mle_final_evaluations()[0];
        self.commit_prover
            .add_claim(info.poly_id, Claim::new(point, eval))
            .context("unable to add claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        self.proofs.push(StepProof::Dense(DenseProof {
            sumcheck: proof,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
    }

    pub fn prove<'b>(mut self, trace: InferenceTrace<'b, Element>) -> anyhow::Result<Proof<E>> {
        // First write commitments and polynomials info to transcript
        self.ctx.write_to_transcript(self.transcript)?;
        // create the context for the witness polys -
        self.instantiate_witness_ctx(&trace)?;
        let trace = trace.to_field::<E>();
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let r_i = self
            .transcript
            .read_challenges(trace.final_output().get_data().len().ilog2() as usize);
        let y_i = trace
            .last_step()
            .output
            .clone()
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&r_i);
        let mut last_claim = Claim {
            point: r_i,
            eval: y_i,
        };

        // we start by the output to prove up to the input, GKR style
        for ((input, step), info) in trace.iter().rev().zip(self.ctx.steps_info.iter()) {
            last_claim = self.prove_step(last_claim, input, step, &info)?;
        }

        // Now we have to make the table proofs
        self.prove_tables()?;

        // now provide opening proofs for all claims accumulated during the proving steps
        let commit_proof = self
            .commit_prover
            .prove(&self.ctx.weights, self.transcript)?;
        let mut output_proof = Proof {
            steps: self.proofs,
            table_proofs: self.table_proofs,
            commit: commit_proof,
            witness: None,
        };
        if let Some(witness_ctx) = self.witness_ctx {
            let witness_proof = self.witness_prover.prove(&witness_ctx, self.transcript)?;
            output_proof.witness = Some((witness_proof, witness_ctx));
        }
        Ok(output_proof)
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    fn instantiate_witness_ctx<'b>(
        &mut self,
        trace: &InferenceTrace<'b, Element>,
    ) -> anyhow::Result<()> {
        let (lookup_witness, polys) = lookup::WitnessContext::<E>::initialise_witness_ctx(
            &self.ctx.lookup,
            trace,
            self.transcript,
        )?;

        if !polys.is_empty() {
            let ctx = precommit::Context::generate(polys)
                .context("unable to generate ctx for witnesses")?;
            self.witness_ctx = Some(ctx);
        } else {
            warn!("no activation functions found - no witness commitment");
        }
        self.lookup_witness = lookup_witness;
        Ok(())
    }
}
