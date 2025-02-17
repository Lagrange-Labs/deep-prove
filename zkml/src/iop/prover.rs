use super::{context::{ActivationInfo, DenseInfo, StepInfo}, Context, Proof, StepProof};
use crate::{
    commit::{same_poly,precommit},
    Claim, Element, VectorTranscript,
    activation::Activation,
    iop::{Matrix2VecProof},
    lookup,
    lookup::LookupProtocol,
    matrix::Matrix,
    model::{InferenceStep, InferenceTrace, Layer},
    vector_to_mle,
};
use anyhow::{bail, Context as CC};
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::{debug, warn};
use multilinear_extensions::{
    mle::{IntoMLE, IntoMLEs, MultilinearExtension},
    virtual_poly::VirtualPolynomial,
};
use serde::{Serialize, de::DeserializeOwned};
use std::cmp::max;
use sumcheck::structs::{IOPProverState, IOPVerifierState};
use transcript::Transcript;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx: &'a Context<E>,
    // proofs for each layer being filled
    proofs: Vec<StepProof<E>>,
    transcript: &'a mut T,
    commit_prover: precommit::CommitProver<E>,
    /// the context of the witness part (IO of lookups, linked with matrix2vec for example)
    /// is generated during proving time. It is first generated and then the fiat shamir starts.
    /// The verifier doesn't know about the individual polys (otherwise it beats the purpose) so
    /// that's why it is generated at proof time.
    witness_ctx: Option<precommit::Context<E>>,
    /// The prover related to proving multiple claims about different witness polyy (io of lookups etc)
    witness_prover: precommit::CommitProver<E>,
}

impl<'a, E, T> Prover<'a, E, T>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new(ctx: &'a Context<E>, transcript: &'a mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            commit_prover: precommit::CommitProver::new(),
            // at this step, we can't build the ctx since we don't know the individual polys
            witness_ctx: None,
            witness_prover: precommit::CommitProver::new(),
        }
    }
    fn prove_step<'b>(
        &mut self,
        last_claim: Claim<E>,
        input: &[E],
        step: &InferenceStep<'b, E>,
        info: &StepInfo<E>,
    ) -> anyhow::Result<Claim<E>> {
        match (step.layer, info) {
            (Layer::Dense(matrix), StepInfo::Dense(info)) => {
                // NOTE: here we treat the ID of the step AS the ID of the polynomial. THat's okay because we only care
                // about these IDs being unique, so as long as the mapping between poly <-> id is correct, all good.
                // This is the case here since we treat each matrix as a different poly
                self.prove_dense_step(last_claim, input, &step.output, info, matrix)
            }
            (Layer::Activation(Activation::Relu(relu)),StepInfo::Activation(info)) => {
                self.prove_relu(last_claim, input, &step.output, info)
            }
            _ => bail!("inconsistent step and info from ctx"),
        }
    }

    fn prove_relu(
        &mut self,
        last_claim: Claim<E>,
        // input to the relu
        input: &[E],
        // output of the relu
        output: &[E],
        info: &ActivationInfo,
    ) -> anyhow::Result<Claim<E>> {
        assert_eq!(input.len(),output.len(),"input/output of lookup don't have same size");
        let padded_size = 1 << info.padded_num_vars;
        // First call the lookup with the right arguments:
        // * table mle: one mle per column
        // * lookup mle: one mle per column, where the evals are just the list of inputs and output ordered by access
        let table_mles = self.ctx.activation.relu_polys();
        let lookup_mles = vec![input.to_vec(), output.to_vec()];
        println!("RELU: BEFORE: lookup[input].len() = {}, table.len() = {}",lookup_mles[0].len(),table_mles[0].len());
        let table_mles = pad2(table_mles,padded_size,E::ZERO).into_mles();
        let lookup_mles = pad2(lookup_mles,padded_size,E::ZERO).into_mles();
        println!("RELU: AFTER: lookup[input].len() = {}, table.len() = {}",1 << lookup_mles[0].num_vars(), 1 << table_mles[0].num_vars());
        // pad the output to the required size
        let padded_output = output.iter().chain(std::iter::repeat(&E::ZERO)).take(padded_size).cloned().collect_vec();

        // TODO: replace via proper lookup protocol
        let mut lookup_proof =
            lookup::DummyLookup::prove(table_mles, lookup_mles, self.transcript)?;
        // in our case, the output of the RELU is ALSO the same poly that previous proving
        // step (likely dense) has "outputted" to evaluate at a random point. So here we accumulate the two claims,
        // the one from previous proving step and the one given by the lookup protocol into one. Since they're claims
        // about the same poly, we can use the "same_poly" protocol.
        let same_poly_ctx = same_poly::Context::<E>::new(info.padded_num_vars);
        let mut same_poly_prover = same_poly::Prover::<E>::new(padded_output.into_mle());
        same_poly_prover.add_claim(last_claim.pad(info.padded_num_vars))?;
        let (input_claim, output_claim) =
            (lookup_proof.claims.remove(0), lookup_proof.claims.remove(0));
        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;
        // order is (output,mult)
        // TODO: add multiplicities, etc...
        self.witness_prover
            .add_claim(info.poly_id, claim_acc_proof.extract_claim())?;
        // the next step is gonna take care of proving the next claim
        // TODO: clarify that bit - here cheating because of fake lookup protocol, but inconsistency
        // between padded input and real inputsize. Next proving step REQUIRES the real input size.
        let next_claim = {
            let input_mle = input.to_vec().into_mle();
            let point = input_claim.pad(input_mle.num_vars()).point;
            let eval = input_mle.evaluate(&point);
            Claim::from(point,eval)
        };
        Ok(next_claim)
    }

    fn prove_dense_step(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &[E],
        // output of dense layer evaluation
        output: &[E],
        info: &DenseInfo<E>,
        matrix: &Matrix<Element>,
    ) -> anyhow::Result<Claim<E>> {
        let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
        assert_eq!(nrows, output.len(), "dense proving: nrows {} vs output {}",nrows,output.len());
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(ncols, input.len(), "something's wrong with the input");
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = vector_to_mle(input.to_vec());
        let max_var = max(mat_mle.num_vars(), input_mle.num_vars());
        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let mut vp = VirtualPolynomial::<E>::new(max_var);
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
            let mut vp = VirtualPolynomial::<E>::new(max_var);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = vector_to_mle(output.to_vec());
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            debug_assert_eq!(claimed_sum, proof.extract_sum(), "sumcheck output weird");
            debug_assert_eq!(claimed_sum, last_claim.eval);

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
            .add_claim(info.poly_id, Claim::from(point, eval))
            .context("unable to add claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        self.proofs.push(StepProof::Dense(Matrix2VecProof {
            sumcheck: proof,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
    }

    pub fn prove<'b>(mut self, trace: InferenceTrace<'b, E>) -> anyhow::Result<Proof<E>> {
        // First, create the context for the witness polys -
        self.instantiate_witness_ctx(&trace,&self.ctx.steps_info)?;
        // write commitments and polynomials info to transcript
        self.ctx.write_to_transcript(self.transcript)?;
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let r_i = self
            .transcript
            .read_challenges(trace.final_output().len().ilog2() as usize);
        let y_i = vector_to_mle(trace.last_step().output.clone()).evaluate(&r_i);
        let mut last_claim = Claim {
            point: r_i,
            eval: y_i,
        };
        // we start by the output to prove up to the input, GKR style
        for (i, ((input, step),info) )in trace.iter().rev().zip(self.ctx.steps_info.iter()).enumerate() {
            last_claim = self.prove_step(last_claim, input, step,&info)?;
        }
        // now provide opening proofs for all claims accumulated during the proving steps
        let commit_proof = self
            .commit_prover
            .prove(&self.ctx.weights, self.transcript)?;
        let mut output_proof = Proof {
            steps: self.proofs,
            commit: commit_proof,
            witness: None,
        };
        if let Some(witness_ctx) = self.witness_ctx {
            let witness_proof = self.witness_prover.prove(&witness_ctx,self.transcript)?;
            output_proof.witness = Some((witness_proof,witness_ctx));
        }
        Ok(output_proof)
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    fn instantiate_witness_ctx<'b>(&mut self, trace: &InferenceTrace<'b, E>,step_infos: &[StepInfo<E>]) -> anyhow::Result<()> {
        let polys = trace
            .iter()
            .rev()
            .zip(step_infos.iter())
            .filter_map(|((_input, step),info)| {
                match (step.layer,info) {
                    (Layer::Activation(Activation::Relu(_)), StepInfo::Activation(info)) => {
                        Some(vec![(info.poly_id, step.output.clone())])
                    }
                    // the dense layer is handling everything "on its own"
                    _ => None,
                }
            })
            .flatten()
            .collect_vec();
        if !polys.is_empty() {
            let ctx = precommit::Context::generate(polys)
                .context("unable to generate ctx for witnesses")?;
            self.witness_ctx = Some(ctx);
        } else {
            warn!("no activation functions found - no witness commitment");
        }
        Ok(())
    }
}

/// Pad all inner vectors to the given size
fn pad2<E:Clone>(a: Vec<Vec<E>>,nsize: usize,with: E) -> Vec<Vec<E>> {
    // check vectors inside are all of the same length respectively
    assert_eq!(a.iter().map(|v|v.len()).sum::<usize>(), a.len() * a[0].len());
    // make sure we're not doing anything wrong
    assert!(a.iter().all(|v| v.len() <= nsize));
    a.into_iter().map(|mut v| { v.resize(nsize, with.clone()); v } ).collect_vec()
}