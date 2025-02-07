use std::cmp::max;

use anyhow::ensure;
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::{debug, info};
use multilinear_extensions::{
    mle::MultilinearExtension,
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::BasicTranscript;

use crate::{
    VectorTranscript,
    matrix::Matrix,
    model::{InferenceStep, InferenceTrace, Layer, Model},
    vector_to_mle,
};

/// Contains all cryptographic material generated by the prover
#[derive(Default, Clone, Serialize, Deserialize)]
struct Proof<E: ExtensionField> {
    /// The successive sumchecks proofs. From output layer to input.
    steps: Vec<StepProof<E>>,
}

/// Contains proof material related to one step of the inference
#[derive(Default, Clone, Serialize, Deserialize)]
struct StepProof<E: ExtensionField> {
    /// the actual sumcheck proof
    proof: IOPProof<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

impl<E: ExtensionField> StepProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
}

impl<E: ExtensionField> Proof<E> {
    /// Appends a new step to the list of proofs
    pub fn push_step_proof(&mut self, proof: IOPProof<E>, individual_claims: Vec<E>) {
        self.steps.push(StepProof {
            proof,
            individual_claims,
        });
    }
}

/// What the verifier must have besides the proof
struct IO<E> {
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

/// Common information between prover and verifier
struct Context<E> {
    /// Dimensions of the polynomials necessary to verify the sumcheck proofs
    polys_aux: Vec<VPAuxInfo<E>>,
    /// TO DISAPPEAR: the model is there to give access to the verifier to the layers
    /// Normally the verifier only has access to the commitment(s)
    model: Model<E>,
}

impl<E: ExtensionField> Context<E> {
    /// Generates a context to give to the verifier that contains informations about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    pub fn generate(model: &Model<E>) -> Self {
        let auxs = model
            .layers()
            .iter()
            .map(|layer| {
                // construct dimension of the polynomial given to the sumcheck
                let (nrows, ncols) = layer.dim();
                // each poly is only two polynomial right now: matrix and vector
                // for matrix, each time we fix the variables related to rows so we are only left
                // with the variables related to columns
                let matrix_num_vars = ncols.ilog2() as usize;
                let vector_num_vars = ncols.ilog2() as usize;
                // there is only one product (i.e. quadratic sumcheck)
                VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                    matrix_num_vars,
                    vector_num_vars,
                ]])
            })
            .rev()
            .collect_vec();
        Self {
            polys_aux: auxs,
            model: model.clone(),
        }
    }

    /// Returns the number of layers there are in the model
    pub fn len(&self) -> usize {
        self.model.layers().len()
    }

    /// This replaces the PCS in the meantime
    pub fn evaluate_layer(&self, index: usize, input: &[E]) -> E {
        self.model
            .layers()
            .get(index)
            .expect("invalid layer addressing")
            .mle()
            .evaluate(input)
    }
}

/// Prover generates a series of sumcheck proofs to prove the inference of a model
struct Prover<E: ExtensionField> {
    transcript: BasicTranscript<E>,
    // proof being filled
    proof: Proof<E>,
}

/// Returns the default transcript the prover and verifier must instantiate to validate a proof.
pub fn default_transcript<E: ExtensionField>() -> BasicTranscript<E> {
    BasicTranscript::new(b"m2vec")
}

impl<E> Prover<E>
where
    E: ExtensionField,
{
    pub fn new() -> Self {
        Self {
            transcript: default_transcript(),
            proof: Default::default(),
        }
    }
    fn prove_step<'a>(
        &mut self,
        random_vars_to_fix: Vec<E>,
        input: &[E],
        step: &InferenceStep<'a, E>,
    ) {
        match step.layer {
            Layer::Dense(matrix) => {
                self.prove_dense_step(random_vars_to_fix, input, &step.output, matrix)
            }
        }
    }
    fn prove_dense_step(
        &mut self,
        random_vars_to_fix: Vec<E>,
        input: &[E],
        output: &[E],
        matrix: &Matrix<E>,
    ) {
        let (nrows, ncols) = (matrix.nrows(), matrix.ncols());
        assert_eq!(nrows, output.len(), "something's wrong with the output");
        assert_eq!(
            nrows.ilog2() as usize,
            random_vars_to_fix.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(ncols, input.len(), "something's wrong with the input");
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&random_vars_to_fix);
        let input_mle = vector_to_mle(input.to_vec());
        let max_var = max(mat_mle.num_vars(), input_mle.num_vars());
        let mut vp = VirtualPolynomial::<E>::new(max_var);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let tmp_transcript = self.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, &mut self.transcript);

        debug_assert!({
            let mut t = tmp_transcript;
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(max_var);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = vector_to_mle(output.to_vec());
            let claimed_sum = mle_output.evaluate(&random_vars_to_fix);

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
                .fold(E::ONE, |mut acc, eval| acc * eval);
            assert_eq!(computed_point, final_prover_point);

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // should/usually done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        self.proof
            .push_step_proof(proof, state.get_mle_final_evaluations());
    }

    pub fn prove<'a>(mut self, trace: InferenceTrace<'a, E>) -> Proof<E> {
        // TODO: input the commitments first to do proper FS

        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let mut randomness_to_fix = self
            .transcript
            .read_challenges(trace.final_output().len().ilog2() as usize);

        // we start by the output to prove up to the input, GKR style
        for (i, (input, step)) in trace.iter().rev().enumerate() {
            info!(
                "prover: step {}: input.len = {:?}, step.matrix {:?}, step.output.len() = {:?}",
                i,
                input.len(),
                step.layer.dim(),
                step.output.len()
            );
            self.prove_step(randomness_to_fix, input, step);
            // this point is the last random point over which to evaluate the original polynomial.
            // In our case, the polynomial is actually 2 for dense layer: the matrix MLE and the
            // vector MLE.
            //
            // So normally the verifier should verify both of these poly at this point. However:
            // 1. For the matrix MLE, we rely on PCS opening proofs, which is another step of the
            //    prover flow.
            // 2. For the vector, we actually do a subsequent sumcheck to prove that the vector MLE
            //    is really equal to the claimed evaluation.
            randomness_to_fix = self.proof.steps.last().unwrap().proof.point.clone();
        }
        self.proof
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<E: ExtensionField>(
    ctx: Context<E>,
    proof: Proof<E>,
    io: IO<E>,
) -> anyhow::Result<()> {
    // TODO: make transcript absorb commitments first
    // 0. Derive the first randomness
    let mut transcript = default_transcript();
    let mut randomness_to_fix = transcript.read_challenges(io.output.len().ilog2() as usize);
    // 1. For the output, we manually evaluate the MLE and check if it's the same as what prover
    //    gave. Note prover could ellude that but it's simpler to avoid that special check right
    //    now.
    let output_mle = vector_to_mle(io.output);
    let computed_sum = output_mle.evaluate(&randomness_to_fix);
    let mut claimed_sum = proof
        .steps
        .first()
        .expect("at least one layer")
        .proof
        // checks that the last g(0) + g(1) is really equal to the output that the verifier's
        // expecting (random evaluation of the output)
        .extract_sum();

    ensure!(
        computed_sum == claimed_sum,
        "output vector evaluation is incorrect"
    );

    let nlayers = ctx.model.layers().len();

    // 2. Verify each proof sequentially
    for (i, (step, aux)) in proof.steps.iter().zip(ctx.polys_aux).enumerate() {
        info!("verify {}: aux {:?}", i, aux);
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &step.proof, &aux, &mut transcript);

        // MATRIX OPENING PART
        // pcs_eval means this evaluation should come from a PCS opening proof
        let pcs_eval_input = subclaim
            .point_flat()
            .iter()
            .chain(randomness_to_fix.iter())
            .cloned()
            .collect_vec();
        // 0 because Matrix comes first in Matrix x Vector
        // Note we don't care about verifying that for the vector since it's verified at the next
        // step.
        let pcs_eval_output = step.individual_claims[0];
        // TODO : replace via PCS
        {
            let computed_output = ctx.model.layers()[nlayers - 1 - i]
                .mle()
                .evaluate(&pcs_eval_input);
            ensure!(
                pcs_eval_output == computed_output,
                "step {}: matrix PCS evaluation failed",
                i
            );
        }

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
        // We compute the evaluation directly from the individual evaluation the prover's giving
        ensure!(
            step.individual_to_virtual_claim() == subclaim.expected_evaluation,
            "step {}: sumcheck claim failed",
            i
        );

        // the new randomness to fix at next layer is the randomness from the sumcheck !
        randomness_to_fix = subclaim.point_flat();
        // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
        // random point. 1 because vector is secondary.
        claimed_sum = step.individual_claims[1];
    }

    let input_mle = vector_to_mle(io.input);
    Ok(())
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::model::Model;

    use super::{Context, IO, Prover, verify};

    type F = GoldilocksExt2;
    use tracing_subscriber;

    #[test]
    fn test_prover_steps() {
        tracing_subscriber::fmt::init();
        let (model, input) = Model::<F>::random(4);
        let trace = model.run(input.clone());
        let output = trace.final_output();
        let ctx = Context::generate(&model);
        let io = IO::new(input, output.to_vec());
        let mut prover = Prover::new();
        let proof = prover.prove(trace);
        verify(ctx, proof, io).expect("invalid proof");
    }

    //#[test]
    // fn test_sumcheck_evals() {
    //    type F = GoldilocksExt2;
    //    let n = (10 as usize).next_power_of_two();
    //    let mat = Matrix::random((2 * n, n)).pad_next_power_of_two();
    //    let vec = random_vector(n);
    //    let sum = mat.matmul(&vec);
    //    let mle1 = mat.to_mle();
    //    let mle2 = vector_to_mle(vec);

    //    let vp = VirtualPolynomial::new(n.ilog2() as usize);
    //    vp.add_mle_list(vec![mle1.clone().into(), mle2.clone().into()], F::ONE);
    //    let poly_info = vp.aux_info.clone();
    //    #[allow(deprecated)]
    //    let (proof, _) = IOPProverState::<F>::prove_parallel(vp.clone(), &mut transcript);

    //    let mut transcript = BasicTranscript::new(b"test");
    //    let subclaim = IOPVerifierState::<F>::verify(sum, &proof, &poly_info, &mut transcript);
    //    assert!(
    //        vp.evaluate(
    //            subclaim
    //                .point
    //                .iter()
    //                .map(|c| c.elements)
    //                .collect::<Vec<_>>()
    //                .as_ref()
    //        ) == subclaim.expected_evaluation,
    //        "wrong subclaim"
    //    );
    //}
}
