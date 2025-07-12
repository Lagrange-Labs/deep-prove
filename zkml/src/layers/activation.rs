use crate::{
    Claim, Context, Element, Prover,
    commit::same_poly,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{QuantizeOp, QuantizeOutput},
    },
    lookup::{
        context::{COLUMN_SEPARATOR, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
        witness::LogUpWitness,
    },
    model::StepData,
    padding::PaddingMode,
    quantization::Fieldizer,
    tensor::{Number, Shape},
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::{DenseMultilinearExtension, IntoMLE};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::marker::PhantomData;
use transcript::Transcript;

use crate::{quantization::BIT_LEN, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx,
};

use anyhow::{Result, anyhow, ensure};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation<N> {
    Relu(Relu),
    Gelu(GELU<N>),
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation<Element>,
    pub node_id: NodeId,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ActivationProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the relu
    pub(crate) lookup: LogUpProof<E>,
    /// The witness commitments from this function
    pub(crate) commits: Vec<PCS::Commitment>,
}

impl<N> OpInfo for Activation<N> {
    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        match self {
            Activation::Relu(_relu) => format!("RELU: {}", 1 << Relu::num_vars()),
            Activation::Gelu(_gelu) => "GELU".to_string(),
        }
    }

    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec() // same as input shapes
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<f32> for Activation<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<f32, E>> {
        match self {
            Activation::Relu(relu) => Ok(LayerOut::from_vec(
                inputs
                    .iter()
                    .map(|input| relu.op(input))
                    .collect::<Vec<_>>(),
            )),
            Activation::Gelu(gelu) => gelu.evaluate::<E>(inputs, _unpadded_input_shapes),
        }
    }
}

impl QuantizeOp for Activation<f32> {
    type QuantizedOp = Activation<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let q_op = match self {
            Activation::Relu(_) => Activation::Relu(Relu),
            Activation::Gelu(_) => Activation::Gelu(GELU::new()),
        };
        Ok(QuantizeOutput::new(q_op, input_scaling.to_vec()))
    }
}

impl Evaluate<Element> for Activation<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<Element, E>> {
        let outputs = match self {
            Activation::Relu(relu) => inputs
                .iter()
                .map(|input| relu.op(input))
                .collect::<Vec<_>>(),
            Activation::Gelu(_) => unimplemented!("GELU not implemented for Element"),
        };
        Ok(LayerOut::from_vec(outputs))
    }
}

impl<N, E> ProveInfo<E> for Activation<N>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        aux.tables.insert(TableType::Relu);

        // `try_fold` would not allow returning of `Err` values from here and would short-circuit
        // instead of looping over all values in the iterator
        #[allow(clippy::manual_try_fold)]
        let num_vars = aux
            .last_output_shape
            .iter_mut()
            .fold(Ok(None), |expected_num_vars, shape| {
                let num_vars = shape.iter().map(|dim| ceil_log2(*dim)).sum::<usize>();
                if let Some(vars) = expected_num_vars? {
                    ensure!(
                        vars == num_vars,
                        "All input shapes for activation must have the same number of variables"
                    );
                }
                Ok(Some(num_vars))
            })?
            .expect("No input shape found for activation layer?");
        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });
        let info = match self {
            Activation::Relu(relu) => LayerCtx::Activation(ActivationCtx {
                op: Activation::Relu(*relu),
                node_id: id,
                num_vars,
            }),
            Activation::Gelu(_) => unimplemented!("GELU not implemented for Element"),
        };
        Ok((info, aux))
    }
}

impl<N> PadOp for Activation<N> {}

impl<N, E, PCS> ProvableOp<E, PCS> for Activation<N>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = ActivationCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.prove_step(
            prover,
            last_claims[0],
            step_data.outputs.outputs()[0].get_data(),
            ctx,
            id,
        )?])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        gen: &mut LookupWitnessGen<E, PCS>,
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<()> {
        ensure!(
            step_data.inputs.len() == 1,
            "Found more than 1 input tensor in inference step of activation layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        // Calculate the column_evals and also the merged lookups
        #[allow(clippy::type_complexity)]
        let (merged_lookups, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) = step_data
            .inputs[0]
            .get_data()
            .iter()
            .zip(step_data.outputs.outputs()[0].get_data().iter())
            .map(|(a, b)| {
                let a_field: E = a.to_field();
                let b_field: E = b.to_field();
                (
                    a + COLUMN_SEPARATOR * b,
                    (a_field.as_bases()[0], b_field.as_bases()[0]),
                )
            })
            .unzip();

        let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) = field.into_iter().unzip();

        let num_vars = ceil_log2(col_one.len());

        // Add the witness polynomials that we need to commit to
        #[allow(clippy::type_complexity)]
        let (commits, column_evals): (
            Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
            Vec<Vec<E::BaseField>>,
        ) = [col_one, col_two]
            .into_par_iter()
            .map(|evaluations| {
                let mle =
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &evaluations);
                let commit = ctx.commitment_ctx.commit(&mle)?;
                Ok(((commit, mle), evaluations))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();
        gen.logup_witnesses.insert(
            id,
            vec![LogUpWitness::<E, PCS>::new_lookup(
                commits,
                column_evals,
                2,
                TableType::Relu,
            )],
        );

        let lookups = gen
            .new_lookups
            .get_mut(&TableType::Relu)
            .ok_or(anyhow!("No table of type Relu was expected"))?;
        lookups.extend(merged_lookups);

        Ok(())
    }
}

impl OpInfo for ActivationCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        self.op.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        self.op.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.op.describe()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for ActivationCtx
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = ActivationProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .get_challenges_by_name(&TableType::Relu.name())
            .ok_or(anyhow!(
                "Couldn't get challenges for LookupType: {}",
                TableType::Relu.name()
            ))?;
        Ok(vec![self.verify_activation(
            verifier,
            last_claims[0],
            proof,
            constant_challenge,
            column_separation_challenge,
        )?])
    }
}

impl<N> Activation<N> {
    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        output: &[E],
        _step: &ActivationCtx,
        node_id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        if let Activation::Gelu(_) = self {
            unimplemented!("GELU not implemented for Element");
        }
        // Should only be one prover_info for this step
        let logup_witnesses = prover.lookup_witness(node_id)?;
        if logup_witnesses.len() != 1 {
            return Err(anyhow!(
                "Activation only requires a lookup into one table type, but node: {} had {} lookup witnesses",
                node_id,
                logup_witnesses.len()
            ));
        }
        let logup_witness = &logup_witnesses[0];
        // Run the lookup protocol and return the lookup proof
        let prover_info = logup_witness.get_logup_input(&prover.challenge_storage)?;

        let commits = logup_witness.get_commitments();
        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        same_poly_prover.add_claim(last_claim.clone())?;
        // Activation proofs have two columns, input and output
        let input_claim = logup_proof.output_claims()[0].clone();
        let output_claim = logup_proof.output_claims()[1].clone();

        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(prover.transcript)?;

        // Add commitment claims to prover
        let commits = [input_claim.clone(), claim_acc_proof.extract_claim()]
            .into_iter()
            .zip(commits)
            .map(|(claim, comm_with_wit)| {
                let comm = PCS::get_pure_commitment(&comm_with_wit.0);
                prover
                    .commit_prover
                    .add_witness_claim(comm_with_wit, claim)?;
                Ok(comm)
            })
            .collect::<Result<Vec<PCS::Commitment>, anyhow::Error>>()?;

        // Add the proof in
        prover.push_proof(
            node_id,
            LayerProof::Activation(ActivationProof {
                io_accumulation: claim_acc_proof,
                lookup: logup_proof,
                commits,
            }),
        );
        Ok(input_claim)
    }
}

impl ActivationCtx {
    pub(crate) fn verify_activation<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &ActivationProof<E, PCS>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
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
            verifier.transcript,
        )?;

        // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
        let sp_ctx = same_poly::Context::<E>::new(self.num_vars);
        let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
        sp_verifier.add_claim(last_claim.clone())?;
        verifier_claims.claims()[1..]
            .iter()
            .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

        let new_output_claim = sp_verifier.verify(&proof.io_accumulation, verifier.transcript)?;
        // 3. Accumulate the new claim into the witness commitment protocol
        verifier_claims
            .claims()
            .iter()
            .take(1)
            .cloned()
            .chain(std::iter::once(new_output_claim))
            .zip(proof.commits.iter())
            .try_for_each(|(claim, commit)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(commit.clone(), claim)
            })?;

        // 4. return the input claim for to be proven at subsequent step
        Ok(verifier_claims.claims()[0].clone())
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Relu;

impl Default for Relu {
    fn default() -> Self {
        Self::new()
    }
}

impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        *BIT_LEN
    }
    pub fn poly_len() -> usize {
        1 << Self::num_vars()
    }
    pub fn shape() -> Shape {
        Shape::new(vec![2, Self::poly_len()])
    }

    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Tensor<T> {
        Tensor::new(
            input.get_shape(),
            input
                .get_data()
                .par_iter()
                .map(|e| Self::apply(*e))
                .collect::<Vec<_>>(),
        )
    }

    #[inline(always)]
    pub fn apply<T: Number>(e: T) -> T {
        if e.is_negative() { T::default() } else { e }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GELU<N> {
    _n: PhantomData<N>,
}

impl<N> Default for GELU<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N> GELU<N> {
    pub fn new() -> Self {
        Self { _n: PhantomData }
    }
}

impl Evaluate<f32> for GELU<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let output_tensors: Vec<Tensor<f32>> = inputs
            .par_iter()
            .map(|t| {
                let d = t.get_data();
                let gelued = d
                    .iter()
                    .map(|&x| {
                        let x_cubed = x * x * x;
                        let inner_term =
                            (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x_cubed);
                        0.5 * x * (1.0 + inner_term.tanh())
                    })
                    .collect::<Vec<_>>();
                Tensor::new(t.get_shape(), gelued)
            })
            .collect();
        Ok(LayerOut::from_vec(output_tensors))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_activation_relu_apply() {
        struct TestCase {
            input: Element,
            output: Element,
        }

        impl TestCase {
            pub fn from(input: Element, output: Element) -> Self {
                Self { input, output }
            }
        }
        for case in [
            TestCase::from(-24, 0),
            TestCase::from(0, 0),
            TestCase::from(124, 124),
            TestCase::from(-127, 0),
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }

    #[test]
    fn test_activation_gelu_evaluate_f32() -> anyhow::Result<()> {
        let gelu = GELU::<f32>::new();
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let input_tensor = Tensor::new(vec![1, input_data.len()].into(), input_data.clone());

        // Expected values calculated using the GELU approximation
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let expected_output_data = vec![
            -0.045500278, // GELU(-2.0)
            -0.15865526,  // GELU(-1.0)
            0.0,          // GELU(0.0)
            0.8413447,    // GELU(1.0)
            1.9544997,    // GELU(2.0)
            2.9963627,    // GELU(3.0)
        ];

        let layer_out = gelu.evaluate::<GoldilocksExt2>(&[&input_tensor], vec![])?;
        assert_eq!(layer_out.outputs().len(), 1);
        let output_tensor = &layer_out.outputs()[0];

        assert_eq!(output_tensor.get_shape(), vec![1, input_data.len()].into());
        let actual_output_data = output_tensor.get_data();

        actual_output_data
            .iter()
            .zip(expected_output_data.iter())
            .for_each(|(actual, expected)| {
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "Actual: {}, Expected: {}",
                    actual,
                    expected
                );
            });
        Ok(())
    }
}
