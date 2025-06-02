use crate::{
    Claim, Element, Prover,
    commit::same_poly,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{LayerCtx, LayerProof, PolyID},
    lookup::{
        context::{COLUMN_SEPARATOR, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
    },
    model::StepData,
    padding::PaddingMode,
    quantization::Fieldizer,
    tensor::Number,
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use multilinear_extensions::mle::IntoMLE;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{marker::PhantomData};
use transcript::Transcript;

use crate::{quantization::BIT_LEN, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx,
};

use anyhow::{Result, anyhow, ensure};

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    Relu(Relu),
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ActivationProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the relu
    pub(crate) lookup: LogUpProof<E>,
}

impl OpInfo for Activation {
    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!("RELU: {}", 1 << Relu::num_vars())
    }

    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        input_shapes.to_vec() // same as input shapes
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for Activation {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<N, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating activation layer"
        );
        let input = inputs[0];
        let output = match self {
            Activation::Relu(relu) => relu.op(input),
        };
        Ok(LayerOut::from_vec(vec![output]))
    }
}

impl<E> ProveInfo<E> for Activation
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: PolyID, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        aux.tables.insert(TableType::Relu);
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
        let info = match self {
            Activation::Relu(relu) => LayerCtx::Activation(ActivationCtx {
                op: Activation::Relu(*relu),
                poly_id: id,
                num_vars,
            }),
        };
        Ok((info, aux))
    }
}

impl PadOp for Activation {}

impl<E> ProvableOp<E> for Activation
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Ctx = ActivationCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T>,
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
        gen: &mut LookupWitnessGen<E>,
        step_data: &StepData<Element, E>,
    ) -> Result<()> {
        gen.tables.insert(TableType::Relu);

        ensure!(
            step_data.inputs.len() == 1,
            "Found more than 1 input tensor in inference step of activation layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        // Calculate the column_evals and also the merged lookups
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
        let table_lookup_map = gen.lookups.entry(TableType::Relu).or_default();

        merged_lookups
            .into_iter()
            .for_each(|lookup| *table_lookup_map.entry(lookup).or_insert(0u64) += 1);

        gen.polys_with_id.push((
            id as PolyID,
            step_data.outputs.outputs()[0]
                .get_data()
                .iter()
                .map(Fieldizer::<E>::to_field)
                .collect(),
        ));
        gen.lookups_no_challenges
            .insert(id, (vec![col_one, col_two], 2, TableType::Relu));

        Ok(())
    }
}

impl OpInfo for ActivationCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        Activation::Relu(Relu).output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Activation::Relu(Relu).num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        Activation::Relu(Relu).describe()
    }

    fn is_provable(&self) -> bool {
        Activation::Relu(Relu).is_provable()
    }
}

impl<E> VerifiableCtx<E> for ActivationCtx
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = ActivationProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .as_ref()
            .unwrap()
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

impl Activation {
    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: &Claim<E>,
        output: &[E],
        step: &ActivationCtx,
        node_id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let prover_info = prover.lookup_witness(node_id)?;

        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;
        // Activation proofs have two columns, input and output
        let input_claim = logup_proof.output_claims()[0].clone();
        let output_claim = logup_proof.output_claims()[1].clone();

        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;
        // order is (output,mult)
        prover
            .witness_prover
            .add_claim(step.poly_id, claim_acc_proof.extract_claim())?;

        // Add the proof in
        prover.push_proof(
            node_id,
            LayerProof::Activation(ActivationProof {
                io_accumulation: claim_acc_proof,
                lookup: logup_proof,
            }),
        );
        Ok(input_claim)
    }
}

impl ActivationCtx {
    pub(crate) fn verify_activation<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: &Claim<E>,
        proof: &ActivationProof<E>,
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
        verifier
            .witness_verifier
            .add_claim(self.poly_id, new_output_claim)?;

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
    pub fn shape() -> Vec<usize> {
        vec![2, Self::poly_len()]
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

pub struct GELU<N> {
    _n: PhantomData<N>,
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
        _unpadded_input_shapes: Vec<Vec<usize>>,
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
    use goldilocks::GoldilocksExt2;

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
        let input_tensor = Tensor::new(vec![1, input_data.len()], input_data.clone());

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

        assert_eq!(output_tensor.get_shape(), vec![1, input_data.len()]);
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
