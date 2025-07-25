use crate::{
    Claim, Context, Element, Prover, ScalingFactor,
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
    quantization,
    quantization::Fieldizer,
    tensor::{Number, Shape},
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::{DenseMultilinearExtension, IntoMLE};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, marker::PhantomData};
use transcript::Transcript;

use crate::{quantization::BIT_LEN, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx,
};

use anyhow::{Result, anyhow, bail, ensure};
const GELU_SCALE_EXP: usize = 12;
const GELU_SCALE_FACTOR: usize = 1 << GELU_SCALE_EXP;

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
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        let q_op = match self {
            Activation::Relu(_) => Activation::Relu(Relu),
            Activation::Gelu(g) => Activation::Gelu(g.quantize(input_scaling[0])?),
        };
        Ok(QuantizeOutput::new(q_op, output_scalings))
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
            Activation::Gelu(g) => inputs
                .iter()
                .map(|input| input.try_map(|e| g.apply(e)))
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(LayerOut::from_vec(outputs))
    }
}

impl<E> ProveInfo<E> for Activation<Element>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        match self {
            Activation::Relu(_) => aux.tables.insert(TableType::Relu),
            // TODO: if we want to save on memory, we can use a pointer to the vector instead
            Activation::Gelu(gelu) => aux
                .tables
                .insert(TableType::GELU(gelu.quant_data.clone().unwrap())),
        };

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
        let act = match self {
            Activation::Relu(relu) => Activation::Relu(*relu),
            Activation::Gelu(g) => Activation::Gelu(g.clone()),
        };
        Ok((
            LayerCtx::Activation(ActivationCtx {
                op: act,
                node_id: id,
                num_vars,
            }),
            aux,
        ))
    }
}

impl<N> PadOp for Activation<N> {}

impl<E, PCS> ProvableOp<E, PCS> for Activation<Element>
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
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        ensure!(
            step_data.inputs.len() == 1,
            "Found more than 1 input tensor in inference step of activation layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        let inputs = step_data.inputs[0].get_data();
        let outputs = step_data.outputs.outputs()[0].get_data();
        debug_assert_eq!(
            inputs.len(),
            outputs.len(),
            "Input and outputs must have the same length",
        );
        let size = inputs.len();

        let mut element_count = HashMap::<Element, u64>::new();
        let mut col_one = Vec::<E::BaseField>::with_capacity(size);
        let mut col_two = Vec::<E::BaseField>::with_capacity(size);
        for (a, b) in inputs.iter().zip(outputs.iter()) {
            let (a, a_field): (Element, E) = match self {
                Activation::Relu(_) => (*a, a.to_field()),
                Activation::Gelu(g) => {
                    let scaled = a * g.quant_data.as_ref().unwrap().multiplier;
                    assert!(
                        scaled >= g.quant_data.as_ref().unwrap().min
                            && scaled <= g.quant_data.as_ref().unwrap().max
                    );
                    (scaled, scaled.to_field())
                }
            };
            // Calculate the lookup element
            let el = a + COLUMN_SEPARATOR * b;
            *element_count.entry(el).or_default() += 1;

            // Calculate the column_evals
            let b_field: E = b.to_field();
            col_one.push(a_field.as_bases()[0]);
            col_two.push(b_field.as_bases()[0]);
        }

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

        let mut gen = LookupWitnessGen::<E, PCS>::default();
        gen.logup_witnesses.insert(
            id,
            vec![LogUpWitness::<E, PCS>::new_lookup(
                commits,
                column_evals,
                2,
                self.table_type(),
            )],
        );
        gen.element_count.insert(self.table_type(), element_count);

        Ok(gen)
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
        let table_type = match &self.op {
            Activation::Relu(_) => TableType::Relu,
            Activation::Gelu(g) => TableType::GELU(g.quant_data.clone().unwrap()),
        };
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .get_challenges_by_name(&table_type.name())
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
    fn table_type(&self) -> TableType {
        match self {
            Activation::Relu(_) => TableType::Relu,
            Activation::Gelu(g) => TableType::GELU(g.quant_data.clone().unwrap()),
        }
    }
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
        // Should only be one prover_info for this step
        let mut logup_witnesses = prover.lookup_witness(node_id)?;
        ensure!(
            logup_witnesses.len() == 1,
            "Activation only requires a lookup into one table type, but node: {} had {} lookup witnesses",
            node_id,
            logup_witnesses.len()
        );
        let logup_witness = logup_witnesses.pop().expect("Length was checked above");
        // Run the lookup protocol and return the lookup proof
        let prover_info = logup_witness.get_logup_input(&prover.challenge_storage)?;

        let commits = logup_witness.into_commitments();
        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        same_poly_prover.add_claim(last_claim.clone())?;
        // Activation proofs have two columns, input and output
        let input_claim = logup_proof.output_claims()[0].clone();
        let input_claim = match &self {
            Activation::Gelu(g) => {
                let m: E = g.quant_data.as_ref().unwrap().multiplier.to_field();
                let mi = m.inverse();
                let eval = input_claim.eval * mi;
                Claim::new(input_claim.point.clone(), eval)
            }
            _ => input_claim,
        };
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
        let input_claim = match &self.op {
            Activation::Relu(_) => verifier_claims.claims()[0].clone(),
            Activation::Gelu(g) => {
                let claim = &verifier_claims.claims()[0];
                let m: E = g.quant_data.as_ref().unwrap().multiplier.to_field();
                let mi = m.inverse();
                let eval = claim.eval * mi;
                Claim::new(claim.point.clone(), eval)
            }
        };
        Ok(input_claim)
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
    quant_data: Option<GELUQuantData>,
    _n: PhantomData<N>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GELUQuantData {
    /// The multiplier used to scale the input
    multiplier: Element,
    /// The minimum value of the input
    pub(crate) min: Element,
    /// The maximum value of the input
    pub(crate) max: Element,
}

impl GELUQuantData {
    pub fn table_size(&self) -> usize {
        (self.max - self.min + 1).ilog2() as usize
    }
    /// Returns the input indexes of the table and the corresponding output values
    pub fn table(&self) -> impl Iterator<Item = (Element, Element)> + use<'_> {
        (self.min..self.max).map(|i| (i, self.table_output(i)))
    }
    /// NOTE: this requires the scaled input
    pub fn table_output(&self, input: Element) -> Element {
        let float_input = input as f32 / GELU_SCALE_FACTOR as f32;
        let float_output = gelu_float(&float_input);
        (float_output * *quantization::MAX as f32).round() as Element
    }
}

impl<N> Default for GELU<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N> GELU<N> {
    pub fn new() -> Self {
        Self {
            quant_data: None,
            _n: PhantomData,
        }
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
                let gelued = d.iter().map(gelu_float).collect::<Vec<_>>();
                Tensor::new(t.get_shape(), gelued)
            })
            .collect();
        Ok(LayerOut::from_vec(output_tensors))
    }
}

fn gelu_float(x: &f32) -> f32 {
    let x_cubed = x * x * x;
    let inner_term = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x_cubed);
    0.5 * x * (1.0 + inner_term.tanh())
}
impl GELU<f32> {
    fn quantize(&self, input_scaling: ScalingFactor) -> anyhow::Result<GELU<Element>> {
        // so we want sf * SCALING = multiplier
        // then we construct the lookup table as  GELU(i / SCALING) * quantization::MAX for
        // all i in the range [-2^{7 + ceil_log2(multiplier)}, 2^{7 + ceil_log2(multiplier)}]
        // This is because the input is already requantized, and we're multipliying the input
        // by the multiplier during quantized inference such that the float input is scaled
        // to that number of bits. So with inputs of 2^7 max, multiplied by multiplier then
        // the output range is 2^{7 + ceil_log2(multiplier)}
        // During lookup, we basically scale down back to the original
        // float value, apply GELU and multiply by 128 which is right now the output maximum range.
        let multiplier = (GELU_SCALE_FACTOR as f32 * input_scaling.scale()).round() as Element;
        let table_min = -2i32.pow(7 + ceil_log2(multiplier as usize) as u32);
        let table_max = 2i32.pow(7 + ceil_log2(multiplier as usize) as u32);
        let table_size = table_max - table_min;
        assert!((table_size as usize).is_power_of_two());
        assert!(
            table_size <= 1 << 20,
            "Table size for GELU is too bigggg: {:?}",
            table_size.ilog2()
        );
        let qd = GELUQuantData {
            multiplier,
            min: table_min as Element,
            max: table_max as Element,
        };
        Ok(GELU {
            quant_data: Some(qd),
            _n: PhantomData,
        })
    }
}

impl GELU<Element> {
    fn apply(&self, input: &Element) -> anyhow::Result<Element> {
        let Some(ref quant_data) = self.quant_data else {
            bail!("GELU not quantized");
        };
        let scaled = input * quant_data.multiplier;
        let within_range = quant_data.min <= scaled && scaled <= quant_data.max;
        ensure!(within_range, "Input out of range");
        Ok(self.quant_data.as_ref().unwrap().table_output(scaled))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{
        Element,
        layers::Layer,
        model::{Model, test::prove_model},
    };

    use super::*;

    #[test]
    fn test_activation_gelu_proving() -> anyhow::Result<()> {
        let input_shape = vec![3, 5].into();
        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        model.add_consecutive_layer(
            Layer::Activation(Activation::Gelu(GELU::<f32>::new())),
            None,
        )?;
        model.route_output(None)?;
        prove_model(model).unwrap();
        Ok(())
    }

    #[test]
    fn test_activation_gelu_quantize() -> anyhow::Result<()> {
        let gelu = GELU::<f32>::new();
        let input_scaling = ScalingFactor::from_scale(1.0, None);
        _ = gelu.quantize(input_scaling)?;
        Ok(())
    }

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
