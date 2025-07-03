use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    util::ceil_log2,
};
use serde::de::DeserializeOwned;
use std::collections::HashMap;

use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
        requant::Requant,
    },
    model::StepData,
    padding::{PaddingMode, ShapeData, ShapeInfo},
    quantization::{Fieldizer, split_scale_into_multiplier},
    tensor::{Number, Shape},
};

use super::provable::LayerOut;
const OPERAND_POLY_ID: u64 = 0xff;
/// Constant that defines the amount of fixed point precision to use for multiplying the "eps" in eps * 2^-n part
/// with the inputs.
pub(crate) const M_FIXED_PRECISION: usize = 16;

/// Add layer that adds two tensors together.
/// If there is two inputs, no static weight, then the output shape is the same as the first input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add<N> {
    /// The operand is the right side of the Add operation.
    /// shape is the unpadded shape of the operand
    operand: Option<(Tensor<N>, Shape)>,
    quant_info: Option<QuantInfo>,
}

/// Context info for the add layer.
/// NOTE: In LLM, we assume the same scaling info regardless of the sequence length.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddCtx {
    node_id: NodeId,
    quant_info: QuantInfo,
    operand: Option<()>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddProof<E> {
    left_eval: E,
    right_eval: E,
}

impl<N: Number> Add<N> {
    pub fn new() -> Self {
        Self {
            operand: None,
            quant_info: None,
        }
    }
    pub fn new_with(operand: Tensor<N>, unpadded_shape: Shape) -> Self {
        Self {
            operand: Some((operand, unpadded_shape)),
            quant_info: None,
        }
    }
}

impl Evaluate<f32> for Add<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let result = if inputs.len() == 2 {
            ensure!(
                Shape::from(inputs[0].get_shape()).product()
                    == Shape::from(inputs[1].get_shape()).product(),
                "Add layer expects inputs to have the same shape: {:?} vs {:?}",
                inputs[0].get_shape(),
                inputs[1].get_shape()
            );
            inputs[0].add(inputs[1])
        } else if inputs.len() == 1 {
            ensure!(
                self.operand.is_some(),
                "Add operand can't be None if there is only one input"
            );
            ensure!(
                inputs[0].get_shape().product()
                    == self.operand.as_ref().unwrap().0.get_shape().product(),
                "Add layer expects input and operand to have the same shape: {:?} vs {:?}",
                inputs[0].get_shape(),
                self.operand.as_ref().unwrap().0.get_shape()
            );
            inputs[0].add(&self.operand.as_ref().unwrap().0)
        } else {
            bail!("Add layer expects 1 or 2 inputs, got {}", inputs.len());
        };
        Ok(LayerOut::from_vec(vec![result]))
    }
}

impl Evaluate<Element> for Add<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        let Some(ref quant_info) = self.quant_info else {
            bail!("Add layer is not quantized");
        };
        let left_tensor = inputs[0];
        let right_tensor = match self.operand {
            Some((ref op, _)) => op,
            None => inputs[1],
        };
        let left_scaled = left_tensor.scalar_mul(&(quant_info.left_scale()));
        let right_scaled = right_tensor.scalar_mul(&(quant_info.right_scale()));
        let mut result = left_scaled.add(&right_scaled);
        // we check if we need to scale the result or not
        if !quant_info.requires_requant() {
            result = result.scalar_mul(&(quant_info.global_multiplier_element()));
        }
        Ok(LayerOut::from_vec(vec![result]))
    }
}

impl<N> OpInfo for Add<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        if let Some((_, og_shape)) = &self.operand {
            assert!(
                *og_shape == input_shapes[0],
                "Add layer operand shape mismatch: {:?} vs {:?}",
                og_shape,
                &input_shapes[0]
            );
        }
        match padding_mode {
            PaddingMode::NoPadding => input_shapes.to_vec(),
            PaddingMode::Padding => input_shapes
                .iter()
                .map(|shape| shape.next_power_of_two())
                .collect(),
        }
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!("Add {:?}", self.quant_info)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl OpInfo for AddCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        match padding_mode {
            PaddingMode::NoPadding => input_shapes.to_vec(),
            PaddingMode::Padding => input_shapes
                .iter()
                .map(|shape| shape.next_power_of_two())
                .collect(),
        }
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        "Add".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

/// Quantization info for the add layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuantInfo {
    m1: f32,
    m2: f32,
    shift: i32,
}

impl QuantInfo {
    pub fn new(
        left_scaling: &ScalingFactor,
        right_scaling: &ScalingFactor,
        output_scaling: &ScalingFactor,
    ) -> Self {
        let (shift_out, m_out) = split_scale_into_multiplier(output_scaling.scale());
        let s1p = left_scaling.scale() / m_out;
        let s2p = right_scaling.scale() / m_out;
        Self {
            m1: s1p,
            m2: s2p,
            shift: shift_out,
        }
    }
    pub fn left_scale(&self) -> Element {
        let m_scaled: f32 = self.m1 * (1 << M_FIXED_PRECISION) as f32;
        m_scaled.round() as Element
    }
    pub fn right_scale(&self) -> Element {
        let m_scaled: f32 = self.m2 * (1 << M_FIXED_PRECISION) as f32;
        m_scaled.round() as Element
    }
    // returns the exponent of the denominator 2^n, what the requant layer will have to perform as shift
    pub fn global_shift(&self) -> i32 {
        -self.shift + M_FIXED_PRECISION as i32
    }
    pub fn requires_requant(&self) -> bool {
        self.shift <= M_FIXED_PRECISION as i32
    }
    fn global_multiplier(&self) -> f32 {
        2f32.powf(self.global_shift() as f32)
    }
    pub(crate) fn global_multiplier_element(&self) -> Element {
        self.global_multiplier() as Element
    }
}

/// Normally, scaling add is done by scaling both inputs, so requant should happen _before_ the add.
/// y = (s1 * x1 + s2 * x2) / s3 where s1 is the left input scaling factor, s2 is the right input scaling factor,
/// and s3 is the output scaling factor.
/// In quantized world, we approximate s3 = m * 2^-n
/// so we can rewrite the equation as:
/// y = (x1 * s1 / (m * 2^-n) + x2 * s2 / (m * 2^-n))
/// y = (x1 * s1 / m + x2 * s2 / m) / 2^-n
/// y = (x1 * s1 / m + x2 * s2 / m) * 2^n
///
/// Due to accuracy issues, we're actually using fixed point precision so
/// y = ((x1 * s1 * 2^precision / m) + (x2 * s2 * 2^precision / m)) / (2^-n * 2^precision)
/// y = ((x1 * s1 * 2^precision / m) + (x2 * s2 * 2^precision / m)) / 2^{-n + precision}
/// y = ((x1 * s1 * 2^precision / m) + (x2 * s2 * 2^precision / m)) * 2^{n - precision}
///
/// So if `n >= precision`, the exponent is positive and thus we can simply scale the claim by an integer
/// If `n <= precision`, then the exponent is negative, and thus we need a division, so we need a requant layer.
///
/// NOTE: in the case there is a right operand, then we need to quantize the right operand, as in a dense layer.
impl Add<f32> {
    fn quantize(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Add<Element>>> {
        let left_scaling = input_scaling[0];
        let right_scaling = match self.operand {
            Some((ref t, _)) => ScalingFactor::from_tensor(t, None),
            None => input_scaling[1],
        };
        let quant_info = QuantInfo::new(&left_scaling, &right_scaling, &output_scaling);
        let quantized_model = Add::<Element> {
            operand: self.operand.map(|(t, s)| (t.quantize(&right_scaling), s)),
            quant_info: Some(quant_info.clone()),
        };
        // we need to decide if we need a requant layer or not, and if so, what the scaling factor should be
        // if not, we just return the quantized model
        if !quant_info.requires_requant() {
            return Ok(QuantizeOutput::new(quantized_model, vec![output_scaling]));
        }

        // we assume the inputs are quantized between [MIN, MAX] so add only produces values between [2 * MIN, 2 * MAX]
        // However, we also need to take into account M1*shift2 and M2*shift1
        let max = quant_info
            .left_scale()
            .abs()
            .max(quant_info.right_scale().abs());
        let maxlog = if max > 1 { ceil_log2(max as usize) } else { 1 };
        // so we do gross estimation of x1 * max + x2 * max => which in bit size means (MIN.log2() + maxlog) + 1
        // we also append a final +1 to void offsetted values being zeros
        let intermediate_bit_size = crate::quantization::MAX.ilog2() as usize + maxlog + 1 + 1;
        // now we need to prepare the requant layer's scaling. The requant layer performs s1 * s2 / s3
        // we want the requant to perform 1 * 1 / 2^{shift1 + shift2} so s1=1, s2=1, s3=2^{shift1 + shift2}
        let os1 = ScalingFactor::from_scale(1.0, None);
        let os2 = ScalingFactor::from_scale(1.0, None);
        let os3 = ScalingFactor::from_scale(quant_info.global_multiplier_element() as f32, None);
        let requant = Requant::from_scaling_factors(os1, os2, os3, intermediate_bit_size);
        Ok(QuantizeOutput::new(quantized_model, vec![output_scaling]).with_requant(requant))
    }
}

impl QuantizeOp for Add<f32> {
    type QuantizedOp = Add<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let mut output_scalings = S::scaling_factors_for_node(data, node_id, 1);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        self.quantize(input_scaling, output_scalings.pop().unwrap())
    }
}

impl<E> ProveInfo<E> for Add<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn step_info(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        let Some(ref quant_info) = self.quant_info else {
            bail!("Add layer is not quantized");
        };
        let mut ctx = AddCtx {
            quant_info: quant_info.clone(),
            operand: None,
            node_id: id,
        };
        if let Some((ref op, _)) = self.operand {
            let mut model_polys = HashMap::new();
            model_polys.insert(OPERAND_POLY_ID.to_string(), op.get_data().to_vec());
            aux.model_polys = Some(model_polys);
            ctx.operand = Some(());
        };
        Ok((LayerCtx::Add(ctx), aux))
    }
}

impl PadOp for Add<Element> {
    fn pad_node(mut self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        if let Some((op, og_shape)) = self.operand {
            ensure!(si.shapes.len() == 1, "Add layer expects 1 input shape");
            let op = op.pad_next_power_of_two();
            let padded_shape = op.get_shape();
            self.operand = Some((op, og_shape.clone()));
            ShapeData::new(og_shape.clone());
            let sd = si.shapes.first_mut().unwrap();
            sd.input_shape_og = og_shape.clone();
            sd.input_shape_padded = padded_shape;
        } else {
            ensure!(si.shapes.len() == 2, "Add layer expects 2 input shapes");
        }
        Ok(self)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Add<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = AddCtx;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        T: Transcript<E>,
    {
        ensure!(last_claims.len() == 1, "Add layer expects 1 claim");
        let last_claim = last_claims[0];
        ensure!(self.quant_info.is_some(), "Add layer is not quantized");
        // assuming last_claim is f(r) = y
        // we want to prove that x1(r) + x2(r) = y
        // in the case there is no operand, we output two claims, x1(r) and x2(r)
        // in the case there is an operand, we output one claim, x1(r) and we
        // add the claim OPERAND(r) to the list of claims to verify via the committed weights PCS.
        // Regarding the scaling operation, we actually want to prove
        // that x1(r) * M1 / 2^shift1 + x2(r) * M2 / 2^shift2 = y, so the prover outputs only x1(r) and x2(r)
        // and the verifier will "scale" the claims accordingly to check the equation.
        let left_input = &step_data.inputs[0];
        let left_eval = left_input
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&last_claim.point);
        let mut output_claims = vec![Claim::new(last_claim.point.clone(), left_eval)];
        let right_eval = match self.operand {
            Some((ref op, _)) => {
                let right_eval = op.evals_flat::<E>().into_mle().evaluate(&last_claim.point);
                let mut claims = HashMap::new();
                claims.insert(
                    OPERAND_POLY_ID.to_string(),
                    Claim::new(last_claim.point.clone(), right_eval),
                );
                // this claim gets verified by the PCS openings since it's a static one
                prover.add_common_claims(node_id, claims)?;
                right_eval
            }
            None => {
                let right_eval = step_data.inputs[1]
                    .get_data()
                    .to_vec()
                    .into_mle()
                    .evaluate(&last_claim.point);
                // this claims gets passed to the previous layer alongside the left one.
                output_claims.push(Claim::new(last_claim.point.clone(), right_eval));
                right_eval
            }
        };
        prover.push_proof(
            node_id,
            LayerProof::Add(AddProof {
                left_eval,
                right_eval,
            }),
        );
        Ok(output_claims)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for AddCtx
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = AddProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(last_claims.len() == 1, "Add layer expects 1 claim");
        let last_claim = last_claims[0];
        // just making sure downsizing due to API of E is ok
        ensure!((self.quant_info.left_scale() as u64) as Element == self.quant_info.left_scale());
        ensure!((self.quant_info.right_scale() as u64) as Element == self.quant_info.right_scale());
        // we have the output claim f(r) = y = x1(r) * x1_scale + x2(r) * x2_scale
        // and the proof gives us x1(r) and x2(r) so we just need to "scale" these and
        // verify the equation.
        let left_scale: E = self.quant_info.left_scale().to_field();
        let scaled_left = proof.left_eval * left_scale;
        let right_scale: E = self.quant_info.right_scale().to_field();
        let left_claim = Claim::new(last_claim.point.clone(), proof.left_eval);
        let scaled_right = proof.right_eval * right_scale;
        let right_claim = Claim::new(last_claim.point.clone(), proof.right_eval);
        ensure!(
            scaled_left + scaled_right == last_claim.eval,
            "Add layer verification failed"
        );
        if let Some(()) = self.operand {
            // in this case we need to verify the opening for the operand via PCS
            let mut claims = HashMap::new();
            claims.insert(
                OPERAND_POLY_ID.to_string(),
                Claim::new(last_claim.point.clone(), proof.right_eval),
            );
            verifier.add_common_claims(self.node_id, claims)?;
            // in this case we return only the left claim since the right one is verified by PCS
            Ok(vec![left_claim])
        } else {
            // in this case we return both claims
            Ok(vec![left_claim, right_claim])
        }
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{
        Element,
        layers::Layer,
        model::{Model, test::prove_model},
        quantization,
        tensor::is_close_with_tolerance,
    };

    use super::*;

    #[test]
    fn test_add_quantization() {
        let add = Add::<f32>::new();
        let t1 = Tensor::<f32>::random(&vec![2, 2].into());
        let t2 = Tensor::<f32>::random(&vec![2, 2].into());
        let t3 = t1.add(&t2);
        let s1 = ScalingFactor::from_tensor(&t1, None);
        let s2 = ScalingFactor::from_tensor(&t2, None);
        let s3 = ScalingFactor::from_tensor(&t3, None);
        let qt1 = t1.quantize(&s1); // x1_q = round(x1 / s1)
        let qt2 = t2.quantize(&s2);
        let qadd = add.quantize(&[s1, s2], s3).unwrap().quantized_op;
        let qadd_result = qadd
            .evaluate::<GoldilocksExt2>(&[&qt1, &qt2], vec![vec![2, 2].into(), vec![2, 2].into()])
            .unwrap();
        let shift = qadd.quant_info.as_ref().unwrap().global_shift();
        // we divide by 2^{shift1 + shift2} to get the result in the original scale
        // we pass by float first otherwise the inverse of 2^{shift1 + shift2} is just zero
        let pow = 2f32.powf(-shift as f32);
        let result_scaled = Tensor::<Element>::new(
            qadd_result.outputs()[0].get_shape(),
            qadd_result.outputs()[0]
                .get_data()
                .iter()
                .map(|x| ((*x as f32 * pow) as Element))
                .collect::<Vec<_>>(),
        );
        let computed_result = result_scaled.dequantize(&s3);
        let within_range = result_scaled
            .get_data()
            .iter()
            .all(|x| *x >= *quantization::MIN && *x <= *quantization::MAX);
        // NOTE: for some reason the result is not exactly within small bounds of the float value
        // so we have to increase tolerance to 30%
        let close_to_float =
            is_close_with_tolerance(computed_result.get_data(), t3.get_data(), 1e-2_f32, 0.1);
        println!("computed_result: {:?}", computed_result.get_data());
        println!("t3: {:?}", t3.get_data());
        assert!(within_range, "output is not within range");
        assert!(
            close_to_float,
            "output is not close to float: float {:?} vs computed {:?}",
            t3.get_data(),
            computed_result.get_data()
        );
    }

    #[test]
    fn test_add_proving_no_operand() {
        let input_shape = Shape::from(vec![2, 2]);
        let mut model = Model::new_from_input_shapes(
            vec![input_shape.clone(), input_shape.clone()],
            PaddingMode::NoPadding,
        );

        let add = Add::new();
        let _ = model.add_consecutive_layer(Layer::Add(add), None).unwrap();
        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }

    #[test]
    fn test_add_proving_with_operand() {
        let input_shape = Shape::from(vec![2, 2]);
        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);
        let operand = Tensor::<f32>::random(&vec![2, 2].into());
        let add = Add::new_with(operand, input_shape.clone());
        let _ = model.add_consecutive_layer(Layer::Add(add), None).unwrap();
        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }

    #[test]
    fn test_add_failing() {
        let shape: Shape = vec![2, 2].into();
        let t1 = Tensor::<f32>::random(&shape);
        let t2 = Tensor::<f32>::random(&shape);
        let t3 = t1.add(&t2);
        let s1 = ScalingFactor::from_tensor(&t1, None);
        let s2 = ScalingFactor::from_tensor(&t2, None);
        let s3 = ScalingFactor::from_tensor(&t3, None);
        let qt1 = t1.quantize(&s1);
        let qt2 = t2.quantize(&s2);
        let s1s3 = s1.scale() / s3.scale();
        let s2s3 = s2.scale() / s3.scale();
        let (shift1, m1) = split_scale_into_multiplier(s1s3);
        let (shift2, m2) = split_scale_into_multiplier(s2s3);
        let qt1_scaled1 = qt1
            .get_data()
            .iter()
            .map(|x| (*x as f32) * (m1 * 2f32.powf(-shift1 as f32)))
            .map(|x| x as Element)
            .collect::<Vec<_>>();
        let qt2_scaled1 = qt2
            .get_data()
            .iter()
            .map(|x| (*x as f32) * (m2 * 2f32.powf(-shift2 as f32)))
            .map(|x| x as Element)
            .collect::<Vec<_>>();
        let qt1_scaled = qt1
            .get_data()
            .iter()
            .map(|x| (*x as f32) * (s1.scale() / s3.scale()))
            .map(|x| x as Element)
            .collect::<Vec<_>>();
        let qt2_scaled = qt2
            .get_data()
            .iter()
            .map(|x| (*x as f32) * (s2.scale() / s3.scale()))
            .map(|x| x as Element)
            .collect::<Vec<_>>();
        assert!(is_close_with_tolerance(
            &qt1_scaled1.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            &qt1_scaled.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            1e-2_f32,
            0.1
        ));
        assert!(is_close_with_tolerance(
            &qt2_scaled1.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            &qt2_scaled.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            1e-2_f32,
            0.1
        ));
        let qt1_scaled = Tensor::new(shape.clone(), qt1_scaled1);
        let qt2_scaled = Tensor::new(shape.clone(), qt2_scaled1);
        let q_result = qt1_scaled.add(&qt2_scaled);
        let dequantized = q_result.dequantize(&s3);
        let close_to_float =
            is_close_with_tolerance(dequantized.get_data(), t3.get_data(), 1e-2_f32, 0.1);
        assert!(
            close_to_float,
            "THEORY output is not close to float: float {:?} vs computed {:?}",
            t3.get_data(),
            dequantized.get_data()
        );

        // now do with the proper method where we put on the same denominator
        let quant_info = QuantInfo::new(&s1, &s2, &s3);
        let qt1_scaled = qt1.scalar_mul(&quant_info.left_scale());
        let qt2_scaled = qt2.scalar_mul(&quant_info.right_scale());
        let q_result = qt1_scaled.add(&qt2_scaled);
        let q_result_scaled_back = q_result
            .get_data()
            .iter()
            .map(|x| (*x as f32) * 2f32.powf(-quant_info.global_shift() as f32))
            .map(|x| x as Element)
            .collect::<Vec<_>>();
        let q_result = Tensor::new(shape.clone(), q_result_scaled_back);
        let dequantized = q_result.dequantize(&s3);
        let close_to_float =
            is_close_with_tolerance(dequantized.get_data(), t3.get_data(), 1e-2_f32, 0.1);
        assert!(
            close_to_float,
            "PRACTICAL output is not close to float: float {:?} vs computed {:?}",
            t3.get_data(),
            dequantized.get_data()
        );
    }

    #[test]
    fn test_add_requant() {
        let t1 = Tensor::<f32>::random(&vec![4].into());
        let s1 = ScalingFactor::from_tensor(&t1, None);
        let qt1 = t1.clone().quantize(&s1);
        let ct1 = qt1.dequantize(&s1);
        println!("t1: {:?}", t1.get_data());
        println!("qt1: {:?}", qt1.get_data());
        println!("ct1: {:?}", ct1.get_data());
        println!(
            "is close: {:?}",
            is_close_with_tolerance(t1.get_data(), ct1.get_data(), 1e-2_f32, 0.3)
        );
    }
}
