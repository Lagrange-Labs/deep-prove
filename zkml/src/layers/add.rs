use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::VPAuxInfo,
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
    quantization::split_scale_into_multiplier,
    tensor::{Number, Shape},
};

use super::provable::LayerOut;
const OPERAND_POLY_ID: u64 = 0xff;

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
        let result = left_scaled.add(&right_scaled);
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
        "Add".to_string()
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
    shift1: usize,
    m2: f32,
    shift2: usize,
}

impl QuantInfo {
    pub fn left_scale(&self) -> Element {
        (self.m1 * 2f32.powf(self.shift2 as f32)) as Element
    }
    pub fn right_scale(&self) -> Element {
        (self.m2 * 2f32.powf(self.shift1 as f32)) as Element
    }
    // returns the denominator, what the requant layer will have to perform as shift
    pub fn global_shift(&self) -> Element {
        2u32.pow((self.shift1 + self.shift2) as u32) as Element
    }
}

/// Normally, scaling add is done by scaling both inputs, so requant should happen _before_ the add.
/// y = (s1 * x1 + s2 * x2) / s3 where s1 is the left input scaling factor, s2 is the right input scaling factor,
/// and s3 is the output scaling factor.
/// In quantized world, we approximate s1' = s1 / s3 = M1 / 2^shift1 and s2' = s2 / s3 = M2 / 2^shift2
/// so we can rewrite the equation as:
/// y = (x1 * M1 / 2^shift1 + x2 * M2 / 2^shift2)
/// so if we put under a common denomiator we have:
/// y = (x1 * M1 * 2^shift2 + x2 * M2 * 2^shift1) / (2^{shift1 + shift2})
///
/// Since the numerators are constant, we can just multiply the claims by the respective M1,M2, and shift1 and shift2
/// and have only a single requant layer after to apply the denominator.
///
/// NOTE: in the case there is a right operand, then we need to quantize the right operand, as in a dense layer.
impl Add<f32> {
    fn quantize(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Add<Element>>> {
        let left_scaling = input_scaling[0];
        // s1p = M1 / 2^shift1
        let s1p = left_scaling.scale() / output_scaling.scale();
        let (shift1, m1) = split_scale_into_multiplier(s1p);
        // s2p = M2 / 2^shift2
        let right_scaling = match self.operand {
            Some((ref t, _)) => ScalingFactor::from_tensor(t, None),
            None => input_scaling[1],
        };
        let s2p = right_scaling.scale() / output_scaling.scale();
        let (shift2, m2) = split_scale_into_multiplier(s2p);
        let quant_info = QuantInfo {
            m1,
            shift1,
            m2,
            shift2,
        };
        let quantized_model = Add::<Element> {
            operand: self.operand.map(|(t, s)| (t.quantize(&right_scaling), s)),
            quant_info: Some(quant_info.clone()),
        };
        // we assume the inputs are quantized between [MIN, MAX] so add only produces values between [2 * MIN, 2 * MAX]
        // since we do symmetric quantization, we can take min. However, we also need to take into account M1*shift2 and M2*shift1
        let left_shift: f32 = m1 * shift2 as f32;
        let right_shift: f32 = m2 * shift1 as f32;
        let max = (left_shift as Element).max(right_shift as Element);
        let maxlog = if max > 1 { max.ilog2() as usize } else { 1 };
        // so we do gross estimation of x1 * max + x2 * max => which in bit size means (MIN.log2() + maxlog) + 1
        // we also append a final +1 to void offsetted values being zeros
        let intermediate_bit_size = (crate::quantization::MAX.ilog2() as usize + maxlog) + 1 + 1;
        // now we need to prepare the requant layer's scaling. The requant layer performs s1 * s2 / s3
        // we want the requant to perform 1 * 1 / 2^{shift1 + shift2} so s1=1, s2=1, s3=1 /2^{shift1 + shift2}
        let os1 = ScalingFactor::from_scale(1.0, None);
        let os2 = ScalingFactor::from_scale(1.0, None);
        // recip => 1 / 2^{shift1 + shift2}
        let os3 = ScalingFactor::from_scale((quant_info.global_shift() as f32).recip(), None);
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
        let Some(ref quant_info) = self.quant_info else {
            bail!("Add layer is not quantized");
        };
        // assuming last_claim is f(r) = y
        // we want to prove that x1(r) + x2(r) = y
        // in the case there is no operand, we output two claims, x1(r) and x2(r)
        // in the case there is an operand, we output one claim, x1(r) and we
        // add the claim OPERAND(r) to the list of claims to verify via the committed weights PCS.
        // Regarding the scaling operation, we actually want to prove
        // that x1(r) * M1 / 2^shift1 + x2(r) * M2 / 2^shift2 = y, so the prover outputs only x1(r) and x2(r)
        // and the verifier will "scale" the claims accordingly to check the equation.
        let output = step_data.outputs.outputs()[0];
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
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(last_claims.len() == 1, "Add layer expects 1 claim");
        let last_claim = last_claims[0];
        // just making sure downsizing due to API of E is ok
        ensure!((self.quant_info.left_scale() as u64) as Element == self.quant_info.left_scale());
        ensure!((self.quant_info.right_scale() as u64) as Element ==  self.quant_info.right_scale());
        // we have the output claim f(r) = y = x1(r) * x1_scale + x2(r) * x2_scale
        // and the proof gives us x1(r) and x2(r) so we just need to "scale" these and
        // verify the equation.
        let scaled_left = proof.left_eval * E::from_canonical_u64(self.quant_info.left_scale() as u64);
        let left_claim = Claim::new(last_claim.point.clone(), scaled_left);
        let scaled_right = proof.right_eval * E::from_canonical_u64(self.quant_info.right_scale() as u64);
        let right_claim = Claim::new(last_claim.point.clone(), scaled_right);
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

    use crate::{Element, quantization, tensor::is_close_with_tolerance, testing::vector_in_range};

    use super::*;

    #[test]
    fn test_add() {
        let add = Add::new();
        let t1 = Tensor::<Element>::random(&vec![2, 2].into());
        let t2 = Tensor::<Element>::random(&vec![2, 2].into());
        let result = add
            .evaluate::<GoldilocksExt2>(&[&t1, &t2], vec![vec![2, 2].into(), vec![2, 2].into()])
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    result.outputs[0].get(vec![i, j]),
                    t1.get(vec![i, j]) + t2.get(vec![i, j])
                );
            }
        }
        let add = Add::new_with(t1.clone(), t1.get_shape().into());
        let result = add
            .evaluate::<GoldilocksExt2>(&[&t2], vec![vec![2, 2].into()])
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    result.outputs[0].get(vec![i, j]),
                    t1.get(vec![i, j]) + t2.get(vec![i, j])
                );
            }
        }
    }

    #[test]
    fn test_add_quantization() {
        let add = Add::<f32>::new();
        let t1 = Tensor::<f32>::new(vec![2, 2].into(), vector_in_range(4));
        let t2 = Tensor::<f32>::new(vec![2, 2].into(), vector_in_range(4));
        let t3 = t1.add(&t2);
        let s1 = ScalingFactor::from_tensor(&t1, None);
        let s2 = ScalingFactor::from_tensor(&t2, None);
        let s3 = ScalingFactor::from_tensor(&t3, None);
        let qt1 = t1.quantize(&s1);
        let qt2 = t2.quantize(&s2);
        let qadd = add.quantize(&[s1, s2], s3).unwrap().quantized_op;
        let qadd_result = qadd
            .evaluate::<GoldilocksExt2>(&[&qt1, &qt2], vec![vec![2, 2].into(), vec![2, 2].into()])
            .unwrap();
        let shift = qadd.quant_info.as_ref().unwrap().global_shift();
        let ishift = (shift as f32).recip();
        // we divide by 2^{shift1 + shift2} to get the result in the original scale
        // we pass by float first otherwise the inverse of 2^{shift1 + shift2} is just zero
        let result_scaled = Tensor::<Element>::new(
            qadd_result.outputs()[0].get_shape(),
            qadd_result.outputs()[0]
                .get_data()
                .iter()
                .map(|x| ((*x as f32 * ishift) as Element))
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
            is_close_with_tolerance(computed_result.get_data(), t3.get_data(), 1e-2_f32, 0.3);
        assert!(within_range, "output is not within range");
        assert!(close_to_float, "output is not close to float");
    }
}
