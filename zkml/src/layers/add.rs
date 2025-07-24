use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};
use serde::de::DeserializeOwned;
use std::{cmp::Ordering, collections::HashMap};

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
    quantization::Fieldizer,
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

impl<N: Number> Default for Add<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Context info for the add layer.
/// NOTE: In LLM, we assume the same scaling info regardless of the sequence length.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddCtx {
    node_id: NodeId,
    quant_info: QuantInfo,
    is_static_operand: bool,
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

impl Add<Element> {
    pub(crate) fn prove_step<
        A: AsRef<Tensor<E>>,
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        inputs: &[A],
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<(Vec<Claim<E>>, AddProof<E>)> {
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
        let left_input = inputs[0].as_ref();
        let left_eval = left_input
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&last_claim.point);
        let mut output_claims = vec![Claim::new(last_claim.point.clone(), left_eval)];
        let right_eval = match self.operand {
            Some((_, _)) => {
                // out = x1 * s1 + x2 * s2
                // so x1 = (out - x2 * s2) / s1
                let a: E = self.quant_info.as_ref().unwrap().left_scale().to_field();
                let left_side: E = left_eval * a;
                let right_side: E = last_claim.eval - left_side;
                let right_eval =
                    right_side / self.quant_info.as_ref().unwrap().right_scale().to_field();
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
                let right_eval = inputs[1]
                    .as_ref()
                    .get_data()
                    .to_vec()
                    .into_mle()
                    .evaluate(&last_claim.point);
                // this claims gets passed to the previous layer alongside the left one.
                output_claims.push(Claim::new(last_claim.point.clone(), right_eval));
                right_eval
            }
        };

        let proof = AddProof {
            left_eval,
            right_eval,
        };

        Ok((output_claims, proof))
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
                inputs[0].get_shape().product() == inputs[1].get_shape().product(),
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
        ensure!(!inputs.is_empty(), "Add layer expects at least 1 input");
        let left_tensor = inputs[0];
        let right_tensor = match self.operand {
            Some((ref op, _)) => op,
            None => {
                ensure!(
                    inputs.len() == 2,
                    "Add layer expects 2 inputs if there is no operand"
                );
                inputs[1]
            }
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
                input_shapes.len() == 1,
                "Add layer expects 1 input if there is an operand"
            );
            assert!(
                *og_shape == input_shapes[0],
                "Add layer operand shape mismatch: {:?} vs {:?}",
                og_shape,
                &input_shapes[0]
            );
        } else {
            assert!(
                input_shapes.len() == 2,
                "Add layer expects 2 inputs if there is no operand"
            );
            assert!(
                input_shapes[0] == input_shapes[1],
                "Add layer input shapes mismatch: {:?} vs {:?}",
                input_shapes[0],
                input_shapes[1]
            );
        }
        match padding_mode {
            PaddingMode::NoPadding => vec![input_shapes[0].clone()],
            PaddingMode::Padding => vec![input_shapes[0].next_power_of_two()],
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
/// When we perform quantised addition between two tensors A and B we need both tensors to be quantised with the same
/// [`ScalingFactor`]. Often this is not the case and so we use [`QuantInfo`] to calculate a suitable common [`ScaleFactor`].
#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuantInfo {
    /// This is the value we must multiply the left input by
    left_multiplier: Element,
    /// This is the value we must multiply the right input by
    right_multiplier: Element,
    /// This is the common scale factor that the output of the addition will have
    common_scale: f32,
    /// Lets us know if we need a requantisation step after the addition
    require_requant: bool,
}

impl QuantInfo {
    /// Calculates the relevant quantisation info to perform [`Add`]. We have to rescale both inputs to the same scaling factor, and then work out if we have to requantise afterwards.
    pub fn new(
        left_scaling: &ScalingFactor,
        right_scaling: &ScalingFactor,
        output_scaling: &ScalingFactor,
    ) -> Self {
        // The common scale factor needs to be worked out here from `left_scaling`, `right_scaling` and `output_scaling`
        // To do this we take the absoloute value of the base 2 logarithm of each (they should all be values in the interval [0,1), thus having negative base 2 logarithm)
        let left_log = left_scaling.scale().log2().abs();
        let right_log = right_scaling.scale().log2().abs();
        let output_log = output_scaling.scale().log2().abs();

        // We make a closure that we can use once we work out whic of the left and right inputs has higher precision. This closure takes both (once it knows which is high precision and which is low)
        // and calculates what we have to multiply the left input by, the right input by, the common scale factor after multiplying by both of these and also whether a requant step is required after the addition.
        let minimum_precision_diff = *crate::quantization::BIT_LEN as f32;
        let scale_comparison = |h_precision: ScalingFactor,
                                l_precision: ScalingFactor|
         -> (Element, Element, f32, bool) {
            let high_log = h_precision.scale().log2().abs();
            let low_log = l_precision.scale().log2().abs();
            match high_log.compare(&output_log) {
                Ordering::Less => {
                    // In this case the common scaling factor should be output_scaling as long as it is suitably more precise
                    // The rationale is that if we have `s3*y = s1*x1 + s2*x2` with `s1 = 2^-a`, `s2 = 2^-b` and `s3 = 2^-c` in this we have `c > a` and `c > b`.
                    // Since we want to work out `y = s1/s3 * x1 + s2/s3 * x2` then `s1/s3 = 2^-a/2^-c = 2^(c-a)`, similaraly `s2/s3 = 2^(c-b)`.
                    // What we are doing is checking that both `2^(c-a)` and `2^(c-b)` are bigger than `2^minimum_precision_diff` so that calling `(s1/s3).round()` and `(s2/s3).round()` retains a reasonable level of precision.
                    let (common_scale, require_requant) =
                        if output_log - high_log >= minimum_precision_diff {
                            // In this case the output scale factor is small enough such that `1/output_scale.scale()` is at least 2^minimum_precision_diff * (1 / h_precision.scale())`
                            // so we can scale up our inputs and don't have to requantise afterwards.
                            let common_scale = output_scaling.scale();
                            (common_scale, false)
                        } else {
                            // The output scaling factor wasn't suitably large so we take the ceiling of high_log and add `minimum_precision_diff`, this way we retain a reasonable
                            // amount of accuracy.
                            let ceiling_round = high_log.ceil() as usize;
                            let common_scale =
                                2.0f32.powf(-(ceiling_round as f32 + minimum_precision_diff));
                            (common_scale, true)
                        };

                    let left_rescale = (left_scaling.scale() / common_scale).round() as Element;
                    let right_rescale = (right_scaling.scale() / common_scale).round() as Element;
                    (left_rescale, right_rescale, common_scale, require_requant)
                }
                Ordering::Equal => {
                    // In this case the output scale factor is small enough such that `1/output_scale.scale()` is at least 2^minimum_precision_diff * (1 / h_precision.scale())`
                    // so we can scale up our inputs and don't have to requantise afterwards.
                    let (common_scale, require_requant) =
                        if output_log - low_log >= minimum_precision_diff {
                            let common_scale = output_scaling.scale();
                            (common_scale, false)
                        } else {
                            // The output scaling factor wasn't suitably large so we take the ceiling of high_log and add 8
                            let ceiling_round = high_log.ceil() as usize;
                            let common_scale =
                                2.0f32.powf(-(ceiling_round as f32 + minimum_precision_diff));
                            (common_scale, true)
                        };

                    let left_rescale = (left_scaling.scale() / common_scale).round() as Element;
                    let right_rescale = (right_scaling.scale() / common_scale).round() as Element;
                    (left_rescale, right_rescale, common_scale, require_requant)
                }
                Ordering::Greater => {
                    // In this case the common scaling factor should be high_log as long as it is suitably more precise than low_log
                    let (common_scale, require_requant) = if high_log - low_log >= 8.0f32 {
                        let common_scale = h_precision.scale();
                        (common_scale, true)
                    } else {
                        // The high scaling factor wasn't suitably large so we take the ceiling of high_log and add 8
                        let ceiling_round = high_log.ceil() as usize;
                        let common_scale = 2.0f32.powf(-(ceiling_round as f32 + 8.0f32));
                        (common_scale, true)
                    };

                    let left_rescale = (left_scaling.scale() / common_scale).round() as Element;
                    let right_rescale = (right_scaling.scale() / common_scale).round() as Element;
                    (left_rescale, right_rescale, common_scale, require_requant)
                }
            }
        };
        let (left_multiplier, right_multiplier, common_scale, require_requant): (
            Element,
            Element,
            f32,
            bool,
        ) = match left_log.compare(&right_log) {
            Ordering::Less | Ordering::Equal => {
                // left input has lower or equal precision, now we work out if the output_scaling is higher or lower precision
                scale_comparison(*right_scaling, *left_scaling)
            }
            Ordering::Greater => {
                // right input has lower precision
                scale_comparison(*left_scaling, *right_scaling)
            }
        };

        Self {
            left_multiplier,
            right_multiplier,
            common_scale,
            require_requant,
        }
    }
    /// The value to scalar multiply the left input by
    pub fn left_scale(&self) -> Element {
        self.left_multiplier
    }
    /// The value to scalar multiply the right input by
    pub fn right_scale(&self) -> Element {
        self.right_multiplier
    }

    /// If the output scale factor was a suitable amount more precise than either of the inputs we do not have to
    /// requantise afterwards (this almost never happens).
    pub fn requires_requant(&self) -> bool {
        self.require_requant
    }
    /// Returns the common scale factor used in the addition
    pub fn common_scale(&self) -> f32 {
        self.common_scale
    }
    /// The absoloute value of intermedaite size before addition is bounded above by `self.common_scale.log2().abs().ceil()`, so we add 2 extra to this.
    /// The first because we need an additional bit for the sign and the second because of the actual addition.
    pub fn intermediate_bit_size(&self) -> usize {
        self.common_scale.log2().abs().ceil() as usize + 2
    }
}

/// Normally, scaling add is done by scaling both inputs, so requant should happen _before_ the add.
/// y = (s1 * x1 + s2 * x2) / s3 where s1 is the left input scaling factor, s2 is the right input scaling factor,
/// and s3 is the output scaling factor.
///
/// If s3 is suitably small (i.e. it retains more bits of precision) then the values s1 / s3 and s2 / s3 are precise enough
/// that we can perform `y = (s1 / s3).round() * x1 + (s2 / s3).round() * x2` and not have to requantise afterwards.
///
/// If this isn't the case we pick some intermediate `common_scale` and obtain
/// `y_int = (s1 / common_scale).round() * x1 + (s2 / common_scale).round() * x2`.
/// Then we perform a requantisation step afterwards to calculate `y = (common_scale / s3).round() * y_int`.
///
/// Currently we require `s3` to be at least 8 bits more precise than `s1` or `s2` in order to not requantise.
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
        // We don't need the quantised domain here as the Requant layer works everything out from scale factors and intermediate bit size.
        let add_scale = ScalingFactor::from_scale(quant_info.common_scale(), None);
        let requant = requant_from_add(
            add_scale,
            output_scaling,
            quant_info.intermediate_bit_size(),
        );
        Ok(QuantizeOutput::new(quantized_model, vec![output_scaling]).with_requant(requant))
    }
}

/// Function used to instantiate a new [`Requant`] from the scaling factors of all tensors involved in an addition layer.
pub fn requant_from_add(
    add_scale: ScalingFactor,
    output_scale: ScalingFactor,
    intermediate_bit_size: usize,
) -> Requant {
    let m = add_scale.scale() / output_scale.scale();
    Requant::from_multiplier(m, intermediate_bit_size)
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
            is_static_operand: false,
            node_id: id,
        };
        if let Some((ref op, _)) = self.operand {
            let mut model_polys = HashMap::new();
            model_polys.insert(OPERAND_POLY_ID.to_string(), op.get_data().to_vec());
            aux.model_polys = Some(model_polys);
            ctx.is_static_operand = true;
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

    fn prove<T>(
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
        let (output_claims, proof) =
            self.prove_step(node_id, last_claims, &step_data.inputs, prover)?;

        prover.push_proof(node_id, LayerProof::Add(proof));
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
        if self.is_static_operand {
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

        let scale = qadd.quant_info.as_ref().unwrap().common_scale() / s3.scale();
        let result_scaled = Tensor::<Element>::new(
            qadd_result.outputs()[0].get_shape(),
            qadd_result.outputs()[0]
                .get_data()
                .iter()
                .map(|x| (*x as f32 * scale).round() as Element)
                .collect::<Vec<_>>(),
        );
        let computed_result = result_scaled.dequantize(&s3);

        let close_to_float = is_close_with_tolerance(
            computed_result.get_data(),
            t3.get_data(),
            1e-2_f32,
            1e-1_f32,
        );
        println!("computed_result: {:?}", computed_result.get_data());
        println!("t3: {:?}", t3.get_data());

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
        for _ in 0..25 {
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
    }

    #[test]
    fn test_add_proving_with_operand() {
        let input_shape = Shape::from(vec![3, 7]);
        for _ in 0..25 {
            let mut model =
                Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);
            let operand = Tensor::<f32>::random(&input_shape);
            let add = Add::new_with(operand, input_shape.clone());
            let _ = model.add_consecutive_layer(Layer::Add(add), None).unwrap();
            model.route_output(None).unwrap();
            model.describe();
            prove_model(model).unwrap();
        }
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
            is_close_with_tolerance(t1.get_data(), ct1.get_data(), 1e-2_f32, 0.1e-2_f32)
        );
    }
}
