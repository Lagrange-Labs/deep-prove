use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Element, ScalingFactor, ScalingStrategy, Tensor,
    layers::{
        provable::{Evaluate, NodeId, OpInfo, QuantizeOp, QuantizeOutput},
        requant::Requant,
    },
    padding::PaddingMode,
    quantization::split_scale_into_multiplier,
    tensor::{Number, Shape},
};

use super::provable::LayerOut;

/// Add layer that adds two tensors together.
/// If there is two inputs, no static weight, then the output shape is the same as the first input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add<N> {
    /// The operand is the right side of the Add operation.
    /// shape is the unpadded shape of the operand
    operand: Option<(Tensor<N>, Shape)>,
    quant_info: Option<QuantInfo>,
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
        if inputs.len() == 2 {
            let left = inputs[0].scalar_mul(&(quant_info.left_scale()));
            let right = inputs[1].scalar_mul(&(quant_info.right_scale()));
            let result = left.add(&right);
            Ok(LayerOut::from_vec(vec![result]))
        } else if inputs.len() == 1 {
            let Some((ref op, _)) = self.operand else {
                bail!("Add layer is not quantized");
            };
            // we dont need to scale the operand since it's already done during the quantization of the layer
            let result = op.add(&inputs[0].scalar_mul(&(quant_info.left_scale())));
            Ok(LayerOut::from_vec(vec![result]))
        } else {
            bail!("Add layer expects 1 or 2 inputs, got {}", inputs.len());
        }
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
/// In that case, we quantize the right operand only with M2 * 2^shift1, not s2.
impl Add<f32> {
    fn quantize(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Add<Element>>> {
        let left_scaling = input_scaling[0];
        let right_scaling = input_scaling[1];
        // s1p = M1 / 2^shift1
        let s1p = left_scaling.scale() / output_scaling.scale();
        let (shift1, m1) = split_scale_into_multiplier(s1p);
        // s2p = M2 / 2^shift2
        let s2p = right_scaling.scale() / output_scaling.scale();
        let (shift2, m2) = split_scale_into_multiplier(s2p);
        let quant_info = QuantInfo {
            m1,
            shift1,
            m2,
            shift2,
        };
        let quantized_model = Add::<Element> {
            operand: self.operand.map(|(t, s)| {
                // we quantize the static operand with m2 * 2^shift1 so the values are "already" ready to be added
                // with the input during inference
                let right_scaling =
                    ScalingFactor::from_scale(quant_info.right_scale() as f32, None);
                (t.quantize(&right_scaling), s)
            }),
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
        let close_to_float =
            is_close_with_tolerance(computed_result.get_data(), t3.get_data(), 1e-2_f32, 0.3);
        assert!(within_range, "output is not within range");
        assert!(close_to_float, "output is not close to float");
    }
}
