//! This layer applies the softmax function to the last dimension of the input tensor
use anyhow::ensure;

use crate::{
    Element, Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo, QuantizeOp},
    tensor::Number,
};

#[derive(Debug, Clone)]
pub struct Softmax<N> {
    // By default, it's equal to 1
    pub scale: N,
    // By default, softmax is going to be applied on the full tensor.
    // You can specificy a dimen to apply softmax on. For example, for a tensor  of shape [2,3,4],
    // if apply_on_dim = 2, then softmax will be applied on every chunks of 3 * 4 elements each.
    pub apply_on_dim: Option<usize>,
}

impl<N: Number> Softmax<N> {
    pub fn new() -> Self {
        Self {
            scale: N::unit(),
            apply_on_dim: None,
        }
    }
    pub fn with_scale(self, scale: N) -> Self {
        Self { scale, ..self }
    }
    pub fn with_dim(self, dim: usize) -> Self {
        Self {
            apply_on_dim: Some(dim),
            ..self
        }
    }
}

impl Evaluate<f32> for Softmax<f32> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&crate::Tensor<f32>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "softmax expects exactly one input tensor currently"
        );
        let input = inputs[0];
        let dim = self.apply_on_dim.unwrap_or(input.get_shape().len() - 1);
        let output = input
            .slice_on_dim(dim)
            .map(|vec| {
                let sum = vec.iter().map(|x| self.scale * x.exp()).sum::<f32>();
                vec.iter()
                    .map(|x| (self.scale * x).exp() / sum)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        let output_tensor = Tensor::new(input.get_shape(), output);
        Ok(LayerOut::from_vec(vec![output_tensor]))
    }
}

impl Evaluate<Element> for Softmax<Element> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        _inputs: &[&crate::Tensor<Element>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        unimplemented!()
    }
}

impl<N: Number> OpInfo for Softmax<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Vec<usize>> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Softmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for Softmax<f32> {
    type QuantizedOp = Softmax<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: crate::layers::provable::NodeId,
        _input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<crate::layers::provable::QuantizeOutput<Self::QuantizedOp>> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_softmax() {
        let softmax = Softmax::new();
        let input = Tensor::new(vec![2, 3], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3]])
            .unwrap();
        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0
            ]
        );
    }

    #[test]
    fn test_softmax_with_scale() {
        let softmax = Softmax::new().with_scale(2.0);
        let input = Tensor::new(vec![2, 3], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3]])
            .unwrap();
        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0
            ]
        );
    }
}
