//! This layer applies the softmax function to the last dimension of the input tensor
use anyhow::ensure;
use serde::{Deserialize, Serialize};

use crate::{
    Element, Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo, QuantizeOp},
    tensor::{Number, Shape},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Softmax<N> {
    // By default, it's equal to 1
    pub scale: N,
    // By default, softmax is going to be applied on the full tensor.
    // You can specificy a dimen to apply softmax on. For example, for a tensor  of shape [2,3,4],
    // if apply_on_dim = 1, then softmax will be applied on every chunks of 4 elements each.
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
    /// Apply softmax on the subset of from this dim
    pub fn on_dim(self, dim: usize) -> Self {
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
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "softmax expects exactly one input tensor currently"
        );
        let input = inputs[0];
        let dim = self.apply_on_dim.unwrap_or(input.get_shape().len() - 1);
        let output = input
            .slice_on_dim(dim)
            .0
            .map(|vec| {
                let scaled = vec
                    .iter()
                    .map(|x| self.scale * x)
                    .map(|x| x.exp())
                    .collect::<Vec<_>>();
                let sum = scaled.iter().sum::<f32>();
                scaled.iter().map(|x| x / sum).collect::<Vec<_>>()
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
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        unimplemented!()
    }
}

impl<N: Number> OpInfo for Softmax<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
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

    use ff_ext::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_softmax() {
        let softmax = Softmax::new();
        let input = Tensor::new(vec![2, 3].into(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3].into()])
            .unwrap();
        assert_eq!(output.outputs[0].get_shape(), vec![2, 3].into());
        // since we dont slice, sum of  prob should be equal to 1
        assert_eq!(output.outputs[0].get_data().iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn test_softmax_with_dim() {
        let softmax = Softmax::new().on_dim(1);
        let input = Tensor::random(&vec![2, 3, 4].into());
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3, 4].into()])
            .unwrap();
        let out = output.outputs()[0];
        assert_eq!(out.get_shape(), vec![2, 3, 4].into());
        let (slices, _) = out.slice_on_dim(1);
        let acceptable_range = 0.99..1.01;
        for slice in slices {
            assert!(
                acceptable_range.contains(&slice.iter().sum::<f32>()),
                "{:?}",
                out.get_data()
            );
        }
    }

    #[test]
    fn test_softmax_with_scale() {
        let scale = 1.0 / 2.0;
        let softmax = Softmax::new().with_scale(scale);
        let input = Tensor::new(vec![2, 3].into(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3].into()])
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
