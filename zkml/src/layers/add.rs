use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Element, NextPowerOfTwo, ScalingFactor, ScalingStrategy, Tensor,
    layers::provable::{Evaluate, NodeId, OpInfo, QuantizeOp, QuantizeOutput},
    padding::PaddingMode,
    tensor::{Number, Shape},
};

use super::provable::LayerOut;

/// Add layer that adds two tensors together.
/// If there is two inputs, no static weight, then the output shape is the same as the first input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add<N> {
    operand: Option<Tensor<N>>,
}

impl<N: Number> Add<N> {
    pub fn new() -> Self {
        Self { operand: None }
    }
    pub fn new_with(operand: Option<Tensor<N>>) -> Self {
        Self { operand }
    }
}

impl<N: Number> Evaluate<N> for Add<N> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        let result = if inputs.len() == 2 {
            ensure!(
                Shape::from(inputs[0].get_shape()).numel()
                    == Shape::from(inputs[1].get_shape()).numel(),
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
                inputs[0].get_shape().iter().product::<usize>()
                    == self
                        .operand
                        .as_ref()
                        .unwrap()
                        .get_shape()
                        .iter()
                        .product::<usize>(),
                "Add layer expects input and operand to have the same shape: {:?} vs {:?}",
                inputs[0].get_shape(),
                self.operand.as_ref().unwrap().get_shape()
            );
            inputs[0].add(self.operand.as_ref().unwrap())
        } else {
            bail!("Add layer expects 1 or 2 inputs, got {}", inputs.len());
        };
        Ok(LayerOut::from_vec(vec![result]))
    }
}

impl<N> OpInfo for Add<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
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

impl QuantizeOp for Add<f32> {
    type QuantizedOp = Add<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        _input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_add() {
        let add = Add::new();
        let t1 = Tensor::<Element>::random(&vec![2, 2]);
        let t2 = Tensor::<Element>::random(&vec![2, 2]);
        let result = add
            .evaluate::<GoldilocksExt2>(&[&t1, &t2], vec![vec![2, 2], vec![2, 2]])
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    result.outputs[0].get(vec![i, j]),
                    t1.get(vec![i, j]) + t2.get(vec![i, j])
                );
            }
        }
        let add = Add::new_with(Some(t1.clone()));
        let result = add
            .evaluate::<GoldilocksExt2>(&[&t2], vec![vec![2, 2]])
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
}
