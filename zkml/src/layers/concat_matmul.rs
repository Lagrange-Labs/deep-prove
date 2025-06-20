//! Module to infere and prove statements of the form:
//! A = A_1 || A_2 || ... || A_n
//! B = B_1 || B_2 || ... || B_n
//! C = C_1 || C_2 || ... || C_n
//! where C_i = A_i @ B_i
//! Here concatenation means concatenation over the highest dimension, e.g.
//! if A_i is of shape [1, r, s] then A = [A_1, A_2, ... , A_n] is of shape [n, r, s]
//!
//! This module currently only supports the case where A_i and B_i are witnesses values.
//! Transpose: There is the option to transpose the output of the matmul. This is useful for proving to avoid
//! having to prove explicitly the transpose operation with a separate layer, as sumcheck based proving can directly
//! prove the transpose at the same time as the matmul.
use std::borrow::Borrow;

use anyhow::ensure;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Tensor,
    layers::{
        provable::{Evaluate, OpInfo, QuantizeOp},
        requant::Requant,
    },
    padding::PaddingMode,
    tensor::{Number, Shape},
};

use super::provable::LayerOut;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatMatMul {
    permute: Option<Vec<usize>>,
    /// It tells what is the maximum bit size we ever expect the output of this layer to be.
    /// NOTE: This is a config item normally but we need this information during quantization.
    /// Best would be to rework quantization trait to include such config items.
    intermediate_bit_size: usize,
}

const DEFAULT_INTERMEDIATE_BIT_SIZE: usize = 25;

impl ConcatMatMul {
    pub fn new() -> Self {
        Self {
            permute: None,
            intermediate_bit_size: DEFAULT_INTERMEDIATE_BIT_SIZE,
        }
    }
    pub fn new_with_permute(permutation: Vec<usize>) -> Self {
        Self {
            permute: Some(permutation),
            intermediate_bit_size: DEFAULT_INTERMEDIATE_BIT_SIZE,
        }
    }
    pub fn with_max_shapes(self, max_shapes: Vec<Shape>) -> Self {
        Self::ensure_shape_consistency(&max_shapes).unwrap();
        let matrix_shape = max_shapes.into_iter().next().unwrap().slice(1..2);
        let intermediate_bit_size = matrix_shape.matmul_output_bitsize();
        Self {
            permute: None,
            intermediate_bit_size,
        }
    }

    pub fn ensure_shape_consistency<S: Borrow<Shape>>(shapes: &[S]) -> anyhow::Result<()> {
        assert!(shapes.len() == 2, "ConcatMatMul expects 2 inputs");
        ensure!(
            shapes[0].borrow().rank() == shapes[1].borrow().rank(),
            "ConcatMatMul expects input shapes with same rank: {:?} vs {:?}",
            shapes[0].borrow(),
            shapes[1].borrow()
        );
        ensure!(
            shapes[0].borrow().rank() == 3,
            "ConcatMatMul expects inputs of rank 3"
        );
        ensure!(
            shapes[0].borrow().dim(0) == shapes[1].borrow().dim(0),
            "ConcatMatMul expects inputs with same highest dimension"
        );
        ensure!(
            shapes[0].borrow().dim(2) == shapes[1].borrow().dim(1),
            "ConcatMatMul expects submatrices dimensions to match"
        );
        Ok(())
    }
}

impl<N: Number> Evaluate<N> for ConcatMatMul {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "ConcatMatMul expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        let a_shape = a.get_shape();
        let b_shape = b.get_shape();
        Self::ensure_shape_consistency(&[&a_shape, &b_shape])?;
        let results = (0..a_shape[0])
            .map(|batch| {
                let batch_a = a.slice_3d(batch, batch + 1).reshape(a_shape.slice(1..=2));
                let batch_b = b.slice_3d(batch, batch + 1).reshape(b_shape.slice(1..=2));
                batch_a.matmul(&batch_b)
            })
            .collect::<Vec<_>>();
        let mut it = results.into_iter();
        // reshape because concat expects a 3d tensor so he can accumulate in the highest dimension.
        let concat = it
            .next()
            .unwrap()
            .reshape(Shape::new(vec![1, a_shape[1], b_shape[2]]));
        let mut concat = it.fold(concat, |mut acc, x| {
            acc.concat(x);
            acc
        });
        if let Some(ref transpose) = self.permute {
            concat = concat.permute3d(transpose);
        }
        Ok(LayerOut::from_vec(vec![concat]))
    }
}

impl OpInfo for ConcatMatMul {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        let a_shape = &input_shapes[0];
        let b_shape = &input_shapes[1];
        Self::ensure_shape_consistency(&[a_shape, b_shape]).unwrap();
        // inner matrix shapes
        let mut mat_result_shape: Shape = vec![a_shape[0], a_shape[1], b_shape[2]].into();
        if let PaddingMode::Padding = padding_mode {
            mat_result_shape = mat_result_shape.next_power_of_two()
        }
        if let Some(ref permute) = self.permute {
            mat_result_shape = mat_result_shape.permute(permute);
        }
        vec![mat_result_shape]
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!("ConcatMatMul (permute: {:?})", self.permute)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for ConcatMatMul {
    type QuantizedOp = ConcatMatMul;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: super::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<super::provable::QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scale = S::scaling_factors_for_node(data, node_id, num_outputs)[0];
        // normally it's input_scaling * model_scaling / output_scaling, except in this case, we don't have a model_scaling
        // but we have the second matrix scaling, so we use that.
        let input_scale = input_scaling[0];
        let weights_scale = input_scaling[1];
        let intermediate_bit_size = self.intermediate_bit_size;
        let requant = Requant::from_scaling_factors(
            input_scale,
            weights_scale,
            output_scale,
            intermediate_bit_size,
        );
        Ok(super::provable::QuantizeOutput::new(self, vec![output_scale]).with_requant(requant))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_concat_matmul() {
        let concat_matmul = ConcatMatMul::new();
        let a = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let b = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        assert_eq!(
            result.outputs[0].data,
            vec![7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0]
        );
    }

    #[test]
    fn test_concat_matmul_with_transpose() {
        let concat_matmul = ConcatMatMul::new_with_permute(vec![1, 0, 2]);
        let a = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let b = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        let expected = Tensor::new(
            vec![2, 2, 2].into(),
            vec![7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0],
        );
        let expected = expected.permute3d(&vec![1, 0, 2]);
        assert_eq!(result.outputs[0].data, expected.data);
        let expected_shape =
            concat_matmul.output_shapes(&[a.get_shape(), b.get_shape()], PaddingMode::NoPadding);
        assert_eq!(result.outputs[0].get_shape(), expected_shape[0]);
    }
}
