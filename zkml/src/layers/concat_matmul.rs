//! Module to infere and prove statements of the form:
//! A = A_1 || A_2 || ... || A_n
//! B = B_1 || B_2 || ... || B_n
//! C = C_1 || C_2 || ... || C_n
//! where C_i = A_i @ B_i
//! Here concatenation means concatenation over the highest dimension, e.g.
//! if A_i is of shape [1, r, s] then A = [A_1, A_2, ... , A_n] is of shape [n, r, s]
//! 
//! This module currently only supports the case where A_i and B_i are witnesses values.

use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{tensor::Number, Tensor};

use super::provable::LayerOut;
#[derive(Debug, Clone)]
struct ConcatMatMul;

impl ConcatMatMul {
    pub fn evaluate<N: Number, E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "ConcatMatMul expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        let a_shape = a.get_shape();
        let b_shape = b.get_shape();
        ensure!(a_shape.len() == b_shape.len(), "ConcatMatMul expects inputs of the same shape");
        ensure!(a_shape.len() == 3, "ConcatMatMul expects inputs of shape [n, r, s]");
        ensure!(a_shape[0] == b_shape[0], "ConcatMatMul expects inputs with same highest dimension");
        ensure!(a_shape[2] == b_shape[1], "ConcatMatMul expects submatrices dimensions to match");
        let results = (0..a_shape[0]).map(|batch| {
            let batch_a = a.slice_3d(batch, batch + 1).reshape(vec![a_shape[1], a_shape[2]]);
            let batch_b = b.slice_3d(batch, batch + 1).reshape(vec![b_shape[1], b_shape[2]]);
            batch_a.matmul(&batch_b)
        }).collect::<Vec<_>>();
        let mut it = results.into_iter();
        // reshape because concat expects a 3d tensor so he can accumulate in the highest dimension.
        let concat = it.next().unwrap().reshape(vec![1, a_shape[1], b_shape[2]]);
        let concat = it.fold(concat, |mut acc, x| {
            acc.concat(x);
            acc
        });
        Ok(LayerOut::from_vec(vec![concat]))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_concat_matmul() {
        let concat_matmul = ConcatMatMul;
        let a = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = concat_matmul.evaluate::<_, GoldilocksExt2>(&[&a, &b]).unwrap();
        assert_eq!(result.outputs[0].data, vec![7.0,10.0,15.0,22.0,67.0,78.0,91.0,106.0]);
    }
}
