//! Temporary matmul for testing purposes while awaiting for the actual matmul to be merged on this branch.

use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{Tensor, tensor::Number};

use super::provable::LayerOut;

// MatMul that multiplies two witnesses values
pub struct MatMul;

impl MatMul {
    pub fn evaluate<N: Number, E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "MatMul expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        ensure!(a.get_shape().len() == 2, "MatMul expects a 2D tensor");
        ensure!(b.get_shape().len() == 2, "MatMul expects a 2D tensor");
        let result = a.matmul(b);
        Ok(LayerOut::from_vec(vec![result]))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use super::*;

    #[test]
    fn test_matmul() {
        let matmul = MatMul;
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matmul.evaluate::<_, GoldilocksExt2>(&[&a, &b]).unwrap();
        assert_eq!(result.outputs[0].data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
