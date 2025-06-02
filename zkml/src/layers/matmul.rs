//! Temporary matmul for testing purposes while awaiting for the actual matmul to be merged on this branch.

use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{layers::provable::Evaluate, tensor::Number, Tensor};

use super::provable::LayerOut;

pub enum Config {
    TransposeB,
}
// MatMul that multiplies two witnesses values
pub struct MatMul {
    option: Option<Config>,
}

impl MatMul {
    pub fn new_with_config(option: Config) -> Self {
        Self { option: Some(option) }
    }
    pub fn new() -> Self {
        Self { option: None }
    }
}
impl<N: Number> Evaluate<N> for MatMul {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "MatMul expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        ensure!(a.get_shape().len() == 2, "MatMul expects a 2D tensor");
        ensure!(b.get_shape().len() == 2, "MatMul expects a 2D tensor");
        let result = if let Some(ref config) = self.option {
            match config {
                Config::TransposeB => {
                    assert_eq!(a.get_shape()[1], b.get_shape()[1]);
                    a.matmul(&b.transpose())
                }
            }
        } else {
            assert_eq!(a.get_shape()[1], b.get_shape()[0]);
            a.matmul(b)
        };
        Ok(LayerOut::from_vec(vec![result]))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use super::*;

    #[test]
    fn test_matmul() {
        let matmul = MatMul::new();
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matmul.evaluate::<GoldilocksExt2>(&[&a, &b], vec![]).unwrap();
        assert_eq!(result.outputs[0].data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_transpose_b() {
        let matmul = MatMul::new_with_config(Config::TransposeB);
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose();
        let result = matmul.evaluate::<GoldilocksExt2>(&[&a, &b],vec![]).unwrap();
        assert_eq!(result.outputs[0].data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
