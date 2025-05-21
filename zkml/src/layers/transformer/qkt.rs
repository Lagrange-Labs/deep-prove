//! Performs the Q @ K^T operation
//! where Q is a vector tensor and K is the matrix tensor output from the qkv layer

use ff_ext::ExtensionField;

use crate::layers::provable::{Evaluate, ProvableOpError};
use crate::layers::LayerOut;
use crate::tensor::Number;
use crate::Tensor;

#[derive(Clone, Debug)]
struct QKT;

impl<N: Number> Evaluate<N> for QKT {
    fn evaluate<E: ExtensionField>(&self, inputs: &[&Tensor<N>],_unpadded_input_shapes: Vec<Vec<usize>>) -> Result<LayerOut<N, E>, ProvableOpError> {
        let q = inputs[0];
        let k = inputs[1];
        let q = if q.get_shape().len() == 1 {
            let mat_q = q.clone();
            mat_q.reshape(vec![1, q.get_shape()[0]])
        } else {
            q.clone()
        };
        if q.get_shape()[1] != k.get_shape()[1] {
            return Err(ProvableOpError::InvalidInputShape(format!("qkt expects the second dimension of q and k to be the same, got {:?} and {:?}", q.get_shape(), k.get_shape())));
        }
        let qkt = q.matmul(&k.transpose());
        Ok(LayerOut::from_vec(vec![qkt]))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::{Element, Tensor};

    use super::*;

    #[test]
    fn test_qkt() {
        let mut q = Tensor::<Element>::new(vec![10], vec![1,2,3,4,5,6,7,8,9,10]);
        let k = Tensor::<Element>::new(vec![2, 10], vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
        let qkt_layer = QKT;
        let output = qkt_layer.evaluate::<GoldilocksExt2>(&[&q, &k], vec![vec![1, 10], vec![2, 10]]).expect("qkt shouldn't fail");
        let kt = k.transpose();
        // just to treat it as a matrix for matmul
        q.shape = vec![1, 10];
        let expected = q.matmul(&kt);
        assert_eq!(expected.get_shape(), vec![1,2]);
        assert_eq!(output.outputs[0].get_shape(), expected.get_shape());
        assert_eq!(output.outputs[0].get_data(), expected.get_data());
    }
}