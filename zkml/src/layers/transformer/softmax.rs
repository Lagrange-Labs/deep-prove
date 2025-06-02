//! This layer applies the softmax function to the last dimension of the input tensor
use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut},
};

#[derive(Debug, Clone)]
pub struct Softmax;

impl Evaluate<f32> for Softmax {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&crate::Tensor<f32>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let input = inputs[0];
        let output = input
            .slices_last_dim()
            .map(|vec| {
                let sum = vec.iter().map(|x| x.exp()).sum::<f32>();
                vec.iter().map(|x| x.exp() / sum).collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        let output_tensor = Tensor::new(input.get_shape(), output);
        Ok(LayerOut::from_vec(vec![output_tensor]))
    }
}

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_softmax() {
        let softmax = Softmax;
        let input = Tensor::new(vec![2, 3], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3]])
            .unwrap();
        assert_eq!(output.outputs[0].get_data(), vec![
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0
        ]);
    }
}
