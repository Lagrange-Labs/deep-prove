use anyhow::ensure;
use crate::padding::PaddingMode;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{Tensor, tensor::Number};

use super::provable::{Evaluate, LayerOut, OpInfo, ProvableOpError};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reshape {
    new_dim: Vec<Vec<usize>>,
}

impl OpInfo for Reshape {
    fn output_shapes(&self, input_shapes: &[Vec<usize>], _padding_mode: PaddingMode) -> Vec<Vec<usize>> {
        assert!(self.new_dim.len() == input_shapes.len());
        assert!(
            self.new_dim
                .iter()
                .zip(input_shapes.iter())
                .all(|(new_dim, input_shape)| new_dim.iter().product::<usize>()
                    == input_shape.iter().product::<usize>())
        );
        self.new_dim.clone()
    }
    
    fn num_outputs(&self, num_inputs: usize) -> usize {
       num_inputs
    }
    
    fn describe(&self) -> String {
        format!("Reshape: {:?}", self.new_dim)
    }
    
    fn is_provable(&self) -> bool {
        false
    }
}

impl<N: Number> Evaluate<N> for Reshape {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<N, E>, ProvableOpError> {
        if self.new_dim.len() != inputs.len() {
            return Err(ProvableOpError::InvalidInputShape(format!(
                "new dims {:?} vs inputs.len() {}",
                self.new_dim,
                inputs.len()
            )));
        }
        assert!(
            self.new_dim
                .iter()
                .zip(inputs.iter())
                .all(|(new_dim, input_tensor)| new_dim.iter().product::<usize>()
                    == input_tensor.get_shape().iter().product::<usize>())
        );
        let out_tensors = inputs.iter().map(|x| x.clone().clone()).collect::<Vec<_>>();
        let out_tensors = self
            .new_dim
            .iter()
            .zip(out_tensors.into_iter())
            .map(|(new_dim, input_tensor)| {
                input_tensor.reshape(new_dim.clone())
            })
            .collect();
        Ok(LayerOut::from_vec(out_tensors))
    }
}


#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_reshape() {
        let input = Tensor::<Element>::new(vec![2, 3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
        let reshape = Reshape {
            new_dim: vec![vec![3,2,3]],
        };
        let output = reshape.evaluate::<GoldilocksExt2>(&[&input], vec![vec![]]).expect("reshape shouldn't fail");
        assert_eq!(output.outputs[0].get_shape(), vec![3, 2, 3]);
        assert_eq!(output.outputs[0].get_data(), input.get_data());
    }
}