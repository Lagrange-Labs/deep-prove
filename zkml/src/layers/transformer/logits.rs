use anyhow::ensure;
use crate::argmax_slice;
use serde::{Deserialize, Serialize};

use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo},
    tensor::Number,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Logits {
    Argmax,
}

impl<N: Number> Evaluate<N> for Logits {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&crate::Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<crate::layers::provable::LayerOut<N, E>> {
        match self {
            Logits::Argmax => {
                let indices = inputs
                    .iter()
                    .map(|input| {
                        ensure!(input.get_shape().len() >= 2, "Argmax is for tensors of rank >= 2");
                        let last_row = input.slice_on_dim(1).last().unwrap();
                        Ok(Tensor::new(vec![1], vec![N::from_usize(argmax_slice(last_row).unwrap())]))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                Ok(LayerOut::from_vec(indices))
            }
        }
    }
}

impl OpInfo for Logits {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Vec<usize>> {
        vec![vec![1]; input_shapes.len()]
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Logits".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use super::*;
    use crate::{layers::provable::Evaluate, tensor::Tensor};

    #[test]
    fn test_logits_argmax() -> anyhow::Result<()> {
        let input = Tensor::new(vec![3, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let logits = Logits::Argmax;
        let out = logits.evaluate::<GoldilocksExt2>(&[&input], vec![])?;
        // the last dimension is [4,5] so argmax here is 1
        assert_eq!(out.outputs()[0].get_data(), vec![1.0]);
        Ok(())
    }
}
