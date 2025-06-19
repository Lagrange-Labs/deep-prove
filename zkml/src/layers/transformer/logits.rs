use crate::argmax_slice;
use anyhow::ensure;
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
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(
            inputs.iter().all(|i| i.get_shape().len() >= 2),
            "Argmax is for tensors of rank >= 2"
        );

        match self {
            Logits::Argmax => {
                let indices = inputs
                    .iter()
                    .map(|input| {
                        let rows = input
                            .slice_last_dim()
                            .map(|row| {
                                Tensor::new(
                                    // we want to stack along a new dimension so we unsqueeze manually here
                                    vec![1, 1],
                                    vec![N::from_usize(argmax_slice(row).unwrap())],
                                )
                            })
                            .collect::<Vec<_>>();
                        Tensor::stack_all(rows)
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
        let input = Tensor::new(vec![3, 2], vec![0.0, 1.0, 3.0, 2.0, 4.0, 5.0]);
        let logits = Logits::Argmax;
        let out = logits.evaluate::<GoldilocksExt2>(&[&input], vec![])?;
        // first slice is [0,1] so argmax here is 1
        // second slice is [3,2] so argmax here is 0
        // the last dimension is [4,5] so argmax here is 1
        assert_eq!(out.outputs()[0].get_data(), vec![1.0, 0.0, 1.0]);
        Ok(())
    }
}
