use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{
    NextPowerOfTwo, Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo},
    padding::PaddingMode,
    tensor::Number,
};

#[derive(Debug, Clone)]
pub enum Positional<N> {
    Learned(Tensor<N>),
    // TODO
    Rope,
}

impl<N: Number> Positional<N> {
    pub fn get_shape(&self) -> Vec<usize> {
        match self {
            Self::Learned(pos) => pos.get_shape().to_vec(),
            Self::Rope => unimplemented!("Rope not implemented"),
        }
    }
}

impl<N: Number> Evaluate<N> for Positional<N> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(
            inputs.iter().all(|x| x.get_shape().len() == 2),
            "positional embeddings only support 2d tensors"
        );

        let outputs = inputs
            .iter()
            .map(|x| {
                match self {
                    Self::Learned(pos) => {
                        let sub_pos = pos.slice_2d(0, x.get_shape()[0]);
                        assert_eq!(sub_pos.get_shape(), x.get_shape());
                        // we basically add the positional embeddings for each position in the input tensor
                        Ok(x.add(&sub_pos))
                    }
                    Self::Rope => {
                        anyhow::bail!("Rope not implemented");
                    }
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(LayerOut::from_vec(outputs))
    }
}

impl<N: Number> OpInfo for Positional<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        let s = input_shapes.to_vec();
        if let PaddingMode::Padding = padding_mode {
            s.into_iter().map(|s| s.next_power_of_two()).collect()
        } else {
            s
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "Positional({:?}x{:?})",
            self.get_shape()[0],
            self.get_shape()[1]
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}
