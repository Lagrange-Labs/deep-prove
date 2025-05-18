use anyhow::ensure;

use crate::{Tensor, padding::PaddingMode, parser::gguf::FileTensorLoader, tensor::Number};

use super::provable::{Evaluate, OpInfo};

#[derive(Debug, Clone)]
pub struct LayerNorm<N: Number> {
    gamma: Tensor<N>,
    beta: Tensor<N>,
}

impl LayerNorm<f32> {
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_loader(loader: &FileTensorLoader, exp_size: usize) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        ensure!(
            gamma.get_shape().as_slice() == &[exp_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            exp_size,
            gamma.get_shape()
        );
        ensure!(
            beta.get_shape().as_slice() == &[exp_size],
            "norm_beta must have shape [{}] vs given {:?}",
            exp_size,
            beta.get_shape()
        );
        Ok(Self { gamma, beta })
    }
}

impl<N: Number> OpInfo for LayerNorm<N> {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "LayerNorm({:?},{:?})",
            self.gamma.get_shape(),
            self.beta.get_shape()
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for LayerNorm<N> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<super::provable::LayerOut<N, E>> {
        unimplemented!()
    }
}
