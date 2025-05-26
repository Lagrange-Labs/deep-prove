use anyhow::ensure;

use crate::{
    Element, Tensor,
    padding::PaddingMode,
    parser::gguf::{FileTensorLoader, LLMConfig},
    tensor::Number,
};

use crate::layers::provable::{Evaluate, LayerOut, OpInfo, ProvableOpError};
use burn::{
    backend::Wgpu,
    module::Param,
    nn::{LayerNorm as BLayerNorm, LayerNormConfig as BLayerNormConfig},
    tensor::{Tensor as BTensor, TensorData},
};

#[derive(Debug, Clone)]
pub struct LayerNorm<N> {
    pub gamma: Tensor<N>,
    pub beta: Tensor<N>,
    pub eps: f32,
}

impl<N: Number> LayerNorm<N> {
    pub fn new(gamma: Tensor<N>, beta: Tensor<N>, eps: f32) -> Self {
        assert_eq!(gamma.get_shape(), beta.get_shape());
        Self { gamma, beta, eps }
    }
}

impl LayerNorm<f32> {
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        ensure!(
            gamma.get_shape().as_slice() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.get_shape()
        );
        ensure!(
            beta.get_shape().as_slice() == &[c.embedding_size],
            "norm_beta must have shape [{}] vs given {:?}",
            c.embedding_size,
            beta.get_shape()
        );
        let eps = loader.metadata::<f32>(c.specific_config.norm_epsilon_key());
        Ok(Self::new(gamma, beta, eps))
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

// Type alias for the backend to use.
type Backend = Wgpu;

impl Evaluate<f32> for LayerNorm<f32> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<f32, E>, ProvableOpError> {
        assert!(inputs.len() == 1);
        let input = inputs[0];
        assert!(input.get_shape().len() == 2);
        let embedding_size = input.get_shape()[1];
        let device = Default::default();
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/layer.rs.html#67
        let input = BTensor::<Backend, 2>::from_data(
            TensorData::new(input.get_data().to_vec(), input.get_shape()),
            &device,
        );
        let gamma = BTensor::<Backend, 1>::from_data(
            TensorData::new(self.gamma.get_data().to_vec(), self.gamma.get_shape()),
            &device,
        );
        let beta = BTensor::<Backend, 1>::from_data(
            TensorData::new(self.beta.get_data().to_vec(), self.beta.get_shape()),
            &device,
        );
        let config = BLayerNormConfig::new(embedding_size as usize).with_epsilon(self.eps as f64);
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Param::from_tensor(beta);
        let output = norm.forward(input);
        let data: Vec<f32> = output.to_data().into_vec().map_err(|e| {
            ProvableOpError::ConversionError("failed to convert to f32".to_string())
        })?;
        let output_shape = output.shape().dims.to_vec();
        Ok(LayerOut::from_tensor(Tensor::<f32>::new(
            output_shape,
            data,
        )))
    }
}

impl Evaluate<Element> for LayerNorm<Element> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<Element, E>, ProvableOpError> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;

    use super::*;

    impl<N: Number> LayerNorm<N> {
        pub fn random(size: usize) -> Self {
            let gamma = Tensor::<N>::random(&[size]);
            let beta = Tensor::<N>::random(&[size]);
            let eps = 1e-5;
            Self::new(gamma, beta, eps)
        }
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_layernorm() {
        let gamma = Tensor::<f32>::new(vec![1024], vec![1.0; 1024]);
        let beta = Tensor::<f32>::new(vec![1024], vec![0.0; 1024]);
        let eps = 1e-5;
        let layernorm = LayerNorm { gamma, beta, eps };
        let input = Tensor::<f32>::new(vec![1, 1024], vec![0.0; 1024]);
        let output = layernorm.evaluate::<E>(&[&input], vec![]).unwrap();
        assert_eq!(output.outputs[0].get_shape(), vec![1, 1024]);
        assert_eq!(output.outputs[0].get_data(), vec![0.0; 1024]);
    }
}
