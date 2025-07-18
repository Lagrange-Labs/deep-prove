use anyhow::ensure;
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::{
    Element, ScalingFactor, Tensor,
    layers::provable::{QuantizeOp, QuantizeOutput},
    padding::PaddingMode,
    parser::{gguf::FileTensorLoader, json, llm::LLMConfig},
    tensor::{Number, Shape},
};

use crate::layers::provable::{Evaluate, LayerOut, OpInfo};
use burn::{
    module::Param,
    nn::LayerNormConfig as BLayerNormConfig,
    tensor::{Tensor as BTensor, TensorData},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn from_json(l: &json::FileTensorLoader, _c: &LLMConfig) -> anyhow::Result<Self> {
        trace!("from_json: current path: {:?}", l.prefix);
        let gamma = l.get_tensor("norm.weight")?;
        let beta = l.get_tensor("norm.bias")?;
        let eps = l.metadata_to_f32("norm_epsilon")?;
        Ok(Self::new(gamma, beta, eps))
    }
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        ensure!(
            gamma.get_shape().as_ref() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.get_shape()
        );
        ensure!(
            beta.get_shape().as_ref() == &[c.embedding_size],
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
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
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
type Backend = burn::backend::NdArray;

impl Evaluate<f32> for LayerNorm<f32> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        assert!(inputs.len() == 1);
        let input = inputs[0];
        ensure!(
            input.get_shape().len() == 2,
            "layernorm input must have shape [seq_len, embedding_size]: found {:?}",
            input.get_shape()
        );
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
        let config = BLayerNormConfig::new(embedding_size).with_epsilon(self.eps as f64);
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Param::from_tensor(beta);
        let output = norm.forward(input);
        let Ok(data): Result<Vec<f32>, _> = output.to_data().into_vec() else {
            anyhow::bail!("failed to convert to f32");
        };
        let output_shape = Shape::new(output.shape().dims);
        Ok(LayerOut::from_tensor(Tensor::<f32>::new(
            output_shape,
            data,
        )))
    }
}

impl Evaluate<Element> for LayerNorm<Element> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        _inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        unimplemented!()
    }
}

impl QuantizeOp for LayerNorm<f32> {
    type QuantizedOp = LayerNorm<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: crate::layers::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<crate::layers::provable::QuantizeOutput<Self::QuantizedOp>> {
        // TODO: write the layernorm quantization rule depending on proving
        // Currently still working since we want to test quantization of layers.
        warn!("LayerNorm quantization not implemented");
        let gamma = self.gamma.quantize(&ScalingFactor::default());
        let beta = self.beta.quantize(&ScalingFactor::default());
        let quantized = LayerNorm::new(gamma, beta, self.eps);
        Ok(QuantizeOutput::new(quantized, input_scaling.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;

    use super::*;

    impl<N: Number> LayerNorm<N> {
        pub fn random(size: usize) -> Self {
            let gamma = Tensor::<N>::random(&vec![size].into());
            let beta = Tensor::<N>::random(&vec![size].into());
            let eps = 1e-5;
            Self::new(gamma, beta, eps)
        }
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_layernorm() {
        let gamma = Tensor::<f32>::new(vec![1024].into(), vec![1.0; 1024]);
        let beta = Tensor::<f32>::new(vec![1024].into(), vec![0.0; 1024]);
        let eps = 1e-5;
        let layernorm = LayerNorm { gamma, beta, eps };
        let input = Tensor::<f32>::new(vec![1, 1024].into(), vec![0.0; 1024]);
        let output = layernorm.evaluate::<E>(&[&input], vec![]).unwrap();
        assert_eq!(output.outputs[0].get_shape(), vec![1, 1024].into());
        assert_eq!(output.outputs[0].get_data(), vec![0.0; 1024]);
    }
}
