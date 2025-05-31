use anyhow::ensure;

use crate::{
    Tensor,
    parser::gguf::{FileTensorLoader, LLMConfig, LLMVariant},
    parser::json,
    tensor::Number,
};

#[derive(Debug, Clone)]
pub enum Positional<N: Number> {
    Learned(Tensor<N>),
    // TODO
    Rope,
}

impl Positional<f32> {
    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let position_embd = l.get_tensor("position_embd.weight")?;
        ensure!(position_embd.get_shape().len() == 2, "position_embd must be 2d");
        ensure!(position_embd.get_shape()[0] == c.context_length, "position_embd must have shape [0] [{}] vs given {:?}", c.context_length, position_embd.get_shape());
        ensure!(position_embd.get_shape()[1] == c.embedding_size, "position_embd must have shape [1] [{}] vs given {:?}", c.embedding_size, position_embd.get_shape());
        Ok(Self::Learned(position_embd))
    }
        
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        match c.specific_config {
            LLMVariant::GPT2 => {
                let position_embd = loader.get_tensor("position_embd.weight")?;
                let shape = position_embd.get_shape();
                ensure!(
                    shape[0] == c.context_length,
                    "position_embd must have shape [{}] vs given {:?}",
                    c.context_length,
                    position_embd.get_shape()
                );
                ensure!(
                    shape[1] == c.embedding_size,
                    "position_embd must have shape [{}] vs given {:?}",
                    c.embedding_size,
                    position_embd.get_shape()
                );
                Ok(Self::Learned(position_embd))
            }
            _ => anyhow::bail!(
                "Unsupported variant for positional embeddings: {:?}",
                c.specific_config
            ),
        }
    }
}

impl<N: Number> Positional<N> {
    pub fn forward(&self, x: &Tensor<N>) -> anyhow::Result<Tensor<N>> {
        ensure!(
            x.get_shape().len() == 2,
            "positional embeddings only support 2d tensors"
        );
        match self {
            Self::Learned(pos) => {
                let sub_pos = pos.slice_2d(0, x.get_shape()[0]);
                assert_eq!(sub_pos.get_shape(), x.get_shape());
                Ok(x.add(&sub_pos))
            }
            Self::Rope => {
                anyhow::bail!("Rope not implemented");
            }
        }
    }
}
