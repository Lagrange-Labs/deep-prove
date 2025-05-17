use crate::{padding::PaddingMode, parser::gguf::FileTensorLoader, tensor::Number, Tensor};

use super::provable::OpInfo;


#[derive(Debug, Clone)]
pub struct Embeddings<N: Number> {
    pub emb: Tensor<N>,
}

impl<N: Number> Embeddings<N> {
    pub fn new(emb: Tensor<N>) -> Self {
        Self { emb }
    }
}

impl<N: Number> OpInfo for Embeddings<N> {
    fn output_shapes(&self, input_shapes: &[Vec<usize>], _padding_mode: PaddingMode) -> Vec<Vec<usize>> {
        assert_eq!(input_shapes.len(), 1);
        assert_eq!(input_shapes[0].len(), 1);
        // for each input, we output an embedding vector
        vec![vec![input_shapes[0][0],self.emb.get_shape()[1]]]
    }
    
    fn num_outputs(&self, num_inputs: usize) -> usize {
        assert_eq!(num_inputs, 1,"no batch support for now");
        num_inputs
    }
    
    fn describe(&self) -> String {
        format!("Embeddings({:?})", self.emb.get_shape())
    }
    
    fn is_provable(&self) -> bool {
        true
    }
}

impl Embeddings<f32> {
    // TODO: make that a trait ? or part of the Layer enum ?
    pub fn from_loader(loader: &FileTensorLoader) -> anyhow::Result<Self> {
        let emb_tensor = loader.get_tensor("token_embd.weight")?;
        Ok(Embeddings::new(emb_tensor))
    }
}