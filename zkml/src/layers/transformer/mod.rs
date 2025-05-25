use crate::tensor::Number;

pub mod embeddings;
pub mod layernorm;
pub mod positional;
pub mod qkv;
pub mod qkt;
pub mod mha;
pub mod softmax;

#[cfg(test)]
mod test {
    use crate::{layers::reshape, parser::gguf};

    use super::qkv;


    // temporary test to implement a transformer logic.
    // Once this is working, next step is to form the graph with the framework and compare outputs
    // if both outputs are the same, we can get rid of that test logic.
    // The goal is to see in one function what is the transformer logic
    fn gguf_to_inference_logic(c: gguf::LLMConfig, att: gguf::Attention<f32>) {
        let layer_norm = att.norm;
        let qkv = qkv::QKV::new(att.q, att.q_bias, att.k, att.k_bias, att.v, att.v_bias);
        // [1, d_model] → reshape → [1, h, head_dim]
        let reshape_q = reshape::Reshape::new_fixed(vec![vec![c.embedding_size]]);
        // [seq_len, d_model] → reshape → [seq_len, h, head_dim]
        let reshape_kv = reshape::Reshape::new_subspace(1..2, vec![c.num_heads, c.head_dim()]);
    } 
}