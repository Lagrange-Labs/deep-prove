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
    use anyhow::bail;
    use goldilocks::GoldilocksExt2;

    use crate::layers::mul;
    use crate::layers::provable::Evaluate;
    use crate::Tensor;
    use crate::{layers::reshape, parser::gguf};

    use super::{layernorm, softmax};
    use super::qkv;
    use super::mha;

    // Test structure to just have a flat forward pass for the attention layer. 
    // Goal is to move that structure to the graph structure once this produces the same
    // output as candle or burn with the same config and weights.
    // Once this flat impl is consistent, then we can compare with the graph version.
    // Once that is consistent too, we can delete.
    struct FlatAttention {
        qkv: qkv::QKV<f32>,
        scaler: mul::ScalarMul<f32>,
        reshape_q: reshape::Reshape,
        reshape_kv: reshape::Reshape,
        layernorm: layernorm::LayerNorm<f32>,
        mha: mha::MHA_QK,
        cache: qkv::CacheQKV<f32>,
    }

    impl FlatAttention {
        pub fn new(c: gguf::LLMConfig, att: gguf::Attention<f32>) -> Self {
            let qkv = qkv::QKV::new(att.q, att.q_bias, att.k, att.k_bias, att.v, att.v_bias);
            // [1, d_model] → reshape → [1, h, head_dim]
            let reshape_q = reshape::Reshape::new_fixed(vec![vec![c.embedding_size]]);
            // [seq_len, d_model] → reshape → [seq_len, h, head_dim]
            let reshape_kv = reshape::Reshape::new_subspace(1..2, vec![c.num_heads, c.head_dim()]);
            let mha = mha::MHA_QK::new(c.num_heads, c.head_dim());
            let scaler = mul::ScalarMul::new((1.0 / (c.head_dim() as f32)).sqrt());
            Self { qkv, reshape_q, reshape_kv, scaler, layernorm: att.norm, cache: qkv::CacheQKV::new(), mha }
        }

        pub fn forward(&mut self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
            let input = self.layernorm.evaluate::<GoldilocksExt2>(&vec![input], vec![])?;
            let qkv = self.qkv.evaluate::<GoldilocksExt2>(&input.outputs(), &mut self.cache)?;
            let mha = self.mha.evaluate::<_,GoldilocksExt2>(&qkv.outputs())?;
            // apply softmax on the first output, Q @ K^T
            let qkt = vec![mha.outputs()[0]];
            // but first, we need to scale it down by 1/sqrt(head_dim)
            let scaled = self.scaler.evaluate::<_, GoldilocksExt2>(&qkt)?;
            let softmax_layer= softmax::Softmax;
            let softmaxed = softmax_layer.evaluate::<GoldilocksExt2>(&scaled.outputs(), vec![])?;
            // now we can project back with V
            
            
            bail!("Not implemented");
        }
    }

    #[test]
    fn test_flat_attention() {

    }
}