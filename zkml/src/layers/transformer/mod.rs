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
    use crate::{Element, Tensor};
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

    /// Test if the following two operations are equivalent:
    /// 1. reshape, transpose, partition, merge back, transpose back
    /// 2. reshape, partition, merge back
    #[test]
    fn test_multihead_transpose() {
        let seq_len = 10;
        let emb_size = 16;
        let head_dim = 4;
        let num_heads = emb_size / head_dim;
        // only do one token for Q
        let q = Tensor::<Element>::random(&[1, emb_size]);
        let k = Tensor::<Element>::random(&[seq_len, emb_size]);
        let v = Tensor::<Element>::random(&[seq_len, emb_size]);
        // -----------------------
        // first technique:
        // -----------------------
        let qkt_v_1 =  {
        // go from [1, emb_size] to [1, num_heads, head_dim] to [num_heads, 1, head_dim]
        let q_reshaped = q.clone().reshape(vec![1, num_heads, head_dim]).permute3d(&vec![1,0,2]);
        // go from [seq_len, emb_size] to [seq_len, num_heads, head_dim]  to [num_heads, seq_len, head_dim]
        let k_reshaped = k.clone().reshape(vec![seq_len, num_heads, head_dim]).permute3d(&vec![1,0,2]);
        let v_reshaped = v.clone().reshape(vec![seq_len, num_heads, head_dim]).permute3d(&vec![1,0,2]);
        // now we split by heads and multiply each mini q with mini k transposed
        let qkt_heads = (0..num_heads).map(|head| {
            let mini_q = q_reshaped.slice_3d(head, head + 1).reshape(vec![1, head_dim]);
            let mini_k = k_reshaped.slice_3d(head, head + 1).reshape(vec![seq_len, head_dim]).transpose();
            // [1, head_dim] @ [head_dim, seq_len] = [1, seq_len]
            mini_q.matmul(&mini_k)
        }).collect::<Vec<_>>();
        let v_heads = (0..num_heads).map(|head| {
            let mini_v = v_reshaped.slice_3d(head, head + 1).reshape(vec![seq_len, head_dim]);
            // [seq_len, head_dim]
            mini_v
        }).collect::<Vec<_>>();
        let mut qkt_it = qkt_heads.into_iter();
        let mut v_it = v_heads.into_iter();
        // now we do the matmul with v, so each heads is now of [1, seq_len] @ [seq_len, head_dim] = [1, head_dim]
        // we then concat the results together so the end results is [num_heads, 1, head_dim]
        // here we add the third coordinate for the concat to "work", e.g. [1,a,b] || [1, a, b] == [2, a, b]
        let qkt = qkt_it.next().unwrap().matmul(&v_it.next().unwrap()).reshape(vec![1,1,head_dim]);
        let qkt_v= qkt_it.zip(v_it).fold(qkt, |mut acc, (qkt,v)| { 
            acc.concat(qkt.matmul(&v));
            acc 
        });
        println!("qkt_v shape: {:?}", qkt_v.get_shape());
        // transpose back to [1, num_heads, head_dim] and then reshape to [1, emb_size]
        let qkt_v = qkt_v.permute3d(&vec![1,0,2]).reshape(vec![1, emb_size]);
        qkt_v
        };

        // -----------------------
        // second technique: same but without any permutation
        // -----------------------

        let qkt_v_2 = {
                    // go from [1, emb_size] to to [num_heads, 1, head_dim]
        let q_reshaped = q.reshape(vec![num_heads, 1, head_dim]);
        // go from [seq_len, emb_size] to [num_heads, seq_len, head_dim]
        let k_reshaped = k.reshape(vec![num_heads, seq_len, head_dim]);
        let v_reshaped = v.reshape(vec![num_heads, seq_len, head_dim]);
        // now we split by heads and multiply each mini q with mini k transposed
        let qkt_heads = (0..num_heads).map(|head| {
            let mini_q = q_reshaped.slice_3d(head, head + 1).reshape(vec![1, head_dim]);
            let mini_k = k_reshaped.slice_3d(head, head + 1).reshape(vec![seq_len, head_dim]).transpose();
            // [1, head_dim] @ [head_dim, seq_len] = [1, seq_len]
            mini_q.matmul(&mini_k)
        }).collect::<Vec<_>>();
        let v_heads = (0..num_heads).map(|head| {
            let mini_v = v_reshaped.slice_3d(head, head + 1).reshape(vec![seq_len, head_dim]);
            // [seq_len, head_dim]
            mini_v
        }).collect::<Vec<_>>();
        let mut qkt_it = qkt_heads.into_iter();
        let mut v_it = v_heads.into_iter();
        let qkt = qkt_it.next().unwrap().matmul(&v_it.next().unwrap());
        // now we do the matmul with v, so each heads is now of [1, seq_len] @ [seq_len, head_dim] = [1, head_dim]
        // we then concat the results together so the end results is [num_heads, 1, head_dim]
        let qkt_v= qkt_it.zip(v_it).fold(qkt, |mut acc, (qkt,v)| { 
            acc.concat(qkt.matmul(&v));
            acc 
        });
        // transpose back to [1, num_heads, head_dim] and then reshape to [1, emb_size]
        let qkt_v = qkt_v.reshape(vec![1, emb_size]);
            qkt_v
        };

        assert_eq!(qkt_v_1, qkt_v_2);

    }
}