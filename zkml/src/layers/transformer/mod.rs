use crate::tensor::Number;

pub mod embeddings;
pub mod layernorm;
pub mod mha;
pub mod positional;
pub mod qkt;
pub mod qkv;
pub mod softmax;

#[cfg(test)]
mod test {
    use anyhow::bail;
    use goldilocks::GoldilocksExt2;

    use crate::{
        layers::{concat_matmul, mul, provable::Evaluate, reshape}, parser::gguf, tensor::Number, Element, Tensor
    };

    use super::{layernorm, mha, qkv, softmax};

    // Test structure to just have a flat forward pass for the attention layer.
    // Goal is to move that structure to the graph structure once this produces the same
    // output as candle or burn with the same config and weights.
    // Once this flat impl is consistent, then we can compare with the graph version.
    // Once that is consistent too, we can delete.
    struct FlatAttention<N> {
        num_heads: usize,
        head_dim: usize,
        qkv: qkv::QKV<N>,
        scaler: mul::ScalarMul<f32>,
        reshape_q: reshape::Reshape,
        reshape_kv: reshape::Reshape,
        layernorm: layernorm::LayerNorm<N>,
        mha: mha::MHA_QK,
        cache: qkv::CacheQKV<N>,
    }

    impl FlatAttention<f32> {
        pub fn new_from_gguf(c: gguf::LLMConfig, att: gguf::Attention<f32>) -> Self {
            let qkv = qkv::QKV::new(att.q, att.q_bias, att.k, att.k_bias, att.v, att.v_bias);
            // [1, d_model] → reshape → [1, h, head_dim]
            let reshape_q = reshape::Reshape::new_fixed(vec![vec![c.embedding_size]]);
            // [seq_len, d_model] → reshape → [seq_len, h, head_dim]
            let reshape_kv = reshape::Reshape::new_subspace(1..2, vec![c.num_heads, c.head_dim()]);
            let mha = mha::MHA_QK::new(c.num_heads, c.head_dim());
            let scaler = mul::ScalarMul::new((1.0 / (c.head_dim() as f32)).sqrt());
            Self {
                num_heads: c.num_heads,
                head_dim: c.head_dim(),
                qkv,
                reshape_q,
                reshape_kv,
                scaler,
                layernorm: att.norm,
                cache: qkv::CacheQKV::new(),
                mha,
            }
        }

        /// currently hardcoded for f32 - need to implement layernorm and softmax in quantized world to be generic over N
        pub fn forward(&mut self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
            assert_eq!(input.get_shape().len(), 2);
            let seq_len = input.get_shape()[0];
            let input = self
                .layernorm
                .evaluate::<GoldilocksExt2>(&vec![input], vec![])?;
            let qkv = self
                .qkv
                .evaluate::<GoldilocksExt2>(&input.outputs(), &mut self.cache)?;
            let mha = self.mha.evaluate::<_, GoldilocksExt2>(&qkv.outputs())?;
            // apply softmax on the first output, Q @ K^T
            let qkt = vec![mha.outputs()[0]];
            // but first, we need to scale it down by 1/sqrt(head_dim)
            let scaled = self.scaler.evaluate::<_, GoldilocksExt2>(&qkt)?;
            let softmax_layer = softmax::Softmax;
            // then we apply softmax row by row
            let softmaxed = softmax_layer.evaluate::<GoldilocksExt2>(&scaled.outputs(), vec![])?;
            let qkt_shape = softmaxed.outputs()[0].get_shape();
            let v_shape = mha.outputs()[1].get_shape();
            println!("qkt shape: {:?}, v shape: {:?}", qkt_shape, v_shape);
            assert_eq!(v_shape, vec![self.num_heads, seq_len, self.head_dim]);
             // qk is now of shape [num_heads,seq_len]
            assert_eq!(qkt_shape, vec![self.num_heads, seq_len]);
            // We reshape to [num_heads, 1, seq_len] such concat_matmul can work, since it expects tensors of same shape
            let qkt_reshaped = softmaxed.outputs()[0].clone().reshape(vec![self.num_heads, 1, seq_len]);
            // now we can project back with V
            let op = concat_matmul::ConcatMatMul;
            let qkt_v = op
                .evaluate::<f32, GoldilocksExt2>(&vec![&qkt_reshaped, mha.outputs()[1]])?;
            Ok(qkt_v.outputs()[0].clone())
        }
    }

    impl<N: Number> FlatAttention<N> {
        pub fn random(emb_size: usize, num_heads: usize, hidden_size: usize) -> Self {
            let head_size = hidden_size / num_heads;
            let qkv = qkv::QKV::random(emb_size, hidden_size);
            let reshape_q = reshape::Reshape::new_fixed(vec![vec![emb_size]]);
            let reshape_kv = reshape::Reshape::new_subspace(1..2, vec![num_heads, head_size]);
            let mha = mha::MHA_QK::new(num_heads, head_size);
            let scaler = mul::ScalarMul::new((1.0 / (head_size as f32)).sqrt());
            let layernorm = layernorm::LayerNorm::random(emb_size);
            Self {
                num_heads,
                head_dim: head_size,
                qkv,
                reshape_q,
                reshape_kv,
                scaler,
                layernorm,
                cache: qkv::CacheQKV::new(),
                mha,
            }
        }
    }

    #[test]
    fn test_flat_attention() {
        let emb_size = 10;
        let num_heads = 2;
        let hidden_size = 16;
        let mut att = FlatAttention::random(emb_size, num_heads, hidden_size);
        let input = Tensor::<f32>::random(&[1, emb_size]);
        let output = att.forward(&input).unwrap();
        println!("output shape: {:?}", output.get_shape());
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
        let qkt_v_1 = {
            // go from [1, emb_size] to [1, num_heads, head_dim] to [num_heads, 1, head_dim]
            let q_reshaped = q
                .clone()
                .reshape(vec![1, num_heads, head_dim])
                .permute3d(&vec![1, 0, 2]);
            // go from [seq_len, emb_size] to [seq_len, num_heads, head_dim]  to [num_heads, seq_len, head_dim]
            let k_reshaped = k
                .clone()
                .reshape(vec![seq_len, num_heads, head_dim])
                .permute3d(&vec![1, 0, 2]);
            let v_reshaped = v
                .clone()
                .reshape(vec![seq_len, num_heads, head_dim])
                .permute3d(&vec![1, 0, 2]);
            // now we split by heads and multiply each mini q with mini k transposed
            let qkt_heads = (0..num_heads)
                .map(|head| {
                    let mini_q = q_reshaped
                        .slice_3d(head, head + 1)
                        .reshape(vec![1, head_dim]);
                    let mini_k = k_reshaped
                        .slice_3d(head, head + 1)
                        .reshape(vec![seq_len, head_dim])
                        .transpose();
                    // [1, head_dim] @ [head_dim, seq_len] = [1, seq_len]
                    mini_q.matmul(&mini_k)
                })
                .collect::<Vec<_>>();
            let v_heads = (0..num_heads)
                .map(|head| {
                    let mini_v = v_reshaped
                        .slice_3d(head, head + 1)
                        .reshape(vec![seq_len, head_dim]);
                    // [seq_len, head_dim]
                    mini_v
                })
                .collect::<Vec<_>>();
            let mut qkt_it = qkt_heads.into_iter();
            let mut v_it = v_heads.into_iter();
            // now we do the matmul with v, so each heads is now of [1, seq_len] @ [seq_len, head_dim] = [1, head_dim]
            // we then concat the results together so the end results is [num_heads, 1, head_dim]
            // here we add the third coordinate for the concat to "work", e.g. [1,a,b] || [1, a, b] == [2, a, b]
            let qkt = qkt_it
                .next()
                .unwrap()
                .matmul(&v_it.next().unwrap())
                .reshape(vec![1, 1, head_dim]);
            let qkt_v = qkt_it.zip(v_it).fold(qkt, |mut acc, (qkt, v)| {
                acc.concat(qkt.matmul(&v));
                acc
            });
            println!("qkt_v shape: {:?}", qkt_v.get_shape());
            // transpose back to [1, num_heads, head_dim] and then reshape to [1, emb_size]
            let qkt_v = qkt_v.permute3d(&vec![1, 0, 2]).reshape(vec![1, emb_size]);
            qkt_v
        };
    }
}