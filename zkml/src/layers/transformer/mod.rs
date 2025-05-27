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
        layers::{activation::{Activation, Relu}, add::{self, Add}, concat_matmul::{self, ConcatMatMul}, dense::Dense, mul, provable::Evaluate, reshape::{self, Reshape}}, parser::gguf::{self, tests::GPT2_Q8_0_PATH, FileTensorLoader, LLMConfig, LLMModel}, tensor::Number, Element, Tensor
    };

    use super::{layernorm, mha, qkv, softmax};

    // === FFN Block === //
    // LayerNorm before FFN
    // let ff_in = x_resid1.layer_norm(eps2); // [hidden_size]

    // // FFN: up -> activation -> down
    // let ff_up = ff_in.matmul(w_ff1); // [ff_dim]
    // let act = gelu(ff_up);           // [ff_dim]
    // let ff_down = act.matmul(w_ff2); // [hidden_size]

    // // Residual connection
    // let x_out = x_resid1 + ff_down; // [hidden_size]
    struct FlatFFN<N> {
        layernorm: layernorm::LayerNorm<N>,
        up: Dense<N>,
        activation: Activation,
        down: Dense<N>,
        add: Add<N>,
    }

    impl FlatFFN<f32> {
        pub fn new_from_gguf(c: &gguf::LLMConfig, ffn: gguf::FeedForward<f32>) -> Self {
            let layernorm = ffn.norm;
            let up = Dense::new(ffn.up, ffn.up_bias);
            let activation = Activation::Relu(Relu);
            let down = Dense::new(ffn.down, ffn.down_bias);
            let add = add::Add::new();
            Self {
                layernorm,
                up,
                activation,
                down,
                add,
            }
        }

        pub fn evaluate(&mut self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
            let normed = self.layernorm.evaluate::<GoldilocksExt2>(&vec![input], vec![])?;
            let up = self.up.evaluate::<GoldilocksExt2>(&normed.outputs(), vec![])?;
            let act = self.activation.evaluate::<GoldilocksExt2>(&up.outputs(), vec![])?;
            let down = self.down.evaluate::<GoldilocksExt2>(&act.outputs(), vec![])?;
            let out = self.add.evaluate::<GoldilocksExt2>(&vec![input, down.outputs()[0]])?;
            Ok(out.outputs()[0].clone())
        }
    }

    impl<N: Number>FlatFFN<N> {
        pub fn random(hidden_size: usize,up_size: usize) -> Self {
            let layernorm = layernorm::LayerNorm::random(hidden_size);
            let up = Dense::random(vec![up_size, hidden_size]);
            let activation = Activation::Relu(Relu);
            let down = Dense::random(vec![hidden_size, up_size]);
            let add = add::Add::new();
            Self {
                layernorm,
                up,
                activation,
                down,
                add,
            }
        }
    }

    // Test structure to just have a flat forward pass for the attention layer.
    // Goal is to move that structure to the graph structure once this produces the same
    // output as candle or burn with the same config and weights.
    // Once this flat impl is consistent, then we can compare with the graph version.
    // Once that is consistent too, we can delete.
    struct FlatAttention<N> {
        num_heads: usize,
        head_dim: usize,
        hidden_size: usize,
        qkv: qkv::QKV<N>,
        qkt_v: ConcatMatMul,
        scaler: mul::ScalarMul<f32>,
        layernorm: layernorm::LayerNorm<N>,
        mha: mha::MHA_QK,
        cache: qkv::CacheQKV<N>,
        out: Dense<N>,
        reshape_merged: Reshape,
        reshape_qkt: Reshape,
        add: add::Add<N>,
        ffn: FlatFFN<N>,
    }

    impl FlatAttention<f32> {
        pub fn new_from_gguf(c: &gguf::LLMConfig, att: gguf::Attention<f32>) -> Self {
            let qkv = qkv::QKV::new(att.q, att.q_bias, att.k, att.k_bias, att.v, att.v_bias);
            let reshape_qkt = reshape::Reshape::new_squeeze(1);
            let mha = mha::MHA_QK::new(c.num_heads, c.head_dim());
            let scaler = mul::ScalarMul::new((1.0 / (c.head_dim() as f32)).sqrt());
            let ffn = FlatFFN::new_from_gguf(c, att.feedforward);
            Self {
                out: Dense::new(att.out, att.out_bias),
                hidden_size: c.hidden_size,
                num_heads: c.num_heads,
                head_dim: c.head_dim(),
                qkv,
                qkt_v: concat_matmul::ConcatMatMul::new_with_transpose(vec![1, 0, 2]),
                scaler,
                layernorm: att.norm,
                cache: qkv::CacheQKV::new(),
                mha,
                reshape_merged: Reshape::new_fixed(vec![vec![1, c.hidden_size]]),
                reshape_qkt,
                ffn,
                add: add::Add::new(),
            }
        }

        /// currently hardcoded for f32 - need to implement layernorm and softmax in quantized world to be generic over N
        pub fn forward(&mut self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
            assert_eq!(input.get_shape().len(), 2);
            let seq_len = input.get_shape()[0];
            let normed = self
                .layernorm
                .evaluate::<GoldilocksExt2>(&vec![input], vec![])?;
            let qkv = self
                .qkv
                .evaluate::<GoldilocksExt2>(&normed.outputs(), &mut self.cache)?;
            let mha = self.mha.evaluate::<_, GoldilocksExt2>(&qkv.outputs())?;
            // apply softmax on the first output, Q @ K^T
            let qkt = vec![mha.outputs()[0]];
            // but first, we need to scale it down by 1/sqrt(head_dim)
            let scaled = self.scaler.evaluate::<_, GoldilocksExt2>(&qkt)?;
            let softmax_layer = softmax::Softmax;
            // then we apply softmax row by row
            let softmaxed = softmax_layer.evaluate::<GoldilocksExt2>(&scaled.outputs(), vec![])?;
            #[cfg(test)]
            {
                let qkt_shape = softmaxed.outputs()[0].get_shape();
                let v_shape = mha.outputs()[1].get_shape();
                println!("qkt shape: {:?}, v shape: {:?}", qkt_shape, v_shape);
                assert_eq!(v_shape, vec![self.num_heads, seq_len, self.head_dim]);
                 // qk is now of shape [num_heads,seq_len]
                assert_eq!(qkt_shape, vec![self.num_heads, seq_len]);
            }
            // We reshape to [num_heads, 1, seq_len] such concat_matmul can work, since it expects tensors of same shape
            let qkt_reshaped = self.reshape_qkt.evaluate::<_,GoldilocksExt2>(&softmaxed.outputs())?;
            // now we can project back with V
            // We go from [num_heads, 1, head_dim] → transpose back to [1, h, head_dim] 
            let qkt_v = self.qkt_v.evaluate::<_, GoldilocksExt2>(&vec![qkt_reshaped.outputs()[0], mha.outputs()[1]])?;
            // → and reshape to [1, hidden_size]
            let merged = self.reshape_merged.evaluate::<_,GoldilocksExt2>(&qkt_v.outputs())?;
            // now we do the final projection - still [1,hidden_size]
            let projected = self.out.evaluate::<GoldilocksExt2>(&merged.outputs(), vec![])?;
            // and then residual connection, [1, hidden_size]
            let out = self.add.evaluate::<GoldilocksExt2>(&vec![input, &projected.outputs()[0]])?;
            // and then FFN
            let ffn_out = self.ffn.evaluate(&out.outputs()[0])?;
            Ok(ffn_out)
        }
    }

    impl<N: Number> FlatAttention<N> {
        pub fn random(emb_size: usize, num_heads: usize) -> Self {
            // Note in LLM, it's always the case that hidden_size = emb_size so we can apply residual 
            let hidden_size = emb_size;
            let head_size = hidden_size / num_heads;
            let qkv = qkv::QKV::random(emb_size, hidden_size);
            let mha = mha::MHA_QK::new(num_heads, head_size);
            let scaler = mul::ScalarMul::new((1.0 / (head_size as f32)).sqrt());
            let layernorm = layernorm::LayerNorm::random(emb_size);
            let out= Dense::random(vec![hidden_size, hidden_size]);
            let ffn = FlatFFN::random(hidden_size, hidden_size);
            Self {
                out,
                hidden_size,
                num_heads,
                head_dim: head_size,
                qkv,
                qkt_v: concat_matmul::ConcatMatMul::new_with_transpose(vec![1, 0, 2]),
                scaler,
                layernorm,
                cache: qkv::CacheQKV::new(),
                mha,
                reshape_merged: Reshape::new_fixed(vec![vec![1, hidden_size]]),
                reshape_qkt: Reshape::new_squeeze(1),
                add: Add::new(),
                ffn,
            }
        }
    }

    #[test]
    fn test_flat_attention_random() {
        let emb_size = 10;
        let num_heads = 2;
        let mut att = FlatAttention::random(emb_size, num_heads);
        let input = Tensor::<f32>::random(&[1, emb_size]);
        let output = att.forward(&input).unwrap();
        println!("output shape: {:?}", output.get_shape());
    }

    #[test]
    fn test_flat_attention_from_gguf() -> anyhow::Result<()> {
        let loader = FileTensorLoader::from_path(GPT2_Q8_0_PATH)?;
        let config = LLMConfig::from_content(&loader)?;
        let LLMModel::GPT2(mut model) = config.model(&loader)? else {
            bail!("Model is not a GPT2 model");
        };
        println!("model: {:?}", config.specific_config);
        let mut att = FlatAttention::new_from_gguf(&config, model.blocks.remove(0));
        let input = Tensor::<f32>::random(&[1, config.embedding_size]);
        let output = att.forward(&input).unwrap();
        println!("output shape: {:?}",output.get_shape());
        Ok(())
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