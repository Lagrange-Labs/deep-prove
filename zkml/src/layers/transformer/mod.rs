pub mod embeddings;
pub mod layernorm;
pub mod mha;
pub mod positional;
// pub mod qkt;
pub mod qkv;
pub mod softmax;

#[cfg(test)]
mod test {
    use std::{fs::File, path::PathBuf};

    use anyhow::Context;
    use goldilocks::GoldilocksExt2;
    use serde::Deserialize;

    use crate::{
        layers::{
            activation::GELU,
            add::{self, Add},
            concat_matmul::{self, ConcatMatMul},
            dense::Dense,
            provable::Evaluate,
            reshape::{self, Reshape},
        }, model::Model, padding::PaddingMode, parser::{
            file_cache, gguf::{tests::GPT2_Q8_0_URL, FileTensorLoader}, json::test::{TINY_GPT2_DEBUG_NAME, TINY_GPT2_NAME}, llm::{Attention, FeedForward, LLMConfig, LLMModel}
        }, tensor::Number, Tensor
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
        activation: GELU<N>,
        down: Dense<N>,
        add: Add<N>,
    }

    impl FlatFFN<f32> {
        pub fn new_from_gguf(_c: &LLMConfig, ffn: FeedForward<f32>) -> Self {
            let layernorm = ffn.norm;
            let up = Dense::new(ffn.up, ffn.up_bias);
            let activation = GELU::new();
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

        pub fn evaluate(
            &mut self,
            input: &Tensor<f32>,
            output: Option<&GPT2LayerOutput>,
        ) -> anyhow::Result<Tensor<f32>> {
            let normed = self
                .layernorm
                .evaluate::<GoldilocksExt2>(&vec![input], vec![])?;
            if let Some(gpt2_output) = output {
                gpt2_output.is_prefnn_layernorm_close(normed.outputs());
            }
            let up = self
                .up
                .evaluate::<GoldilocksExt2>(&normed.outputs(), vec![])?;
            if let Some(gpt2_output) = output {
                assert!(gpt2_output.is_ffn_up_close(up.outputs()));
            }
            let act = self
                .activation
                .evaluate::<GoldilocksExt2>(&up.outputs(), vec![])?;
            let down = self
                .down
                .evaluate::<GoldilocksExt2>(&act.outputs(), vec![])?;
            let out = self.add.evaluate::<GoldilocksExt2>(
                &vec![input, &down.outputs()[0]],
                vec![
                    input.get_shape().to_vec(),
                    down.outputs()[0].get_shape().to_vec(),
                ],
            )?;
            Ok(out.outputs()[0].clone())
        }
    }

    impl<N: Number> FlatFFN<N> {
        pub fn random(hidden_size: usize, up_size: usize) -> Self {
            let layernorm = layernorm::LayerNorm::random(hidden_size);
            let up = Dense::random(vec![up_size, hidden_size]);
            let activation = GELU::new();
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
        #[allow(dead_code)]
        hidden_size: usize,
        qkv: qkv::QKV<N>,
        qkt_v: ConcatMatMul,
        layernorm: layernorm::LayerNorm<N>,
        mha: mha::MhaQK,
        softmax: softmax::Softmax<N>,
        out: Dense<N>,
        reshape_merged: Reshape,
        reshape_qkt: Reshape,
        add: add::Add<N>,
        ffn: FlatFFN<N>,
    }

    impl FlatAttention<f32> {
        pub fn new_from_gguf(c: &LLMConfig, att: Attention<f32>) -> Self {
            let qkv = qkv::QKV::new(att.q, att.q_bias, att.k, att.k_bias, att.v, att.v_bias);
            let reshape_qkt = reshape::Reshape::new_squeeze(1);
            let mha = mha::MhaQK::new(c.num_heads, c.head_dim());
            let ffn = FlatFFN::new_from_gguf(c, att.feedforward);
            Self {
                out: Dense::new(att.out, att.out_bias),
                hidden_size: c.hidden_size,
                num_heads: c.num_heads,
                head_dim: c.head_dim(),
                qkv,
                qkt_v: concat_matmul::ConcatMatMul::new_with_permute(vec![1, 0, 2]),
                softmax: softmax::Softmax::new_with_scale((1.0 / (c.head_dim() as f32)).sqrt()),
                layernorm: att.norm,
                mha,
                reshape_merged: Reshape::new_fixed(vec![vec![1, c.hidden_size]]),
                reshape_qkt,
                ffn,
                add: add::Add::new(),
            }
        }

        /// currently hardcoded for f32 - need to implement layernorm and softmax in quantized world to be generic over N
        pub fn forward(
            &mut self,
            input: &Tensor<f32>,
            gpt2_output: Option<&GPT2LayerOutput>,
        ) -> anyhow::Result<Tensor<f32>> {
            assert_eq!(input.get_shape().len(), 2);
            let seq_len = input.get_shape()[0];
            let normed = self
                .layernorm
                .evaluate::<GoldilocksExt2>(&vec![input], vec![])?;
            if let Some(gpt2_output) = gpt2_output {
                assert!(gpt2_output.is_layernorm_close(normed.outputs()));
            }
            let qkv = self
                .qkv
                .evaluate::<GoldilocksExt2>(&normed.outputs(), vec![])?;
            if let Some(gpt2_output) = gpt2_output {
                assert!(gpt2_output.is_qkv_close(qkv.outputs()));
            }
            let mha = self
                .mha
                .evaluate::<GoldilocksExt2>(&qkv.outputs(), vec![])?;
            // apply softmax + rescale on the first output, Q @ K^T
            // NOTE that we apply softmax row by row
            let softmaxed = self
                .softmax
                .evaluate::<GoldilocksExt2>(&[mha.outputs()[0]], vec![])?;
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
            let qkt_reshaped = self
                .reshape_qkt
                .evaluate::<GoldilocksExt2>(&softmaxed.outputs(), vec![])?;
            // now we can project back with V
            // We go from [num_heads, 1, head_dim] → transpose back to [1, h, head_dim]
            let qkt_v = self.qkt_v.evaluate::<GoldilocksExt2>(
                &vec![qkt_reshaped.outputs()[0], mha.outputs()[1]],
                vec![],
            )?;
            // → and reshape to [1, hidden_size]
            let merged = self
                .reshape_merged
                .evaluate::<GoldilocksExt2>(&qkt_v.outputs(), vec![])?;
            if let Some(gpt2_output) = gpt2_output {
                assert!(gpt2_output.is_attention_mha_output_close(merged.outputs()));
            }
            // now we do the final projection - still [1,hidden_size]
            let projected = self
                .out
                .evaluate::<GoldilocksExt2>(&merged.outputs(), vec![])?;
            if let Some(gpt2_output) = gpt2_output {
                assert!(gpt2_output.is_attention_output_proj_close(projected.outputs()));
            }
            // and then residual connection, [1, hidden_size]
            let out = self.add.evaluate::<GoldilocksExt2>(
                &vec![input, &projected.outputs()[0]],
                vec![
                    input.get_shape().to_vec(),
                    projected.outputs()[0].get_shape().to_vec(),
                ],
            )?;
            if let Some(gpt2_output) = gpt2_output {
                assert!(gpt2_output.is_residual_attn_close(out.outputs()));
            }
            // and then FFN
            let ffn_out = self.ffn.evaluate(&out.outputs()[0], gpt2_output)?;
            Ok(ffn_out)
        }
    }

    impl<N: Number> FlatAttention<N> {
        pub fn random(emb_size: usize, num_heads: usize) -> Self {
            // Note in LLM, it's always the case that hidden_size = emb_size so we can apply residual
            let hidden_size = emb_size;
            let head_size = hidden_size / num_heads;
            let qkv = qkv::QKV::random(emb_size, hidden_size);
            let mha = mha::MhaQK::new(num_heads, head_size);
            let layernorm = layernorm::LayerNorm::random(emb_size);
            let out = Dense::random(vec![hidden_size, hidden_size]);
            let ffn = FlatFFN::random(hidden_size, hidden_size);
            Self {
                out,
                hidden_size,
                num_heads,
                head_dim: head_size,
                qkv,
                qkt_v: concat_matmul::ConcatMatMul::new_with_permute(vec![1, 0, 2]),
                softmax: softmax::Softmax::new_with_scale(
                    N::from_f32((1.0 / (head_size as f32)).sqrt()).unwrap(),
                ),
                layernorm,
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
        let output = att.forward(&input, None).unwrap();
        println!("output shape: {:?}", output.get_shape());
    }

    #[test]
    fn test_flat_attention_from_gguf() -> anyhow::Result<()> {
        let path = file_cache::ensure_downloaded(GPT2_Q8_0_URL)?;
        let loader = FileTensorLoader::from_path(path)?;
        let config = LLMConfig::from_content(&loader)?;
        let LLMModel::GPT2(mut model) = config.model(&loader)?;
        println!("model: {:?}", config.specific_config);
        let mut att = FlatAttention::new_from_gguf(&config, model.blocks.remove(0));
        let input = Tensor::<f32>::random(&[1, config.embedding_size]);
        let output = att.forward(&input, None).unwrap();
        println!("output shape: {:?}", output.get_shape());
        Ok(())
    }

    #[derive(Debug, Deserialize)]
    struct GPT2Output {
        #[allow(dead_code)]
        token: String,
        #[allow(dead_code)]
        input_ids: u32,
        inputs_embeds: Vec<f32>,
        layers: Vec<GPT2LayerOutput>,
    }
    #[derive(Debug, Deserialize)]
    struct GPT2LayerOutput {
        ln1_out: Vec<f32>,
        ln2_out: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        attn_output: Vec<f32>,
        attn_output_proj: Vec<f32>,
        residual_attn: Vec<f32>,
        ffn_up: Vec<f32>,
        manual_output: Vec<f32>,
        // Optional field for the final layer with LayerNorm applied
        manual_output_with_final_ln: Option<Vec<f32>>,
    }

    impl GPT2LayerOutput {
        pub fn is_qkv_close(&self, qkv: Vec<&Tensor<f32>>) -> bool {
            let q = qkv[0];
            let k = qkv[1];
            let v = qkv[2];
            let q_close = is_close(q.get_data(), &self.q);
            let k_close = is_close(k.get_data(), &self.k);
            let v_close = is_close(v.get_data(), &self.v);
            q_close && k_close && v_close
        }

        pub fn is_layernorm_close(&self, layernorm: Vec<&Tensor<f32>>) -> bool {
            is_close(layernorm[0].get_data(), &self.ln1_out)
        }
        pub fn is_attention_mha_output_close(&self, mha_output: Vec<&Tensor<f32>>) -> bool {
            is_close(mha_output[0].get_data(), &self.attn_output)
        }
        pub fn is_attention_output_proj_close(&self, output_proj: Vec<&Tensor<f32>>) -> bool {
            is_close(output_proj[0].get_data(), &self.attn_output_proj)
        }
        pub fn is_residual_attn_close(&self, residual_attn: Vec<&Tensor<f32>>) -> bool {
            is_close(residual_attn[0].get_data(), &self.residual_attn)
        }
        pub fn is_prefnn_layernorm_close(&self, ln2_out: Vec<&Tensor<f32>>) -> bool {
            is_close(ln2_out[0].get_data(), &self.ln2_out)
        }
        pub fn is_ffn_up_close(&self, ffn_up: Vec<&Tensor<f32>>) -> bool {
            is_close(ffn_up[0].get_data(), &self.ffn_up)
        }
    }

    // taken from https://docs.pytorch.org/docs/stable/generated/torch.isclose.html
    fn is_close(a: &[f32], b: &[f32]) -> bool {
        let atol = 1e-8_f32;
        let rtol = 1e-5_f32;
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| {
            let diff = (x - y).abs();
            diff <= atol + rtol * y.abs()
        })
    }

    use crate::parser::json;

    #[test]
    fn test_read_gpt2_pytorch_output() -> anyhow::Result<()> {
        let model_weights_path = json::test::get_json_file(TINY_GPT2_NAME)?;
        let debug_output_path = json::test::get_json_file(TINY_GPT2_DEBUG_NAME)?;

        let loader = json::FileTensorLoader::new_from_path(model_weights_path)?;
        let config = LLMConfig::from_json(&loader)?;
        println!("config: {:?}", config);

        let gpt2_output = serde_json::from_reader::<_, GPT2Output>(
            File::open(debug_output_path.clone()).context(format!("failed to open file {}",debug_output_path.clone()))?
        )?;
        let input = Tensor::new(
            vec![1, config.embedding_size],
            gpt2_output.inputs_embeds.clone(),
        );
        let LLMModel::GPT2(mut model) = config.model_json(&loader)?;
        println!("model: {:?}", config.specific_config);
        // Try to run with the flat attention implementation
        let first_attention = model.blocks.remove(0);
        let mut att = FlatAttention::new_from_gguf(&config, first_attention.clone());
        let first_layer_output = gpt2_output.layers.get(0).expect("no layers in output");
        let output = att.forward(&input, Some(first_layer_output)).unwrap();
        let expected_output = &first_layer_output.manual_output;
        assert!(is_close(expected_output, &output.get_data()));
        // Now try to run with the graph implementation
        let mut model = Model::new_from_input_shapes(vec![input.get_shape()], PaddingMode::NoPadding);
        let _last_node_id = first_attention.write_to_model(&mut model, None, &config)?;
        model.route_output(None)?;
        let output = model.run_float(&[input.clone()])?;
        assert!(is_close(expected_output, &output[0].get_data()),"graph output differs");
        Ok(())
    }

    
}
