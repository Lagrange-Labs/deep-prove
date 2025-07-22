use crate::Tensor;

pub mod embeddings;
pub mod layernorm;
pub mod logits;
pub mod mha;
pub mod positional;
pub mod qkv;
pub mod softmax;

/// Normally q_len == seq_len when the input is contains multiple tokens.
/// However, if the input contains 1 token, (e.g. with caching for example) then
/// q_len == 1 and seq_len > 1 if we are down multiple inference steps.
pub fn causal_mask(num_heads: usize, q_len: usize, seq_len: usize) -> Tensor<f32> {
    assert!(q_len == 1 || q_len == seq_len);
    let mask_len = num_heads * q_len * seq_len;
    let mut mask = vec![0.0; mask_len];
    for h in 0..num_heads {
        for i in 0..q_len {
            for j in 0..seq_len {
                if j > i {
                    mask[h * q_len * seq_len + i * seq_len + j] = -1e9;
                }
            }
        }
    }
    Tensor::new(vec![num_heads, q_len, seq_len].into(), mask)
}

#[cfg(test)]
pub(crate) mod test {
    use std::fs::File;

    use anyhow::{Context, ensure};
    use ark_std::rand::{Rng, thread_rng};
    use ff_ext::GoldilocksExt2;
    use serde::Deserialize;

    use crate::{
        Tensor, init_test_logging,
        layers::{
            activation::GELU,
            add::{self, Add},
            matrix_mul::{MatMul, OperandMatrix},
            provable::Evaluate,
        },
        model::Model,
        padding::PaddingMode,
        parser::{
            file_cache,
            gguf::{FileTensorLoader, tests::GPT2_Q8_0_URL},
            json::test::{TINY_GPT2_DEBUG_NAME, TINY_GPT2_NAME},
            llm::{Attention, FeedForward, LLMConfig, LLMModel},
        },
        tensor::{Number, Shape, is_close},
    };

    use super::{layernorm, mha, qkv};

    struct FlatFFN<N> {
        layernorm: layernorm::LayerNorm<N>,
        up: MatMul<N>,
        activation: GELU<N>,
        down: MatMul<N>,
        add: Add<N>,
    }

    impl FlatFFN<f32> {
        pub fn new_from_gguf(_c: &LLMConfig, ffn: FeedForward<f32>) -> Self {
            let layernorm = ffn.norm;
            let up = {
                // normally we would do this
                // Dense::new(ffn.up, ffn.up_bias);
                // but since we are multiplying this matrix FOR EACH TOKEN in the sequence,
                // this becomes a matrix multiplication in practice.
                let weight = OperandMatrix::new_weight_matrix(ffn.up);
                let left = OperandMatrix::Input;
                MatMul::new(left, weight).expect("failed to create MatMul")
            };
            let activation = GELU::new();
            let down = {
                // normally we would do this
                // Dense::new(ffn.down, ffn.down_bias);
                let weight = OperandMatrix::new_weight_matrix(ffn.down);
                let left = OperandMatrix::Input;
                MatMul::new(left, weight).expect("failed to create MatMul")
            };
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
            if let Some(gpt2_output) = output {
                assert!(gpt2_output.is_ffn_after_gelu_close(act.outputs()));
            }
            let down = self
                .down
                .evaluate::<GoldilocksExt2>(&act.outputs(), vec![])?;
            if let Some(gpt2_output) = output {
                assert!(gpt2_output.is_ffn_after_down_close(down.outputs()));
            }
            let out = self.add.evaluate::<GoldilocksExt2>(
                &vec![input, &down.outputs()[0]],
                vec![input.get_shape(), down.outputs()[0].get_shape()],
            )?;
            Ok(out.outputs()[0].clone())
        }
    }

    impl<N: Number> FlatFFN<N> {
        pub fn random(hidden_size: usize, up_size: usize) -> Self {
            let layernorm = layernorm::LayerNorm::random(hidden_size);
            let up = {
                let weight = OperandMatrix::new_weight_matrix(Tensor::random(
                    &vec![up_size, hidden_size].into(),
                ));
                let left = OperandMatrix::Input;
                MatMul::new(left, weight).expect("failed to create MatMul")
            };
            let activation = GELU::new();
            let down = {
                // Dense::random(vec![up_size, hidden_size]);
                let weight = OperandMatrix::new_weight_matrix(Tensor::random(
                    &vec![hidden_size, up_size].into(),
                ));
                let left = OperandMatrix::Input;
                MatMul::new(left, weight).expect("failed to create MatMul")
            };
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
        #[allow(dead_code)]
        num_heads: usize,
        #[allow(dead_code)]
        head_dim: usize,
        #[allow(dead_code)]
        hidden_size: usize,
        qkv: qkv::QKV<N>,
        layernorm: layernorm::LayerNorm<N>,
        mha: mha::Mha<N>,
        out: MatMul<N>,
        add: add::Add<N>,
        ffn: FlatFFN<N>,
    }

    impl FlatAttention<f32> {
        pub fn new_from_parser(c: &LLMConfig, att: Attention<f32>) -> anyhow::Result<Self> {
            let qkv = qkv::QKV::new(
                att.q,
                att.q_bias,
                att.k,
                att.k_bias,
                att.v,
                att.v_bias,
                c.num_heads,
            )?;
            let mha = mha::Mha::new(c.context_length, c.num_heads, c.head_dim())?;
            let ffn = FlatFFN::new_from_gguf(c, att.feedforward);

            let out = {
                let weight = OperandMatrix::new_weight_matrix(att.out);
                let left = OperandMatrix::Input;
                MatMul::new(left, weight).expect("failed to create MatMul")
            };

            Ok(Self {
                out,
                hidden_size: c.hidden_size,
                num_heads: c.num_heads,
                head_dim: c.head_dim(),
                qkv,
                layernorm: att.norm,
                mha,
                add: Add::new(),
                ffn,
            })
        }

        /// currently hardcoded for f32 - need to implement layernorm and softmax in quantized world to be generic over N
        pub fn forward(
            &mut self,
            input: &Tensor<f32>,
            gpt2_output: Option<&GPT2LayerOutput>,
        ) -> anyhow::Result<Tensor<f32>> {
            ensure!(input.get_shape().len() == 2);

            let normed = self
                .layernorm
                .evaluate::<GoldilocksExt2>(&vec![input], vec![])?;

            if let Some(gpt2_output) = gpt2_output {
                ensure!(gpt2_output.is_layernorm_close(normed.outputs()));
            }
            let qkv = self
                .qkv
                .evaluate::<GoldilocksExt2>(&normed.outputs(), vec![])?;

            if let Some(gpt2_output) = gpt2_output {
                ensure!(gpt2_output.is_qkv_close(qkv.outputs()));
            }
            let (mha, _, softmax_out, _) = self
                .mha
                .evaluate_with_intermediate_outputs::<GoldilocksExt2>(&qkv.outputs(), vec![])?;

            if let Some(gpt2_output) = gpt2_output {
                assert!(
                    gpt2_output.is_attention_weights_close(&softmax_out.outputs()[0]),
                    "attention_weights_close given {:?} vs computed {:?}",
                    gpt2_output.attn_weights,
                    softmax_out.outputs()[0].get_data()
                );
            }

            if let Some(gpt2_output) = gpt2_output {
                ensure!(
                    gpt2_output.is_attention_mha_output_close(mha.outputs()),
                    "attention_mha_outputcomputed: {:?} vs  expected {:?}",
                    mha.outputs(),
                    gpt2_output.attn_output
                );
            }
            // now we do the final projection - still [seq_len,hidden_size]
            let projected = self
                .out
                .evaluate::<GoldilocksExt2>(&mha.outputs(), vec![])?;
            if let Some(gpt2_output) = gpt2_output {
                ensure!(gpt2_output.is_attention_output_proj_close(projected.outputs()));
            }

            // and then residual connection, [1, hidden_size]
            let out = self.add.evaluate::<GoldilocksExt2>(
                &vec![input, &projected.outputs()[0]],
                vec![input.get_shape(), projected.outputs()[0].get_shape()],
            )?;

            if let Some(gpt2_output) = gpt2_output {
                ensure!(gpt2_output.is_residual_attn_close(out.outputs()));
            }
            // and then FFN
            let ffn_out = self.ffn.evaluate(&out.outputs()[0], gpt2_output)?;
            Ok(ffn_out)
        }
    }

    impl<N: Number> FlatAttention<N> {
        pub fn random(emb_size: usize, num_heads: usize) -> anyhow::Result<Self> {
            // Note in LLM, it's always the case that hidden_size = emb_size so we can apply residual
            let hidden_size = emb_size;
            let head_size = hidden_size / num_heads;
            let context_length = 1024;
            let qkv = qkv::QKV::random(num_heads, emb_size, hidden_size)?;
            let mha = mha::Mha::new(context_length, num_heads, head_size)?;
            let layernorm = layernorm::LayerNorm::random(emb_size);
            // let out = Dense::random(vec![hidden_size, hidden_size]);
            let out = {
                let weight = OperandMatrix::new_weight_matrix(Tensor::random(
                    &vec![hidden_size, hidden_size].into(),
                ));
                let left = OperandMatrix::Input;
                MatMul::new(left, weight).expect("failed to create MatMul")
            };

            let ffn = FlatFFN::random(hidden_size, hidden_size);
            Ok(Self {
                out,
                hidden_size,
                num_heads,
                head_dim: head_size,
                qkv,
                layernorm,
                mha,
                add: Add::new(),
                ffn,
            })
        }
    }

    #[test]
    fn test_flat_attention_random() {
        let emb_size = 10;
        let num_heads = 2;
        let mut att = FlatAttention::random(emb_size, num_heads).unwrap();
        let input = Tensor::<f32>::random(&vec![1, emb_size].into());
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
        for seq_len in [1, 10] {
            let input = Tensor::<f32>::random(&vec![seq_len, config.embedding_size].into());
            let mut att = FlatAttention::new_from_parser(&config, model.blocks.remove(0)).unwrap();
            let flat_output = att.forward(&input, None).unwrap();
            println!("output shape: {:?}", flat_output.get_shape());
        }
        Ok(())
    }

    #[derive(Debug, Deserialize)]
    pub(crate) struct GPT2Output {
        #[allow(dead_code)]
        token: String,
        pub(crate) input_ids: Vec<u32>,
        // flattened input embeddings
        pub(crate) inputs_embeds: Vec<f32>,
        layers: Vec<GPT2LayerOutput>,
        // output of final projection before logits selection
        logits: Vec<f32>,
        // the new token generated for this input
        next_token_id: u32,
    }

    impl GPT2Output {
        pub fn final_output(&self) -> Vec<f32> {
            // self.logits.clone()
            vec![self.next_token_id as f32]
            // self.layers
            //    .last()
            //    .unwrap()
            //    .manual_output_with_final_ln
            //    .as_ref(
            //    .unwrap()
        }
    }

    #[derive(Debug, Deserialize)]
    struct GPT2LayerOutput {
        ln1_out: Vec<f32>,
        ln2_out: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        attn_output: Vec<f32>,
        attn_weights: Vec<f32>,
        attn_output_proj: Vec<f32>,
        residual_attn: Vec<f32>,
        ffn_up: Vec<f32>,
        manual_output: Vec<f32>,
        // Optional field for the final layer with LayerNorm applied
        #[allow(dead_code)]
        manual_output_with_final_ln: Option<Vec<f32>>,
        ffn_after_gelu: Vec<f32>,
        ffn_after_down: Vec<f32>,
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

        pub fn is_attention_weights_close(&self, attention_weights: &Tensor<f32>) -> bool {
            is_close(attention_weights.get_data(), &self.attn_weights)
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
        pub fn is_ffn_after_gelu_close(&self, output: Vec<&Tensor<f32>>) -> bool {
            is_close(output[0].get_data(), &self.ffn_after_gelu)
        }
        pub fn is_ffn_after_down_close(&self, output: Vec<&Tensor<f32>>) -> bool {
            is_close(output[0].get_data(), &self.ffn_after_down)
        }
    }

    use crate::parser::json;
    #[test]
    fn test_read_gpt2_pytorch_embeddings() -> anyhow::Result<()> {
        let model_weights_path = json::test::get_json_file(TINY_GPT2_NAME)?;
        let debug_output_path = json::test::get_json_file(TINY_GPT2_DEBUG_NAME)?;
        let loader = json::FileTensorLoader::new_from_path(model_weights_path)?;
        let config = LLMConfig::from_json(&loader)?;
        let LLMModel::GPT2(llm_model) = config.model_json(&loader)?;
        let gpt2_output = serde_json::from_reader::<_, GPT2Output>(
            File::open(debug_output_path.clone())
                .context(format!("failed to open file {}", debug_output_path.clone()))?,
        )?;
        let input = Tensor::new(
            vec![gpt2_output.input_ids.len()].into(),
            gpt2_output.input_ids.iter().map(|x| *x as f32).collect(),
        );
        let embedded = llm_model
            .embeddings
            .evaluate::<GoldilocksExt2>(&vec![&input], vec![])?;
        let positioned = llm_model
            .positional
            .evaluate::<GoldilocksExt2>(&vec![embedded.outputs()[0]], vec![])?;
        assert!(is_close(
            &positioned.outputs()[0].get_data(),
            &gpt2_output.inputs_embeds
        ));
        Ok(())
    }

    /// Compares the flat implementation vs the graph implementation for the first layer
    #[test]
    fn test_read_gpt2_pytorch_output_first() -> anyhow::Result<()> {
        let model_weights_path = json::test::get_json_file(TINY_GPT2_NAME)?;
        let debug_output_path = json::test::get_json_file(TINY_GPT2_DEBUG_NAME)?;

        let loader = json::FileTensorLoader::new_from_path(model_weights_path)?;
        let config = LLMConfig::from_json(&loader)?;
        println!("config: {:?}", config);

        let gpt2_output = serde_json::from_reader::<_, GPT2Output>(
            File::open(debug_output_path.clone())
                .context(format!("failed to open file {}", debug_output_path.clone()))?,
        )?;
        let input = Tensor::new(
            vec![gpt2_output.input_ids.len(), config.embedding_size].into(),
            gpt2_output.inputs_embeds.clone(),
        );
        let LLMModel::GPT2(mut model) = config.model_json(&loader)?;
        println!("model: {:?}", config.specific_config);
        // Try to run with the flat attention implementation
        let first_attention = model.blocks.remove(0);
        let mut att = FlatAttention::new_from_parser(&config, first_attention.clone()).unwrap();
        let first_layer_output = gpt2_output.layers.get(0).expect("no layers in output");
        let output = att.forward(&input, Some(first_layer_output))?;
        println!("flat output: {:?}", output.get_shape());
        let expected_output = &first_layer_output.manual_output;
        assert!(is_close(expected_output, &output.get_data()));
        // Now try to run with the graph implementation
        let mut model =
            Model::new_from_input_shapes(vec![input.get_shape()], PaddingMode::NoPadding);
        let _last_node_id = first_attention.write_to_model(&mut model, None, &config)?;
        model.route_output(None)?;
        let output1 = model.run::<GoldilocksExt2>(&[input.clone()])?;
        let output = model.run_float(&[input.clone()])?;
        assert_eq!(output1.outputs()?[0].get_data(), output[0].get_data());
        println!("graph output: {:?}", output[0].get_shape());
        assert!(
            is_close(expected_output, &output[0].get_data()),
            "graph output differs"
        );
        Ok(())
    }

    /// Compares the graph implementation vs the output of the pytorch model over
    /// all passes including to the final logits selection.
    #[test]
    fn test_gpt2_model_full_pass() -> anyhow::Result<()> {
        init_test_logging("INFO");
        let model_weights_path = json::test::get_json_file(TINY_GPT2_NAME)?;
        let debug_output_path = json::test::get_json_file(TINY_GPT2_DEBUG_NAME)?;
        // too big to run in JSON mode - need to switch format
        // let model_weights_path = json::test::get_json_file(DISTIL_GPT2_NAME)?;
        // let debug_output_path = json::test::get_json_file(DISTIL_GPT2_DEBUG_NAME)?;
        let loader = json::FileTensorLoader::new_from_path(model_weights_path)?;
        let config = LLMConfig::from_json(&loader)?;
        let LLMModel::GPT2(llm_model) = config.model_json(&loader)?;
        let gpt2_output = serde_json::from_reader::<_, GPT2Output>(
            File::open(debug_output_path.clone())
                .context(format!("failed to open file {}", debug_output_path.clone()))?,
        )?;
        let expected_output = &&gpt2_output.final_output();
        let input = Tensor::new(
            // setup 1 as last dimension since embeddings iterate over last dimension
            // or call unsqueeze
            vec![gpt2_output.input_ids.len()].into(),
            gpt2_output.input_ids.iter().map(|x| *x as f32).collect(),
        );
        // also test on a single random token
        let max_token = thread_rng().gen_range(0..llm_model.embeddings.vocab_size);
        let single_input = Tensor::new(vec![1].into(), vec![max_token as f32]);
        let model = llm_model
            .clone()
            .into_provable_model(&config, Shape::from(single_input.get_shape()))?;
        model.describe();
        model.run_float(&[single_input.clone()])?;

        let model = llm_model.into_provable_model(&config, Shape::from(input.get_shape()))?;
        model.describe();
        let output = model.run_float(&[input.clone()])?[0].clone();
        // since the expected output is only for one token, but our model generates logits for all tokens,
        // we take the last element of the model output
        let output = output.slice_last_dim().last().unwrap();
        assert!(
            is_close(expected_output, &output),
            "graph output differs: {:?} vs {:?}: LOGITS {:?}",
            expected_output,
            output,
            &gpt2_output.logits[0..5]
        );
        Ok(())
    }
}
