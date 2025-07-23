use std::{collections::HashMap, path::Path};

use anyhow::{Context, bail, ensure};
use serde::Deserialize;
use tracing::trace;

use crate::{
    Tensor,
    layers::{
        matrix_mul::MatMul,
        transformer::{embeddings::Embeddings, layernorm::LayerNorm, positional::Positional},
    },
    parser::llm::{Attention, FeedForward, GPT2Model, LLMConfig, LLMModel, LLMVariant},
    tensor::Shape,
};

impl LLMConfig {
    pub fn from_json(l: &FileTensorLoader) -> anyhow::Result<Self> {
        let variant = LLMVariant::from_json(l)?;
        let hidden_size = l.metadata_to_u32("hidden_dim")? as usize;
        let embedding_size = hidden_size;
        let num_heads = l.metadata_to_u32("num_attention_heads")? as usize;
        let num_blocks = l.metadata_to_u32("num_hidden_layers")? as usize;
        let context_length = l.metadata_to_u32("max_seq_len")? as usize;
        let norm_epsilon = l.metadata_to_f32("norm_epsilon")?;
        Ok(Self {
            embedding_size,
            hidden_size,
            num_heads,
            num_block: num_blocks,
            context_length,
            norm_epsilon,
            specific_config: variant,
        })
    }
}

impl LLMVariant {
    pub fn model_json(&self, l: &FileTensorLoader, config: &LLMConfig) -> anyhow::Result<LLMModel> {
        match self {
            Self::GPT2 => Ok(LLMModel::GPT2(GPT2Model::from_json(l, config)?)),
        }
    }
    pub fn from_json(l: &FileTensorLoader) -> anyhow::Result<Self> {
        let variant_value = l
            .get_metadata("model_name")
            .ok_or_else(|| anyhow::anyhow!("Metadata key 'model_name' not found"))?;

        let model_name_str = variant_value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Metadata 'model_name' is not a string value"))?;

        match model_name_str
            .trim() // Keep the trim, it's good practice
        {
            "gpt2" => Ok(Self::GPT2),
            "sshleifer/tiny-gpt2" => Ok(Self::GPT2),
            a => bail!("unsupported architecture: {:?}", a),
        }
    }
}
impl GPT2Model {
    pub fn from_json(l: &FileTensorLoader, config: &LLMConfig) -> anyhow::Result<Self> {
        let embeddings = Embeddings::from_json(l)?;
        let positional = Positional::from_json(l, config)?;
        let num_layers = config.num_block;
        let blocks = (0..num_layers)
            .map(|i| Attention::from_json(&l.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<Attention<f32>>>>()?;
        let final_norm = LayerNorm::from_json(&l.pp("output_"), config)?;
        let proj_weights = l.get_tensor("output.weight")?.transpose();
        let proj_bias = l.get_tensor("output.bias").ok();
        let final_proj = MatMul::new_constant(proj_weights, proj_bias)?;
        Ok(Self::new(
            embeddings, positional, blocks, final_norm, final_proj,
        ))
    }
}

impl FeedForward<f32> {
    pub fn from_json(l: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let norm = LayerNorm::from_json(&l.pp("ffn_"), c)?;
        let up = l.get_tensor("ffn_up.weight")?;
        let up_bias = l.get_tensor("ffn_up.bias")?;
        let down = l.get_tensor("ffn_down.weight")?;
        let down_bias = l.get_tensor("ffn_down.bias")?;
        ensure!(
            up.get_shape()[0] == c.hidden_size,
            "up have shape {:?} but in features should be equal to hidden_size: {}",
            up.get_shape(),
            c.hidden_size
        );
        ensure!(
            down.get_shape()[1] == c.embedding_size,
            "down have shape {:?} but out features should be equal to embedding_size: {}",
            down.get_shape(),
            c.embedding_size
        );
        Ok(Self {
            norm,
            up,
            up_bias,
            down,
            down_bias,
        })
    }
}

impl Attention<f32> {
    pub fn from_json(l: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let norm = LayerNorm::from_json(&l.pp("attn_"), c)
            .context("Failed to load LayerNorm for attention in from_json")?;

        let fused_qkv_weight = l
            .get_tensor("attn_qkv.weight")
            .context("Failed to load attn_qkv.weight in from_json")?;
        let fused_qkv_bias = l
            .get_tensor("attn_qkv.bias")
            .context("Failed to load attn_qkv.bias in from_json")?;

        let hidden_size = c.hidden_size; // embedding_dim for GPT-2

        // Unfuse weights:
        // Expected shape of fused_qkv_weight is [3 * hidden_size, hidden_size] after python script transpose.
        // Each individual q, k, v weight matrix should be [hidden_size, hidden_size].
        // So, each chunk has hidden_size * hidden_size elements.
        let weight_chunk_elements = hidden_size * hidden_size;
        let mut unfused_weights_data =
            unfuse_crate_tensors(fused_qkv_weight.clone(), weight_chunk_elements, 3)
                .context("Failed to unfuse QKV weights in from_json")?;

        let q_weight = Tensor::new(
            vec![c.embedding_size, hidden_size].into(),
            unfused_weights_data.remove(0),
        );
        let k_weight = Tensor::new(
            vec![c.embedding_size, hidden_size].into(),
            unfused_weights_data.remove(0),
        );
        let v_weight = Tensor::new(
            vec![c.embedding_size, hidden_size].into(),
            unfused_weights_data.remove(0),
        );
        trace!("fused qkv: {fused_qkv_weight:?}");
        trace!("qkv full tensor {unfused_weights_data:?}");
        trace!("q_weight {:?}", q_weight.get_data());

        // Unfuse biases:
        // Expected shape of fused_qkv_bias is [3 * hidden_size].
        // Each individual q, k, v bias vector should be [hidden_size].
        // So, each chunk has hidden_size elements.
        let bias_chunk_elements = hidden_size;
        let mut unfused_biases_data = unfuse_crate_tensors(fused_qkv_bias, bias_chunk_elements, 3)
            .context("Failed to unfuse QKV biases in from_json")?;

        let q_bias_vec = Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0));
        let k_bias_vec = Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0));
        let v_bias_vec = Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0));

        // These are the individual Q, K, V matrices and biases now.
        // The QKV struct or logic that consumes these will handle them.
        // For now, let's assume Attention struct stores these directly if QKV is not used here.
        // Or, construct the QKV layer if that's the design.
        // The original struct for Attention<f32> directly stores q, q_bias, k, k_bias, v, v_bias.

        let out = l
            .get_tensor("attn_output.weight")
            .context("Failed to load attn_output.weight in from_json")?;
        let out_bias = l
            .get_tensor("attn_output.bias")
            .context("Failed to load attn_output.bias in from_json")?;

        // Shape check for attn_output.weight: [hidden_size, hidden_size] for GPT-2
        // Python script exports it as [out_features, in_features]
        // For c_proj (attn_output), out_features = hidden_size, in_features = hidden_size
        ensure!(
            out.get_shape().as_ref() == &[hidden_size, hidden_size],
            "Attention output weight tensor shape mismatch in from_json. Expected [{}, {}], got {:?}",
            hidden_size,
            hidden_size,
            out.get_shape()
        );
        ensure!(
            out_bias.get_shape().as_ref() == &[hidden_size],
            "Attention output bias tensor shape mismatch in from_json. Expected [{}], got {:?}",
            hidden_size,
            out_bias.get_shape()
        );

        let feedforward =
            FeedForward::from_json(l, c).context("Failed to load FeedForward in from_json")?;

        Ok(Self {
            norm,
            q: q_weight,
            q_bias: q_bias_vec,
            k: k_weight,
            k_bias: k_bias_vec,
            v: v_weight,
            v_bias: v_bias_vec,
            out,
            out_bias,
            feedforward,
        })
    }
}

/// Generic helper function to unfuse a tensor's data into multiple chunks.
/// Expects the input tensor `fused_tensor` (crate::Tensor<f32>) to contain flat data.
fn unfuse_crate_tensors(
    fused_tensor: Tensor<f32>,
    expected_chunk_len_elements: usize,
    num_chunks: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let data = fused_tensor.get_data();
    let total_elements = data.len();

    ensure!(
        expected_chunk_len_elements > 0,
        "expected_chunk_len_elements must be positive, got {}",
        expected_chunk_len_elements
    );
    ensure!(
        num_chunks > 0,
        "num_chunks must be positive, got {}",
        num_chunks
    );

    let expected_total_elements = expected_chunk_len_elements * num_chunks;
    ensure!(
        total_elements == expected_total_elements,
        "Tensor data size ({}) does not match expected total size ({} chunks * {} elements_per_chunk = {}). Original tensor shape: {:?}",
        total_elements,
        num_chunks,
        expected_chunk_len_elements,
        expected_total_elements,
        fused_tensor.get_shape()
    );

    let tensors_data: Vec<Vec<f32>> = data
        .chunks_exact(expected_chunk_len_elements)
        .map(|chunk| chunk.to_vec())
        .collect();

    ensure!(
        tensors_data.len() == num_chunks,
        "Unfused into {} tensors, expected {}",
        tensors_data.len(),
        num_chunks
    );

    Ok(tensors_data)
}

impl Positional<f32> {
    pub fn from_json(l: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let position_embd = l.get_tensor("position_embd.weight")?;
        ensure!(
            position_embd.get_shape().len() == 2,
            "position_embd must be 2d"
        );
        ensure!(
            position_embd.get_shape()[0] == c.context_length,
            "position_embd must have shape [0] [{}] vs given {:?}",
            c.context_length,
            position_embd.get_shape()
        );
        ensure!(
            position_embd.get_shape()[1] == c.embedding_size,
            "position_embd must have shape [1] [{}] vs given {:?}",
            c.embedding_size,
            position_embd.get_shape()
        );
        Ok(Self::new_learned(position_embd))
    }
}

impl Embeddings<f32> {
    pub fn from_json(l: &FileTensorLoader) -> anyhow::Result<Self> {
        let emb_tensor = l.get_tensor("token_embd.weight")?;
        Embeddings::new(emb_tensor)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonTensor {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl JsonTensor {
    pub fn as_tensor(&self) -> Tensor<f32> {
        Tensor::new(self.shape.clone(), self.data.clone())
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct JsonModel {
    pub metadata: HashMap<String, serde_json::Value>,
    pub tensors: HashMap<String, JsonTensor>,
}

#[derive(Clone, Debug)]
pub struct FileTensorLoader {
    pub content: JsonModel,
    pub prefix: String, // current path scope (e.g., "blk.00.")
}

impl FileTensorLoader {
    pub fn new_from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path.as_ref()).with_context(|| {
            format!("Failed to open JSON file at: {:?}", path.as_ref().display())
        })?;
        let content: JsonModel = serde_json::from_reader(file).with_context(|| {
            format!(
                "Failed to parse JSON from file at: {:?}",
                path.as_ref().display()
            )
        })?;
        Ok(Self {
            content,
            prefix: "".to_string(),
        })
    }

    pub fn pp(&self, sub: &str) -> Self {
        let mut new = self.clone();
        new.prefix = format!("{}{}", self.prefix, sub);
        new
    }

    fn resolve_key(&self, key: &str) -> Option<&JsonTensor> {
        let full_key = format!("{}{}", self.prefix, key);
        self.content.tensors.get(&full_key)
    }

    pub fn get_tensor(&self, key: &str) -> anyhow::Result<Tensor<f32>> {
        let tensor = self
            .resolve_key(key)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {key}"))?;
        Ok(Tensor::new(tensor.shape.clone(), tensor.data.clone()))
    }

    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.content.metadata.get(key)
    }

    pub fn metadata_to_u32(&self, key: &str) -> anyhow::Result<u32> {
        Ok(self
            .get_metadata(key)
            .ok_or_else(|| anyhow::anyhow!("missing metadata {key}"))?
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("metadata {key} not a u32"))? as u32)
    }

    pub fn metadata_to_f32(&self, key: &str) -> anyhow::Result<f32> {
        Ok(self
            .get_metadata(key)
            .ok_or_else(|| anyhow::anyhow!("missing metadata {key}"))?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("metadata {key} not a f32"))? as f32)
    }
}

#[cfg(test)]
pub mod test {
    use crate::parser::llm::LLMConfig;
    use std::{env, path::PathBuf};

    use super::*;

    pub const TINY_GPT2_NAME: &str = "tiny_gpt2_weights.json";
    pub const TINY_GPT2_DEBUG_NAME: &str = "tiny_gpt2_debug_output.json";
    #[allow(dead_code)]
    pub const DISTIL_GPT2_NAME: &str = "distilgpt2_weights.json";
    #[allow(dead_code)]
    pub const DISTIL_GPT2_DEBUG_NAME: &str = "distilgpt2_debug_output.json";

    pub fn get_json_file(name: &str) -> anyhow::Result<String> {
        let path = if env::var("DEEPPROVE_CI").unwrap_or_default() == "true" {
            let ci_asset_dir = env::var("DEEPPROVE_ASSET_DIR")
                .context("DEEPPROVE_ASSET_DIR not set in CI environment")?;
            PathBuf::from(ci_asset_dir).join(name)
        } else {
            PathBuf::from("assets/scripts/llms/").join(name)
        };
        assert!(
            path.exists(),
            "Missing model locally, create a venv and run `python3 gpt2_internal.py --output-dir ./assets/scripts/llms/ --export-model` to retrieve it",
        );
        Ok(path.to_str().unwrap().to_string())
    }

    #[test]
    fn test_json_tensor_loader() -> anyhow::Result<()> {
        let path = get_json_file(TINY_GPT2_NAME)?;
        let loader = FileTensorLoader::new_from_path(path)?;
        let config = LLMConfig::from_json(&loader)?;
        println!("tiny gpt2 config: {:?}", config);
        config.model_json(&loader)?;
        Ok(())
    }
}
