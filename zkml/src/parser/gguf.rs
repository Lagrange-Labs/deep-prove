use candle_core::quantized::{
    QTensor,
    gguf_file::{Value, ValueType},
};
use candle_transformers::{models::deepseek2::SplitOp, quantized_var_builder::VarBuilder};
use std::{
    any::TypeId,
    fs::File,
    io::{BufReader, Read, Seek},
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{Context, bail, ensure};
use candle_core::{
    CpuStorage, DType, Device, Storage,
    quantized::{GgmlDType, gguf_file::Content},
};

use crate::{
    Tensor,
    layers::transformer::{embeddings::Embeddings, layernorm::LayerNorm, positional::Positional},
    tensor::Number,
};

fn parse_gguf(path: &str) -> anyhow::Result<()> {
    Ok(())
}

// 1. LayerNorm (Pre-Norm)
// Input: [seq_len, hidden_size] (e.g. [N, 768])
// Operation: Normalize across hidden_size (i.e., last axis)
//
// Parameters:
// blk.N.attn_norm.weight (scale)
// blk.N.attn_norm.bias (shift)
//
// 2. Fused QKV Projection
// Operation: Linear layer projecting input into Q, K, V vectors in one go.
// Matmul: x @ W_qkv.T + b_qkv
// Shapes:
// x: [N, 768]
// W_qkv: [3 * hidden_size, 768] → [2304, 768]
// b_qkv: [2304]
//
// Tensors:
// blk.N.attn_qkv.weight
// blk.N.attn_qkv.bias
//
// After matmul: Output [N, 2304] → split into:
// q: [N, 768]
// k: [N, 768]
// v: [N, 768]
//
// 3. Attention Scores
// Matmul: attn_scores = q @ k.T / sqrt(d_k)
// Shape:
// q: [N, 768] reshaped to [N, n_heads, head_dim]
// k: [N, 768] reshaped to [N, n_heads, head_dim]
//
// Result: [N, n_heads, N]
//
// Optionally add attention mask (causal or padding mask)
//
// 4. Softmax + Dropout
// Apply softmax over the last axis (tokens)
// Optional dropout during training
//
// 5. Attention Output
// Weighted sum: attn_output = softmax_scores @ v
// Shape: [N, n_heads, head_dim] → merge heads → [N, 768]
//
// 6. Output Projection
// Linear projection back to hidden size
// Matmul: attn_output @ W_o.T + b_o
// Shapes:
// W_o: [768, 768]
// b_o: [768]
// Tensors:
// blk.N.attn_output.weight
// blk.N.attn_output.bias
//
// 7. Residual Add
// Output: x + attn_output → shape [N, 768]

#[derive(Debug, Clone)]
pub struct LLMConfig {
    /// The size of an embedding vector (each token gets translated to an embedding vector of this size)
    pub embedding_size: usize,
    /// Size of the attention layer matrices.
    pub hidden_size: usize,
    /// The number of "heads" that are used within each attention layer.
    pub num_heads: usize,
    /// The number of blocks / attention layers there is in the model
    pub num_block: usize,
    /// The maximum size that the tensor containing input + generated token can have. Beyond that, we should not
    /// run the tensor through the model anymore.
    pub context_length: usize,
    /// LayerNorm needs an epsilon value to determine the precision. This is it.
    pub norm_epsilon: f32,
    /// The specific config for the variant.
    pub specific_config: LLMVariant,
}

impl LLMConfig {
    pub fn from_content(l: &FileTensorLoader) -> anyhow::Result<Self> {
        let variant = LLMVariant::from_content(l)?;
        let embedding_size = l.content.metadata[variant.embedding_size_key()].to_u32()? as usize;
        let hidden_size = l.content.metadata[variant.hidden_size_key()].to_u32()? as usize;
        let num_heads = l.content.metadata[variant.num_heads_key()].to_u32()? as usize;
        let context_length = l.content.metadata[variant.context_length_key()].to_u32()? as usize;
        let norm_epsilon = l.content.metadata[variant.norm_epsilon_key()].to_f32()?;
        let num_block = l.content.metadata[variant.num_block_key()].to_u32()? as usize;
        Ok(Self {
            hidden_size,
            embedding_size,
            num_heads,
            context_length,
            norm_epsilon,
            num_block,
            specific_config: variant,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn model(&self, l: &FileTensorLoader) -> anyhow::Result<LLMModel> {
        self.specific_config.model(l, &self)
    }
}

#[derive(Debug, Clone)]
pub enum LLMVariant {
    GPT2,
}

impl LLMVariant {
    pub fn from_content(l: &FileTensorLoader) -> anyhow::Result<Self> {
        let Some(variant) = l
            .content
            .metadata
            .get("general.name")
            .or_else(|| l.content.metadata.get("general.architecture"))
        else {
            bail!("no variant found");
        };
        match variant
            .to_string()
            .context("unable to get variant into string")?
            .as_str()
        {
            "gpt2" => Ok(Self::GPT2),
            a => bail!("unsupported architecture: {:?}", a),
        }
    }

    pub fn num_heads_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.attention.head_count",
        }
    }

    pub fn context_length_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.context_length",
        }
    }
    pub fn num_block_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.block_count",
        }
    }
    pub fn embedding_size_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.embedding_length",
        }
    }
    pub fn hidden_size_key(&self) -> &str {
        match self {
            // same size as embedding for gpt2
            Self::GPT2 => self.embedding_size_key(),
        }
    }
    pub fn norm_epsilon_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.attention.layer_norm_epsilon",
        }
    }
    pub fn model(&self, l: &FileTensorLoader, config: &LLMConfig) -> anyhow::Result<LLMModel> {
        match self {
            Self::GPT2 => Ok(LLMModel::GPT2(GPT2Model::from_loader(l, config)?)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LLMModel {
    GPT2(GPT2Model),
}

#[derive(Debug, Clone)]
pub struct GPT2Model {
    embeddings: Embeddings<f32>,
    positional: Positional<f32>,
    pub blocks: Vec<Attention<f32>>,
}

impl GPT2Model {
    pub fn from_loader(loader: &FileTensorLoader, config: &LLMConfig) -> anyhow::Result<Self> {
        let embeddings = Embeddings::from_loader(loader)?;
        let positional = Positional::from_loader(loader, config)?;
        let num_layers = config.num_block;
        let blocks = (0..num_layers)
            .map(|i| Attention::from_loader(&loader.pp(&format!("blk.{i}.")), &config))
            .collect::<anyhow::Result<Vec<Attention<f32>>>>()?;
        Ok(Self {
            embeddings,
            positional,
            blocks,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FeedForward<N: Number> {
    pub norm: LayerNorm<N>,
    pub up: Tensor<N>,
    pub up_bias: Tensor<N>,
    pub down: Tensor<N>,
    pub down_bias: Tensor<N>,
}

impl FeedForward<f32> {
    // Replaces from_var_builder and from_tensor_loader
    // 'loader' is expected to be the block-level loader (e.g., scoped to "blk.N.")
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        // Create a sub-scope for the feed-forward network's LayerNorm
        let ffn_norm_loader = loader.pp("ffn_");
        // Use the new LayerNorm::from_loader
        let norm = LayerNorm::from_loader(&ffn_norm_loader, &c)?;

        let up = loader.get_tensor("ffn_up.weight")?;
        let up_bias = loader.get_tensor("ffn_up.bias")?;
        let down = loader.get_tensor("ffn_down.weight")?;
        let down_bias = loader.get_tensor("ffn_down.bias")?;

        ensure!(
            down.get_shape()[0] == c.embedding_size,
            "down must have shape {:?} vs embedding_size: {}",
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

#[derive(Debug, Clone)]
pub struct Attention<N: Number> {
    pub q: Tensor<N>,
    pub q_bias: Tensor<N>,
    pub k: Tensor<N>,
    pub k_bias: Tensor<N>,
    pub v: Tensor<N>,
    pub v_bias: Tensor<N>,
    pub out: Tensor<N>,
    pub out_bias: Tensor<N>,
    pub norm: LayerNorm<N>,
    pub feedforward: FeedForward<N>,
}

impl Attention<f32> {
    // Replaces from_var_builder and from_tensor_loader
    // 'loader' is expected to be the block-level loader (e.g., scoped to "blk.N.")
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let embedding_size = c.embedding_size;
        let hidden_size = c.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );

        let qkv_weight_qtensor = loader.get_qtensor("attn_qkv.weight")?;
        let qkv_weight_candle = qkv_weight_qtensor.dequantize(&Device::Cpu)?;
        let mut unfused_weights =
            unfuse_tensors(qkv_weight_candle, embedding_size * embedding_size)?;
        ensure!(unfused_weights.len() == 3, "qkv_weight must have 3 chunks");
        let q = crate::Tensor::new(vec![embedding_size, hidden_size], unfused_weights.remove(0));
        let k = crate::Tensor::new(vec![embedding_size, hidden_size], unfused_weights.remove(0));
        let v = crate::Tensor::new(vec![embedding_size, hidden_size], unfused_weights.remove(0));

        let qkv_bias_qtensor = loader.get_qtensor("attn_qkv.bias")?;
        let qkv_bias_candle = qkv_bias_qtensor.dequantize(&Device::Cpu)?;
        let mut unfused_biases = unfuse_tensors(qkv_bias_candle, embedding_size)?;
        ensure!(unfused_biases.len() == 3, "qkv_bias must have 3 chunks");
        let q_bias = crate::Tensor::new(vec![hidden_size], unfused_biases.remove(0));
        let k_bias = crate::Tensor::new(vec![hidden_size], unfused_biases.remove(0));
        let v_bias = crate::Tensor::new(vec![hidden_size], unfused_biases.remove(0));

        let attn_norm_loader = loader.pp("attn_");
        // Use new LayerNorm::from_loader
        let norm = LayerNorm::from_loader(&attn_norm_loader, &c)?;

        let out = loader.get_tensor("attn_output.weight")?;
        let out_bias = loader.get_tensor("attn_output.bias")?;
        ensure!(
            out.get_shape().as_slice() == &[embedding_size, embedding_size],
            "out must have shape [hidden_size, hidden_size]"
        );
        ensure!(
            out_bias.get_shape().as_slice() == &[embedding_size],
            "out_bias must have shape [hidden_size]"
        );

        // Use new FeedForward::from_loader
        let ff = FeedForward::from_loader(loader, &c)?;

        Ok(Self {
            out,
            out_bias,
            norm,
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            feedforward: ff,
        })
    }
}

fn dequantize(qtensor: Arc<QTensor>) -> anyhow::Result<Tensor<f32>> {
    let shape = qtensor.shape().dims().to_vec();
    let tensor = match qtensor.dtype() {
        // it's a no op for f32 or f16 to dequantize
        GgmlDType::Q8_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0 | GgmlDType::F16 | GgmlDType::F32 => {
            qtensor.dequantize(&Device::Cpu)?
        }
        _ => {
            bail!("unsupported dtype");
        }
    };
    let (s, l) = tensor.storage_and_layout();
    let data = match s.deref() {
        Storage::Cpu(cpu) => match cpu {
            CpuStorage::F32(d) => d.to_vec(),
            CpuStorage::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
            _ => bail!("unsupported storage type (only f32 or f16 is supported)"),
        },
        _ => bail!("unsupported storage backend (only cpu is supported)"),
    };
    Ok(Tensor::new(shape, data))
}

fn unfuse_tensors(fused: candle_core::Tensor, chunk_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let (s, l) = fused.storage_and_layout();
    // let shape = l.shape().dims().to_vec();
    let data = match s.deref() {
        Storage::Cpu(cpu) => match cpu {
            CpuStorage::F32(d) => d.to_vec(),
            CpuStorage::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
            _ => bail!("unsupported storage type (only f32 or f16 is supported)"),
        },
        _ => bail!("unsupported storage backend (only cpu is supported)"),
    };
    let tensors: Vec<Vec<f32>> = data.chunks(chunk_len).map(|chunk| chunk.to_vec()).collect();
    ensure!(
        tensors.iter().all(|t| t.len() == chunk_len),
        "all chunks must have the same length"
    );
    Ok(tensors)
}

trait FromValue<T> {
    fn from_value(v: &Value) -> T;
}

impl FromValue<f32> for Value {
    fn from_value(v: &Value) -> f32 {
        v.to_f32().expect("failed to convert f32 to f32")
    }
}

impl FromValue<f64> for Value {
    fn from_value(v: &Value) -> f64 {
        v.to_f64().expect("failed to convert f64 to f64")
    }
}
impl FromValue<usize> for Value {
    fn from_value(v: &Value) -> usize {
        v.to_u32().expect("failed to convert u32 to u32") as usize
    }
}

/// Type alias for a TensorLoader specialized for reading from a BufReader<File>.
/// This simplifies the instantiation when loading tensors directly from a file path.
pub type FileTensorLoader = TensorLoader<BufReader<File>>;

#[derive(Clone)]
/// Manages lazy loading of tensors from a GGUF file.
///
/// This structure allows for efficient, on-demand loading of tensor data.
/// It supports sub-scoping for tensor names (e.g., `blk.0.attn_norm.weight`)
/// by maintaining an internal prefix. It is designed to be cloneable, making
/// it easy to pass around or use in different parts of a model definition.
/// Tensor loading is deferred until a specific tensor is requested via `get_tensor`.
pub struct TensorLoader<R: Read + Seek> {
    /// Parsed GGUF metadata and tensor information.
    /// This is an `Arc` to allow cheap cloning of the `TensorLoader`.
    content: Arc<Content>,
    /// Reader for the GGUF file, allowing lazy loading of tensor data.
    /// It's wrapped in `Arc<Mutex<>>` to enable shared, mutable access
    /// across cloned instances and for thread-safety if used in concurrent contexts.
    reader: Arc<Mutex<R>>,
    /// Current prefix for tensor names. When a tensor is requested,
    /// this prefix is prepended to the requested name to form the full tensor name.
    current_prefix: String,
    /// The `Device` on which `QTensor`s (quantized tensors) should be initially loaded.
    /// Note: The existing `dequantize` function subsequently converts these to `crate::Tensor<f32>`
    /// and currently materializes them on the CPU.
    device: Device,
}

impl<R: Read + Seek + Send + 'static> TensorLoader<R> {
    /// Creates a new `TensorLoader` from a given reader and device.
    /// The reader must be positioned at the beginning of the GGUF file.
    ///
    /// # Arguments
    /// * `reader` - A type implementing `Read` and `Seek` for the GGUF file (e.g., `BufReader<File>`).
    ///
    /// # Errors
    /// Returns an error if reading the GGUF content metadata fails.
    pub fn from_reader(mut reader: R) -> anyhow::Result<Self> {
        let content = Content::read(&mut reader)?;
        Ok(Self {
            content: Arc::new(content),
            reader: Arc::new(Mutex::new(reader)),
            current_prefix: String::new(),
            device: Device::Cpu,
        })
    }

    /// Creates a new `TensorLoader` instance representing a sub-scope.
    /// The new scope's prefix is formed by concatenating the current loader's prefix
    /// with the `prefix_extension`. For example, if the current prefix is `blk.0.`
    /// and `prefix_extension` is `attn_`, the new prefix will be `blk.0.attn_`.
    ///
    /// # Arguments
    /// * `prefix_extension` - The string to append to the current prefix to define the new scope.
    ///
    /// # Returns
    /// A new `TensorLoader` instance for the specified sub-scope.
    pub fn pp(&self, prefix_extension: &str) -> Self {
        Self {
            content: Arc::clone(&self.content),
            reader: Arc::clone(&self.reader),
            current_prefix: format!("{}{}", self.current_prefix, prefix_extension),
            device: self.device.clone(),
        }
    }

    /// Retrieves a quantized tensor (`QTensor`) by its name relative to the current scope.
    /// The full tensor name is formed by `current_prefix + name`.
    /// This method is primarily for internal use or advanced scenarios where the `QTensor` is needed directly.
    ///
    /// # Arguments
    /// * `name` - The name of the tensor, relative to the current scope (e.g., `weight`).
    ///
    /// # Errors
    /// Returns an error if the reader lock cannot be acquired or if `Content::tensor` fails to load the `QTensor`.
    pub(crate) fn get_qtensor(&self, name: &str) -> anyhow::Result<Arc<QTensor>> {
        let full_name = format!("{}{}", self.current_prefix, name);
        let mut reader_guard = self.reader.lock().map_err(|e| {
            anyhow::anyhow!(
                "Failed to acquire reader lock for tensor '{}': {}",
                full_name,
                e
            )
        })?;
        self.content
            .tensor(&mut *reader_guard, &full_name, &self.device)
            .map(|qtensor| Arc::new(qtensor))
            .map_err(|e| anyhow::anyhow!("Failed to load QTensor '{}' from GGUF: {}", full_name, e))
    }

    /// Retrieves and dequantizes a tensor by its name relative to the current scope.
    ///
    /// This method first loads the quantized tensor (`QTensor`) using `get_qtensor`,
    /// then calls the `dequantize` function (expected to be available in the same module)
    /// to convert it into a `crate::Tensor<f32>`.
    ///
    /// # Arguments
    /// * `name` - The name of the tensor, relative to the current scope (e.g., `attn_norm.weight`).
    ///
    /// # Errors
    /// Returns an error if `get_qtensor` fails or if the subsequent dequantization fails.
    pub fn get_tensor(&self, name: &str) -> anyhow::Result<crate::Tensor<f32>> {
        let qtensor = self.get_qtensor(name)?;
        dequantize(qtensor)
    }

    pub fn metadata<T>(&self, key: &str) -> T
    where
        Value: FromValue<T>,
    {
        let v = &self.content.metadata[key];
        Value::from_value(v)
    }
}

impl TensorLoader<BufReader<File>> {
    /// Creates a new `TensorLoader` by opening and reading a GGUF file from the specified path.
    ///
    /// # Arguments
    /// * `path` - The file system path to the GGUF file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or if reading the GGUF content metadata fails.
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to open file {:?}: {}", path.as_ref(), e))?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }
}

#[cfg(test)]
pub mod tests {
    use candle_core::{CpuStorage, Device, Storage, Tensor, quantized::gguf_file::Content};
    use candle_transformers::quantized_var_builder::VarBuilder;
    use gguf_rs::get_gguf_container;
    use std::{fs::File, io::Read, ops::Deref, path::Path};

    use crate::{
        layers::transformer::embeddings::Embeddings,
        parser::gguf::{FeedForward, LLMConfig},
    };

    use super::{Attention, TensorLoader};
    // download at https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF
    pub const GPT2_Q8_0_PATH: &str = "assets/scripts/llms/gpt2.q8_0.gguf";

    #[test]
    fn test_gguf_load_model() -> anyhow::Result<()> {
        let loader = FileTensorLoader::from_path(GPT2_Q8_0_PATH)?;
        let config = LLMConfig::from_content(&loader)?;
        let model = config.model(&loader)?;
        println!("model: {:?}", config.specific_config);
        Ok(())
    }

    #[test]
    fn test_gguf_load_attention() -> anyhow::Result<()> {
        let loader = FileTensorLoader::from_path(GPT2_Q8_0_PATH)?;
        let config = LLMConfig::from_content(&loader)?;
        // println!("config: {:?}", config); // Keep original println if desired for debugging
        let block0_loader = loader.pp("blk.0.");

        let _attention = Attention::from_loader(&block0_loader, &config)?;
        Ok(())
    }

    #[test]
    fn test_gguf_load_config() -> anyhow::Result<()> {
        let loader = FileTensorLoader::from_path(GPT2_Q8_0_PATH)?;
        let config = LLMConfig::from_content(&loader)?;
        println!("config: {:?}", config);
        Ok(())
    }

    #[test]
    fn test_gguf_load_embedding() -> anyhow::Result<()> {
        let gguf_path = GPT2_Q8_0_PATH;
        let loader = FileTensorLoader::from_path(gguf_path)?;
        let _embedding = Embeddings::from_loader(&loader)?;
        Ok(())
    }

    // https://docs.rs/candle-transformers/latest/src/candle_transformers/models/llama.rs.html#517-535
    #[test]
    fn test_load_and_inspect_gpt2_gguf() -> anyhow::Result<()> {
        // Path to the GGUF file
        let gguf_path = GPT2_Q8_0_PATH;

        let mut container = get_gguf_container(&gguf_path)?;
        let model = container.decode()?;

        println!("GGUF version: {}", model.get_version());
        println!("GGUF metadata: {:?}", model.metadata());
        let mut r = File::open(gguf_path)?;
        let gguf_candle = Content::read(&mut r)?;
        println!("GGUF metadata: {:?}", gguf_candle.metadata.keys());
        println!("GGUF tensors: {:?}", gguf_candle.tensor_infos);
        for tensor in model.tensors() {
            println!("Tensor name: {}", tensor.name);
            println!("Tensor kind: {}", tensor.kind);
            let num_elements = tensor.shape.iter().product::<u64>();
            println!(
                "Tensor shape: {:?} -> total {:?}",
                tensor.shape, num_elements
            );
            let qtensor = gguf_candle.tensor(&mut r, &tensor.name, &Device::Cpu)?;
            let tensor = qtensor.dequantize(&Device::Cpu)?;
            let (s, l) = tensor.storage_and_layout();
            let data = match s.deref() {
                Storage::Cpu(s) => match s {
                    CpuStorage::F32(d) => d.to_vec(),
                    CpuStorage::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
                    _ => {
                        panic!("unsupported type of tensor: {:?}", s);
                    }
                },
                _ => {
                    panic!("only cpu storage type is supported");
                }
            };
        }
        Ok(())
    }

    use crate::parser::gguf::FileTensorLoader;
    #[test]
    fn test_tensor_loader_subscoping_and_lazy_load() -> anyhow::Result<()> {
        let gguf_path = GPT2_Q8_0_PATH;

        // Create TensorLoader using the type alias
        let loader = FileTensorLoader::from_path(gguf_path)?;

        // Test loading a tensor from the root scope
        let embedding_tensor = loader.get_tensor("token_embd.weight")?;
        // Expected shape for gpt2 token_embd.weight: [vocab_size, embedding_length] = [50257, 768]
        assert_eq!(
            embedding_tensor.get_shape(),
            vec![50257usize, 768usize],
            "Shape mismatch for token_embd.weight"
        );

        // Test sub-scoping with a trailing dot (VarBuilder style)
        let blk0_loader = loader.pp("blk.0.");
        let attn_norm_weight = blk0_loader.get_tensor("attn_norm.weight")?;
        // Expected shape for blk.0.attn_norm.weight: [embedding_length] = [768]
        assert_eq!(
            attn_norm_weight.get_shape(),
            vec![768usize],
            "Shape mismatch for blk.0.attn_norm.weight"
        );

        let qkv_weight = blk0_loader.get_tensor("attn_qkv.weight")?;
        // Expected shape for blk.0.attn_qkv.weight: [3 * embedding_length, embedding_length] = [2304, 768]
        assert_eq!(
            qkv_weight.get_shape(),
            vec![2304usize, 768usize],
            "Shape mismatch for blk.0.attn_qkv.weight"
        );

        // Test sub-scoping with custom prefix as requested ("attn_", "ffn_")
        // Current prefix of blk0_loader is "blk.0."
        let blk0_attn_loader = blk0_loader.pp("attn_"); // New prefix: "blk.0.attn_"
        let attn_norm_weight_v2 = blk0_attn_loader.get_tensor("norm.weight")?; // Full name: "blk.0.attn_norm.weight"
        assert_eq!(
            attn_norm_weight_v2.get_shape(),
            vec![768usize],
            "Shape mismatch for blk.0.attn_norm.weight via custom subscope"
        );

        let blk0_ffn_loader = blk0_loader.pp("ffn_"); // New prefix: "blk.0.ffn_"
        let ffn_norm_weight = blk0_ffn_loader.get_tensor("norm.weight")?; // Full name: "blk.0.ffn_norm.weight"
        // Expected shape for blk.0.ffn_norm.weight: [embedding_length] = [768]
        assert_eq!(
            ffn_norm_weight.get_shape(),
            vec![768usize],
            "Shape mismatch for blk.0.ffn_norm.weight via custom subscope"
        );

        // Test that loading a non-existent tensor fails
        let non_existent_tensor_result = blk0_loader.get_tensor("non_existent_tensor.weight");
        assert!(
            non_existent_tensor_result.is_err(),
            "Expected error for non-existent tensor"
        );

        Ok(())
    }
}
