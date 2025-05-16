use candle_transformers::{models::deepseek2::SplitOp, quantized_var_builder::VarBuilder};
use candle_core::{quantized::QTensor};
use std::{
    io::{Read, Seek},
    ops::Deref,
    sync::Arc,
};

use anyhow::{bail, ensure};
use candle_core::{
    CpuStorage, DType, Device, Storage,
    quantized::{GgmlDType, gguf_file::Content},
};

use crate::{Tensor, tensor::Number};

fn parse_gguf(path: &str) -> anyhow::Result<()> {
    Ok(())
}

#[derive(Debug, Clone)]
struct Gpt2Config {
    embedding_size: usize,
    num_heads: usize,
}

impl Gpt2Config {
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }
    // In gpt2 and llama, Q K V are square
    pub fn hidden_size(&self) -> usize {
        self.embedding_size
    }
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn from_content(content: &Content) -> anyhow::Result<Self> {
        let embedding_size = content.metadata["gpt2.embedding_length"].to_u32()? as usize;
        let num_heads = content.metadata["gpt2.attention.head_count"].to_u32()? as usize;
        Ok(Self { embedding_size, num_heads })
    }
}

struct MatMul<N: Number> {
    pub rhs: Tensor<N>,
}

impl<N: Number> MatMul<N> {
    pub fn new(rhs: Tensor<N>) -> Self {
        Self { rhs }
    }
}

struct Attention<N: Number> {
    q: Tensor<N>,
    q_bias: Tensor<N>,
    k: Tensor<N>,
    k_bias: Tensor<N>,
    v: Tensor<N>,
    v_bias: Tensor<N>,
}

impl Attention<f32> {
    pub fn from_var_builder(
        var_builder: &VarBuilder,
        embedding_size: usize,
        hidden_size: usize,
    ) -> anyhow::Result<Self> {
        ensure!(embedding_size == hidden_size, "embedding_size must be equal to hidden_size");
        //var_builder.get_no_shape(name)
        let qkv = var_builder.get_no_shape("attn_qkv.weight")?;
        let qkv = qkv.dequantize(&Device::Cpu)?;
        println!("qkv: {:?} (elem count: {})", qkv.shape(), qkv.elem_count());
        // Dimension is [hidden_size * 3, embedding_size] or [output_dim, input_dim]
        // Since hidden_size == embedding_size, we can just use the embedding_size * embedding_size
        let mut unfused = unfuse_tensors(qkv, embedding_size * embedding_size)?;
        ensure!(unfused.len() == 3, "bias must have 3 chunks");
        let q = Tensor::new(vec![embedding_size, hidden_size], unfused.remove(0));
        let k = Tensor::new(vec![embedding_size, hidden_size], unfused.remove(0));
        let v = Tensor::new(vec![embedding_size, hidden_size], unfused.remove(0));
        let bias = var_builder.get_no_shape("attn_qkv.bias")?;
        let bias = bias.dequantize(&Device::Cpu)?;
        let mut unfused = unfuse_tensors(bias, embedding_size)?;
        ensure!(unfused.len() == 3, "bias must have 3 chunks");
        let q_bias = Tensor::new(vec![hidden_size], unfused.remove(0));
        let k_bias = Tensor::new(vec![hidden_size], unfused.remove(0));
        let v_bias = Tensor::new(vec![hidden_size], unfused.remove(0));
        Ok(
            Self {
                q,
                q_bias,
                k,
                k_bias,
                v,
                v_bias,
            },
        )
    }
}

#[derive(Debug, Clone)]
struct Embeddings<N: Number> {
    pub emb: Tensor<N>,
}

impl<N: Number> Embeddings<N> {
    pub fn new(emb: Tensor<N>) -> Self {
        Self { emb }
    }
}

impl Embeddings<f32> {
    pub fn from_var_builder(b: &VarBuilder) -> anyhow::Result<Self> {
        let qtensor = b.get_no_shape(&"token_embd.weight")?;
        Ok(Embeddings::new(dequantize(qtensor)?))
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
        ensure!(tensors.iter().all(|t| t.len() == chunk_len), "all chunks must have the same length");
        Ok(tensors)
}

#[cfg(test)]
mod tests {
    use candle_core::{CpuStorage, Device, Storage, Tensor, quantized::gguf_file::Content};
    use candle_transformers::quantized_var_builder::VarBuilder;
    use gguf_rs::get_gguf_container;
    use std::{fs::File, io::Read, ops::Deref, path::Path};

    use crate::parser::gguf::Gpt2Config;

    use super::{Attention, Embeddings};
    // download at https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF
    const GPT2_Q8_0_PATH: &str = "assets/scripts/llms/gpt2.q8_0.gguf";

    #[test]
    fn test_gguf_load_attention() -> anyhow::Result<()> {
        let gguf_path = GPT2_Q8_0_PATH;
        let mut file = File::open(gguf_path)?;
        let gguf_candle = Content::read(&mut file)?;
        let config = Gpt2Config::from_content(&gguf_candle)?;
        println!("config: {:?}", config);
        let var_builder = VarBuilder::from_gguf(&gguf_path, &Device::Cpu)?;
        let attention = Attention::from_var_builder(&var_builder.pp("blk.0"), config.embedding_size(), config.hidden_size())?;
        Ok(())
    }

    #[test]
    fn test_gguf_load_config() -> anyhow::Result<()> {
        let gguf_path = GPT2_Q8_0_PATH;
        let mut file = File::open(gguf_path)?;
        let gguf_candle = Content::read(&mut file)?;
        let config = Gpt2Config::from_content(&gguf_candle)?;
        println!("config: {:?}", config);
        Ok(())
    }

    #[test]
    fn test_gguf_load_embedding() -> anyhow::Result<()> {
        let gguf_path = GPT2_Q8_0_PATH;
        /// VarBuilder has the disadvantage of loading everything into memory first, but it has the
        /// advantage of sub scoping the naming.
        /// TODO: make a lazy var builder that combine benefits from current VarBuilder and just COntent
        /// that lazy load
        let gguf_candle = VarBuilder::from_gguf(&gguf_path, &Device::Cpu)?;
        let embedding = Embeddings::from_var_builder(&gguf_candle)?;
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
}
