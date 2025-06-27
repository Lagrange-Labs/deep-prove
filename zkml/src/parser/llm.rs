use anyhow::bail;

use crate::{
    Tensor,
    layers::{
        Layer,
        activation::{Activation, GELU},
        add,
        concat_matmul::ConcatMatMul,
        matrix_mul::MatMul,
        provable::{Edge, Node, NodeId},
        reshape::Reshape,
        transformer::{
            embeddings::Embeddings, layernorm::LayerNorm, logits::Logits, mha::MhaQK,
            positional::Positional, qkv::QKV, softmax::Softmax,
        },
    },
    model::Model,
    padding::PaddingMode,
    tensor::{Number, Shape},
};

/// Intermediary struct to hold the config of the model.
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

#[derive(Debug, Clone)]
pub enum LLMVariant {
    GPT2,
}
impl LLMVariant {
    pub fn from_content(variant: &str) -> anyhow::Result<Self> {
        // Convert gguf_file::Value to String, then get &str
        let variant_str = variant.to_string();
        match variant_str.as_str() {
            "gpt2" => Ok(Self::GPT2),
            "distilgpt2" => Ok(Self::GPT2),
            a => bail!("unsupported architecture: {:?}", a),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LLMModel {
    GPT2(GPT2Model),
}

#[derive(Debug, Clone)]
pub struct GPT2Model {
    pub embeddings: Embeddings<f32>,
    pub positional: Positional<f32>,
    pub blocks: Vec<Attention<f32>>,
    /// Final LayerNorm applied after all transformer blocks (ln_f in GPT-2)
    pub final_norm: LayerNorm<f32>,
    /// final projection on token sizes to before selecting next token
    pub final_proj: MatMul<f32>,
}

impl GPT2Model {
    pub fn new(
        embeddings: Embeddings<f32>,
        positional: Positional<f32>,
        blocks: Vec<Attention<f32>>,
        final_norm: LayerNorm<f32>,
        final_proj: MatMul<f32>,
    ) -> Self {
        Self {
            embeddings,
            positional,
            blocks,
            final_norm,
            final_proj,
        }
    }
    /// Creates a Model<f32> from the GPT2Model. Currently it does NOT support the embeddings and positional nor
    /// multiple passes.
    /// User input shape is the shape of the user input tensor.
    pub fn to_provable_model(
        self,
        c: &LLMConfig,
        user_input_shape: Shape,
    ) -> anyhow::Result<Model<f32>> {
        let mut model =
            Model::new_from_input_shapes(vec![user_input_shape], PaddingMode::NoPadding);

        let mut last_node_id =
            Some(model.add_consecutive_layer(Layer::Embeddings(self.embeddings), None)?);
        last_node_id =
            Some(model.add_consecutive_layer(Layer::Positional(self.positional), last_node_id)?);
        for block in self.blocks {
            last_node_id = Some(block.write_to_model(&mut model, last_node_id, c)?);
        }
        last_node_id =
            Some(model.add_consecutive_layer(Layer::LayerNorm(self.final_norm), last_node_id)?);
        last_node_id =
            Some(model.add_consecutive_layer(Layer::MatMul(self.final_proj), last_node_id)?);
        model.add_consecutive_layer(Layer::Logits(Logits::Argmax), last_node_id)?;
        model.route_output(None)?;
        Ok(model)
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
#[derive(Debug, Clone)]
pub struct FeedForward<N: Number> {
    pub norm: LayerNorm<N>,
    pub up: Tensor<N>,
    pub up_bias: Tensor<N>,
    pub down: Tensor<N>,
    pub down_bias: Tensor<N>,
}
impl FeedForward<f32> {
    pub fn write_to_model(
        self,
        model: &mut Model<f32>,
        input_node_id: NodeId,
    ) -> anyhow::Result<NodeId> {
        let layernorm = self.norm;
        // let up = MatMul::new_constant(self.up, self.up_bias);
        // TODO bias
        let up = MatMul::new_constant(self.up, Some(self.up_bias))?;
        let activation = GELU::new();
        // let down = MatMul::new_constant(self.down, self.down_bias);
        let down = MatMul::new_constant(self.down, Some(self.down_bias))?;
        let add = add::Add::new();
        let last_node_id =
            model.add_consecutive_layer(Layer::LayerNorm(layernorm), Some(input_node_id))?;
        let last_node_id = model.add_consecutive_layer(Layer::MatMul(up), Some(last_node_id))?;
        let last_node_id = model.add_consecutive_layer(
            Layer::Activation(Activation::Gelu(activation)),
            Some(last_node_id),
        )?;
        let last_node_id = model.add_consecutive_layer(Layer::MatMul(down), Some(last_node_id))?;
        model.add_node(Node::new(
            vec![Edge::new(input_node_id, 0), Edge::new(last_node_id, 0)],
            Layer::Add(add),
        ))
    }
}

impl Attention<f32> {
    pub fn write_to_model(
        self,
        model: &mut Model<f32>,
        input_node_id: Option<NodeId>,
        c: &LLMConfig,
    ) -> anyhow::Result<NodeId> {
        let qkv = QKV::new(
            self.q,
            self.q_bias,
            self.k,
            self.k_bias,
            self.v,
            self.v_bias,
        );
        let mha = MhaQK::new(c.num_heads, c.head_dim());
        let softmax = Softmax::<f32>::new()
            .with_scale((1.0 / (c.head_dim() as f32)).sqrt())
            .on_dim(1);
        let qkt_v = ConcatMatMul::new_with_permute(vec![1, 0, 2]);
        let out = MatMul::new_constant(self.out, Some(self.out_bias))?;
        let reshape_merged = Reshape::new_subspace(1..=2, vec![c.hidden_size]);
        // input is [seq_len, emb_size]
        let last_node_id =
            model.add_consecutive_layer(Layer::LayerNorm(self.norm), input_node_id)?;
        // shape goes to [seq_len, hidden_size] for each, Q K and V
        let last_node_id = model.add_consecutive_layer(Layer::QKV(qkv), Some(last_node_id))?;
        // then this output two tensors:
        // * first one is [num_heads, seq_len] (Q @ K^T - all heads concatenated)
        // * second one is [num_heads, seq_len, head_dim] (V)
        let mha_id = model.add_consecutive_layer(Layer::MhaQK(mha), Some(last_node_id))?;

        // same output shape as QKT but now we have softmaxed values
        let last_node_id = model.add_node(Node::new(
            vec![Edge::new(mha_id, 0)],
            Layer::Softmax(softmax),
        ))?;

        let last_node_id = model.add_node(Node::new(
            // here we take the first output of softmax (QKT) and the second output of MhaQK (V)
            vec![Edge::new(last_node_id, 0), Edge::new(mha_id, 1)],
            Layer::ConcatMatMul(qkt_v),
        ))?;
        let last_node_id =
            model.add_consecutive_layer(Layer::Reshape(reshape_merged), Some(last_node_id))?;
        let last_node_id = model.add_consecutive_layer(Layer::MatMul(out), Some(last_node_id))?;
        let last_node_id = model.add_node(Node::new(
            vec![
                Edge {
                    // here we dont know if the input is the input to the model or an input coming from previous layers
                    // so if there is no layer before this attention, we take the input of the model
                    node: input_node_id,
                    index: 0,
                },
                Edge::new(last_node_id, 0),
            ],
            Layer::Add(add::Add::new()),
        ))?;
        self.feedforward.write_to_model(model, last_node_id)
    }
}
