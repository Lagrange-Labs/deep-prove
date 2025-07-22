//! Multihead attention layer:
//! The module performs all the operations inside the multi-head attention layer, relying on
//! ConcatMatMul and Softmax layers as building blocks.
use std::iter::once;

use crate::{
    Claim, Element, Prover, ScalingFactor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        concat_matmul::{
            ConcatMatMul, ConcatMatMulCtx, ConcatMatMulProof, InputMatrixDimensions, Permutation,
        },
        provable::{
            Evaluate, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
        reshape::{Reshape, ReshapeCtx},
        transformer::softmax::{
            OUTPUT_SCALE_FACTOR, Softmax, SoftmaxCtx, SoftmaxData, SoftmaxProof,
        },
    },
    lookup::context::LookupWitnessGen,
    model::StepData,
    padding::{GarbagePad, PaddingMode, ShapeInfo},
    quantization::{Fieldizer, TensorFielder},
    tensor::{Number, Shape},
};
use anyhow::{anyhow, ensure};
use ff_ext::{ExtensionField, FieldFrom, SmallField};
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use p3_field::FieldAlgebra;
use p3_goldilocks::Goldilocks;
use poseidon::poseidon_hash::PoseidonHash;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{Tensor, layers::provable::LayerOut};

#[derive(Clone, Debug)]
pub struct MhaData<E: ExtensionField> {
    // Output tensor of Mha before final reshape
    pre_reshaping_out: Tensor<E>,
    softmax_out: Tensor<Element>, // this needs to be an `Element` to call Softmax::lookup_witness
    softmax_data: SoftmaxData<E>,
    softmax_in: Tensor<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MhaCtx<E> {
    node_id: NodeId,
    inputs_reshape: ReshapeCtx,
    final_mul: ConcatMatMulCtx<E>,
    softmax: SoftmaxCtx,
    qk: ConcatMatMulCtx<E>,
    final_reshape: ReshapeCtx,
}

struct MhaOutputShaper<'a> {
    inputs_reshape: &'a dyn OpInfo,
    final_mul: &'a dyn OpInfo,
    softmax: &'a dyn OpInfo,
    qk: &'a dyn OpInfo,
    final_reshape: &'a dyn OpInfo,
}

impl<'a, N: Number> From<&'a Mha<N>> for MhaOutputShaper<'a> {
    fn from(value: &'a Mha<N>) -> Self {
        Self {
            inputs_reshape: &value.inputs_reshape,
            final_mul: &value.final_mul,
            softmax: &value.softmax,
            qk: &value.qk,
            final_reshape: &value.final_reshape,
        }
    }
}

impl<'a, E: ExtensionField> From<&'a MhaCtx<E>> for MhaOutputShaper<'a> {
    fn from(value: &'a MhaCtx<E>) -> Self {
        Self {
            inputs_reshape: &value.inputs_reshape,
            final_mul: &value.final_mul,
            softmax: &value.softmax,
            qk: &value.qk,
            final_reshape: &value.final_reshape,
        }
    }
}

impl<'a> MhaOutputShaper<'a> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let reshaped_input_shapes = self
            .inputs_reshape
            .output_shapes(input_shapes, padding_mode);

        let linear_out_shapes = self
            .qk
            .output_shapes(&reshaped_input_shapes[..2], padding_mode);

        let soft_out_shapes = self.softmax.output_shapes(&linear_out_shapes, padding_mode);

        let final_mul_input_shapes = vec![
            soft_out_shapes[0].clone(),
            reshaped_input_shapes[2].clone(), // V
        ];

        let final_mul_shapes = self
            .final_mul
            .output_shapes(&final_mul_input_shapes, padding_mode);

        self.final_reshape
            .output_shapes(&final_mul_shapes, padding_mode)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct MhaProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    final_mul_proof: ConcatMatMulProof<E>,
    softmax_proof: SoftmaxProof<E, PCS>,
    qk_proof: ConcatMatMulProof<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> MhaProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        self.softmax_proof.get_lookup_data()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mha<N> {
    inputs_reshape: Reshape, // ToDo: can be removed once we will include padding in QKV layer
    context_length: usize,
    num_heads: usize,
    head_dim: usize,
    qk: ConcatMatMul,
    softmax: Softmax<N>,
    final_mul: ConcatMatMul,
    final_reshape: Reshape, /* ToDo: can be removed once we will handle unpadding in subsequent linear layer */
}

impl<N: Number> Mha<N> {
    pub fn new(context_length: usize, num_heads: usize, head_dim: usize) -> anyhow::Result<Self> {
        let inputs_reshape = Reshape::new_subspace(1..2, vec![num_heads, head_dim]);
        let qk = ConcatMatMul::new(
            InputMatrixDimensions::new(1, 2, 0),
            InputMatrixDimensions::new(1, 2, 0),
        )
        .update_intermediate_bit_size(
            vec![
                vec![context_length, num_heads, head_dim].into(),
                vec![context_length, num_heads, head_dim].into(),
            ],
            None,
            None, // use the default quantization range
        );
        let softmax = Softmax::new().with_scale(N::from_f32((1.0 / (head_dim as f32)).sqrt())?);
        let final_mul = ConcatMatMul::new_with_permute(
            InputMatrixDimensions::new(0, 2, 1),
            InputMatrixDimensions::new(1, 0, 2),
            Permutation::new(vec![1, 0, 2]),
        )
        .update_intermediate_bit_size(
            vec![
                vec![num_heads, context_length, context_length].into(),
                vec![context_length, num_heads, head_dim].into(),
            ],
            Some(OUTPUT_SCALE_FACTOR), // here instead the output of softmax can be up
            // to `OUTPUT_SCALE_FACTOR` rather than the usual quantization range
            None,
        );
        // reshape the output from [q_len, num_heads, head_dim] to [q_len, num_heads*head_dim]
        let final_reshape = Reshape::new_subspace(1..=2, vec![num_heads * head_dim]);
        Ok(Self {
            inputs_reshape,
            context_length,
            num_heads,
            head_dim,
            qk,
            softmax,
            final_mul,
            final_reshape,
        })
    }

    // compute ephemeral node ids to be employed for the sub-layers called in MHA.
    // It uses a collision-resistant hash function to pseudo-randomly select an ephemeral id
    // The id is ephemeral in the sense that it will not correspond to an actual node in the
    // model
    fn compute_ephemeral_node_id(node_id: NodeId, domain_separator: &str) -> NodeId {
        let payload = once(Goldilocks::from_canonical_u64(node_id as u64))
            .chain(
                domain_separator
                    .as_bytes()
                    .iter()
                    .map(|b| Goldilocks::from_canonical_u8(*b)),
            )
            .collect_vec();
        PoseidonHash::hash_or_noop(&payload).0[0].to_canonical_u64() as usize
    }

    fn qk_node_id(node_id: NodeId) -> NodeId {
        Self::compute_ephemeral_node_id(node_id, "qk")
    }

    fn softmax_node_id(node_id: NodeId) -> NodeId {
        Self::compute_ephemeral_node_id(node_id, "softmax")
    }

    fn final_mul_node_id(node_id: NodeId) -> NodeId {
        Self::compute_ephemeral_node_id(node_id, "final_mul")
    }

    /// Core method to evaluate the layer; it returns also the intermediate outputs of final_mul, softmax
    /// and qk sub-layers, which might be necessary to build the proving data
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_with_intermediate_outputs<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<(
        LayerOut<N, E>,
        LayerOut<N, E>,
        LayerOut<N, E>,
        LayerOut<N, E>,
    )>
    where
        Softmax<N>: Evaluate<N>,
    {
        let unpadded_input_shapes = if unpadded_input_shapes.is_empty() {
            // take input shapes from inputs
            inputs.iter().map(|input| input.get_shape()).collect()
        } else {
            unpadded_input_shapes
        };

        ensure!(
            inputs.len() == 3,
            "MHA layer expects 3 inputs, found {}",
            inputs.len()
        );

        let reshaped_input_shapes = self
            .inputs_reshape
            .output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);

        let reshaped_inputs = self
            .inputs_reshape
            .evaluate::<E>(inputs, unpadded_input_shapes)?;

        let qk_out_shapes = self
            .qk
            .output_shapes(&reshaped_input_shapes, PaddingMode::NoPadding);

        let qk_out = self.qk.evaluate::<E>(
            &reshaped_inputs.outputs()[..2],
            reshaped_input_shapes[..2].to_vec(),
        )?;

        // apply softmax
        let soft_out_shapes = self
            .softmax
            .output_shapes(&qk_out_shapes, PaddingMode::NoPadding);

        let soft_out = self
            .softmax
            .evaluate::<E>(&qk_out.outputs(), qk_out_shapes)?;

        ensure!(
            soft_out.outputs().len() == 1,
            "Softmax should return one output"
        );

        let final_mul_input_shapes =
            vec![soft_out_shapes[0].clone(), reshaped_input_shapes[2].clone()];

        let out_shapes = self
            .final_mul
            .output_shapes(&final_mul_input_shapes, PaddingMode::NoPadding);

        let final_mul_out = self.final_mul.evaluate::<E>(
            &[soft_out.outputs()[0], reshaped_inputs.outputs()[2]],
            final_mul_input_shapes,
        )?;

        let out = self
            .final_reshape
            .evaluate(&final_mul_out.outputs(), out_shapes)?;

        Ok((out, final_mul_out, soft_out, qk_out))
    }
}

impl<N: Number> OpInfo for Mha<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        MhaOutputShaper::from(self).output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "MHA({}, {}): \t {} \t {}, \t {}",
            self.num_heads,
            self.head_dim,
            self.qk.describe(),
            self.softmax.describe(),
            self.final_mul.describe(),
        )
        .to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<f32> for Mha<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let (out, _, _, _) =
            self.evaluate_with_intermediate_outputs(inputs, unpadded_input_shapes)?;

        Ok(out)
    }
}

impl Evaluate<Element> for Mha<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        let (out, final_mul_out, soft_out, qk_out) =
            self.evaluate_with_intermediate_outputs(inputs, unpadded_input_shapes)?;

        let LayerOut {
            outputs,
            proving_data,
        } = soft_out;
        let ProvingData::Softmax(softmax_data) = proving_data else {
            Err(anyhow!("Softmax data not found while evaluating MhaLayer"))?
        };
        let data = MhaData {
            pre_reshaping_out: final_mul_out.outputs()[0].to_fields(),
            softmax_data,
            softmax_out: outputs[0].clone(),
            softmax_in: qk_out.outputs[0].to_fields(),
        };
        Ok(out.with_proving_data(ProvingData::Mha(data)))
    }
}

impl QuantizeOp for Mha<f32> {
    type QuantizedOp = Mha<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: crate::layers::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 3,
            "Expected 3 input scaling factors for MHA layer, found {}",
            input_scaling.len()
        );

        // for the first concat mat mul, we simply need to compute the scaling factor of the product, without requantization
        let product_scaling = {
            let scale = input_scaling[0].scale() * input_scaling[1].scale();
            let output_domain = self.qk.output_domain();
            let quantized_domain = Some((-output_domain, output_domain));
            ScalingFactor::from_scale(scale, quantized_domain)
        };

        // quantize data for softmax
        let QuantizeOutput {
            quantized_op: quantized_softmax,
            output_scalings,
            ..
        } = self
            .softmax
            .quantize_op::<S>(data, node_id, &[product_scaling])?;

        ensure!(
            output_scalings.len() == 1,
            "Expected 1 output scaling for softmax, found {}",
            output_scalings.len()
        );

        // prepare input scaling for final multiplication operation
        let final_mul_scalings = vec![output_scalings[0], input_scaling[2]];
        let quantized_out = self
            .final_mul
            .quantize_op::<S>(data, node_id, &final_mul_scalings)?;

        let quantized_mha = Self::QuantizedOp {
            inputs_reshape: self.inputs_reshape,
            context_length: self.context_length,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            qk: self.qk,
            softmax: quantized_softmax,
            final_mul: quantized_out.quantized_op,
            final_reshape: self.final_reshape,
        };
        Ok(QuantizeOutput {
            quantized_op: quantized_mha,
            output_scalings: quantized_out.output_scalings,
            requant_layer: quantized_out.requant_layer,
        })
    }
}

impl<E: ExtensionField> ProveInfo<E> for Mha<Element> {
    fn step_info(&self, id: NodeId, aux: ContextAux) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        let (ctx, mut reshaped_aux) = self.inputs_reshape.step_info(
            id, // No need to have an ad-hoc id
            aux,
        )?;

        let LayerCtx::<E>::Reshape(inputs_reshape_ctx) = ctx else {
            unreachable!()
        };

        ensure!(
            reshaped_aux.last_output_shape.len() == 3,
            "Expected 3 input shapes in Mha layer ctx, found {}",
            reshaped_aux.last_output_shape.len(),
        );

        // save v_shape as it is going to be used later on for `final_mul`
        let v_shape = reshaped_aux.last_output_shape.pop().unwrap();

        let qk_aux = ContextAux {
            tables: reshaped_aux.tables,
            last_output_shape: reshaped_aux.last_output_shape[..2].to_vec(),
            model_polys: reshaped_aux.model_polys,
            max_poly_len: reshaped_aux.max_poly_len,
        };

        let (ctx, aux) = self.qk.step_info(Self::qk_node_id(id), qk_aux)?;

        let LayerCtx::ConcatMatMul(qk_ctx) = ctx else {
            unreachable!()
        };

        let (ctx, mut aux) = self.softmax.step_info(Self::softmax_node_id(id), aux)?;

        let LayerCtx::<E>::Softmax(softmax_ctx) = ctx else {
            unreachable!()
        };

        ensure!(
            aux.last_output_shape.len() == 1,
            "Expected 1 output shape from softmax when building Mha layer ctx, found {}",
            aux.last_output_shape.len(),
        );

        // prepare `last_output_shape` in final mul `ContextAux`: we need to add `v_shape`
        aux.last_output_shape.push(v_shape);
        let (ctx, aux) = self.final_mul.step_info(Self::final_mul_node_id(id), aux)?;

        let LayerCtx::ConcatMatMul(final_mul_ctx) = ctx else {
            unreachable!()
        };

        let (ctx, aux) = self.final_reshape.step_info(
            id, // No need to have an ad-hoc id
            aux,
        )?;

        let LayerCtx::<E>::Reshape(final_reshape_ctx) = ctx else {
            unreachable!()
        };

        let ctx = LayerCtx::Mha(MhaCtx {
            node_id: id,
            inputs_reshape: inputs_reshape_ctx,
            final_mul: final_mul_ctx,
            softmax: softmax_ctx,
            qk: qk_ctx,
            final_reshape: final_reshape_ctx,
        });

        Ok((ctx, aux))
    }
}

pub(crate) fn pad_matrix_to_ignore_mha_garbage<T>(
    matrix: &Tensor<T>,
    unpadded_mha_shape: &Shape,
    padded_mha_shape: &Shape,
    padded_shape: Shape,
) -> anyhow::Result<Tensor<T>>
where
    T: Copy + Clone + Send + Sync + Default,
{
    ensure!(
        unpadded_mha_shape.rank() == padded_mha_shape.rank(),
        "Rank of padded and unpadded shapes in garbage pad of Mha layer differ: unpadded = {}, padded {}",
        unpadded_mha_shape.rank(),
        padded_mha_shape.rank(),
    );

    ensure!(
        unpadded_mha_shape.rank() == 3,
        "Rank of shapes in garbage pad of Mha layer must be 3, found {}",
        unpadded_mha_shape.rank()
    );

    ensure!(
        padded_shape.is_matrix(),
        "Target padded shape to remove garbage for Mha layer is not a matrix"
    );

    let num_heads = unpadded_mha_shape[1];
    let padded_num_heads = padded_mha_shape[1];
    let head_dim = unpadded_mha_shape[2];
    let padded_head_dim = padded_mha_shape[2];

    let nrows = padded_shape[0].max(padded_num_heads * padded_head_dim);
    let ncols = padded_shape[1];

    let unpadded_shape = matrix.get_shape();

    ensure!(
        unpadded_shape.is_matrix(),
        "Tensor to be padded to remove garbage for Mha layer is not a matrix"
    );

    let padded_matrix_data = (0..nrows * ncols)
        .into_par_iter()
        .map(|i| {
            let row = i / ncols;
            let col = i % ncols;
            // check if this row corresponds to a garbage entry in the matrix produced by Mha layer
            let is_not_garbage_row =
                row % padded_head_dim < head_dim && row / padded_head_dim < num_heads;
            // check it the column of the row corresponds to an entry in the original matrix or it's a padding column
            let is_not_padding_column = col < unpadded_shape[1];
            let new_item = if is_not_garbage_row && is_not_padding_column {
                // we need to get an entry from the original matrix
                let original_row = row / padded_head_dim * head_dim + row % padded_head_dim;
                matrix.get_data()[original_row * unpadded_shape[1] + col]
            } else {
                // it's either an entry in a garbage row or a padded column in a non-garbage row: in both cases,
                // we fill it with 0
                T::default()
            };
            new_item
        })
        .collect();

    Ok(Tensor::new(vec![nrows, ncols].into(), padded_matrix_data))
}

impl PadOp for Mha<Element> {
    fn pad_node(mut self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let inputs_reshape = self.inputs_reshape.pad_node(si)?;

        // Save shape about input tensor v for later use; we can remove it from
        // `si`, and leave only shapes about q and k inputs, because v is not
        // necessary for the next operation
        let v_shape = si.shapes.pop().unwrap();

        let qk = self.qk.pad_node(si)?;

        // softmax takes as input the output of qk, so we can just provide
        // shape info `si`
        let softmax = self.softmax.pad_node(si)?;

        // now we need to build shape info for final_mul from softmax output
        // and from `v`
        si.shapes.push(v_shape);
        let final_mul = self.final_mul.pad_node(si)?;

        // add garbage pad information to `si`
        ensure!(
            si.shapes.len() == 1,
            "Expected 1 output shape after padding Mha, found {}",
            si.shapes.len()
        );

        let garbage_pad = GarbagePad::MHA((
            si.unpadded_input_shapes().pop().unwrap(),
            si.padded_input_shapes().pop().unwrap(),
        ));
        si.shapes = vec![si.shapes.pop().unwrap().with_garbage_pad(garbage_pad)];

        // need to properly pad the reshape new dimension
        let padded_num_heads = self.num_heads.next_power_of_two();
        let padded_head_dim = self.head_dim.next_power_of_two();

        let Reshape::Subspace(subspace) = &mut self.final_reshape else {
            unreachable!("Final reshape in MHA layer must be Subspace variant")
        };
        subspace.to_add = vec![padded_head_dim * padded_num_heads];

        let final_reshape = self.final_reshape.pad_node(si)?;

        Ok(Self {
            inputs_reshape,
            context_length: self.context_length,
            num_heads: padded_num_heads,
            head_dim: padded_head_dim,
            qk,
            softmax,
            final_mul,
            final_reshape,
        })
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS> for Mha<Element> {
    type Ctx = MhaCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let inputs = &step_data.inputs;

        ensure!(
            inputs.len() == 3,
            "Expected 3 inputs when proving MHA layer, found {}",
            inputs.len()
        );

        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Expected 1 one output when proving MHA layer, found {}",
            step_data.outputs.outputs().len()
        );

        // apply reshaping to input and output tensors before employing them in proving logic
        let reshaped_inputs = self.inputs_reshape.evaluate_layer::<E, E>(
            &inputs.iter().collect_vec(),
            inputs.iter().map(|input| input.get_shape()).collect(),
        )?;

        let reshaped_inputs = reshaped_inputs.outputs();

        let mha_data = step_data
            .outputs
            .try_mha_data()
            .ok_or(anyhow!("MhaData not found when proving Mha layer"))?;

        let (mut claims, final_mul_proof) = self.final_mul.prove_step(
            last_claims,
            &mha_data.pre_reshaping_out,
            &[&mha_data.softmax_out.to_fields(), reshaped_inputs[2]],
            prover,
        )?;

        ensure!(
            claims.len() == 2,
            "Expected 2 input claims for mul with V in MhaLayer, found {}",
            claims.len()
        );

        let v_input_claim = claims.pop().unwrap();
        let softmax_out_claim = claims.pop().unwrap();

        let (claims, softmax_proof) = self.softmax.prove_step(
            node_id,
            vec![&softmax_out_claim],
            &mha_data.softmax_data,
            prover,
        )?;

        ensure!(
            claims.len() == 1,
            "Expected 1 input claim for Softmax in MhaLayer, found {}",
            claims.len()
        );

        let (mut input_claims, qk_proof) = self.qk.prove_step(
            vec![&claims[0]],
            &mha_data.softmax_in,
            &reshaped_inputs[..2],
            prover,
        )?;

        ensure!(
            input_claims.len() == 2,
            "Expected 2 input claims for QK matrix multiplication in Mha layer, found {}",
            input_claims.len(),
        );

        // append claim about V to claims about Q and K
        input_claims.push(v_input_claim);

        // add proof for this node
        let proof = MhaProof {
            final_mul_proof,
            softmax_proof,
            qk_proof,
        };

        prover.push_proof(node_id, LayerProof::Mha(proof));

        Ok(input_claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &crate::Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> anyhow::Result<LookupWitnessGen<E, PCS>> {
        let mha_data = step_data
            .outputs
            .try_mha_data()
            .ok_or(anyhow!("MhaData not found when proving Mha layer"))?;
        self.softmax
            .lookup_witness(id, ctx, &mha_data.softmax_out, &mha_data.softmax_data)
    }
}

impl<E: ExtensionField> OpInfo for MhaCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        MhaOutputShaper::from(self).output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "MHACtx({}): \t {} \t {}, \t {}",
            self.node_id,
            self.qk.describe(),
            self.softmax.describe(),
            self.final_mul.describe(),
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for MhaCtx<E> {
    type Proof = MhaProof<E, PCS>;

    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        // we first need to reconstruct the `ShapeStep` for each intermediate layer
        ensure!(
            shape_step.unpadded_input_shape.len() == 3,
            "Expected 3 unpadded input shapes for MhaLayer, found {}",
            shape_step.unpadded_input_shape.len()
        );

        ensure!(
            shape_step.padded_input_shape.len() == 3,
            "Expected 3 padded input shapes for MhaLayer, found {}",
            shape_step.padded_input_shape.len()
        );

        let reshaped_inputs = LayerCtx::<E>::Reshape(self.inputs_reshape.clone()).shape_step(
            &shape_step.unpadded_input_shape,
            &shape_step.padded_input_shape,
        );

        let qk_shapes = LayerCtx::ConcatMatMul(self.qk.clone()).shape_step(
            &reshaped_inputs.unpadded_output_shape[..2],
            &reshaped_inputs.padded_output_shape[..2],
        );

        let softmax_shapes = LayerCtx::<E>::Softmax(self.softmax.clone()).shape_step(
            &qk_shapes.unpadded_output_shape,
            &qk_shapes.padded_output_shape,
        );

        let final_mul_shapes = LayerCtx::ConcatMatMul(self.final_mul.clone()).shape_step(
            &[
                softmax_shapes.unpadded_output_shape[0].clone(),
                reshaped_inputs.unpadded_output_shape[2].clone(),
            ],
            &[
                softmax_shapes.padded_output_shape[0].clone(),
                reshaped_inputs.padded_output_shape[2].clone(),
            ],
        );

        // now we call the verifier of each sub-layer
        let mut claims = self.final_mul.verify(
            &proof.final_mul_proof,
            last_claims,
            verifier,
            &final_mul_shapes,
        )?;

        ensure!(
            claims.len() == 2,
            "Expected 2 input claims for multiplication with V when verifying Mha layer, found {} claims",
            claims.len(),
        );

        let v_input_claim = claims.pop().unwrap();

        let softmax_out_claim = claims.pop().unwrap();

        let claims = self.softmax.verify(
            &proof.softmax_proof,
            &[&softmax_out_claim],
            verifier,
            &softmax_shapes,
        )?;

        let mut input_claims =
            self.qk
                .verify(&proof.qk_proof, &[&claims[0]], verifier, &qk_shapes)?;

        // add claim about V to input claims

        input_claims.push(v_input_claim);

        Ok(input_claims)
    }
}

pub fn zeroifier<N: Number>(num_heads: usize, q_len: usize, seq_len: usize) -> Tensor<N> {
    let zeroified = (0..num_heads)
        .into_par_iter()
        .flat_map(|_head| {
            (0..q_len)
                .into_par_iter()
                .flat_map(|q| {
                    (0..seq_len)
                        .map(|e| if e > q { N::default() } else { N::unit() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tensor::new(vec![num_heads, q_len, seq_len].into(), zeroified)
}

/// Sets to minus infinity the part that should be ignored on each Q "sequence" for each head
pub fn infinitizer<N: Number>(
    num_heads: usize,
    q_len: usize,
    seq_len: usize,
    minus_infinity: N,
) -> Tensor<N> {
    let zeroified = (0..num_heads)
        .into_par_iter()
        .flat_map(|_head| {
            (0..q_len)
                .into_par_iter()
                .flat_map(|q| {
                    (0..seq_len)
                        .map(|e| if e > q { minus_infinity } else { N::default() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tensor::new(vec![num_heads, q_len, seq_len].into(), zeroified)
}

/// Method to efficiency evaluate the MLE of the zeroifier matrix over a random
/// point. The point is provided already split between coordinates referring to the
/// columns and coordinates referring to the rows of the matrix.
/// Currently, it works only for a square zeroifier matrix
pub fn eval_zeroifier_mle<F: ExtensionField>(column_point: &[F], row_point: &[F]) -> F {
    column_point
        .iter()
        .zip(row_point)
        .fold(F::ONE, |acc, (&c, &r)| {
            acc * (F::ONE - c - r + F::from_canonical_u64(2) * c * r) + (F::ONE - c) * r
        })
}

/// Method to efficiency evaluate the MLE of the infinitizer matrix over a random
/// point. The point is provided already split between coordinates referring to the
/// columns and coordinates referring to the rows of the matrix.
/// Currently, it works only for a square infinitizer matrix
pub fn eval_infinitizer_mle<F: ExtensionField + FieldFrom<u64>>(
    column_point: &[F],
    row_point: &[F],
    minus_infinity: Element,
) -> F {
    <Element as Fieldizer<F>>::to_field(&minus_infinity)
        * (F::ONE - eval_zeroifier_mle(column_point, row_point))
}

#[cfg(test)]
mod test {
    use std::fs::File;

    use anyhow::Context;
    use ff_ext::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{
        Element, init_test_logging,
        layers::{
            Layer,
            matrix_mul::{MatMul, OperandMatrix},
            transformer::{qkv::QKV, test::GPT2Output},
        },
        model::{
            Model, ToIterator,
            test::{prove_model, prove_quantized_model, quantize_model},
        },
        padding::pad_model,
        parser::{
            file_cache,
            gguf::{FileTensorLoader, tests::GPT2_Q8_0_URL},
            json::{self, test::TINY_GPT2_DEBUG_NAME},
            llm::{LLMConfig, LLMModel},
        },
        quantization::{self, Fieldizer},
        testing::random_field_vector,
        to_bit_sequence_le,
    };

    use super::*;

    #[test]
    fn test_mha_qk_vector_and_matrix() {
        struct Params {
            seq_len: usize,
            q_len: usize,
        }
        for params in vec![
            Params {
                seq_len: 2,
                q_len: 1,
            },
            Params {
                seq_len: 2,
                q_len: 2,
            },
            Params {
                seq_len: 2,
                q_len: 3,
            },
        ] {
            let num_heads = 2;
            let head_dim = 4;
            let q_len = params.q_len;
            let seq_len = params.seq_len;
            let mha_qk = Mha::<Element>::new(seq_len, num_heads, head_dim)
                .unwrap()
                .qk;
            let q = Tensor::<Element>::random(&vec![q_len, num_heads, head_dim].into());
            let k = Tensor::<Element>::random(&vec![seq_len, num_heads, head_dim].into());
            let mut output = mha_qk
                .evaluate::<GoldilocksExt2>(&[&q, &k], vec![])
                .unwrap();
            assert_eq!(output.outputs.len(), 1);
            let qk = output.outputs.remove(0);
            // normally [1,seq_len] per head, so with all heads [num_heads, 1, seq_len]
            assert_eq!(qk.get_shape(), vec![num_heads, q_len, seq_len].into());
            let output_shapes =
                mha_qk.output_shapes(&[q.get_shape(), k.get_shape()], PaddingMode::NoPadding);
            assert_eq!(output_shapes, vec![qk.get_shape()]);
        }
    }

    #[test]
    fn test_mha_final_mul() {
        struct Params {
            seq_len: usize,
            q_len: usize,
        }
        for params in vec![
            Params {
                seq_len: 2,
                q_len: 1,
            },
            Params {
                seq_len: 2,
                q_len: 2,
            },
        ] {
            let num_heads = 2;
            let head_dim = 4;
            let q_len = params.q_len;
            let seq_len = params.seq_len;
            let qk = Tensor::<Element>::random(&vec![num_heads, q_len, seq_len].into());
            let v = Tensor::<Element>::random(&vec![seq_len, num_heads, head_dim].into());
            let mha_mul = Mha::<Element>::new(seq_len, num_heads, head_dim)
                .unwrap()
                .final_mul;
            let mut output = mha_mul
                .evaluate::<GoldilocksExt2>(&[&qk, &v], vec![qk.get_shape(), v.get_shape()])
                .expect("mha_final_mul should not fail");
            assert_eq!(output.outputs.len(), 1);
            let out = output.outputs.remove(0);
            assert_eq!(out.get_shape(), vec![q_len, num_heads, head_dim].into());
            let output_shapes =
                mha_mul.output_shapes(&[qk.get_shape(), v.get_shape()], PaddingMode::NoPadding);
            assert_eq!(output_shapes, vec![out.get_shape()]);
        }
    }

    #[test]
    fn test_zeroifier_and_infinitizer() {
        let num_heads = 2;
        let q_len = 4;
        let seq_len = 4;
        let input = Tensor::<Element>::random(&vec![num_heads, q_len, seq_len].into());
        let zeros = zeroifier(num_heads, q_len, seq_len);
        let minus_infinity = infinitizer(num_heads, q_len, seq_len, Element::MIN);
        let zeroified = input.mul(&zeros);
        let infinitized = zeroified.add(&minus_infinity);
        assert_eq!(zeroified.get_shape(), input.get_shape());
        assert_eq!(infinitized.get_shape(), input.get_shape());
        let (slice_it, _) = infinitized.slice_on_dim(0);
        slice_it.enumerate().all(|(head_idx, head)| {
            head.chunks(q_len).enumerate().all(|(q_idx, q)| {
                q.iter().enumerate().all(|(i, v)| {
                    let input_value = input.get(vec![head_idx, q_idx, i]);
                    // if we are less than the q_len, we dont have causal mask
                    if i <= q_idx {
                        input_value == *v
                    } else {
                        // otherwise we have causal mask
                        *v == Element::MIN
                    }
                })
            })
        });
    }

    // Testing method which, given as input the little-endian bit representations of 2 integers `x`, `y`,
    // returns 1 if x <= y, 0 otherwise. The output is computed through a multi-linear polynomial, which
    // should correspond to the MLE of the zeroifier matrix
    fn eval_lteq_poly(x_i: &[Element], y_i: &[Element]) -> Element {
        assert_eq!(x_i.len(), y_i.len());
        x_i.into_iter()
            .zip(y_i.into_iter())
            .fold(Element::from(1), |acc, (x, y)| {
                acc * (1 - x - y + 2 * x * y) + (1 - x) * y
            })
    }

    fn test_zeroifier_evaluation_for_num_heads<const NUM_HEADS_BITS: usize>() {
        // create zeroifier matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let num_heads = 1 << NUM_HEADS_BITS;

        let zeroifier = zeroifier::<Element>(num_heads, num_columns, num_columns);

        let zeroifier_heads = {
            let (it, shape) = zeroifier.slice_on_dim(0);
            it.map(|data| Tensor::new(shape.clone(), data.to_vec()))
                .collect_vec()
        };

        assert_eq!(zeroifier_heads[0].get_2d(0, 0), Element::from(1));
        assert_eq!(
            zeroifier_heads[0].get_2d(num_columns - 1, num_columns - 1),
            Element::from(1)
        );
        assert_eq!(zeroifier_heads[0].get_2d(0, 1), Element::from(0));
        assert_eq!(zeroifier_heads[0].get_2d(1, 1), Element::from(1));
        assert_eq!(zeroifier_heads[0].get_2d(1, 2), Element::from(0));

        let mle = zeroifier.to_mle_flat::<GoldilocksExt2>();

        for i in 0..num_columns {
            for j in 0..num_columns {
                for h in 0..num_heads {
                    let x_i = to_bit_sequence_le(i, NUM_BITS)
                        .map(|x| x as Element)
                        .collect_vec();
                    let y_i = to_bit_sequence_le(j, NUM_BITS)
                        .map(|x| x as Element)
                        .collect_vec();
                    let h_i = to_bit_sequence_le(h, NUM_HEADS_BITS).map(|x| x as Element);
                    // check that the zeroifier matrix is equivalent to the lteq function
                    let cmp = eval_lteq_poly(&y_i, &x_i);
                    assert_eq!(
                        zeroifier_heads[h].get_2d(i, j),
                        cmp,
                        "Zeroifier evaluation failed for ({}, {})",
                        i,
                        j
                    );
                    // build point for MLE: first column bits in little-endian order, then rows bits in little-endian order,
                    // then head bits in little-endian order
                    let point = y_i
                        .into_iter()
                        .chain(x_i)
                        .chain(h_i)
                        .map(|bit| GoldilocksExt2::from_v(bit as u64))
                        .collect_vec();
                    let eval = mle.evaluate(&point);
                    assert_eq!(eval, cmp.to_field());
                    // check that the MLE evaluation with the formula is the same as `eval`.
                    // Note that the evaluation is independent from `num_heads` dimension, as the
                    // zeroifier matrix is repeated across all the heads
                    let quick_eval =
                        eval_zeroifier_mle(&point[..NUM_BITS], &point[NUM_BITS..NUM_BITS * 2]);
                    assert_eq!(eval, quick_eval);
                }
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<GoldilocksExt2>(NUM_BITS * 2 + NUM_HEADS_BITS);
            assert_eq!(
                mle.evaluate(&point),
                eval_zeroifier_mle(&point[..NUM_BITS], &point[NUM_BITS..NUM_BITS * 2],),
            );
        }
    }

    #[test]
    fn test_zeroifier_evaluation() {
        // test with a single head
        test_zeroifier_evaluation_for_num_heads::<0>();
        // test with multiple heads
        test_zeroifier_evaluation_for_num_heads::<2>();
    }

    // Testing method which, given as input the big-endian bit representations of 2 integers `x`, `y`,
    // returns `minus_infinity` if x > y, 0 otherwise. The output is computed through a multi-linear polynomial, which
    // should correspond to the MLE of the infinitizer matrix
    fn eval_gt_poly(x_i: &[Element], y_i: &[Element], minus_infinity: Element) -> Element {
        minus_infinity * (Element::unit() - eval_lteq_poly(x_i, y_i))
    }

    fn test_infinitizer_evaluation_for_num_heads<const NUM_HEADS_BITS: usize>() {
        // create infinitizer matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let num_heads = 1 << NUM_HEADS_BITS;

        let minus_infinity = *quantization::MIN;

        let infinitizer =
            infinitizer::<Element>(num_heads, num_columns, num_columns, minus_infinity);

        let infinitizer_heads = {
            let (it, shape) = infinitizer.slice_on_dim(0);
            it.map(|data| Tensor::new(shape.clone(), data.to_vec()))
                .collect_vec()
        };

        assert_eq!(infinitizer_heads[0].get_2d(0, 0), Element::from(0));
        assert_eq!(
            infinitizer_heads[0].get_2d(num_columns - 1, num_columns - 1),
            Element::from(0)
        );
        assert_eq!(
            infinitizer_heads[0].get_2d(0, 1),
            Element::from(minus_infinity)
        );
        assert_eq!(infinitizer_heads[0].get_2d(1, 1), Element::from(0));
        assert_eq!(
            infinitizer_heads[0].get_2d(1, 2),
            Element::from(minus_infinity)
        );

        let mle = infinitizer.to_mle_flat::<GoldilocksExt2>();

        for i in 0..num_columns {
            for j in 0..num_columns {
                for h in 0..num_heads {
                    let x_i = to_bit_sequence_le(i, NUM_BITS)
                        .map(|x| x as Element)
                        .collect_vec();
                    let y_i = to_bit_sequence_le(j, NUM_BITS)
                        .map(|x| x as Element)
                        .collect_vec();
                    let h_i = to_bit_sequence_le(h, NUM_HEADS_BITS).map(|x| x as Element);
                    // check that the zeroifier matrix is equivalent to the gt function with output being minus_infinity
                    let cmp = eval_gt_poly(&y_i, &x_i, minus_infinity);
                    assert_eq!(
                        infinitizer_heads[h].get_2d(i, j),
                        cmp,
                        "Zeroifier evaluation failed for ({}, {})",
                        i,
                        j
                    );
                    // build point for MLE: first column bits in little-endian order, then rows bits in little-endian order,
                    // then head bits in little-endian order
                    let point = y_i
                        .into_iter()
                        .chain(x_i.into_iter())
                        .chain(h_i.into_iter())
                        .map(|bit| bit.to_field())
                        .collect_vec();
                    let eval = mle.evaluate(&point);
                    println!("{cmp} {} {}", cmp as u64, u64::MAX - 2);
                    assert_eq!(eval, cmp.to_field());
                    // check that the MLE evaluation with the formula is the same as `eval`.
                    // Note that the evaluation is independent from `num_heads` dimension, as the
                    // zeroifier matrix is repeated across all the heads
                    let quick_eval = eval_infinitizer_mle(
                        &point[..NUM_BITS],
                        &point[NUM_BITS..NUM_BITS * 2],
                        minus_infinity,
                    );
                    assert_eq!(eval, quick_eval);
                }
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<GoldilocksExt2>(NUM_BITS * 2 + NUM_HEADS_BITS);
            assert_eq!(
                mle.evaluate(&point),
                eval_infinitizer_mle(
                    &point[..NUM_BITS],
                    &point[NUM_BITS..NUM_BITS * 2],
                    minus_infinity,
                ),
            );
        }
    }

    #[test]
    fn test_infinitizer_evaluation() {
        // test infinitizer with a single head
        test_infinitizer_evaluation_for_num_heads::<0>();
        // test infinitizer with multiple heads
        test_infinitizer_evaluation_for_num_heads::<3>();
    }

    #[test]
    fn test_proven_mha() {
        let num_heads = 5;
        let head_dim = 7;
        let seq_len = 10;

        let hidden_size = num_heads * head_dim;

        let input_shape = Shape::new(vec![seq_len, hidden_size]);

        let mut model = Model::new_from_input_shapes(vec![input_shape; 3], PaddingMode::NoPadding);

        let mha = Mha::new(seq_len, num_heads, head_dim).unwrap();

        _ = model.add_consecutive_layer(Layer::Mha(mha), None).unwrap();
        model.route_output(None).unwrap();

        _ = prove_model(model).unwrap();
    }

    #[test]
    fn test_proven_mha_with_padding_and_unpadding() {
        init_test_logging("info");
        let num_heads = 12;
        let head_dim = 64;
        let seq_len = 1024;
        let embedding_size = 768;
        let hidden_size = num_heads * head_dim;

        let input_shape = Shape::new(vec![seq_len, embedding_size]);

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

        // we pad in QKV node
        let qkv_node_id = model
            .add_consecutive_layer(
                Layer::QKV(QKV::random(num_heads, embedding_size, hidden_size).unwrap()),
                None,
            )
            .unwrap();

        let mha_node_id = model
            .add_consecutive_layer(
                Layer::Mha(Mha::new(seq_len, num_heads, head_dim).unwrap()),
                Some(qkv_node_id),
            )
            .unwrap();

        // we add MatMul node to remove the padding garbage
        let matmul_size = 37;

        let matmul = MatMul::new(
            OperandMatrix::Input,
            OperandMatrix::new_weight_matrix(Tensor::random(
                &vec![hidden_size, matmul_size].into(),
            )),
        )
        .unwrap();

        let _mat_mul_id = model
            .add_consecutive_layer(Layer::MatMul(matmul), Some(mha_node_id))
            .unwrap();

        model.route_output(None).unwrap();

        // sample input for the model, and compute expected output
        let input = vec![Tensor::random(&input_shape)];

        let (quantized_model, quantized_input) =
            quantize_model(model.clone(), input, None).unwrap();

        let trace = quantized_model
            .run::<GoldilocksExt2>(&quantized_input)
            .unwrap();
        let outputs = trace.outputs().unwrap();

        assert_eq!(outputs.len(), 1);

        let expected_output = outputs[0].clone();

        let outputs = prove_quantized_model(quantized_model.clone(), quantized_input).unwrap();

        assert_eq!(outputs.len(), 1);

        let output = &outputs[0];

        let padded_output_shape = output.get_shape();

        for i in 0..padded_output_shape[0] {
            for j in 0..padded_output_shape[1] {
                if i < seq_len {
                    // it's a non-garbage row
                    if j < matmul_size {
                        // it's an actual entry, so we check it's the same as the unpadded output
                        assert_eq!(
                            expected_output.get_2d(i, j),
                            output.get_2d(i, j),
                            "Failed for {i} {j}"
                        );
                    } else {
                        // it's a padded entry, so check it's zero
                        assert_eq!(0, output.get_2d(i, j));
                    }
                }
            }
        }
    }

    #[test]
    fn test_removing_garbage() {
        let seq_len = 12;
        let num_heads = 5;
        let head_dim = 6;
        let hidden_size = num_heads * head_dim;

        let matmul_size = 37;

        let matrix_shape = Shape::new(vec![hidden_size, matmul_size]);

        let matrix = Tensor::<Element>::random(&matrix_shape);

        let padded_matrix_shape = matrix_shape.next_power_of_two();

        let unpadded_mha_shape = Shape::new(vec![seq_len, num_heads, head_dim]);
        let padded_mha_shape = unpadded_mha_shape.next_power_of_two();

        let padded_matrix = pad_matrix_to_ignore_mha_garbage(
            &matrix,
            &unpadded_mha_shape,
            &padded_mha_shape,
            padded_matrix_shape,
        )
        .unwrap();

        let input_shape = Shape::new(vec![seq_len, hidden_size]);
        let input = Tensor::random(&input_shape);

        let output = input.matmul(&matrix);

        println!("Matrix: {matrix:?}, padded: {padded_matrix:?}");

        let padded_num_heads = num_heads.next_power_of_two();
        let padded_head_dim = head_dim.next_power_of_two();
        let padded_input_shape = Shape::new(vec![
            seq_len.next_power_of_two(),
            padded_head_dim * padded_num_heads,
        ]);

        let mut padded_input_data = vec![Element::default(); padded_input_shape.product()];

        input.data.iter().enumerate().for_each(|(i, value)| {
            let col = i % input_shape[1];
            let row = i / input_shape[1];
            let padded_col = col / head_dim * padded_head_dim + col % head_dim;
            padded_input_data[row * padded_input_shape[1] + padded_col] = *value
        });

        let padded_input = Tensor::new(padded_input_shape, padded_input_data);

        let padded_output = padded_input.matmul(&padded_matrix);

        let padded_out_shape = padded_output.get_shape();

        for i in 0..padded_out_shape[0] {
            for j in 0..padded_out_shape[1] {
                if i < seq_len {
                    // it's a non-garbage row
                    if j < matmul_size {
                        // it's an actual entry, so we check it's the same as the unpadded output
                        assert_eq!(
                            output.get_2d(i, j),
                            padded_output.get_2d(i, j),
                            "Failed for {i} {j}"
                        );
                    } else {
                        // it's a padded entry, so check it's zero
                        assert_eq!(0, padded_output.get_2d(i, j));
                    }
                }
            }
        }
    }

    #[test]
    fn test_mha_with_real_values() -> anyhow::Result<()> {
        // let model_weights_path = json::test::get_json_file(TINY_GPT2_NAME)?;
        let debug_output_path = json::test::get_json_file(TINY_GPT2_DEBUG_NAME)?;
        // let loader = json::FileTensorLoader::new_from_path(model_weights_path)?;
        // let config = LLMConfig::from_json(&loader)?;
        // let LLMModel::GPT2(llm_model) = config.model_json(&loader)?;
        let model_path = file_cache::ensure_downloaded(GPT2_Q8_0_URL)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let config = LLMConfig::from_content(&loader)?;
        let model = config.model(&loader)?;
        let LLMModel::GPT2(llm_model) = model;
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

        let input_shape = positioned.outputs()[0].get_shape();

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

        let qkv_node_id = model
            .add_consecutive_layer(
                Layer::QKV(QKV::new(
                    llm_model.blocks[0].q.clone(),
                    llm_model.blocks[0].q_bias.clone(),
                    llm_model.blocks[0].k.clone(),
                    llm_model.blocks[0].k_bias.clone(),
                    llm_model.blocks[0].v.clone(),
                    llm_model.blocks[0].v_bias.clone(),
                    config.num_heads,
                )?),
                None,
            )
            .unwrap();

        let mha = Mha::new(config.context_length, config.num_heads, config.head_dim())?;

        let _mha_id = model
            .add_consecutive_layer(Layer::Mha(mha), Some(qkv_node_id))
            .unwrap();

        model.route_output(None).unwrap();

        let inputs = vec![positioned.outputs()[0].clone()];
        let (quantized_model, inputs) =
            quantize_model(model, inputs.clone(), Some(inputs)).unwrap();

        prove_quantized_model(quantized_model, inputs)?;

        Ok(())
    }

    #[test]
    fn test_mha_padding() {
        let seq_len = 12;
        let num_heads = 5;
        let head_dim = 6;
        let embedding_size = 727;
        let hidden_size = num_heads * head_dim;

        let input_shape = Shape::new(vec![seq_len, embedding_size]);

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

        let qkv_node_id = model
            .add_consecutive_layer(
                Layer::QKV(QKV::random(num_heads, embedding_size, hidden_size).unwrap()),
                None,
            )
            .unwrap();

        let mha = Mha::<f32>::new(seq_len, num_heads, head_dim).unwrap();

        let _mha_id = model
            .add_consecutive_layer(Layer::Mha(mha), Some(qkv_node_id))
            .unwrap();
        println!("qkv id: {qkv_node_id}");
        println!("mha id: {_mha_id}");
        model.route_output(None).unwrap();

        let inputs = vec![Tensor::random(&input_shape)];

        let (quantized_model, inputs) = quantize_model(model, inputs, None).unwrap();
        quantized_model
            .to_forward_iterator()
            .for_each(|(node_id, node)| {
                println!("node with id {node_id}, node name: {}", node.describe())
            });
        // run to get unpadded output
        let mut outputs = quantized_model
            .run::<GoldilocksExt2>(&inputs)
            .unwrap()
            .outputs()
            .unwrap()
            .into_iter()
            .cloned()
            .collect_vec();

        assert_eq!(outputs.len(), 1);

        let unpadded_out = outputs.pop().unwrap();

        // pad model
        let padded_model = pad_model(quantized_model).unwrap();

        // pad inputs
        let padded_inputs = padded_model.prepare_inputs(inputs).unwrap();

        // compute padded evaluation, with garbage removal in matmul
        let mut outputs = padded_model
            .run::<GoldilocksExt2>(&padded_inputs)
            .unwrap()
            .outputs()
            .unwrap()
            .into_iter()
            .cloned()
            .collect_vec();

        assert_eq!(outputs.len(), 1);

        let padded_out = outputs.pop().unwrap();

        // check that non-garbabe entries in padded output are the same as corresponding entries
        // in unpadded_out, i.e., the padding didn't affect the results of MHA
        let padded_out_shape = padded_out.get_shape();

        let padded_head_dim = head_dim.next_power_of_two();
        for i in 0..padded_out_shape[0] {
            for j in 0..padded_out_shape[1] {
                if i < seq_len
                    && j % padded_head_dim < head_dim
                    && j / padded_head_dim * head_dim < num_heads
                {
                    let original_matrix_index =
                        j / padded_head_dim * head_dim + j % padded_head_dim;
                    assert_eq!(
                        unpadded_out.get_2d(i, original_matrix_index),
                        padded_out.get_2d(i, j),
                        "Failed for {i} {j} {original_matrix_index}"
                    );
                }
            }
        }
    }
}
