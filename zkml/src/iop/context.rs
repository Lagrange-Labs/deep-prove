use crate::{
    Element,
    iop::precommit::{self, PolyID},
    layers::LayerCtx,
    lookup::context::{LookupContext, TableType},
    model::Model,
};
use anyhow::Context as CC;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::BasefoldCommitment;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeSet;
use tracing::{debug, info, trace};
use transcript::Transcript;

/// Info related to the lookup protocol tables.
/// Here `poly_id` is the multiplicity poly for this table.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct TableCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub poly_id: PolyID,
    pub num_vars: usize,
    pub table_commitment: BasefoldCommitment<E>,
}

pub const RESHAPE_FS_ID: u64 = 0xdeadbeef;

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: Vec<LayerCtx<E>>,
    /// Context related to the commitment and accumulation of claims related to the weights of model.
    /// This part contains the commitment of the weights.
    pub weights: precommit::Context<E>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
    /// unpadded shape of the first initial input
    pub unpadded_input_shape: Vec<usize>,
}

/// Similar to the InferenceStep but only records the input and output shapes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeStep {
    pub unpadded_input_shape: Vec<usize>,
    pub unpadded_output_shape: Vec<usize>,
    pub padded_input_shape: Vec<usize>,
    pub padded_output_shape: Vec<usize>,
}

impl ShapeStep {
    pub fn new(
        unpadded_input: Vec<usize>,
        padded_input: Vec<usize>,
        unpadded_output: Vec<usize>,
        padded_output: Vec<usize>,
    ) -> ShapeStep {
        Self {
            unpadded_input_shape: unpadded_input,
            padded_input_shape: padded_input,
            unpadded_output_shape: unpadded_output,
            padded_output_shape: padded_output,
        }
    }
    pub fn next_step(
        last_step: &ShapeStep,
        unpadded_output: Vec<usize>,
        padded_output: Vec<usize>,
    ) -> ShapeStep {
        ShapeStep {
            unpadded_input_shape: last_step.unpadded_output_shape.clone(),
            unpadded_output_shape: unpadded_output,
            padded_input_shape: last_step.padded_output_shape.clone(),
            padded_output_shape: padded_output,
        }
    }
}

/// Auxiliary information for the context creation
#[derive(Clone, Debug)]
pub(crate) struct ContextAux {
    pub tables: BTreeSet<TableType>,
    pub last_output_shape: Vec<usize>,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Generates a context to give to the verifier that contains informations about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    pub fn generate(
        model: &Model<Element>,
        input_shape: Option<Vec<usize>>,
    ) -> anyhow::Result<Self> {
        let tables = BTreeSet::new();
        let last_output_shape = if let Some(shape) = input_shape {
            shape
        } else {
            model.input_shape()
        };
        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape,
        };
        let mut step_infos = Vec::with_capacity(model.layer_count());
        debug!("Context : layer info generation ...");
        for (id, layer) in model.layers() {
            trace!(
                "Context : {}-th layer {}info generation ...",
                id,
                layer.describe()
            );
            let (info, new_aux) = layer.step_info(id, ctx_aux);
            step_infos.push(info);
            ctx_aux = new_aux;
        }
        info!(
            "step_infos: {:?}",
            step_infos.iter().map(|x| x.variant_name()).join(", ")
        );
        debug!("Context : commitment generating ...");
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        debug!("Context : lookup generation ...");
        let lookup_ctx = LookupContext::new(&ctx_aux.tables);
        Ok(Self {
            steps_info: step_infos.into_iter().rev().collect_vec(),
            weights: commit_ctx,
            lookup: lookup_ctx,
            unpadded_input_shape: model.unpadded_input_shape(),
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for steps in self.steps_info.iter() {
            match steps {
                LayerCtx::Dense(info) => {
                    t.append_field_element(&E::BaseField::from(info.matrix_poly_id as u64));
                    info.matrix_poly_aux.write_to_transcript(t);
                }
                LayerCtx::Requant(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                LayerCtx::Activation(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                LayerCtx::Pooling(info) => {
                    t.append_field_element(&E::BaseField::from(info.poolinfo.kernel_size as u64));
                    t.append_field_element(&E::BaseField::from(info.poolinfo.stride as u64));
                }
                LayerCtx::Table(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                    t.append_field_elements(info.table_commitment.root().0.as_slice());
                }
                LayerCtx::Convolution(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.bias_poly_id as u64));

                    for i in 0..info.delegation_fft.len() {
                        info.delegation_fft[i].write_to_transcript(t);
                    }
                    for i in 0..info.delegation_ifft.len() {
                        info.delegation_ifft[i].write_to_transcript(t);
                    }
                    info.fft_aux.write_to_transcript(t);
                    info.ifft_aux.write_to_transcript(t);
                    info.hadamard.write_to_transcript(t);
                }
                LayerCtx::SchoolBookConvolution(_info) => {}
                LayerCtx::Reshape => {
                    t.append_field_element(&E::BaseField::from(RESHAPE_FS_ID as u64));
                }
            }
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}
