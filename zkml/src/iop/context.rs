use crate::{
    Element,
    commit::context::{CommitmentContext, PolyId},
    layers::provable::{Node, NodeCtx, NodeId, OpInfo},
    lookup::context::{LookupContext, TableType},
    model::{Model, ModelCtx, ToIterator},
    tensor::Shape,
    to_base,
};
use anyhow::{anyhow, ensure};
use ff_ext::ExtensionField;
use mpcs::{BasefoldCommitment, PolynomialCommitmentScheme};
use multilinear_extensions::{mle::DenseMultilinearExtension, util::ceil_log2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use tracing::{debug, trace};
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
    pub num_vars: usize,
    pub table_commitment: BasefoldCommitment<E>,
}

pub const RESHAPE_FS_ID: u64 = 0xdeadbeef;

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: ModelCtx<E>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentContext<E, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
    /// unpadded shape of the first initial input
    pub unpadded_input_shapes: Vec<Shape>,
}

/// Similar to the InferenceStep but only records the input and output shapes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeStep {
    pub unpadded_input_shape: Vec<Shape>,
    pub unpadded_output_shape: Vec<Shape>,
    pub padded_input_shape: Vec<Shape>,
    pub padded_output_shape: Vec<Shape>,
}

impl ShapeStep {
    pub fn new(
        unpadded_input: Vec<Shape>,
        padded_input: Vec<Shape>,
        unpadded_output: Vec<Shape>,
        padded_output: Vec<Shape>,
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
        unpadded_output: Vec<Shape>,
        padded_output: Vec<Shape>,
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
pub struct ContextAux {
    pub tables: BTreeSet<TableType>,
    pub last_output_shape: Vec<Shape>,
    pub model_polys: Option<HashMap<PolyId, Vec<Element>>>,
    /// THis field is only used in macro layers like MHA
    pub max_poly_len: usize,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Context<E, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Generates a context to give to the verifier that contains information about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    pub fn generate(
        model: &Model<Element>,
        input_shapes: Option<Vec<Shape>>,
        params: Option<(
            <PCS as PolynomialCommitmentScheme<E>>::ProverParam,
            <PCS as PolynomialCommitmentScheme<E>>::VerifierParam,
        )>,
    ) -> anyhow::Result<Self> {
        let tables = BTreeSet::new();
        let input_shapes = if let Some(shape) = input_shapes {
            shape
        } else {
            model.input_shapes.clone()
        };
        let mut max_poly_len = input_shapes
            .iter()
            .fold(0usize, |acc, shapes| acc.max(shapes.product()));

        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape: input_shapes.clone(),
            model_polys: None,
            max_poly_len,
        };

        // If we already have params loaded from store, we only need to
        // construct the commitment context from node shapes.
        // Otherwise, we also need generate the params for which we need to find
        // the largest polynomial.
        let (step_infos, commitment_ctx, lookup) = match params {
            Some((prover_params, verifier_params)) => {
                let mut model_polys =
                    Vec::<(NodeId, HashMap<PolyId, DenseMultilinearExtension<E>>)>::new();
                let mut step_infos = BTreeMap::new();
                let mut shapes: HashMap<NodeId, Vec<Shape>> = HashMap::new();
                debug!("Context : layer info generation ...");
                for (id, node) in model.to_forward_iterator() {
                    ctx_aux = compute_node_shape::<E>(
                        ctx_aux,
                        &mut model_polys,
                        &mut step_infos,
                        &mut shapes,
                        &input_shapes,
                        id,
                        node,
                    )?;
                }

                debug!("Context : lookup generation ...");
                let lookup_ctx = LookupContext::new(&ctx_aux.tables);

                debug!("Context : commitment generating ...");
                let commitment_ctx = CommitmentContext::<E, PCS>::new_with_params(
                    model_polys,
                    &lookup_ctx,
                    prover_params,
                    verifier_params,
                )?;
                (step_infos, commitment_ctx, lookup_ctx)
            }
            None => {
                let mut model_polys =
                    Vec::<(NodeId, HashMap<PolyId, DenseMultilinearExtension<E>>)>::new();
                let mut step_infos = BTreeMap::new();
                let mut shapes: HashMap<NodeId, Vec<Shape>> = HashMap::new();
                debug!("Context : layer info generation ...");
                for (id, node) in model.to_forward_iterator() {
                    ctx_aux = compute_node_shape::<E>(
                        ctx_aux,
                        &mut model_polys,
                        &mut step_infos,
                        &mut shapes,
                        &input_shapes,
                        id,
                        node,
                    )?;
                    max_poly_len = max_poly_len.max(ctx_aux.max_poly_len);
                }
                // Check to see if we use a lookup table alrger than any of the individual polynomials
                ctx_aux.tables.iter().for_each(|table_type| {
                    let multiplicity_vars = table_type.multiplicity_poly_vars();
                    max_poly_len = max_poly_len.max(1 << multiplicity_vars)
                });

                debug!("Context : lookup generation ...");
                let lookup_ctx = LookupContext::new(&ctx_aux.tables);

                debug!("Context : commitment generating ...");
                let commitment_ctx =
                    CommitmentContext::<E, PCS>::new(max_poly_len, model_polys, &lookup_ctx)?;
                (step_infos, commitment_ctx, lookup_ctx)
            }
        };

        Ok(Self {
            steps_info: ModelCtx { nodes: step_infos },
            commitment_ctx,
            lookup,
            unpadded_input_shapes: model.unpadded_input_shapes(),
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        self.commitment_ctx.write_to_transcript(t)?;
        Ok(())
    }
}

fn compute_node_shape<E: ExtensionField>(
    mut ctx_aux: ContextAux,
    model_polys: &mut Vec<(NodeId, HashMap<PolyId, DenseMultilinearExtension<E>>)>,
    step_infos: &mut BTreeMap<NodeId, NodeCtx<E>>,
    shapes: &mut HashMap<NodeId, Vec<Shape>>,
    input_shapes: &[Shape],
    id: usize,
    node: &Node<Element>,
) -> anyhow::Result<ContextAux> {
    trace!(
        "Context : {}-th layer {}info generation ...",
        id,
        node.operation.describe()
    );
    trace!(
        "Generating context node with id {id}: {:?}",
        node.describe()
    );
    // compute input shapes for this node
    let node_input_shapes = node
        .inputs
        .iter()
        .map(|edge| {
            Ok(if let Some(node_id) = &edge.node {
                let node_shapes = shapes.get(node_id).ok_or(anyhow!(
                    "Node {} not found in set of previous shapes",
                    node_id
                ))?;
                ensure!(
                    edge.index < node_shapes.len(),
                    "Input for node {} is coming from output {} of node {}, 
                        but this node has only {} outputs",
                    id,
                    edge.index,
                    node_id,
                    node_shapes.len()
                );
                node_shapes[edge.index].clone()
            } else {
                // input node
                ensure!(
                    edge.index < input_shapes.len(),
                    "Input for node {} is the input {} of the model, 
                        but the model has only {} inputs",
                    id,
                    edge.index,
                    input_shapes.len()
                );
                input_shapes[edge.index].clone()
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    ctx_aux.last_output_shape = node_input_shapes;
    let (info, mut new_aux) = node.step_info(id, ctx_aux)?;
    // Retrieve any model polynomials that need to be committed
    if new_aux.model_polys.is_some() {
        model_polys.push((
            id,
            new_aux
                .model_polys
                .as_mut()
                .unwrap()
                .drain()
                .map(|(poly_id, evals)| {
                    let num_vars = ceil_log2(evals.len());
                    (
                        poly_id,
                        DenseMultilinearExtension::<E>::from_evaluations_vec(
                            num_vars,
                            to_base::<E, _>(evals),
                        ),
                    )
                })
                .collect::<HashMap<PolyId, DenseMultilinearExtension<E>>>(),
        ));
    }
    step_infos.insert(
        id,
        NodeCtx {
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            ctx: info,
        },
    );
    shapes.insert(id, new_aux.last_output_shape.clone());
    Ok(new_aux)
}
