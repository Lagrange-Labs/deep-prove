use std::{
    array,
    collections::{BTreeMap, HashMap},
};

use ff_ext::ExtensionField;
use goldilocks::SmallField;
use multilinear_extensions::virtual_poly::ArcMultilinearExtension;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{DeserializeOwned, Error},
};
use simple_frontend::structs::{CellId, ChallengeConst, ConstantType, LayerId};

pub(crate) type SumcheckProof<F> = sumcheck::structs::IOPProof<F>;

/// A point is a vector of num_var length
pub type Point<F> = Vec<F>;

/// A point and the evaluation of this point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointAndEval<F> {
    pub point: Point<F>,
    pub eval: F,
}

impl<E: ExtensionField> Default for PointAndEval<E> {
    fn default() -> Self {
        Self {
            point: vec![],
            eval: E::ZERO,
        }
    }
}

impl<F: Clone> PointAndEval<F> {
    /// Construct a new pair of point and eval.
    /// Caller gives up ownership
    pub fn new(point: Point<F>, eval: F) -> Self {
        Self { point, eval }
    }

    /// Construct a new pair of point and eval.
    /// Performs deep copy.
    pub fn new_from_ref(point: &Point<F>, eval: &F) -> Self {
        Self {
            point: (*point).clone(),
            eval: eval.clone(),
        }
    }
}

/// Represent the prover state for each layer in the IOP protocol. To support
/// gates between non-adjacent layers, we leverage the techniques in
/// [Virgo++](https://eprint.iacr.org/2020/1247).
pub struct IOPProverState<E: ExtensionField> {
    pub(crate) layer_id: LayerId,
    /// Evaluations to the next phase.
    pub(crate) to_next_phase_point_and_evals: Vec<PointAndEval<E>>,
    /// Evaluations of subsets from layers __closer__ to the output.
    /// __closer__ as in the layer that the subset elements lie in has not been processed.
    ///
    /// LayerId is the layer id of the incoming subset point and evaluation.
    pub(crate) subset_point_and_evals: Vec<Vec<(LayerId, PointAndEval<E>)>>,

    /// The point to the next step.
    pub(crate) to_next_step_point: Point<E>,

    // Especially for output phase1.
    pub(crate) assert_point: Point<E>,
}

/// Represent the verifier state for each layer in the IOP protocol.
pub struct IOPVerifierState<E: ExtensionField> {
    pub(crate) layer_id: LayerId,
    /// Evaluations from the next layer.
    pub(crate) to_next_phase_point_and_evals: Vec<PointAndEval<E>>,
    /// Evaluations of subsets from layers closer to the output. LayerId is the
    /// layer id of the incoming subset point and evaluation.
    pub(crate) subset_point_and_evals: Vec<Vec<(LayerId, PointAndEval<E>)>>,

    pub(crate) challenges: HashMap<ChallengeConst, Vec<E::BaseField>>,
    pub(crate) instance_num_vars: usize,

    pub(crate) to_next_step_point_and_eval: PointAndEval<E>,

    // Especially for output phase1.
    pub(crate) assert_point: Point<E>,
    // Especially for phase2.
    pub(crate) out_point: Point<E>,
    pub(crate) eq_y_ry: Vec<E>,
    pub(crate) eq_x1_rx1: Vec<E>,
    pub(crate) eq_x2_rx2: Vec<E>,
}

#[allow(clippy::too_long_first_doc_paragraph)]
/// Phase 1 is a sumcheck protocol merging the subset evaluations from the
/// layers closer to the circuit output to an evaluation to the output of the
/// current layer.
/// Phase 2 is several sumcheck protocols (depending on the degree of gates),
/// reducing the correctness of the output of the current layer to the input of
/// the current layer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct IOPProverStepMessage<E: ExtensionField> {
    /// Sumcheck proofs for each sumcheck protocol.
    pub sumcheck_proof: SumcheckProof<E>,
    pub sumcheck_eval_values: Vec<E>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct IOPProof<E: ExtensionField> {
    pub sumcheck_proofs: Vec<IOPProverStepMessage<E>>,
}

/// Represent the point at the final step and the evaluations of the subsets of
/// the input layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct GKRInputClaims<E: ExtensionField> {
    pub point_and_evals: Vec<PointAndEval<E>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum SumcheckStepType {
    OutputPhase1Step1,
    Phase1Step1,
    Phase2Step1,
    Phase2Step2,
    Phase2Step2NoStep3,
    Phase2Step3,
    LinearPhase2Step1,
    InputPhase2Step1,
    Undefined,
}

pub(crate) enum Step {
    Step1 = 0,
    Step2,
    Step3,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Layer<E: ExtensionField> {
    pub(crate) layer_id: u32,
    pub(crate) sumcheck_steps: Vec<SumcheckStepType>,
    pub(crate) num_vars: usize,

    // Gates. Should be all None if it's the input layer.
    pub(crate) add_consts: Vec<GateCIn<ConstantType<E>>>,
    pub(crate) adds: Vec<Gate1In<ConstantType<E>>>,
    pub(crate) adds_fanin_mapping: [BTreeMap<CellId, Vec<Gate1In<ConstantType<E>>>>; 1], /* grouping for 1 fanins */
    pub(crate) mul2s: Vec<Gate2In<ConstantType<E>>>,
    pub(crate) mul2s_fanin_mapping: [BTreeMap<CellId, Vec<Gate2In<ConstantType<E>>>>; 2], /* grouping for 2 fanins */
    pub(crate) mul3s: Vec<Gate3In<ConstantType<E>>>,
    pub(crate) mul3s_fanin_mapping: [BTreeMap<CellId, Vec<Gate3In<ConstantType<E>>>>; 3], /* grouping for 3 fanins */

    /// The corresponding wires copied from this layer to later layers. It is
    /// (later layer id -> current wire id to be copied). It stores the non-zero
    /// entry of copy_to[layer_id] for each row.
    pub(crate) copy_to: BTreeMap<LayerId, Vec<CellId>>,
    /// The corresponding wires from previous layers pasted to this layer. It is
    /// (shallower layer id -> pasted to the current id). It stores the non-zero
    /// entry of paste_from[layer_id] for each column. Undefined for the input.
    pub(crate) paste_from: BTreeMap<LayerId, Vec<CellId>>,
    /// Maximum size of the subsets pasted from the previous layers, rounded up
    /// to the next power of two. This is the logarithm of the rounded size.
    /// Undefined for the input layer.
    pub(crate) max_previous_num_vars: usize,
}

impl<E: ExtensionField> Default for Layer<E> {
    fn default() -> Self {
        Layer::<E> {
            layer_id: 0,
            sumcheck_steps: vec![],
            add_consts: vec![],
            adds: vec![],
            adds_fanin_mapping: [BTreeMap::new(); 1],
            mul2s: vec![],
            mul2s_fanin_mapping: array::from_fn(|_| BTreeMap::new()),
            mul3s: vec![],
            mul3s_fanin_mapping: array::from_fn(|_| BTreeMap::new()),
            copy_to: BTreeMap::new(),
            paste_from: BTreeMap::new(),
            num_vars: 0,
            max_previous_num_vars: 0,
        }
    }
}

#[derive(Clone, Serialize, Default, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Circuit<E: ExtensionField> {
    pub layers: Vec<Layer<E>>,

    pub n_witness_in: usize,
    pub n_witness_out: usize,
    /// The endpoints in the input layer copied from each input witness.
    pub paste_from_wits_in: Vec<(CellId, CellId)>,
    /// The endpoints in the input layer copied from counter.
    pub paste_from_counter_in: Vec<(usize, (CellId, CellId))>,
    /// The endpoints in the input layer copied from constants
    pub paste_from_consts_in: Vec<(i64, (CellId, CellId))>,
    /// The wires copied to the output witness
    pub copy_to_wits_out: Vec<Vec<CellId>>,
    pub assert_consts: Vec<GateCIn<ConstantType<E>>>,
    pub max_wit_in_num_vars: Option<usize>,
}

pub type GateCIn<C> = Gate<C, 0>;
pub type Gate1In<C> = Gate<C, 1>;
pub type Gate2In<C> = Gate<C, 2>;
pub type Gate3In<C> = Gate<C, 3>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Macro struct for Gate
pub struct Gate<C, const FAN_IN: usize> {
    #[serde(
        serialize_with = "serialize_array",
        deserialize_with = "deserialize_array"
    )]
    pub(crate) idx_in: [CellId; FAN_IN],
    pub(crate) idx_out: CellId,
    pub(crate) scalar: C,
}

fn serialize_array<T: Clone + Serialize, S: Serializer, const N: usize>(
    input: &[T; N],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    input.to_vec().serialize(serializer)
}

fn deserialize_array<'a, T: Deserialize<'a>, D: Deserializer<'a>, const N: usize>(
    deserializer: D,
) -> Result<[T; N], D::Error> {
    Vec::<T>::deserialize(deserializer)?
        .try_into()
        .map_err(|_| D::Error::custom("failed to deserialize array: wrong number of items found"))
}

#[derive(Clone)]
pub struct CircuitWitness<'a, E: ExtensionField> {
    /// Three vectors denote 1. layer_id, 2. instance_id || wire_id.
    pub(crate) layers: Vec<ArcMultilinearExtension<'a, E>>,
    /// Three vectors denote 1. wires_in id, 2. instance_id || wire_id.
    pub(crate) witness_in: Vec<ArcMultilinearExtension<'a, E>>,
    /// Three vectors denote 1. wires_out id, 2. instance_id || wire_id.
    pub(crate) witness_out: Vec<ArcMultilinearExtension<'a, E>>,
    /// Challenges
    pub(crate) challenges: HashMap<ChallengeConst, Vec<E::BaseField>>,
    /// The number of instances for the same sub-circuit.
    pub(crate) n_instances: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LayerWitness<F: SmallField> {
    pub instances: Vec<LayerInstanceWit<F>>,
}

pub type LayerInstanceWit<F> = Vec<F>;
