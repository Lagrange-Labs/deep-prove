use anyhow::anyhow;
use ark_std::rand::thread_rng;
use ff::Field;
use ff_ext::ExtensionField;
use gkr::{
    structs::{Circuit, CircuitWitness, IOPProof, IOPProverState, IOPVerifierState, PointAndEval},
    util::ceil_log2,
};
use goldilocks::SmallField;
use mpcs::{BasefoldCommitment, PolynomialCommitmentScheme};
use poseidon::{digest::Digest, poseidon_hash::hash_n_to_hash_no_pad};
use std::collections::{BTreeMap, HashMap, HashSet};
use utils::compute_multiplicity_poly;

use crate::{
    Claim, Element,
    activation::{Activation, Relu},
    commit::Pcs,
    iop::context::{StepInfo, TableInfo},
    model::{InferenceTrace, Layer, StepIdx},
    quantization::{Fieldizer, Requant},
};
use gkr_circuits::{
    lookups_circuit::lookup_wire_fractional_sumcheck, table_circuit::table_fractional_sumcheck,
};
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

pub mod gkr_circuits;
pub mod logup;
pub mod utils;

pub use logup::LogUp;

type MLE<E> = DenseMultilinearExtension<E>;

const NO_LOOKUP_DOM_SEP: &'static [u8] = b"";
const RANGE_CHECK_DOM_SEP: &'static [u8] = b"range_check";
const RELU_DOM_SEP: &'static [u8] = b"relu";
const FINAL_TABLE_DOM_SEP: &'static [u8] = b"";

/// Proof from a GKR based lookup.
/// The commitment is a batch commitment to all of the witness wires.
/// When this is a "normal" lookup proof (not the final table) this is a batch commitment to all of the inputs
/// if its the final table proof its just a commitment to the multiplicity poly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// one commitment for all columns in the lookups
    commitment: BasefoldCommitment<E>,
    /// Claims about the witness polynomials
    claims: Vec<Claim<E>>,
    /// The actual GKR proof
    gkr_proof: IOPProof<E>,
    /// numerators for all the fractional sumchecks, in the same order as the claims for witness polys
    numerators: Vec<E>,
    /// denominators for all the fractional sumchecks, in the same order as the claims for witness polys
    denominators: Vec<E>,
    /// the challenges to be used for this lookup, verified at the very end
    challenges: Vec<E>,
    /// The number of instances of this circuit this proof is for
    instance_num_vars: usize,
}
impl<E: ExtensionField> Proof<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Retireve the claims about the input columns
    pub fn claims(&self) -> &[Claim<E>] {
        &self.claims
    }

    /// Retrieves the challenges used in this proof
    pub fn challenges(&self) -> &[E] {
        &self.challenges
    }

    /// Retrieves the [`BasefoldCommitment`] digest
    pub fn get_digest(&self) -> Digest<E::BaseField> {
        self.commitment.root()
    }

    /// Retrieves the numerators
    pub fn numerators(&self) -> Vec<E> {
        self.numerators.clone()
    }

    /// Retrieves the denominators
    pub fn denominators(&self) -> Vec<E> {
        self.denominators.clone()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierClaims<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    claims: Vec<Claim<E>>,
    commitment: BasefoldCommitment<E>,
    numerators: Vec<E>,
    denominators: Vec<E>,
}

impl<E: ExtensionField> VerifierClaims<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn claims(&self) -> &[Claim<E>] {
        &self.claims
    }

    pub fn commitment(&self) -> &BasefoldCommitment<E> {
        &self.commitment
    }

    pub fn numerators(&self) -> &[E] {
        &self.numerators
    }

    pub fn denominators(&self) -> &[E] {
        &self.denominators
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Copy)]
pub enum LookupType {
    Relu(usize),
    Requant(Requant, usize),
    RequantTable(usize),
    ReluTable,
    NoLookup,
}

impl<E: ExtensionField> From<&StepInfo<E>> for LookupType
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn from(value: &StepInfo<E>) -> LookupType {
        match value {
            StepInfo::Requant(info) => LookupType::Requant(info.requant, info.num_vars),
            StepInfo::Activation(info) => LookupType::Relu(info.num_vars),
            _ => LookupType::NoLookup,
        }
    }
}

impl LookupType {
    pub fn make_circuit<E: ExtensionField>(&self) -> Circuit<E> {
        match self {
            LookupType::Requant(.., num_vars) => lookup_wire_fractional_sumcheck(1, *num_vars),
            LookupType::Relu(num_vars) => lookup_wire_fractional_sumcheck(2, *num_vars),
            LookupType::ReluTable => table_fractional_sumcheck(2, 8),
            LookupType::RequantTable(num_vars) => table_fractional_sumcheck(1, *num_vars),
            LookupType::NoLookup => Circuit::<E>::default(),
        }
    }

    pub fn number_of_columns(&self) -> usize {
        match self {
            LookupType::Requant(info, ..) => {
                let max_value = info.range >> 1;
                let max_bit_range = info.range << 1;
                let maximum_range = max_bit_range + max_value;

                ((ceil_log2(maximum_range) - info.right_shift - 1) / ceil_log2(info.after_range))
                    + 2
            }
            LookupType::Relu(..) => 2,
            LookupType::ReluTable => 2,
            LookupType::RequantTable(..) => 1,
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_columns_per_instance(&self) -> usize {
        match self {
            LookupType::Requant(..) => 1,
            LookupType::Relu(..) => 2,
            LookupType::ReluTable => 3,
            LookupType::RequantTable(..) => 2,
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_vars(&self) -> usize {
        match self {
            LookupType::Requant(.., num_vars) => *num_vars,
            LookupType::Relu(num_vars) => *num_vars,
            LookupType::ReluTable => 8,
            LookupType::RequantTable(num_vars) => *num_vars,
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_witness_mles(&self) -> usize {
        match self {
            LookupType::ReluTable => 3,
            LookupType::RequantTable(..) => 2,
            _ => self.number_of_columns(),
        }
    }

    pub fn get_dom_sep(&self) -> &[u8] {
        match &self {
            LookupType::NoLookup => NO_LOOKUP_DOM_SEP,
            LookupType::Requant(..) | LookupType::RequantTable(..) => RANGE_CHECK_DOM_SEP,
            LookupType::Relu(..) | LookupType::ReluTable => RELU_DOM_SEP,
        }
    }

    pub fn name(&self) -> String {
        match self {
            LookupType::Requant(info, ..) => {
                format!("Requant_{}", info.after_range.ilog2() as usize)
            }
            LookupType::Relu(..) | LookupType::ReluTable => "Relu".to_string(),
            LookupType::RequantTable(num_vars) => format!("Requant_{}", *num_vars),
            LookupType::NoLookup => "NoLookup".to_string(),
        }
    }

    pub fn get_mles<E: ExtensionField>(&self) -> Option<Vec<Vec<E::BaseField>>> {
        match self {
            LookupType::ReluTable => {
                let (relu_in, relu_out) = Relu::to_mle::<E>();
                let base_in = relu_in
                    .into_iter()
                    .map(|val| val.as_bases()[0])
                    .collect::<Vec<E::BaseField>>();
                let base_out = relu_out
                    .into_iter()
                    .map(|val| val.as_bases()[0])
                    .collect::<Vec<E::BaseField>>();
                Some(vec![base_in, base_out])
            }
            LookupType::RequantTable(num_vars) => {
                let requant_info = Requant {
                    range: 0,
                    right_shift: 0,
                    after_range: 1 << num_vars,
                };
                let mle = requant_info.to_mle::<E>();
                Some(vec![
                    mle.into_iter()
                        .map(|val| val.as_bases()[0])
                        .collect::<Vec<E::BaseField>>(),
                ])
            }
            _ => None,
        }
    }

    pub fn is_table(&self) -> bool {
        match self {
            LookupType::ReluTable | &LookupType::RequantTable(..) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context<E: ExtensionField> {
    circuits: HashMap<StepIdx, (LookupType, Circuit<E>)>,
}

#[derive(Clone, Default)]
pub struct WitnessContext<'a, E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub current_step: StepIdx,
    witness_storage: BTreeMap<StepIdx, LookupProverInfo<'a, E>>,
}

#[derive(Clone)]
pub struct LookupProverInfo<'a, E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub lookup_type: LookupType,
    pub batch_commitment: BasefoldCommitment<E>,
    pub circuit_witness: CircuitWitness<'a, E>,
    pub challenges: Vec<E>,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: DeserializeOwned,
{
    /// Getter for the lookup circuit depending on the [`StepIdx`]
    pub fn get_circuit(&self, step_idx: StepIdx) -> anyhow::Result<&Circuit<E>> {
        self.circuits
            .get(&step_idx)
            .and_then(|(_, circuit)| Some(circuit))
            .ok_or(anyhow::anyhow!(
                "Context does not contain a circuit for the step: {:?}",
                step_idx
            ))
    }

    /// Getter for the lookup circuit and type depending on the [`StepIdx`]
    pub fn get_circuit_and_type(
        &self,
        step_idx: StepIdx,
    ) -> anyhow::Result<&(LookupType, Circuit<E>)> {
        self.circuits.get(&step_idx).ok_or(anyhow::anyhow!(
            "Context does not contain a circuit for the step: {:?}",
            step_idx
        ))
    }

    /// Generate [`Context`] from a [ModelContext](`crate::iop::context::Context`)
    pub fn generate(steps_info: &mut Vec<StepInfo<E>>) -> anyhow::Result<Context<E>> {
        let mut requant_sizes = HashSet::<usize>::new();
        let mut requant_table_sizes = vec![];
        let mut uses_relu = false;

        // In the context the step infos go backwards when you iterate over them so we need to store the max length
        let num_steps = steps_info.len();

        let mut info = steps_info
            .iter()
            .enumerate()
            .filter_map(|(stp_idx, step)| match step {
                StepInfo::Dense(..) => None,
                StepInfo::Activation(..) => {
                    let lookup_type = LookupType::from(step);
                    let circuit = lookup_type.make_circuit::<E>();
                    uses_relu = true;
                    Some((num_steps - 1 - stp_idx, (lookup_type, circuit)))
                }
                StepInfo::Requant(info) => {
                    if requant_sizes.insert(info.requant.after_range) {
                        requant_table_sizes.push(info.requant.after_range.ilog2() as usize);
                    }

                    let lookup_type = LookupType::from(step);
                    let circuit = lookup_type.make_circuit::<E>();

                    Some((num_steps - 1 - stp_idx, (lookup_type, circuit)))
                }
                _ => unreachable!(),
            })
            .collect::<Vec<(StepIdx, (LookupType, Circuit<E>))>>();

        let largest_table = requant_table_sizes.iter().max().and_then(|value| {
            if uses_relu {
                Some(std::cmp::max(*value, 8))
            } else {
                Some(*value)
            }
        });

        let max_vars = if let Some(max_vars) = largest_table {
            max_vars
        } else {
            // If the option returned None then there are no Lookups
            return Ok(Context {
                circuits: HashMap::default(),
            });
        };

        let params = Pcs::<E>::setup(1 << max_vars)?;

        requant_table_sizes
            .iter()
            .enumerate()
            .try_for_each(|(j, &num_vars)| {
                let idx = num_steps + j;
                let lookup_type = LookupType::RequantTable(num_vars);
                let circuit = lookup_type.make_circuit::<E>();
                let mles = lookup_type
                    .get_mles::<E>()
                    .ok_or(anyhow!(
                        "Got none table LookupType when only tables should be made: {:?}",
                        lookup_type
                    ))?
                    .into_iter()
                    .map(|evaluations| {
                        DenseMultilinearExtension::<E>::from_evaluations_vec(num_vars, evaluations)
                    })
                    .collect::<Vec<DenseMultilinearExtension<E>>>();
                let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                let commit = Pcs::<E>::batch_commit(&pp, &mles)?.to_commitment();
                let table_step = StepInfo::<E>::Table(TableInfo {
                    poly_id: idx,
                    num_vars,
                    table_commitment: commit,
                });
                steps_info.push(table_step);
                info.push((idx, (lookup_type, circuit)));
                Result::<(), anyhow::Error>::Ok(())
            })?;

        let final_idx = num_steps + requant_table_sizes.len();

        let final_lookup_type = LookupType::ReluTable;
        let mles = final_lookup_type
            .get_mles::<E>()
            .ok_or(anyhow!(
                "Got none table LookupType when only tables should be made: {:?}",
                final_lookup_type
            ))?
            .into_iter()
            .map(|evaluations| DenseMultilinearExtension::<E>::from_evaluations_vec(8, evaluations))
            .collect::<Vec<DenseMultilinearExtension<E>>>();
        let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << 8)?;
        let commit = Pcs::<E>::batch_commit(&pp, &mles)?.to_commitment();
        let table_step = StepInfo::<E>::Table(TableInfo {
            poly_id: final_idx,
            num_vars: 8,
            table_commitment: commit,
        });
        steps_info.push(table_step);
        let final_circuit = final_lookup_type.make_circuit::<E>();
        info.push((final_idx, (final_lookup_type, final_circuit)));

        let circuits = HashMap::<StepIdx, (LookupType, Circuit<E>)>::from_iter(info.into_iter());

        Ok(Context { circuits })
    }
}

impl<'a, E: ExtensionField> WitnessContext<'a, E>
where
    E: Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Initialises [`WitnessContext`] from an [`InferenceTrace`] and a [`Context`]
    pub fn initialise_witness_ctx<'b>(
        ctx: &'b Context<E>,
        trace: &InferenceTrace<E>,
    ) -> anyhow::Result<(WitnessContext<'a, E>, Vec<(usize, Vec<E>)>)>
    where
        'b: 'a,
    {
        // First we quickly iterate through all the steps to find the largest number of variables that will be used as input to a lookup

        // we also pick some random field element to compute "merged" lookups so that we can construct the multiplicity poly
        let tmp_const_challenge = E::random(thread_rng());
        let tmp_relu_challenge = E::random(thread_rng());
        let tmp_relu_challenges = vec![tmp_relu_challenge, tmp_relu_challenge * tmp_relu_challenge];
        // We would have a different requant lookup table for each different `requant.after_range`.
        // We make this  BTreeMap so we order them in increasing table size
        let mut tmp_requant_challenges = BTreeMap::<usize, E>::new();

        let max_vars = trace
            .iter()
            .fold(0, |current_max, (_, step)| match step.layer {
                Layer::Dense(..) => current_max,
                Layer::Activation(Activation::Relu(..)) => std::cmp::max(current_max, 8),
                Layer::Requant(info) => {
                    let need_to_insert = tmp_requant_challenges.get(&info.after_range).is_none();
                    if need_to_insert {
                        tmp_requant_challenges.insert(info.after_range, E::random(thread_rng()));
                    }
                    std::cmp::max(current_max, info.after_range.ilog2())
                }
                _ => unreachable!(),
            });

        // make the basefold params to commit to all the polys (we won't open them this is just to save some hashing in the verifier).
        let params = Pcs::<E>::setup(1 << max_vars)?;

        let mut tmp_witness_storage = HashMap::new();
        let mut commits_in_order = vec![];
        let mut final_table_lookups = BTreeMap::<LookupType, Vec<E>>::new();
        // Initiate a vec to hold all the witness poly info
        let mut polys_with_id = Vec::<(usize, Vec<E>)>::new();
        // For each step in the inference trace construct the witness MLEs and also batch commit to them so we can generate seperation challenges.
        trace.iter().try_for_each(|(step_input, step)| {
            let step_idx = step.id;
            match step.layer {
                Layer::Dense(..) => Result::<(), anyhow::Error>::Ok(()),
                Layer::Activation(Activation::Relu(..)) => {
                    let (lookup_type, _) = ctx
                        .circuits
                        .get(&step_idx)
                        .expect("The index didn't have a circuit when it was expected to");
                    let num_vars = lookup_type.num_vars();

                    let mle_evals = [step_input, step.output.as_slice()]
                        .iter()
                        .map(|val| {
                            val.iter()
                                .map(|eval| eval.as_bases()[0])
                                .collect::<Vec<E::BaseField>>()
                        })
                        .collect::<Vec<Vec<E::BaseField>>>();
                    let mles = mle_evals
                        .iter()
                        .map(|val| DenseMultilinearExtension::from_evaluations_slice(num_vars, val))
                        .collect::<Vec<MLE<E>>>();
                    let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                    let batch_commit = Pcs::<E>::batch_commit(&pp, &mles)?.to_commitment();
                    commits_in_order.extend_from_slice(batch_commit.root().0.as_slice());

                    let merged_evals = (0..1 << num_vars)
                        .map(|i| {
                            mle_evals
                                .iter()
                                .enumerate()
                                .fold(tmp_const_challenge, |acc, (j, col)| {
                                    acc + E::from(col[i]) * tmp_relu_challenges[j]
                                })
                        })
                        .collect::<Vec<E>>();

                    tmp_witness_storage
                        .insert(step_idx, (lookup_type.clone(), batch_commit, mle_evals));

                    final_table_lookups
                        .entry(LookupType::ReluTable)
                        .or_insert_with(|| vec![])
                        .extend(merged_evals.into_iter());

                    // Add the poly we have to merge claims about
                    polys_with_id.push((step_idx, step.output.clone()));
                    Ok(())
                }
                Layer::Requant(requant) => {
                    let (lookup_type, _) = ctx
                        .circuits
                        .get(&step_idx)
                        .expect("The index didn't have a circuit when it was expected to");
                    let num_columns = lookup_type.number_of_columns();
                    let num_vars = lookup_type.num_vars();

                    let mut relu_input = vec![E::BaseField::ZERO; 1 << num_vars];

                    let mut discarded_chunks =
                        vec![vec![E::BaseField::ZERO; 1 << num_vars]; num_columns - 1];

                    // Bit mask for the bytes
                    let bit_mask = requant.after_range as i128 - 1;

                    let max_bit = requant.range << 1;
                    let subtract = max_bit >> requant.right_shift;

                    step_input.iter().enumerate().for_each(|(index, val)| {
                        // First we take the relu input
                        let u64val = val.to_canonical_u64_vec()[0];
                        let i128val = if u64val > E::BaseField::MODULUS_U64 >> 1 {
                            u64val as i128 - E::BaseField::MODULUS_U64 as i128
                        } else {
                            u64val as i128
                        };
                        let pre_shift = i128val + max_bit as i128;
                        let tmp = pre_shift >> requant.right_shift;
                        let relu_in = tmp - subtract as i128;
                        let relu_in_field: E = relu_in.to_field();

                        relu_input[index] = relu_in_field.as_bases()[0];
                        // the value of an input should always be basefield elements
                        let to_subtract = tmp << requant.right_shift;

                        let mut remainder_vals = pre_shift - to_subtract;
                        discarded_chunks
                            .iter_mut()
                            .rev()
                            .for_each(|discarded_chunk| {
                                let chunk = remainder_vals & bit_mask;
                                let value = chunk as i128 - 128;

                                let field_elem: E = value.to_field();
                                discarded_chunk[index] = field_elem.as_bases()[0];
                                remainder_vals >>= requant.after_range.ilog2();
                            });
                    });
                    // Debug check the recombination
                    debug_assert!({
                        let mut checker = true;
                        step_input
                            .iter()
                            .zip(step.output.iter())
                            .enumerate()
                            .for_each(|(i, (value, out_value))| {
                                let first_val = discarded_chunks[0][i];
                                checker &= E::from(first_val) == *out_value;
                                let calc_first_val = E::from(1u64 << requant.right_shift)
                                    * (E::from(first_val)
                                        + <Element as Fieldizer<E>>::to_field(&(subtract as i128)));

                                let acc_vals = discarded_chunks
                                    .iter()
                                    .skip(1)
                                    .rev()
                                    .enumerate()
                                    .fold(E::ZERO, |acc, (j, col)| {
                                        acc + E::from((requant.after_range.pow(j as u32)) as u64)
                                            * (E::from(col[i]) + E::from(128u64))
                                    });

                                let should_be_equal = calc_first_val + acc_vals
                                    - <Element as Fieldizer<E>>::to_field(&(max_bit as i128));
                                checker &= should_be_equal == *value;
                                if !checker {
                                    println!("value was: {:?}", value);
                                    println!("calculated: {:?}", should_be_equal);
                                    println!(
                                        "calculated u64: {}",
                                        should_be_equal.to_canonical_u64_vec()[0]
                                    );
                                };
                            });

                        checker
                    });
                    let padded_len = discarded_chunks.len().next_power_of_two();
                    // Pad out to a power of two
                    while discarded_chunks.len() != padded_len {
                        discarded_chunks.insert(0, vec![E::BaseField::ZERO; 1 << num_vars]);
                    }

                    let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                    let witness_mles = discarded_chunks
                        .iter()
                        .map(|evals| {
                            DenseMultilinearExtension::from_evaluations_slice(num_vars, evals)
                        })
                        .collect::<Vec<MLE<E>>>();
                    let batch_commit = Pcs::<E>::batch_commit(&pp, &witness_mles)?.to_commitment();
                    commits_in_order.extend_from_slice(batch_commit.root().0.as_slice());

                    // Get the requant challenge for this table size from the map
                    let challenge = tmp_requant_challenges
                        .get(&requant.after_range)
                        .ok_or(anyhow!("Could not get challenge"))?;
                    let lookups = discarded_chunks
                        .iter()
                        .flat_map(|col| {
                            col.iter()
                                .map(|val| tmp_const_challenge + E::from(*val) * *challenge)
                                .collect::<Vec<E>>()
                        })
                        .collect::<Vec<E>>();

                    tmp_witness_storage.insert(
                        step_idx,
                        (lookup_type.clone(), batch_commit, discarded_chunks),
                    );

                    final_table_lookups
                        .entry(LookupType::RequantTable(
                            requant.after_range.ilog2() as usize
                        ))
                        .or_insert_with(|| vec![])
                        .extend(lookups.into_iter());

                    // Add the poly we have to merge claims about
                    polys_with_id.push((step_idx, step.output.clone()));

                    Ok(())
                }
            }
        })?;

        // Produce all the table columns
        let mut final_step = trace.last_step().id + 1;
        final_table_lookups
            .iter()
            .try_for_each(|(table_type, merged_lookups)| {
                let num_vars = table_type.num_vars();
                let step_id = final_step;
                final_step += 1;
                let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                if let LookupType::ReluTable = table_type {
                    let mut table_mles = table_type
                        .get_mles::<E>()
                        .ok_or(anyhow!("couldn't get mles for table type"))?;

                    let merged_table = (0..1 << num_vars)
                        .map(|i| {
                            table_mles
                                .iter()
                                .enumerate()
                                .fold(tmp_const_challenge, |acc, (j, col)| {
                                    acc + tmp_relu_challenges[j] * E::from(col[i])
                                })
                        })
                        .collect::<Vec<E>>();
                    let multiplicity_poly =
                        compute_multiplicity_poly(&merged_table, merged_lookups);
                    let m_evals = multiplicity_poly.get_base_field_vec().to_vec();
                    let m_commit = Pcs::<E>::commit(&pp, &multiplicity_poly)?.to_commitment();

                    // Add the multiplicity poly that we have to open
                    polys_with_id.push((
                        step_id,
                        m_evals.iter().copied().map(E::from).collect::<Vec<E>>(),
                    ));

                    table_mles.push(m_evals);
                    // Add the mutliplicity commit to `commits_in_order`
                    commits_in_order.extend_from_slice(m_commit.root().0.as_slice());
                    tmp_witness_storage.insert(step_id, (*table_type, m_commit, table_mles));
                    Ok(())
                } else if let LookupType::RequantTable(log_size) = table_type {
                    let mut table_mles = table_type
                        .get_mles::<E>()
                        .ok_or(anyhow!("couldn't get mles for table type"))?;

                    let challenge = tmp_requant_challenges
                        .get(&(1 << log_size))
                        .ok_or(anyhow!("Couldn't get tmp requant challenge"))?;
                    let merged_table = table_mles[0]
                        .iter()
                        .map(|val| tmp_const_challenge + *challenge * E::from(*val))
                        .collect::<Vec<E>>();
                    let multiplicity_poly =
                        compute_multiplicity_poly(&merged_table, merged_lookups);
                    let m_evals = multiplicity_poly.get_base_field_vec().to_vec();
                    let m_commit = Pcs::<E>::commit(&pp, &multiplicity_poly)?.to_commitment();
                    // Add the multiplicity poly that we have to open
                    polys_with_id.push((
                        step_id,
                        m_evals.iter().copied().map(E::from).collect::<Vec<E>>(),
                    ));
                    table_mles.push(m_evals);
                    // Add the mutliplicity commit to `commits_in_order`
                    commits_in_order.extend_from_slice(m_commit.root().0.as_slice());
                    tmp_witness_storage.insert(step_id, (*table_type, m_commit, table_mles));
                    Ok(())
                } else {
                    Err(anyhow!(
                        "Encountered incrrect lookup type when computing multiplicity polys: {:?}",
                        table_type
                    ))
                }
            })?;

        // Now we work out the final challenges using the commitments
        let constant_challenge = E::from_bases(&hash_n_to_hash_no_pad(&commits_in_order).0[..2]);

        // Make one challenge for each requant table size, store in a hashmap
        let mut challenge_map = tmp_requant_challenges
            .iter()
            .map(|(after_size, _)| {
                let bit_size = after_size.ilog2() as u8;
                let requant_extra = RANGE_CHECK_DOM_SEP
                    .iter()
                    .chain(std::iter::once(&bit_size))
                    .map(|byte| E::BaseField::from(*byte as u64))
                    .collect::<Vec<E::BaseField>>();
                let input = [commits_in_order.clone(), requant_extra].concat();
                let requant_challenge = E::from_bases(&hash_n_to_hash_no_pad(&input).0[..2]);

                let identifier = format!("Requant_{}", bit_size);

                (identifier, vec![requant_challenge, constant_challenge])
            })
            .collect::<HashMap<String, Vec<E>>>();
        let relu_extra = RELU_DOM_SEP
            .iter()
            .map(|byte| E::BaseField::from(*byte as u64))
            .collect::<Vec<E::BaseField>>();
        let input = [commits_in_order.clone(), relu_extra].concat();
        let relu_challenge = E::from_bases(&hash_n_to_hash_no_pad(&input).0[..2]);

        challenge_map.insert("Relu".to_string(), vec![relu_challenge, constant_challenge]);

        let to_store: Vec<(StepIdx, LookupProverInfo<'a, E>)> = tmp_witness_storage
            .into_iter()
            .map(|(step_idx, (lt, batch_commit, mle_evals))| {
                let challenges = challenge_map
                    .get(&lt.name())
                    .expect("No challenges stored for lookup type");

                let circuit: &Circuit<E> = ctx
                    .get_circuit(step_idx)
                    .expect("No circuit stored for lookup type");
                let mut circuit_witness = CircuitWitness::new(circuit, challenges.to_vec());
                // Workout how many of the evals we feed into each instance as we will need to pad the evals to the correct power of two length.
                let mles_per_instance = lt.num_columns_per_instance();
                let padded_instances_size =
                    (mle_evals.len() / mles_per_instance).next_power_of_two();

                let required_padding = padded_instances_size * mles_per_instance - mle_evals.len();

                let padded_mle_evals = mle_evals
                    .into_iter()
                    .chain(
                        std::iter::repeat(vec![E::BaseField::ZERO; 1 << lt.num_vars()])
                            .take(required_padding),
                    )
                    .collect::<Vec<Vec<E::BaseField>>>();
                padded_mle_evals
                    .chunks(lt.num_columns_per_instance())
                    .for_each(|chunk| {
                        let wits_in = chunk
                            .iter()
                            .map(|evaluations| {
                                DenseMultilinearExtension::from_evaluations_slice(
                                    lt.num_vars(),
                                    evaluations,
                                )
                            })
                            .collect::<Vec<DenseMultilinearExtension<E>>>();
                        circuit_witness.add_instance(circuit, wits_in);
                    });

                let lookup_witness_data = LookupProverInfo {
                    lookup_type: lt,
                    batch_commitment: batch_commit,
                    circuit_witness,
                    challenges: challenges.to_vec(),
                };

                (step_idx, lookup_witness_data)
            })
            .collect();

        let witness_storage = BTreeMap::from_iter(to_store.into_iter());
        Ok((
            Self {
                current_step: 0,
                witness_storage,
            },
            polys_with_id,
        ))
    }

    pub fn continue_proving(&self) -> bool {
        let step_idx = self.current_step;
        self.witness_storage.get(&step_idx).is_some()
    }
}

pub trait LookupProtocol<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove<T: Transcript<E>>(
        lookup_ctx: &Context<E>,
        witness_ctx: &WitnessContext<E>,
        t: &mut T,
    ) -> anyhow::Result<Proof<E>> {
        let current_step = witness_ctx.current_step;
        // Get all the witness info from the context
        let LookupProverInfo {
            lookup_type,
            batch_commitment,
            circuit_witness,
            challenges,
            ..
        } = witness_ctx
            .witness_storage
            .get(&current_step)
            .ok_or(anyhow!(
                "Was not expecting to prove a lookup at step: {:?}",
                current_step
            ))?;
        // Get the circuit based on the lookup type
        let circuit = lookup_ctx.get_circuit(current_step)?;

        // Get the witness out and use this to calculate a challenge, we shall supply this to the verifier as well
        let witness_out_ref = circuit_witness.witness_out_ref();
        let witness_in = circuit_witness.witness_in_ref();
        // Append the outputs to the transcript
        let (witness_output_vec, stuff): (Vec<Vec<E::BaseField>>, Vec<Vec<E>>) = witness_out_ref
            .iter()
            .map(|mle| {
                let evals = mle.get_base_field_vec().to_vec();
                let stuff = evals
                    .chunks(2)
                    .map(|chunk| E::from_bases(chunk))
                    .collect::<Vec<E>>();
                (evals, stuff)
            })
            .unzip();

        let witness_outputs = witness_output_vec
            .into_iter()
            .flatten()
            .collect::<Vec<E::BaseField>>();
        let numerators = stuff[0].clone();
        let denominators = stuff[1].clone();

        // If the lookup type is a table type we append the challenges to the transcript here
        if lookup_type.is_table() {
            t.append_field_element_exts(&challenges);
        }

        t.append_field_elements(&witness_outputs);
        // Work out how many instances of the same circuit we a re proving at the same time
        // we then squeeze 1 + num_instance_vars challenges from the transcript
        let num_instance_vars = circuit_witness.instance_num_vars();
        // Squeeze a challenge to be used in evaluating the output mle (this is denom_prod flattened ot basefield elements)
        let output_point =
            std::iter::repeat_with(|| t.get_and_append_challenge(b"lookup_challenge").elements)
                .take(1 + num_instance_vars)
                .collect::<Vec<E>>();

        // Evaluate all the output MLEs at the the challenge point
        let wires_out_evals = witness_out_ref
            .iter()
            .map(|mle| PointAndEval::new(output_point.clone(), mle.evaluate(&output_point)))
            .collect::<Vec<PointAndEval<E>>>();

        // make the GKR proof
        let (gkr_proof, _) = IOPProverState::prove_parallel(
            circuit,
            circuit_witness,
            vec![],
            wires_out_evals,
            1 << num_instance_vars,
            t,
        );

        // Group the claims so that we know what our initial sumcheck eval should be at the next step
        let last_step_message = gkr_proof.sumcheck_proofs.last().unwrap();
        let point = &last_step_message.sumcheck_proof.point;
        let witness_in_vars = witness_in[0].num_vars() - num_instance_vars;
        let claims = witness_in
            .iter()
            .flat_map(|w_in| {
                w_in.get_base_field_vec()
                    .chunks(1 << witness_in_vars)
                    .map(|mle_evals| {
                        let mle = DenseMultilinearExtension::from_evaluations_slice(
                            witness_in_vars,
                            mle_evals,
                        );
                        Claim::new(
                            point[..witness_in_vars].to_vec(),
                            mle.evaluate(&point[..witness_in_vars]),
                        )
                    })
                    .collect::<Vec<Claim<E>>>()
            })
            .collect::<Vec<Claim<E>>>();

        Ok(Proof {
            commitment: batch_commitment.clone(),
            claims,
            gkr_proof,
            numerators,
            denominators,
            challenges: challenges.clone(),
            instance_num_vars: num_instance_vars,
        })
    }

    // commitments to the lookups, one commitment per "column"
    fn verify<T: Transcript<E>>(
        lookup_ctx: &Context<E>,
        challenges: &[E],
        step: usize,
        proof: Proof<E>,
        t: &mut T,
    ) -> anyhow::Result<VerifierClaims<E>> {
        // Get the circuit by the lookup type
        let (lookup_type, circuit) = lookup_ctx.get_circuit_and_type(step)?;

        // Split the proof into parts
        let Proof {
            commitment,
            gkr_proof,
            numerators,
            denominators,
            instance_num_vars,
            claims,
            ..
        } = proof;
        // Compute the expectted output values as the prover should have

        let output_values = numerators
            .iter()
            .chain(denominators.iter())
            .flat_map(|val| val.as_bases().to_vec())
            .collect::<Vec<E::BaseField>>();
        let numerator_mle = DenseMultilinearExtension::from_evaluations_vec(
            1 + instance_num_vars,
            numerators
                .iter()
                .flat_map(|val| val.as_bases().to_vec())
                .collect::<Vec<E::BaseField>>(),
        );
        let denominator_mle = DenseMultilinearExtension::from_evaluations_vec(
            1 + instance_num_vars,
            denominators
                .iter()
                .flat_map(|val| val.as_bases().to_vec())
                .collect::<Vec<E::BaseField>>(),
        );

        // If the lookup type is a table append the challenges to the transcript
        if lookup_type.is_table() {
            t.append_field_element_exts(&challenges);
        }

        t.append_field_elements(&output_values);
        // Squeeze the challenge
        let output_point =
            std::iter::repeat_with(|| t.get_and_append_challenge(b"lookup_challenge").elements)
                .take(1 + instance_num_vars)
                .collect::<Vec<E>>();

        // We directly evaluate as all the output MLEs
        let witness_out_evals = vec![
            PointAndEval::<E>::new(output_point.clone(), numerator_mle.evaluate(&output_point)),
            PointAndEval::<E>::new(
                output_point.clone(),
                denominator_mle.evaluate(&output_point),
            ),
        ];
        // Run the GKR verification
        let _gkr_claims = IOPVerifierState::verify_parallel(
            circuit,
            challenges,
            vec![],
            witness_out_evals,
            gkr_proof,
            instance_num_vars,
            t,
        )
        .map_err(|e| anyhow!("Error when verifying GKR {{ inner: {:?}}}", e))?;

        // // Convert to our `Claim` type
        // let claims = gkr_claims
        //     .point_and_evals
        //     .iter()
        //     .map(Claim::from)
        //     .collect::<Vec<Claim<E>>>();

        Ok(VerifierClaims {
            claims: claims.clone(),
            commitment,
            numerators,
            denominators,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{default_transcript, model::Model};

    use super::*;

    type F = GoldilocksExt2;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use tracing_subscriber;

    #[test]
    fn test_prover_steps() {
        tracing_subscriber::fmt::init();
        let (model, input) = Model::random(4);
        model.describe();
        let trace = model.run::<F>(input.clone());
        let output = trace.final_output();
        let ctx = crate::iop::Context::generate(&model).expect("unable to generate context");

        let (mut witness_context, _) =
            WitnessContext::<F>::initialise_witness_ctx(&ctx.lookup, &trace).unwrap();
        let mut prover_transcript = default_transcript();
        let step_indices = witness_context
            .witness_storage
            .keys()
            .copied()
            .collect::<Vec<usize>>();
        let lookup_proofs = step_indices
            .into_iter()
            .map(|step| {
                witness_context.current_step = step;
                let proof = LogUp::prove(&ctx.lookup, &witness_context, &mut prover_transcript)?;
                Result::<(StepIdx, Proof<F>), anyhow::Error>::Ok((step, proof))
            })
            .collect::<Result<Vec<(StepIdx, Proof<F>)>, _>>()
            .unwrap();

        let mut lookup_commits = vec![];
        let mut numerators = vec![];
        let mut denominators = vec![];
        lookup_proofs.iter().for_each(|(_, proof)| {
            lookup_commits.extend_from_slice(proof.get_digest().0.as_slice());
            numerators.extend_from_slice(proof.numerators.as_slice());
            denominators.extend_from_slice(proof.denominators.as_slice());
        });

        let (num, denom) = numerators.iter().zip(denominators.iter()).fold(
            (F::ZERO, F::ONE),
            |(acc_num, acc_denom), (num_i, denom_i)| {
                (
                    acc_num * *denom_i + acc_denom * *num_i,
                    acc_denom * *denom_i,
                )
            },
        );

        assert_eq!(num, F::ZERO);
        assert_ne!(denom, F::ZERO);

        let constant_challenge = F::from_bases(&hash_n_to_hash_no_pad(&lookup_commits).0[..2]);

        let mut lookup_challenges = HashMap::<String, Vec<F>>::new();

        (0..ctx.steps_info.len()).for_each(|i| {
            let res = ctx.lookup.get_circuit_and_type(i);
            if let Ok((LookupType::RequantTable(bit_size), _)) = res {
                let lookup_type = LookupType::RequantTable(*bit_size);
                let bit_size = *bit_size as u8;
                let requant_extra = lookup_type
                    .get_dom_sep()
                    .iter()
                    .chain(std::iter::once(&bit_size))
                    .map(|byte| Goldilocks::from(*byte as u64))
                    .collect::<Vec<Goldilocks>>();
                let input = [lookup_commits.clone(), requant_extra].concat();
                let requant_challenge = F::from_bases(&hash_n_to_hash_no_pad(&input).0[..2]);
                println!("Inserting: {}", lookup_type.name());
                lookup_challenges.insert(lookup_type.name(), vec![
                    requant_challenge,
                    constant_challenge,
                ]);
            } else if let Ok((LookupType::ReluTable, _)) = res {
                let lookup_type = LookupType::ReluTable;
                let relu_extra = lookup_type
                    .get_dom_sep()
                    .iter()
                    .map(|byte| Goldilocks::from(*byte as u64))
                    .collect::<Vec<Goldilocks>>();
                let input = [lookup_commits.clone(), relu_extra].concat();
                let relu_challenge = F::from_bases(&hash_n_to_hash_no_pad(&input).0[..2]);
                println!("Inserting: {}", lookup_type.name());
                lookup_challenges
                    .insert(lookup_type.name(), vec![relu_challenge, constant_challenge]);
            }
        });

        let mut verifier_transcript = default_transcript();
        let (final_numerator, final_denominator) = lookup_proofs
            .iter()
            .try_fold((F::ZERO, F::ONE), |(acc_num, acc_denom), (step, proof)| {
                let (lookup_type, _) = ctx.lookup.get_circuit_and_type(*step)?;
                let challenges = lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                    "Couldn't get challenges for lookup type: {}",
                    lookup_type.name()
                ))?;
                let verifier_claim = LogUp::verify(
                    &ctx.lookup,
                    challenges,
                    *step,
                    proof.clone(),
                    &mut verifier_transcript,
                )?;

                let (out_num, out_denom) = verifier_claim
                    .numerators()
                    .iter()
                    .zip(verifier_claim.denominators().iter())
                    .fold(
                        (acc_num, acc_denom),
                        |(acc_num_i, acc_denom_i), (num, denom)| {
                            (
                                acc_num_i * *denom + acc_denom_i * *num,
                                acc_denom_i * *denom,
                            )
                        },
                    );
                Result::<(F, F), anyhow::Error>::Ok((out_num, out_denom))
            })
            .unwrap();

        assert_eq!(final_numerator, F::ZERO);
        assert_ne!(final_denominator, F::ZERO);
    }
}
