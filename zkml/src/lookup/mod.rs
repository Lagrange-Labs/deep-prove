use anyhow::anyhow;
use ark_std::rand::thread_rng;
use ff::Field;
use ff_ext::ExtensionField;
use gkr::{
    structs::{Circuit, CircuitWitness, IOPProof, IOPProverState, IOPVerifierState, PointAndEval},
    util::ceil_log2,
};

use mpcs::{BasefoldCommitment, PolynomialCommitmentScheme};
use poseidon::digest::Digest;
use std::collections::{HashMap, HashSet};
use utils::compute_multiplicity_poly;

use crate::{
    Claim, Element,
    activation::Relu,
    commit::{Pcs, precommit::PolyID},
    iop::context::StepInfo,
    model::{InferenceTrace, StepIdx},
    pooling::Maxpool2D,
    quantization::{Fieldizer, Requant},
    tensor::Tensor,
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
    Maxpool2D(Maxpool2D, usize), // Maxpool update
    Maxpool2DTable(usize),       // Maxpool update
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
            StepInfo::Pooling(info) => LookupType::Maxpool2D(info.poolinfo, info.num_vars),
            // Maxpool update
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
            // Maxpool update
            LookupType::Maxpool2D(_, num_vars) => lookup_wire_fractional_sumcheck(1, *num_vars),
            LookupType::Maxpool2DTable(num_vars) => table_fractional_sumcheck(1, *num_vars),
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
            // Maxpool update
            LookupType::Maxpool2D(..) => 1,
            LookupType::Maxpool2DTable(..) => 1,
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_columns_per_instance(&self) -> usize {
        match self {
            LookupType::Requant(..) => 1,
            LookupType::Relu(..) => 2,
            LookupType::ReluTable => 3,
            LookupType::RequantTable(..) => 2,
            // Maxpool update
            LookupType::Maxpool2D(..) => 1,
            LookupType::Maxpool2DTable(..) => 1,
            LookupType::NoLookup => 0,
        }
    }

    pub fn num_vars(&self) -> usize {
        match self {
            LookupType::Requant(.., num_vars) => *num_vars,
            LookupType::Relu(num_vars) => *num_vars,
            LookupType::ReluTable => 8,
            LookupType::RequantTable(num_vars) => *num_vars,
            // Maxpool update
            LookupType::Maxpool2D(.., num_vars) => *num_vars,
            LookupType::Maxpool2DTable(num_vars) => *num_vars,
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

    pub fn name(&self) -> String {
        match self {
            LookupType::Requant(info, ..) => {
                format!("Requant_{}", info.after_range.ilog2() as usize)
            }
            LookupType::Relu(..) | LookupType::ReluTable => "Relu".to_string(),
            LookupType::RequantTable(num_vars) => format!("Requant_{}", *num_vars),
            // Maxpool update
            LookupType::Maxpool2D(.., num_vars) | LookupType::Maxpool2DTable(num_vars) => {
                format!("Maxpool2D_{}", *num_vars)
            }
            LookupType::NoLookup => "NoLookup".to_string(),
        }
    }

    pub fn get_mles<E: ExtensionField>(&self) -> Option<Vec<Vec<E::BaseField>>> {
        match self {
            LookupType::ReluTable => {
                let (relu_in, relu_out) = Relu::to_mle::<E>();

                Some(vec![relu_in, relu_out])
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
            // Maxpool update
            LookupType::Maxpool2DTable(_) => {
                unimplemented!()
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

    pub fn table_type(&self) -> LookupType {
        match self {
            LookupType::Relu(..) | LookupType::ReluTable => LookupType::ReluTable,
            LookupType::Requant(info, ..) => {
                LookupType::RequantTable(info.after_range.ilog2() as usize)
            }
            LookupType::RequantTable(..) => *self,
            // Maxpool update
            LookupType::Maxpool2D(..) => *self,
            LookupType::Maxpool2DTable(..) => *self,
            LookupType::NoLookup => LookupType::NoLookup,
        }
    }

    pub fn prep_lookup_polys<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
    ) -> Vec<Vec<E::BaseField>> {
        match self {
            LookupType::Relu(..) => {
                let output = Relu::op(&Relu::new(), input);
                vec![input.get_data(), output.get_data()]
                    .into_iter()
                    .map(|values| {
                        values
                            .iter()
                            .map(|val| {
                                let field_val: E = val.to_field();
                                field_val.as_bases()[0]
                            })
                            .collect::<Vec<E::BaseField>>()
                    })
                    .collect::<Vec<Vec<E::BaseField>>>()
            }
            LookupType::Requant(info, ..) => info.prep_for_requantize::<E>(input.get_data()),
            // Maxpool update
            LookupType::Maxpool2D(..) => unimplemented!(),
            _ => vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Map relates [`StepIdx`] to the position of the relevant lookup info in `lookup_circuits`.
    step_index_map: HashMap<StepIdx, usize>,
    /// Storage for all the lookup circuits used as well as the their [`LookupType`].
    lookup_circuits: Vec<(LookupType, Circuit<E>)>,
    /// Storage for all the tables used by the [`Model`] prover.
    table_circuits: Vec<TableInfo<E>>,
}

/// Info related to the lookup protocol tables.
/// Here `poly_id` is the multiplicity poly for this table.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct TableInfo<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub poly_id: PolyID,
    pub num_vars: usize,
    pub table_commitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::Commitment,
    pub circuit: Circuit<E>,
    pub lookup_type: LookupType,
}

#[derive(Clone, Default)]
pub struct WitnessContext<'a, E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    lookup_witnesses: Vec<LookupProverInfo<'a, E>>,
    table_witnesses: Vec<LookupProverInfo<'a, E>>,
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
        let index = self.step_index_map.get(&step_idx).ok_or(anyhow::anyhow!(
            "Context does not contain a circuit for the step: {:?}",
            step_idx
        ))?;

        Ok(&self.lookup_circuits[*index].1)
    }

    /// Gets circuit by type
    pub fn get_circuit_by_type(&self, lookup_type: &LookupType) -> Option<&Circuit<E>> {
        self.lookup_circuits
            .iter()
            .find(|(lt, _)| *lt == *lookup_type)
            .and_then(|(_, circuit)| Some(circuit))
    }

    /// Getter for the lookup circuit and type depending on the [`StepIdx`]
    pub fn get_circuit_and_type(
        &self,
        step_idx: StepIdx,
    ) -> anyhow::Result<&(LookupType, Circuit<E>)> {
        let index = self.step_index_map.get(&step_idx).ok_or(anyhow::anyhow!(
            "Context does not contain a circuit for the step: {:?}",
            step_idx
        ))?;

        Ok(&self.lookup_circuits[*index])
    }

    /// Getter for the [`TableInfo`]s
    pub fn get_table_circuits(&self) -> &[TableInfo<E>] {
        &self.table_circuits
    }

    /// Generate [`Context`] from a [ModelContext](`crate::iop::context::Context`)
    pub fn generate(steps_info: &[StepInfo<E>]) -> anyhow::Result<Context<E>> {
        // We make a hashset that we use to check if we have already seen a table.
        let mut tables_used = HashSet::<LookupType>::new();
        // Make all the constiuent parts of the `Context`
        let mut lookup_circuits = Vec::<(LookupType, Circuit<E>)>::new();
        let mut table_circuits = Vec::<TableInfo<E>>::new();
        let mut step_index_map = HashMap::<StepIdx, usize>::new();
        // Make some Pcs params so we can commit to items that should be passed to the transcript, we
        // pick 17 as the max number of variables for now as we should never have to deal with a table bigger than this.
        let params = Pcs::<E>::setup(1 << 17)?;

        // We start table related poly ids after the ones relating to layers of the model
        let mut table_poly_id = steps_info.len();

        // Iterate through the step info and make the relevant circuit for each.
        steps_info
            .iter()
            .enumerate()
            .try_for_each::<_, Result<(), anyhow::Error>>(|(idx, step)| match step {
                // Skip Dense steps
                StepInfo::Dense(..) => Ok(()),
                StepInfo::Activation(..) => {
                    let lookup_type = LookupType::from(step);
                    let circuit = lookup_type.make_circuit::<E>();
                    let vec_position = lookup_circuits.len();
                    step_index_map.insert(idx, vec_position);
                    lookup_circuits.push((lookup_type, circuit));
                    if tables_used.insert(LookupType::ReluTable) {
                        let lookup_type = LookupType::ReluTable;
                        let mles = lookup_type
                            .get_mles::<E>()
                            .ok_or(anyhow!(
                                "Got none table LookupType when only tables should be made: {:?}",
                                lookup_type
                            ))?
                            .into_iter()
                            .map(|evaluations| {
                                DenseMultilinearExtension::<E>::from_evaluations_vec(8, evaluations)
                            })
                            .collect::<Vec<DenseMultilinearExtension<E>>>();
                        let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << 8)?;
                        let commit = Pcs::<E>::batch_commit(&pp, &mles)?.to_commitment();

                        let relu_table_info = TableInfo {
                            poly_id: table_poly_id,
                            num_vars: 8,
                            table_commitment: commit,
                            circuit: lookup_type.make_circuit::<E>(),
                            lookup_type,
                        };
                        table_circuits.push(relu_table_info);
                        table_poly_id += 1;
                    }
                    Ok(())
                }
                StepInfo::Requant(info) => {
                    let lookup_type = LookupType::from(step);
                    let circuit = lookup_type.make_circuit::<E>();
                    let vec_position = lookup_circuits.len();
                    step_index_map.insert(idx, vec_position);
                    lookup_circuits.push((lookup_type, circuit));

                    let num_vars = info.requant.after_range.ilog2() as usize;
                    if tables_used.insert(LookupType::RequantTable(num_vars)) {
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
                                DenseMultilinearExtension::<E>::from_evaluations_vec(
                                    num_vars,
                                    evaluations,
                                )
                            })
                            .collect::<Vec<DenseMultilinearExtension<E>>>();
                        let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                        let commit = Pcs::<E>::batch_commit(&pp, &mles)?.to_commitment();
                        let table_info = TableInfo {
                            poly_id: table_poly_id,
                            num_vars,
                            table_commitment: commit,
                            circuit,
                            lookup_type,
                        };
                        table_circuits.push(table_info);
                        table_poly_id += 1;
                    }
                    Ok(())
                }
                // Maxpool update
                StepInfo::Pooling(_) => unimplemented!(),
                _ => unreachable!(),
            })?;

        Ok(Context {
            step_index_map,
            lookup_circuits,
            table_circuits,
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for table_info in self.table_circuits.iter() {
            t.append_field_element(&E::BaseField::from(table_info.poly_id as u64));
            t.append_field_element(&E::BaseField::from(table_info.num_vars as u64));
            t.append_field_elements(table_info.table_commitment.root().0.as_slice());
        }

        Ok(())
    }
}

impl<'a, E: ExtensionField> WitnessContext<'a, E>
where
    E: Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Initialises [`WitnessContext`] from an [`InferenceTrace`] and a [`Context`]
    pub fn initialise_witness_ctx<'b, T: Transcript<E>>(
        ctx: &'b Context<E>,
        trace: &InferenceTrace<Element>,
        t: &mut T,
    ) -> anyhow::Result<(WitnessContext<'a, E>, Vec<(usize, Vec<E>)>)>
    where
        'b: 'a,
    {
        // We look at all the table circuits and make the corresponding challenges for them all
        let tmp_const_challenge = E::BaseField::random(thread_rng());
        let tmp_challenges = ctx
            .table_circuits
            .iter()
            .map(|table_info| {
                (
                    table_info.lookup_type.name(),
                    E::BaseField::random(thread_rng()),
                )
            })
            .collect::<HashMap<String, E::BaseField>>();

        // make the basefold params to commit to all the polys (we won't open them this is just to save some hashing in the verifier).
        let params = Pcs::<E>::setup(1 << 17)?;

        let mut final_table_lookups = HashMap::<LookupType, Vec<E::BaseField>>::new();
        // Initiate a vec to hold all the witness poly info
        let mut polys_with_id = Vec::<(usize, Vec<E>)>::new();
        // For each step in the inference trace construct the witness MLEs and also batch commit to them so we can generate seperation challenges.
        let tmp_witness_storage = trace
            .iter()
            .filter_map(|(step_input, step)| {
                let step_idx = step.id;

                if let Ok((lookup_type, _)) = ctx.get_circuit_and_type(step_idx) {
                    let num_vars = step_input.get_data().len().ilog2() as usize;

                    let mut mle_evals = lookup_type.prep_lookup_polys::<E>(step_input);
                    let padded_len = mle_evals.len().next_power_of_two();
                    mle_evals.resize(padded_len, vec![E::BaseField::ZERO; 1 << num_vars]);

                    let mles = mle_evals
                        .iter()
                        .map(|val| DenseMultilinearExtension::from_evaluations_slice(num_vars, val))
                        .collect::<Vec<MLE<E>>>();
                    let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars).ok()?;
                    let batch_commit = Pcs::<E>::batch_commit(&pp, &mles).ok()?.to_commitment();
                    t.append_field_elements(batch_commit.root().0.as_slice());

                    // Compute a temporary merged eval to make the multiplicity polys
                    let challenge = tmp_challenges.get(&lookup_type.name())?;
                    let challenges =
                        std::iter::successors(Some(*challenge), |prev| Some(*prev * *challenge))
                            .take(lookup_type.num_columns_per_instance())
                            .collect::<Vec<E::BaseField>>();

                    let merged_evals = mle_evals
                        .chunks(lookup_type.num_columns_per_instance())
                        .flat_map(|chunk| {
                            (0..1 << num_vars).map(|i| {
                                chunk
                                    .iter()
                                    .enumerate()
                                    .fold(tmp_const_challenge, |acc, (j, col)| {
                                        acc + col[i] * challenges[j]
                                    })
                            })
                        })
                        .collect::<Vec<E::BaseField>>();

                    final_table_lookups
                        .entry(lookup_type.table_type())
                        .or_insert_with(|| vec![])
                        .extend(merged_evals.into_iter());

                    // Add the poly we have to merge claims about
                    polys_with_id.push((
                        step_idx,
                        step.output
                            .get_data()
                            .iter()
                            .map(Fieldizer::<E>::to_field)
                            .collect(),
                    ));

                    Some((step_idx, *lookup_type, batch_commit, mle_evals))
                } else {
                    None
                }
            })
            .collect::<Vec<(_, _, _, _)>>();

        // Produce all the table columns and work out challeneges
        let mut challenge_map = HashMap::<String, E>::new();
        let constant_challenge = t
            .get_and_append_challenge(b"table_constant_challenge")
            .elements;

        let table_witnesses = ctx
            .table_circuits
            .iter()
            .map(|table_info| {
                let TableInfo {
                    poly_id,
                    circuit,
                    lookup_type,
                    ..
                } = table_info;
                let num_vars = lookup_type.num_vars();
                let step_id = *poly_id;

                let (pp, _) = Pcs::<E>::trim(params.clone(), 1 << num_vars)?;
                let merged_lookups = final_table_lookups
                    .get(lookup_type)
                    .ok_or(anyhow!("No lookups for this table type: {:?}", lookup_type))?;
                let mut table_mles = lookup_type
                    .get_mles::<E>()
                    .ok_or(anyhow!("couldn't get mles for table type"))?;

                let temp_challenge = tmp_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                    "No temporary challenge for table type: {:?}",
                    lookup_type
                ))?;
                let tmp_cs = std::iter::successors(Some(*temp_challenge), |prev| {
                    Some(*prev * *temp_challenge)
                })
                .take(table_mles.len())
                .collect::<Vec<E::BaseField>>();
                let merged_table = (0..1 << num_vars)
                    .map(|i| {
                        table_mles
                            .iter()
                            .enumerate()
                            .fold(tmp_const_challenge, |acc, (j, col)| {
                                acc + tmp_cs[j] * col[i]
                            })
                    })
                    .collect::<Vec<E::BaseField>>();
                let multiplicity_poly = compute_multiplicity_poly(&merged_table, merged_lookups);
                let m_evals = multiplicity_poly.get_base_field_vec().to_vec();
                let m_commit = Pcs::<E>::commit(&pp, &multiplicity_poly)?.to_commitment();

                // Add the multiplicity poly that we have to open
                polys_with_id.push((
                    step_id,
                    m_evals.iter().copied().map(E::from).collect::<Vec<E>>(),
                ));

                table_mles.push(m_evals);
                // Add the mutliplicity commit to `commits_in_order`
                t.append_field_elements(m_commit.root().0.as_slice());
                // Squeeze the combining challenge related to this table type
                let actual_challenege = t.get_and_append_challenge(b"table_challenge").elements;
                // Put it in the hashmap for later
                challenge_map.insert(lookup_type.name(), actual_challenege);
                let mut circuit_witness =
                    CircuitWitness::new(circuit, vec![actual_challenege, constant_challenge]);

                let wits_in = table_mles
                    .iter()
                    .map(|evaluations| {
                        DenseMultilinearExtension::from_evaluations_slice(num_vars, evaluations)
                    })
                    .collect::<Vec<DenseMultilinearExtension<E>>>();

                circuit_witness.add_instance(circuit, wits_in);

                let lookup_witness_data = LookupProverInfo {
                    lookup_type: *lookup_type,
                    batch_commitment: m_commit,
                    circuit_witness,
                    challenges: vec![actual_challenege, constant_challenge],
                };

                Ok(lookup_witness_data)
            })
            .collect::<Result<Vec<LookupProverInfo<E>>, anyhow::Error>>()?;

        let lookup_witnesses: Vec<LookupProverInfo<'_, E>> = tmp_witness_storage
            .into_iter()
            .map(|(step_idx, lt, batch_commit, mle_evals)| {
                let challenge = challenge_map
                    .get(&lt.name())
                    .expect("No challenges stored for lookup type");

                let circuit: &Circuit<E> = ctx
                    .get_circuit(step_idx)
                    .expect("No circuit stored for lookup type");
                let mut circuit_witness =
                    CircuitWitness::new(circuit, vec![*challenge, constant_challenge]);
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
                    challenges: vec![*challenge, constant_challenge],
                };

                lookup_witness_data
            })
            .collect();

        Ok((
            Self {
                lookup_witnesses,
                table_witnesses,
            },
            polys_with_id,
        ))
    }

    pub fn next(&mut self) -> Option<LookupProverInfo<'a, E>> {
        self.lookup_witnesses.pop()
    }

    pub fn get_table_witnesses(&self) -> &[LookupProverInfo<E>] {
        &self.table_witnesses
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
        prover_info: &LookupProverInfo<E>,
        t: &mut T,
    ) -> anyhow::Result<Proof<E>> {
        // Get all the witness info from the context
        let LookupProverInfo {
            lookup_type,
            batch_commitment,
            circuit_witness,
            challenges,
            ..
        } = prover_info;
        // Get the circuit based on the lookup type
        let circuit = lookup_ctx.get_circuit_by_type(&lookup_type).ok_or(anyhow!(
            "Couldn't retrieve a circuit during proving for lookup type: {:?}",
            lookup_type
        ))?;

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

    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove_table<T: Transcript<E>>(
        circuit: &Circuit<E>,
        prover_info: &LookupProverInfo<E>,
        t: &mut T,
    ) -> anyhow::Result<Proof<E>> {
        // Get all the witness info from the context
        let LookupProverInfo {
            lookup_type,
            batch_commitment,
            circuit_witness,
            challenges,
            ..
        } = prover_info;
        // Get the circuit based on the lookup type

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

    // commitments to the lookups, one commitment per "column"
    fn verify_table<T: Transcript<E>>(
        challenges: &[E],
        lookup_type: &LookupType,
        circuit: &Circuit<E>,
        proof: Proof<E>,
        t: &mut T,
    ) -> anyhow::Result<VerifierClaims<E>> {
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
        // Compute the expected output values as the prover should have

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
    use crate::{default_transcript, init_test_logging, model::Model};

    use super::*;

    type F = GoldilocksExt2;
    use goldilocks::GoldilocksExt2;

    use tracing_subscriber;

    #[test]
    fn test_prover_steps() -> anyhow::Result<()> {
        init_test_logging();
        let (model, input) = Model::random(4);
        model.describe();
        let trace = model.run(input);

        let ctx = crate::iop::Context::generate(&model).expect("unable to generate context");

        let mut prover_transcript = default_transcript();
        ctx.write_to_transcript(&mut prover_transcript)?;
        let (witness_context, _) = WitnessContext::<F>::initialise_witness_ctx(
            &ctx.lookup,
            &trace,
            &mut prover_transcript,
        )
        .unwrap();

        let lookup_proofs = witness_context
            .lookup_witnesses
            .iter()
            .map(|prover_info| LogUp::prove(&ctx.lookup, prover_info, &mut prover_transcript))
            .collect::<Result<Vec<Proof<F>>, _>>()
            .unwrap();

        let mut numerators = vec![];
        let mut denominators = vec![];
        lookup_proofs.iter().for_each(|proof| {
            numerators.extend_from_slice(proof.numerators.as_slice());
            denominators.extend_from_slice(proof.denominators.as_slice());
        });

        let table_proofs = witness_context
            .table_witnesses
            .iter()
            .zip(ctx.lookup.table_circuits.iter())
            .map(|(prover_info, table_info)| {
                LogUp::prove_table(&table_info.circuit, prover_info, &mut prover_transcript)
            })
            .collect::<Result<Vec<Proof<F>>, _>>()
            .unwrap();

        table_proofs.iter().for_each(|proof| {
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

        let mut verifier_transcript = default_transcript();
        ctx.write_to_transcript(&mut verifier_transcript)?;
        lookup_proofs.iter().for_each(|proof| {
            verifier_transcript.append_field_elements(proof.commitment.root().0.as_slice())
        });
        let constant_challenge = verifier_transcript
            .get_and_append_challenge(b"table_constant_challenge")
            .elements;

        let mut lookup_challenges = HashMap::<String, Vec<F>>::new();

        table_proofs
            .iter()
            .zip(ctx.lookup.table_circuits.iter())
            .for_each(|(proof, table_info)| {
                let table_type = table_info.lookup_type;
                verifier_transcript.append_field_elements(proof.get_digest().0.as_slice());
                let challenege = verifier_transcript
                    .get_and_append_challenge(b"table_challenge")
                    .elements;
                lookup_challenges.insert(table_type.name(), vec![challenege, constant_challenge]);
            });

        let mut proof_iter = lookup_proofs.into_iter();
        let (final_lookup_numerator, final_lookup_denominator) =
            ctx.steps_info.iter().rev().enumerate().try_fold(
                (F::ZERO, F::ONE),
                |(acc_num, acc_denom), (step, step_info)| {
                    if step_info.requires_lookup() {
                        let proof = proof_iter.next().unwrap();
                        let (lookup_type, _) = ctx.lookup.get_circuit_and_type(step)?;
                        let challenges =
                            lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                                "Couldn't get challenges for lookup type: {}",
                                lookup_type.name()
                            ))?;
                        let verifier_claim = LogUp::verify(
                            &ctx.lookup,
                            challenges,
                            step,
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
                    } else {
                        Result::<(F, F), anyhow::Error>::Ok((acc_num, acc_denom))
                    }
                },
            )?;

        let (final_numerator, final_denominator) = table_proofs
            .iter()
            .zip(ctx.lookup.table_circuits.iter())
            .try_fold(
                (final_lookup_numerator, final_lookup_denominator),
                |(acc_num, acc_denom), (proof, table_info)| {
                    let TableInfo {
                        circuit,
                        lookup_type,
                        ..
                    } = table_info;
                    let challenges = lookup_challenges.get(&lookup_type.name()).ok_or(anyhow!(
                        "Couldn't get challenges for lookup type: {}",
                        lookup_type.name()
                    ))?;
                    let verifier_claim = LogUp::verify_table(
                        challenges,
                        lookup_type,
                        circuit,
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
                },
            )?;

        assert_eq!(final_numerator, F::ZERO);
        assert_ne!(final_denominator, F::ZERO);

        Ok(())
    }
}
