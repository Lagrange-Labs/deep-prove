//! File containing code for lookup witness generation.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
};
use p3_field::{Field, FieldAlgebra};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;
use utils::Metrics;

use super::{logup_gkr::error::LogUpError, witness::LogUpWitness};
use crate::{
    Claim, Context, Element,
    iop::ChallengeStorage,
    layers::{
        activation::Relu,
        provable::{NodeId, ProvableOp},
        transformer::softmax::{LOG_SCALE_FACTOR, OUTPUT_SCALE_FACTOR, SCALE_FACTOR},
    },
    model::{InferenceTrace, ToIterator},
    quantization::{self, Fieldizer},
};
use rayon::prelude::*;
pub const TABLE_POLY_ID_OFFSET: usize = 666;

type LookupAndColumns<BaseField> = (Vec<Element>, (Vec<BaseField>, Vec<BaseField>));

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Enum used for establishing the different table types needed to prove non-linear functions in a model.
pub enum TableType {
    /// Table used for the Relu activation function
    Relu,
    /// Table used for range checking (its size is determined by the quantisation bit size)
    Range,
    /// Table used for clamping values, the inner [`usize`] denotes the maximum bit length a value can be before clamping to use this table
    Clamping(usize),
    /// Table type used for computing Softmax, see the [`SoftmaxTableData`] struct for more info.
    Softmax(SoftmaxTableData),
    /// Table used for checking the normalisation error in Softmax operations, the first inner [`Element`] is `1` quantised by the scale factor, the second inner [`Element`] is the absoloute value of the allowable error
    ErrorTable(Element, Element),
    /// Table use to check if a value is zero or not, returns 1 if the value is zero and zero otherwise, the [`usize`] indicates how many variables the table has.
    ZeroTable(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Struct used to store Softmax table data
pub struct SoftmaxTableData {
    /// This is the result of calling [`f32::to_bits`] on the temperature value.
    float_bits: u32,
    /// The bit length of table size.
    table_size: usize,
    /// Any value larger than this gets mapped to zero.
    bkm: Element,
}

impl SoftmaxTableData {
    pub(crate) fn new(float_bits: u32, table_size: usize, bkm: Element) -> SoftmaxTableData {
        SoftmaxTableData {
            float_bits,
            table_size,
            bkm,
        }
    }

    pub(crate) fn float_temperature(&self) -> f32 {
        f32::from_bits(self.float_bits)
    }

    pub(crate) fn full_table_size(&self) -> Element {
        1 << self.table_size
    }

    pub(crate) fn size(&self) -> usize {
        self.table_size
    }

    pub(crate) fn bkm(&self) -> Element {
        self.bkm
    }

    pub(crate) fn table_output(&self, j: Element) -> Element {
        let float_temperature = self.float_temperature();
        let base = 1i128 << (LOG_SCALE_FACTOR - 8);
        let bkm = self.bkm();
        let prod = base * j;
        if prod >= bkm {
            0i128
        } else {
            let float_exp = (-prod as f32 / (SCALE_FACTOR as f32 * float_temperature)).exp();
            (float_exp * OUTPUT_SCALE_FACTOR as f32).round() as Element
        }
    }
}

impl TableType {
    pub fn get_merged_table_column<E: ExtensionField>(
        &self,
        column_separator: Element,
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        match self {
            TableType::Relu => {
                #[allow(clippy::type_complexity)]
                let (comb, (col_one, col_two)): (
                    Vec<Element>,
                    (Vec<E::BaseField>, Vec<E::BaseField>),
                ) = (*quantization::MIN - 1..=*quantization::MAX)
                    .map(|i| {
                        let out = Relu::apply(i);
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (
                            i + out * column_separator,
                            (i_field.as_bases()[0], out_field.as_bases()[0]),
                        )
                    })
                    .unzip();
                (comb, vec![col_one, col_two])
            }
            TableType::Range => {
                let (element_out, field): (Vec<Element>, Vec<E::BaseField>) = (0..1
                    << *quantization::BIT_LEN)
                    .map(|i| {
                        let i_field: E = i.to_field();
                        (i, i_field.as_bases()[0])
                    })
                    .unzip();
                (element_out, vec![field])
            }
            TableType::Clamping(size) => {
                let max = 1i128 << (size - 1);
                let min = -max;
                #[allow(clippy::type_complexity)]
                let (comb, (col_one, col_two)): (
                    Vec<Element>,
                    (Vec<E::BaseField>, Vec<E::BaseField>),
                ) = (min..max)
                    .map(|i| {
                        let out = if i < *quantization::MIN {
                            *quantization::MIN
                        } else if i > *quantization::MAX {
                            *quantization::MAX
                        } else {
                            i
                        };
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (
                            i + out * column_separator,
                            (i_field.as_bases()[0], out_field.as_bases()[0]),
                        )
                    })
                    .unzip();
                (comb, vec![col_one, col_two])
            }
            TableType::Softmax(table_data) => {
                let table_size = table_data.full_table_size();

                let (merged_lookup, (in_column, out_column)): LookupAndColumns<E::BaseField> =
                    (0i128..table_size)
                        .map(|j| {
                            let out_elem = table_data.table_output(j);
                            let in_field: E = j.to_field();
                            let out_field: E = out_elem.to_field();
                            (
                                j + COLUMN_SEPARATOR * out_elem,
                                (in_field.as_bases()[0], out_field.as_bases()[0]),
                            )
                        })
                        .unzip();
                (merged_lookup, vec![in_column, out_column])
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                // Work out the minimum and maximum elements of the table
                let table_min = *quant_one - *allowable_error;
                let table_max = *quant_one + *allowable_error;
                // Work out the full table size
                let table_size = 1usize << ceil_log2(2 * *allowable_error as usize);
                let (element_out, field): (Vec<Element>, Vec<E::BaseField>) = (table_min
                    ..=table_max)
                    .map(|elem| {
                        let f: E = elem.to_field();
                        (elem, f.as_bases()[0])
                    })
                    .chain(std::iter::repeat((0i128, E::BaseField::ZERO)))
                    .take(table_size)
                    .unzip();
                (element_out, vec![field])
            }
            TableType::ZeroTable(bit_size) => {
                let table_size: Element = 1 << bit_size;
                let (merged_lookup, (in_column, out_column)): LookupAndColumns<E::BaseField> = (0
                    ..table_size)
                    .map(|i| {
                        let out: Element = if i != 0 { 0 } else { 1 };
                        let merged_val = i + COLUMN_SEPARATOR * out;
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (merged_val, (i_field.as_bases()[0], out_field.as_bases()[0]))
                    })
                    .unzip();
                (merged_lookup, vec![in_column, out_column])
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::Range => "Range".to_string(),
            TableType::Clamping(size) => format!("Clamping: {size}"),
            TableType::Softmax(table_data) => {
                format!("Softmax - temperature: {}", table_data.float_temperature())
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                format!(
                    "Error Table - quantised one: {quant_one}, allowable error: {allowable_error}",
                )
            }
            TableType::ZeroTable(bit_size) => format!("Zero: {bit_size}"),
        }
    }

    pub fn evaluate_table_columns<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<Vec<E>, LogUpError> {
        match self {
            TableType::Range => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a range table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                Ok(vec![
                    point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index)
                    }),
                ])
            }
            TableType::Relu => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a relu table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                let first_column = point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                    acc + *p * E::from_canonical_u64(1u64 << index)
                }) - E::from_canonical_u64(1u64 << (*quantization::BIT_LEN - 1));

                let second_column = point.iter().enumerate().take(point.len() - 1).fold(
                    E::ZERO,
                    |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index) * point[point.len() - 1]
                    },
                );
                Ok(vec![first_column, second_column])
            }
            TableType::Clamping(size) => {
                if point.len() != *size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a clamping table evaluation, point size: {}, expected: {}",
                        point.len(),
                        size
                    )));
                }

                let first_column = point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                    acc + *p * E::from_canonical_u64(1u64 << index)
                }) - E::from_canonical_u64(1u64 << (size - 1));

                let max = 1i128 << (size - 1);
                let min = -max;

                let second_col_eval = (min..max)
                    .map(|i| {
                        let out = if i < *quantization::MIN {
                            *quantization::MIN
                        } else if i > *quantization::MAX {
                            *quantization::MAX
                        } else {
                            i
                        };

                        let out_field: E = out.to_field();
                        out_field.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>()
                    .into_mle()
                    .evaluate(point);

                Ok(vec![first_column, second_col_eval])
            }
            TableType::Softmax(table_data) => {
                let size = table_data.size();
                if point.len() != size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a softmax table evaluation, point size: {}, expected: {}",
                        point.len(),
                        size
                    )));
                }

                Ok(vec![
                    point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index)
                    }),
                ])
            }
            TableType::ErrorTable(..) => Ok(vec![]),
            TableType::ZeroTable(bit_size) => {
                if point.len() != *bit_size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a softmax table evaluation, point size: {}, expected: {}",
                        point.len(),
                        bit_size
                    )));
                }

                let (in_column_eval, out_column_eval) = point.iter().enumerate().fold(
                    (E::ZERO, E::ONE),
                    |(in_acc, out_acc), (index, p)| {
                        (
                            in_acc + *p * E::from_canonical_u64(1u64 << index),
                            out_acc * (E::ONE - *p),
                        )
                    },
                );
                Ok(vec![in_column_eval, out_column_eval])
            }
        }
    }

    pub fn generate_challenge<E: ExtensionField, T: Transcript<E>>(&self, transcript: &mut T) -> E {
        match self {
            TableType::Relu => transcript.get_and_append_challenge(b"Relu").elements,
            TableType::Range | TableType::ErrorTable(..) => {
                // Theres only one column for a range check so we don't need to generate a challenge
                E::ONE
            }
            TableType::Clamping(_) => transcript.get_and_append_challenge(b"Clamping").elements,
            TableType::Softmax(..) => transcript.get_and_append_challenge(b"Softmax").elements,
            TableType::ZeroTable(..) => transcript.get_and_append_challenge(b"Zero").elements,
        }
    }

    /// Gets the number of variables that the multiplicity polynomial will have for this table
    pub fn multiplicity_poly_vars(&self) -> usize {
        match self {
            TableType::Range | TableType::Relu => *quantization::BIT_LEN,
            TableType::Clamping(bits) => *bits,
            TableType::Softmax(table_data) => table_data.size(),
            TableType::ErrorTable(_, allowable_error) => ceil_log2(2 * *allowable_error as usize),
            TableType::ZeroTable(bits) => *bits,
        }
    }

    /// Function that returns any MLEs that have to be committed for this [`TableType`]
    pub fn committed_columns<E: ExtensionField>(&self) -> Option<DenseMultilinearExtension<E>> {
        match self {
            TableType::Softmax(table_data) => {
                let table_size = table_data.full_table_size();

                let out_column = (0i128..table_size)
                    .map(|j| {
                        let out_elem = table_data.table_output(j);
                        let out_field: E = out_elem.to_field();
                        out_field.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>();
                Some(DenseMultilinearExtension::<E>::from_evaluations_vec(
                    table_data.size(),
                    out_column,
                ))
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                // Work out the minimum and maximum elements of the table
                let table_min = quant_one - allowable_error;
                let table_max = quant_one + allowable_error;
                // Work out the full table size
                let num_vars = ceil_log2(2 * *allowable_error as usize);
                let table_size = 1usize << num_vars;
                let column = (table_min..=table_max)
                    .map(|elem| {
                        let f: E = elem.to_field();
                        f.as_bases()[0]
                    })
                    .chain(std::iter::repeat(E::BaseField::ZERO))
                    .take(table_size)
                    .collect::<Vec<E::BaseField>>();
                Some(DenseMultilinearExtension::<E>::from_evaluations_vec(
                    num_vars, column,
                ))
            }
            _ => None,
        }
    }

    /// Method that takes all of the claims output by a logup table proof and outputs only those that need to be checked via commitment opening (excluding the multiplicity poly claim)
    pub fn table_claims<E: ExtensionField>(&self, claims: &[Claim<E>]) -> Vec<Claim<E>> {
        match self {
            TableType::Softmax(..) | TableType::ErrorTable(..) => {
                // For Softmax and Error Table we just need the output column claim so the last of the slice
                vec![claims.last().cloned().unwrap()]
            }

            _ => vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    tables: Vec<TableType>,
}

impl LookupContext {
    pub fn new(set: &BTreeSet<TableType>) -> LookupContext {
        LookupContext {
            tables: set.iter().copied().collect(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &TableType> {
        self.tables.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

pub struct LookupWitnessGen<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    pub(crate) new_lookups: BTreeMap<TableType, Vec<Element>>,
    pub(crate) logup_witnesses: HashMap<NodeId, Vec<LogUpWitness<E, PCS>>>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LookupWitnessGen<E, PCS> {
    pub fn new(lookup_ctx: &LookupContext) -> Self {
        let new_lookups = lookup_ctx
            .iter()
            .map(|&table_type| (table_type, Vec::<Element>::new()))
            .collect::<BTreeMap<TableType, Vec<Element>>>();
        Self {
            new_lookups,
            logup_witnesses: HashMap::new(),
        }
    }
}

pub(crate) const COLUMN_SEPARATOR: Element = 1i128 << 32;

pub struct LookupWitness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    pub challenge_storage: ChallengeStorage<E>,

    pub logup_witnesses: HashMap<NodeId, Vec<LogUpWitness<E, PCS>>>,

    pub table_witnesses: Vec<LogUpWitness<E, PCS>>,
}

pub fn generate_lookup_witnesses<'a, E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    trace: &InferenceTrace<'a, E, Element>,
    ctx: &Context<E, PCS>,
    transcript: &mut T,
) -> Result<LookupWitness<E, PCS>, LogUpError>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    // If the lookup context is empty then there are no lookup witnesses to generate so we return default values
    if ctx.lookup.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok(LookupWitness {
            challenge_storage: ChallengeStorage {
                constant_challenge: E::ZERO,
                challenge_map: HashMap::new(),
            },
            logup_witnesses: HashMap::new(),
            table_witnesses: vec![],
        });
    }

    // Make the witness gen struct that stores relevant table lookup data
    debug!("== Witness poly fields generation ==");
    let metrics = Metrics::new();
    let mut witness_gen = LookupWitnessGen::<E, PCS>::new(&ctx.lookup);

    for (node_id, _) in ctx.steps_info.to_forward_iterator() {
        let step = trace
            .get_step(&node_id)
            .ok_or(LogUpError::ProvingError(format!(
                "Node {node_id} not found in trace"
            )))?;
        step.op
            .gen_lookup_witness(node_id, &mut witness_gen, ctx, &step.step_data)
            .map_err(|e| {
                LogUpError::ParameterError(format!(
                    "Error generating lookup witness for node {node_id} with error: {e}"
                ))
            })?;
    }
    debug!(
        "== Witness poly fields generation metrics {} ==",
        metrics.to_span()
    );

    debug!("== Witness table multiplicities generation ==");
    let metrics = Metrics::new();
    // calculate the table multiplicities
    let table_witnesses = witness_gen
        .new_lookups
        .par_iter()
        .map(|(table_type, lookups)| {
            let table_lookup_data =
                lookups
                    .iter()
                    .fold(HashMap::<Element, u64>::new(), |mut map, elem| {
                        *map.entry(*elem).or_insert(0) += 1;
                        map
                    });
            let (table_column, column_evals) =
                table_type.get_merged_table_column::<E>(COLUMN_SEPARATOR);

            // Check to see that all the lookup values are present in the table
            #[cfg(test)]
            {
                for key in table_lookup_data.keys() {
                    let check = table_column.contains(key);
                    if !check {
                        println!(
                            "Tried to lookup key: {}, for table: {}",
                            key,
                            table_type.name()
                        );
                    }
                }
            }
            // We have to account for repeated entries in the lookup table. This is usually the case if the table we want to lookup from is not a power of two size, in that case we pick a row from the table
            // and repeat it until the table has the desired size.
            let table_column_map =
                table_column
                    .iter()
                    .fold(BTreeMap::<Element, u64>::new(), |mut map, elem| {
                        *map.entry(*elem).or_insert(0) += 1;
                        map
                    });
            let multiplicities = table_column
                .iter()
                .map(|table_val| {
                    if let Some(lookup_count) = table_lookup_data.get(table_val) {
                        let table_count = *table_column_map.get(table_val).unwrap();
                        let inv = if table_count != 1 {
                            E::BaseField::from_canonical_u64(table_count).inverse()
                        } else {
                            E::BaseField::ONE
                        };
                        E::BaseField::from_canonical_u64(*lookup_count) * inv
                    } else {
                        E::BaseField::ZERO
                    }
                })
                .collect::<Vec<E::BaseField>>();
            let num_vars = ceil_log2(multiplicities.len());
            let mle =
                DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &multiplicities);
            let commit = ctx.commitment_ctx.commit(&mle).map_err(|e| {
                LogUpError::PolynomialError(format!(
                    "Error while committing to {} table multiplicity polynomial: {:?}",
                    table_type.name(),
                    e
                ))
            })?;
            Ok(LogUpWitness::<E, PCS>::new_table(
                (commit, mle),
                multiplicities,
                column_evals,
                *table_type,
            ))
        })
        .collect::<Result<Vec<LogUpWitness<E, PCS>>, LogUpError>>()?;

    debug!(
        "== Witness table multiplicities metrics {} ==",
        metrics.to_span()
    );

    debug!("== Challenge storage ==");
    let metrics = Metrics::new();
    let challenge_storage =
        initialise_from_table_set::<E, T, _>(witness_gen.new_lookups.keys(), transcript);
    debug!("== Challenge storage metrics {} ==", metrics.to_span());

    Ok(LookupWitness {
        challenge_storage,
        logup_witnesses: witness_gen.logup_witnesses,
        table_witnesses,
    })
}

fn initialise_from_table_set<
    'a,
    E: ExtensionField,
    T: Transcript<E>,
    I: Iterator<Item = &'a TableType>,
>(
    set: I,
    transcript: &mut T,
) -> ChallengeStorage<E> {
    let constant_challenge = transcript
        .get_and_append_challenge(b"table_constant")
        .elements;
    let challenge_map = set
        .map(|table_type| {
            let challenge = table_type.generate_challenge(transcript);

            (table_type.name(), challenge)
        })
        .collect::<HashMap<String, E>>();
    ChallengeStorage::<E> {
        constant_challenge,
        challenge_map,
    }
}
