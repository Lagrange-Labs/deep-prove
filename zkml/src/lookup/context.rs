//! File containing code for lookup witness generation.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
};
use p3_field::FieldAlgebra;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Enum used for establishing the different table types needed to prove non-linear functions in a model.
pub enum TableType {
    /// Table used for the Relu activation function
    Relu,
    /// Table used for range checking (its size is determined by the quantisation bit size)
    Range,
    /// Table used for clamping values, the inner [`usize`] denotes the maximum bit length a value can be before clamping to use this table
    Clamping(usize),
    /// Table type used for computing Softmax, the first inner [`usize`] denotes the value such that the temprature is 1/(sqrt(val)), the second [`usize`] is the table size and the [`Element`] is the point at which we map everything to zero
    Softmax(usize, usize, Element),
}

impl TableType {
    fn get_merged_table_column<E: ExtensionField>(
        &self,
        column_separator: Element,
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        match self {
            TableType::Relu => {
                let (comb, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) =
                    (*quantization::MIN - 1..=*quantization::MAX)
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
                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                    field.into_iter().unzip();
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
                let (comb, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) = (min..max)
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
                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                    field.into_iter().unzip();
                (comb, vec![col_one, col_two])
            }
            TableType::Softmax(val, size, bkm) => {
                let float_temperature = 1.0f32 / (*val as f32).sqrt();
                let table_size = 1i128 << size;
                let base = 1i128 << (LOG_SCALE_FACTOR - 8);
                let (merged_lookup, (in_column, out_column)): (
                    Vec<Element>,
                    (Vec<E::BaseField>, Vec<E::BaseField>),
                ) = (0i128..table_size)
                    .map(|j| {
                        let prod = base * j;
                        let out_elem = if prod > *bkm {
                            0i128
                        } else {
                            let float_exp =
                                (-prod as f32 / (SCALE_FACTOR as f32 * float_temperature)).exp();
                            (float_exp * OUTPUT_SCALE_FACTOR as f32).round() as Element
                        };
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
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::Range => "Range".to_string(),
            TableType::Clamping(size) => format!("Clamping: {}", size),
            TableType::Softmax(temp, ..) => format!("Softmax - temperature: {}", temp),
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
            TableType::Softmax(_, size, ..) => {
                if point.len() != *size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a softmax table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *size
                    )));
                }

                Ok(vec![
                    point
                        .iter()
                        .enumerate()
                        .fold(E::ZERO, |acc, (index, p)| acc + *p * E::from(1u64 << index)),
                ])
            }
        }
    }

    pub fn generate_challenge<E: ExtensionField, T: Transcript<E>>(&self, transcript: &mut T) -> E {
        match self {
            TableType::Relu => transcript.get_and_append_challenge(b"Relu").elements,
            TableType::Range => {
                // Theres only one column for a range check so we don't need to generate a challenge
                E::ONE
            }
            TableType::Clamping(_) => transcript.get_and_append_challenge(b"Clamping").elements,
            TableType::Softmax(..) => transcript.get_and_append_challenge(b"Softmax").elements,
        }
    }

    /// Gets the number of variables that the multiplicity polynomial will have for this table
    pub fn multiplicity_poly_vars(&self) -> usize {
        match self {
            TableType::Range | TableType::Relu => *quantization::BIT_LEN,
            TableType::Clamping(bits) => *bits,
            TableType::Softmax(_, bits, _) => *bits,
        }
    }

    /// Function that returns any MLEs that have to be committed for this [`TableType`]
    pub fn committed_columns<E: ExtensionField>(&self) -> Option<DenseMultilinearExtension<E>> {
        match self {
            TableType::Softmax(val, size, bkm) => {
                let float_temperature = 1.0f32 / (*val as f32).sqrt();
                let table_size = 1i128 << size;
                let base = 1i128 << (LOG_SCALE_FACTOR - 8);
                let out_column = (0i128..table_size)
                    .map(|j| {
                        let prod = base * j;
                        let out_elem = if prod > *bkm {
                            0i128
                        } else {
                            let float_exp =
                                (-prod as f32 / (SCALE_FACTOR as f32 * float_temperature)).exp();
                            (float_exp * OUTPUT_SCALE_FACTOR as f32).round() as Element
                        };

                        let out_field: E = out_elem.to_field();
                        out_field.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>();
                Some(DenseMultilinearExtension::<E>::from_evaluations_vec(
                    *size, out_column,
                ))
            }
            _ => None,
        }
    }

    /// Method that takes all of the claims output by a logup table proof and outputs only those that need to be checked via commitment opening (excluding the multiplicity poly claim)
    pub fn table_claims<E: ExtensionField>(&self, claims: &[Claim<E>]) -> Vec<Claim<E>> {
        match self {
            TableType::Softmax(..) => {
                // For Softmax we just need the output column claim so the last of the slice
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

pub fn generate_lookup_witnesses<
    'a,
    E: ExtensionField,
    T: Transcript<E>,
    PCS: PolynomialCommitmentScheme<E>,
>(
    trace: &InferenceTrace<'a, E, Element>,
    ctx: &Context<E, PCS>,
    transcript: &mut T,
) -> Result<
    (
        ChallengeStorage<E>,
        HashMap<NodeId, Vec<LogUpWitness<E, PCS>>>,
        Vec<LogUpWitness<E, PCS>>,
    ),
    LogUpError,
>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    // If the lookup context is empty then there are no lookup witnesses to generate so we return default values
    if ctx.lookup.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok((
            ChallengeStorage {
                constant_challenge: E::ZERO,
                challenge_map: HashMap::new(),
            },
            HashMap::new(),
            vec![],
        ));
    }

    // Make the witness gen struct that stores relevant table lookup data
    let mut witness_gen = LookupWitnessGen::<E, PCS>::new(&ctx.lookup);

    debug!("Lookup witness generation: generating poly fields...");
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
                    "Error generating lookup witness for node {} with error: {}",
                    node_id,
                    e.to_string()
                ))
            })?;
    }

    debug!("Lookup witness generation: generating table multiplicities...");
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

            let multiplicities = table_column
                .iter()
                .map(|table_val| {
                    if let Some(lookup_count) = table_lookup_data.get(table_val) {
                        E::BaseField::from_canonical_u64(*lookup_count)
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

    debug!("Lookup witness generation: commit context generation...");

    debug!("Lookup witness generation: challenge storage...");
    let challenge_storage =
        initialise_from_table_set::<E, T, _>(witness_gen.new_lookups.keys(), transcript);

    Ok((
        challenge_storage,
        witness_gen.logup_witnesses,
        table_witnesses,
    ))
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
