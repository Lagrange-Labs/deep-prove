//! File containg code for lookup witness generation.

use std::collections::{BTreeSet, HashMap};

use ff::Field;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::Context,
    iop::ChallengeStorage,
    layers::{
        activation::Relu,
        provable::{NodeId, ProvableOp},
    },
    lookup::logup_gkr::structs::LogUpInput,
    model::{InferenceTrace, ModelCtx, ToIterator},
    quantization::{self, Fieldizer},
};

use super::logup_gkr::error::LogUpError;
pub const TABLE_POLY_ID_OFFSET: usize = 666;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TableType {
    Relu,
    Range,
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
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::Range => "Range".to_string(),
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
                    point
                        .iter()
                        .enumerate()
                        .fold(E::ZERO, |acc, (index, p)| acc + *p * E::from(1u64 << index)),
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

                let first_column = point
                    .iter()
                    .enumerate()
                    .fold(E::ZERO, |acc, (index, p)| acc + *p * E::from(1u64 << index))
                    - E::from(1u64 << (*quantization::BIT_LEN - 1));

                let second_column = point
                    .iter()
                    .enumerate()
                    .take(point.len() - 1)
                    .fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from(1u64 << index) * point[point.len() - 1]
                    });
                Ok(vec![first_column, second_column])
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
}

pub struct LookupWitnessGen<E: ExtensionField> {
    pub(crate) tables: BTreeSet<TableType>,
    pub(crate) lookups: HashMap<TableType, HashMap<Element, u64>>,
    pub(crate) polys_with_id: Vec<(usize, Vec<E>)>,
    pub(crate) lookups_no_challenges: HashMap<NodeId, (Vec<Vec<E::BaseField>>, usize, TableType)>,
}

impl<E: ExtensionField> Default for LookupWitnessGen<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: ExtensionField> LookupWitnessGen<E> {
    pub fn new() -> Self {
        Self {
            tables: BTreeSet::new(),
            lookups: HashMap::new(),
            polys_with_id: Vec::new(),
            lookups_no_challenges: HashMap::new(),
        }
    }
}

pub(crate) const COLUMN_SEPARATOR: Element = 1i128 << 32;

pub fn generate_lookup_witnesses<'a, E, T: Transcript<E>>(
    trace: &InferenceTrace<'a, E, Element>,
    ctx: &ModelCtx<E>,
    transcript: &mut T,
) -> Result<
    (
        Option<Context<E>>,
        ChallengeStorage<E>,
        HashMap<NodeId, LogUpInput<E>>,
        Vec<LogUpInput<E>>,
    ),
    LogUpError,
>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    let mut witness_gen = LookupWitnessGen::<E>::new();

    debug!("Lookup witness generation: generating poly fields...");
    for (node_id, _) in ctx.to_forward_iterator() {
        let step = trace
            .get_step(&node_id)
            .ok_or(LogUpError::ProvingError(format!(
                "Node {node_id} not found in trace"
            )))?;
        step.op
            .gen_lookup_witness(node_id, &mut witness_gen, &step.step_data)
            .map_err(|e| {
                LogUpError::ParameterError(format!(
                    "Error generating lookup witness for node {node_id} with error: {e}"
                ))
            })?;
    }

    if witness_gen.tables.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok((
            None,
            ChallengeStorage {
                constant_challenge: E::ZERO,
                challenge_map: HashMap::new(),
            },
            HashMap::new(),
            vec![],
        ));
    }

    debug!("Lookup witness generation: generating table multiplicities...");
    // calculate the table multiplicities
    let tables_no_challenges = witness_gen.tables.iter().enumerate().map(|(i,table_type)| {
        let (table_column, column_evals) = table_type.get_merged_table_column::<E>(COLUMN_SEPARATOR);

        let table_lookup_data = witness_gen.lookups.get(table_type).ok_or(LogUpError::ParameterError(format!("Tried to retrieve lookups for a table of type: {table_type:?}, but no table of that type exists")))?;

        let (multiplicities, mults_ext)  = table_column.iter().map(|table_val| {
            if let Some(lookup_count) = table_lookup_data.get(table_val) {
                (E::BaseField::from(*lookup_count), E::from(*lookup_count))
            } else {
                (E::BaseField::ZERO, E::ZERO)
            }
        }).unzip();

        witness_gen.polys_with_id.push((i + TABLE_POLY_ID_OFFSET, mults_ext));
        Ok((column_evals, multiplicities, *table_type))
    }).collect::<Result<Vec<(Vec<Vec<E::BaseField>>, Vec<E::BaseField>, TableType)>, LogUpError>>()?;

    debug!("Lookup witness generation: commit context generation...");
    let ctx = Context::generate(witness_gen.polys_with_id).map_err(|e| {
        LogUpError::ParameterError(format!(
            "Could not generate Lookup witness commit context {{ inner: {e:?}}}"
        ))
    })?;

    ctx.write_to_transcript(transcript).map_err(|e| {
        LogUpError::ParameterError(format!(
            "Unable to write lookup witness commit context to transcript, {{ inner: {e:?}}}"
        ))
    })?;

    debug!("Lookup witness generation: challenge storage...");
    let challenge_storage = initialise_from_table_set::<E, T>(&witness_gen.tables, transcript);

    let lookup_inputs = witness_gen
        .lookups_no_challenges
        .into_iter()
        .map(
            |(node_id, (column_evals, columns_per_instance, table_type))| {
                let (constant_challenge, column_challenge) = challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(LogUpError::ParameterError(format!(
                        "No challegnes found for table type: {} when generating lookup witness",
                        table_type.name()
                    )))?;

                Ok((
                    node_id,
                    LogUpInput::<E>::new_lookup(
                        column_evals,
                        constant_challenge,
                        column_challenge,
                        columns_per_instance,
                    )?,
                ))
            },
        )
        .collect::<Result<HashMap<NodeId, LogUpInput<E>>, LogUpError>>()?;

    let table_inputs = tables_no_challenges
        .into_iter()
        .map(|(column_evals, multiplicities, table_type)| {
            let (constant_challenge, column_challenge) = challenge_storage
                .get_challenges_by_name(&table_type.name())
                .ok_or(LogUpError::ParameterError(format!(
                    "No challegnes found for table type: {} when generating table witness",
                    table_type.name()
                )))?;

            LogUpInput::<E>::new_table(
                column_evals,
                multiplicities,
                constant_challenge,
                column_challenge,
            )
        })
        .collect::<Result<Vec<LogUpInput<E>>, LogUpError>>()?;
    Ok((Some(ctx), challenge_storage, lookup_inputs, table_inputs))
}

fn initialise_from_table_set<E: ExtensionField, T: Transcript<E>>(
    set: &BTreeSet<TableType>,
    transcript: &mut T,
) -> ChallengeStorage<E> {
    let constant_challenge = transcript
        .get_and_append_challenge(b"table_constant")
        .elements;
    let challenge_map = set
        .iter()
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
