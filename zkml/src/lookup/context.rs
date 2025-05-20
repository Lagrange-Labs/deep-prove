//! File containg code for lookup witness generation.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ff::Field;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;

use rayon::prelude::*;

use crate::{
    Element,
    commit::{PCSError},
    iop::ChallengeStorage,
    layers::{Layer, activation::Relu, requant::RequantLookupWitness},
    lookup::{witness::LogUpWitness},
    model::InferenceTrace,
    quantization::{self, Fieldizer},
};

use super::logup_gkr::error::LogUpError;
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
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::Range => "Range".to_string(),
            TableType::Clamping(size) => format!("Clamping: {}", size),
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
            TableType::Clamping(size) => {
                if point.len() != *size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a clamping table evaluation, point size: {}, expected: {}",
                        point.len(),
                        size
                    )));
                }

                let first_column = point
                    .iter()
                    .enumerate()
                    .fold(E::ZERO, |acc, (index, p)| acc + *p * E::from(1u64 << index))
                    - E::from(1u64 << (size - 1));

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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    tables: BTreeSet<TableType>,
}

impl LookupContext {
    pub fn new(set: BTreeSet<TableType>) -> LookupContext {
        LookupContext { tables: set }
    }

    pub fn iter(&self) -> impl Iterator<Item = &TableType> {
        self.tables.iter()
    }
}

pub fn generate_lookup_witnesses<E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    ctx: &crate::iop::context::Context<E, PCS>,
    trace: &InferenceTrace<Element, E>,
    transcript: &mut T,
) -> Result<
    (
        ChallengeStorage<E>,
        Vec<LogUpWitness<E, PCS>>,
        Vec<LogUpWitness<E, PCS>>,
    ),
    LogUpError,
>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    let tables = &ctx.lookup.tables;

    if tables.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok((
            ChallengeStorage {
                constant_challenge: E::ZERO,
                challenge_map: HashMap::new(),
            },
            vec![],
            vec![],
        ));
    }

    let column_separator = 1i128 << 32;
    debug!("Lookup witness generation: generating poly fields...");

    // We iterate through the trace in parallel, for each layer that requires a lookup we commit to all needed polynomials,
    // calculate all the values that will be looked up and then return a HashMap that stores all the lookups into each table
    // together with a vector of witnesses that should iterate in reverse trace order (i.e. the first element in the vector is the witness for the
    // last layer in the trace that requires a lookup).
    let now = std::time::Instant::now();
    let (multiplicity_map, witnesses) = trace
        .iter()
        .map(|(step_input, step)| {
            match step.layer {
                Layer::Activation(..) => {
                    if !tables.contains(&TableType::Relu) {
                        return Err(PCSError::ParameterError("Model context did not contain a Relu table so shouldn't be producing a Relu witness".to_string()));
                    }
                    // Calculate the column_evals and also the merged lookups
                    let (merged_lookups, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) =
                        step_input
                            .get_data()
                            .iter()
                            .zip(step.output.get_data().iter())
                            .map(|(a, b)| {
                                let a_field: E = a.to_field();
                                let b_field: E = b.to_field();
                                (
                                    a + column_separator * b,
                                    (a_field.as_bases()[0], b_field.as_bases()[0]),
                                )
                            })
                            .unzip();

                    let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                        field.into_iter().unzip();

                    let num_vars = ceil_log2(col_one.len());

                    // Add the witness polynomials that we need to commit to
                    let (commits, column_evals): (Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>, Vec<Vec<E::BaseField>>) = [col_one, col_two]
                        .into_par_iter()
                        .map(|evaluations| {
                            let mle = DenseMultilinearExtension::<E>::from_evaluations_slice(
                                num_vars,
                                &evaluations,
                            );
                            let commit = ctx.weights.commit(&mle)?;
                            Ok(((commit, mle), evaluations))
                        })
                        .collect::<Result<Vec<_>, PCSError>>()?.into_iter().unzip();

                    Result::<
                            (Vec<(Vec<Element>, TableType)>, Vec<LogUpWitness<E, PCS>>),
                            PCSError,
                        >::Ok((
                            vec![(merged_lookups, TableType::Relu)],
                            vec![LogUpWitness::<E, PCS>::new_lookup(
                                commits,
                                column_evals,
                                2,
                                TableType::Relu,
                            )],
                        ))
                }

                Layer::Requant(requant) => {
                    if !tables.contains(&TableType::Clamping(requant.clamping_size())) {
                        return Err(PCSError::ParameterError(format!("Model context did not contain a Clamping table of size {}, so shouldn't be producing a Clamping witness", requant.clamping_size())));
                    }
                    if !tables.contains(&TableType::Range) {
                        return Err(PCSError::ParameterError("Model context did not contain a Range table, so shouldn't be producing a Range witness".to_string()));
                    }
                    let RequantLookupWitness {
                        clamping_in,
                        clamping_out,
                        shifted_chunks,
                    } = requant.gen_lookup_witness::<E>(step_input.get_data());

                    let merged_shifted = shifted_chunks
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<Element>>();

                    let merged_clamping = clamping_in
                        .iter()
                        .zip(clamping_out.iter())
                        .map(|(&c_in, &c_out)| c_in + c_out * column_separator)
                        .collect::<Vec<Element>>();
                    let num_vars = ceil_log2(clamping_in.len());
                    // Add the witnesses to be committed

                    let (clamping_commits, clamping_evals): (Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>, Vec<Vec<E::BaseField>>) = [clamping_in, clamping_out]
                        .into_par_iter()
                        .map(|vals| {
                            let evaluations = vals
                                .into_iter()
                                .map(|v| {
                                    let f: E = v.to_field();
                                    f.as_bases()[0]
                                })
                                .collect::<Vec<E::BaseField>>();
                            let mle = DenseMultilinearExtension::<E>::from_evaluations_slice(
                                num_vars,
                                &evaluations,
                            );
                            let commit = ctx.weights.commit(&mle)?;
                            Ok(((commit, mle), evaluations))
                        })
                        .collect::<Result<Vec<_>, PCSError>>()?.into_iter().unzip();

                    let (shifted_chunk_commits, shifted_chunks_evals): (Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>, Vec<Vec<E::BaseField>>) = shifted_chunks
                        .into_par_iter()
                        .map(|chunk| {
                            let evaluations = chunk
                                .into_iter()
                                .map(|v| {
                                    let f: E = v.to_field();
                                    f.as_bases()[0]
                                })
                                .collect::<Vec<E::BaseField>>();
                            let mle = DenseMultilinearExtension::<E>::from_evaluations_slice(
                                num_vars,
                                &evaluations,
                            );
                            let commit = ctx.weights.commit(&mle)?;
                            Ok(((commit, mle), evaluations))
                        })
                        .collect::<Result<Vec<_>, PCSError>>()?.into_iter().unzip();
                    Ok((
                        vec![
                            (
                                merged_clamping,
                                TableType::Clamping(requant.clamping_size()),
                            ),
                            (merged_shifted, TableType::Range),
                        ],
                        vec![
                            LogUpWitness::<E, PCS>::new_lookup(
                                clamping_commits,
                                clamping_evals,
                                2,
                                TableType::Clamping(requant.clamping_size()),
                            ),
                            LogUpWitness::<E, PCS>::new_lookup(
                                shifted_chunk_commits,
                                shifted_chunks_evals,
                                1,
                                TableType::Range,
                            ),
                        ],
                    ))
                }
                Layer::Pooling(pooling) => {
                    if !tables.contains(&TableType::Range) {
                        return Err(PCSError::ParameterError("Model context did not contain a Range table, so shouldn't be producing a Range witness".to_string()));
                    }
                    let (merged_lookups, column_evals) =
                        pooling.gen_lookup_witness::<E>(step_input);
                    
                    // Commit to the witnes polys
                    let output_poly = step
                        .output
                        .get_data()
                        .iter()
                        .map(|val| {
                            let f: E = val.to_field();
                            f.as_bases()[0]
                        })
                        .collect::<Vec<E::BaseField>>();
                    let num_vars = ceil_log2(output_poly.len());
                    
                    let commit_evals = column_evals.iter().chain(std::iter::once(&output_poly)).collect::<Vec<_>>();
                    let commits = commit_evals
                        .into_par_iter()
                        .map(|evaluations| {
                            let mle = DenseMultilinearExtension::<E>::from_evaluations_slice(
                                num_vars,
                                evaluations,
                            );
                            let commit = ctx.weights.commit(&mle)?;
                            Ok((commit, mle))
                        })
                        .collect::<Result<Vec<_>, PCSError>>()?;

                    Ok((vec![(merged_lookups, TableType::Range)], vec![
                        LogUpWitness::<E, PCS>::new_lookup(commits, column_evals, 1, TableType::Range),
                    ]))
                }
                _ => Ok((vec![], vec![])),
            }
        })
        .try_fold(
            (BTreeMap::<TableType, Vec<Element>>::default(), vec![]),
            |(mut multiplicity_map, mut witnesses), items| {
                let (inner_lookups, inner_witnesses): (
                    Vec<(Vec<Element>, TableType)>,
                    Vec<LogUpWitness<E, PCS>>,
                ) = items?;
                inner_lookups.into_iter().for_each(|(lookups, table_type)| {
                    multiplicity_map
                        .entry(table_type)
                        .or_insert_with(|| Vec::<Element>::default())
                        .extend(lookups);
                });
                witnesses.extend(inner_witnesses);
                Result::<
                    (
                        BTreeMap<TableType, Vec<Element>>,
                        Vec<LogUpWitness<E, PCS>>,
                    ),
                    PCSError,
                >::Ok((multiplicity_map, witnesses))
            },
        )?;
        // .try_reduce(
        //     || (BTreeMap::<TableType, Vec<Element>>::default(), vec![]),
        //     |(mut hashmap_a, mut witness_a), (hashmap_b, witness_b)| {
        //         hashmap_a.extend(hashmap_b);
        //         witness_a.extend(witness_b);
        //         Ok((hashmap_a, witness_a))
        //     },
        // )?;
    println!("time to commit to witness: {:?}", now.elapsed());
    debug!("Lookup witness generation: generating table multiplicities...");
    let now = std::time::Instant::now();
    let table_witnesses =
        multiplicity_map
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
                    table_type.get_merged_table_column::<E>(column_separator);

                let multiplicities = table_column
                    .iter()
                    .map(|table_val| {
                        if let Some(lookup_count) = table_lookup_data.get(table_val) {
                            E::BaseField::from(*lookup_count)
                        } else {
                            E::BaseField::ZERO
                        }
                    })
                    .collect::<Vec<E::BaseField>>();
                let num_vars = ceil_log2(multiplicities.len());
                let mle = DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &multiplicities);
                let commit = ctx.weights.commit(
                    &mle,
                )?;
                Ok(LogUpWitness::<E, PCS>::new_table(
                    (commit, mle),
                    multiplicities,
                    column_evals,
                    *table_type,
                ))
            })
            .collect::<Result<Vec<LogUpWitness<E, PCS>>, PCSError>>()?;
        println!("Time to commit to multiplicities: {:?}", now.elapsed());
    // Write all the witness commitments to the transcript
    witnesses
        .iter()
        .chain(table_witnesses.iter())
        .try_for_each(|witness| witness.write_to_transcript(transcript))?;
    
    debug!("Lookup witness generation: challenge storage...");
    let challenge_storage = initialise_from_table_set::<E, T>(&tables, transcript);

    Ok((challenge_storage, witnesses, table_witnesses))
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
