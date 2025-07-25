//! Code used to define structs relating to witnesses for lookup layers

use anyhow::{Result, anyhow};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{iop::ChallengeStorage, lookup::context::ProverCommitment};

use super::{
    context::TableType,
    logup_gkr::{error::LogUpError, structs::LogUpInput},
};

#[derive(Clone, Debug)]
/// Enum used for storing witness data for LogUp GKR, it is equivalent to [`LogUpInput`] but with no challenges stored.
pub enum LogUpWitness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    /// Lookup variant can have multiple instances in one [`LogUpInput::Lookup`], `columns_per_instance` is used to work out how many batches we need to prove.
    Lookup {
        commits: Vec<ProverCommitment<PCS, E>>,
        column_evals: Vec<Vec<E::BaseField>>,
        columns_per_instance: usize,
        table_type: TableType,
    },
    /// Input for a Table proof.
    Table {
        multiplicity_commit: ProverCommitment<PCS, E>,
        multiplicity_evals: Vec<E::BaseField>,
        column_evals: Vec<Vec<E::BaseField>>,
        table_type: TableType,
    },
}

impl<E, PCS> LogUpWitness<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Creates a new lookup witness
    pub fn new_lookup(
        commits: Vec<ProverCommitment<PCS, E>>,
        column_evals: Vec<Vec<E::BaseField>>,
        columns_per_instance: usize,
        table_type: TableType,
    ) -> LogUpWitness<E, PCS> {
        LogUpWitness::Lookup {
            commits,
            column_evals,
            columns_per_instance,
            table_type,
        }
    }

    /// Creates a new table witness
    pub fn new_table(
        multiplicity_commit: ProverCommitment<PCS, E>,
        multiplicity_evals: Vec<E::BaseField>,
        column_evals: Vec<Vec<E::BaseField>>,
        table_type: TableType,
    ) -> LogUpWitness<E, PCS> {
        LogUpWitness::Table {
            multiplicity_commit,
            multiplicity_evals,
            column_evals,
            table_type,
        }
    }

    /// Writes this witness to the transcript
    pub fn write_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        match self {
            LogUpWitness::Lookup { commits, .. } => commits.iter().try_for_each(|comm| {
                PCS::write_commitment(&PCS::get_pure_commitment(&comm.0), transcript)
                    .map_err(|e| anyhow!("Could not write lookup witness to transcript: {:?}", e))
            }),
            LogUpWitness::Table {
                multiplicity_commit,
                ..
            } => PCS::write_commitment(
                &PCS::get_pure_commitment(&multiplicity_commit.0),
                transcript,
            )
            .map_err(|e| anyhow!("Could not write table witness to transcript: {:?}", e)),
        }
    }

    /// Getter for the [`TableType`]
    pub fn table_type(&self) -> TableType {
        match self {
            LogUpWitness::Lookup { table_type, .. } | LogUpWitness::Table { table_type, .. } => {
                table_type.clone()
            }
        }
    }

    /// Extracts the [`LogUpInput`] given a [`ChallengeStorage`]
    pub fn get_logup_input(
        &self,
        challenge_storage: &ChallengeStorage<E>,
    ) -> Result<LogUpInput<E>, LogUpError> {
        match self {
            LogUpWitness::Lookup {
                column_evals,
                columns_per_instance,
                table_type,
                ..
            } => {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(LogUpError::ParameterError(format!(
                        "No challenges for table type: {}",
                        table_type.name()
                    )))?;
                LogUpInput::<E>::new_lookup(
                    column_evals.clone(),
                    constant_challenge,
                    column_separation_challenge,
                    *columns_per_instance,
                )
            }
            LogUpWitness::Table {
                multiplicity_evals,
                column_evals,
                table_type,
                ..
            } => {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(LogUpError::ParameterError(format!(
                        "No challenges for table type: {}",
                        table_type.name()
                    )))?;
                LogUpInput::<E>::new_table(
                    column_evals.clone(),
                    multiplicity_evals.clone(),
                    constant_challenge,
                    column_separation_challenge,
                )
            }
        }
    }

    /// Consumes the witness and returns its commitments.
    pub fn into_commitments(self) -> Vec<ProverCommitment<PCS, E>> {
        match self {
            LogUpWitness::Lookup { commits, .. } => commits,
            LogUpWitness::Table {
                multiplicity_commit,
                ..
            } => vec![multiplicity_commit],
        }
    }
}
