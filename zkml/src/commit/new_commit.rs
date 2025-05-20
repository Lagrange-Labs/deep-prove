//! This module contains logic to prove the correct opening of several claims from several independent
//! polynomials.

use std::collections::HashMap;

use super::PCSError;
use crate::Claim;
use ff_ext::ExtensionField;

use mpcs::{Evaluation, PolynomialCommitmentScheme};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, FieldType, MultilinearExtension},
    util::ceil_log2,
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::debug;
use transcript::Transcript;

/// A polynomial has an unique ID associated to it.
pub type PolyID = usize;

/// Constant we use to determine when a BaseFold commitment is trivial
const TRIVIAL_COMMIT_SIZE: usize = 7;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Struct that stores general information about commitments used for proving inference in a [`Model`].
pub struct CommitmentContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    prover_params: PCS::ProverParam,
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    verifier_params: PCS::VerifierParam,
    /// This field contains a [`HashMap`] where the key is a [`PolyID`] and the value is the [`PolynomialCommitmentScheme::CommitmentWithWitness`] corresponding to that ID.
    /// We use an option because some commitments are generated at proving time.
    model_comms: Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Struct that stores prover information about commitments used for proving inference in a [`Model`].
pub struct VerifierContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    params: PCS::VerifierParam,
    /// The mapping between [`PolyID`] and [`PolynomialCommitmentScheme::Commitment`] used by the verifier.
    poly_info: HashMap<PolyID, PCS::Commitment>,
}

impl<E, PCS> CommitmentContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Make a new [`CommitmentContext`]
    pub fn new(
        witness_poly_size: usize,
        polys: Vec<DenseMultilinearExtension<E>>,
    ) -> Result<CommitmentContext<E, PCS>, PCSError> {
        // Find the maximum size so we can generate params
        let max_poly_size = polys
            .iter()
            .map(|poly| 1 << poly.num_vars())
            .max()
            .ok_or(PCSError::ParameterError(
                "No polynomials were provided".to_string(),
            ))?
            .max(witness_poly_size);

        let param = PCS::setup(max_poly_size)?;
        let (prover_params, verifier_params) = PCS::trim(param, max_poly_size)?;

        let model_comms = polys
            .into_par_iter()
            .map(|poly| {
                let commit = PCS::commit(&prover_params, &poly)?;
                Result::<(_, _), PCSError>::Ok((commit, poly))
            })
            .collect::<Result<Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>, _>>(
            )?;

        Ok(CommitmentContext {
            prover_params,
            verifier_params,
            model_comms,
        })
    }

    /// Getter for the PCS prover params
    pub fn prover_params(&self) -> &PCS::ProverParam {
        &self.prover_params
    }

    /// Getter for the PCS verifier params
    pub fn verifier_params(&self) -> &PCS::VerifierParam {
        &self.verifier_params
    }

    pub fn commit(
        &self,
        mle: &DenseMultilinearExtension<E>,
    ) -> Result<PCS::CommitmentWithWitness, PCSError> {
        PCS::commit(&self.prover_params, mle).map_err(|e| e.into())
    }
}

#[derive(Clone, Debug)]
pub struct CommitmentClaim<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    commitment: PCS::CommitmentWithWitness,
    poly: DenseMultilinearExtension<E>,
    claim: Claim<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierClaim<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    commitment: PCS::Commitment,
    claim: Claim<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelOpeningProof<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    batch_proof: PCS::Proof,
    trivial_proofs: Vec<PCS::Proof>,
}

impl<E, PCS> ModelOpeningProof<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    /// Creates a new [`ModelOpeningProof`] from constituent parts.
    pub fn new(
        batch_proof: PCS::Proof,
        trivial_proofs: Vec<PCS::Proof>,
    ) -> ModelOpeningProof<E, PCS> {
        ModelOpeningProof {
            batch_proof,
            trivial_proofs,
        }
    }

    /// Returns a [`bool`] specifying whether there are trivial proofs
    pub fn has_trivial(&self) -> bool {
        !self.trivial_proofs.is_empty()
    }

    /// Getter for the batch proof
    pub fn batch_proof(&self) -> &PCS::Proof {
        &self.batch_proof
    }

    /// Getter for the trivial proofs
    pub fn trivial_proofs(&self) -> &[PCS::Proof] {
        &self.trivial_proofs
    }
}

#[derive(Debug, Clone)]
pub struct CommitmentProver<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    model_commits: Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
    claims: Vec<CommitmentClaim<E, PCS>>,
    trivial_claims: Vec<CommitmentClaim<E, PCS>>,
}

impl<E, PCS> CommitmentProver<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    pub fn new(ctx: &CommitmentContext<E, PCS>) -> CommitmentProver<E, PCS> {
        let model_commits = ctx.model_comms.clone();

        CommitmentProver {
            model_commits,
            claims: vec![],
            trivial_claims: vec![],
        }
    }
    pub fn add_witness_claim(
        &mut self,
        (commitment, mle): (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>),
        claim: Claim<E>,
    ) -> Result<(), PCSError> {
        if mle.num_vars() <= TRIVIAL_COMMIT_SIZE {
            self.trivial_claims.push(CommitmentClaim {
                commitment,
                poly: mle,
                claim,
            });
        } else {
            self.claims.push(CommitmentClaim {
                commitment,
                poly: mle,
                claim,
            });
        }
        Ok(())
    }

    pub fn add_common_claim(&mut self, claim: Claim<E>) -> Result<(), PCSError> {
        let (commitment, poly) = self.model_commits.pop().ok_or(PCSError::ParameterError(
            "Tried to commit to a model commitment but there were none left!".to_string(),
        ))?;
        self.claims.push(CommitmentClaim {
            commitment,
            poly,
            claim,
        });
        Ok(())
    }

    pub fn prove<T: Transcript<E>>(
        &mut self,
        commitment_context: &CommitmentContext<E, PCS>,
        transcript: &mut T,
    ) -> Result<ModelOpeningProof<E, PCS>, PCSError> {
        // Prepare the parts that go into the batch proof
        let (comms, (polys, (points, evaluations))): (
            Vec<PCS::CommitmentWithWitness>,
            (
                Vec<DenseMultilinearExtension<E>>,
                (Vec<Vec<E>>, Vec<Evaluation<E>>),
            ),
        ) = self
            .claims
            .par_drain(..)
            .enumerate()
            .map(|(i, claim)| {
                let CommitmentClaim {
                    commitment,
                    poly,
                    claim,
                } = claim;
                let Claim { point, eval } = claim;

                let evaluation = Evaluation::<E>::new(i, i, eval);
                (commitment, (poly, (point, evaluation)))
            })
            .unzip();

        let trivial_proofs = self
            .trivial_claims
            .iter()
            .map(|claim| {
                let CommitmentClaim {
                    commitment,
                    poly,
                    claim: inner_claim,
                } = claim;
                let Claim { point, eval } = inner_claim;
                PCS::open(
                    commitment_context.prover_params(),
                    poly,
                    commitment,
                    point,
                    eval,
                    transcript,
                )
                .map_err(PCSError::from)
            })
            .collect::<Result<Vec<PCS::Proof>, PCSError>>()?;

        let batch_proof = PCS::batch_open(
            commitment_context.prover_params(),
            &polys,
            &comms,
            &points,
            &evaluations,
            transcript,
        )?;

        Ok(ModelOpeningProof::new(batch_proof, trivial_proofs))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentVerifier<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    model_commits: Vec<PCS::Commitment>,
    claims: Vec<VerifierClaim<E, PCS>>,
    trivial_claims: Vec<VerifierClaim<E, PCS>>,
}

impl<E, PCS> CommitmentVerifier<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    pub fn new(ctx: &CommitmentContext<E, PCS>) -> CommitmentVerifier<E, PCS> {
        let model_commits = ctx
            .model_comms
            .iter()
            .map(|(comm, _)| PCS::get_pure_commitment(comm))
            .collect::<Vec<PCS::Commitment>>();
        CommitmentVerifier {
            model_commits,
            claims: vec![],
            trivial_claims: vec![],
        }
    }
    pub fn add_witness_claim(
        &mut self,
        commitment: PCS::Commitment,
        claim: Claim<E>,
    ) -> Result<(), PCSError> {
        if claim.point.len() <= TRIVIAL_COMMIT_SIZE {
            self.trivial_claims
                .push(VerifierClaim { commitment, claim });
        } else {
            self.claims.push(VerifierClaim { commitment, claim });
        }
        Ok(())
    }

    pub fn add_common_claim(&mut self, claim: Claim<E>) -> Result<(), PCSError> {
        let commitment = self.model_commits.pop().ok_or(PCSError::ParameterError(
            "Tried to commit to a model commitment but there were none left!".to_string(),
        ))?;
        self.claims.push(VerifierClaim { commitment, claim });
        Ok(())
    }

    pub fn verify<T: Transcript<E>>(
        &mut self,
        commitment_context: &CommitmentContext<E, PCS>,
        proof: &ModelOpeningProof<E, PCS>,
        transcript: &mut T,
    ) -> Result<(), PCSError> {
        // Check that all the model commitments have been used
        if !self.model_commits.is_empty() {
            return Err(PCSError::ParameterError(format!(
                "Not all mdoel commits have been used, had {} remaining",
                self.model_commits.len()
            )));
        }
        // Prepare the parts that go into the batch proof
        let (comms, points, evaluations) = self.claims.drain(..).enumerate().fold(
            (vec![], vec![], vec![]),
            |(mut comms_acc, mut points_acc, mut evals_acc), (i, claim)| {
                let VerifierClaim { commitment, claim } = claim;
                let Claim { point, eval } = claim;

                let evaluation = Evaluation::<E>::new(i, i, eval);
                comms_acc.push(commitment);

                points_acc.push(point);
                evals_acc.push(evaluation);
                (comms_acc, points_acc, evals_acc)
            },
        );

        // Ensure that if we have trivial claims then we also have the same number of trivial proofs
        let trivial_proofs = proof.trivial_proofs();

        if self.trivial_claims.len() != trivial_proofs.len() {
            return Err(PCSError::ParameterError(format!(
                "Openign proof had {} trivial proofs, but the verifier has {} trivial claims",
                trivial_proofs.len(),
                self.trivial_claims.len()
            )));
        }

        self.trivial_claims
            .iter()
            .zip(trivial_proofs)
            .try_for_each(|(claim, proof)| {
                let VerifierClaim {
                    commitment,
                    claim: inner_claim,
                } = claim;
                let Claim { point, eval } = inner_claim;
                // Checke that the commitments align
                PCS::verify(
                    commitment_context.verifier_params(),
                    commitment,
                    point,
                    eval,
                    proof,
                    transcript,
                )?;
                Result::<(), PCSError>::Ok(())
            })?;

        PCS::batch_verify(
            commitment_context.verifier_params(),
            &comms,
            &points,
            &evaluations,
            proof.batch_proof(),
            transcript,
        )
        .map_err(PCSError::from)
    }
}
