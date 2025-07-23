//! This module contains logic to prove the correct opening of several claims from several independent
//! polynomials.

use std::collections::{BTreeMap, HashMap};

use crate::{
    Claim, default_transcript,
    layers::provable::NodeId,
    lookup::context::{LookupContext, TableType},
};
use ff_ext::ExtensionField;

use anyhow::{Context, Result, anyhow, ensure};
use mpcs::{Evaluation, PolynomialCommitmentScheme};
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use transcript::Transcript;

pub type PolyId = String;

type ModelCommitmentsMap<PCS, E> = BTreeMap<
    NodeId,
    BTreeMap<
        PolyId,
        (
            <PCS as PolynomialCommitmentScheme<E>>::CommitmentWithWitness,
            DenseMultilinearExtension<E>,
        ),
    >,
>;
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
    /// This field contains a [`BTreeMap`] where the key is a [`NodeId`] and the value is a vector of tuples of [`PolynomialCommitmentScheme::CommitmentWithWitness`]  and [`DenseMultilinearExtension<E>`] corresponding to that ID.
    model_comms_map: ModelCommitmentsMap<PCS, E>,
    /// This field contains a [`BTreeMap`] relating to lookup tables used by the model
    table_comms_map: HashMap<TableType, (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
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
        polys: Vec<(NodeId, HashMap<PolyId, DenseMultilinearExtension<E>>)>,
        lookup_ctx: &LookupContext,
    ) -> Result<CommitmentContext<E, PCS>> {
        // Find the maximum size so we can generate params
        let max_poly_size = polys
            .iter()
            .fold(witness_poly_size, |mut acc, (_, poly_vec)| {
                poly_vec
                    .iter()
                    .for_each(|(_, poly)| acc = acc.max(1 << poly.num_vars()));
                acc
            })
            .next_power_of_two();

        let param = PCS::setup(max_poly_size).context("setting up params")?;
        let (prover_params, verifier_params) =
            PCS::trim(param, max_poly_size).context("trimming params")?;

        let model_comms_map = polys
            .into_par_iter()
            .map(|(node_id, polys_vec)| {
                let model_comms = polys_vec
                    .into_iter()
                    .map(|(id, poly)| {
                        let commit = PCS::commit(&prover_params, &poly)
                            .with_context(|| format!("committing to polynomial {id}"))?;
                        Result::<(_, _), anyhow::Error>::Ok((id, (commit, poly)))
                    })
                    .collect::<Result<
                        BTreeMap<
                            PolyId,
                            (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>),
                        >,
                        _,
                    >>()
                    .with_context(|| format!("collecting node {node_id} commitments"))?;
                Result::<(NodeId, BTreeMap<PolyId, (_, _)>), anyhow::Error>::Ok((
                    node_id,
                    model_comms,
                ))
            })
            .collect::<Result<BTreeMap<NodeId, _>, _>>()
            .context("collecting model commitments")?;

        let table_comms_map = lookup_ctx.iter().filter_map(|table_type| {
            table_type.committed_columns().and_then(|poly| {let commit = PCS::commit(&prover_params, &poly).ok()?; Some((table_type.clone(), (commit, poly)))})
        }).collect::<HashMap<TableType, (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>>();

        Ok(CommitmentContext {
            prover_params,
            verifier_params,
            model_comms_map,
            table_comms_map,
        })
    }

    /// Make a new [`CommitmentContext`] from known params
    pub fn new_with_params(
        polys: Vec<(NodeId, HashMap<PolyId, DenseMultilinearExtension<E>>)>,
        lookup_ctx: &LookupContext,
        prover_params: PCS::ProverParam,
        verifier_params: PCS::VerifierParam,
    ) -> Result<CommitmentContext<E, PCS>> {
        let model_comms_map = polys
            .into_par_iter()
            .map(|(node_id, polys_vec)| {
                let model_comms = polys_vec
                    .into_iter()
                    .map(|(id, poly)| {
                        let commit = PCS::commit(&prover_params, &poly)
                            .with_context(|| format!("committing to polynomial {id}"))?;
                        Result::<(_, _), anyhow::Error>::Ok((id, (commit, poly)))
                    })
                    .collect::<Result<
                        BTreeMap<
                            PolyId,
                            (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>),
                        >,
                        _,
                    >>()
                    .with_context(|| format!("collecting node {node_id} commitments"))?;
                Result::<(NodeId, BTreeMap<PolyId, (_, _)>), anyhow::Error>::Ok((
                    node_id,
                    model_comms,
                ))
            })
            .collect::<Result<BTreeMap<NodeId, _>, _>>()
            .context("collecting model commitments")?;

        let table_comms_map = lookup_ctx.iter().filter_map(|table_type| {
            table_type.committed_columns().and_then(|poly| {
                let commit = PCS::commit(&prover_params, &poly).ok()?;
                Some((table_type.clone(), (commit, poly)))
            })
        }).collect::<HashMap<TableType, (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>>();

        Ok(CommitmentContext {
            prover_params,
            verifier_params,
            model_comms_map,
            table_comms_map,
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

    /// Helper method to commit to polynomial.
    pub fn commit(&self, mle: &DenseMultilinearExtension<E>) -> Result<PCS::CommitmentWithWitness> {
        PCS::commit(&self.prover_params, mle).map_err(|e| e.into())
    }

    /// Write the commitment context to the transcript
    pub fn write_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        self.model_comms_map
            .iter()
            .try_for_each(|(node_id, comms_vec)| {
                comms_vec.iter().try_for_each(|(id, (comm, _))| {
                    let v_comm = PCS::get_pure_commitment(comm);
                    PCS::write_commitment(&v_comm, transcript).context(format!(
                        "Could not write commitment for polynomial {id} of node {node_id}"
                    ))
                })
            })
    }
}

#[derive(Clone, Debug)]
/// Claim about a polynomial used by the prover (so contain witness as well)
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
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Claim about a commitment used by the verifier (so no witness is included).
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
/// The opening proof for a model inference. We may have trivial proofs that occur when the prover has to commit
/// to small witness polynomials.
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
/// Struct used to batch prove all commitment openings in a model proof.
pub struct CommitmentProver<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Claims that are made about non-trivial commitments
    claims: Vec<CommitmentClaim<E, PCS>>,
    /// Claims about trivial commitments (fewer than 8 variables, in this case its more efficient just to evaluate the polynomial)
    trivial_claims: Vec<CommitmentClaim<E, PCS>>,
}

impl<E, PCS> CommitmentProver<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Create a new [`CommitmentProver`] from the [`CommitmentContext`] for the model.
    pub fn new() -> CommitmentProver<E, PCS> {
        CommitmentProver {
            claims: vec![],
            trivial_claims: vec![],
        }
    }
    /// Add a claim about a witness polynomial.
    pub fn add_witness_claim(
        &mut self,
        (commitment, mle): (PCS::CommitmentWithWitness, DenseMultilinearExtension<E>),
        claim: Claim<E>,
    ) -> Result<()> {
        if mle.num_vars() <= PCS::trivial_num_vars() {
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
    /// Add claims about model weights and biases for a certain node
    pub fn add_common_claims(
        &mut self,
        ctx: &CommitmentContext<E, PCS>,
        node_id: NodeId,
        mut claims: HashMap<PolyId, Claim<E>>,
    ) -> Result<()> {
        if claims.is_empty() {
            // No claims to be added
            return Ok(());
        }
        let node_commitments = ctx.model_comms_map.get(&node_id).cloned().ok_or(anyhow!(
            "No commitments stored for node with id: {}",
            node_id
        ))?;
        node_commitments
            .into_iter()
            .try_for_each(|(id, comm_with_wit)| {
                let claim = claims.remove(&id).ok_or_else(|| {
                    anyhow!("No claim found for poly id {} in node {}", id, node_id)
                })?;
                self.add_witness_claim(comm_with_wit, claim)
            })
    }

    /// Adds a claim about a table polynomial
    pub fn add_table_claim(
        &mut self,
        ctx: &CommitmentContext<E, PCS>,
        table_type: TableType,
        claim: Claim<E>,
    ) -> Result<()> {
        let table_commitment = ctx
            .table_comms_map
            .get(&table_type)
            .cloned()
            .ok_or(anyhow!(
                "No table commitments stored for table of type: {}",
                table_type.name()
            ))?;

        self.add_witness_claim(table_commitment, claim)
    }

    /// Produce the [`ModelOpeningProof`] for this inference trace.
    pub fn prove<T: Transcript<E>>(
        &mut self,
        commitment_context: &CommitmentContext<E, PCS>,
        transcript: &mut T,
    ) -> Result<ModelOpeningProof<E, PCS>> {
        // Prepare the parts that go into the batch proof
        #[allow(clippy::type_complexity)]
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
        // Make the trivial proofs.
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
                .map_err(|e| anyhow!("Could not open trivial commitment: {:?}", e))
            })
            .collect::<Result<Vec<PCS::Proof>, anyhow::Error>>()?;

        // Make the batch proof
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
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// The struct used to verify all of the commitment openings in a model proof.
pub struct CommitmentVerifier<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    model_comms_map: HashMap<NodeId, BTreeMap<PolyId, PCS::Commitment>>,
    table_comms_map: HashMap<TableType, PCS::Commitment>,
    claims: Vec<VerifierClaim<E, PCS>>,
    trivial_claims: Vec<VerifierClaim<E, PCS>>,
}

impl<E, PCS> CommitmentVerifier<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Create a new [`CommitmentVerifier`] from the models [`CommitmentContext`].
    pub fn new(ctx: &CommitmentContext<E, PCS>) -> CommitmentVerifier<E, PCS> {
        let model_comms_map = ctx
            .model_comms_map
            .iter()
            .map(|(node_id, comms_vec)| {
                (
                    *node_id,
                    comms_vec
                        .iter()
                        .map(|(id, (comm, _))| (id.clone(), PCS::get_pure_commitment(comm)))
                        .collect::<BTreeMap<PolyId, PCS::Commitment>>(),
                )
            })
            .collect::<HashMap<NodeId, _>>();

        let table_comms_map = ctx
            .table_comms_map
            .iter()
            .map(|(table_type, (comm, _))| (table_type.clone(), PCS::get_pure_commitment(comm)))
            .collect::<HashMap<TableType, PCS::Commitment>>();

        CommitmentVerifier {
            model_comms_map,
            table_comms_map,
            claims: vec![],
            trivial_claims: vec![],
        }
    }
    /// Add a claim about a witness poly to be verified.
    pub fn add_witness_claim(
        &mut self,
        commitment: PCS::Commitment,
        claim: Claim<E>,
    ) -> Result<()> {
        if claim.point.len() <= PCS::trivial_num_vars() {
            self.trivial_claims
                .push(VerifierClaim { commitment, claim });
        } else {
            self.claims.push(VerifierClaim { commitment, claim });
        }
        Ok(())
    }

    /// Add claims about model weights and biases for a certain node
    pub fn add_common_claims(
        &mut self,
        node_id: NodeId,
        mut claims: HashMap<PolyId, Claim<E>>,
    ) -> Result<()> {
        if claims.is_empty() {
            // No claims to be added
            return Ok(());
        }
        let node_commitments = self.model_comms_map.remove(&node_id).ok_or(anyhow!(
            "No commitments stored for node with id: {}",
            node_id
        ))?;

        node_commitments
            .into_iter()
            .try_for_each(|(id, comm_with_wit)| {
                let claim = claims.remove(&id).ok_or_else(|| {
                    anyhow!("No claim found for poly id {} in node {}", id, node_id)
                })?;
                self.add_witness_claim(comm_with_wit, claim)
            })
    }

    /// Adds a claim about a table polynomial
    pub fn add_table_claim(&mut self, table_type: TableType, claim: Claim<E>) -> Result<()> {
        let table_commitment = self.table_comms_map.remove(&table_type).ok_or(anyhow!(
            "No table commitments stored for table of type: {}",
            table_type.name()
        ))?;

        self.add_witness_claim(table_commitment, claim)
    }

    /// Verify the [`ModelOpeningProof`] for this inference trace.
    pub fn verify<T: Transcript<E>>(
        &mut self,
        commitment_context: &CommitmentContext<E, PCS>,
        proof: &ModelOpeningProof<E, PCS>,
        transcript: &mut T,
    ) -> Result<()> {
        // Check that all the model commitments have been used
        ensure!(
            self.model_comms_map.is_empty(),
            "Not all model commits have been used, had {} remaining",
            self.model_comms_map.len()
        );
        // Check all the table commitments have been used
        ensure!(
            self.table_comms_map.is_empty(),
            "Not all table commits have been used, had {} remaining",
            self.table_comms_map.len()
        );

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

        ensure!(
            self.trivial_claims.len() == trivial_proofs.len(),
            "Openign proof had {} trivial proofs, but the verifier has {} trivial claims",
            trivial_proofs.len(),
            self.trivial_claims.len()
        );

        // Check all trivial commitments are correct
        self.trivial_claims
            .par_iter()
            .zip(trivial_proofs.par_iter())
            .try_for_each(|(claim, proof)| {
                let VerifierClaim {
                    commitment,
                    claim: inner_claim,
                } = claim;
                let Claim { point, eval } = inner_claim;
                // Check that the commitments align, we can use a default transcript because trivial openings don't require a transcript
                let mut t = default_transcript::<E>();
                PCS::verify(
                    commitment_context.verifier_params(),
                    commitment,
                    point,
                    eval,
                    proof,
                    &mut t,
                )?;
                Result::<(), anyhow::Error>::Ok(())
            })?;
        // Verify the batch opening
        PCS::batch_verify(
            commitment_context.verifier_params(),
            &comms,
            &points,
            &evaluations,
            proof.batch_proof(),
            transcript,
        )
        .map_err(|e| anyhow!("Error in PCS batch verification: {:?}", e))
    }
}
