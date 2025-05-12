//! This module contains logic to prove the correct opening of several claims from several independent
//! polynomials.

use std::collections::HashMap;

use crate::{
    Claim, Element, VectorTranscript,
    commit::{aggregated_rlc, compute_beta_eval_poly, compute_betas_eval},
    layers::{Layer, convolution, dense},
    model::Model,
};
use anyhow::{Context as CC, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::debug;
use transcript::Transcript;

use super::Pcs;

/// A polynomial has an unique ID associated to it.
pub type PolyID = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Struct that stores general information about commitments used for proving inference in a [`Model`].
pub struct CommitmentContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// General parameters for the [`PolynomialCommitmentScheme`]
    params: PCS::Param,
    /// Commitments to constant tensors used in model inference e.g. weights and biases
    model_commitments: Vec<PCS::CommitmentWithWitness>,
    /// The multilinear extensions of the corresponding commitments stored in [`CommitmentContext::model_commitments`]
    model_polys: Vec<DenseMultilinearExtension<E>>,
    /// This field contains a [`HashMap`] where the key is a [`PolyID`] and the value is the index of the corresponding polynomial/commitment in [`CommitmentContext::model_polys`]/[`CommitmentContext::model_commitments`]
    /// together with a [`bool`] that indicates if commitment is trivial or not
    poly_info: HashMap<PolyID, (usize, bool)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Struct that stores prover information about commitments used for proving inference in a [`Model`].
pub struct ProverContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    params: PCS::ProverParam,
    /// Commitments to tensors used in model inference (including weights and biases)
    commitments: Vec<PCS::CommitmentWithWitness>,
    /// Trivial commitments to tensors used in inference
    trivial_commitments: Vec<PCS::CommitmentWithWitness>,
    /// The multilinear extensions of the corresponding commitments stored in [`ProverContext::commitments`]
    polys: Vec<DenseMultilinearExtension<E>>,
    /// The multilinear extensions of the corresponding commitments stored in [`ProverContext::trivial_commitments`]
    trivial_polys: Vec<DenseMultilinearExtension<E>>,
    /// This field contains a [`HashMap`] where the key is a [`PolyID`] and the value is the index of the corresponding polynomial/commitment in either the commitments/polys list or the trivial commitments/polys list.
    /// The [`bool`] is `true` if the commitment is trivial
    poly_info: HashMap<PolyID, (usize, bool)>,
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
    /// Commitments to tensors used in model inference (including weights and biases)
    commitments: Vec<PCS::Commitment>,
    /// Trivial commitments to tensors used in inference
    trivial_commitments: Vec<PCS::Commitment>,
    /// This field contains a [`HashMap`] where the key is a [`PolyID`] and the value is the index of the corresponding polynomial/commitment in either the commitments/polys list or the trivial commitments/polys list.
    /// The [`bool`] is `true` if the commitment is trivial
    poly_info: HashMap<PolyID, (usize, bool)>,
}

impl<E, PCS> Default for CommitmentContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    fn default() -> Self {
        let params = PCS::setup(2).expect("unable to setup commitment");
        CommitmentContext {
            params,
            model_commitments: vec![],
            model_polys: vec![],
            poly_info: HashMap::default(),
        }
    }
}
