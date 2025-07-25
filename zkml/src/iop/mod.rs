use std::collections::HashMap;

use crate::{
    commit::context::ModelOpeningProof,
    layers::{LayerProof, provable::NodeId},
    lookup::logup_gkr::structs::LogUpProof,
};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
pub mod context;
pub mod prover;
pub mod verifier;

pub use context::Context;
use transcript::Transcript;

/// Contains all cryptographic material generated by the prover
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Proof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// The successive sumchecks proofs. From output layer to input.
    steps: HashMap<NodeId, LayerProof<E, PCS>>,
    /// The proofs for any lookup tables used
    table_proofs: Vec<TableProof<E, PCS>>,
    /// the commitment proofs related to the weights
    commit: ModelOpeningProof<E, PCS>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct TableProof<E: ExtensionField, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// The commitment to the multiplicity polynomial
    multiplicity_commit: PCS::Commitment,
    /// the lookup protocol proof for the table fractional sumcheck
    lookup: LogUpProof<E>,
}

impl<E, PCS> TableProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// gets a reference to the inner commitment
    pub fn get_commitment(&self) -> &PCS::Commitment {
        &self.multiplicity_commit
    }
}
#[derive(Debug, Clone, Default)]
pub struct ChallengeStorage<E: ExtensionField> {
    /// This is the constant challenge looked in the lookup PIOPs
    pub constant_challenge: E,
    /// This is the map containing different values related to different tables/lookups
    pub challenge_map: HashMap<String, E>,
}

impl<E> ChallengeStorage<E>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn initialise<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        ctx: &Context<E, PCS>,
        transcript: &mut T,
    ) -> Self {
        let constant_challenge = transcript
            .get_and_append_challenge(b"table_constant")
            .elements;
        let challenge_map = ctx
            .lookup
            .iter()
            .map(|table_type| {
                let challenge = table_type.generate_challenge(transcript);

                (table_type.name(), challenge)
            })
            .collect::<HashMap<String, E>>();
        Self {
            constant_challenge,
            challenge_map,
        }
    }

    pub fn get_challenges_by_name(&self, name: &String) -> Option<(E, E)> {
        self.challenge_map
            .get(name)
            .map(|challenges| (self.constant_challenge, *challenges))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{default_transcript, init_test_logging_default, model::Model, testing::Pcs};

    use super::{Context, prover::Prover, verifier::verify};

    type F = GoldilocksExt2;

    #[test]
    fn test_prover_steps_generic() {
        init_test_logging_default();
        let (model, input) = Model::random(4).unwrap();
        model.describe();
        let trace = model.run(&input).unwrap();
        let io = trace.to_verifier_io();
        let ctx =
            Context::<F, Pcs<F>>::generate(&model, None, None).expect("unable to generate context");
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, _>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }

    #[test]
    fn test_prover_steps_pooling() {
        init_test_logging_default();
        let (model, input) = Model::random_pooling(4).unwrap();
        model.describe();
        let trace = model.run(&input).unwrap();
        let io = trace.to_verifier_io();
        let ctx =
            Context::<F, Pcs<F>>::generate(&model, None, None).expect("unable to generate context");
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, _>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }
}
