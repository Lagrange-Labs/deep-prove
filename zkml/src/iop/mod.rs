use std::collections::HashMap;

use crate::{commit::precommit, layers::LayerProof, lookup::logup_gkr::structs::LogUpProof};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
pub mod context;
pub mod prover;
pub mod split_sumcheck;
pub mod verifier;

pub use context::Context;
use transcript::Transcript;

/// Contains all cryptographic material generated by the prover
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Proof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// The successive sumchecks proofs. From output layer to input.
    steps: Vec<LayerProof<E>>,
    /// The proofs for any lookup tables used
    table_proofs: Vec<TableProof<E>>,
    /// the commitment proofs related to the weights
    commit: precommit::CommitProof<E>,
    /// the proofs related to the witnesses from RELU and link with dense layer
    witness: Option<(precommit::CommitProof<E>, precommit::Context<E>)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TableProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// the lookup protocol proof for the table fractional sumcheck
    lookup: LogUpProof<E>,
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
    pub fn initialise<T: Transcript<E>>(ctx: &Context<E>, transcript: &mut T) -> Self {
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
            .and_then(|challenges| Some((self.constant_challenge, *challenges)))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::{default_transcript, init_test_logging, model::Model, quantization::TensorFielder};

    use super::{
        Context,
        prover::Prover,
        verifier::{IO, verify},
    };

    type F = GoldilocksExt2;

    #[test]
    fn test_prover_steps() {
        init_test_logging();
        let (model, input) = Model::random(4);
        model.describe();
        let trace = model.run(input.clone());
        let output = trace.final_output();
        let ctx = Context::<F>::generate(&model, None).expect("unable to generate context");
        let io = IO::new(input.to_fields(), output.clone().to_fields());
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }

    #[test]
    fn test_prover_steps_pooling() {
        init_test_logging();
        let (model, input) = Model::random_pooling(4);
        model.describe();
        let trace = model.run(input.clone());
        let output = trace.final_output();
        let ctx = Context::<F>::generate(&model, Some(input.get_shape()))
            .expect("unable to generate context");
        let io = IO::new(input.to_fields(), output.clone().to_fields());
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }
}
