use std::collections::HashMap;

use crate::{
    Claim,
    commit::{precommit, same_poly},
    lookup::logup_gkr::structs::LogUpProof,
};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPProof;

pub mod context;
pub mod prover;
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
    steps: Vec<StepProof<E>>,
    /// The proofs for any lookup tables used
    table_proofs: Vec<TableProof<E>>,
    /// the commitment proofs related to the weights
    commit: precommit::CommitProof<E>,
    /// the proofs related to the witnesses from RELU and link with dense layer
    witness: Option<(precommit::CommitProof<E>, precommit::Context<E>)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum StepProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    Dense(DenseProof<E>),
    Convolution(ConvProof<E>),
    Activation(ActivationProof<E>),
    Requant(RequantProof<E>),
    Pooling(PoolingProof<E>),
}

impl<E: ExtensionField> StepProof<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            StepProof::Dense(..) => None,
            StepProof::Convolution(..) => None,
            StepProof::Activation(ActivationProof { lookup, .. })
            | StepProof::Requant(RequantProof { lookup, .. })
            | StepProof::Pooling(PoolingProof { lookup, .. }) => Some(lookup.fractional_outputs()),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ActivationProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the relu
    lookup: LogUpProof<E>,
}

/// Contains proof material related to one step of the inference

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ConvProof<E: ExtensionField> {
    // Sumcheck proof for the FFT layer
    fft_proof: IOPProof<E>,
    // Proof for the evaluation delegation of the omegas matrix
    // It consists of multiple sumcheck proofs
    fft_delegation_proof: Vec<IOPProof<E>>,
    // Likewise for fft, we define ifft proofs
    ifft_proof: IOPProof<E>,
    ifft_delegation_proof: Vec<IOPProof<E>>,
    // Sumcheck proof for the hadamard product
    hadamard_proof: IOPProof<E>,
    // The evaluation claims produced by the corresponding sumchecks
    fft_claims: Vec<E>,
    ifft_claims: Vec<E>,
    fft_delegation_claims: Vec<Vec<E>>,
    ifft_delegation_claims: Vec<Vec<E>>,
    hadamard_clams: Vec<E>,
    bias_claim: E,
}
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct DenseProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the mat2vec protocol
    sumcheck: IOPProof<E>,
    /// The evaluation of the bias at the previous claims in the proving flow.
    /// The verifier substracts this from the previous claim to end up with one claim only
    /// about the matrix, without the bias.
    bias_eval: E,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

impl<E: ExtensionField> DenseProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
}

/// Contains proof material related to one step of the inference
#[derive(Clone, Serialize, Deserialize)]
pub struct PoolingProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// the actual sumcheck proof proving that the product of correct terms is always zero
    sumcheck: IOPProof<E>,
    /// The lookup proof showing that the diff is always in the correct range
    lookup: LogUpProof<E>,
    /// proof for the accumulation of the claim from the zerocheck + claim from lookup for the same poly for both input and output
    io_accumulation: same_poly::Proof<E>,
    /// The claims that are accumulated for the output of this step
    output_claims: Vec<Claim<E>>,
    /// The output evaluations of the diff polys produced by the zerocheck
    zerocheck_evals: Vec<E>,
    /// This tells the verifier how far apart the variables get fixed on the input MLE
    variable_gap: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RequantProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from activation + claim from lookup for the same poly
    /// e.g. the "link" between an activation and requant layer
    io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the requantization
    lookup: LogUpProof<E>,
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
                let challenge = transcript
                    .get_and_append_challenge(b"table_challenge")
                    .elements;

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

    use crate::{
        default_transcript, init_test_logging, lookup::LogUp, model::Model,
        quantization::TensorFielder,
    };

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
        let prover = Prover::<_, _, LogUp>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _, LogUp>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }

    #[test]
    fn test_prover_steps_pooling() {
        init_test_logging();
        let (model, input) = Model::random_pooling(4);
        model.describe();
        let trace = model.run(input.clone());
        let output = trace.final_output();
        let ctx =
            Context::<F>::generate(&model, Some(input.dims())).expect("unable to generate context");
        let io = IO::new(input.to_fields(), output.clone().to_fields());
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, LogUp>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _, LogUp>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }
}
