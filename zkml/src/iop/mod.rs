use crate::{
    Claim,
    commit::{precommit, same_poly},
    lookup,
};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPProof;

pub mod context;
pub mod prover;
pub mod verifier;

pub use context::Context;

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

    pub fn get_lookup_data(&self) -> Option<(Vec<E::BaseField>, Vec<E>, Vec<E>)> {
        match self {
            StepProof::Dense(..) => None,
            StepProof::Convolution(..) => None,
            StepProof::Activation(ActivationProof { lookup, .. })
            | StepProof::Requant(RequantProof { lookup, .. })
            | StepProof::Pooling(PoolingProof { lookup, .. }) => Some((
                lookup.get_digest().0.to_vec(),
                lookup.numerators(),
                lookup.denominators(),
            )),
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
    lookup: lookup::Proof<E>,
}

/// Contains proof material related to one step of the inference

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ConvProof<E: ExtensionField>{
    // Sumcheck proof for the FFT layer
    fft_proof : IOPProof<E>,
    // Proof for the evaluation delegation of the omegas matrix
    // It consists of multiple sumcheck proofs
    fft_delegation_proof: Vec<IOPProof<E>>,
    // Likewise for fft, we define ifft proofs
    ifft_proof : IOPProof<E>,
    ifft_delegation_proof: Vec<IOPProof<E>>,
    // Sumcheck proof for the hadamard product
    hadamard_proof: IOPProof<E>,
    // The evaluation claims produced by the corresponding sumchecks
    fft_claims: Vec<E>,
    ifft_claims: Vec<E>,
    fft_delegation_claims: Vec<Vec<E>>,
    ifft_delegation_claims: Vec<Vec<E>>,
    hadamard_clams: Vec<E>

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
    lookup: lookup::Proof<E>,
    /// proof for the accumulation of the claim from the zerocheck + claim from lookup for the same poly for both input and output
    io_accumulation: [same_poly::Proof<E>; 2],
    /// The claims that are accumulated for the output of this step
    output_claims: Vec<Claim<E>>,
    /// The claim that are accumulated for the input of this step
    input_claims: Vec<Claim<E>>,
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
    lookup: lookup::Proof<E>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TableProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// the lookup protocol proof for the table fractional sumcheck
    lookup: lookup::Proof<E>,
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
    use tracing_subscriber;

    #[test]
    fn test_prover_steps() {
        init_test_logging();
        let (model, input) = Model::random(4);
        model.describe();
        let trace = model.run(input.clone());
        let output = trace.final_output();
        let ctx = Context::<F>::generate(&model).expect("unable to generate context");
        let io = IO::new(input.to_fields(), output.clone().to_fields());
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, LogUp>::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify::<_, _, LogUp>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }

    //#[test]
    // fn test_sumcheck_evals() {
    //    type F = GoldilocksExt2;
    //    let n = (10 as usize).next_power_of_two();
    //    let mat = Matrix::random((2 * n, n)).pad_next_power_of_two();
    //    let vec = random_vector(n);
    //    let sum = mat.matmul(&vec);
    //    let mle1 = mat.to_mle();
    //    let mle2 = vector_to_mle(vec);

    //    let vp = VirtualPolynomial::new(n.ilog2() as usize);
    //    vp.add_mle_list(vec![mle1.clone().into(), mle2.clone().into()], F::ONE);
    //    let poly_info = vp.aux_info.clone();
    //    #[allow(deprecated)]
    //    let (proof, _) = IOPProverState::<F>::prove_parallel(vp.clone(), &mut transcript);

    //    let mut transcript = BasicTranscript::new(b"test");
    //    let subclaim = IOPVerifierState::<F>::verify(sum, &proof, &poly_info, &mut transcript);
    //    assert!(
    //        vp.evaluate(
    //            subclaim
    //                .point
    //                .iter()
    //                .map(|c| c.elements)
    //                .collect::<Vec<_>>()
    //                .as_ref()
    //        ) == subclaim.expected_evaluation,
    //        "wrong subclaim"
    //    );
    //}
}
