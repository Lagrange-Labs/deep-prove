use ff_ext::ExtensionField;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sumcheck::structs::IOPProof;
use crate::commit::precommit;
use crate::commit::same_poly;
use crate::lookup;

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
    /// the commitment proofs related to the weights
    commit: precommit::CommitProof<E>,
    /// the proofs related to the witnesses from RELU and link with dense layer
    witness: Option<(precommit::CommitProof<E>, precommit::Context<E>)>,
}

#[derive(Clone,Serialize,Deserialize)]
pub enum StepProof<E: ExtensionField> {
    Dense(Matrix2VecProof<E>),
    Activation(ActivationProof<E>),
}

impl<E: ExtensionField> StepProof<E> {
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::Activation(_) => "Activation".to_string(),
        }
    }
}

#[derive(Clone,Serialize,Deserialize)]
pub struct ActivationProof<E: ExtensionField> {
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the relu
    lookup: lookup::Proof<E>,
}

/// Contains proof material related to one step of the inference
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Matrix2VecProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the mat2vec protocol
    sumcheck: IOPProof<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

impl<E: ExtensionField> Matrix2VecProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::{default_transcript, model::Model, vector_to_field_par};

    use super::{prover::Prover, verifier::{verify, IO}, Context};

    type F = GoldilocksExt2;
    use tracing_subscriber;

    #[test]
    fn test_prover_steps() {
        tracing_subscriber::fmt::init();
        let (model, input) = Model::random(4);
        model.describe();
        let trace = model.run::<F>(input.clone());
        let output = trace.final_output();
        let ctx = Context::generate(&model).expect("unable to generate context");
        let io = IO::new(vector_to_field_par(&input), output.to_vec());
        let mut prover_transcript = default_transcript();
        let prover = Prover::new(&ctx, &mut prover_transcript);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript = default_transcript();
        verify(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
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
