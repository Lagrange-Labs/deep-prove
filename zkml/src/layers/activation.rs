use crate::{
    Claim, Prover,
    commit::{PCSError, same_poly},
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerCtx, LayerProof, PolyID},
    lookup::{
        context::TableType,
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
    },
    tensor::Number,
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::IntoMLE;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{quantization::BIT_LEN, tensor::Tensor};

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    Relu(Relu),
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ActivationProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the relu
    pub(crate) lookup: LogUpProof<E>,
    /// The witness commitments from this function
    pub(crate) commits: Vec<PCS::Commitment>,
}

impl Activation {
    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Tensor<T> {
        match self {
            Activation::Relu(relu) => relu.op(input),
        }
    }
    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux<E>,
    ) -> (LayerCtx<E>, ContextAux<E>)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.tables.insert(TableType::Relu);
        let info = match self {
            Activation::Relu(relu) => LayerCtx::Activation(ActivationCtx {
                op: Activation::Relu(*relu),
                poly_id: id,
                num_vars: aux
                    .last_output_shape
                    .iter()
                    .map(|dim| ceil_log2(*dim))
                    .sum::<usize>(),
            }),
        };
        // There are no fixed tensors for Relu so set model_polys to be empty
        aux.model_polys = vec![];

        (info, aux)
    }

    pub(crate) fn prove_step<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        output: &[E],
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let logup_witness = prover.next_lookup_witness()?;
        let prover_info = logup_witness.get_logup_input(&prover.challenge_storage)?;

        let commits = logup_witness.get_commitments();
        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;
        // Activation proofs have two columns, input and output
        let input_claim = logup_proof.output_claims()[0].clone();
        let output_claim = logup_proof.output_claims()[1].clone();

        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;

        // Add commitment claims to prover
        let commits = [input_claim.clone(), claim_acc_proof.extract_claim()]
            .into_iter()
            .zip(commits)
            .map(|(claim, comm_with_wit)| {
                let comm = PCS::get_pure_commitment(&comm_with_wit.0);
                prover
                    .commit_prover
                    .add_witness_claim(comm_with_wit, claim)?;
                Ok(comm)
            })
            .collect::<Result<Vec<PCS::Commitment>, PCSError>>()?;

        // Add the proof in
        prover.push_proof(LayerProof::Activation(ActivationProof {
            io_accumulation: claim_acc_proof,
            lookup: logup_proof,
            commits,
        }));
        Ok(input_claim)
    }
}

impl ActivationCtx {
    pub(crate) fn verify_activation<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: Claim<E>,
        proof: &ActivationProof<E, PCS>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proof
        let verifier_claims = verify_logup_proof(
            &proof.lookup,
            1,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;

        // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
        let sp_ctx = same_poly::Context::<E>::new(self.num_vars);
        let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
        sp_verifier.add_claim(last_claim)?;
        verifier_claims.claims()[1..]
            .iter()
            .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

        let new_output_claim = sp_verifier.verify(&proof.io_accumulation, verifier.transcript)?;
        // 3. Accumulate the new claim into the witness commitment protocol
        verifier_claims
            .claims()
            .iter()
            .take(1)
            .cloned()
            .chain(std::iter::once(new_output_claim))
            .zip(proof.commits.iter())
            .try_for_each(|(claim, commit)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(commit.clone(), claim)
            })?;

        // 4. return the input claim for to be proven at subsequent step
        Ok(verifier_claims.claims()[0].clone())
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Relu;

impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        *BIT_LEN
    }
    pub fn poly_len() -> usize {
        1 << Self::num_vars()
    }
    pub fn shape() -> Vec<usize> {
        vec![2, Self::poly_len()]
    }

    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Tensor<T> {
        Tensor::new(
            input.get_shape(),
            input
                .get_data()
                .par_iter()
                .map(|e| Self::apply(*e))
                .collect::<Vec<_>>(),
        )
    }

    #[inline(always)]
    pub fn apply<T: Number>(e: T) -> T {
        if e.is_negative() { T::default() } else { e }
    }
}

#[cfg(test)]
mod test {
    use crate::Element;

    use super::*;

    #[test]
    fn test_activation_relu_apply() {
        struct TestCase {
            input: Element,
            output: Element,
        }

        impl TestCase {
            pub fn from(input: Element, output: Element) -> Self {
                Self { input, output }
            }
        }
        for case in [
            TestCase::from(-24, 0),
            TestCase::from(0, 0),
            TestCase::from(124, 124),
            TestCase::from(-127, 0),
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }
}
