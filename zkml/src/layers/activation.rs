use crate::{
    Claim, Context, Element, Prover,
    commit::same_poly,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{LayerCtx, LayerProof},
    lookup::{
        context::{COLUMN_SEPARATOR, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
        witness::LogUpWitness,
    },
    model::StepData,
    padding::PaddingMode,
    quantization::Fieldizer,
    tensor::Number,
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::{DenseMultilinearExtension, IntoMLE};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{quantization::BIT_LEN, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx,
};

use anyhow::{Result, anyhow, ensure};

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    Relu(Relu),
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation,
    pub node_id: NodeId,
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

impl OpInfo for Activation {
    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!("RELU: {}", 1 << Relu::num_vars())
    }

    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        input_shapes.to_vec() // same as input shapes
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for Activation {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<N, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating activation layer"
        );
        let input = inputs[0];
        let output = match self {
            Activation::Relu(relu) => relu.op(input),
        };
        Ok(LayerOut::from_vec(vec![output]))
    }
}

impl<E> ProveInfo<E> for Activation
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        aux.tables.insert(TableType::Relu);
        let num_vars = aux
            .last_output_shape
            .iter_mut()
            .fold(Ok(None), |expected_num_vars, shape| {
                let num_vars = shape.iter().map(|dim| ceil_log2(*dim)).sum::<usize>();
                if let Some(vars) = expected_num_vars? {
                    ensure!(
                        vars == num_vars,
                        "All input shapes for activation must have the same number of variables"
                    );
                }
                Ok(Some(num_vars))
            })?
            .expect("No input shape found for activation layer?");
        // Set the model polys to be empty
        aux.model_polys = vec![];
        let info = match self {
            Activation::Relu(relu) => LayerCtx::Activation(ActivationCtx {
                op: Activation::Relu(*relu),
                node_id: id,
                num_vars,
            }),
        };
        Ok((info, aux))
    }
}

impl PadOp for Activation {}

impl<E, PCS> ProvableOp<E, PCS> for Activation
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = ActivationCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.prove_step(
            prover,
            last_claims[0],
            step_data.outputs.outputs()[0].get_data(),
            ctx,
            id,
        )?])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        gen: &mut LookupWitnessGen<E, PCS>,
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<()> {
        ensure!(
            step_data.inputs.len() == 1,
            "Found more than 1 input tensor in inference step of activation layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        // Calculate the column_evals and also the merged lookups
        let (merged_lookups, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) = step_data
            .inputs[0]
            .get_data()
            .iter()
            .zip(step_data.outputs.outputs()[0].get_data().iter())
            .map(|(a, b)| {
                let a_field: E = a.to_field();
                let b_field: E = b.to_field();
                (
                    a + COLUMN_SEPARATOR * b,
                    (a_field.as_bases()[0], b_field.as_bases()[0]),
                )
            })
            .unzip();

        let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) = field.into_iter().unzip();

        let num_vars = ceil_log2(col_one.len());

        // Add the witness polynomials that we need to commit to
        let (commits, column_evals): (
            Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
            Vec<Vec<E::BaseField>>,
        ) = [col_one, col_two]
            .into_par_iter()
            .map(|evaluations| {
                let mle =
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &evaluations);
                let commit = ctx.commitment_ctx.commit(&mle)?;
                Ok(((commit, mle), evaluations))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();
        gen.logup_witnesses.insert(
            id,
            vec![LogUpWitness::<E, PCS>::new_lookup(
                commits,
                column_evals,
                2,
                TableType::Relu,
            )],
        );

        let lookups = gen
            .new_lookups
            .get_mut(&TableType::Relu)
            .ok_or(anyhow!("No table of type Relu was expected"))?;
        lookups.extend(merged_lookups);

        Ok(())
    }
}

impl OpInfo for ActivationCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        Activation::Relu(Relu).output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Activation::Relu(Relu).num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        Activation::Relu(Relu).describe()
    }

    fn is_provable(&self) -> bool {
        Activation::Relu(Relu).is_provable()
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for ActivationCtx
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = ActivationProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .get_challenges_by_name(&TableType::Relu.name())
            .ok_or(anyhow!(
                "Couldn't get challenges for LookupType: {}",
                TableType::Relu.name()
            ))?;
        Ok(vec![self.verify_activation(
            verifier,
            last_claims[0],
            proof,
            constant_challenge,
            column_separation_challenge,
        )?])
    }
}

impl Activation {
    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        output: &[E],
        _step: &ActivationCtx,
        node_id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // Should only be one prover_info for this step
        let logup_witnesses = prover.lookup_witness(node_id)?;
        if logup_witnesses.len() != 1 {
            return Err(anyhow!(
                "Activation only requires a lookup into one table type, but node: {} had {} lookup witnesses",
                node_id,
                logup_witnesses.len()
            ));
        }
        let logup_witness = &logup_witnesses[0];
        // Run the lookup protocol and return the lookup proof
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
            .collect::<Result<Vec<PCS::Commitment>, anyhow::Error>>()?;

        // Add the proof in
        prover.push_proof(
            node_id,
            LayerProof::Activation(ActivationProof {
                io_accumulation: claim_acc_proof,
                lookup: logup_proof,
                commits,
            }),
        );
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
        last_claim: &Claim<E>,
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
        sp_verifier.add_claim(last_claim.clone())?;
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

impl Default for Relu {
    fn default() -> Self {
        Self::new()
    }
}

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
