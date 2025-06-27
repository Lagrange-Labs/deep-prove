//! This module implements the proving and verifying of a hadamard product
//! as v[i] = v1[i] * v2[i]
//! It can not be integrated as a single layer in the model, but rather it's a
//! a module that can be used within other layers. FOr example it is used within
//! convolution.

use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use crate::{Claim, Element, Tensor, commit::compute_betas_eval};

pub struct HadamardCtx<F: ExtensionField> {
    sumcheck_aux: VPAuxInfo<F>,
}

impl<F: ExtensionField> HadamardCtx<F> {
    pub fn new(v1: &Tensor<Element>, v2: &Tensor<Element>) -> Self {
        assert_eq!(v1.get_shape(), v2.get_shape());
        let num_vars = if v1.get_data().len().is_power_of_two() {
            v1.get_data().len().ilog2() as usize
        } else {
            v1.get_data().len().next_power_of_two().ilog2() as usize
        };
        Self {
            sumcheck_aux: VPAuxInfo::from_mle_list_dimensions(&[vec![
                // v1, v2, beta
                num_vars, num_vars, num_vars,
            ]]),
        }
    }
    pub fn from_len(vector_len: usize) -> Self {
        let num_vars = vector_len.next_power_of_two().ilog2() as usize;
        Self {
            // v1, v2, beta
            sumcheck_aux: VPAuxInfo::from_mle_list_dimensions(&[vec![
                num_vars, num_vars, num_vars,
            ]]),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct HadamardProof<E: ExtensionField> {
    sumcheck: IOPProof<E>,
    individual_claim: Vec<E>,
}

impl<F: ExtensionField> HadamardProof<F> {
    #[allow(unused)]
    pub fn random_point(&self) -> &[F] {
        &self.sumcheck.point
    }
    #[allow(unused)]
    pub fn v1_eval(&self) -> F {
        self.individual_claim[0]
    }
    #[allow(unused)]
    pub fn v2_eval(&self) -> F {
        self.individual_claim[1]
    }
}

// - v1, the two input vectors
// - v3 = v1 ° v2, the output vector
// - the last claim on the output vector v3(r) = y
// .  - remember the proving is done backwards
// - We want to prove that $\sum_i v_1(i) * v_2(i) * eq(r,i) = y = v_3(r)$
// - This output one point to evaluate on v1 and v2 and eq.
// NOTE: output claim MUST be valid on v3 but v3 is never computed on this method !
// In the conv application:
//   - v1 is the output of the conv
// . - v2 clearing garbage tensor. the verifier can evaluate v2 and eq easily
//   - the v1 claim is now passed to the previous layer (we prove in reverse)
#[allow(unused)]
pub fn prove<F: ExtensionField, T: Transcript<F>>(
    transcript: &mut T,
    output_claim: &Claim<F>,
    v1: &Tensor<Element>,
    v2: &Tensor<Element>,
) -> HadamardProof<F> {
    assert_eq!(
        output_claim.point.len(),
        v1.get_data().len().ilog2() as usize
    );
    assert_eq!(
        output_claim.point.len(),
        v2.get_data().len().ilog2() as usize
    );
    assert!(
        v1.get_shape()
            .into_vec()
            .iter()
            .all(|x| x.is_power_of_two())
    );
    assert!(
        v2.get_shape()
            .into_vec()
            .iter()
            .all(|x| x.is_power_of_two())
    );
    let beta_poly = compute_betas_eval(&output_claim.point).into_mle();
    let v1_mle = v1.to_mle_flat::<F>();
    let v2_mle = v2.to_mle_flat::<F>();
    let mut vp = VirtualPolynomial::<F>::new(v1_mle.num_vars());
    vp.add_mle_list(vec![v1_mle.into(), v2_mle.into(), beta_poly.into()], F::ONE);
    #[allow(deprecated)]
    let (proof, state) = IOPProverState::<F>::prove_parallel(vp, transcript);
    debug_assert!({
        let computed_eval = proof.extract_sum();
        let given_eval = output_claim.eval;
        computed_eval == given_eval
    });
    HadamardProof {
        sumcheck: proof,
        individual_claim: state.get_mle_final_evaluations()[..2].to_vec(),
    }
}

#[allow(unused)]
pub fn verify<F: ExtensionField, T: Transcript<F>>(
    ctx: &HadamardCtx<F>,
    transcript: &mut T,
    proof: &HadamardProof<F>,
    output_claim: &Claim<F>,
    expected_v2_eval: F,
) -> Result<Claim<F>> {
    let subclaim = IOPVerifierState::<F>::verify(
        output_claim.eval,
        &proof.sumcheck,
        &ctx.sumcheck_aux,
        transcript,
    );
    // TODO: closed formula for beta evaluation
    let beta = compute_betas_eval(&output_claim.point).into_mle();
    let beta_eval = beta.evaluate(&proof.sumcheck.point);
    // [v1,v2,beta]
    ensure!(
        expected_v2_eval == proof.v2_eval(),
        "Hadamard verification failed for v2 eval"
    );
    /// Given the evaluations are given outside of the sumcheck proof, by the prover,
    /// we need to verify that they match what the sumcheck have been computed.
    let product = beta_eval * proof.v1_eval() * proof.v2_eval();
    ensure!(
        product == subclaim.expected_evaluation,
        "Hadamard verification failed for product eval"
    );
    Ok(Claim::new(proof.sumcheck.point.clone(), proof.v1_eval()))
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use super::*;
    use crate::{default_transcript, testing::random_field_vector};

    #[test]
    fn test_hadamard_proving() {
        let mut transcript = default_transcript();
        let n: usize = 10;
        let v1 = Tensor::random(&vec![n].into()).pad_next_power_of_two();
        let v2 = Tensor::random(&vec![n].into()).pad_next_power_of_two();
        let r = random_field_vector(n.next_power_of_two().ilog2() as usize);
        let expected_output = v1.mul(&v2);
        let output_mle = expected_output.to_mle_flat::<GoldilocksExt2>();
        let output_eval = output_mle.evaluate(&r);
        let output_claim = Claim::new(r, output_eval);
        let proof = prove(
            &mut transcript,
            &output_claim.clone(),
            &v1.clone(),
            &v2.clone(),
        );

        let ctx = HadamardCtx::new(&v1, &v2);
        // NOTE: find closed formula to evaluate it efficiently OR use PCS
        let v2_eval = v2
            .to_mle_flat::<GoldilocksExt2>()
            .evaluate(&proof.random_point());
        // NOTE: this has to be done by the component integrating the hadamard logic
        // normally by verifying this input claim via another sumcheck.
        let input_claim = verify(
            &ctx,
            &mut default_transcript(),
            &proof,
            &output_claim,
            v2_eval,
        )
        .unwrap();
        let expected_v1_eval = v1
            .to_mle_flat::<GoldilocksExt2>()
            .evaluate(&input_claim.point);
        assert_eq!(expected_v1_eval, input_claim.eval);
    }
}
