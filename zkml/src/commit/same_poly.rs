//! This module contains logic to prove the opening of several claims related to the _same_ polynomial.
//! e.g. a set of (r_i,y_i) such that f(r_i) = y_i for all i's.
//! a_i = randomness() for i:0 -> |r_i|
//! for r_i, compute Beta_{r_i} = [beta_{r_i}(0),(1),...(2^|r_i|)]
//! then Beta_j = SUM_j a_i * Beta_{r_i}
//! final_y = SUM a_i * y_i
//!
//! Note the output of the verifier is a claim that needs to be verified outside of this protocol.
//! It could be via an opening directly OR via an accumulation scheme.

use crate::{Claim, VectorTranscript, commit::identity_eval};
use anyhow::{Ok, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use super::{aggregated_rlc, compute_betas_eval};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField> {
    vp_info: VPAuxInfo<E>,
}

impl<E: ExtensionField> Context<E> {
    /// number of variables of the poly in question
    pub fn new(num_vars: usize) -> Self {
        Self {
            vp_info: VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Proof<E: ExtensionField> {
    sumcheck: IOPProof<E>,
    // [0] about the betas, [1] about the poly
    evals: Vec<E>,
}

impl<E: ExtensionField> Proof<E> {
    pub fn extract_claim(&self) -> Claim<E> {
        Claim {
            point: self.sumcheck.point.clone(),
            eval: self.evals[1],
        }
    }
}

pub struct Prover<E: ExtensionField> {
    claims: Vec<Claim<E>>,
    poly: DenseMultilinearExtension<E>,
}

impl<E> Prover<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// The polynomial over which the claims are to be accumulated and proven
    /// Note the prover also _commits_ to this polynomial.
    pub fn new(poly: DenseMultilinearExtension<E>) -> Self {
        Self {
            claims: Default::default(),
            poly,
        }
    }
    pub fn add_claim(&mut self, claim: Claim<E>) -> anyhow::Result<()> {
        ensure!(
            claim.point.len() == self.poly.num_vars(),
            format!(
                "Invalid claim length: input.len() = {} vs poly.num_vars = {} ",
                claim.point.len(),
                self.poly.num_vars()
            )
        );
        self.claims.push(claim);
        Ok(())
    }
    pub fn prove<T: Transcript<E>>(self, t: &mut T) -> anyhow::Result<Proof<E>> {
        let challenges = t.read_challenges(self.claims.len());

        let beta_evals = challenges
            .into_par_iter()
            .zip(self.claims.into_par_iter())
            .map(|(a_i, c_i)| {
                // c_i.input = r_i
                compute_betas_eval(&c_i.point)
                    .into_iter()
                    .map(|b_i| a_i * b_i)
                    .collect_vec()
            })
            .collect::<Vec<_>>();
        let final_beta = (0..1 << self.poly.num_vars())
            .into_par_iter()
            .map(|i| {
                beta_evals
                    .iter()
                    .map(|beta_for_r_i| beta_for_r_i[i])
                    .fold(E::ZERO, |acc, b| acc + b)
            })
            .collect::<Vec<_>>();

        // then run the sumcheck on it
        let mut vp = VirtualPolynomial::new(self.poly.num_vars());
        vp.add_mle_list(vec![final_beta.into_mle().into(), self.poly.into()], E::ONE);
        #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(vp, t);

        Ok(Proof {
            sumcheck: sumcheck_proof,
            evals: state.get_mle_final_evaluations(),
        })
    }
}

pub struct Verifier<'a, E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    claims: Vec<Claim<E>>,
    ctx: &'a Context<E>,
}

impl<'a, E: ExtensionField> Verifier<'a, E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new(ctx: &'a Context<E>) -> Self {
        Self {
            claims: Default::default(),
            ctx,
        }
    }

    pub fn add_claim(&mut self, claim: Claim<E>) -> anyhow::Result<()> {
        ensure!(
            claim.point.len() == self.ctx.vp_info.max_num_variables,
            "invalid input len wrt to poly in ctx, claim point length: {}, expected point length: {}",
            claim.point.len(),
            self.ctx.vp_info.max_num_variables
        );
        self.claims.push(claim);
        Ok(())
    }

    pub fn verify<T: Transcript<E>>(self, proof: &Proof<E>, t: &mut T) -> anyhow::Result<Claim<E>> {
        let fs_challenges = t.read_challenges(self.claims.len());
        let (rs, ys): (Vec<_>, Vec<_>) = self.claims.into_iter().map(|c| (c.point, c.eval)).unzip();
        let y_res = aggregated_rlc(&ys, &fs_challenges);
        // check sumcheck proof
        let subclaim = IOPVerifierState::<E>::verify(y_res, &proof.sumcheck, &self.ctx.vp_info, t);
        // check sumcheck output: first check for the betas we can compute
        // for(int i = 0; i < a.size(); i++){y += a[i]*identity_eval(claims[i].first,P.randomness[0]);}
        let computed_y = fs_challenges
            .into_iter()
            .zip(rs)
            .fold(E::ZERO, |acc, (a_i, r_i)| {
                acc + a_i * identity_eval(&r_i, &proof.sumcheck.point)
            });
        let given_y = proof.evals[0];
        ensure!(computed_y == given_y, "beta evaluation do not match");
        // here instead of checking this claim via PCS, we actually put it in the output of the verify function.
        // That claims will be accumulated and verified elsewhere in the protocol.
        // Note the claim is only about the actual poly, not the betas since it has been verified just ^
        let claim = proof.extract_claim();

        // then check that both betas and poly evaluation lead to the outcome of the sumcheck, e.g. the sum
        let expected = proof.evals[0] * proof.evals[1];
        let computed = subclaim.expected_evaluation;
        ensure!(expected == computed, "final evals of sumcheck is not valid");
        Ok(claim)
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;
    use mpcs::PolynomialCommitmentScheme;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use crate::{
        Claim, default_transcript,
        testing::{Pcs, random_field_vector},
    };
    use itertools::Itertools;

    use super::{Context, Prover, Verifier};

    type F = GoldilocksExt2;

    #[test]
    fn test_pcs() {
        let num_vars = 10;
        let len = 1 << num_vars;
        let _param = Pcs::<F>::setup(len).expect("unable to setup");
    }

    #[test]
    fn test_same_poly_proof() -> anyhow::Result<()> {
        // number of vars
        let num_vars = 10 as usize;
        let poly_len = 1 << num_vars;
        let poly = random_field_vector::<F>(poly_len);
        let poly_mle = poly.clone().into_mle();
        // number of clains
        let m = 14;
        let claims = (0..m)
            .map(|_| {
                let r_i = random_field_vector::<F>(num_vars);
                let y_i = poly_mle.evaluate(&r_i);
                (r_i, y_i)
            })
            .collect_vec();
        // COMMON PART
        assert_eq!(poly.len(), 1 << num_vars);
        let ctx = Context::new(num_vars);
        // PROVER PART
        let mut t = default_transcript();
        let mut prover = Prover::new(poly_mle.clone());
        for (r_i, y_i) in claims.clone().into_iter() {
            prover.add_claim(Claim::new(r_i, y_i))?;
        }
        let proof = prover.prove(&mut t)?;
        // VERIFIER PART
        let mut t = default_transcript();
        let mut verifier = Verifier::new(&ctx);
        for (r_i, y_i) in claims.into_iter() {
            verifier.add_claim(Claim::new(r_i, y_i))?;
        }
        let claim = verifier.verify(&proof, &mut t)?;
        let expected = poly_mle.evaluate(&claim.point);
        assert_eq!(claim.eval, expected);
        Ok(())
    }
}
