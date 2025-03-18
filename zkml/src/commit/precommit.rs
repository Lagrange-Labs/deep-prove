//! This module contains logic to prove the correct opening of several claims from several independent
//! polynomials. These polynomials are committed at setup time. The proof contains only a single PCS
//! opening proofs and a sumcheck proof.
#![allow(dead_code)]

use std::collections::HashMap;

use crate::{
    Claim, VectorTranscript,
    commit::{aggregated_rlc, compute_beta_eval_poly, compute_betas_eval},
    iop::context::BIAS_POLY_ID,
    layers::Layer,
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
use transcript::Transcript;

use super::Pcs;

/// A polynomial has an unique ID associated to it.
pub type PolyID = usize;

// TODO: separate context into verifier and prover ctx once thing is working
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // PROVER PART
    pp: <Pcs<E> as PolynomialCommitmentScheme<E>>::ProverParam,
    commitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::CommitmentWithWitness,
    /// already flattened out polys evals by decreasing order
    polys: DenseMultilinearExtension<E>,
    // COMMON PART
    /// Needed to verify the sumcheck proof
    poly_aux: VPAuxInfo<E>,
    // keeps track of which layer do we layout first in the sequence of witness/poly we commit to
    // key is the id of the polynomial (we associated each polynomial to an "ID" so verifier and prover can
    // add any claim about any polynomial before proving/verifying)
    // value is a tuple:
    //  * the index of the poly in the vector of poly when ordered by decreasing order
    // .* the length of the polynomial
    poly_info: HashMap<PolyID, (usize, usize)>,
    // VERIFIER PART
    vp: <Pcs<E> as PolynomialCommitmentScheme<E>>::VerifierParam,
    vcommitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::Commitment,
}

impl<E> Default for Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    fn default() -> Self {
        let params = Pcs::setup(2).expect("unable to setup commitment");
        let (pp, vp) = Pcs::trim(params, 2).unwrap();
        Self {
            pp,
            vp,
            commitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::CommitmentWithWitness::default(),
            polys: DenseMultilinearExtension::<E>::default(),
            poly_aux: VPAuxInfo::<E>::default(),
            poly_info: HashMap::default(),
            vcommitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::Commitment::default(),
        }
    }
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// NOTE: it assumes the model's layers are already padded to power of two
    pub fn generate_from_model(m: &Model) -> anyhow::Result<Self> {
        Self::generate(
            m.layers()
                .flat_map(|(id, l)| match l {
                    Layer::Dense(dense) => vec![
                        Some((id, dense.matrix.evals_2d())),
                        Some((BIAS_POLY_ID + id, dense.bias.evals_flat())),
                    ],
                    Layer::Convolution(m) => vec![
                        Some((id, m.filter.get_conv_weights())),
                        Some((BIAS_POLY_ID + id, m.bias.evals_flat())),
                    ],
                    _ => vec![None],
                })
                .flatten()
                .collect_vec(),
        )
    }

    /// Generates the context given the set of individual polys that we need to commit to.
    /// It generates the parameters and commits to the MLEs of the polys.
    /// It also orders the polys by decreasing size and keep the ordering information.
    /// NOTE: it assumes each individual poly is padded to a power of two (they don't need to be of
    /// equal size)
    pub fn generate(mut polys: Vec<(PolyID, Vec<E>)>) -> anyhow::Result<Self> {
        assert!(polys.iter().all(|(_, w_i)| w_i.len().is_power_of_two()));
        // we pad the concatenated evals to the next power of two as well
        let padded_size = polys
            .iter()
            .map(|(_, w_i)| w_i.len())
            .sum::<usize>()
            .next_power_of_two();
        // sort in decreasing order
        polys.sort_by(|(_, w_i), (_, y_i)| y_i.len().cmp(&w_i.len()));
        let sorted_ids = polys.iter().map(|(id, poly)| (id, poly.len()));
        let id_order = HashMap::from_iter(
            sorted_ids
                .into_iter()
                .enumerate()
                .map(|(idx, (id, poly_len))| (*id, (idx, poly_len))),
        );
        let flattened = polys
            .into_iter()
            .map(|(_, w_i)| w_i)
            .flatten()
            .chain(std::iter::repeat(E::ZERO))
            .take(padded_size)
            .collect_vec();
        assert!(flattened.len().is_power_of_two());
        let num_vars = flattened.len().ilog2() as usize;
        let params = Pcs::setup(flattened.len()).expect("unable to setup commitment");
        let (pp, vp) = Pcs::trim(params, flattened.len()).unwrap();
        let mle = DenseMultilinearExtension::from_evaluations_ext_vec(num_vars, flattened);
        let comm = Pcs::commit(&pp, &mle).context("unable to commit")?;
        let vcommitment = Pcs::get_pure_commitment(&comm);
        Ok(Self {
            pp,
            poly_aux: VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
            vp,
            commitment: comm,
            vcommitment,
            polys: mle,
            poly_info: id_order,
        })
    }

    /// Write the relevant information to transcript, necessary for both prover and verifier.
    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        Pcs::write_commitment(&self.vcommitment, t).context("can't write commtiment")?;
        // TODO: write the rest of the struct
        Ok(())
    }
    fn sort_claims(
        &self,
        claims: Vec<IndividualClaim<E>>,
    ) -> anyhow::Result<Vec<IndividualClaim<E>>> {
        assert_eq!(
            claims.len(),
            self.poly_info.len(),
            "claims.len() = {} vs poly.len() = {}",
            claims.len(),
            self.poly_info.len()
        );
        let mut sorted_claims = claims.clone();
        for (idx, claim) in claims.into_iter().enumerate() {
            let (sorted_idx, poly_size) = self
                .poly_info
                .get(&claim.poly_id)
                .context("claim refers to unknown poly")?;
            let given_size = 1 << claim.claim.point.len() as usize;
            // verify the consistency of the individual polys lens with the claims
            ensure!(
                *poly_size == given_size,
                format!(
                    "claim {idx} doesn't have right format: poly {} has size {poly_size} vs input {given_size}",
                    claim.poly_id
                )
            );
            // order the claims according to the order of the poly defined in the setup phase
            sorted_claims[*sorted_idx] = claim;
        }
        Ok(sorted_claims)
    }
}

/// Structure that can prove the opening of multiple polynomials at different points.
pub struct CommitProver<E: ExtensionField> {
    /// all individual claims accumulated so far ordered by decreasing size of the poly
    claims: Vec<IndividualClaim<E>>,
}

impl<E: ExtensionField> CommitProver<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new() -> Self {
        Self {
            claims: Default::default(),
        }
    }

    /// Add a claim to be accumulated and checked via PCS
    /// The layer must be existing in the context, i.e. the setup phase must have processed the
    /// corresponding poly.
    /// TODO: add context so it can check the correct shape of the claim
    pub fn add_claim(&mut self, id: PolyID, claim: Claim<E>) -> anyhow::Result<()> {
        let claim = IndividualClaim { poly_id: id, claim };
        self.claims.push(claim);
        Ok(())
    }

    pub fn prove<T: Transcript<E>>(
        self,
        ctx: &Context<E>,
        t: &mut T,
    ) -> anyhow::Result<CommitProof<E>> {
        let sorted_claims = ctx.sort_claims(self.claims)?;

        ctx.write_to_transcript(t)?;
        let debug_transcript = t.clone();
        let fs_challenges = t.read_challenges(sorted_claims.len());
        let (full_r, full_y): (Vec<Vec<_>>, Vec<_>) = sorted_claims
            .into_iter()
            .map(|c| (c.claim.point, c.claim.eval))
            .multiunzip();

        // construct the matrix with the betas scaled
        let beta_mle = beta_matrix_mle(&full_r, &fs_challenges);
        assert_eq!(beta_mle.num_vars(), ctx.polys.num_vars());
        let mut full_poly = VirtualPolynomial::new(ctx.polys.num_vars());

        // TODO: remove that clone
        full_poly.add_mle_list(vec![beta_mle.into(), ctx.polys.clone().into()], E::ONE);

        assert_eq!(full_poly.aux_info, ctx.poly_aux);

        #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(full_poly.clone(), t);

        debug_assert!({
            let computed_result = aggregated_rlc(&full_y, &fs_challenges);
            debug_assert_eq!(sumcheck_proof.extract_sum(), computed_result,);

            let mut t = debug_transcript;
            let y_agg = aggregated_rlc(&full_y, &fs_challenges);
            let subclaim =
                IOPVerifierState::<E>::verify(y_agg, &sumcheck_proof, &ctx.poly_aux, &mut t);
            let computed = full_poly.evaluate(&subclaim.point_flat());
            debug_assert_eq!(computed, subclaim.expected_evaluation);
            true
        });
        // now we need to produce a proof of opening the witness MLE at the requested point
        // 1 because first poly is the betas poly, second is the witness one and we are only
        //   interested in producing a PCS opening proof for the witness one.
        let eval = state.get_mle_final_evaluations()[1];
        let point = sumcheck_proof.point.clone();
        let pcs_proof = Pcs::open(&ctx.pp, &ctx.polys, &ctx.commitment, &point, &eval, t)
            .context("can't produce opening proof")?;

        Ok(CommitProof {
            sumcheck: sumcheck_proof,
            opening: pcs_proof,
            individual_evals: state.get_mle_final_evaluations(),
        })
    }
}

pub struct CommitVerifier<E> {
    claims: Vec<IndividualClaim<E>>,
}

impl<E: ExtensionField> CommitVerifier<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new() -> Self {
        Self {
            claims: Default::default(),
        }
    }
    pub fn add_claim(&mut self, id: PolyID, claim: Claim<E>) -> anyhow::Result<()> {
        let claim = IndividualClaim { poly_id: id, claim };
        self.claims.push(claim);
        Ok(())
    }

    pub fn verify<T: Transcript<E>>(
        self,
        ctx: &Context<E>,
        proof: CommitProof<E>,
        t: &mut T,
    ) -> anyhow::Result<()> {
        let sorted_claims = ctx.sort_claims(self.claims)?;
        ctx.write_to_transcript(t)?;
        // 1. verify sumcheck proof
        let fs_challenges = t.read_challenges(sorted_claims.len());
        // pairs of (r,y) = (point,eval) claims
        // these are ordered in the decreasing order of the corresponding poly
        let (full_r, full_y): (Vec<Vec<_>>, Vec<_>) = sorted_claims
            .iter()
            .cloned()
            .map(|c| (c.claim.point, c.claim.eval))
            .multiunzip();
        let y_agg = aggregated_rlc(&full_y, &fs_challenges);
        let subclaim = IOPVerifierState::<E>::verify(y_agg, &proof.sumcheck, &ctx.poly_aux, t);

        // 2. verify PCS opening proof on the committed poly.
        let point = proof.sumcheck.point.clone();
        let eval = proof.individual_evals[1];
        Pcs::verify(&ctx.vp, &ctx.vcommitment, &point, &eval, &proof.opening, t)
            .context("invalid pcs opening")?;

        // 3. Manually evaluate the beta matrix MLE to get the output to check the final sumcheck
        //    claim

        // Size of each poly, ORDERED by decreasing size of poly
        let pairs: Vec<usize> = sorted_claims
            .iter()
            .enumerate()
            .map(|(_idx, claim)| {
                let (_, poly_len) = *ctx
                    .poly_info
                    .get(&claim.poly_id)
                    .expect("invalid layer - this is a bug");
                poly_len
            })
            .collect();
        let computed =
            compute_beta_eval_poly(pairs, &fs_challenges, &full_r, &proof.sumcheck.point);
        // 0 since poly is f_beta(..) * f_w(..) so beta comes firt
        let expected = proof.individual_evals[0];
        ensure!(computed == expected, "Error in beta evaluation check");
        // 4. just make sure the final claim of the sumcheck is consistent with f_beta(r) * f_w(r)
        // now that we've verified both individually we can just multiply and compare
        let full_eval = proof.individual_evals[0] * proof.individual_evals[1];
        ensure!(
            full_eval == subclaim.expected_evaluation,
            "Error in final evaluation check"
        );
        Ok(())
    }
}

/// Holds the proofs generated by the prover in the commit module for proving the correctness of a
/// vector of (point,eval) claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct CommitProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    sumcheck: IOPProof<E>,
    opening: <Pcs<E> as PolynomialCommitmentScheme<E>>::Proof,
    // output of the sumcheck prover: f_1(r) * f_2(r)
    individual_evals: Vec<E>,
}

/// Individual claim to accumulate with others in a single sumcheck + PCS opening
/// It implements equality traits for sorting in decreasing order
#[derive(Clone, Debug)]
struct IndividualClaim<E> {
    poly_id: PolyID,
    claim: Claim<E>,
}

/// compute the beta matrix from individual challenges and betas.
/// NOTE: currently the method is a bit convoluted since it uses the Matrix API which takes a
/// list of rows, while this beta matrix is defined from a list of columns.
fn beta_matrix_mle<E: ExtensionField>(ris: &[Vec<E>], ais: &[E]) -> DenseMultilinearExtension<E> {
    assert_eq!(ais.len(), ris.len());
    let padded_len = ris
        .iter()
        .map(|r_i| 1 << r_i.len())
        .sum::<usize>()
        .next_power_of_two() as usize;
    // compute the betas, and scale them by the associated verifier randomness
    // We just flatten them so when we do the combined sumcheck f_b(x) * f_w(x) then since both
    // are flattened, it's like a dot product.
    let betas = ris
        .iter()
        .zip(ais)
        .map(|(ri, a_i)| {
            compute_betas_eval(ri.as_slice())
                .into_iter()
                .map(move |b_i| b_i * a_i)
        })
        .flatten()
        .chain(std::iter::repeat(E::ZERO))
        .take(padded_len)
        .collect_vec();
    DenseMultilinearExtension::from_evaluations_ext_vec(betas.len().ilog2() as usize, betas)
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use ff::Field;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use super::compute_betas_eval;
    use crate::{
        Claim, default_transcript, pad_vector,
        tensor::Tensor,
        testing::{random_bool_vector, random_field_vector},
    };

    use super::{CommitProver, CommitVerifier, Context};

    type F = GoldilocksExt2;

    #[test]
    fn test_commit_matrix() -> anyhow::Result<()> {
        let mut rng = thread_rng();
        let n_poly = 2;
        // let range = thread_rng().gen_range(3..15);
        let matrices = (0..n_poly)
            .map(|_| {
                Tensor::random(vec![
                    rng.gen_range(3..24) as usize,
                    rng.gen_range(3..24) as usize,
                ])
                .pad_next_power_of_two_2d()
            })
            .enumerate()
            .collect_vec();
        let claims = (0..n_poly)
            .map(|i| {
                let point = matrices[i].1.random_eval_point();
                let eval = matrices[i].1.to_mle_2d().evaluate(&point);
                (matrices[i].0, point, eval)
            })
            .collect_vec();

        let polys = matrices
            .iter()
            .map(|(id, m)| (*id, m.evals_2d()))
            .collect_vec();
        let ctx = Context::generate(polys.clone())?;

        let mut prover = CommitProver::new();
        for (id, point, eval) in claims.iter() {
            prover.add_claim(*id, Claim::new(point.clone(), eval.clone()))?;
        }

        let mut t = default_transcript();
        let proof = prover.prove(&ctx, &mut t)?;

        // VERIFIER
        let mut verifier = CommitVerifier::new();
        let mut t = default_transcript();
        for (id, point, eval) in claims {
            verifier.add_claim(id, Claim::new(point, eval))?;
        }
        verifier.verify(&ctx, proof, &mut t)?;

        Ok(())
    }

    #[test]
    fn test_commit_batch() -> anyhow::Result<()> {
        let n_poly = 7;
        // let range = thread_rng().gen_range(3..15);
        let polys = (0..n_poly)
            .map(|_| pad_vector(random_field_vector::<F>(thread_rng().gen_range(3..24))))
            .enumerate()
            .collect_vec();
        let ctx = Context::generate(polys.clone())?;

        let mut claims = Vec::new();
        let mut prover = CommitProver::new();
        for (id, poly) in polys {
            let p = random_bool_vector::<F>(poly.len().ilog2() as usize);
            let eval = poly.clone().into_mle().evaluate(&p);
            claims.push((id, p.clone(), eval.clone()));
            prover.add_claim(id, Claim::new(p, eval))?;
        }

        let mut t = default_transcript();
        let proof = prover.prove(&ctx, &mut t)?;

        // VERIFIER
        let mut verifier = CommitVerifier::new();
        let mut t = default_transcript();
        for (id, point, eval) in claims {
            verifier.add_claim(id, Claim::new(point, eval))?;
        }
        verifier.verify(&ctx, proof, &mut t)?;

        Ok(())
    }

    #[test]
    fn test_beta_compute() {
        let n = 2 * 8;
        let r = random_bool_vector::<F>(n / 2);
        let betas = compute_betas_eval(&r);
        let beta_mle = betas.into_mle();
        assert_eq!(beta_mle.evaluate(&r), F::ONE);
        let r2 = random_bool_vector::<F>(n / 2);
        assert_ne!(beta_mle.evaluate(&r2), F::ONE);
        assert_eq!(beta_mle.evaluate(&r2), F::ZERO);
    }
}
