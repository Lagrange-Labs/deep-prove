use std::collections::{BTreeSet, HashMap};

use crate::{
    VectorTranscript,
    model::{Model, PolyID},
};
use anyhow::{ensure, Context as CC};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::{Basefold, BasefoldBasecodeParams, PolynomialCommitmentScheme};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

type Pcs<E> = Basefold<E, BasefoldBasecodeParams>;

// TODO: separate context into verifier and prover ctx once thing is working
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    // PROVER PART
    pp: <Pcs<E> as PolynomialCommitmentScheme<E>>::ProverParam,
    commitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::CommitmentWithWitness,
    /// already flattened out polys evals by decreasing order
    polys: DenseMultilinearExtension<E>,
    //
    // COMMON PART
    /// Needed to verify the sumcheck proof
    poly_aux: VPAuxInfo<E>,
    // keeps track of which layer do we layout first in the sequence of witness/poly we commit to
    // key is the id of the polynomial (we associated each polynomial to an "ID" so verifier and prover can 
    // add any claim about any polynomial before proving/verifying)
    // value is a tuple:
    //  * 0: the index of the poly in the vector of poly when ordered by decreasing order
    //  * 1: the size of the poly - needed by verifier to efficiently compute the beta MLE eval at random point
    poly_info: HashMap<PolyID, (usize,usize)>,
    // 
    // VERIFIER PART
    vp: <Pcs<E> as PolynomialCommitmentScheme<E>>::VerifierParam,
    vcommitment: <Pcs<E> as PolynomialCommitmentScheme<E>>::Commitment,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// NOTE: it assumes the model's layers are already padded to power of two
    pub fn generate_from_model(m: &Model<E>) -> anyhow::Result<Self> {
        Self::generate(m.layers().map(|(id, l)| (id, l.evals())).collect_vec())
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
        let sorted_ids = polys.iter().map(|(id, poly)| (id,poly.len()));
        println!("CONTEXT: sorted_ids: {:?}",sorted_ids);
        let id_order =
            HashMap::from_iter(sorted_ids.into_iter().enumerate().map(|(i, (id,poly_len))| (*id, (i,poly_len))));
        println!("CONTEXT: id_order: {:?}",id_order);

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
        Pcs::write_commitment(&Pcs::get_pure_commitment(&self.commitment), t)
            .context("can't write commtiment")?;
        Ok(())
    }
}

/// Structure that can prove the opening of multiple polynomials at different points.
struct CommitProver<E: ExtensionField> {
    /// all individual claims accumulated so far ordered by decreasing size of the poly
    claims: BTreeSet<IndividualClaim<E>>,
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
    pub fn add_claim(
        &mut self,
        ctx: &Context<E>,
        id: PolyID,
        point: Vec<E>,
        // Note this one is not necessary and should be removed down the line for the prover
        eval: E,
    ) -> anyhow::Result<()> {
        let index = ctx
            .poly_info
            .get(&id)
            .context("no layer saved in ctx for {layer}")?.0;
        let claim = IndividualClaim {
            index_order: index,
            poly_id: id,
            point,
            eval,
        };
        self.claims.insert(claim);
        Ok(())
    }

    pub fn prove<T: Transcript<E>>(
        self,
        ctx: &Context<E>,
        t: &mut T,
    ) -> anyhow::Result<CommitProof<E>> {
        println!("\n");
        println!("PROVER: claims: {:?}",self.claims);
        ctx.write_to_transcript(t)?;
        let mut debug_transcript = t.clone();
        let fs_challenges = t.read_challenges(self.claims.len());
        let (full_r, full_y): (Vec<Vec<_>>, Vec<_>) = self
            .claims
            .into_iter()
            .map(|c| (c.point, c.eval))
            .multiunzip();

        println!("PROVER: fs_challenges: {:?}",fs_challenges);
        println!("PROVER: full_r: {:?}",full_r);
        // construct the matrix with the betas scaled
        let beta_mle = beta_matrix_mle(&full_r, &fs_challenges);
        assert_eq!(beta_mle.num_vars(), ctx.polys.num_vars());
        let mut full_poly = VirtualPolynomial::new(ctx.polys.num_vars());

        full_poly.add_mle_list(vec![beta_mle.into(), ctx.polys.clone().into()], E::ONE);
        assert_eq!(full_poly.aux_info, ctx.poly_aux);

        #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(full_poly.clone(), t);

        debug_assert!({
            debug_assert_eq!(
                sumcheck_proof.extract_sum(),
                aggregated_rlc(&full_y, &fs_challenges)
            );

            let mut t = debug_transcript;
            let y_agg = aggregated_rlc(&full_y, &fs_challenges);
            let subclaim =
                IOPVerifierState::<E>::verify(y_agg, &sumcheck_proof, &ctx.poly_aux, &mut t);
            let computed = full_poly.evaluate(&subclaim.point_flat());
            debug_assert_eq!(computed, subclaim.expected_evaluation);
            println!("PROVER: final_beta_eval(r) {:?}",state.get_mle_final_evaluations()[0]);
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

struct CommitVerifier<E> {
    claims: BTreeSet<IndividualClaim<E>>,
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
    pub fn add_claim(
        &mut self,
        ctx: &Context<E>,
        id: PolyID,
        point: Vec<E>,
        eval: E,
    ) -> anyhow::Result<()> {
        let index = ctx
            .poly_info
            .get(&id)
            .context("no layer saved in ctx for {layer}")?.0;
        let claim = IndividualClaim {
            index_order: index,
            poly_id: id,
            point,
            eval,
        };
        self.claims.insert(claim);
        Ok(())
    }

    pub fn verify<T: Transcript<E>>(
        self,
        ctx: &Context<E>,
        proof: CommitProof<E>,
        t: &mut T,
    ) -> anyhow::Result<()> {
        println!("\n");
        println!("VERIFIER: claims: {:?}",self.claims);
        ctx.write_to_transcript(t)?;
        // 1. verify sumcheck proof
        let fs_challenges = t.read_challenges(self.claims.len());
        println!("VERIFIER: fs_challenge: {:?}",fs_challenges);
        // pairs of (r,y) = (point,eval) claims
        // these are ordered in the decreasing order of the corresponding poly
        let (full_r, full_y): (Vec<Vec<_>>, Vec<_>) = self
            .claims
            .iter()
            .cloned()
            .map(|c| (c.point, c.eval))
            .multiunzip();
        println!("VERIFIER: full_r: {:?}",full_r);
        let y_agg = aggregated_rlc(&full_y, &fs_challenges);
        let subclaim = IOPVerifierState::<E>::verify(y_agg, &proof.sumcheck, &ctx.poly_aux, t);

        // 2. verify PCS opening proof on the committed poly.
        let point = proof.sumcheck.point.clone();
        let eval = proof.individual_evals[1];
        Pcs::verify(&ctx.vp, &ctx.vcommitment, &point, &eval, &proof.opening, t)
            .context("invalid pcs opening")?;

        // 3. Manually evaluate the beta matrix MLE to get the output to check the final sumcheck
        //    claim

        // index of the poly and size
        let pairs: Vec<(usize, usize)> = self.claims
        .iter()
        .enumerate()
        .map(|(idx,claim)| {
            let info = *ctx.poly_info.get(&claim.poly_id).expect("invalid layer - this is a bug");
            assert_eq!(info.0,idx);
            info
        })
        .collect();
        println!("VERIFIER: pairs: {:?}",pairs);

        // Final verification step
        let mut beta_evals = fs_challenges.iter()
            .zip(&full_r)
            .map(|(&x_i, r_i)| x_i * identity_eval(r_i, &proof.sumcheck.point))
            .collect::<Vec<_>>();

        let mut pos = 0;
        for (idx, poly_size) in pairs.iter() {
            let prod = get_offset_product(*poly_size, pos, &proof.sumcheck.point);
            pos += poly_size;
            beta_evals[*idx] *= prod;
        }

        let computed = beta_evals.iter().fold(E::ZERO, |acc, &eval| acc + eval);
        let expected = proof.individual_evals[0];
        println!("VERIFIER: computed final beta(r) : {:?}", computed);
        ensure!(computed == expected, "Error in final check");
        Ok(())
    }
}

/// Holds the proofs generated by the prover in the commit module for proving the correctness of a
/// vector of (point,eval) claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
struct CommitProof<E: ExtensionField>
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
#[derive(Clone,Debug)]
struct IndividualClaim<E> {
    // the index in which this individual claim has been committed to in the aggregated poly
    // it's used to order the claims and polynomials by decreasing size
    index_order: usize,
    poly_id: PolyID,
    point: Vec<E>,
    eval: E,
}

impl<E: ExtensionField> PartialEq for IndividualClaim<E> {
    fn eq(&self, other: &Self) -> bool {
        self.index_order == other.index_order
    }
}

impl<E: ExtensionField> Eq for IndividualClaim<E> {}

impl<E: ExtensionField> PartialOrd for IndividualClaim<E> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.index_order.partial_cmp(&other.index_order)
    }
}

impl<E: ExtensionField> Ord for IndividualClaim<E> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index_order.cmp(&other.index_order)
    }
}

/// Random linear combination of claims and random elements derived from transcript
fn aggregated_rlc<E: ExtensionField>(claims: &[E], challenges: &[E]) -> E {
    assert_eq!(claims.len(), challenges.len());
    claims
        .iter()
        .zip(challenges)
        .fold(E::ZERO, |acc, (claim, r)| acc + *claim * r)
}

/// Compute the vector (beta(r,1),....,beta(r,2^{|r|}))
/// This function uses the dynamic programing technique of Libra
fn compute_betas_eval<E: ExtensionField>(r: &[E]) -> Vec<E> {
    let n = r.len();
    let size = 1 << n;
    let mut betas = vec![E::ZERO; size];
    betas[0] = E::ONE;

    for i in 0..n {
        let current_size = 1 << i;
        let temp = betas[..current_size].to_vec();
        let r_elem = r[r.len() - 1 - i];
        for j in 0..current_size {
            let idx = j << 1;
            let t = r_elem * temp[j];
            betas[idx] = temp[j] - t;
            betas[idx + 1] = t;
        }
    }
    betas
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

/// Compute multilinear identity test between two points: returns 1 if points are equal, 0 if different.
/// Used as equality checker in polynomial commitment verification.
/// Compute Beta(r1,r2) = prod_{i \in [n]}((1-r1[i])(1-r2[i]) + r1[i]r2[i])
/// NOTE: the two vectors don't need to be of equal size. It compute the identity eval on the 
/// minimum size between the two vector
fn identity_eval<E: ExtensionField>(r1: &[E], r2: &[E]) -> E {
    let max_elem = std::cmp::min(r1.len(),r2.len());
    let v1 = &r1[..max_elem];
    let v2 = &r2[..max_elem];
    v1.iter()
        .zip(v2)
        .fold(E::ONE, |eval, (r1_i, r2_i)| {
            let one = E::ONE;
            eval * (*r1_i * r2_i + (one - r1_i) * (one - r2_i))
        })
}

/// Computes the product of offset terms for a given position and random vector
fn get_offset_product<E: ExtensionField>(size: usize, mut pos: usize, r: &[E]) -> E {
    // 1. Convert 'pos' into its binary representation
    let mut bits = vec![E::ZERO; r.len()];
    // Convert position to binary representation in little-endian order
    for i in (0..r.len()).rev() {  // Changed: iterate in reverse
        bits[i] = if pos & 1 == 1 { E::ONE } else { E::ZERO };
        pos >>= 1;
    }
    
    let num_vars_needed = r.len() - size.ilog2() as usize;
    // Compute product for the required number of variables
    (0..num_vars_needed).fold(E::ONE, |prod, i| {
        let bit = bits[r.len() - 1 - i];
        let r_i = r[r.len() - 1 - i];
        prod * (bit * r_i + (E::ONE - bit) * (E::ONE - r_i))
    })
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use ff::Field;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{
        commit::{compute_betas_eval, get_offset_product, identity_eval},
        model::test::{random_bool_vector, random_vector},
        pad_vector,
        prover::default_transcript, vector_to_mle,
    };

    use super::{CommitProver, CommitVerifier, Context};

    type F = GoldilocksExt2;

    #[test]
    fn test_commit_batch() -> anyhow::Result<()> {
        let n_poly = 2;
        let polys = (0..n_poly)
            .map(|_| pad_vector(random_vector::<F>(thread_rng().gen_range(3..15))))
            .enumerate()
            .collect_vec();
        let ctx = Context::generate(polys.clone())?;

        let mut claims = Vec::new();
        let mut prover = CommitProver::new();
        for (id, poly) in polys {
            let p = random_bool_vector(poly.len().ilog2() as usize);
            let eval = vector_to_mle(poly.clone()).evaluate(&p);
            claims.push((id, p.clone(), eval.clone()));
            prover.add_claim(&ctx, id, p, eval)?;
        }

        let mut t = default_transcript();
        let proof = prover.prove(&ctx, &mut t)?;

        // VERIFIER
        let mut verifier = CommitVerifier::new();
        let mut t = default_transcript();
        for (id, point, eval) in claims {
            verifier.add_claim(&ctx, id, point, eval)?;
        }
        verifier.verify(&ctx, proof, &mut t)?;

        Ok(())
    }

    #[test]
    fn test_beta_compute() {
        let n = 2 * 8;
        let r = random_bool_vector::<F>(n / 2);
        let betas = compute_betas_eval(&r);
        let beta_mle = vector_to_mle(betas);
        assert_eq!(beta_mle.evaluate(&r), F::ONE);
        let r2 = random_bool_vector::<F>(n / 2);
        assert_ne!(beta_mle.evaluate(&r2), F::ONE);
        assert_eq!(beta_mle.evaluate(&r2), F::ZERO);
    }

    #[test]
    fn test_identity_eval() {
        let n = 4;
        let r1 = random_bool_vector::<F>(n);
        
        // When vectors are identical, should return 1
        let r2 = r1.clone();
        let result = identity_eval(&r1, &r2);
        assert_eq!(result, F::ONE);
    
        // When vectors are different, should return 0
        let r2 = random_bool_vector::<F>(n);
        let result = identity_eval(&r1, &r2);
        assert_eq!(result, F::ZERO);
    }

    #[test]
    fn test_get_offset_product() {
        let size = 4;  // Original polynomial of size 4 (2^2 variables)
        let n = 4;     // Total variables in concatenated space
        let r = vec![F::ONE, F::ZERO, F::ONE, F::ZERO];  // Some test point
        
        // When evaluating at position 0 (first slice)
        let result0 = get_offset_product(size, 0, &r);
        
        // When evaluating at position 4 (second slice)
        let result4 = get_offset_product(size, 4, &r);
        
        // Results will be different because they're enforcing different slices
        assert_ne!(result0, result4);
    }
}
