use ff_ext::ExtensionField;
use mpcs::{Basefold, BasefoldRSParams};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

pub mod context;
pub mod precommit;
pub mod same_poly;

pub(crate) type Pcs<E> = Basefold<E, BasefoldRSParams>;
/// Compute the vector (beta(r,1), ... ,beta(r,2^{|r|}))
/// This function uses the dynamic programing technique of Libra
pub fn compute_betas_eval<E: ExtensionField>(r: &[E]) -> Vec<E> {
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

/// Random linear combination of claims and random elements derived from transcript
pub fn aggregated_rlc<E: ExtensionField>(claims: &[E], challenges: &[E]) -> E {
    assert_eq!(claims.len(), challenges.len());
    claims
        .par_iter()
        .zip(challenges)
        .fold(|| E::ZERO, |acc, (claim, r)| acc + *claim * r)
        .reduce(|| E::ZERO, |res, acc| res + acc)
}

/// Verifier logic to cheaply compute the evaluation of a matrix of betas vector at a given point.
/// * poly_lens is the vector of the length of the poly, ordered by how the poly were used by the prover
/// * challenges for the random linear combination
/// * ris: vector of r_i from the list of claim
/// * point: evaluation point desired, usually comes from the last step of the sumcheck.
pub fn compute_beta_eval_poly<E: ExtensionField>(
    poly_lens: Vec<usize>,
    fs_challenges: &[E],
    ris: &[Vec<E>],
    point: &[E],
) -> E {
    let mut beta_evals = fs_challenges
        .into_par_iter()
        .zip(ris.into_par_iter())
        .map(|(x_i, r_i)| *x_i * identity_eval(r_i, point))
        .collect::<Vec<_>>();

    let mut pos = 0;
    for (idx, poly_size) in poly_lens.iter().enumerate() {
        let prod = get_offset_product(*poly_size, pos, point);
        pos += poly_size;
        beta_evals[idx] *= prod;
    }

    beta_evals
        .par_iter()
        .fold(|| E::ZERO, |acc, &eval| acc + eval)
        .reduce(|| E::ZERO, |res, acc| res + acc)
}

/// Compute multilinear identity test between two points: returns 1 if points are equal, 0 if different.
/// Used as equality checker in polynomial commitment verification.
/// Compute Beta(r1,r2) = prod_{i \in [n]}((1-r1[i])(1-r2[i]) + r1[i]r2[i])
/// NOTE: the two vectors don't need to be of equal size. It compute the identity eval on the
/// minimum size between the two vector
pub(crate) fn identity_eval<E: ExtensionField>(r1: &[E], r2: &[E]) -> E {
    let max_elem = std::cmp::min(r1.len(), r2.len());
    let v1 = &r1[..max_elem];
    let v2 = &r2[..max_elem];
    v1.iter().zip(v2).fold(E::ONE, |eval, (r1_i, r2_i)| {
        let one = E::ONE;
        eval * (*r1_i * r2_i + (one - r1_i) * (one - r2_i))
    })
}

/// Computes the product of offset terms for a given position and random vector
/// fn get_offset_product<E: ExtensionField>(size: usize, mut pos: usize, r: &[E]) -> E {
///    // 1. Convert 'pos' into its binary representation
///    let mut bits = vec![E::ZERO; r.len()];
///    // Convert position to binary representation in little-endian order
///    for i in (0..r.len()).rev() {  // Changed: iterate in reverse
///        bits[i] = if pos & 1 == 1 { E::ONE } else { E::ZERO };
///        pos >>= 1;
///    }
///
///    let num_vars_needed = r.len() - size.ilog2() as usize;
///    // Compute product for the required number of variables
///    (0..num_vars_needed).fold(E::ONE, |prod, i| {
///        let bit = bits[r.len() - 1 - i];
///        let r_i = r[r.len() - 1 - i];
///        prod * (bit * r_i + (E::ONE - bit) * (E::ONE - r_i))
///    })
/// }
///
/// Computes the offset product for a given claim.
///
/// # Parameters
/// - `claim_size`: The size (number of entries) corresponding to the claim.
/// - `mut pos`: The position (an integer) whose binary representation will be used.
/// - `rand_vec`: The vector of random field elements (corresponding to `r` in the C++ code).
///
/// # Returns
/// The product computed from the bits of `pos` and corresponding values in `rand_vec`.
pub(crate) fn get_offset_product<E: ExtensionField>(
    claim_size: usize,
    mut pos: usize,
    rand_vec: &[E],
) -> E {
    // Create a vector to hold the bits.
    // In the C++ code, bits are pushed in LSB-first order;
    // here we fill the vector in reverse so that the most significant end of the vector
    // contains what C++ would later pick from bits[r.size()-1 - i].
    let mut bits = vec![E::ZERO; rand_vec.len()];

    // Fill 'bits' such that bits[0] becomes the LSB, bits[len-1] the MSB.
    // By iterating in reverse, we mimic the eventual reversal in the C++ code.
    for i in 0..rand_vec.len() {
        bits[i] = if pos & 1 == 1 { E::ONE } else { E::ZERO };
        pos >>= 1;
    }

    // The number of variables to be "folded" is determined by log2(claim_size).
    // This is equivalent to 'r.size() - (int)log2(size)' in C++.
    let num_vars_needed = rand_vec.len() - claim_size.ilog2() as usize;

    // Now, accumulate the product similar to the C++ loop.
    (0..num_vars_needed).fold(E::ONE, |prod, i| {
        // Access from the end of the vector (i.e. effectively reversing the bits again)
        let bit = bits[rand_vec.len() - 1 - i];
        let r_i = rand_vec[rand_vec.len() - 1 - i];
        prod * (bit * r_i + (E::ONE - bit) * (E::ONE - r_i))
    })
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::{
        commit::{get_offset_product, identity_eval},
        testing::random_bool_vector,
    };
    use ff::Field;
    type F = GoldilocksExt2;

    #[test]
    fn test_identity_eval() {
        // FAILING WITH THIS POINT
        let n = 4;
        let r1 = random_bool_vector::<F>(n);
        println!("r1: {:?}", r1);

        // When vectors are identical, should return 1
        let r2 = r1.clone();
        let result = identity_eval(&r1, &r2);
        assert_eq!(result, F::ONE);

        // When vectors are different, should return 0
        let r2 = random_bool_vector::<F>(n);
        println!("r2: {:?}", r2);
        let result = identity_eval(&r1, &r2);
        assert!(r1 == r2 || result == F::ZERO);
    }

    #[test]
    fn test_get_offset_product() {
        let size = 4; // Original polynomial of size 4 (2^2 variables)
        let r = vec![F::ONE, F::ZERO, F::ONE, F::ZERO]; // Some test point

        // When evaluating at position 0 (first slice)
        let result0 = get_offset_product(size, 0, &r);

        // When evaluating at position 4 (second slice)
        let result4 = get_offset_product(size, 4, &r);

        // Results will be different because they're enforcing different slices
        assert_ne!(result0, result4);
    }
}
