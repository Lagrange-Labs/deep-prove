use ff_ext::ExtensionField;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub mod context;
pub mod same_poly;

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
        for (j, item) in temp.iter().enumerate().take(current_size) {
            let idx = j << 1;
            let t = r_elem * *item;
            betas[idx] = *item - t;
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
        .fold(|| E::ZERO, |acc, (claim, r)| acc + *claim * *r)
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
        eval * (*r1_i * *r2_i + (one - *r1_i) * (one - *r2_i))
    })
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{commit::identity_eval, testing::random_bool_vector};
    use p3_field::FieldAlgebra;
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
}
