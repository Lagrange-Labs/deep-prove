use goldilocks::GoldilocksExt2;
use mpcs::{Basefold, BasefoldBasecodeParams};

type PcsGoldilocksBasecode = Basefold<GoldilocksExt2, BasefoldBasecodeParams>;

#[cfg(test)]
mod test {
    use ark_std::rand::thread_rng;
    use ff::Field;
    use ff_ext::ExtensionField;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use mpcs::PolynomialCommitmentScheme;
    use multilinear_extensions::{
        mle::{DenseMultilinearExtension, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};
    use transcript::{BasicTranscript, Transcript};

    use crate::{
        VectorTranscript,
        matrix::Matrix,
        model::test::{random_bool_vector, random_vector},
        prover::default_transcript,
        scale_vector, vector_to_mle,
    };

    use super::PcsGoldilocksBasecode;
    type F = GoldilocksExt2;

    #[test]
    fn test_commit_batch() {
        let mut rng = thread_rng();
        let m1 = Matrix::<F>::random((10, 10)).pad_next_power_of_two();
        let m1_r = m1.random_eval_point();
        let m1_y = m1.to_mle().evaluate(&m1_r);
        let m2 = Matrix::<F>::random((20, 20)).pad_next_power_of_two();
        let m2_r = m1.random_eval_point();
        let m2_y = m1.to_mle().evaluate(&m1_r);

        let mut ws = vec![m1.evals(), m2.evals()];
        ws.sort_by_key(|w| w.len());
        let mut sorted = ws.into_iter().rev().flatten().collect_vec();
        let full_mle = vector_to_mle(sorted);

        let mut t = default_transcript();
        let params = PcsGoldilocksBasecode::setup(1 << full_mle.num_vars()).unwrap();
        let (pp, vp) = PcsGoldilocksBasecode::trim(params, 1 << full_mle.num_vars()).unwrap();
        let comm = PcsGoldilocksBasecode::commit(&pp, &full_mle).expect("unable to commit");
        PcsGoldilocksBasecode::write_commitment(
            &PcsGoldilocksBasecode::get_pure_commitment(&comm),
            &mut t,
        )
        .expect("can't write commitment");

        let nclaims = 2; // one for each poly
        // order changed since we sort by decreasing size
        let inputs = vec![m2_r, m1_r];
        let claims = vec![m2_y, m1_y];

        let fs_challenges = t.read_challenges(nclaims);
        let y_agg = aggregated_rlc(&claims, &fs_challenges);
        // construct the matrix with the betas scaled
        let beta_mle = beta_matrix_mle(&inputs, &fs_challenges);
        let witness_mle = full_mle;
        let mut full_poly =
            VirtualPolynomial::new(std::cmp::max(beta_mle.num_vars(), witness_mle.num_vars()));
        full_poly.add_mle_list(vec![beta_mle.into(), witness_mle.into()], F::ONE);

        let (proof, state) = IOPProverState::<F>::prove_parallel(full_poly.clone(), &mut t);

        // VERIFIER part
        let mut t = default_transcript();
        PcsGoldilocksBasecode::write_commitment(
            &PcsGoldilocksBasecode::get_pure_commitment(&comm),
            &mut t,
        )
        .expect("can't write commitment");

        // claimed_sum = y_agg
        let subclaim = IOPVerifierState::<F>::verify(y_agg, &proof, &full_poly.aux_info, &mut t);
    }

    /// compute the beta matrix from individual challenges and betas.
    /// NOTE: currently the method is a bit convoluted since it uses the Matrix API which takes a
    /// list of rows, while this beta matrix is defined from a list of columns.
    fn beta_matrix_mle<E: ExtensionField>(
        ris: &[Vec<E>],
        ais: &[E],
    ) -> DenseMultilinearExtension<E> {
        assert_eq!(ais.len(), ris.len());
        let ncols = ris.len();
        let nrows = ris
            .iter()
            .max_by_key(|v| v.len())
            .expect("always at least one claim to aggregate")
            .len();
        // compute the betas, and scale them by the associated verifier randomness
        let betas = ris.iter().zip(ais).map(|(ri, a_i)| {
            compute_betas_eval(ri.as_slice())
                .into_iter()
                .map(move |b_i| b_i * a_i)
                .chain(std::iter::repeat(E::ZERO))
                .take(nrows)
        });
        // pass from a representation of vector of columns to vector of rows
        // NOTE: review the relevancy of that step and the general matrix organisation - sounds
        // silly to do that but want to keep same addressing scheme as in Matrix struct.
        let mut rows: Vec<Vec<E>> = vec![Vec::with_capacity(ncols); nrows];
        for (col_idx, col) in betas.enumerate() {
            for (row_idx, value) in col.enumerate() {
                rows[row_idx].push(value)
            }
        }
        DenseMultilinearExtension::from_evaluations_ext_vec(
            nrows.ilog2() as usize,
            rows.into_iter().flatten().collect_vec(),
        )
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
}
