use std::{array, sync::Arc};

use ark_std::{rand::RngCore, test_rng};
use ff_ext::{ExtensionField, FromUniformBytes, GoldilocksExt2};
use itertools::Itertools;
use multilinear_extensions::{
    mle::DenseMultilinearExtension,
    op_mle,
    util::max_usable_threads,
    virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
};
use p3_field::FieldAlgebra;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use transcript::{BasicTranscript, Transcript};

use crate::{
    structs::{IOPProverState, IOPVerifierState},
    util::{ceil_log2, interpolate_uni_poly},
};

// TODO add more tests related to various num_vars combination after PR #162

fn test_sumcheck<E: ExtensionField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) {
    let mut rng = test_rng();
    let mut transcript = BasicTranscript::new(b"test");

    let (poly, asserted_sum) =
        VirtualPolynomial::<E>::random(nv, num_multiplicands_range, num_products, &mut rng);
    let poly_info = poly.aux_info.clone();
    #[allow(deprecated)]
    let (proof, _) = IOPProverState::<E>::prove_parallel(poly.clone(), &mut transcript);

    let mut transcript = BasicTranscript::new(b"test");
    let subclaim = IOPVerifierState::<E>::verify(asserted_sum, &proof, &poly_info, &mut transcript);
    assert!(
        poly.evaluate(
            subclaim
                .point
                .iter()
                .map(|c| c.elements)
                .collect::<Vec<_>>()
                .as_ref()
        ) == subclaim.expected_evaluation,
        "wrong subclaim"
    );
    // The point given in the proof is the same as the point returned by the verifier method
    // It is the point over which to evaluate the original polynomial over
    assert_eq!(
        proof.point,
        subclaim.point.iter().map(|c| c.elements).collect_vec()
    );
}

fn test_sumcheck_internal<E: ExtensionField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) {
    let mut rng = test_rng();
    let (poly, asserted_sum) =
        VirtualPolynomial::<E>::random(nv, num_multiplicands_range, num_products, &mut rng);
    let (poly_info, num_variables) = (poly.aux_info.clone(), poly.aux_info.max_num_variables);
    #[allow(deprecated)]
    let mut prover_state = IOPProverState::prover_init_parallel(poly.clone());
    let mut verifier_state = IOPVerifierState::verifier_init(&poly_info);
    let mut challenge = None;

    let mut transcript = BasicTranscript::new(b"test");

    transcript.append_message(b"initializing transcript for testing");

    for _ in 0..num_variables {
        let prover_message =
            IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge);

        challenge = Some(IOPVerifierState::verify_round_and_update_state(
            &mut verifier_state,
            &prover_message,
            &mut transcript,
        ));
    }
    // pushing the last challenge point to the state
    if let Some(p) = challenge {
        prover_state.challenges.push(p);
        // fix last challenge to collect final evaluation
        prover_state
            .poly
            .flattened_ml_extensions
            .par_iter_mut()
            .for_each(|mle| {
                if num_variables == 1 {
                    // first time fix variable should be create new instance
                    if mle.num_vars() > 0 {
                        *mle = mle.fix_variables(&[p.elements]).into();
                    } else {
                        *mle = Arc::new(DenseMultilinearExtension::from_evaluation_vec_smart(
                            0,
                            mle.get_base_field_vec().to_vec(),
                        ))
                    }
                } else {
                    let mle = Arc::get_mut(mle).unwrap();
                    if mle.num_vars() > 0 {
                        mle.fix_variables_in_place(&[p.elements]);
                    }
                }
            });
    };
    let subclaim = IOPVerifierState::check_and_generate_subclaim(&verifier_state, &asserted_sum);
    assert!(
        poly.evaluate(
            subclaim
                .point
                .iter()
                .map(|c| c.elements)
                .collect::<Vec<_>>()
                .as_ref()
        ) == subclaim.expected_evaluation,
        "wrong subclaim"
    );
}

#[test]
fn test_trivial_polynomial() {
    test_trivial_polynomial_helper::<GoldilocksExt2>();
}

fn test_trivial_polynomial_helper<E: ExtensionField>() {
    let nv = 1;
    let num_multiplicands_range = (3, 5);
    let num_products = 5;

    test_sumcheck::<E>(nv, num_multiplicands_range, num_products);
    test_sumcheck_internal::<E>(nv, num_multiplicands_range, num_products);
}

#[test]
fn test_normal_polynomial() {
    test_normal_polynomial_helper::<GoldilocksExt2>();
}

fn test_normal_polynomial_helper<E: ExtensionField>() {
    let nv = 12;
    let num_multiplicands_range = (3, 5);
    let num_products = 5;

    test_sumcheck::<E>(nv, num_multiplicands_range, num_products);
    test_sumcheck_internal::<E>(nv, num_multiplicands_range, num_products);
}

// #[test]
// fn zero_polynomial_should_error() {
//     let nv = 0;
//     let num_multiplicands_range = (4, 13);
//     let num_products = 5;

//     assert!(test_sumcheck(nv, num_multiplicands_range, num_products).is_err());
//     assert!(test_sumcheck_internal(nv, num_multiplicands_range, num_products).is_err());
// }

#[test]
fn test_extract_sum() {
    test_extract_sum_helper::<GoldilocksExt2>();
}

fn test_extract_sum_helper<E: ExtensionField>() {
    let mut rng = test_rng();
    let mut transcript = BasicTranscript::new(b"test");
    let (poly, asserted_sum) = VirtualPolynomial::<E>::random(8, (2, 3), 3, &mut rng);
    #[allow(deprecated)]
    let (proof, _) = IOPProverState::<E>::prove_parallel(poly, &mut transcript);
    assert_eq!(proof.extract_sum(), asserted_sum);
}

struct DensePolynomial(Vec<GoldilocksExt2>);

impl DensePolynomial {
    fn rand(degree: usize, mut rng: &mut impl RngCore) -> Self {
        Self(
            (0..degree)
                .map(|_| GoldilocksExt2::random(&mut rng))
                .collect(),
        )
    }

    fn evaluate(&self, p: &GoldilocksExt2) -> GoldilocksExt2 {
        let mut powers_of_p = *p;
        let mut res = self.0[0];
        for &c in self.0.iter().skip(1) {
            res += powers_of_p * c;
            powers_of_p *= *p;
        }
        res
    }
}

#[test]
fn test_interpolation() {
    let mut prng = ark_std::test_rng();

    // test a polynomial with 20 known points, i.e., with degree 19
    let poly = DensePolynomial::rand(20 - 1, &mut prng);
    let evals = (0..20)
        .map(|i| poly.evaluate(&GoldilocksExt2::from_canonical_u64(i as u64)))
        .collect::<Vec<GoldilocksExt2>>();
    let query = GoldilocksExt2::random(&mut prng);

    assert_eq!(poly.evaluate(&query), interpolate_uni_poly(&evals, query));

    // test a polynomial with 33 known points, i.e., with degree 32
    let poly = DensePolynomial::rand(33 - 1, &mut prng);
    let evals = (0..33)
        .map(|i| poly.evaluate(&GoldilocksExt2::from_canonical_u64(i as u64)))
        .collect::<Vec<GoldilocksExt2>>();
    let query = GoldilocksExt2::random(&mut prng);

    assert_eq!(poly.evaluate(&query), interpolate_uni_poly(&evals, query));

    // test a polynomial with 64 known points, i.e., with degree 63
    let poly = DensePolynomial::rand(64 - 1, &mut prng);
    let evals = (0..64)
        .map(|i| poly.evaluate(&GoldilocksExt2::from_canonical_u64(i as u64)))
        .collect::<Vec<GoldilocksExt2>>();
    let query = GoldilocksExt2::random(&mut prng);

    assert_eq!(poly.evaluate(&query), interpolate_uni_poly(&evals, query));
}

const NUM_DEGREE: usize = 3;
const NV: usize = 20;
type E = GoldilocksExt2;

#[test]
fn test_nikko_devirgo() {
    let (_, _ceno, devirgo) = { prepare_input::<E>(NV) };
    println!("INPUTS {} prepared", NV);
    run_devirgo("devirgo", devirgo);
}

#[test]
fn test_nikko_ceno() {
    let (_, ceno, _devirgo) = { prepare_input::<E>(NV) };
    println!("INPUTS {} prepared", NV);
    run_ceno_prover("ceno", ceno);
}

pub fn run_devirgo<'a, E: ExtensionField>(name: &str, ps: Vec<VirtualPolynomial<'a, E>>) {
    let mut prover_transcript = BasicTranscript::<E>::new(b"test");
    let threads = max_usable_threads();
    let instant = std::time::Instant::now();
    let (_sumcheck_proof_v2, _) =
        IOPProverState::<E>::prove_batch_polys(threads, ps, &mut prover_transcript);
    let elapsed = instant.elapsed().as_millis();
    println!("{}: elapsed: {}ms", name, elapsed);
}

pub fn run_ceno_prover<'a, E: ExtensionField>(name: &str, p: VirtualPolynomial<'a, E>) {
    let instant = std::time::Instant::now();
    let mut prover_transcript = BasicTranscript::new(b"test");
    #[allow(deprecated)]
    let (_sumcheck_proof_v1, _) = IOPProverState::<E>::prove_parallel(p, &mut prover_transcript);
    let elapsed = instant.elapsed();
    println!("{}: elapsed: {}ms", name, elapsed.as_millis());
}

/// transpose 2d vector without clone
pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}
fn prepare_input<'a, E: ExtensionField>(
    nv: usize,
) -> (E, VirtualPolynomial<'a, E>, Vec<VirtualPolynomial<'a, E>>) {
    let mut rng = test_rng();
    let max_thread_id = max_usable_threads();
    let size_log2 = ceil_log2(max_thread_id);
    let fs: [ArcMultilinearExtension<'a, E>; NUM_DEGREE] = array::from_fn(|_| {
        let mle: ArcMultilinearExtension<'a, E> =
            DenseMultilinearExtension::<E>::random(nv, &mut rng).into();
        mle
    });

    let mut virtual_poly_v1 = VirtualPolynomial::new(nv);
    virtual_poly_v1.add_mle_list(fs.to_vec(), E::ONE);

    // devirgo version
    let virtual_poly_v2: Vec<Vec<ArcMultilinearExtension<'a, E>>> = transpose(
        fs.iter()
            .map(|f| match &f.evaluations() {
                multilinear_extensions::mle::FieldType::Base(evaluations) => evaluations
                    .chunks((1 << nv) >> size_log2)
                    .map(|chunk| {
                        let mle: ArcMultilinearExtension<'a, E> =
                            DenseMultilinearExtension::<E>::from_evaluations_vec(
                                nv - size_log2,
                                chunk.to_vec(),
                            )
                            .into();
                        mle
                    })
                    .collect_vec(),
                _ => unreachable!(),
            })
            .collect(),
    );
    let virtual_poly_v2: Vec<VirtualPolynomial<E>> = virtual_poly_v2
        .into_iter()
        .map(|fs| {
            let mut virtual_polynomial = VirtualPolynomial::new(fs[0].num_vars());
            virtual_polynomial.add_mle_list(fs, E::ONE);
            virtual_polynomial
        })
        .collect();

    let asserted_sum = fs
        .iter()
        .fold(vec![E::ONE; 1 << nv], |mut acc, f| {
            op_mle!(f, |f| {
                (0..f.len()).zip(acc.iter_mut()).for_each(|(i, acc)| {
                    *acc *= f[i];
                });
                acc
            })
        })
        .into_iter()
        .sum::<E>();

    (asserted_sum, virtual_poly_v1, virtual_poly_v2)
}
