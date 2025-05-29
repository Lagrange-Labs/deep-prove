use crate::{
    Error,
    sum_check::classic::{ClassicSumCheckProver, ClassicSumCheckRoundMessage, ProverState},
    util::{
        arithmetic::{div_ceil, horner_field_type},
        expression::{CommonPolynomial, Expression, Rotation},
        impl_index,
        parallel::{num_threads, parallelize_iter},
        poly_index_ext, poly_iter_ext,
    },
};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::{FieldType, MultilinearExtension};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{fmt::Debug, iter, ops::AddAssign};
use transcript::Transcript;

macro_rules! zip_self {
    (@ $iter:expr, $step:expr, $skip:expr) => {
        $iter.skip($skip).step_by($step).zip($iter.skip($skip + ($step >> 1)).step_by($step))
    };
    ($iter:expr) => {
        zip_self!(@ $iter, 2, 0)
    };
    ($iter:expr, $step:expr) => {
        zip_self!(@ $iter, $step, 0)
    };
    ($iter:expr, $step:expr, $skip:expr) => {
        zip_self!(@ $iter, $step, $skip)
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct Coefficients<E: ExtensionField>(FieldType<E>);

impl<E: ExtensionField> ClassicSumCheckRoundMessage<E> for Coefficients<E> {
    type Auxiliary = ();

    fn write(&self, transcript: &mut impl Transcript<E>) -> Result<(), Error> {
        match &self.0 {
            FieldType::Ext(coeffs) => transcript.append_field_element_exts(coeffs),
            FieldType::Base(coeffs) => coeffs
                .iter()
                .for_each(|c| transcript.append_field_element(c)),
            FieldType::Unreachable => unreachable!(),
        };
        Ok(())
    }

    fn sum(&self) -> E {
        self[0] + self[..].iter().copied().sum::<E>()
    }

    fn evaluate(&self, _: &Self::Auxiliary, challenge: &E) -> E {
        horner_field_type(&self.0, challenge)
    }
}

impl<'rhs, E: ExtensionField> AddAssign<&'rhs E> for Coefficients<E> {
    fn add_assign(&mut self, rhs: &'rhs E) {
        match &mut self.0 {
            FieldType::Ext(coeffs) => coeffs[0] += *rhs,
            FieldType::Base(_) => panic!("Cannot add extension element to base coefficients"),
            FieldType::Unreachable => unreachable!(),
        }
    }
}

impl<'rhs, E: ExtensionField> AddAssign<(&'rhs E, &'rhs Coefficients<E>)> for Coefficients<E> {
    fn add_assign(&mut self, (scalar, rhs): (&'rhs E, &'rhs Coefficients<E>)) {
        match (&mut self.0, &rhs.0) {
            (FieldType::Ext(lhs), FieldType::Ext(rhs)) => {
                if scalar == &E::ONE {
                    lhs.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(lhs, rhs)| *lhs += *rhs)
                } else if scalar != &E::ZERO {
                    lhs.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(lhs, rhs)| *lhs += *scalar * *rhs)
                }
            }
            _ => panic!("Cannot add base coefficients to extension coefficients"),
        }
    }
}

impl_index!(Coefficients, 0);

/// A CoefficientsProver is represented as a polynomial of the form c + sum_i c_i poly_i, where
/// poly_i are represented as product of polynomial expressions.
#[derive(Clone, Debug)]
pub struct CoefficientsProver<E: ExtensionField>(E, Vec<(E, Vec<Expression<E>>)>);

impl<E: ExtensionField> CoefficientsProver<E> {
    fn evals(&self, state: &ProverState<E>) -> Vec<E> {
        let mut result = vec![self.0; 1 << state.num_vars];
        // Next, for every product of polynomials, where each product is assumed to be exactly 2
        // put this into h(X).
        if self.1.iter().all(|(_, products)| products.len() == 2) {
            for (scalar, products) in self.1.iter() {
                let [lhs, rhs] = [0, 1].map(|idx| &products[idx]);
                match (lhs, rhs) {
                    (
                        Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)),
                        Expression::Polynomial(query),
                    )
                    | (
                        Expression::Polynomial(query),
                        Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)),
                    ) if query.rotation() == Rotation::cur() => {
                        let lhs = &state.eq_xys[*idx];
                        let rhs = &state.polys[query.poly()][state.num_vars];
                        assert_eq!(lhs.num_vars, rhs.num_vars);
                        result.iter_mut().enumerate().for_each(|(i, v)| {
                            *v += poly_index_ext(lhs, i % lhs.evaluations.len())
                                * poly_index_ext(rhs, i % rhs.evaluations.len())
                                * *scalar;
                        })
                    }
                    _ => unimplemented!(),
                }
            }
        } else {
            unimplemented!()
        }
        result
    }
}

impl<E: ExtensionField> ClassicSumCheckProver<E> for CoefficientsProver<E> {
    type RoundMessage = Coefficients<E>;

    fn new(state: &ProverState<E>) -> Self {
        let (constant, flattened) = state.expression.evaluate(
            &|constant| (constant, vec![]),
            &|poly| {
                (
                    E::ZERO,
                    vec![(E::ONE, vec![Expression::CommonPolynomial(poly)])],
                )
            },
            &|query| (E::ZERO, vec![(E::ONE, vec![Expression::Polynomial(query)])]),
            &|challenge| (state.challenges[challenge], vec![]),
            &|(constant, mut products)| {
                products.iter_mut().for_each(|(scalar, _)| {
                    *scalar = -*scalar;
                });
                (-constant, products)
            },
            &|(lhs_constant, mut lhs_products), (rhs_constant, rhs_products)| {
                lhs_products.extend(rhs_products);
                (lhs_constant + rhs_constant, lhs_products)
            },
            &|(lhs_constant, lhs_products), (rhs_constant, rhs_products)| {
                let mut outputs =
                    Vec::with_capacity((lhs_products.len() + 1) * (rhs_products.len() + 1));
                for (constant, products) in
                    [(lhs_constant, &rhs_products), (rhs_constant, &lhs_products)]
                {
                    if constant != E::ZERO {
                        outputs.extend(
                            products
                                .iter()
                                .map(|(scalar, polys)| (constant * *scalar, polys.clone())),
                        )
                    }
                }
                for ((lhs_scalar, lhs_polys), (rhs_scalar, rhs_polys)) in
                    lhs_products.iter().cartesian_product(rhs_products.iter())
                {
                    outputs.push((
                        *lhs_scalar * *rhs_scalar,
                        iter::empty()
                            .chain(lhs_polys)
                            .chain(rhs_polys)
                            .cloned()
                            .collect_vec(),
                    ));
                }
                (lhs_constant * rhs_constant, outputs)
            },
            &|(constant, mut products), rhs| {
                products.iter_mut().for_each(|(lhs, _)| {
                    *lhs *= rhs;
                });
                (constant * rhs, products)
            },
        );
        Self(constant, flattened)
    }

    fn prove_round(&self, state: &ProverState<E>) -> Self::RoundMessage {
        // Initialize h(X) to zero
        let mut coeffs = Coefficients(FieldType::Ext(vec![E::ZERO; state.expression.degree() + 1]));
        // First, sum the constant over the hypercube and add to h(X)
        coeffs += &(E::from_canonical_u64(state.size() as u64) * self.0);
        // Next, for every product of polynomials, where each product is assumed to be exactly 2
        // put this into h(X).
        if self.1.iter().all(|(_, products)| products.len() == 2) {
            for (scalar, products) in self.1.iter() {
                let [lhs, rhs] = [0, 1].map(|idx| &products[idx]);
                if cfg!(feature = "sanity-check") {
                    // When LAZY = false, coeffs[1] will also be computed during the process
                    coeffs += (scalar, &self.karatsuba::<false>(state, lhs, rhs));
                } else {
                    coeffs += (scalar, &self.karatsuba::<true>(state, lhs, rhs));
                }
            }
            if cfg!(feature = "sanity-check") {
                assert_eq!(coeffs[0].double() + coeffs[1] + coeffs[2], state.sum);
            } else {
                coeffs[1] = state.sum - coeffs[0].double() - coeffs[2];
                assert_eq!(coeffs[0].double() + coeffs[1] + coeffs[2], state.sum);
            }
        } else {
            unimplemented!()
        }
        coeffs
    }

    fn sum(&self, state: &ProverState<E>) -> E {
        self.evals(state).iter().copied().sum()
    }
}

impl<E: ExtensionField> CoefficientsProver<E> {
    /// Given two polynomials, represented as polynomial expressions, compute the coefficients
    /// of their product, with certain variables fixed and other variables summed according to
    /// the state.
    fn karatsuba<const LAZY: bool>(
        &self,
        state: &ProverState<E>,
        lhs: &Expression<E>,
        rhs: &Expression<E>,
    ) -> Coefficients<E> {
        let mut coeffs = [E::ZERO; 3];
        // let mut new_coeffs = [E::ZERO; 3];
        match (lhs, rhs) {
            (
                Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)),
                Expression::Polynomial(query),
            )
            | (
                Expression::Polynomial(query),
                Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)),
            ) if query.rotation() == Rotation::cur() => {
                let lhs = &state.eq_xys[*idx];
                let rhs = &state.polys[query.poly()][state.num_vars];

                // lhs and rhs are guaranteed to have the same number of variables and are both
                // multilinear. However, their number of variables may be smaller than the total
                // number of variables of this sum-check protocol. In that case, simply pretend
                // that the evaluation representations are of the full sizes, by repeating the
                // existing evaluations.

                let poly_len = 1usize << lhs.num_vars();
                if poly_len == 1 {
                    let mul = state.size();
                    let mul_field = E::from_canonical_u64(mul as u64);
                    let lhs_eval = poly_index_ext(lhs, 0);
                    let rhs_eval = poly_index_ext(rhs, 0);
                    coeffs[0] = lhs_eval * rhs_eval * mul_field;
                } else {
                    let (poly_size, multiple) = if state.size() < poly_len || state.size() == 1 {
                        (state.size(), 1)
                    } else if state.size() == poly_len {
                        let shifted = poly_len >> 1;
                        (shifted, 2)
                    } else {
                        let shifted = poly_len >> 1;
                        let multiple = if shifted != 0 {
                            state.size() / shifted
                        } else {
                            1
                        };
                        (shifted, multiple)
                    };

                    let evaluate_serial = |coeffs: &mut [E; 3], start: usize, n: usize| {
                        zip_self!(poly_iter_ext(lhs), 2, start * 2)
                            .zip(zip_self!(poly_iter_ext(rhs), 2, start * 2))
                            .take(n)
                            .for_each(|((lhs_0, lhs_1), (rhs_0, rhs_1))| {
                                let coeff_0 = lhs_0 * rhs_0;
                                let coeff_2 = (lhs_1 - lhs_0) * (rhs_1 - rhs_0);
                                coeffs[0] += coeff_0;
                                coeffs[2] += coeff_2;
                                if !LAZY {
                                    coeffs[1] += lhs_1 * rhs_1 - coeff_0 - coeff_2;
                                }
                            });
                    };

                    let num_threads = num_threads();
                    let multiple_field = E::from_canonical_u64(multiple as u64);
                    if poly_size < num_threads {
                        evaluate_serial(&mut coeffs, 0, poly_size);
                        if multiple != 1 {
                            coeffs[0] *= multiple_field;
                            coeffs[2] *= multiple_field;
                            if !LAZY {
                                coeffs[1] *= multiple_field;
                            }
                        }
                    } else {
                        let chunk_size = div_ceil(poly_size, num_threads);
                        let mut partials = vec![[E::ZERO; 3]; num_threads];
                        parallelize_iter(
                            partials.iter_mut().zip((0..).step_by(chunk_size)),
                            |(partial, start)| {
                                // It is possible that the previous chunks already covers all
                                // the positions
                                if poly_size > start {
                                    let chunk_size = chunk_size.min(poly_size - start);
                                    evaluate_serial(partial, start, chunk_size);
                                }
                            },
                        );
                        partials.iter().for_each(|partial| {
                            coeffs[0] += partial[0];
                            coeffs[2] += partial[2];
                            if !LAZY {
                                coeffs[1] += partial[1];
                            }
                        });
                        if multiple != 1 {
                            coeffs[0] *= multiple_field;
                            coeffs[2] *= multiple_field;
                            if !LAZY {
                                coeffs[1] *= multiple_field;
                            }
                        }
                    };
                }
            }
            _ => unimplemented!(),
        }
        Coefficients(FieldType::Ext(coeffs.to_vec()))
    }
}
