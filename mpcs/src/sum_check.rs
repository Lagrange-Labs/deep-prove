use crate::{
    Error,
    util::{
        BitIndex,
        arithmetic::{BooleanHypercube, inner_product, powers, product},
        expression::{CommonPolynomial, Expression, Query},
    },
};
use std::{collections::HashMap, fmt::Debug};

use classic::{ClassicSumCheckRoundMessage, SumcheckProof};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
use p3_field::Field;
use serde::{Serialize, de::DeserializeOwned};
use transcript::Transcript;

pub mod classic;

#[derive(Clone, Debug)]
pub struct VirtualPolynomial<'a, E: ExtensionField> {
    expression: &'a Expression<E>,
    polys: Vec<&'a DenseMultilinearExtension<E>>,
    challenges: &'a [E],
    ys: &'a [Vec<E>],
}

impl<'a, E: ExtensionField> VirtualPolynomial<'a, E> {
    pub fn new(
        expression: &'a Expression<E>,
        polys: impl IntoIterator<Item = &'a DenseMultilinearExtension<E>>,
        challenges: &'a [E],
        ys: &'a [Vec<E>],
    ) -> Self {
        Self {
            expression,
            polys: polys.into_iter().collect(),
            challenges,
            ys,
        }
    }
}

pub type SumCheckProverOutput<E, SC> = (
    Vec<E>,
    Vec<E>,
    SumcheckProof<E, <SC as SumCheck<E>>::RoundMessage>,
);

pub trait SumCheck<E: ExtensionField>: Clone + Debug
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type ProverParam: Clone + Debug;
    type VerifierParam: Clone + Debug;
    type RoundMessage: ClassicSumCheckRoundMessage<E> + Clone + Debug;

    #[allow(clippy::type_complexity)]
    fn prove(
        pp: &Self::ProverParam,
        num_vars: usize,
        virtual_poly: VirtualPolynomial<E>,
        sum: E,
        transcript: &mut impl Transcript<E>,
    ) -> Result<SumCheckProverOutput<E, Self>, Error>;

    fn verify(
        vp: &Self::VerifierParam,
        num_vars: usize,
        degree: usize,
        sum: E,
        proof: &SumcheckProof<E, Self::RoundMessage>,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(E, Vec<E>), Error>;
}

pub fn evaluate<E: ExtensionField>(
    expression: &Expression<E>,
    num_vars: usize,
    evals: &HashMap<Query, E>,
    challenges: &[E],
    ys: &[&[E]],
    x: &[E],
) -> E {
    assert!(num_vars > 0 && expression.max_used_rotation_distance() <= num_vars);
    let identity = identity_eval(x);
    let lagranges = {
        let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
        expression
            .used_langrange()
            .into_iter()
            .map(|i| {
                let b = bh[i.rem_euclid(1 << num_vars as i32) as usize];
                (i, lagrange_eval(x, b))
            })
            .collect::<HashMap<_, _>>()
    };
    let eq_xys = ys.iter().map(|y| eq_xy_eval(x, y)).collect_vec();
    expression.evaluate(
        &|scalar| scalar,
        &|poly| match poly {
            CommonPolynomial::Identity => identity,
            CommonPolynomial::Lagrange(i) => lagranges[&i],
            CommonPolynomial::EqXY(idx) => eq_xys[idx],
        },
        &|query| evals[&query],
        &|idx| challenges[idx],
        &|scalar| -scalar,
        &|lhs, rhs| lhs + rhs,
        &|lhs, rhs| lhs * rhs,
        &|value, scalar| scalar * value,
    )
}

pub fn lagrange_eval<F: Field>(x: &[F], b: usize) -> F {
    assert!(!x.is_empty());

    product(x.iter().enumerate().map(
        |(idx, x_i)| {
            if b.nth_bit(idx) { *x_i } else { F::ONE - *x_i }
        },
    ))
}

pub fn eq_xy_eval<F: Field>(x: &[F], y: &[F]) -> F {
    assert!(!x.is_empty());
    assert_eq!(x.len(), y.len());

    product(
        x.iter()
            .zip(y)
            .map(|(x_i, y_i)| (*x_i * *y_i).double() + F::ONE - *x_i - *y_i),
    )
}

fn identity_eval<F: Field>(x: &[F]) -> F {
    inner_product(
        x,
        &powers(F::from_canonical_u64(2)).take(x.len()).collect_vec(),
    )
}
