use ff_ext::ExtensionField;
use multilinear_extensions::mle::FieldType;
use num_integer::Integer;
use p3_field::Field;
use std::{borrow::Borrow, iter};

mod bh;
mod hypercube;
pub use bh::BooleanHypercube;
pub use hypercube::{
    interpolate_field_type_over_boolean_hypercube, interpolate_over_boolean_hypercube,
};
use p3_field::FieldAlgebra;

use itertools::Itertools;

pub fn horner_field_type<E: ExtensionField>(coeffs: &FieldType<E>, x: &E) -> E {
    match coeffs {
        FieldType::Ext(coeffs) => horner(coeffs.as_slice(), x),
        FieldType::Base(coeffs) => horner_base(coeffs.as_slice(), x),
        _ => unreachable!(),
    }
}

/// Evaluate the given coeffs as a univariate polynomial at x
pub fn horner<F: Field>(coeffs: &[F], x: &F) -> F {
    coeffs.iter().rev().fold(F::ZERO, |acc, c| acc * *x + *c)
}

/// Evaluate the given coeffs as a univariate polynomial at x
pub fn horner_base<E: ExtensionField>(coeffs: &[E::BaseField], x: &E) -> E {
    let mut acc = E::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * *x + E::from(*c);
    }
    acc
    // 2
    //.fold(F::ZERO, |acc, coeff| acc * x + coeff)
}

pub fn steps<F: Field>(start: F) -> impl Iterator<Item = F> {
    steps_by(start, F::ONE)
}

pub fn steps_by<F: Field>(start: F, step: F) -> impl Iterator<Item = F> {
    iter::successors(Some(start), move |state| Some(step + *state))
}

pub fn powers<F: Field>(scalar: F) -> impl Iterator<Item = F> {
    iter::successors(Some(F::ONE), move |power| Some(scalar * *power))
}

pub fn squares<F: Field>(scalar: F) -> impl Iterator<Item = F> {
    iter::successors(Some(scalar), move |scalar| Some(scalar.square()))
}

pub fn product<F: Field>(values: impl IntoIterator<Item = impl Borrow<F>>) -> F {
    values
        .into_iter()
        .fold(F::ONE, |acc, value| acc * *value.borrow())
}

pub fn sum<F: Field>(values: impl IntoIterator<Item = impl Borrow<F>>) -> F {
    values
        .into_iter()
        .fold(F::ZERO, |acc, value| acc + *value.borrow())
}

pub fn inner_product<'a, 'b, F: Field>(
    lhs: impl IntoIterator<Item = &'a F>,
    rhs: impl IntoIterator<Item = &'b F>,
) -> F {
    lhs.into_iter()
        .zip_eq(rhs)
        .map(|(lhs, rhs)| *lhs * *rhs)
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

pub fn inner_product_three<'a, 'b, 'c, F: Field>(
    a: impl IntoIterator<Item = &'a F>,
    b: impl IntoIterator<Item = &'b F>,
    c: impl IntoIterator<Item = &'c F>,
) -> F {
    a.into_iter()
        .zip_eq(b)
        .zip_eq(c)
        .map(|((a, b), c)| *a * *b * *c)
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

pub fn fe_from_bool<F: Field>(value: bool) -> F {
    if value { F::ONE } else { F::ZERO }
}

pub fn usize_from_bits_le(bits: &[bool]) -> usize {
    bits.iter()
        .rev()
        .fold(0, |int, bit| (int << 1) + (*bit as usize))
}

pub fn div_rem(dividend: usize, divisor: usize) -> (usize, usize) {
    Integer::div_rem(&dividend, &divisor)
}

pub fn div_ceil(dividend: usize, divisor: usize) -> usize {
    Integer::div_ceil(&dividend, &divisor)
}

#[allow(unused)]
pub fn interpolate2_weights_base<E: ExtensionField>(
    points: [(E, E); 2],
    weight: E::BaseField,
    x: E,
) -> E {
    interpolate2_weights(points, E::from(weight), x)
}

pub fn interpolate2_weights<F: Field>(points: [(F, F); 2], weight: F, x: F) -> F {
    // a0 -> a1
    // b0 -> b1
    // x  -> a1 + (x-a0)*(b1-a1)/(b0-a0)
    let (a0, a1) = points[0];
    let (b0, b1) = points[1];
    if cfg!(feature = "sanity-check") {
        assert_ne!(a0, b0);
        assert_eq!(weight * (b0 - a0), F::ONE);
    }
    // Here weight = 1/(b0-a0). The reason for precomputing it is that inversion is expensive
    a1 + (x - a0) * (b1 - a1) * weight
}

pub fn interpolate2<F: Field>(points: [(F, F); 2], x: F) -> F {
    // a0 -> a1
    // b0 -> b1
    // x  -> a1 + (x-a0)*(b1-a1)/(b0-a0)
    let (a0, a1) = points[0];
    let (b0, b1) = points[1];
    assert_ne!(a0, b0);
    a1 + (x - a0) * (b1 - a1) * (b0 - a0).inverse()
}

pub fn degree_2_zero_plus_one<F: Field>(poly: &[F]) -> F {
    poly[0] + poly[0] + poly[1] + poly[2]
}

pub fn degree_2_eval<F: Field>(poly: &[F], point: F) -> F {
    poly[0] + point * poly[1] + point * point * poly[2]
}

pub fn base_from_raw_bytes<E: ExtensionField>(bytes: &[u8]) -> E::BaseField {
    let mut res = E::BaseField::ZERO;
    bytes.iter().for_each(|b| {
        res += E::BaseField::from_canonical_u8(*b);
    });
    res
}
