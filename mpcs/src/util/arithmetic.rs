use ff::{BatchInvert, Field, PrimeField};

use ff_ext::ExtensionField;
use multilinear_extensions::mle::FieldType;
use num_integer::Integer;
use std::{borrow::Borrow, iter};

mod bh;
mod hypercube;
pub use bh::BooleanHypercube;
pub use bitvec::field::BitField;
pub use hypercube::{
    interpolate_field_type_over_boolean_hypercube, interpolate_over_boolean_hypercube,
};
use num_bigint::BigUint;

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
    coeffs.iter().rev().fold(F::ZERO, |acc, c| acc * x + c)
}

/// Evaluate the given coeffs as a univariate polynomial at x
pub fn horner_base<E: ExtensionField>(coeffs: &[E::BaseField], x: &E) -> E {
    let mut acc = E::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * x + E::from(*c);
    }
    acc
    // 2
    //.fold(F::ZERO, |acc, coeff| acc * x + coeff)
}

pub fn steps<F: Field>(start: F) -> impl Iterator<Item = F> {
    steps_by(start, F::ONE)
}

pub fn steps_by<F: Field>(start: F, step: F) -> impl Iterator<Item = F> {
    iter::successors(Some(start), move |state| Some(step + state))
}

pub fn powers<F: Field>(scalar: F) -> impl Iterator<Item = F> {
    iter::successors(Some(F::ONE), move |power| Some(scalar * power))
}

pub fn squares<F: Field>(scalar: F) -> impl Iterator<Item = F> {
    iter::successors(Some(scalar), move |scalar| Some(scalar.square()))
}

pub fn product<F: Field>(values: impl IntoIterator<Item = impl Borrow<F>>) -> F {
    values
        .into_iter()
        .fold(F::ONE, |acc, value| acc * value.borrow())
}

pub fn sum<F: Field>(values: impl IntoIterator<Item = impl Borrow<F>>) -> F {
    values
        .into_iter()
        .fold(F::ZERO, |acc, value| acc + value.borrow())
}

pub fn inner_product<'a, 'b, F: Field>(
    lhs: impl IntoIterator<Item = &'a F>,
    rhs: impl IntoIterator<Item = &'b F>,
) -> F {
    lhs.into_iter()
        .zip_eq(rhs)
        .map(|(lhs, rhs)| *lhs * rhs)
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
        .map(|((a, b), c)| *a * b * c)
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

pub fn barycentric_weights<F: Field>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|&(i, _point_i)| (i != j))
                .map(|(_i, point_i)| *point_j - point_i)
                .reduce(|acc, value| acc * value)
                .unwrap_or(F::ONE)
        })
        .collect_vec();
    weights.iter_mut().batch_invert();
    weights
}

pub fn barycentric_interpolate<F: Field>(weights: &[F], points: &[F], evals: &[F], x: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *x - point).collect_vec();
        coeffs.iter_mut().batch_invert();
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().fold(F::ZERO, |sum, coeff| sum + coeff);
        (coeffs, sum_inv.invert().unwrap())
    };
    inner_product(&coeffs, evals) * sum_inv
}

pub fn modulus<F: PrimeField>() -> BigUint {
    BigUint::from_bytes_le((-F::ONE).to_repr().as_ref()) + 1u64
}

pub fn fe_from_bool<F: Field>(value: bool) -> F {
    if value { F::ONE } else { F::ZERO }
}

pub fn fe_mod_from_le_bytes<F: PrimeField>(bytes: impl AsRef<[u8]>) -> F {
    fe_from_le_bytes((BigUint::from_bytes_le(bytes.as_ref()) % modulus::<F>()).to_bytes_le())
}

pub fn fe_truncated_from_le_bytes<F: PrimeField>(bytes: impl AsRef<[u8]>, num_bits: usize) -> F {
    let mut big = BigUint::from_bytes_le(bytes.as_ref());
    (num_bits as u64..big.bits()).for_each(|idx| big.set_bit(idx, false));
    fe_from_le_bytes(big.to_bytes_le())
}

pub fn fe_from_le_bytes<F: PrimeField>(bytes: impl AsRef<[u8]>) -> F {
    let bytes = bytes.as_ref();
    let mut repr = F::Repr::default();
    assert!(bytes.len() <= repr.as_ref().len());
    repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
    F::from_repr(repr).unwrap()
}

pub fn fe_to_fe<F1: PrimeField, F2: PrimeField>(fe: F1) -> F2 {
    debug_assert!(BigUint::from_bytes_le(fe.to_repr().as_ref()) < modulus::<F2>());
    let mut repr = F2::Repr::default();
    repr.as_mut().copy_from_slice(fe.to_repr().as_ref());
    F2::from_repr(repr).unwrap()
}

pub fn fe_truncated<F: PrimeField>(fe: F, num_bits: usize) -> F {
    let (num_bytes, num_bits_last_byte) = div_rem(num_bits, 8);
    let mut repr = fe.to_repr();
    repr.as_mut()[num_bytes + 1..].fill(0);
    repr.as_mut()[num_bytes] &= (1 << num_bits_last_byte) - 1;
    F::from_repr(repr).unwrap()
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
    a1 + (x - a0) * (b1 - a1) * (b0 - a0).invert().unwrap()
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
        res += E::BaseField::from(u64::from(*b));
    });
    res
}
