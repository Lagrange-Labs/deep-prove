//! Module that takes care of (re)quantizing
use derive_more::From;
use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::Itertools;
use once_cell::sync::Lazy;
use std::{env};

use crate::{Element, tensor::Tensor};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: Lazy<usize> = Lazy::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

/// symmetric quantization range
pub static MIN: Lazy<Element> = Lazy::new(|| -(1 << (*BIT_LEN - 1)) + 1 );
pub static MAX: Lazy<Element> = Lazy::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static ZERO: Lazy<Element> = Lazy::new(|| 0);

/// Symmetric quantization scaling
#[derive(Debug, Clone, From, Copy)]
pub struct ScalingFactor {
    abs_max: f32,
}

trait MinMax {
    fn zero() -> Self;
    fn absolute_value(&self) -> Self;
    fn cmp_min(&self, other: Self) -> Self;
    fn cmp_max(&self, other: Self) -> Self;
    fn to_f32(&self) -> f32;
}

impl MinMax for f32 {
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn zero() -> Self {
        0.0
    }  
    fn cmp_min(&self, other: Self) -> Self {
        self.min(other)
    }
    fn cmp_max(&self, other: Self) -> Self {
        self.max(other)
    }
    fn to_f32(&self) -> f32 {
        *self
    }
}

impl MinMax for Element {
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn cmp_min(&self, other: Self) -> Self {
        std::cmp::min(*self, other)
    }
    fn cmp_max(&self, other: Self) -> Self {
        std::cmp::max(*self, other)
    }
    fn zero() -> Self {
        0
    }
    fn to_f32(&self) -> f32 {
        *self as f32
    }
}
impl ScalingFactor {
    pub fn from_tensor<T: MinMax>(t: &Tensor<T>) -> Self {
        let max_abs = t.get_data().iter().fold(T::zero(), |a, b| a.cmp_max(b.absolute_value()));
        Self { abs_max: max_abs.to_f32() }
    }

    pub fn from_span(min: f32, max: f32) -> Self {
        let abs_max = min.abs().max(max.abs());
        Self { abs_max }
    }

    fn scale(&self) -> f32 {
        (self.abs_max - ( - self.abs_max)) / (*MAX - *MIN) as f32
    }

    /// Derives the right shift to apply to values to requantize them
    /// S_i = (a_i - b_i) / 2^Q = 2^k_i / 2^Q
    /// So S1 * S2 / S3 = 2^k1 * 2^k2 * 2^Q / [2^k3 * 2^Q * 2^Q]
    ///                 = 2^(k1 + k2 - k3 - Q)
    ///                 = 2^{-n} where n = k3 + Q - k1 - k2
    /// n is the number of bits to shift right
    pub fn shift(&self, s2: Self, s3: Self) -> usize {
        let full = self.scale() * s2.scale() / s3.scale();
        assert!(
            full >= 0.0 && full <= 1.0,
            "Full is not in the range [0, 1]. This should not happen."
        );
        let exp = (-full.log2()).ceil() as usize;
        exp
    }

    /// Take a floating point number and quantize it to an BIT_LEN-bit integer
    pub fn quantize(&self, value: &f32) -> Element {
        assert!(
            *value >= -1.0 && *value <= 1.0,
            "Input value must be between -1.0 and 1.0"
        );
        let zero_point = 0;

        // formula is q = round(r/S) + z
        //let scaled =((value.clamp(self.min,self.max) - self.min) / self.scale()).round() * self.scale() + self.min;
        let scaled = (*value / self.scale()).round() as Element + zero_point;

        scaled.clamp(*MIN, *MAX)
    }
}

impl Default for ScalingFactor {
    fn default() -> Self {
        Self::from_span(-1.0, 1.0)
    }
}

pub(crate) trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl<F: ExtensionField> Fieldizer<F> for Element {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from(<F::BaseField as SmallField>::MODULUS_U64 - self.unsigned_abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}
pub(crate) trait IntoElement {
    fn into_element(&self) -> Element;
}

impl<F: ExtensionField> IntoElement for F {
    fn into_element(&self) -> Element {
        let e = self.to_canonical_u64_vec()[0] as Element;
        let modulus_half = <F::BaseField as SmallField>::MODULUS_U64 >> 1;
        // That means he's a positive number
        if *self == F::ZERO {
            0
        // we dont assume any bounds on the field elements, requant might happen at a later stage
        // so we assume the worst case
        } else if e <= modulus_half as Element {
            e
        } else {
            // That means he's a negative number - so take the diff with the modulus and recenter around 0
            let diff = <F::BaseField as SmallField>::MODULUS_U64 - e as u64;
            -(diff as Element)
        }
    }
}

impl<F: ExtensionField> Fieldizer<F> for u8 {
    fn to_field(&self) -> F {
        F::from(*self as u64)
    }
}

pub trait TensorFielder<F> {
    fn to_fields(self) -> Tensor<F>;
}

impl<F: ExtensionField, T> TensorFielder<F> for Tensor<T>
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Tensor<F> {
        Tensor::new(
            self.get_shape(),
            self.get_data()
                .into_iter()
                .map(|i| i.to_field())
                .collect_vec(),
        )
    }
}

pub fn range_from_weight(weight: &Element) -> (Element, Element) {
    let min = if weight.is_negative() {
        weight * *MAX as Element
    } else {
        weight * *MIN as Element
    };
    let max = if weight.is_negative() {
        weight * *MIN as Element
    } else {
        weight * *MAX as Element
    };
    (min, max)
}

#[cfg(test)]
mod test {
    use crate::quantization::Fieldizer;

    use crate::Element;
    type F = goldilocks::GoldilocksExt2;

    #[test]
    fn test_wrapped_field() {
        // for case in vec![-12,25,i8::MIN,i8::MAX] {
        //     let a: i8 = case;
        //     let af: F= a.to_field();
        //     let f = af.to_canonical_u64_vec()[0];
        //     let exp = if a.is_negative() {
        //         MODULUS - (a as i64).unsigned_abs()
        //     } else {
        //         a as u64
        //     };
        //     assert_eq!(f,exp);
        // }
    }

    #[test]
    fn test_wrapped_arithmetic() {
        #[derive(Clone, Debug)]
        struct TestCase {
            a: Element,
            b: Element,
            res: Element,
        }

        let cases = vec![
            TestCase {
                a: -53,
                b: 10,
                res: -53 * 10,
            },
            TestCase {
                a: -45,
                b: -56,
                res: 45 * 56,
            },
        ];
        for (i, case) in cases.iter().enumerate() {
            // cast them to handle overflow
            let ap: F = case.a.to_field();
            let bp: F = case.b.to_field();
            let res = ap * bp;
            let expected = case.res.to_field();
            assert_eq!(res, expected, "test case {}: {:?}", i, case);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type F = goldilocks::GoldilocksExt2;
    #[test]
    fn test_element_field_roundtrip() {
        // Also test a few specific values explicitly
        let test_values = [*MIN, -100, -50, -1, 0, 1, 50, 100, *MAX];
        for &val in &test_values {
            let field_val: F = val.to_field();
            let roundtrip = field_val.into_element();

            assert_eq!(
                val, roundtrip,
                "Element {} did not roundtrip correctly (got {})",
                val, roundtrip
            );
        }
    }

    #[test]
    fn test_scaling_factor() {
        let s1 = ScalingFactor::from_span(0.8, -0.2);
        let s2 = ScalingFactor::from_span(2.0, -2.0);
        let s3 = ScalingFactor::from_span(128.0, -45.0);
        s1.shift(s2, s3);
    }
}
