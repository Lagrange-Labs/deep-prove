//! Module that takes care of (re)quantizing
mod metadata;
mod strategy;
use derive_more::From;
use ff_ext::{ExtensionField, SmallField};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{env, sync::LazyLock};
use tracing::warn;

use crate::{
    Element,
    tensor::{Number, Tensor, TensorSlice, is_close},
};
pub use metadata::ModelMetadata;
pub(crate) use strategy::InferenceTracker;
pub use strategy::{AbsoluteMax, InferenceObserver, ScalingStrategy, ScalingStrategyKind};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: LazyLock<usize> = LazyLock::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

/// symmetric quantization range
pub static MIN: LazyLock<Element> = LazyLock::new(|| -(1 << (*BIT_LEN - 1)) + 1);
pub static MAX: LazyLock<Element> = LazyLock::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static RANGE: LazyLock<Element> = LazyLock::new(|| *MAX - *MIN);
pub static ZERO: LazyLock<Element> = LazyLock::new(|| 0);
pub const MIN_FLOAT: f32 = -1.0;
pub const MAX_FLOAT: f32 = 1.0;
pub const QUANTIZATION_RANGE: std::ops::RangeInclusive<f32> = MIN_FLOAT..=MAX_FLOAT;

/// Symmetric quantization scaling
/// go from float [-a;a] to int [-2^BIT_LEN;2^BIT_LEN]
/// S = (a - (-a)) / (2^{BIT_LEN-1}- (-2^{BIT_LEN-1})) = 2a / 2^BIT_LEN
#[derive(Debug, Clone, From, Copy, Serialize, Deserialize)]
pub struct ScalingFactor {
    min: f32,
    max: f32,
    scale: f32,
    quantized_domain: (Element, Element),
}

impl ScalingFactor {
    pub fn from_absolute_max(abs_max: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        Self::from_span(-(abs_max.abs()), abs_max.abs(), quantized_domain)
    }
    pub fn from_tensor<T: MinMax>(
        t: &Tensor<T>,
        quantized_domain: Option<(Element, Element)>,
    ) -> Self {
        let max_abs = t
            .get_data()
            .iter()
            .fold(T::zero(), |a, b| a.cmp_max(b.absolute_value()));
        Self::from_absolute_max(max_abs.to_f32(), quantized_domain)
    }

    pub fn from_span(min: f32, max: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        let quantized_domain = quantized_domain.unwrap_or((*MIN, *MAX));
        let scale = (max - min) / (quantized_domain.1 - quantized_domain.0) as f32;
        Self {
            min,
            max,
            scale,
            quantized_domain,
        }
    }
    // Initialize a scaling factor in such a way that `self.scale()` is equal to the `scale` value
    // provided as input.
    pub(crate) fn from_scale(scale: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        let (min_quantized, max_quantized) = quantized_domain.unwrap_or((*MIN, *MAX));
        let max = scale / 2.0 * (max_quantized - min_quantized) as f32;
        let min = -(max.abs());
        Self {
            max,
            min,
            scale,
            quantized_domain: (min_quantized, max_quantized),
        }
    }

    /// Create a [`ScalingFactor`] from its constituent parts. Useful for operations like Softmax where its
    /// important to preserve its structure as a probability distribution.
    pub(crate) fn from_parts(
        max: f32,
        min: f32,
        scale: f32,
        quantized_domain: (Element, Element),
    ) -> ScalingFactor {
        Self {
            max,
            min,
            scale,
            quantized_domain,
        }
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn max(&self) -> f32 {
        self.max
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn domain(&self) -> (Element, Element) {
        self.quantized_domain
    }
    /// M = S1 * S2 / S3
    pub fn m(&self, s2: &Self, s3: &Self) -> f32 {
        self.scale() * s2.scale() / s3.scale()
    }

    /// Derives the right shift to apply to values to requantize them
    /// M = S1 * S2 / S3 = 2^-n * eps
    /// n is the number of bits to shift right
    pub fn shift(&self, s2: &Self, s3: &Self) -> usize {
        (-self.m(s2, s3).log2()).ceil() as usize
    }

    /// Take a floating point number and quantize it to an BIT_LEN-bit integer
    /// S = (a - (-a)) / (2^{BIT_LEN-1}- (-2^{BIT_LEN-1})) = 2a / 2^BIT_LEN
    pub fn quantize(&self, value: &f32) -> Element {
        // assert!(
        //    *value >= -1.0 && *value <= 1.0,
        //    "Input value must be between -1.0 and 1.0"
        //);
        let zero_point = 0;

        // formula is q = round(r/S) + z
        // let scaled =((value.clamp(self.min,self.max) - self.min) / self.scale()).round() * self.scale() + self.min;
        let scaled = (*value / self.scale()).round() as Element + zero_point;
        if scaled < self.quantized_domain.0 || scaled > self.quantized_domain.1 {
            warn!(
                "Quantized value {} from {} is out of range [{}, {}]",
                scaled, value, self.quantized_domain.0, self.quantized_domain.1
            );
        }
        scaled.clamp(self.quantized_domain.0, self.quantized_domain.1)
    }

    pub fn dequantize(&self, value: &Element) -> f32 {
        *value as f32 * self.scale()
    }
}

impl Default for ScalingFactor {
    fn default() -> Self {
        let default_scale = 2.0f32 / (*MAX - *MIN) as f32;
        Self {
            min: -1.0,
            max: 1.0,
            scale: default_scale,
            quantized_domain: (*MIN, *MAX),
        }
    }
}

// s = m *  2^-shift, it returns the shift and the multiplier
pub fn split_scale_into_multiplier(s: f32) -> (i32, f32) {
    let log_s = s.log2();
    let (shift, m) = ((-log_s.trunc()), 2.0f32.powf(log_s.fract()));

    assert!(
        is_close(&[m * (2f32.powf(-shift))], &[s]),
        "m * 2^shift != s -> m: {}, s: {}, shift: {}, m * 2^shift: {}",
        m,
        s,
        shift,
        m * 2f32.powf(-shift)
    );
    (shift as i32, m)
}

/// Returns the scaling factors for the main tensor and for the bias tensor. These are the "model" scaling factors, or
/// S2 in the formula S1 * S2 / S3.
pub fn model_scaling_factor_from_tensor_and_bias(
    input: &ScalingFactor,
    main: &Tensor<f32>,
    bias: &Tensor<f32>,
) -> (ScalingFactor, ScalingFactor) {
    let max_weight = main.max_abs_output();
    let max_bias = bias.max_abs_output();
    let main_sf = ScalingFactor::from_absolute_max(max_weight.max(max_bias), None);
    let bias_sf = bias_scaling_matmul(input, &main_sf);
    (main_sf, bias_sf)
}

pub fn bias_scaling_matmul(input: &ScalingFactor, model: &ScalingFactor) -> ScalingFactor {
    let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
    let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
    ScalingFactor::from_scale(
        input.scale() * model.scale(),
        Some((min_quantized, max_quantized)),
    )
}

pub trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl<F: ExtensionField> Fieldizer<F> for Element {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from_canonical_u64(<F::BaseField as SmallField>::MODULUS_U64 - self.unsigned_abs())
        } else {
            // for positive and zero, it's just the number
            F::from_canonical_u64(*self as u64)
        }
    }
}
pub(crate) trait IntoElement {
    fn to_element(&self) -> Element;
}

impl<F: ExtensionField> IntoElement for F {
    fn to_element(&self) -> Element {
        let e = self.to_canonical_u64_vec()[0];
        let modulus_half = <F::BaseField as SmallField>::MODULUS_U64 >> 1;
        // That means he's a positive number
        if *self == F::ZERO {
            0
        // we dont assume any bounds on the field elements, requant might happen at a later stage
        // so we assume the worst case
        } else if e <= modulus_half {
            e as Element
        } else {
            // That means he's a negative number - so take the diff with the modulus and recenter around 0
            let diff = <F::BaseField as SmallField>::MODULUS_U64 - e;
            -(diff as Element)
        }
    }
}

pub trait TensorFielder<F> {
    fn to_fields(self) -> Tensor<F>;
}

impl<F: ExtensionField, T> TensorFielder<F> for &Tensor<T>
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Tensor<F> {
        TensorSlice::from(self).to_fields()
    }
}

impl<'a, F: ExtensionField, T> TensorFielder<F> for &TensorSlice<'a, T>
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Tensor<F> {
        Tensor::new(
            self.get_shape(),
            self.get_data().iter().map(|i| i.to_field()).collect_vec(),
        )
    }
}

pub fn max_range_from_weight<T: Number>(weight: &T, min_input: &T, max_input: &T) -> (T, T) {
    let min = if weight.is_negative() {
        *weight * *max_input
    } else {
        *weight * *min_input
    };
    let max = if weight.is_negative() {
        *weight * *min_input
    } else {
        *weight * *max_input
    };
    (min, max)
}

pub trait MinMax {
    fn zero() -> Self;
    fn absolute_value(&self) -> Self;
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

#[cfg(test)]
mod test {
    use crate::quantization::{Fieldizer, IntoElement, split_scale_into_multiplier};

    use crate::Element;

    use super::{MAX, MIN};
    type F = ff_ext::GoldilocksExt2;

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

    #[test]
    fn test_element_field_roundtrip() {
        // Also test a few specific values explicitly
        let test_values = [*MIN, -100, -50, -1, 0, 1, 50, 100, *MAX];
        for &val in &test_values {
            let field_val: F = val.to_field();
            let roundtrip = field_val.to_element();

            assert_eq!(
                val, roundtrip,
                "Element {} did not roundtrip correctly (got {})",
                val, roundtrip
            );
        }
    }

    #[test]
    fn test_split_scale_into_multiplier() {
        for s in vec![0.125, 0.075] {
            let (shift, m) = split_scale_into_multiplier(s);

            assert!((m * (2f32.powf(-shift as f32) as f32) - s).abs() <= f32::EPSILON);
        }
    }
}
