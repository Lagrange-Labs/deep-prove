//! Module that takes care of (re)quantizing

use ff::Field;
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use goldilocks::SmallField;
use itertools::Itertools;
use log::debug;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::env;
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: Lazy<usize> = Lazy::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

// These values depend on BIT_LEN and need to be computed at runtime
pub static MIN: Lazy<Element> = Lazy::new(|| -(1 << (*BIT_LEN - 1)));
pub static MAX: Lazy<Element> = Lazy::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static ZERO: Lazy<Element> = Lazy::new(|| 0);

/// Trait used to quantize original floating point number to integer
pub trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
    fn from_f32_unsafe_clamp(e: &f32, max_abs: f64) -> Output;
    fn from_f32_unsafe_range(e: &f32, min_val: f64, max_val: f64) -> Output;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        assert!(
            *e >= -1.0 && *e <= 1.0,
            "Input value must be between -1.0 and 1.0"
        );
        // even tho we are requantizing starting from Element, we only want to requantize for QuantInteger
        // the reason we have these two types is to handle overflow
        // (a -b) / 2^Q
        let scale = (1.0 - (-1.0)) / (1 << *BIT_LEN) as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e as f64 / scale).round() as Element + zero_point;
        scaled as Element
    }

    fn from_f32_unsafe_clamp(e: &f32, max_abs: f64) -> Self {
        let e = *e as f64;
        assert!(
            max_abs > 0.0,
            "max_abs should be greater than zero. Domain range is between [-max_abs, max_abs]."
        );

        let scale = (2.0 * max_abs * 1.05) / (*MAX - *MIN) as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (e / scale).round() as Element + zero_point;
        if scaled < *MIN || scaled > *MAX {
            panic!(
                "Quantization error: Value {} scales to {}, which is outside the valid range [{}, {}].",
                e, scaled, *MIN, *MAX
            );
        }
        let scaled = scaled.clamp(*MIN, *MAX);

        if e < -max_abs || e > max_abs {
            debug!(
                "Quantization: Value {} is out of [-{}, {}]. But quantized to {}.",
                e, max_abs, max_abs, scaled
            );
        }
        scaled as Element
    }

    fn from_f32_unsafe_range(e: &f32, min_val: f64, max_val: f64) -> Self {
        let e = *e as f64;
        assert!(
            min_val < max_val,
            "min_val must be less than max_val. Domain range is between [min_val, max_val]."
        );
    
        // Safety factor for avoiding edge cases
        let safety_factor = 1.05;
        
        // Find the maximum absolute value to ensure zero-centered scaling
        let abs_max = max_val.abs().max(min_val.abs()) * safety_factor;
        
        // This scale maps [-abs_max, abs_max] to [MIN, MAX]
        let scale = (2.0 * abs_max) / (*MAX - *MIN) as f64;
        assert!(scale > 0.0, "Scale must be positive");
        
        // Simple scaling ensures that 0 maps to 0 in the output
        let scaled = (e / scale).round() as Element;
        
        // Bounds check
        if scaled < *MIN || scaled > *MAX {
            panic!(
                "Quantization error: Value {} (from range [{}, {}]) scales to {}, which is outside the valid range [{}, {}].",
                e, min_val, max_val, scaled, *MIN, *MAX
            );
        }
        
        scaled
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
            self.dims(),
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

/// Information about a requantization step:
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub struct Requant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    // this is the range we expect the values to be in pre shift
    pub range: usize,
    /// The range we want the values to be in post requantizing
    pub after_range: usize,
}

impl Requant {
    pub fn op(&self, input: &crate::tensor::Tensor<Element>) -> crate::tensor::Tensor<Element> {
        crate::tensor::Tensor::<Element>::new(
            input.dims(),
            input.get_data().iter().map(|e| self.apply(e)).collect_vec(),
        )
    }

    /// Applies requantization to a single element.
    ///
    /// This function performs the following steps:
    /// 1. Adds a large offset (max_bit) to ensure all values are positive
    /// 2. Right-shifts by the specified amount to reduce the bit width
    /// 3. Subtracts the shifted offset to restore the correct value range
    ///
    /// The result is a value that has been scaled down to fit within the
    /// target bit width while preserving the relative magnitudes.
    #[inline(always)]
    pub fn apply(&self, e: &Element) -> Element {
        let max_bit = (self.range << 1) as i128;
        let tmp = e + max_bit;
        let tmp = tmp >> self.right_shift;
        tmp - (max_bit >> self.right_shift)
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![1, self.range]
    }

    pub fn write_to_transcript<E: ExtensionField, T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from(self.right_shift as u64));
        t.append_field_element(&E::BaseField::from(self.range as u64));
    }

    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values --> shifted to the right !
    /// TODO: have a "cache" of lookups for similar ranges
    pub fn to_mle<E: ExtensionField>(&self) -> Vec<E> {
        // TODO: make a +1 or -1 somewhere
        let min_range = -(self.after_range as Element) / 2;
        let max_range = (self.after_range as Element) / 2 - 1;
        (min_range..=max_range)
            .map(|i| i.to_field())
            .collect::<Vec<E>>()
    }
    /// Function that takes a list of field elements that need to be requantized (i.e. the output of a Dense layer)
    /// and splits each value into the correct decomposition for proving via lookups.
    pub fn prep_for_requantize<E: ExtensionField>(
        &self,
        input: &[Element],
    ) -> Vec<Vec<E::BaseField>> {
        // We calculate how many chunks we will split each entry of `input` into.
        // Since outputs of a layer are centered around zero (i.e. some are negative) in order for all the shifting
        // and the like to give the correct result we make sure that everything is positive.

        // The number of bits that get "sliced off" is equal to `self.right_shift`, we want to know how many limbs it takes to represent
        // this sliced off chunk in base `self.after_range`. To calculate this we perform ceiling division on `self.right_shift` by
        // `ceil_log2(self.after_range)` and then add one for the column that represents the output we will take to the next layer.
        let num_columns = (self.right_shift - 1) / ceil_log2(self.after_range) + 2;

        let num_vars = ceil_log2(input.len());

        let mut mle_evals = vec![vec![E::BaseField::ZERO; 1 << num_vars]; num_columns];

        // Bit mask for the bytes
        let bit_mask = self.after_range as i128 - 1;

        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        input.iter().enumerate().for_each(|(index, val)| {
            let pre_shift = val + max_bit as i128;
            let tmp = pre_shift >> self.right_shift;
            let input = tmp - subtract as i128;
            let input_field: E = input.to_field();

            mle_evals[0][index] = input_field.as_bases()[0];
            // the value of an input should always be basefield elements

            // This leaves us with only the part that is "discarded"
            let mut remainder_vals = pre_shift - (tmp << self.right_shift);
            mle_evals
                .iter_mut()
                .skip(1)
                .rev()
                .for_each(|discarded_chunk| {
                    let chunk = remainder_vals & bit_mask;
                    let value = chunk as i128 - (self.after_range as i128 >> 1);
                    let field_elem: E = value.to_field();
                    discarded_chunk[index] = field_elem.as_bases()[0];
                    remainder_vals >>= self.after_range.ilog2();
                });
            debug_assert_eq!(remainder_vals, 0);
        });

        debug_assert!({
            input.iter().enumerate().fold(true, |acc, (i, value)| {
                let calc_evals = mle_evals
                    .iter()
                    .map(|col| E::from(col[i]))
                    .collect::<Vec<E>>();

                let field_value: E = value.to_field();
                acc & (self.recombine_claims(&calc_evals) == field_value)
            })
        });
        mle_evals
    }

    /// Function to recombine claims of constituent MLEs into a single value to be used as the initial sumcheck evaluation
    /// of the subsequent proof.
    pub fn recombine_claims<E: ExtensionField>(&self, eval_claims: &[E]) -> E {
        // We calculate how many chunks we will split each entry of `input` into.
        // Since outputs of a layer are centered around zero (i.e. some are negative) in order for all the shifting
        // and the like to give the correct result we make sure that everything is positive.

        // The number of bits that get "sliced off" is equal to `self.right_shift`, we want to know how many limbs it takes to represent
        // this sliced off chunk in base `self.after_range`. To calculate this we perform ceiling division on `self.right_shift` by
        // `ceil_log2(self.after_range)` and then add one for the column that represents the output we will take to the next layer.
        let num_columns = (self.right_shift - 1) / ceil_log2(self.after_range) + 2;

        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        // There may be padding claims so we only take the first `num_columns` claims
        let actual_claims = &eval_claims[..num_columns];
        let tmp_eval =
            E::from(1 << self.right_shift as u64) * (actual_claims[0] + E::from(subtract as u64))
                + actual_claims.iter().skip(1).rev().enumerate().fold(
                    E::ZERO,
                    |acc, (i, &claim)| {
                        acc + E::from((self.after_range.pow(i as u32)) as u64)
                            * (claim + E::from(self.after_range as u64 >> 1))
                    },
                );
        tmp_eval - E::from(max_bit as u64)
    }
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
