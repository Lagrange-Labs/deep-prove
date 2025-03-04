//! Module that takes care of (re)quantizing

use ff::Field;
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use goldilocks::SmallField;
use itertools::Itertools;

use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

/// The type of integer we do arithmetics over. Note this is NOT the type we run the inference over actually
/// We run it over `[Element]` since we need to handle overflows. But all the quantization and requantization
/// is done over this QuantInteger.
pub type QuantInteger = i8;
pub const BIT_LEN: usize = QuantInteger::BITS as usize;
pub const MAX: QuantInteger = QuantInteger::MAX;
pub const MIN: QuantInteger = QuantInteger::MIN;
pub const ZERO: QuantInteger = 0;

/// Trait used to quantize original floating point number to integer
pub trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        // even tho we are requantizing starting from Element, we only want to requantize for QuantInteger
        // the reason we have these two types is to handle overflow
        let max = QuantInteger::MAX as Element;
        let min = QuantInteger::MIN as Element;
        // (a -b) / 2^Q
        let scale = (1.0 - (-1.0)) / (max - min) as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e as f64 / scale).round() as Element + zero_point;
        scaled as Element
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

impl<F: ExtensionField> Fieldizer<F> for QuantInteger {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number

            -F::from(self.unsigned_abs() as u64)
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

/// QuantRange is an intermediary struct to compute the final quantization information.
/// This struct is gonna be useful down the road to handle more precise requantization techniques.
/// BIT_LEN is the target bit length we want to reduce any number to
#[derive(Clone, Debug)]
struct QuantRange<const BIT_LEN: usize> {
    // a - b: at the beginning a power of two to simplify requantization
    pub(crate) max_range: usize,
}

impl<const BIT_LEN: usize> Default for QuantRange<BIT_LEN> {
    fn default() -> Self {
        QuantRange { max_range: 2 }
    }
}

impl<const BIT_LEN: usize> QuantRange<BIT_LEN> {
    /// Computes the quantization info that a matrix x vec will produce
    /// self should be the quant info of the matrix
    /// The quantization info is depending on the number of columns in the matrix
    /// NOTE: this is assuming the vector has the same quantization factor as the matrix coeff
    ///       and it assumes these are the default range.
    /// NOTE2: It is using the simplfiication of finding the max range which is a power of two
    /// so we only need to "right shift" during requant
    fn compute_matvec_quant(m: &crate::tensor::Tensor<Element>) -> Requant {
        let nrows = m.nrows_2d();
        let ncols = m.ncols_2d();
        let max_output_range = m
            .get_data()
            .iter()
            .chunks(ncols)
            .into_iter()
            .map(|row| {
                let row_range = row
                    .map(|weight| {
                        let min = if weight.is_negative() {
                            weight * MAX as Element
                        } else {
                            weight * MIN as Element
                        };
                        let max = if weight.is_negative() {
                            weight * MIN as Element
                        } else {
                            weight * MAX as Element
                        };
                        (min, max)
                    })
                    .fold((0, 0), |(min, max), (wmin, wmax)| (min + wmin, max + wmax));
                // weight * MIN can be positive and higher then MAX*weight if weight's negative
                // so we take the absolute value of the difference
                (row_range.1 - row_range.0).unsigned_abs() as usize
            })
            .max()
            .expect("No max range found")
            .next_power_of_two();
        // trace!("max_output_range: {} - ilog2: {}", max_output_range, max_output_range.ilog2());
        let shift = max_output_range.ilog2() as usize - BIT_LEN;
        Requant {
            range: max_output_range,
            right_shift: shift,
            after_range: 1 << BIT_LEN,
        }
    }
    /// Computes the right shift required to perform after multiplying two numbers
    /// Here `output` should be the range that the value is in AFTER requantizing
    fn mult_shift(&self, rhs: &QuantRange<BIT_LEN>, output: &QuantRange<BIT_LEN>) -> usize {
        assert!(self.max_range.is_power_of_two());
        assert!(rhs.max_range.is_power_of_two());
        assert!(output.max_range.is_power_of_two());
        let slog = self.max_range.ilog2() as usize;
        let rlog = rhs.max_range.ilog2() as usize;
        let olog = output.max_range.ilog2() as usize;
        olog + BIT_LEN - slog - rlog
    }
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
    pub fn from_matrix_default(m: &crate::tensor::Tensor<Element>) -> Self {
        QuantRange::<BIT_LEN>::compute_matvec_quant(m)
    }

    pub fn op(&self, input: &crate::tensor::Tensor<Element>) -> crate::tensor::Tensor<Element> {
        crate::tensor::Tensor::<Element>::new(
            input.dims(),
            input.get_data().iter().map(|e| self.apply(e)).collect_vec(),
        )
    }

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
