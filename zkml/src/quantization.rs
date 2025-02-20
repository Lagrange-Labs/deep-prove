//! Module that takes care of (re)quantizing 

use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{matrix::Matrix, Element};

/// The type of integer we do arithmetics over. Note this is NOT the type we run the inference over actually
/// We run it over `[Element]` since we need to handle overflows. But all the quantization and requantization
/// is done over this QuantInteger.
pub type QuantInteger = i8;
pub const BIT_LEN :usize = QuantInteger::BITS as usize;
pub const MAX: QuantInteger = QuantInteger::MAX;
pub const MIN: QuantInteger = QuantInteger::MIN;
pub const ZERO: QuantInteger = 0;

/// Trait used to quantize original floating point number to integer
pub(crate) trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
}

pub(crate) trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        // even tho we are requantizing starting from Element, we only want to requantize for QuantInteger
        // the reason we have these two types is to handle overflow
        let max = QuantInteger::MAX;
        // (a -b) / 2^Q
        let scale = (1.0 - (-1.0)) / max as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e as f64 / scale).round() as Element + zero_point;
        scaled as Element
    }
}

impl<F: ExtensionField> Fieldizer<F> for Element {
    fn to_field(&self) -> F {
        // make sure we're in range still
        assert!(*self >= QuantInteger::MIN as Element && *self <= QuantInteger::MAX as Element);
        (*self as QuantInteger).to_field()
    }
}

impl<F: ExtensionField> Fieldizer<F> for QuantInteger {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from(<F::BaseField as SmallField>::MODULUS_U64 - self.abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}

/// QuantRange is an intermediary struct to compute the final quantization information.
/// This struct is gonna be useful down the road to handle more precise requantization techniques.
/// BIT_LEN is the target bit length we want to reduce any number to
#[derive(Clone,Debug)]
struct QuantRange<const BIT_LEN: usize> {
    // a - b: at the beginning a power of two to simplify requantization
    pub(crate) max_range: usize,
}


impl<const BIT_LEN:usize > QuantRange<BIT_LEN> {
    pub fn default() -> Self {
        QuantRange {
            max_range: (2 as usize).pow(BIT_LEN as u32),
        }
    }
    /// Computes the quantization info that a matrix x vec will produce
    /// self should be the quant info of the matrix
    /// The quantization info is depending on the number of columns in the matrix
    /// NOTE: this is assuming the vector has the same quantization factor as the matrix coeff
    ///       and it assumes these are the default range.
    /// NOTE2: It is using the simplfiication of finding the max range which is a power of two
    /// so we only need to "right shift" during requant
    fn compute_matvec_quant(m: &Matrix<Element>) -> Requant {
        // BIT_LEN * 2 because of multiplication
        // log because of additions
        let bit_len = BIT_LEN * 2  + m.ncols().ilog2() as usize;
        let output_range = Self {
            max_range: (2 as usize).pow(bit_len as u32),
        };
        let shift = Self::default().mult_shift(&Self::default(), &output_range);
        Requant {
            range: output_range.max_range,
            right_shift: shift,
        }
    }
    /// Computes the right shift required to perform after multiplying two numbers
    fn mult_shift(&self, rhs: &QuantRange<BIT_LEN>,output: &QuantRange<BIT_LEN>) -> usize {
        assert!(self.max_range.is_power_of_two());
        assert!(rhs.max_range.is_power_of_two());
        assert!(output.max_range.is_power_of_two());
        let slog = self.max_range.ilog2() as usize;
        let rlog = rhs.max_range.ilog2() as usize;
        let olog = output.max_range.ilog2() as usize;
        slog + rlog - olog - BIT_LEN
    }

    
}

/// Information about a requantization step: 
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
pub struct Requant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    pub range: usize,
}

impl Requant {
    pub fn from_matrix_default(m: &Matrix<Element>) -> Self {
        QuantRange::<BIT_LEN>::compute_matvec_quant(m)
    }

    pub fn op(&self, input: &[Element]) -> Vec<Element> {
        input.iter().map(|e| self.apply(e)).collect_vec()
    }

    #[inline(always)]
    pub fn apply(&self, e: &Element) -> Element {
        e >> self.right_shift 
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![1, self.range]
    }

    pub fn write_to_transcript<E: ExtensionField,T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from(self.right_shift as u64));
        t.append_field_element(&E::BaseField::from(self.range as u64));
    }

    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values --> shifted to the right !
    /// TODO: have a "cache" of lookups for similar ranges
    pub fn to_mle<E: ExtensionField>(&self) -> (Vec<E>, Vec<E>) {
        // TODO: make a +1 or -1 somewhere
        let min_range = - (self.range as Element) / 2;
        let max_range = (self.range as Element) / 2;
        (min_range..=max_range)
            .map(|i| {
                let input : E = i.to_field();
                // conversion from QuantInteger -> u64 OK because result is either 0 or strictly positive.
                let output = E::from(self.apply(&i) as u64);
                (input,output)
            })
            .unzip()
    }


}