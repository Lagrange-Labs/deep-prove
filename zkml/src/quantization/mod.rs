//! Module that takes care of (re)quantizing
mod metadata;
mod strategy;
use derive_more::From;
use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::Itertools;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::env;
use tracing::warn;

use crate::{
    Element,
    tensor::{Number, Tensor},
};
pub use metadata::ModelMetadata;
pub use strategy::{AbsoluteMax, InferenceObserver, ScalingStrategy};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: Lazy<usize> = Lazy::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

/// symmetric quantization range
pub static MIN: Lazy<Element> = Lazy::new(|| -(1 << (*BIT_LEN - 1)) + 1);
pub static MAX: Lazy<Element> = Lazy::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static RANGE: Lazy<Element> = Lazy::new(|| *MAX - *MIN);
pub static ZERO: Lazy<Element> = Lazy::new(|| 0);
pub const MIN_FLOAT: f32 = -1.0;
pub const MAX_FLOAT: f32 = 1.0;

/// Symmetric quantization scaling
/// go from float [-a;a] to int [-2^BIT_LEN;2^BIT_LEN]
/// S = (a - (-a)) / (2^{BIT_LEN-1}- (-2^{BIT_LEN-1})) = 2a / 2^BIT_LEN
#[derive(Debug, Clone, From, Copy, Serialize, Deserialize)]
pub struct ScalingFactor {
    min: f32,
    max: f32,
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
        Self {
            min: min,
            max: max,
            quantized_domain: quantized_domain.unwrap_or((*MIN, *MAX)),
        }
    }
    // Initialize a scaling factor in such a way that `self.scale()` is equal to the `scale` value
    // provided as input.
    pub(crate) fn from_scale(scale: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        let (min_quantized, max_quantized) = quantized_domain.clone().unwrap_or((*MIN, *MAX));
        let max = scale / 2.0 * (max_quantized - min_quantized) as f32;
        Self::from_absolute_max(max, quantized_domain)
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn max(&self) -> f32 {
        self.max
    }

    pub fn scale(&self) -> f32 {
        (self.max - self.min) / (self.quantized_domain.1 - self.quantized_domain.0) as f32
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
        Self {
            min: -1.0,
            max: 1.0,
            quantized_domain: (*MIN, *MAX),
        }
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
    use ark_std::{
        Zero,
        rand::{Rng, thread_rng},
    };
    use ff::Field;
    use ff_ext::ExtensionField;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};
    use transcript::Transcript;

    use crate::{
        commit::{compute_beta_eval_poly, compute_betas_eval},
        default_transcript,
        quantization::{Fieldizer, IntoElement},
    };

    use crate::{Element, Tensor, tensor::Number};

    use super::{MAX, MIN};
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
    fn test_quant_idea() {
        for bit_size in 8..11 {
            test_idea_helper(bit_size);
        }
    }

    fn test_idea_helper(bit_size: usize) {
        let mut rel_error_accum = 0.0f32;
        let mut full_error = 0.0f32;
        let mut classic_rel_error_accum = 0.0f32;
        let mut classic_full_error = 0.0f32;

        let dim = 1000usize;
        let shape = vec![dim, dim];

        let tensor_float = Tensor::<f32>::random(shape.clone());
        let max = tensor_float.max_value();
        let min = tensor_float.min_value();

        let weight_scale = (max - min) / 254.0f32;

        let max_row_sum = tensor_float
            .get_data()
            .chunks(dim)
            .map(|chunk| chunk.iter().map(|val| val.abs()).sum::<f32>())
            .fold(0.0f32, |max, x| max.cmp_max(&x.absolute_value()));

        let alpha = max_row_sum / (weight_scale * (1 << (2 * bit_size)) as f32);
        let tester = thread_rng().gen_range((0..1i128 << (2 * bit_size)));

        let alpha_log = alpha.log2();
        let fract_part = alpha_log.fract();
        let int_part = alpha_log.trunc().abs() as usize;

        let shifted = tester >> int_part;
        let outbit = shifted as f32 * 2.0f32.powf(fract_part);

        let fixed_point = 2.0f32.powf(fract_part) * (1u64 << 32) as f32;
        let traditional = tester as f32 * alpha;
        println!("shift then mul: {}, trad: {}", outbit, traditional);
        println!(
            "fixed point: {}, trad shifted: {}",
            fixed_point * shifted as f32,
            traditional * (1u64 << 32) as f32
        );
        let quant_mat_data = tensor_float
            .get_data()
            .iter()
            .map(|val| (val / weight_scale).round() as Element)
            .collect::<Vec<Element>>();

        let quant_mat = Tensor::<Element>::new(vec![dim, dim], quant_mat_data);
        for _ in 0..100 {
            let float_vec = Tensor::<f32>::random(vec![dim]);
            let max_vec_f = float_vec.max_abs_output();

            let source_scale = (2.0f32 * max_vec_f) / 254.0f32;

            let quant_vec_data = float_vec
                .get_data()
                .iter()
                .map(|val| {
                    let out = (val / alpha).round() as Element;
                    // println!("quantised input vec elem: {}", out);
                    out
                })
                .collect::<Vec<Element>>();
            let quant_vec = Tensor::<Element>::new(vec![dim], quant_vec_data);
            let output = quant_mat.matvec(&quant_vec);
            let float_output = tensor_float.matvec(&float_vec);
            let (num, denom) = output
                .get_data()
                .iter()
                .zip(float_output.get_data().iter())
                .fold((0.0f32, 0.0f32), |(num_acc, denom_acc), (v, f)| {
                    assert!(v.unsigned_abs() <= (1 << (2 * bit_size)));
                    let dequant = (*v as f32) * alpha * weight_scale;
                    (num_acc + (*f - dequant) * (*f - dequant), denom_acc + f * f)
                });

            let num_sqrt = num.sqrt();
            let denom_sqrt = denom.sqrt();

            let rel_error = num_sqrt / denom_sqrt;

            rel_error_accum += rel_error;
            full_error += num_sqrt;

            let quant_vec_data = float_vec
                .get_data()
                .iter()
                .map(|val| {
                    let out = (val / source_scale).round() as Element;
                    // println!("quantised input vec elem: {}", out);
                    out
                })
                .collect::<Vec<Element>>();
            let quant_vec = Tensor::<Element>::new(vec![dim], quant_vec_data);

            let output = quant_mat.matvec(&quant_vec);

            let new_scale = weight_scale * source_scale;

            let float_output = tensor_float.matvec(&float_vec);
            let (num, denom) = output
                .get_data()
                .iter()
                .zip(float_output.get_data().iter())
                .fold((0.0f32, 0.0f32), |(num_acc, denom_acc), (v, f)| {
                    // println!("quantised output: {}", v);
                    // println!("dequantised output: {}", (*v as f32) * alpha * weight_scale);
                    // println!("float output: {}", f);

                    let dequant = (*v as f32) * new_scale;
                    (num_acc + (*f - dequant) * (*f - dequant), denom_acc + f * f)
                });

            let num_sqrt = num.sqrt();
            let denom_sqrt = denom.sqrt();

            let rel_error = num_sqrt / denom_sqrt;

            classic_rel_error_accum += rel_error;
            classic_full_error += num_sqrt;
        }
        let average_rel_error = rel_error_accum / 25.;
        let average_full_error = full_error / 25.;
        let classic_average_rel_error = classic_rel_error_accum / 25.;
        let classic_average_full_error = classic_full_error / 25.;
        println!("input bit length: {}", bit_size);
        println!(
            "Average relative error: {}%, log relative error: {}",
            average_rel_error * 100.,
            average_rel_error.log2()
        );
        println!("classic average actual error was: {}", average_full_error);
        println!(
            "classic Average relative error: {}%, log relative error: {}",
            classic_average_rel_error * 100.,
            classic_average_rel_error.log2()
        );
        println!(
            "classic average actual error was: {}",
            classic_average_full_error
        );
        println!(
            "diff between classic and new: {}",
            (classic_average_rel_error - average_rel_error) * 100.
        );
    }

    #[test]
    fn test_decomp_sumcheck() {
        decomp_sumcheck_helper::<GoldilocksExt2>();
    }

    fn decomp_sumcheck_helper<E: ExtensionField>() {
        let dim = 32usize;
        let shape = vec![dim, dim];

        let tensor_float = Tensor::<f32>::random(shape.clone());

        let quant_tensor_data = tensor_float
            .get_data()
            .iter()
            .map(|val| (val * 127.0f32).round() as Element)
            .collect::<Vec<Element>>();

        let bit_size = 7;
        let mut sign_bits = vec![];
        let mut bit_vecs = vec![vec![]; 7];

        quant_tensor_data.iter().for_each(|elem| {
            if elem.is_positive() || elem.is_zero() {
                sign_bits.push(0i128);
            } else {
                sign_bits.push(1i128);
            }
            let mut val = elem.absolute_value();
            (0..bit_size).for_each(|i| {
                bit_vecs[i].push(val & 1);
                val >>= 1;
            });
        });

        for (index, (orig, sign)) in quant_tensor_data.iter().zip(sign_bits.iter()).enumerate() {
            let (accum, _) = bit_vecs.iter().fold((0i128, 1i128), |(acc, pow_2), bv| {
                (acc + bv[index] * pow_2, pow_2 * 2)
            });

            let calc = (1 - 2 * sign) * accum;

            assert_eq!(*orig, calc);
        }

        let quant_tensor_mle = quant_tensor_data
            .iter()
            .map(|val| {
                let field: E = val.to_field();
                field.as_bases()[0]
            })
            .collect::<Vec<E::BaseField>>()
            .into_mle();

        let mut transcript = default_transcript::<E>();

        let r_point = (0..quant_tensor_mle.num_vars())
            .map(|_| {
                let chal = transcript.get_and_append_challenge(b"point");
                chal.elements
            })
            .collect::<Vec<E>>();

        let og_eval = quant_tensor_mle.evaluate(&r_point);
        println!("og eval: {:?}", og_eval);

        let eq_poly: ArcMultilinearExtension<E> = compute_betas_eval(&r_point).into_mle().into();

        let alpha_chal = transcript.get_and_append_challenge(b"batching").elements;

        let mut vp = VirtualPolynomial::<E>::new(r_point.len());

        let field_bit_vecs = bit_vecs
            .iter()
            .enumerate()
            .map(|(i, bits)| {
                let field_bits: ArcMultilinearExtension<E> = bits
                    .iter()
                    .map(|val| {
                        let f: E = val.to_field();
                        f.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>()
                    .into_mle()
                    .into();
                let coeff = E::from(1u64 << i);
                vp.add_mle_list(vec![field_bits.clone()], coeff);
                field_bits
            })
            .collect::<Vec<_>>();

        let sign_mle: ArcMultilinearExtension<E> = sign_bits
            .iter()
            .map(|b| {
                let f: E = (1 - 2 * b).to_field();
                f.as_bases()[0]
            })
            .collect::<Vec<E::BaseField>>()
            .into_mle()
            .into();

        vp.mul_by_mle(sign_mle.clone(), E::BaseField::ONE);

        let mut batch_chal = alpha_chal;

        let one_mle: ArcMultilinearExtension<E> =
            vec![E::BaseField::ONE; dim * dim].into_mle().into();

        vp.add_mle_list(vec![one_mle.clone()], batch_chal);

        vp.add_mle_list(vec![sign_mle.clone(), sign_mle.clone()], -batch_chal);
        batch_chal *= alpha_chal;

        field_bit_vecs.iter().for_each(|mle| {
            vp.add_mle_list(vec![mle.clone()], batch_chal);
            vp.add_mle_list(vec![mle.clone(), mle.clone()], -batch_chal);
            batch_chal *= alpha_chal;
        });

        vp.mul_by_mle(eq_poly.clone(), E::BaseField::ONE);

        let aux = vp.aux_info.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, &mut transcript);

        let evals = state.get_mle_final_evaluations();

        let mut verifier_transcript = default_transcript::<E>();

        let r_point_ver = (0..quant_tensor_mle.num_vars())
            .map(|_| {
                let chal = verifier_transcript.get_and_append_challenge(b"point");
                chal.elements
            })
            .collect::<Vec<E>>();

        let alpha_chal_ver = verifier_transcript
            .get_and_append_challenge(b"batching")
            .elements;

        let initial_eval = quant_tensor_mle.evaluate(&r_point_ver);
        println!("initial eval: {:?}", initial_eval);
        let verifier_claim =
            IOPVerifierState::<E>::verify(initial_eval, &proof, &aux, &mut verifier_transcript);

        let v_point = verifier_claim.point_flat();

        let eq_eval_ver = eq_poly.evaluate(&v_point);

        let bit_evals = &evals[..evals.len() - 3];
        println!("bit evals len: {}", bit_evals.len());
        let sign_eval = evals[evals.len() - 3];

        let field_two = E::from(2);
        let decomp_eval = bit_evals
            .iter()
            .rev()
            .skip(1)
            .fold(bit_evals[bit_evals.len() - 1], |acc, e| {
                field_two * acc + *e
            });
        let first_part = decomp_eval * sign_eval * eq_eval_ver;

        let sign_part = alpha_chal_ver * (E::ONE - sign_eval * sign_eval) * eq_eval_ver;
        let batch_chal_ver = alpha_chal_ver * alpha_chal_ver * eq_eval_ver;

        let (boolean_check, _) =
            bit_evals
                .iter()
                .fold((E::ZERO, batch_chal_ver), |(acc, chal_acc), &be| {
                    let step_eval = chal_acc * (be - be * be);
                    (acc + step_eval, chal_acc * alpha_chal_ver)
                });

        let full_claim = first_part + sign_part + boolean_check;

        assert_eq!(verifier_claim.expected_evaluation, full_claim);
    }

    #[test]
    fn test_elem_decomp() {
        let a = 1000000000i128;
        let b = -a;

        let (a_bits, b_bits): (Vec<i128>, Vec<i128>) = (0..128)
            .map(|i| {
                let a_shift = a >> i;
                let b_shift = b >> i;
                (a_shift & 1, b_shift & 1)
            })
            .unzip();

        for (i, (a_bit, b_bit)) in a_bits.into_iter().zip(b_bits.into_iter()).enumerate() {
            println!("Bit number: {i}, a: {a_bit}, b: {b_bit}");
        }
        let mut rng = thread_rng();
        let rand_float: f32 = rng.gen_range(0.0..1.0f32);

        let b_float_mul = (b as f32) * rand_float;

        let log_f = rand_float.log2();
        let shift = log_f.trunc().abs() as usize;
        let mult = 2.0f32.powf(log_f.fract());

        let fpm = mult * (1u64 << 25) as f32;

        let fpm_res = (-(1i128 << (shift + 25 - 1)) + b * fpm as i128) >> (shift + 25);

        println!("fpm res: {}", fpm_res);
        println!("rounded float res: {}", b_float_mul.round());
        println!("truncated float res: {}", b_float_mul.trunc());

        let elem_b = 14i128;
        let div = elem_b / -3i128;
        println!("div: {}", div);
    }
}
