use crate::{
    Claim, Prover, ScalingFactor,
    commit::{compute_betas_eval, same_poly},
    iop::verifier::Verifier,
    layers::LayerProof,
    lookup::logup_gkr::{prover::batch_prove as logup_batch_prove, verifier::verify_logup_proof},
    quantization::{self, IntoElement},
    tensor::Tensor,
};
use anyhow::{Result, anyhow, ensure};
use ark_std::Zero;
use ff::Field;
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;

use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial, eq_eval},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use statrs::statistics::{Data, Distribution};
use std::ops::{Add, Mul, Sub};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::{debug, warn};
use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::ContextAux,
    lookup::{context::TableType, logup_gkr::structs::LogUpProof},
    quantization::Fieldizer,
};

use super::LayerCtx;

enum RequantResult {
    Ok(Element),
    OutOfRange(Element),
}
/// Information about a requantization step:
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
pub struct FullRequant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    pub fixed_point_multiplier: Element,
    /// TEST ONLY: this can be given to simulate a perfect requantization during inference. Note that it CAN NOT
    /// be proven currently.
    pub multiplier: f32,
    /// This field represents how many bits the max absoloute value can be
    intermediate_bit_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FullRequantProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// the actual sumcheck proof proving the requant
    pub(crate) first_sumcheck: IOPProof<E>,
    /// the actual sumcheck proof proving the requant
    pub(crate) second_sumcheck: IOPProof<E>,
    /// The claimed bit vectors of the less than check
    pub(crate) less_than_vals: Vec<E>,
    /// The output bit mle evals
    pub(crate) output_bit_mle_evals: Vec<E>,
    /// The evals of the bit vecs
    pub(crate) bit_mle_evals: Vec<E>,
}

impl FullRequant {
    pub fn from_scaling_factors(
        input_scale: ScalingFactor,
        weights_scale: ScalingFactor,
        output_scale: ScalingFactor,
        intermediate_bit_size: usize,
    ) -> FullRequant {
        let m = input_scale.m(&weights_scale, &output_scale);
        let log_m = m.log2();
        // This is the right shift
        let int_part = log_m.trunc().abs() as usize;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fixed_point_multiplier = (epsilon * (1u64 << 25) as f32).round() as Element;

        FullRequant {
            right_shift: int_part,
            fixed_point_multiplier,
            multiplier: m,
            intermediate_bit_size,
        }
    }

    fn shift(&self) -> usize {
        self.right_shift + 25
    }

    fn apply(&self, elem: &Element) -> Element {
        let rounding = 1i128 << (self.shift() - 1);
        let unclamped = (rounding + elem * self.fixed_point_multiplier) >> self.shift();
        let sign = if unclamped.is_positive() || unclamped.is_zero() {
            1i128
        } else {
            -1i128
        };

        let clamped = if unclamped.abs() >= *quantization::MAX {
            *quantization::MAX * sign
        } else {
            unclamped
        };

        clamped
    }

    pub fn op(
        &self,
        input: &crate::tensor::Tensor<Element>,
    ) -> Result<crate::tensor::Tensor<Element>> {
        let res = input
            .get_data()
            .iter()
            .map(|e| self.apply(e))
            .collect::<Vec<Element>>();

        Ok(crate::tensor::Tensor::<Element>::new(
            input.get_shape(),
            res,
        ))
    }

    pub(crate) fn prove_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: &Claim<E>,
        input: &Tensor<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // First we make the sign MLE and multiply by the fixed point multiplier
        // This value is used to round correctly
        let rounding_constant = 1i128 << (self.shift() - 1);
        let max_bits = self.intermediate_bit_size + 25;
        let bits_after_shift = max_bits - self.shift();

        let less_than_const = 1i128 << bits_after_shift;

        let (scaled_vals, top_parts): (Vec<Element>, Vec<Element>) = input
            .get_data()
            .iter()
            .map(|field_elem| {
                let element: Element = field_elem.into_element();

                // We scale the value and add the fixed constant that corrects for rounding
                let scaled_val = rounding_constant + element * self.fixed_point_multiplier;
                let tmp = scaled_val >> self.shift();
                let top_part = if tmp.is_positive() || tmp.is_zero() {
                    less_than_const + (*quantization::MAX - tmp)
                } else {
                    less_than_const + (*quantization::MAX + tmp)
                };

                (scaled_val, top_part)
            })
            .unzip();

        // If the value represented by bits after shift is greater than quantization::MAX or smaller than quantization::MIN we need to clamp.
        // To do this we need to build a `clamp_selector` which is `0` when we should clamp and `1` otherwise
        //
        // We calculate this by first working out 2^bits_after_shift +(quantization::MAX + (2 * bit_vecs[last] - 1) * recomb(bit_vecs[bits_after_shift..]))
        // Which will be a value with top bit equal to `1` if no clamping is required and `0` if clampin is required
        //
        // We then bit decompose this value and constrain that all the decomposed mles are boolean valued via batched zero-check

        // Now we decompose scaled_vals into bits
        let bit_mles = (0..self.intermediate_bit_size + 25)
            .into_par_iter()
            .map(|j| {
                let mle: ArcMultilinearExtension<E> = scaled_vals
                    .iter()
                    .map(|val| {
                        let bit: E = ((val >> j) & 1).to_field();
                        bit.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>()
                    .into_mle()
                    .into();
                mle
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        let first_sumcheck_decomp = (0..bits_after_shift + 1)
            .into_par_iter()
            .map(|j| {
                let mle: ArcMultilinearExtension<E> = top_parts
                    .iter()
                    .map(|val| {
                        let bit: E = ((val >> j) & 1).to_field();
                        bit.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>()
                    .into_mle()
                    .into();
                mle
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();
        // Now we assemble the (rather large) virtual polynomial
        let mut vp = VirtualPolynomial::<E>::new(bit_mles[0].num_vars());

        // Squeeze a challenge for batching purposes
        let alpha_chal = prover
            .transcript
            .get_and_append_challenge(b"requant_batch_challenge")
            .elements;

        let mut batch_chal = alpha_chal;

        // These terms are used to prove that every element of each `bit_mle` is either 0 or 1
        first_sumcheck_decomp.iter().for_each(|bm| {
            vp.add_mle_list(vec![bm.clone()], batch_chal);
            vp.add_mle_list(vec![bm.clone(), bm.clone()], -batch_chal);
            batch_chal *= alpha_chal;
        });

        // For the output we want to use the polys from `self.shift()..self.shift() + BIT_SIZE - 1`
        let output_calc_mles = &bit_mles[self.shift()..];
        // This mle tells us whether to clamp or not (if it has value 1 we should clamp, else we shouldn't)
        let clamping_mle = first_sumcheck_decomp.last().unwrap();

        // The output claim should be equal to `eq_poly * (clamping_mle * (SUM 2^{i} * output_calc_mles[i]) + *quantization::MAX * (1 - clamping_mle) * (1 - 2 * msb_mle))`
        output_calc_mles
            .iter()
            .enumerate()
            .take(output_calc_mles.len() - 1)
            .for_each(|(i, output_term)| {
                let pow_two = E::from(1u64 << i);
                vp.add_mle_list(vec![output_term.clone(), clamping_mle.clone()], pow_two);
            });

        let msb_mle = &output_calc_mles[output_calc_mles.len() - 1];
        let pow_two = E::from(1u64 << (output_calc_mles.len() - 1));
        vp.add_mle_list(vec![msb_mle.clone(), clamping_mle.clone()], -pow_two);

        let clamping_coeff: E = (*quantization::MAX).to_field();
        let const_mle: ArcMultilinearExtension<E> =
            vec![clamping_coeff; 1 << bit_mles[0].num_vars()]
                .into_mle()
                .into();
        vp.add_mle_list(vec![const_mle], E::ONE);
        vp.add_mle_list(
            vec![clamping_mle.clone(), msb_mle.clone()],
            clamping_coeff * E::from(2u64),
        );
        vp.add_mle_list(vec![clamping_mle.clone()], -clamping_coeff);
        vp.add_mle_list(vec![msb_mle.clone()], -clamping_coeff * E::from(2u64));

        let eq_poly: ArcMultilinearExtension<E> =
            compute_betas_eval(&last_claim.point).into_mle().into();

        vp.mul_by_mle(eq_poly, E::BaseField::ONE);

        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        let all_mle_evals = state.get_mle_final_evaluations();
        let first_sumcheck_decomp_evals = all_mle_evals[..first_sumcheck_decomp.len()].to_vec();
        let output_bit_mle_evals = all_mle_evals
            [first_sumcheck_decomp.len()..first_sumcheck_decomp.len() + bits_after_shift]
            .to_vec();

        let first_sumcheck_point = &proof.point;
        let second_eq_poly: ArcMultilinearExtension<E> =
            compute_betas_eval(first_sumcheck_point).into_mle().into();

        // Now we assemble the (rather large) virtual polynomial
        let mut vp = VirtualPolynomial::<E>::new(bit_mles[0].num_vars());

        // Squeeze a challenge for batching purposes
        let alpha_chal = prover
            .transcript
            .get_and_append_challenge(b"requant_batch_challenge")
            .elements;

        let mut batch_chal = alpha_chal;

        // These terms are used to prove that every element of each `bit_mle` is either 0 or 1
        bit_mles.iter().for_each(|bm| {
            vp.add_mle_list(vec![bm.clone()], batch_chal);
            vp.add_mle_list(vec![bm.clone(), bm.clone()], -batch_chal);
            batch_chal *= alpha_chal;
        });

        // For the output we want to use the polys from `self.shift()..self.shift() + BIT_SIZE - 1`
        let output_calc_mles = &bit_mles[self.shift()..];

        // This is to relate evals from previous sumcheck to this one
        output_calc_mles.iter().for_each(|bm| {
            vp.add_mle_list(vec![bm.clone()], batch_chal);
            batch_chal *= alpha_chal;
        });

        let msb_mle = bit_mles.last().unwrap();

        // We calculate this by first working out 2^bits_after_shift +(quantization::MAX + (2 * bit_vecs[last] - 1) * recomb(bit_vecs[bits_after_shift..]))
        let field_two = E::from(2u64);
        output_calc_mles
            .iter()
            .enumerate()
            .take(output_calc_mles.len() - 1)
            .for_each(|(i, mle)| {
                let pow_two = E::from(1u64 << i);
                vp.add_mle_list(vec![mle.clone(), msb_mle.clone()], pow_two * field_two);
                vp.add_mle_list(vec![mle.clone()], -pow_two);
            });

        let pow_two = E::from(1u64 << (output_calc_mles.len() - 1));
        vp.add_mle_list(vec![msb_mle.clone(), msb_mle.clone()], -pow_two * field_two);
        vp.add_mle_list(vec![msb_mle.clone()], pow_two);

        let ltc_field: E = less_than_const.to_field();
        let quant_max_field: E = (*quantization::MAX).to_field();

        let const_poly: ArcMultilinearExtension<E> =
            vec![ltc_field + quant_max_field; 1 << bit_mles[0].num_vars()]
                .into_mle()
                .into();
        vp.add_mle_list(vec![const_poly], E::ONE);

        vp.mul_by_mle(second_eq_poly, E::BaseField::ONE);

        #[allow(deprecated)]
        let (proof_two, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        let all_mle_evals = state.get_mle_final_evaluations();
        let bit_mle_evals = all_mle_evals[..bit_mles.len()].to_vec();

        let rounding_const_field: E = rounding_constant.to_field();

        let (non_sign_part, last_pow_two) = bit_mle_evals
            .iter()
            .take(bit_mle_evals.len() - 1)
            .fold((E::ZERO, E::ONE), |(acc, pow_two_acc), &eval| {
                (acc + eval * pow_two_acc, pow_two_acc * field_two)
            });
        let with_sign = non_sign_part
            - last_pow_two * bit_mle_evals[bit_mle_evals.len() - 1]
            - rounding_const_field;

        let fpm: E = self.fixed_point_multiplier.to_field();
        let fpm_inverse = fpm.invert().unwrap();

        let claim_eval = fpm_inverse * with_sign;

        let input_claim = Claim {
            point: proof_two.point.clone(),
            eval: claim_eval,
        };

        prover.push_proof(LayerProof::FullRequant(FullRequantProof {
            first_sumcheck: proof,
            second_sumcheck: proof_two,
            less_than_vals: first_sumcheck_decomp_evals,
            output_bit_mle_evals,
            bit_mle_evals,
        }));

        Ok(input_claim)
    }

    pub(crate) fn verify_full_requant<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &FullRequantProof<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // Squeeze the batching challenge from the transcript
        let alpha_chal = verifier
            .transcript
            .get_and_append_challenge(b"requant_batch_challenge")
            .elements;

        let num_vars = last_claim.point.len();

        let aux_info =
            VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![num_vars, num_vars, num_vars]]);

        let first_subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.first_sumcheck,
            &aux_info,
            verifier.transcript,
        );

        // These terms are used to prove that every element of each `bit_mle` is either 0 or 1
        let (first_booleanity, _) = proof
            .less_than_vals
            .iter()
            .fold((E::ZERO, alpha_chal), |(acc, chal_acc), &bm| {
                (acc + chal_acc * (bm - bm * bm), chal_acc * alpha_chal)
            });

        // The output claim should be equal to `eq_poly * ((1 - clamping_mle) * (SUM 2^{i} * output_calc_mles[i]) + *quantization::MAX * clamping_mle)`
        let clamping_eval = *proof.less_than_vals.last().unwrap();
        let field_two = E::from(2u64);
        let (no_sign, final_pow_two) = proof
            .output_bit_mle_evals
            .iter()
            .take(proof.output_bit_mle_evals.len() - 1)
            .fold((E::ZERO, E::ONE), |(acc, pow_two_acc), &output_term| {
                (
                    acc + pow_two_acc * (output_term * clamping_eval),
                    pow_two_acc * field_two,
                )
            });

        let msb_mle = proof.output_bit_mle_evals[proof.output_bit_mle_evals.len() - 1];

        let with_sign = no_sign - final_pow_two * (msb_mle * clamping_eval);

        let clamping_coeff: E = (*quantization::MAX).to_field();

        let clamping_term =
            clamping_coeff * (E::ONE - clamping_eval) * (E::ONE - field_two * msb_mle);

        let eq_poly_eval: E = eq_eval(&last_claim.point, &first_subclaim.point_flat());

        let full_claim_calc = (clamping_term + with_sign + first_booleanity) * eq_poly_eval;

        ensure!(
            full_claim_calc == first_subclaim.expected_evaluation,
            "First subclaim evaluation wasn't what was expected, calculated: {:?}, expected: {:?}",
            full_claim_calc,
            first_subclaim.expected_evaluation
        );

        // Now we calculate the initial evaluation for the second sumcheck
        let (initial_eval_first_term, _) = proof
            .less_than_vals
            .iter()
            .fold((E::ZERO, E::ONE), |(acc, pow_two_acc), &eval| {
                (acc + pow_two_acc * eval, pow_two_acc * field_two)
            });
        // Squeeze the batching challenge from the transcript
        let alpha_chal = verifier
            .transcript
            .get_and_append_challenge(b"requant_batch_challenge")
            .elements;

        let (second_booleanity_check, challenge) = proof
            .bit_mle_evals
            .iter()
            .fold((E::ZERO, alpha_chal), |(acc, chal_acc), &eval| {
                (acc + chal_acc * (eval - eval * eval), chal_acc * alpha_chal)
            });

        let (initial_eval, output_eval_part, _) = proof
            .output_bit_mle_evals
            .iter()
            .zip(proof.bit_mle_evals[self.shift()..].iter())
            .fold(
                (initial_eval_first_term, E::ZERO, challenge),
                |(acc, mid_acc, chal_acc), (&eval, &out_eval)| {
                    (
                        acc + chal_acc * eval,
                        mid_acc + chal_acc * out_eval,
                        chal_acc * alpha_chal,
                    )
                },
            );

        let second_subclaim = IOPVerifierState::<E>::verify(
            initial_eval,
            &proof.second_sumcheck,
            &aux_info,
            verifier.transcript,
        );

        let msb_eval = *proof.bit_mle_evals.last().unwrap();

        // We calculate this by first working out 2^bits_after_shift +(quantization::MAX + (2 * bit_vecs[last] - 1) * recomb(bit_vecs[bits_after_shift..]))
        let (no_sign_accum, pow_two_last) = proof.bit_mle_evals[self.shift()..]
            .iter()
            .take(proof.output_bit_mle_evals.len() - 1)
            .fold((E::ZERO, E::ONE), |(acc, pow_two_acc), &eval| {
                (acc + pow_two_acc * eval, pow_two_acc * field_two)
            });
        let full_no_const =
            (no_sign_accum - (msb_eval * pow_two_last)) * (field_two * msb_eval - E::ONE);

        let max_bits = self.intermediate_bit_size + 25;
        let bits_after_shift = max_bits - self.shift();

        let less_than_const = 1i128 << bits_after_shift;

        let ltc_field: E = less_than_const.to_field();
        let quant_max_field: E = (*quantization::MAX).to_field();

        let const_term = quant_max_field + ltc_field;

        let second_eq_eval = eq_eval(&proof.first_sumcheck.point, &proof.second_sumcheck.point);
        let full_second_eval =
            (const_term + full_no_const + output_eval_part + second_booleanity_check)
                * second_eq_eval;

        ensure!(
            full_second_eval == second_subclaim.expected_evaluation,
            "Second subclaim evaluation wasn't what was expected, calculated: {:?}, expected: {:?}",
            full_second_eval,
            second_subclaim.expected_evaluation
        );

        let (sum_no_sign, final_pow_two) = proof
            .bit_mle_evals
            .iter()
            .take(proof.bit_mle_evals.len() - 1)
            .fold((E::ZERO, E::ONE), |(acc, pow_two_acc), &eval| {
                (acc + pow_two_acc * eval, pow_two_acc * field_two)
            });
        let full_sum = sum_no_sign - final_pow_two * msb_eval;
        let rounding_const_field: E = (1i128 << (self.shift() - 1)).to_field();
        let fpm_field: E = self.fixed_point_multiplier.to_field();
        let fpm_inverse = fpm_field.invert().unwrap();

        let next_claim_eval = (full_sum - rounding_const_field) * fpm_inverse;

        let next_claim = Claim::<E> {
            point: proof.second_sumcheck.point.clone(),
            eval: next_claim_eval,
        };
        Ok(next_claim)
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        _id: PolyID,
        aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        (LayerCtx::<E>::FullRequant(*self), aux)
    }
}

/// Information about a requantization step:
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
pub struct Requant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    // this is the range we expect the values to be in pre shift
    // This is a magnitude: e.g. [-4;8] gives range = 12.
    // This is to make sure to offset the values to be positive integers before doing the shift
    // That info is used to construct a lookup table for the requantization so the size of the lookup table
    // is directly correlated to the range of the input data.
    pub range: usize,
    /// The range we want the values to be in post requantizing
    pub after_range: usize,
    /// TEST ONLY: this can be given to simulate a perfect requantization during inference. Note that it CAN NOT
    /// be proven currently.
    pub multiplier: Option<f32>,
}

/// Info related to the lookup protocol necessary to requantize
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequantCtx {
    pub requant: Requant,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RequantProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from activation + claim from lookup for the same poly
    /// e.g. the "link" between an activation and requant layer
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the requantization
    pub(crate) lookup: LogUpProof<E>,
}
impl Requant {
    pub fn new(min_value: usize, right_shift: usize) -> Self {
        Self {
            right_shift,
            range: min_value,
            after_range: *quantization::RANGE as usize,
            multiplier: None,
        }
    }

    pub fn set_test_multiplier(&mut self, multiplier: f32) {
        self.multiplier = Some(multiplier);
    }
    pub fn op(
        &self,
        input: &crate::tensor::Tensor<Element>,
    ) -> Result<crate::tensor::Tensor<Element>> {
        let mut not_ok_count = 0;
        let res = input
            .get_data()
            .iter()
            .map(|e| match self.apply(e) {
                RequantResult::Ok(res) => res,
                RequantResult::OutOfRange(res) => {
                    not_ok_count += 1;
                    res
                }
            })
            .collect_vec();
        let d = Data::new(res.iter().map(|e| *e as f64).collect_vec());
        // Debug information to uncomment when debugging scaling factor. Sometimes the right shift is too high
        // and we can observe values being null'd, e.g. set to 0 very quickly. Which messes up the distribution and
        // thus the inference.
        let stats = (d.mean().unwrap(), d.variance().unwrap());
        debug!(
            "AFTER REQUANT: shift {} : {:.2} % OUT OF RANGE (over total {})-> stats mean {:?} var {:?} \n\t->{:?}\n\t->{:?}",
            self.right_shift,
            not_ok_count as f32 / res.len() as f32 * 100.0,
            res.len(),
            stats.0,
            stats.1,
            &input.get_data()[..10.min(input.get_data().len())],
            &res[..10.min(res.len())],
        );
        // ensure!(
        //    not_ok_count == 0,
        //    "Requantization led to out of range values"
        //);
        Ok(crate::tensor::Tensor::<Element>::new(
            input.get_shape(),
            res,
        ))
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.tables.insert(TableType::Range);
        (
            LayerCtx::Requant(RequantCtx {
                requant: *self,
                poly_id: id,
                num_vars: aux
                    .last_output_shape
                    .iter()
                    .map(|dim| ceil_log2(*dim))
                    .sum::<usize>(),
            }),
            aux,
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
    fn apply(&self, e: &Element) -> RequantResult {
        if let Some(_multiplier) = self.multiplier {
            panic!("this is only for test - disable manually");
            #[allow(unreachable_code)]
            let _res = (*e as f64 * _multiplier as f64).round() as Element;
            if !(_res >= *quantization::MIN && _res <= *quantization::MAX) {
                return RequantResult::OutOfRange(
                    _res.clamp(*quantization::MIN, *quantization::MAX),
                );
            } else {
                return RequantResult::Ok(_res);
            }
        }
        let max_bit = (self.range << 1) as Element;
        let tmp = e + max_bit;
        assert!(
            tmp >= 0,
            "offset is too small: element {} + {} (self.range << 1) = {}",
            e,
            self.range << 1,
            tmp
        );
        let tmp = tmp >> self.right_shift;
        let res = tmp - (max_bit >> self.right_shift);
        if !(res >= *quantization::MIN && res <= *quantization::MAX) {
            warn!("{} is NOT quantized correctly: res {}", e, res);
            // RequantResult::OutOfRange(res.clamp(*quantization::MIN, *quantization::MAX))
            RequantResult::OutOfRange(res)
        } else {
            // warn!("{} is OK quantized correctl: res {}", e, res);
            RequantResult::Ok(res)
        }
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

    pub fn gen_lookup_witness<E: ExtensionField>(
        &self,
        input: &[Element],
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        // We calculate how many chunks we will split each entry of `input` into.
        // Since outputs of a layer are centered around zero (i.e. some are negative) in order for all the shifting
        // and the like to give the correct result we make sure that everything is positive.

        // The number of bits that get "sliced off" is equal to `self.right_shift`, we want to know how many limbs it takes to represent
        // this sliced off chunk in base `self.after_range`. To calculate this we perform ceiling division on `self.right_shift` by
        // `ceil_log2(self.after_range)` and then add one for the column that represents the output we will take to the next layer.
        let num_columns = (self.right_shift - 1) / ceil_log2(self.after_range) + 2;

        let num_vars = ceil_log2(input.len());

        let mut lookups = vec![vec![0i128; 1 << num_vars]; num_columns];
        let mut lookups_field = vec![vec![E::BaseField::ZERO; 1 << num_vars]; num_columns];
        // Bit mask for the bytes
        let bit_mask = self.after_range.next_power_of_two() as i128 - 1;

        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        input.iter().enumerate().for_each(|(index, val)| {
            let pre_shift = val + max_bit as i128;
            let tmp = pre_shift >> self.right_shift;
            let input = tmp - subtract as i128 + (self.after_range as i128 >> 1);
            let in_field: E = input.to_field();

            lookups[0][index] = input;
            lookups_field[0][index] = in_field.as_bases()[0];
            // the value of an input should always be basefield elements

            // This leaves us with only the part that is "discarded"
            let mut remainder_vals = pre_shift - (tmp << self.right_shift);
            lookups
                .iter_mut()
                .zip(lookups_field.iter_mut())
                .skip(1)
                .rev()
                .for_each(|(discarded_lookup_chunk, discarded_field_chunk)| {
                    let chunk = remainder_vals & bit_mask;
                    let value = chunk as i128;
                    let val_field: E = value.to_field();
                    discarded_lookup_chunk[index] = value;
                    discarded_field_chunk[index] = val_field.as_bases()[0];
                    remainder_vals >>= ceil_log2(self.after_range);
                });
            debug_assert_eq!(remainder_vals, 0);
        });

        debug_assert!({
            input.iter().enumerate().fold(true, |acc, (i, value)| {
                let calc_evals = lookups_field
                    .iter()
                    .map(|col| E::from(col[i]))
                    .collect::<Vec<E>>();

                let field_value: E = value.to_field();
                acc & (self.recombine_claims(&calc_evals) == field_value)
            })
        });
        (lookups.concat(), lookups_field)
    }

    /// Function to recombine claims of constituent MLEs into a single value to be used as the initial sumcheck evaluation
    /// of the subsequent proof.
    pub fn recombine_claims<
        E: From<u64> + Default + Add<Output = E> + Mul<Output = E> + Sub<Output = E> + Copy,
    >(
        &self,
        eval_claims: &[E],
    ) -> E {
        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        // There may be padding claims so we only take the first `num_columns` claims

        let tmp_eval = E::from(1 << self.right_shift as u64)
            * (eval_claims[0] + E::from(subtract as u64) - E::from(self.after_range as u64 >> 1))
            + eval_claims.iter().skip(1).rev().enumerate().fold(
                E::default(),
                |acc, (i, &claim)| {
                    acc + E::from((self.after_range.next_power_of_two().pow(i as u32)) as u64)
                        * (claim)
                },
            );
        tmp_eval - E::from(max_bit as u64)
    }
    pub(crate) fn prove_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: &Claim<E>,
        output: &[E],
        requant_info: &RequantCtx,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let prover_info = prover.next_lookup_witness()?;

        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;
        // For requant layers we have to extract the correct "chunk" from the list of claims
        let eval_claims = logup_proof
            .output_claims()
            .iter()
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();

        let combined_eval = requant_info.requant.recombine_claims(&eval_claims);

        // Pass the eval associated with the poly used in the activation step to the same poly prover
        let first_claim = logup_proof
            .output_claims()
            .first()
            .ok_or(anyhow!("No claims found"))?;
        let point = first_claim.point.clone();

        let corrected_claim = Claim::<E> {
            point: point.clone(),
            eval: first_claim.eval - E::from((*quantization::RANGE / 2) as u64),
        };
        println!("correct claim eval: {:?}", corrected_claim.eval);
        println!(
            "output eval: {:?}",
            output.to_vec().into_mle().evaluate(&corrected_claim.point)
        );
        // Add the claim used in the activation function
        same_poly_prover.add_claim(corrected_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;

        prover
            .witness_prover
            .add_claim(requant_info.poly_id, claim_acc_proof.extract_claim())?;
        println!("REQUANT: WITNESS Poly ID: {}", requant_info.poly_id);

        prover.push_proof(LayerProof::Requant(RequantProof {
            io_accumulation: claim_acc_proof,
            lookup: logup_proof,
        }));

        Ok(Claim {
            point,
            eval: combined_eval,
        })
    }
}

impl RequantCtx {
    pub(crate) fn verify_requant<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &RequantProof<E>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proof
        let num_instances =
            (self.requant.right_shift - 1) / ceil_log2(self.requant.after_range) + 2;
        let verifier_claims = verify_logup_proof(
            &proof.lookup,
            num_instances,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;

        // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
        let sp_ctx = same_poly::Context::<E>::new(self.num_vars);
        let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
        sp_verifier.add_claim(last_claim)?;

        let first_claim = verifier_claims
            .claims()
            .first()
            .ok_or(anyhow::anyhow!("No claims found"))?;
        let point = first_claim.point.clone();
        // The first claim needs to be shifted down as we add a value to make sure that all its evals are in the range 0..1 << BIT_LEn
        let corrected_claim = Claim::<E>::new(
            point.clone(),
            first_claim.eval - E::from((*quantization::RANGE / 2) as u64),
        );
        sp_verifier.add_claim(corrected_claim)?;

        let new_output_claim = sp_verifier.verify(&proof.io_accumulation, verifier.transcript)?;
        // 3. Accumulate the new claim into the witness commitment protocol
        verifier
            .witness_verifier
            .add_claim(self.poly_id, new_output_claim)?;

        // Here we recombine all of the none dummy polynomials to get the actual claim that should be passed to the next layer
        let eval_claims = verifier_claims
            .claims()
            .iter()
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();
        let eval = self.requant.recombine_claims(&eval_claims);
        // 4. return the input claim for to be proven at subsequent step
        Ok(Claim { point, eval })
    }
}

//#[cfg(test)]
// mod tests {
//    use ark_std::rand::rngs::StdRng;
//
//    use super::*;
//    use crate::quantization::range_from_weight;
//    use crate::tensor::Tensor;
//    use crate::ScalingFactor;
//
//    #[test]
//    fn test_requant_shift_element() {
//        let n = 10;
//        let v1 = Tensor::random_seed(vec![n],Some(15420));
//        let v2= Tensor::random_seed(vec![n],Some(1567892312));
//        assert!(v1.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        assert!(v2.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        let res = v1.mul(&v2);
//        let s1 = ScalingFactor::from_tensor(&v1);
//        let s2 = ScalingFactor::from_tensor(&v2);
//        let s_res = ScalingFactor::from_tensor(&res);
//        println!("v1: {:?}", v1.get_data());
//        println!("v2: {:?}", v2.get_data());
//        println!("res: {:?}", res.get_data());
//        println!("s1: {:?}", s1);
//        println!("s2: {:?}", s2);
//        println!("s_res: {:?}", s_res);
//        let shift = s1.shift(s2, s_res);
//        println!("shift: {:?}", shift);
//        let res_max = res.get_data().iter().max().unwrap();
//        let res_min = res.get_data().iter().min().unwrap();
//        let requant_info = Requant {
//            right_shift:  shift,
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requant = requant_info.op(&res);
//        println!("res_requant: {:?}", res_requant.get_data());
//    }
//
//    use ark_std::rand::SeedableRng;
//    use ark_std::rand::Rng;
//    #[test]
//    fn test_requant_shift_model_like() {
//        let n = 10;
//        let mut rng = StdRng::seed_from_u64(15420);
//        let input_min = -1.0;
//        let input_max = 1.0;
//        println!("1");
//        let s_input = ScalingFactor::from_span(input_min, input_max);
//        let inputf :Vec<f32> = (0..n).map(|_| { rng.gen_range(input_min..=input_max) }).collect_vec();
//        let input: Vec<Element> = inputf.iter().map(|e| s_input.quantize(&e)).collect_vec();
//        let min_f32 = -0.2;
//        let max_f32 = 0.2;
//        println!("2");
//        let s_model = ScalingFactor::from_span(min_f32, max_f32);
//        println!("3");
//        let s_input = ScalingFactor::from_span(input_min, input_max);
//        println!("4");
//        let modelf :Vec<f32> = (0..n).map(|_| { rng.gen_range(min_f32..=max_f32) }).collect_vec();
//        let model :Vec<Element> = modelf.iter().map(|e| s_model.quantize(&e)).collect_vec();
//
//        let inputf = Tensor::new(vec![n], inputf);
//        let modelf  = Tensor::new(vec![n], modelf);
//        println!("5");
//        let resf = inputf.mul(&modelf);
//        println!("6");
//        let s_resf = ScalingFactor::from_tensor(&resf);
//        let s_resft = ScalingFactor::new(resf.get_data().iter().map(|e| e.abs()).fold(0.0f32,|a,b| a.max(b)));
//        println!("7");
//        let input = Tensor::new(vec![n], input);
//        let model= Tensor::new(vec![n], model);
//        assert!(input.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        assert!(model.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        let (mins,maxs) : (Vec<_>,Vec<_>)= model.get_data().iter().map(|e| range_from_weight(e)).unzip();
//        let res_min = mins.iter().min().unwrap();
//        let res_max = maxs.iter().max().unwrap();
//        let s_res = ScalingFactor::from_span(*res_min as f32, *res_max as f32);
//        let res = input.mul(&model);
//        println!("input: {:?}", input.get_data());
//        println!("model: {:?}", model.get_data());
//        println!("res: {:?}", res.get_data());
//        println!("s1: {:?}", s_input);
//        println!("s2: {:?}", s_model);
//        println!("s_resf: {:?}", s_resf);
//        println!("s_res: {:?}", s_res);
//        let shift = s_input.shift(s_model, s_res);
//        let shiftf= s_input.shift(s_model, s_resf);
//        let shiftft = s_input.shift(s_model, s_resft);
//        println!("shift: {:?}", shift);
//        println!("shiftf: {:?}", shiftf);
//        println!("shiftft: {:?}", shiftft);
//        let requant = Requant {
//            right_shift:  shift,
//            // theoretical res_max and res_min at this point ! since we dont know the input when we create requant
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requant = requant.op(&res);
//        let requant = Requant {
//            right_shift:  shiftf,
//            // theoretical res_max and res_min at this point ! since we dont know the input when we create requant
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requantf = requant.op(&res);
//        let requant = Requant {
//            right_shift:  shiftft,
//            // theoretical res_max and res_min at this point ! since we dont know the input when we create requant
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requantft= requant.op(&res);
//        println!("res_requant: {:?}", res_requant.get_data());
//        println!("res_requantf: {:?}", res_requantf.get_data());
//        println!("res_requantft: {:?}", res_requantft.get_data());
//        //assert!(res_requant.get_data().iter().filter(|r| **r == 0 || **r == -1).collect::<Vec<_>>().len() < res_requant.get_data().len());
//    }
//
//}
