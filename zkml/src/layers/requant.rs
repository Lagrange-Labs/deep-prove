//! Module containing code for requantising in a provable manner after a layer involving an affine operation of some kind.
//! We have to use fixed point multiplication to prove this requantisation effectively.

use crate::{
    Claim, Prover, ScalingFactor,
    iop::verifier::Verifier,
    layers::LayerProof,
    quantization::{self, IntoElement},
    tensor::Tensor,
};
use anyhow::{Result, ensure};
use ark_std::Zero;

use ff_ext::ExtensionField;

use multilinear_extensions::{
    mle::IntoMLE,
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial, build_eq_x_r, eq_eval},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};

use transcript::Transcript;

use crate::{
    Element, commit::precommit::PolyID, iop::context::ContextAux, quantization::Fieldizer,
};

use super::LayerCtx;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
/// This struct contains the infomation used in requantisation (i.e. rescaling and clamping)
/// The fields are:
/// - `multiplier`: This is the actual [`f32`] value calculated as `S1 * S2 / S3` and in traditional quantisation is what we would multiply by and then round to requantise
/// - `right_shift`: This is `multiplier.log2().trunc().abs()`
/// - `fixed_point_multiplier`: This is `2.0.powf(multiplier.log2().fract()) * (1 << 25)`, 25 is chosen as the [`f32`] mantissa is only 24 bits long so this should retain all bits
/// - `intermediate_bit_size`: This is the maximum number of bits a value can have before its requantised.
pub struct Requant {
    /// After multiplying by `self.fixed_point_multiplier` the value need to be shifted by this plus 25.
    pub right_shift: usize,
    /// The normalised scaling factor represented as a fixed point multiplier (it should have 24 fractional bits)
    pub fixed_point_multiplier: Element,
    /// THe actual multiplier, this is mainly used to compare accuracy, it has no purpose in actual proving
    pub multiplier: f32,
    /// This field represents how many bits the max absoloute value can be
    intermediate_bit_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
/// The infomation needed to verify that rescaling and clamping was performed correctly after a
/// step involving linear algebra (i.e. a fully connected or convolutional layer)
pub struct RequantProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// The first sumcheck proof that takes the output claim and
    /// constructs a claim about whether clamping occured or not.
    pub(crate) first_sumcheck: IOPProof<E>,
    /// The second sumcheck proof that links the clamping claim to a claim about the input tensor.
    pub(crate) second_sumcheck: IOPProof<E>,
    /// The claimed bit vectors of the clamping claim, this is from the first sumcheck
    pub(crate) less_than_vals: Vec<E>,
    /// The output bit mle evals, this is from the first sumcheck
    pub(crate) output_bit_mle_evals: Vec<E>,
    /// The evals of the bit vecs, this is from the second sumcheck
    pub(crate) bit_mle_evals: Vec<E>,
}

impl Requant {
    /// Method used to instantiate a new [`Requant`] from the scaling factors of all tensors involved in a layer.
    /// The `intermediate_bit_size` is layer dependant and so should be passed as input. It can be calculated based on how many times you need to multiply and add
    /// to get each value in the output tensor.
    pub fn from_scaling_factors(
        input_scale: ScalingFactor,
        weights_scale: ScalingFactor,
        output_scale: ScalingFactor,
        intermediate_bit_size: usize,
    ) -> Requant {
        let m = input_scale.m(&weights_scale, &output_scale);
        let log_m = m.log2();
        // This is the right shift
        let int_part = log_m.trunc().abs() as usize;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fixed_point_multiplier = (epsilon * (1u64 << 25) as f32).round() as Element;

        Requant {
            right_shift: int_part,
            fixed_point_multiplier,
            multiplier: m,
            intermediate_bit_size,
        }
    }

    /// This returns the shift (including the part that depends on `S1 * S2/ S3`)
    fn shift(&self) -> usize {
        self.right_shift + 25
    }

    /// Internal method that applies this op to an [`Element`]
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

    /// API for performing this op on a quantised tensor.
    pub fn op(&self, input: &Tensor<Element>) -> Result<Tensor<Element>> {
        let res = input
            .get_data()
            .iter()
            .map(|e| self.apply(e))
            .collect::<Vec<Element>>();

        Ok(Tensor::<Element>::new(input.get_shape(), res))
    }

    /// Logic for proving correct requantisation between an input and output [`Tensor`].
    /// This functions in a GKR like manner over two sumchecks, the first links the claim about the ouput to the bit decomposition
    /// of the values in the input and clamps values if necessary. The second sumcheck verifies that clamping only occured when a value
    /// would overflow the quantised integer bit size.
    #[timed::timed_instrument(level = "info")]
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
        // This value is used to round correctly
        let rounding_constant = 1i128 << (self.shift() - 1);
        // This is the maximum size in bits of any of the values in the input after being multiplied by the fixed point multiplier
        let max_bits = self.intermediate_bit_size + 25;
        // This is the number of bits that will remain after shifting
        let bits_after_shift = max_bits - self.shift();
        // We use this to check if a value needs to be clamped
        let less_than_const = 1i128 << bits_after_shift;

        // Split the input into the pre-shifted part (scaled_vals) and the part used in the clamping check (top_parts)
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

        // Now we decompose scaled_vals and top_parts into bits
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

        // Now we assemble the first (rather large) virtual polynomial
        let eq_poly: ArcMultilinearExtension<E> = build_eq_x_r(&last_claim.point);
        // Squeeze a challenge for batching purposes
        let alpha_chal = prover
            .transcript
            .get_and_append_challenge(b"requant_batch_challenge")
            .elements;

        let clamping_coeff: E = (*quantization::MAX).to_field();

        // These terms are used to prove that we did bit decompose top_parts
        let (first_vp, _) = first_sumcheck_decomp.iter().enumerate().fold(
            (
                VirtualPolynomial::<E>::new(bit_mles[0].num_vars()),
                alpha_chal,
            ),
            |(mut acc, chal_acc), (i, bm)| {
                let first_coeff = if i < first_sumcheck_decomp.len() - 1 {
                    chal_acc
                } else {
                    chal_acc - clamping_coeff
                };
                acc.add_mle_list(vec![eq_poly.clone(), bm.clone()], first_coeff);
                acc.add_mle_list(vec![eq_poly.clone(), bm.clone(), bm.clone()], -chal_acc);
                (acc, chal_acc * alpha_chal)
            },
        );
        // For the output we want to use the polys from `self.shift()` onwards
        let output_calc_mles = &bit_mles[self.shift()..];
        // This mle tells us whether to clamp or not (if it has value 0 we should clamp, else we shouldn't)
        let clamping_mle = first_sumcheck_decomp.last().unwrap();

        // The output claim should be equal to `eq_poly * (clamping_mle * (SUM 2^{i} * output_calc_mles[i] - 2^{top_bit} * msb_mle) + *quantization::MAX * (1 - clamping_mle) * (1 - 2 * msb_mle))`
        // This term has degree 3.
        let (mut vp, final_pow_two) = output_calc_mles
            .iter()
            .take(output_calc_mles.len() - 1)
            .fold((first_vp, 1u64), |(mut acc, pow_two_acc), mle| {
                acc.add_mle_list(
                    vec![eq_poly.clone(), mle.clone(), clamping_mle.clone()],
                    E::from(pow_two_acc),
                );
                (acc, pow_two_acc + pow_two_acc)
            });

        let msb_mle = &output_calc_mles[output_calc_mles.len() - 1];

        vp.add_mle_list(vec![eq_poly.clone()], clamping_coeff);
        vp.add_mle_list(
            vec![eq_poly.clone(), clamping_mle.clone(), msb_mle.clone()],
            clamping_coeff * E::from(2u64) - E::from(final_pow_two),
        );

        vp.add_mle_list(
            vec![eq_poly.clone(), msb_mle.clone()],
            -clamping_coeff * E::from(2u64),
        );

        // Run the first sumcheck proof
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // Extract claims that the verifier will need to calculate the initial input of the next sumcheck GKR style.
        let all_mle_evals = state.get_mle_final_evaluations();
        let first_sumcheck_decomp_evals =
            all_mle_evals[1..1 + first_sumcheck_decomp.len()].to_vec();
        let output_bit_mle_evals = all_mle_evals
            [1 + first_sumcheck_decomp.len()..1 + first_sumcheck_decomp.len() + bits_after_shift]
            .to_vec();

        let first_sumcheck_point = &proof.point;

        let second_eq_poly: ArcMultilinearExtension<E> = build_eq_x_r(first_sumcheck_point);
        // Now we assemble the second (rather large) virtual polynomial

        // Squeeze a challenge for batching purposes
        let alpha_chal = prover
            .transcript
            .get_and_append_challenge(b"requant_batch_challenge")
            .elements;

        // These terms are used to prove that every element of each `bit_mle` is either 0 or 1
        let (first_vp, batch_chal) = bit_mles.iter().fold(
            (
                VirtualPolynomial::<E>::new(bit_mles[0].num_vars()),
                alpha_chal,
            ),
            |(mut acc, chal_acc), bm| {
                acc.add_mle_list(vec![second_eq_poly.clone(), bm.clone()], chal_acc);
                acc.add_mle_list(
                    vec![second_eq_poly.clone(), bm.clone(), bm.clone()],
                    -chal_acc,
                );
                (acc, chal_acc * alpha_chal)
            },
        );

        // For the output we want to use the polys from `self.shift()..self.shift() + BIT_SIZE - 1`
        let output_calc_mles = &bit_mles[self.shift()..];

        // The unwrap here is safe because by construction `bit_mles` will always contain values.
        let msb_mle = bit_mles.last().unwrap();

        // This is to relate evals from previous sumcheck to this one
        let (mut vp, final_batch_chal, final_pow_two) = output_calc_mles
            .iter()
            .take(output_calc_mles.len() - 1)
            .fold(
                (first_vp, batch_chal, E::ONE),
                |(mut acc, chal_acc, pow_two_acc), bm| {
                    acc.add_mle_list(
                        vec![second_eq_poly.clone(), bm.clone()],
                        chal_acc - pow_two_acc,
                    );
                    acc.add_mle_list(
                        vec![second_eq_poly.clone(), bm.clone(), msb_mle.clone()],
                        pow_two_acc + pow_two_acc,
                    );

                    (acc, chal_acc * alpha_chal, pow_two_acc + pow_two_acc)
                },
            );

        // We calculate this by first working out 2^bits_after_shift + (*quantization::MAX + (2 *msb_mle - 1) * (SUM 2^{i} * output_clac_mles[i] - 2^{top_bit} * msb_mle)

        vp.add_mle_list(
            vec![second_eq_poly.clone(), msb_mle.clone(), msb_mle.clone()],
            -(final_pow_two + final_pow_two),
        );
        vp.add_mle_list(
            vec![second_eq_poly.clone(), msb_mle.clone()],
            final_batch_chal + final_pow_two,
        );

        let ltc_field: E = less_than_const.to_field();
        let quant_max_field: E = (*quantization::MAX).to_field();

        vp.add_mle_list(vec![second_eq_poly.clone()], ltc_field + quant_max_field);

        // Run the second sumcheck proof
        #[allow(deprecated)]
        let (proof_two, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // Extract the evaluations of all the bit mles
        let all_mle_evals = state.get_mle_final_evaluations();
        let bit_mle_evals = all_mle_evals[1..1 + bit_mles.len()].to_vec();

        // Construct a claim about the input tensor by combining the claims about `bit_mles` appropriately
        let rounding_const_field: E = rounding_constant.to_field();

        let (non_sign_part, last_pow_two) = bit_mle_evals
            .iter()
            .take(bit_mle_evals.len() - 1)
            .fold((E::ZERO, E::ONE), |(acc, pow_two_acc), &eval| {
                (acc + eval * pow_two_acc, pow_two_acc + pow_two_acc)
            });
        let with_sign = non_sign_part
            - last_pow_two * bit_mle_evals[bit_mle_evals.len() - 1]
            - rounding_const_field;

        let fpm: E = self.fixed_point_multiplier.to_field();
        // The unwrap here is safe as we should panic if the scaling factor is zero
        let fpm_inverse = fpm.invert().unwrap();

        let eval = fpm_inverse * with_sign;
        let point = proof_two.point.clone();

        // Push the proof to the proof list
        prover.push_proof(LayerProof::Requant(RequantProof {
            first_sumcheck: proof,
            second_sumcheck: proof_two,
            less_than_vals: first_sumcheck_decomp_evals,
            output_bit_mle_evals,
            bit_mle_evals,
        }));

        // Return the `Claim` about the Input tensor
        Ok(Claim { point, eval })
    }

    /// Performs logic to verify the [`RequantProof`] creadted by calling the [`Requant::prove_step`] method.
    pub(crate) fn verify_full_requant<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &RequantProof<E>,
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

        // All polynomials have the same number of vars and we expect a degree three VirtualPolynomial so we construct the VPAuxInfo here directly.
        let aux_info =
            VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![num_vars, num_vars, num_vars]]);

        // Generate the first sumcheck subclaim, the initial eval of this sumcheck should always be `last_claim.eval`
        let first_subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.first_sumcheck,
            &aux_info,
            verifier.transcript,
        );

        // These terms are used to prove that every element of each `less_than_vals` is either 0 or 1
        let (first_booleanity, _) = proof
            .less_than_vals
            .iter()
            .fold((E::ZERO, alpha_chal), |(acc, chal_acc), &bm| {
                (acc + chal_acc * (bm - bm * bm), chal_acc * alpha_chal)
            });

        // The output claim should be equal to `eq_poly * (clamping_mle * (SUM 2^{i} * output_calc_mles[i] - 2^{top_bit} * msb_eval) + *quantization::MAX * (1 - clamping_mle) * (1 - 2 * msb_eval))`
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

        let msb_eval = proof.output_bit_mle_evals[proof.output_bit_mle_evals.len() - 1];

        let with_sign = no_sign - final_pow_two * (msb_eval * clamping_eval);

        let clamping_coeff: E = (*quantization::MAX).to_field();

        let clamping_term =
            clamping_coeff * (E::ONE - clamping_eval) * (E::ONE - field_two * msb_eval);

        let eq_poly_eval: E = eq_eval(&last_claim.point, &first_subclaim.point_flat());

        // Sum all the individual terms and multiply by the eq_poly eval
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

        // We can save some work here by calculating part of the second sumcheck subclaim expected evaluation while exponentiating the challenge
        let (second_booleanity_check, challenge) = proof
            .bit_mle_evals
            .iter()
            .fold((E::ZERO, alpha_chal), |(acc, chal_acc), &eval| {
                (acc + chal_acc * (eval - eval * eval), chal_acc * alpha_chal)
            });
        // As above we save some work by doing two things at once
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
        // Generate the second sumcheck subclaim
        let second_subclaim = IOPVerifierState::<E>::verify(
            initial_eval,
            &proof.second_sumcheck,
            &aux_info,
            verifier.transcript,
        );

        let msb_eval = *proof.bit_mle_evals.last().unwrap();

        // We calculate this by first working out 2^bits_after_shift + (*quantization::MAX + (2 *msb_mle - 1) * (SUM 2^{i} * output_clac_mles[i] - 2^{top_bit} * msb_mle)
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

        // Sum all the terms and multiply by the second eq_poly eval
        let full_second_eval =
            (const_term + full_no_const + output_eval_part + second_booleanity_check)
                * second_eq_eval;

        ensure!(
            full_second_eval == second_subclaim.expected_evaluation,
            "Second subclaim evaluation wasn't what was expected, calculated: {:?}, expected: {:?}",
            full_second_eval,
            second_subclaim.expected_evaluation
        );

        // Use the individual claims about the `bit_mles`, which we just showed came from the previous sumcechk, to construct the claim about the input tensor.
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

        // Return the claim about the input tensor.
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
        (LayerCtx::<E>::Requant(*self), aux)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{Rng, thread_rng};

    use super::*;

    #[test]
    fn test_requant_op() -> anyhow::Result<()> {
        for vec_size in 5usize..11 {
            test_op_helper(vec_size)?;
        }
        Ok(())
    }

    fn test_op_helper(vec_size: usize) -> anyhow::Result<()> {
        let mut rng = thread_rng();
        // Create a random float input, a random float matrix and compute the output
        let float_vector = Tensor::<f32>::random(vec![vec_size]);

        let matrix_rows = rng.gen_range(vec_size..2 * vec_size);
        let float_matrix = Tensor::<f32>::random(vec![matrix_rows, vec_size]);

        let float_output = float_matrix.matvec(&float_vector);

        // Now comput the scaling factors for the tensors
        let input_max = float_vector.max_abs_output();
        let matrix_max = float_matrix.max_abs_output();
        let output_max = float_output.max_abs_output();

        let input_scaling = ScalingFactor::from_absolute_max(input_max, None);
        let matrix_scaling = ScalingFactor::from_absolute_max(matrix_max, None);
        let output_scaling = ScalingFactor::from_absolute_max(output_max, None);

        // Create the requantisation info
        let intermediate_bit_size =
            2 * (*quantization::BIT_LEN - 1) + vec_size.ilog2() as usize + 1;
        let requant = Requant::from_scaling_factors(
            input_scaling,
            matrix_scaling,
            output_scaling,
            intermediate_bit_size,
        );

        // Quantise the vectors
        let quantised_input = float_vector.quantize(&input_scaling);
        let quantised_matrix = float_matrix.quantize(&matrix_scaling);
        let quantised_output = float_output.quantize(&output_scaling);

        // Now run the matrix multiplication quantised, then requant and compare results
        let quant_mat_mul = quantised_matrix.matvec(&quantised_input);
        let calculated_output = requant.op(&quant_mat_mul)?;

        for (calculated, expected) in calculated_output
            .get_data()
            .iter()
            .zip(quantised_output.get_data().iter())
        {
            // There may be a slight rounding error due to using fixed point multiplication
            // So we check that the absoloute value of the difference is less than or equal to 1.
            assert!(
                calculated.abs_diff(*expected) <= 1,
                "Absoloute difference was larger than 1, calculated: {}, expected: {}",
                calculated,
                expected
            );
        }
        Ok(())
    }
}
