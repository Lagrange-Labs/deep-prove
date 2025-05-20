//! Module containign code for performing proving friendly requantisation. This is done via a [fixed point multiplication](https://en.wikipedia.org/wiki/Fixed-point_arithmetic#Binary_fixed-point_multiplication) and use of lookup arguments.

use crate::{
    Claim, Prover, ScalingFactor, Tensor,
    commit::{PCSError, compute_betas_eval},
    iop::verifier::Verifier,
    layers::LayerProof,
    lookup::logup_gkr::{prover::batch_prove as logup_batch_prove, verifier::verify_logup_proof},
    quantization,
};
use anyhow::{Result, anyhow, ensure};

use ff_ext::ExtensionField;
use gkr::util::ceil_log2;

use mpcs::{PolynomialCommitmentScheme, sum_check::eq_xy_eval};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE},
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};

use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::ContextAux,
    lookup::{context::TableType, logup_gkr::structs::LogUpProof},
    quantization::Fieldizer,
};

use super::LayerCtx;

/// Constnat used in fixed point multiplication for normalised [`f32`] values
const FIXED_POINT_SCALE: usize = 25;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
/// This struct contains the infomation used in requantisation (i.e. rescaling and clamping)
/// The fields are:
/// - `multiplier`: This is the actual [`f32`] value calculated as `S1 * S2 / S3` and in traditional quantisation is what we would multiply by and then round to requantise
/// - `right_shift`: This is `multiplier.log2().trunc().abs()`
/// - `fixed_point_multiplier`: This is `2.0.powf(multiplier.log2().fract()) * (1 << `fp_scale`)`, `fp_scale` is chosen to be at least 25 bits as the [`f32`] mantissa is only 24 bits long so this should retain all bits.
/// - `fp_scale`: This is calculated so that `fp_scale + right_shift` is a multiple of [`quantization::BIT_LEN`], that way we only need one size of range table.
/// - `intermediate_bit_size`: This is the maximum number of bits a value can have before its requantised.
pub struct Requant {
    /// After multiplying by `self.fixed_point_multiplier` the value need to be shifted by this plus 25.
    pub right_shift: usize,
    /// The normalised scaling factor represented as a fixed point multiplier (it should have 24 fractional bits)
    pub fixed_point_multiplier: Element,
    /// The scale used for the fixed point multiplier, it is calculated to be the smallest value greater than or equal to [`FIXED_POINT_SCALE`] such that
    /// the right shift we perform is a multiple of [`quantization::BIT_LEN`]
    pub fp_scale: usize,
    /// THe actual multiplier, this is mainly used to compare accuracy, it has no purpose in actual proving
    pub multiplier: f32,
    /// This field represents how many bits the max absoloute value can be
    pub(crate) intermediate_bit_size: usize,
}

/// Info related to the lookup protocol necessary to requantize
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequantCtx {
    pub requant: Requant,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
/// Struct holding all the information needed to verify requantisation was performed correctly.
/// This includes both lookup proofs and an additional sumcheck proof that we use so that all evaluations are at the same point.
pub struct RequantProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from activation + claim from lookup for the same poly
    /// e.g. the "link" between an activation and requant layer
    pub(crate) io_accumulation: IOPProof<E>,
    /// The evalaution claims about witness polynomials from the io_accumulation sumcheck
    pub(crate) accumulation_evals: Vec<E>,
    /// The clamping lookup proof for the requantization
    pub(crate) clamping_lookup: LogUpProof<E>,
    /// The range check lookup proof for the chunks that are shifted away
    pub(crate) shifted_lookup: LogUpProof<E>,
    /// COmmitments to lookup polynomials, they are in the order clamping commitments -> shifted commitments
    pub(crate) commitments: Vec<PCS::Commitment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Struct used to store the evaluations of the MLEs used in the lookup argument during requantisation.
/// - `clamping_in` is the input to the clmaping lookup and represents the most significant bits of the input that aren't shifted away
/// - `clamping_out` is the evaluations of the output MLE
/// - `shifted_chunks` is the value that gets shifted away split into [`quantization::BIT_LEN`] size limbs. They are ordered little endian.
pub struct RequantLookupWitness {
    /// The input to the clamping table
    pub(crate) clamping_in: Vec<Element>,
    /// The output of the clamping table
    pub(crate) clamping_out: Vec<Element>,
    /// The chunks that are shifted away
    pub(crate) shifted_chunks: Vec<Vec<Element>>,
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

        // We want the part that gets shifted away to be a multiple of the quantisation bit length (that way we can use the same range table for each chunk)
        let next_multiple = (int_part + FIXED_POINT_SCALE).next_multiple_of(*quantization::BIT_LEN);
        let fp_scale = next_multiple - int_part;
        let fixed_point_multiplier = (epsilon * (1u64 << fp_scale) as f32).round() as Element;

        Requant {
            right_shift: int_part,
            fixed_point_multiplier,
            fp_scale,
            multiplier: m,
            intermediate_bit_size,
        }
    }

    /// This returns the shift (including the part that depends on `S1 * S2/ S3`)
    pub(crate) fn shift(&self) -> usize {
        self.fp_scale + self.right_shift
    }

    /// Internal method that applies this op to an [`Element`]
    fn apply(&self, elem: &Element) -> Element {
        let rounding = 1i128 << (self.shift() - 1);
        let unclamped = (rounding + elem * self.fixed_point_multiplier) >> self.shift();
        let sign = if unclamped.is_positive() || unclamped == 0i128 {
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

    /// Function that tells us how large to make the clamping table
    pub(crate) fn clamping_size(&self) -> usize {
        let fpm_bit_size = ceil_log2(self.fixed_point_multiplier as usize);
        self.intermediate_bit_size + fpm_bit_size - self.shift()
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux<E>,
    ) -> (LayerCtx<E>, ContextAux<E>)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // Specify the two different tables that are required by this step
        aux.tables.insert(TableType::Clamping(self.clamping_size()));
        aux.tables.insert(TableType::Range);
        // There are no model polys for requantisation
        aux.model_polys = vec![];

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

    pub fn write_to_transcript<E: ExtensionField, T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from(self.right_shift as u64));
        t.append_field_element(&E::BaseField::from(self.fixed_point_multiplier as u64));
    }

    /// Method generates the witness that must be passed to lookup arguments.
    pub fn gen_lookup_witness<E: ExtensionField>(&self, input: &[Element]) -> RequantLookupWitness
    where
        E::BaseField: Serialize + DeserializeOwned,
    {
        // We take the input, mutliply by the fixed point multiplier and add the rounding constant. Then we split the resulting values into
        // parts that are either shifted away (these get range checked) or passed to the clamping table.
        let shift = self.shift();
        let rounding_constant = 1i128 << (shift - 1);
        let mask = (1i128 << shift) - 1;
        let (clamping, shifted): (Vec<Element>, Vec<Element>) = input
            .iter()
            .map(|&val| {
                let tmp = val * self.fixed_point_multiplier + rounding_constant;
                let clamp = tmp >> shift;
                let masked = tmp & mask;

                (clamp, masked)
            })
            .unzip();

        // Now we have to calculate the output for the clamped part and break the part that is shifted away into chunks to be range checked.
        // We do the clamping part first.
        let (clamping_in, clamping_out): (Vec<Element>, Vec<Element>) = clamping
            .into_par_iter()
            .map(|elem| {
                let clamp_out = if elem < *quantization::MIN {
                    *quantization::MIN
                } else if elem > *quantization::MAX {
                    *quantization::MAX
                } else {
                    elem
                };

                (elem, clamp_out)
            })
            .unzip();

        // Now we split the shifted part into pieces that fit into the range table
        let range_check_bit_size = *quantization::BIT_LEN;
        let range_mask = (1i128 << range_check_bit_size) - 1;

        // We need to calculate how many cunks we are splitting into, there should never be any remainder from this division.
        let no_chunks = shift / range_check_bit_size;

        let shifted_chunks = (0..no_chunks)
            .into_par_iter()
            .map(|j| {
                shifted
                    .iter()
                    .map(|&elem| {
                        let tmp = elem >> (j * range_check_bit_size);
                        tmp & range_mask
                    })
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Vec<Element>>>();

        RequantLookupWitness {
            clamping_in,
            clamping_out,
            shifted_chunks,
        }
    }

    /// Function to recombine claims of constituent MLEs into a single value to be used as the initial sumcheck evaluation
    /// of the subsequent proof.
    pub fn recombine_claims<E: ExtensionField>(
        &self,
        clamping_claim: E,
        shifted_claims: &[E],
    ) -> E {
        // First we recombine the clamping claim with the shifted chunks
        // We want `clamping_claim * shift_field + SUM 2^{i}*shifted_claims[i]`
        let shift_field = E::from(1u64 << self.shift());
        let (full_val, _) = shifted_claims.iter().fold(
            (shift_field * clamping_claim, E::ONE),
            |(acc, pow_two), &val| {
                (
                    acc + val * pow_two,
                    pow_two * E::from(1u64 << *quantization::BIT_LEN),
                )
            },
        );

        // Now we subtract the rounding constant and then multiply by the inverse of the fixed point multiplier
        // We do this because `input = fpm^{-1}*(full_val - rounding_constant)`
        let rounding_const_field = E::from(1u64 << (self.shift() - 1));

        let fpm_field: E = self.fixed_point_multiplier.to_field();
        let fpm_inverse = fpm_field.invert().unwrap();

        (full_val - rounding_const_field) * fpm_inverse
    }

    /// Method that proves requantisation was performed correctly. First it runs the lookup argument for the clamping claim and batches all the range checks
    /// for the shifted polys together. Then it performs a sumcheck that takes the output claims from the two lookup arguments and produces a new claim where all of the polynomials are
    /// evaluated at the same point. This sumcheck also checks that the output column of the clamping lookup relates to the same polynomial as `last_claim`.
    pub(crate) fn prove_step<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        requant_info: &RequantCtx,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // Retrieve the precalculated lookup witnesses.
        let shifted_logup_witness = prover.next_lookup_witness()?;
        let clamping_logup_witness = prover.next_lookup_witness()?;

        let shifted_prover_info =
            shifted_logup_witness.get_logup_input(&prover.challenge_storage)?;
        let clamping_prover_info =
            clamping_logup_witness.get_logup_input(&prover.challenge_storage)?;

        let shifted_commitments = shifted_logup_witness.get_commitments();
        let clamping_commitments = clamping_logup_witness.get_commitments();
        // Run the lookup protocol and return the lookup proof
        let clamping_logup_proof = logup_batch_prove(&clamping_prover_info, prover.transcript)?;
        let shifted_logup_proof = logup_batch_prove(&shifted_prover_info, prover.transcript)?;
        // We need to prove that the output of this step is the input to following activation function
        // this is done by showing that the `last_claim` and the output column of the clamping lookup both relate to the
        // same polynomial. In addition, we need all the shifted claims to be about the same point as the clamping input claim, so we include these in the sumcheck as well
        if clamping_prover_info.column_evals().len() != 2 {
            return Err(anyhow!(
                "Clamping logup proofs should only have two output evaluations, got: {}",
                clamping_prover_info.column_evals().len()
            ));
        }

        // Extract the individual polynomials from the lookup arguments to be passed to the sumcheck
        let clamping_polys = clamping_prover_info.column_evals();
        let num_vars = clamping_polys[0].len().ilog2() as usize;

        let clamping_mles = clamping_polys
            .iter()
            .map(|evaluations| {
                DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations).into()
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        let shifted_mles = shifted_prover_info
            .column_evals()
            .iter()
            .map(|evaluations| {
                DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations).into()
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        // produce a beta poly for the claim point for the claming lookup argument
        let clamping_beta: ArcMultilinearExtension<E> =
            compute_betas_eval(&clamping_logup_proof.output_claims()[0].point)
                .into_mle()
                .into();
        // Produce a beta poly for the last_claim point
        let last_claim_beta: ArcMultilinearExtension<E> =
            compute_betas_eval(&last_claim.point).into_mle().into();
        // Produce a beta poly for the shifted polys lookup argument
        let shifted_beta: ArcMultilinearExtension<E> =
            compute_betas_eval(&shifted_logup_proof.output_claims()[0].point)
                .into_mle()
                .into();

        // Squeeze a batching challenge from the transcript.
        let batching_challenge = prover
            .transcript
            .get_and_append_challenge(b"requant_batching")
            .elements;

        // Construct the virtual polynomial for the sumcheck
        let mut vp = VirtualPolynomial::<E>::new(num_vars);

        vp.add_mle_list(vec![clamping_mles[1].clone(), last_claim_beta], E::ONE);
        vp.add_mle_list(
            vec![clamping_mles[1].clone(), clamping_beta.clone()],
            batching_challenge,
        );

        let mut combiner = batching_challenge * batching_challenge;
        vp.add_mle_list(vec![clamping_mles[0].clone(), clamping_beta], combiner);

        combiner *= batching_challenge;
        shifted_mles.iter().for_each(|mle| {
            vp.add_mle_list(vec![shifted_beta.clone(), mle.clone()], combiner);
            combiner *= batching_challenge;
        });

        // Run the sumcheck prover for the claims
        // This sumcheck checks that the polynomial `last_claim` relates to is the clamping output while simultaneously providing us with claimns for clamping input and
        // the shifted chunks at the same point (we need them all evalauted at the same point so we can recombine the evaluations and produce the next claim).
        #[allow(deprecated)]
        let (claim_acc_proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // Split out the eq poly evals from witness poly evals
        let final_evals = state.get_mle_final_evaluations();
        let point = claim_acc_proof.point.clone();
        let clamping_out_eval = final_evals[0];
        let clamping_in_eval = final_evals[3];
        let shifted_evals = &final_evals[5..];

        // Calculate the combined evaluation to pass to the next layer
        let combined_eval = requant_info
            .requant
            .recombine_claims(clamping_in_eval, shifted_evals);

        // Add the points and evaluations to open commitments at
        let (accumulation_evals, commitments): (Vec<E>, Vec<PCS::Commitment>) =
            [clamping_in_eval, clamping_out_eval]
                .iter()
                .chain(shifted_evals.iter())
                .zip(clamping_commitments.into_iter().chain(shifted_commitments))
                .map(|(&eval, comm_with_wit)| {
                    let commitment = PCS::get_pure_commitment(&comm_with_wit.0);
                    prover
                        .commit_prover
                        .add_witness_claim(comm_with_wit, Claim::<E>::new(point.clone(), eval))?;

                    Result::<(E, PCS::Commitment), PCSError>::Ok((eval, commitment))
                })
                .collect::<Result<Vec<(E, PCS::Commitment)>, PCSError>>()?
                .into_iter()
                .unzip();

        // Add the layer proof to the list
        prover.push_proof(LayerProof::Requant(RequantProof {
            io_accumulation: claim_acc_proof,
            accumulation_evals,
            clamping_lookup: clamping_logup_proof,
            shifted_lookup: shifted_logup_proof,
            commitments,
        }));

        Ok(Claim {
            point,
            eval: combined_eval,
        })
    }
}

impl RequantCtx {
    /// Method that verifies requantisation has been performed correctly when supplied with a [`RequantProof`].
    /// It verifies both lookup argument proofs, calculates the initial claim for the sumcheck proof using the lookup argument claims
    /// and then verifies the sumcheck using this initial claim. It then takes the output claims provided by the prover, checks they relate to the sumcheck
    /// subclaim, adds them to the list of claims of commitment openings and then calculates the next claim.
    pub(crate) fn verify_requant<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: Claim<E>,
        proof: &RequantProof<E, PCS>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proofs
        let RequantProof {
            io_accumulation,
            accumulation_evals,
            clamping_lookup,
            shifted_lookup,
            commitments,
        } = proof;
        // Work out how many instances of range check are batched into the shifted claims
        let shifted_instances = self.requant.shift() / *quantization::BIT_LEN;
        // Verify both lookup arguments in the same order they are proved.
        let clamping_claims = verify_logup_proof(
            clamping_lookup,
            1,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;
        let shifted_claims = verify_logup_proof(
            shifted_lookup,
            shifted_instances,
            constant_challenge,
            E::ONE,
            verifier.transcript,
        )?;

        // Squeeze the batching challenge for the claim accumulation sumcheck
        let batching_challenge = verifier
            .transcript
            .get_and_append_challenge(b"requant_batching")
            .elements;

        // 2. Verify claim accumulation
        // Work out the initial sumcheck evaluation
        let clamping_point = clamping_claims.point();
        let clamping_evals = clamping_claims
            .claims()
            .iter()
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();

        let shifted_point = shifted_claims.point();
        let shifted_evals = shifted_claims
            .claims()
            .iter()
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();
        let (initial_eval, _) = [last_claim.eval, clamping_evals[1], clamping_evals[0]]
            .iter()
            .chain(shifted_evals.iter())
            .fold((E::ZERO, E::ONE), |(acc, chal), &val| {
                (acc + chal * val, chal * batching_challenge)
            });
        // The verifier can work out the auxiliary information about the sumcheck on their own.
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
            clamping_point.len(),
            clamping_point.len(),
        ]]);
        // Run sumcheck verification to obtaint he subclaim
        let subclaim = IOPVerifierState::<E>::verify(
            initial_eval,
            io_accumulation,
            &aux_info,
            verifier.transcript,
        );

        // Now we check that the evalautions provided by the prover do recombine to the sumcheck subclaim
        let acc_point = subclaim.point_flat();
        // Calculate all th ebeta poly evals ourselves
        let last_claim_beta = eq_xy_eval(&last_claim.point, &acc_point);
        let clamping_beta = eq_xy_eval(clamping_point, &acc_point);
        let shifted_beta = eq_xy_eval(shifted_point, &acc_point);

        // Recombine the evaluations provided by the prover in the expected way
        let clamping_out_part =
            (last_claim_beta + batching_challenge * clamping_beta) * accumulation_evals[1];
        let mut combiner = batching_challenge * batching_challenge;

        let clamping_in_part = combiner * clamping_beta * accumulation_evals[0];

        combiner *= batching_challenge;
        let (calc_claim, _) = accumulation_evals[2..].iter().fold(
            (clamping_in_part + clamping_out_part, combiner),
            |(value_acc, chal), &val| {
                (
                    value_acc + val * shifted_beta * chal,
                    chal * batching_challenge,
                )
            },
        );

        // Error if the calculated claim does not equal the expected evaluation
        ensure!(
            calc_claim == subclaim.expected_evaluation,
            "The calculated claim did not line up with the expected claim, calculated: {:?}, expected: {:?}",
            calc_claim,
            subclaim.expected_evaluation
        );

        // 3. Calculate the next layer claim evaluation
        let next_claim_eval = self
            .requant
            .recombine_claims(accumulation_evals[0], &accumulation_evals[2..]);

        // 4. Add claims to commitment verifier
        accumulation_evals
            .iter()
            .zip(commitments)
            .try_for_each(|(&eval, commit)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(commit.clone(), Claim::<E>::new(acc_point.clone(), eval))
            })?;

        Ok(Claim::<E>::new(acc_point, next_claim_eval))
    }
}
