//! This layer applies the softmax function to the last dimension of the input tensor
use std::marker::PhantomData;

use crate::{
    Claim, Element, ScalingStrategy, Tensor,
    commit::compute_betas_eval,
    iop::{
        context::{Context, ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData,
            QuantizeOp, QuantizeOutput, VerifiableCtx,
        },
    },
    lookup::{
        context::{COLUMN_SEPARATOR, LookupWitnessGen, TableType},
        logup_gkr::{prover::batch_prove, structs::LogUpProof, verifier::verify_logup_proof},
        witness::LogUpWitness,
    },
    model::StepData,
    quantization::{Fieldizer, ScalingFactor},
    tensor::{Number, Shape},
};

use anyhow::{Result, anyhow, ensure};

use ff_ext::ExtensionField;

use mpcs::{PolynomialCommitmentScheme, sum_check::eq_xy_eval};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};

/// The base 2 logarithm of the scale factor used in exponential lookup tables
pub(crate) const LOG_SCALE_FACTOR: usize = 24;
/// The scale factor for our fixed point arithmetic
pub(crate) const SCALE_FACTOR: usize = 1 << LOG_SCALE_FACTOR;
/// The scale factor of the outputs of the `exp` lookup
pub(crate) const OUTPUT_SCALE_FACTOR: usize = 1 << (LOG_SCALE_FACTOR - 1);

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax<N> {
    // By default, it's equal to 1
    /// This is the factor we divide by before exponentiating, when thought of as a Boltzmann distribution this is
    /// often referred to as the "Temperature".
    pub scalar: N,
    // By default, softmax is going to be applied on the full tensor.
    // You can specificy a dimen to apply softmax on. For example, for a tensor  of shape [2,3,4],
    // if apply_on_dim = 1, then softmax will be applied on every chunks of 4 elements each.
    pub apply_on_dim: Option<usize>,
    /// This is the maximum size of dimension that we will normalise over. For example in an Attention layer this would be the maximum context size.
    max_size: usize,
    /// This is the extra information required to compute the quantised version, it defaults to [`None`].
    quant_info: Option<QuantisedSoftmaxData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// This struct is used to store information used when evaluating the quantised version of [`Softmax`] on
/// [`Element`]s.
struct QuantisedSoftmaxData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This stores the output column of the `exp` lookup
    lut: Vec<Element>,
    /// The error bound as calculated by the formulae given in the zkLLM paper
    error_bound: f32,
    /// The float temperature for calculating row normalisation
    float_temperature: f32,
    /// This value indicates the point that we map everything greater than this to zero
    bkm: Element,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`Softmax`] operation.
pub struct SoftmaxProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The proof of the lookup into the exponential table
    pub(crate) exp_lookup: LogUpProof<E>,
    /// The proof for all of the range lookups
    pub(crate) range_lookup: LogUpProof<E>,
    /// The proof for the error table lookup
    pub(crate) error_lookup: LogUpProof<E>,
    /// Witness commitments for this layer
    pub(crate) commitments: Vec<PCS::Commitment>,
    /// The sumcheck proof we use to make sure everything is evaluated at the same point.
    pub(crate) accumulation_proof: IOPProof<E>,
    /// The claimed evaluations of the commitments
    pub(crate) evaluations: Vec<E>,
}

impl<N: Number> Default for Softmax<N> {
    fn default() -> Self {
        Softmax {
            scalar: N::unit(),
            apply_on_dim: None,
            max_size: 1024usize,
            quant_info: None,
        }
    }
}

impl<N: Number> Softmax<N> {
    pub fn new() -> Self {
        Softmax::<N>::default()
    }

    pub fn new_with_scale(scale: N, max_context_size: usize) -> Softmax<N> {
        Softmax {
            scalar: scale,
            apply_on_dim: None,
            max_size: max_context_size,
            quant_info: None,
        }
    }
    pub fn quantise(&self, input_scaling: ScalingFactor) -> Result<Softmax<Element>> {
        // First we work out what we need to multiply by to get the input scale factor to be 2^32
        let input_scale_factor = input_scaling.scale();
        let temperature = self.scalar.to_f32()?;
        let float_temperature = 1.0f32 / temperature;
        let multiplier = (SCALE_FACTOR as f32 * input_scale_factor).round() as Element;

        // minimum_input is calculated as `(input_min - sqrt(d) * ln_n - d * input_max)/sqrt(d)` and then quantised
        let input_min = input_scaling.min();
        let input_max = input_scaling.max();

        let min_input_float = input_min
            - float_temperature * (self.max_size as f32 * (input_max * temperature).exp()).ln();
        // Now that we have the minimum possible input as a float we need to work out how many integral bits we need to account for
        // We know that the minimum input is negative so first we take the absoloute value
        let min_input_abs = min_input_float.abs();

        let int = min_input_abs.round() as usize;
        let integral_bits = ceil_log2(int);

        let table_size = 1i128 << (integral_bits + 8);
        let base = 1i128 << (LOG_SCALE_FACTOR - 8);

        let (float_error, bkm_float) = calc_softmax_error(
            base,
            self.max_size as f32,
            OUTPUT_SCALE_FACTOR as f32,
            SCALE_FACTOR as f32,
            3.0f32,
            0.0f32,
            2.0f32,
            float_temperature,
        );

        let float_error = float_error.abs();
        let bkm = bkm_float.round() as Element;
        // Make the exp lookup table
        let lut = (0i128..table_size)
            .map(|j| {
                let prod = base * j;
                if prod > bkm {
                    0i128
                } else {
                    let float_exp =
                        (-prod as f32 / (SCALE_FACTOR as f32 * float_temperature)).exp();
                    (float_exp * OUTPUT_SCALE_FACTOR as f32).round() as Element
                }
            })
            .collect::<Vec<Element>>();

        // Store all the quantised info for quantised evaluation
        let quant_info = QuantisedSoftmaxData {
            input_scale_factor: input_scaling,
            lut,
            error_bound: float_error,
            float_temperature,
            bkm,
        };

        // Return the quantised `Softmax` operator
        Ok(Softmax::<Element> {
            scalar: multiplier,
            apply_on_dim: self.apply_on_dim,
            max_size: self.max_size,
            quant_info: Some(quant_info),
        })
    }

    fn quant_info(&self) -> Option<&QuantisedSoftmaxData> {
        self.quant_info.as_ref()
    }
    pub fn with_scale(self, scale: N) -> Self {
        Self {
            scalar: scale,
            ..self
        }
    }
    /// Apply softmax on the subset of from this dim
    pub fn on_dim(self, dim: usize) -> Self {
        Self {
            apply_on_dim: Some(dim),
            ..self
        }
    }
}

/// Calculates the error as an [`f32`] when applying softmax as described in zkLLM.
/// This functions returns the error togeter with the value `bkm` such that anything smaller
/// than `bkm` should be mapped to zero.
pub(crate) fn calc_softmax_error(
    bl: i128,
    max_context_size: f32,
    output_sf: f32,
    input_sf: f32,
    k: f32,
    m: f32,
    l: f32,
    temp: f32,
) -> (f32, f32) {
    // First we calculate the optimal point to map everything to zero (to minimise the L1 error)
    let kml = k - m - l;
    let bkm_multiplier = kml * (2.0f32 * max_context_size).ln() + output_sf.ln();
    let bkm = input_sf * temp * bkm_multiplier / (kml + 1.0f32);
    // Now that we have bkm we calculate the allowable float error
    let common_denom = kml * input_sf * temp;
    let first_term = (bl as f32 / common_denom).exp();
    let second_term = (bkm / common_denom).exp() / (2.0f32 * output_sf.powf(1.0 / kml));
    // This is the C constant referenced in the appendix of zkLLM
    let c = (first_term + second_term).powf(kml) - 1.0f32;
    // These terms are used to give the L1 error bound
    let term_one = c * (1.0f32 / (2.0f32 * input_sf * temp)).exp();
    let term_two = (max_context_size - 1.0f32) * ((-bkm as f32) / input_sf * temp).exp();
    (term_one + term_two, bkm)
}

impl Evaluate<f32> for Softmax<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "softmax expects exactly one input tensor currently"
        );
        let input = inputs[0];
        let dim = self.apply_on_dim.unwrap_or(input.get_shape().len() - 1);
        let output = input
            .slice_on_dim(dim)
            .0
            .map(|vec| {
                let scaled = vec
                    .iter()
                    .map(|x| self.scalar * x)
                    .map(|x| x.exp())
                    .collect::<Vec<_>>();
                let sum = scaled.iter().sum::<f32>();
                scaled.iter().map(|x| x / sum).collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        let output_tensor = Tensor::new(input.get_shape(), output);
        Ok(LayerOut::from_vec(vec![output_tensor]))
    }
}

impl<N: Number> OpInfo for Softmax<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Softmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
/// Struct containing data useful for proving correctness of [`Softmax`]. This is data that we compute anyway
/// during quantised evaluation.
pub struct SoftmaxData<E>
where
    E: Clone + ExtensionField,
{
    /// This is the natural logarithm of the sum of the exponentiated input along the given dimension
    shift_data: Vec<Element>,
    /// The lowest 8-bits of the input (after rescaling)
    low_range_check: Vec<Element>,
    /// The second lowest 8 bits of the input (after rescaling)
    high_range_check: Vec<Element>,
    /// The inputs and outputs of the exponential lookup table
    exp_lookup: (Vec<Element>, Vec<Element>),
    _phantom: PhantomData<E>,
}

impl Evaluate<Element> for Softmax<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<Element, E>> {
        // First we heck that we have some quantisation info.
        ensure!(
            self.quant_info.is_some(),
            "Could not evaluate quantised softmax because the operation has not been quantised"
        );
        // Check that we only have one input
        ensure!(
            inputs.len() == 1,
            "Exepected a single input to quantised softmax, got: {}",
            inputs.len()
        );

        // Since we have checked that quant info exists this unwrap is safe
        let QuantisedSoftmaxData {
            input_scale_factor,
            lut,
            float_temperature,
            bkm,
            ..
        } = self.quant_info().unwrap();

        let input = inputs[0];
        let chunk_size = *input.shape.last().ok_or(anyhow!(
            "Could not evaluate Softmax, Input tensor had no shape"
        ))?;

        let unpadded_chunk_size = *unpadded_input_shapes[0].last().ok_or(anyhow!(
            "Could not evaluate Softmax, unpadded input shape was empty for input"
        ))?;
        let unpadded_size = unpadded_input_shapes[0].iter().product::<usize>();
        let padded_size = input.shape.iter().product::<usize>();
        let chunk_sizes = input
            .shape
            .iter()
            .zip(unpadded_input_shapes[0].iter())
            .take(input.shape.len() - 1)
            .scan(
                (padded_size, unpadded_size),
                |(padded_state, unpadded_state), (padded_dim, unpadded_dim)| {
                    *padded_state /= padded_dim;
                    *unpadded_state /= unpadded_dim;
                    Some((*padded_state, *unpadded_state))
                },
            )
            .collect::<Vec<(usize, usize)>>();
        // Calculate the shift chunk by chunk
        let shift_data = input
            .get_data()
            .chunks(chunk_size)
            .map(|vec| {
                let sum = vec
                    .iter()
                    .take(unpadded_chunk_size)
                    .map(|x| (input_scale_factor.dequantize(x) / float_temperature).exp())
                    .sum::<f32>();
                let log_sum = sum.ln();
                let shift = -(SCALE_FACTOR as f32 * float_temperature * log_sum).round() as Element;
                vec![shift; chunk_size]
            })
            .flatten()
            .collect::<Vec<_>>();
        // We use the mask to extract 8-bit chunks of the input, these are the smallest fractional bits
        // and so we can assume that they get mapped to 1 under `exp`
        let mask = 255i128;
        // Now we rescale and chunk the `softmax_input`
        let ((lookups, outputs), (high_range_check, low_range_check)): (
            (Vec<Element>, Vec<Element>),
            (Vec<Element>, Vec<Element>),
        ) = input
            .get_data()
            .iter()
            .zip(shift_data.iter())
            .map(|(&input_elem, &shift)| {
                // We take the absoloute value as this is guaranteed to be negative
                let rescaled = (input_elem * self.scalar + shift).abs();
                // The lest significant chunk (fractional bits 17 to 24)
                let lsc = rescaled & mask;
                // The second lest significant chunk (fractional bits 9 to 16)
                let lsc2 = (rescaled >> 8) & mask;
                // The most significant chunk (all the integral bits (usually around 7 for GPT2) + fractional bits 1 to 8)
                let lookup = rescaled >> 16;

                ((lookup, lut[lookup as usize]), (lsc2, lsc))
            })
            .unzip();

        // We store all the information that has been computed in this step that will be useful later for proving.
        let proving_data = ProvingData::Softmax(SoftmaxData {
            shift_data,
            low_range_check,
            high_range_check,
            exp_lookup: (lookups, outputs.clone()),
            _phantom: PhantomData::<E>,
        });

        // Make the output tensor
        let output = Tensor::<Element>::new(input.get_shape(), outputs);

        Ok(LayerOut {
            outputs: vec![output],
            proving_data,
        })
    }
}

impl PadOp for Softmax<Element> {}

impl<E, PCS> ProvableOp<E, PCS> for Softmax<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = SoftmaxCtx;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        _step_data: &StepData<E, E>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        // Check we have the correct number of claims
        ensure!(
            last_claims.len() == 1,
            "Softmax only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];

        let logup_witnesses = prover.lookup_witness(node_id)?;
        // Check that we have two witnesses for Softmax
        if logup_witnesses.len() != 3 {
            return Err(anyhow!(
                "There should be three lookup witnesses during Softmax proving, node: {}, number of witnesses: {}",
                node_id,
                logup_witnesses.len()
            ));
        }
        // Run the lookup protocol and return the lookup proof
        let exp_logup_witness = &logup_witnesses[0];
        let range_logup_witness = &logup_witnesses[1];
        let error_logup_witness = &logup_witnesses[2];
        let exp_commitments = exp_logup_witness.get_commitments();
        let range_commitments = range_logup_witness.get_commitments();
        let initial_shift = error_logup_witness.get_commitments();

        // Run the lookup protocol and return the lookup proof
        let exp_prover_info = exp_logup_witness.get_logup_input(&prover.challenge_storage)?;
        let range_prover_info = range_logup_witness.get_logup_input(&prover.challenge_storage)?;
        let error_prover_info = error_logup_witness.get_logup_input(&prover.challenge_storage)?;
        // Make the LogUp proofs
        let exp_logup_proof = batch_prove(&exp_prover_info, prover.transcript)?;
        let range_logup_proof = batch_prove(&range_prover_info, prover.transcript)?;
        let error_logup_proof = batch_prove(&error_prover_info, prover.transcript)?;

        // Now we need to run a sumcheck to combine evaluations on the same polynomials and so that we can construct a claim about the input
        let exp_claims = exp_logup_proof.output_claims();
        let range_claims = range_logup_proof.output_claims();
        let error_claims = error_logup_proof.output_claims();

        let exp_point = exp_claims
            .first()
            .and_then(|claim| Some(&claim.point))
            .ok_or(anyhow!("Exponential lookup in Softmax should have claims"))?;
        let range_point = range_claims
            .first()
            .and_then(|claim| Some(&claim.point))
            .ok_or(anyhow!("Range lookup in Softmax should have claims"))?;
        let error_point = error_claims
            .first()
            .and_then(|claim| Some(&claim.point))
            .ok_or(anyhow!("Error lookup in Softmax should have claims"))?;
        // We use the difference in point length between the error point and the exp point to work out how many variables correspond to the normalisation dimension
        let extra_vars = exp_point.len() - error_point.len();

        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();
        let two_mult = E::from_canonical_u64(1u64 << extra_vars);

        let full_error_point = std::iter::repeat_n(two_inv, extra_vars)
            .chain(error_point.iter().copied())
            .collect::<Vec<E>>();

        // Squeeze a batching cahllenge from the transcript
        let alpha = prover
            .transcript
            .get_and_append_challenge(b"batching_challenge")
            .elements;

        let exp_beta: ArcMultilinearExtension<E> = compute_betas_eval(exp_point).into_mle().into();
        let range_beta: ArcMultilinearExtension<E> =
            compute_betas_eval(range_point).into_mle().into();
        let error_beta: ArcMultilinearExtension<E> =
            compute_betas_eval(&full_error_point).into_mle().into();
        let last_claim_beta: ArcMultilinearExtension<E> =
            compute_betas_eval(&last_claim.point).into_mle().into();

        // Start to build the virtual polynomial, begin with exponential polys
        let (vp, batch_challenge) = exp_commitments.iter().fold(
            (VirtualPolynomial::<E>::new(exp_point.len()), E::ONE),
            |(mut vp_acc, bc), (_, poly)| {
                vp_acc.add_mle_list(vec![poly.clone().into(), exp_beta.clone()], bc);
                (vp_acc, bc * alpha)
            },
        );
        // Add range polys
        let (mut vp, batch_challenge) =
            range_commitments
                .iter()
                .fold((vp, batch_challenge), |(mut vp_acc, bc), (_, poly)| {
                    vp_acc.add_mle_list(vec![poly.clone().into(), range_beta.clone()], bc);
                    (vp_acc, bc * alpha)
                });

        // Fianlly add the error check and the last claim, for this we need the output column of the exponential lookup
        let (_, exp_output) = exp_commitments
            .last()
            .ok_or(anyhow!("Exponential lookup in Softmax had no commitments"))?;

        vp.add_mle_list(
            vec![exp_output.clone().into(), error_beta.clone()],
            -batch_challenge * two_mult,
        );
        let output_quantised_one: E = (OUTPUT_SCALE_FACTOR as Element).to_field();
        vp.add_mle_list(vec![error_beta], batch_challenge * output_quantised_one);
        vp.add_mle_list(
            vec![exp_output.clone().into(), last_claim_beta],
            batch_challenge * alpha,
        );
        // Run the sumcheck proof
        #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);
        // We need the point and all the poly evals (excluding beta polys)
        let sumcheck_point = &sumcheck_proof.point;
        let all_evals = state.get_mle_final_evaluations();
        let exp_evals = &[all_evals[0], all_evals[2]];
        let range_evals = &[all_evals[3], all_evals[5]];
        let shift_eval = initial_shift[0].1.evaluate(sumcheck_point);

        // Work out the input eval
        let two_to_the_16 = E::from_canonical_u64(1u64 << 16);
        let two_to_the_8 = E::from_canonical_u64(1u64 << 8);
        let field_multiplier: E = self.scalar.to_field();
        let field_multiplier_inv = field_multiplier.inverse();
        let input_eval = -field_multiplier_inv
            * (exp_evals[0] * two_to_the_16
                + range_evals[1] * two_to_the_8
                + range_evals[0]
                + shift_eval);

        let input_claim = Claim::<E>::new(sumcheck_point.clone(), input_eval);

        // Add the commitments to be opened to the commitment prover
        let exp_commits = exp_commitments
            .iter()
            .zip(exp_evals.iter())
            .map(|(comm_with_wit, eval)| {
                let comm = PCS::get_pure_commitment(&comm_with_wit.0);
                prover.commit_prover.add_witness_claim(
                    comm_with_wit.clone(),
                    Claim::<E>::new(sumcheck_point.clone(), *eval),
                )?;
                Ok(comm)
            })
            .collect::<Result<Vec<PCS::Commitment>, anyhow::Error>>()?;

        let range_commits = range_commitments
            .iter()
            .zip(range_evals.iter())
            .map(|(comm_with_wit, eval)| {
                let comm = PCS::get_pure_commitment(&comm_with_wit.0);
                prover.commit_prover.add_witness_claim(
                    comm_with_wit.clone(),
                    Claim::<E>::new(sumcheck_point.clone(), *eval),
                )?;
                Ok(comm)
            })
            .collect::<Result<Vec<PCS::Commitment>, anyhow::Error>>()?;

        prover.commit_prover.add_witness_claim(
            initial_shift[0].clone(),
            Claim::<E>::new(sumcheck_point.clone(), shift_eval),
        )?;
        let shift_commit = PCS::get_pure_commitment(&initial_shift[0].0);

        let commitments = exp_commits
            .into_iter()
            .chain(range_commits.into_iter())
            .chain(std::iter::once(shift_commit))
            .collect::<Vec<PCS::Commitment>>();
        let evaluations = exp_evals
            .into_iter()
            .chain(range_evals.into_iter())
            .copied()
            .chain(std::iter::once(shift_eval))
            .collect::<Vec<E>>();

        // Add the proof to the proof list
        prover.push_proof(
            node_id,
            LayerProof::<E, PCS>::Softmax(SoftmaxProof::<E, PCS> {
                exp_lookup: exp_logup_proof,
                range_lookup: range_logup_proof,
                error_lookup: error_logup_proof,
                commitments,
                accumulation_proof: sumcheck_proof,
                evaluations,
            }),
        );

        Ok(vec![input_claim])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        gen: &mut LookupWitnessGen<E, PCS>,
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<()> {
        ensure!(
            step_data.inputs.len() == 1,
            "Found more than 1 input in inference step of Softmax layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output in inference step of Softmax layer"
        );
        // Get the data generated during quantised evaluation
        let SoftmaxData {
            shift_data,
            low_range_check,
            high_range_check,
            exp_lookup: (exp_input, exp_output),
            ..
        } = step_data.outputs.try_softmax_data().ok_or(anyhow!(
            "Could not get SoftmaxData during Softmax lookup witness generation"
        ))?;
        let num_vars = ceil_log2(exp_input.len());
        // We need to work out how many chunks to split the normalisation into to be range checked.
        let QuantisedSoftmaxData {
            error_bound,
            float_temperature,
            bkm,
            lut,
            ..
        } = self.quant_info().ok_or(anyhow!(
            "Could not prove Softmax because it had no quantisation data"
        ))?;
        let allowable_error = (*error_bound * OUTPUT_SCALE_FACTOR as f32).round() as Element;

        // Now we construct the polynomials used in the lookups
        // To do this we need the size of the last dimension
        let final_dim_size = *step_data.outputs.outputs()[0]
            .get_shape()
            .last()
            .ok_or(anyhow!("Softmax output tensor did not have a shape"))?;
        let normalisation_lookup = exp_output
            .chunks(final_dim_size)
            .map(|chunk| {
                let sum = chunk.iter().sum::<Element>();
                OUTPUT_SCALE_FACTOR as Element - sum
            })
            .collect::<Vec<Element>>();

        let merged_range_check = low_range_check
            .iter()
            .chain(high_range_check.iter())
            .copied()
            .collect::<Vec<Element>>();
        let merged_softmax = exp_input
            .iter()
            .zip(exp_output.iter())
            .map(|(input, output)| input + output * COLUMN_SEPARATOR)
            .collect::<Vec<Element>>();

        // Make the commitments to the exp lookup
        let (exp_commits, exp_evals): (
            Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
            Vec<Vec<E::BaseField>>,
        ) = [exp_input, exp_output]
            .into_par_iter()
            .map(|vals| {
                let evaluations = vals
                    .into_iter()
                    .map(|v| {
                        let f: E = v.to_field();
                        f.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>();
                let mle =
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &evaluations);
                let commit = ctx.commitment_ctx.commit(&mle)?;
                Ok(((commit, mle), evaluations))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // Make the commitments to the range checks
        let (range_commits, range_evals): (
            Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>,
            Vec<Vec<E::BaseField>>,
        ) = [low_range_check, high_range_check]
            .into_par_iter()
            .map(|vals| {
                let evaluations = vals
                    .into_iter()
                    .map(|v| {
                        let f: E = v.to_field();
                        f.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>();
                let mle =
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &evaluations);
                let commit = ctx.commitment_ctx.commit(&mle)?;
                Ok(((commit, mle), evaluations))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // For the error we actually use the exp output table commitment so here we only need to make the evaluations
        // but we will store the `shift` polynomial and its commitment in the `LogUpWitness` that we create
        let error_evals = normalisation_lookup
            .par_iter()
            .map(|val| {
                let f: E = val.to_field();
                f.as_bases()[0]
            })
            .collect::<Vec<E::BaseField>>();

        let shift_mle = DenseMultilinearExtension::<E>::from_evaluations_vec(
            num_vars,
            shift_data
                .iter()
                .map(|v| {
                    let f: E = v.to_field();
                    f.as_bases()[0]
                })
                .collect::<Vec<E::BaseField>>(),
        );
        let shift_commit = ctx.commitment_ctx.commit(&shift_mle)?;
        // Add the looked up values to the generator so we can make multiplicity polys later
        let lookups = gen.new_lookups.get_mut(&TableType::Range).ok_or(anyhow!(
            "No table of type Range was expected, error occured during Softmax step"
        ))?;
        lookups.extend(merged_range_check);

        // Need to recreate the parameters for the Softmax table
        let float_temp_bits = float_temperature.to_bits();

        let lookups = gen
            .new_lookups
            .get_mut(&TableType::Softmax(
                float_temp_bits,
                ceil_log2(lut.len()),
                *bkm,
            ))
            .ok_or(anyhow!(
                "No table of type {} was expected",
                TableType::Softmax(float_temp_bits, ceil_log2(lut.len()), *bkm,).name()
            ))?;
        lookups.extend(merged_softmax);

        let lookups = gen
            .new_lookups
            .get_mut(&TableType::ErrorTable(allowable_error))
            .ok_or(anyhow!(
                "No table of type {} was expected",
                TableType::ErrorTable(allowable_error).name()
            ))?;
        lookups.extend(normalisation_lookup);

        gen.logup_witnesses.insert(
            id,
            vec![
                LogUpWitness::<E, PCS>::new_lookup(
                    exp_commits,
                    exp_evals,
                    2,
                    TableType::Softmax(float_temp_bits, ceil_log2(lut.len()), *bkm),
                ),
                LogUpWitness::<E, PCS>::new_lookup(range_commits, range_evals, 1, TableType::Range),
                LogUpWitness::<E, PCS>::new_lookup(
                    vec![(shift_commit, shift_mle)],
                    vec![error_evals],
                    1,
                    TableType::ErrorTable(allowable_error),
                ),
            ],
        );
        Ok(())
    }
}

impl QuantizeOp for Softmax<f32> {
    type QuantizedOp = Softmax<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "More than one input scaling factor provided for Softmax. Received {} input scaling factor",
            input_scaling.len()
        );

        let quantised_op = self.quantise(input_scaling[0])?;

        let output_scaling = ScalingFactor::from_parts(
            1.0f32,
            0.0f32,
            1.0f32 / OUTPUT_SCALE_FACTOR as f32,
            (0i128, OUTPUT_SCALE_FACTOR as Element),
        );
        Ok(QuantizeOutput::<Softmax<Element>> {
            quantized_op: quantised_op,
            output_scalings: vec![output_scaling],
            requant_layer: None,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftmaxCtx {
    node_id: NodeId,
    /// The absoloute value of the allowable error
    allowable_error: Element,
    /// The value that determines when we map to zero in the exp lookup
    bkm: Element,
    /// The result of calling [`f32::to_bits`] on the temperature
    temperature_bits: u32,
    /// The number of variables used for the lookup table
    size: usize,
    /// The scalar multiplier used to ensure that the inputs have the correct scale factor
    scalar: Element,
}

impl SoftmaxCtx {
    /// Getter function to retrive the [`TableType`]
    pub(crate) fn table_type(&self) -> TableType {
        TableType::Softmax(self.temperature_bits, self.size, self.bkm)
    }
}

impl OpInfo for SoftmaxCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Softmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E> ProveInfo<E> for Softmax<Element>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        if let Some(quant_info) = self.quant_info() {
            let QuantisedSoftmaxData {
                lut,
                error_bound,
                float_temperature,
                bkm,
                ..
            } = quant_info;

            // We convert the `f32` to bits so that the compiler doesn't complain about trait implementations
            let float_temp_bits = float_temperature.to_bits();
            // Calculate the allowable error in normalisation as an Element
            let allowable_error = (*error_bound * OUTPUT_SCALE_FACTOR as f32).round() as Element;
            // Calculate the lookup table number of variables
            let size = ceil_log2(lut.len());
            // Add the tables that Softmax requires
            aux.tables.insert(TableType::Range);
            aux.tables
                .insert(TableType::Softmax(float_temp_bits, size, *bkm));
            aux.tables.insert(TableType::ErrorTable(allowable_error));

            // There are no common commitments for this layer
            aux.model_polys = None;

            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux

            Ok((
                LayerCtx::<E>::Softmax(SoftmaxCtx {
                    node_id: id,
                    allowable_error,
                    bkm: *bkm,
                    temperature_bits: float_temp_bits,
                    size,
                    scalar: self.scalar,
                }),
                aux,
            ))
        } else {
            return Err(anyhow!(
                "Softmax operation has not been quantised so no proving info available"
            ));
        }
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for SoftmaxCtx
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = SoftmaxProof<E, PCS>;
    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        // First we check that we only have one claim in `last_claims`
        ensure!(
            last_claims.len() == 1,
            "Softmax only outputs 1 claim, received {} while verifying Softmax step",
            last_claims.len()
        );

        let last_claim = last_claims[0];

        // Retrieve the challenges used in the lookup argument
        let table_type = self.table_type();
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .get_challenges_by_name(&table_type.name())
            .ok_or(anyhow!(
                "Couldn't get challenges for LookupType: {}",
                table_type.name()
            ))?;

        // First we verify the LogUp proofs
        let SoftmaxProof {
            exp_lookup,
            range_lookup,
            error_lookup,
            commitments,
            accumulation_proof,
            evaluations,
        } = proof;

        // Verify both lookup arguments in the same order they are proved.
        let exp_claims = verify_logup_proof(
            exp_lookup,
            1,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;
        let range_claims = verify_logup_proof(
            range_lookup,
            2,
            constant_challenge,
            E::ONE,
            verifier.transcript,
        )?;
        let error_claims = verify_logup_proof(
            error_lookup,
            1,
            constant_challenge,
            E::ONE,
            verifier.transcript,
        )?;

        // Now we squeeze the batching challenge
        let alpha = verifier
            .transcript
            .get_and_append_challenge(b"batching_challenge")
            .elements;

        // Recreate the initial evaluation of the sumcheck
        let (claimed_sum, _) = exp_claims
            .claims()
            .iter()
            .chain(range_claims.claims().iter())
            .chain(error_claims.claims().iter())
            .map(|claim| claim.eval)
            .chain(std::iter::once(last_claim.eval))
            .fold((E::ZERO, E::ONE), |(acc, chal_acc), eval| {
                (acc + chal_acc * eval, chal_acc * alpha)
            });

        let exp_point = exp_claims.point();
        let range_point = range_claims.point();
        let error_point = error_claims.point();

        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();

        let extra_vars = exp_point.len() - error_point.len();
        let two_mult = E::from_canonical_u64(1u64 << extra_vars);
        let full_error_point = std::iter::repeat_n(two_inv, extra_vars)
            .chain(error_point.iter().copied())
            .collect::<Vec<E>>();

        // Verify the sumcheck proof
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![exp_point.len(); 2]]);

        let sumcheck_subclaim = IOPVerifierState::<E>::verify(
            claimed_sum,
            accumulation_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = sumcheck_subclaim.point_flat();

        let last_claim_beta_eval = eq_xy_eval(&last_claim.point, &sumcheck_point);
        let exp_beta_eval = eq_xy_eval(exp_point, &sumcheck_point);
        let range_beta_eval = eq_xy_eval(range_point, &sumcheck_point);
        let error_beta_eval = eq_xy_eval(&full_error_point, &sumcheck_point);

        // The evaluations supplied by the prover are in the order exp_input, exp_output, low_range, high_range, shift
        ensure!(
            evaluations.len() == 5,
            "Expected 5 evaluations from the prover during Softmax verification, got {}",
            evaluations.len()
        );

        // Start to build the virtual polynomial, begin with exponential polys
        let (calc_subclaim, batch_challenge) = evaluations[..2]
            .iter()
            .fold((E::ZERO, E::ONE), |(sublcaim_acc, bc), &claim| {
                (sublcaim_acc + claim * bc, bc * alpha)
            });
        // Add range polys
        let (mut calc_subclaim, batch_challenge) = evaluations[2..4].iter().fold(
            (exp_beta_eval * calc_subclaim, batch_challenge),
            |(subclaim_acc, bc), &claim| (subclaim_acc + range_beta_eval * claim * bc, bc * alpha),
        );

        // Fianlly add the error check and the last claim, for this we need the output column of the exponential lookup
        let exp_output_claim = evaluations[1];

        let output_quantised_one: E = (OUTPUT_SCALE_FACTOR as Element).to_field();
        calc_subclaim += batch_challenge
            * error_beta_eval
            * (output_quantised_one - two_mult * exp_output_claim);
        calc_subclaim += batch_challenge * alpha * last_claim_beta_eval * exp_output_claim;

        ensure!(
            sumcheck_subclaim.expected_evaluation == calc_subclaim,
            "Sumcheck verification output claim did not match calculated claim in Softmax verification, expected: {:?}, calculated: {:?}",
            sumcheck_subclaim.expected_evaluation,
            calc_subclaim
        );

        // Now we work out the claim on the input to pass to the next layer
        let two_to_the_16 = E::from_canonical_u64(1u64 << 16);
        let two_to_the_8 = E::from_canonical_u64(1u64 << 8);
        let field_multiplier: E = self.scalar.to_field();
        let field_multiplier_inv = field_multiplier.inverse();
        let input_eval = -field_multiplier_inv
            * (evaluations[0] * two_to_the_16
                + evaluations[3] * two_to_the_8
                + evaluations[2]
                + evaluations[4]);

        // Add the commitments to the commitment verifier
        commitments
            .iter()
            .zip(evaluations.iter())
            .try_for_each(|(comm, &eval)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(comm.clone(), Claim::<E>::new(sumcheck_point.clone(), eval))
            })?;

        Ok(vec![Claim::<E>::new(sumcheck_point, input_eval)])
    }
}

#[cfg(test)]
mod tests {

    use ff_ext::GoldilocksExt2;

    use crate::{
        Tensor,
        layers::Layer,
        model::{Model, test::prove_model},
        padding::PaddingMode,
    };

    use super::*;

    #[test]
    fn test_softmax() {
        let softmax = Softmax::default();
        let input = Tensor::new(vec![2, 3].into(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3].into()])
            .unwrap();
        assert_eq!(output.outputs[0].get_shape(), vec![2, 3].into());
        // since we dont slice, sum of  prob should be equal to 1
        assert_eq!(output.outputs[0].get_data().iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn test_softmax_with_dim() {
        let softmax = Softmax::new().on_dim(1);
        let input = Tensor::random(&vec![2, 3, 4].into());
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3, 4].into()])
            .unwrap();
        let out = output.outputs()[0];
        assert_eq!(out.get_shape(), vec![2, 3, 4].into());
        let (slices, _) = out.slice_on_dim(1);
        let acceptable_range = 0.99..1.01;
        for slice in slices {
            assert!(
                acceptable_range.contains(&slice.iter().sum::<f32>()),
                "{:?}",
                out.get_data()
            );
        }
    }

    #[test]
    fn test_quantise() {
        // For now we test with GPT2 like parameters
        let scale = 1.0f32 / 768.0f32.sqrt();
        let softmax = Softmax::<f32>::new_with_scale(scale, 1024);

        for num_tokens in 1015..1025 {
            // Make random q and k vectors
            let test_q = Tensor::<f32>::random(&vec![num_tokens, 768].into());
            let test_k = Tensor::<f32>::random(&vec![768, num_tokens].into());

            let q_scaling = ScalingFactor::from_tensor(&test_q, None);
            let k_scaling = ScalingFactor::from_tensor(&test_k, None);

            // Pick the quantised domain to be Some((-1i128 << 24, 1i128 << 24)) since matrix multiplication on 768 columns adds at most 10 to the bit size
            // (already at bit size 14 before this due to multiplication of two 8 bit quant integers)
            let qk_scaling = ScalingFactor::from_scale(
                q_scaling.scale() * k_scaling.scale(),
                Some((-1i128 << 24, 1i128 << 24)),
            );

            let test_q_quant = test_q.clone().quantize(&q_scaling);
            let test_k_quant = test_k.clone().quantize(&k_scaling);

            let test_qk_quant = test_q_quant.matmul(&test_k_quant);

            let test_qk_dequant = test_qk_quant.dequantize(&qk_scaling);

            // Now to test the quantised softmax we quantise `float_input` and run the quantised evaluation.
            // We also quantise and dequantise `float_input` and run this data through the float evaluation and then compare the two results.

            let quant_softmax = softmax.quantise(qk_scaling).unwrap();

            // Obtain the quantised output
            let quant_output = quant_softmax
                .evaluate::<GoldilocksExt2>(
                    &[&test_qk_quant],
                    vec![vec![num_tokens, num_tokens].into()],
                )
                .unwrap();
            // The result of running the quantised input as floats
            let dequant_output = softmax
                .evaluate::<GoldilocksExt2>(
                    &[&test_qk_dequant],
                    vec![vec![num_tokens, num_tokens].into()],
                )
                .unwrap();

            for (q_chunk, f_chunk) in quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .zip(dequant_output.outputs[0].get_data().chunks(num_tokens))
            {
                for (&q, f) in q_chunk.iter().zip(f_chunk.iter()) {
                    let float_q = q as f32 / OUTPUT_SCALE_FACTOR as f32;

                    let quant_dequant_diff = (float_q - f).abs();

                    // Make sure we are always withing 1/100 th of the actual value
                    assert!(quant_dequant_diff < 0.01);
                }
            }

            let max_error =
                quant_softmax.quant_info.as_ref().unwrap().error_bound * OUTPUT_SCALE_FACTOR as f32;

            quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .for_each(|chunk| {
                    let row_sum = chunk.iter().sum::<Element>();

                    let diff_from_one = (row_sum - OUTPUT_SCALE_FACTOR as Element).abs();

                    assert!(diff_from_one < max_error.round() as Element);
                });
        }
    }

    #[test]
    fn test_softmax_with_scale() {
        let softmax = Softmax::new_with_scale(1.0 / 2.0, 1024);
        let input = Tensor::new(vec![2, 3].into(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3].into()])
            .unwrap();

        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0
            ]
        );
    }

    #[test]
    fn test_softmax_proving() {
        let input_shape = vec![200, 12, 200];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        let softmax = Softmax::<f32>::new_with_scale(1.0f32 / 768.0f32.sqrt(), 1024);

        let _ = model
            .add_consecutive_layer(Layer::Softmax(softmax), None)
            .unwrap();

        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }
}
