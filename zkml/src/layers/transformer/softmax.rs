//! This layer applies the softmax function to the last dimension of the input tensor
use core::f32;
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
        transformer::mha::eval_zeroifier_mle,
    },
    lookup::{
        context::{
            COLUMN_SEPARATOR, CommsAndEvals, CommsAndProofs, LookupWitnessGen, SoftmaxTableData,
            TableType, count_elements,
        },
        logup_gkr::{
            prover::batch_prove,
            structs::{LogUpProof, LogUpVerifierClaim},
            verifier::verify_logup_proof,
        },
        witness::LogUpWitness,
    },
    model::StepData,
    padding::PaddingMode,
    quantization::{Fieldizer, ScalingFactor},
    tensor::{Number, Shape},
    to_base,
};

use anyhow::{Result, anyhow, ensure};

use ark_std::Zero;
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use mpcs::{PolynomialCommitmentScheme, sum_check::eq_xy_eval};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
};
use p3_field::FieldAlgebra;
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};

/// The base 2 logarithm of the scale factor used in exponential lookup tables
pub(crate) const LOG_SCALE_FACTOR: usize = 24;
/// The scale factor for our fixed point arithmetic
pub(crate) const SCALE_FACTOR: usize = 1 << LOG_SCALE_FACTOR;
/// The scale factor of the outputs of the `exp` lookup
pub(crate) const OUTPUT_SCALE_FACTOR: usize = 1 << 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax<N> {
    // By default, it's equal to 1
    /// In the floating point case this is the factor we multiply by before exponentiating, when thought of as a Boltzmann distribution this is
    /// often referred to as the "Temperature".
    ///
    /// For the quantised version this is the factor we must rescale by in order to make use of the lookup table.
    pub scalar: N,
    /// This is the maximum size of dimension that we will normalise over. For example in an Attention layer this would be the maximum context size.
    max_size: usize,
    /// This is the extra information required to compute the quantised version, it defaults to [`None`].
    quant_info: Option<QuantisedSoftmaxData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
/// This struct is used to store information used when evaluating the quantised version of [`Softmax`] on
/// [`Element`]s.
struct QuantisedSoftmaxData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This stores the [`SoftmaxTableData`]
    lut: SoftmaxTableData,
    /// The error bound as calculated by the formulae given in the zkLLM paper
    error_bound: f32,
    /// This is the inverse of the float temperature for calculating row normalisation
    inv_float_temperature: f32,
    /// This value indicates the point that we map everything greater than this to zero
    bkm: Element,
    /// This value tells use how many chunks we need to make after the exp lookup chunk
    number_zero_chunks: usize,
    /// This value tells us how many variables the zeroing table has
    zero_table_vars: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`Softmax`] operation.
pub struct SoftmaxProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The LogUp proofs for Softmax, they are ordered `exp_lookup`, `range_lookup`, `error_lookup` and then `zero_table_lookup` if it exists
    pub(crate) logup_proofs: Vec<LogUpProof<E>>,
    /// Witness commitments for this layer
    pub(crate) commitments: Vec<PCS::Commitment>,
    /// The sumcheck proof we use to make sure everything is evaluated at the same point.
    pub(crate) accumulation_proof: IOPProof<E>,
    /// Sumcheck proof used to prove that the mask was applied correctly
    pub(crate) mask_proof: IOPProof<E>,
    /// The claimed evaluations of the commitments
    pub(crate) evaluations: Vec<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> SoftmaxProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        let (nums, denoms): (Vec<Vec<E>>, Vec<Vec<E>>) = self
            .logup_proofs
            .iter()
            .map(|p| p.fractional_outputs())
            .unzip();

        (nums.concat(), denoms.concat())
    }
}

impl<N: Number> Default for Softmax<N> {
    fn default() -> Self {
        Softmax {
            scalar: N::unit(),
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
            max_size: max_context_size,
            quant_info: None,
        }
    }
    pub fn quantise(&self, input_scaling: ScalingFactor) -> Result<Softmax<Element>> {
        // First we work out what we need to multiply by to get the input scale factor to be `SCALE_FACTOR`
        let input_scale_factor = input_scaling.scale();
        let temperature = self.scalar.to_f32()?;
        let inv_float_temperature = 1.0f32 / temperature;
        let multiplier = (SCALE_FACTOR as f32 * input_scale_factor).round() as Element;

        // We want to be able to cover all possible inputs, to do this we need to work out what the minimum quantised input is.
        // this can be calculated by taking `input_scaling.domain().0` and then subtracting the maximum possible shift for normalisation.
        let (quantised_min, _) = input_scaling.domain();

        // The maximum shift would be if every element in the row is `quantised_max`, in this case it can be calculated as
        // (-SCALE_FACTOR as f32) * (inv_float_temperature * (self.max_size as f32).ln() + input_scaling.max())
        let max_shift = (-(SCALE_FACTOR as f32)
            * (inv_float_temperature * (self.max_size as f32).ln() + input_scaling.max()))
        .round() as Element;

        // So the minimum possible input is `quantised_min * multiplier + max_shift`, we multiply by `multiplier` so everything has scaling factor `SCALE_FACTOR`.
        let min_softmax_input = quantised_min * multiplier + max_shift;

        // The smallest 16 bits of `min_softmax_input` relate to values that are so small that after exponentiating they are so close to 1 that we just map them all to 1.
        // Due to this the bottom 16 bits gets sliced off and are just range checked, so for the actual softmax input we only need `min_softmax_input >> 16`.
        let significant_min_input = min_softmax_input >> 16;

        // Now we work out how many bits it takes to represent this number (it will always be less than zero so we take an abs() first)
        let min_input_bits = ceil_log2(significant_min_input.unsigned_abs() as usize);

        // Now we want to work out the value "bkm" such that anything with absolute value greater than bkm should just be mapped to zero
        // by the exponential. We will have K total tables, L of which are used for values that are so insignificant they get mapped to 1 and M of which
        // contain values that are all greater than bkm. We aim to make K - M - L = 1 because results from testing tell us that this allows
        // us to make an exp table with 17 variables which isn't too large (as it gets reused across every softmax in something like Multiheaded attention).
        let base: Element = 1 << (LOG_SCALE_FACTOR - 8);
        let (float_error, bkm_float) = calc_softmax_error(
            base,
            self.max_size as f32,
            OUTPUT_SCALE_FACTOR as f32,
            SCALE_FACTOR as f32,
            inv_float_temperature,
        );

        let float_error = float_error.abs();
        let bkm = bkm_float.round() as Element;
        // Now that we have bkm we set the Softmax table size as `ceil_log2(bkm as usize >> 16)` (which is 17 in practice)
        let softmax_table_size = ceil_log2(bkm as usize >> 16);
        // We also work out how many additional chunks we need to cover anything between bkm >> 16 and significant_min_input
        let (number_zero_chunks, zero_table_vars) = if min_input_bits > softmax_table_size {
            let remaining_bits = min_input_bits - softmax_table_size;
            // Here we ceiling divide
            let number_chunks = (remaining_bits - 1) / softmax_table_size + 1;
            // If number of tables is 1 we check to see if we can use < softmax_table_size bits
            let zeroing_table_bit_size = if number_chunks == 1 {
                remaining_bits
            } else {
                softmax_table_size
            };
            (number_chunks, zeroing_table_bit_size)
        } else {
            (0usize, 0usize)
        };

        // Make the exp lookup table
        let table_data =
            SoftmaxTableData::new(inv_float_temperature.to_bits(), softmax_table_size, bkm);

        // Store all the quantised info for quantised evaluation
        let quant_info = QuantisedSoftmaxData {
            input_scale_factor: input_scaling,
            lut: table_data,
            error_bound: float_error,
            inv_float_temperature,
            bkm,
            number_zero_chunks,
            zero_table_vars,
        };

        // Return the quantised `Softmax` operator
        Ok(Softmax::<Element> {
            scalar: multiplier,
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
}

impl Softmax<Element> {
    /// Method that given a quantised input [`Tensor`] calculates the `shift` we apply along each dim and returns the result as the `bias` field of
    /// as [`AttentionMask`].
    pub(crate) fn calculate_shift_data(
        &self,
        input: &Tensor<Element>,
        unpadded_input_shape: &[usize],
    ) -> Result<(Tensor<Element>, AttentionMask<Element>)> {
        let QuantisedSoftmaxData {
            input_scale_factor,
            inv_float_temperature,
            bkm,
            ..
        } = self.quant_info().ok_or(anyhow!("Attempted to calculate shift data for quantised Softmax with no QuantisedSoftmaxData present"))?;

        // We need to calculate the shift we should apply together with the mask
        // To do this we:
        // 1. dequantise the input
        // 2. apply a float mask
        // 3. sum along the desired dim
        let negative_infinity = -((bkm >> 16) + 1) << 16;

        // New way is calculate shift row by row (as if a mask is being used)
        // apply shift
        // apply mask

        // We need a mask
        let final_dim = *input
            .shape
            .last()
            .ok_or(anyhow!("Input tensor had no shape in quantised Softmax"))?;
        // We also need the second to last dim
        let second_dim = input.shape[input.shape.len() - 2];
        let shift_data = input
            .get_data()
            .chunks(final_dim)
            .enumerate()
            .map(|(i, chunk)| {
                // We add the check here to see if we are in the first row of a new channel, the first row has to be calculated
                // differently so as to avoid getting rounding errors that lead to values we can't lookup.
                if i % second_dim == 0 {
                    -chunk[0] * self.scalar
                } else {
                    let max = *chunk.iter().take(i % second_dim + 1).max().unwrap();
                    let sum = chunk
                        .iter()
                        .take(i % second_dim + 1)
                        .map(|x| {
                            (input_scale_factor.dequantize(&(x - max)) / inv_float_temperature)
                                .exp()
                        })
                        .sum::<f32>();
                    let log_sum = sum.ln();
                    -(SCALE_FACTOR as f32 * inv_float_temperature * log_sum).round() as Element
                        - max * self.scalar
                }
            })
            .collect::<Vec<Element>>();
        // Make a tensor for the shift data
        let shift_shape = input
            .shape
            .iter()
            .take(unpadded_input_shape.len() - 1)
            .copied()
            .chain(std::iter::once(1usize))
            .collect::<Vec<usize>>();
        let shift_tensor = Tensor::<Element>::new(shift_shape.into(), shift_data);
        let mask = AttentionMask::<Element>::new(input.shape.as_slice(), negative_infinity)?;

        Ok((shift_tensor, mask))
    }
}

/// Calculates the error as an [`f32`] when applying softmax as described in zkLLM.
/// This functions returns the error together with the value `bkm` such that anything smaller
/// than `bkm` should be mapped to zero.
pub(crate) fn calc_softmax_error(
    bl: Element,
    max_context_size: f32,
    output_sf: f32,
    input_sf: f32,
    temp: f32,
) -> (f32, f32) {
    // First we calculate the optimal point to map everything to zero (to minimise the L1 error)
    // we assume the total number of tables that don't map everything to 1 or 0 is exactly 1.
    let kml = 1.0f32;
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
    let term_two = (max_context_size - 1.0f32) * (-bkm / input_sf * temp).exp();
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
        // Make the attention mask
        let mask = AttentionMask::<f32>::new(&input.shape, f32::NEG_INFINITY)?;
        let masked_input = mask.apply(input)?;

        let chunk_size = *input
            .shape
            .last()
            .ok_or(anyhow!("Input shape was empty for float Softmax"))?;
        let output = masked_input
            .get_data()
            .chunks(chunk_size)
            .flat_map(|vec| {
                let max: f32 = *vec
                    .iter()
                    .max_by(|i, j| i.partial_cmp(j).unwrap_or(std::cmp::Ordering::Less))
                    .unwrap();
                let scaled = vec
                    .iter()
                    .map(|x| {
                        if *x != f32::NEG_INFINITY {
                            self.scalar * (x - max)
                        } else {
                            *x
                        }
                    })
                    .map(|x| x.exp())
                    .collect::<Vec<_>>();
                let sum = scaled.iter().sum::<f32>();
                scaled.iter().map(|x| x / sum).collect::<Vec<_>>()
            })
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

#[derive(Debug, Clone)]
/// Struct containing data useful for proving correctness of [`Softmax`]. This is data that we compute anyway
/// during quantised evaluation.
pub struct SoftmaxData<E>
where
    E: Clone + ExtensionField,
{
    /// This is the natural logarithm of the sum of the exponentiated input along the given dimension
    shift_tensor: Tensor<Element>,
    /// This is the input tensor after applying the shift
    shifted_input: Tensor<Element>,
    /// This is the mask used during the attention process
    mask: AttentionMask<Element>,
    /// The lowest 8-bits of the input (after rescaling)
    low_range_check: Vec<Element>,
    /// The second lowest 8 bits of the input (after rescaling)
    high_range_check: Vec<Element>,
    /// The inputs and outputs of the exponential lookup table
    exp_lookup: (Vec<Element>, Vec<Element>),
    /// The inputs and outputs of the most significant chunks lookups
    zero_table_lookups: (Vec<Vec<Element>>, Vec<Vec<Element>>),
    _phantom: PhantomData<E>,
}

impl<E: Clone + ExtensionField> Default for SoftmaxData<E> {
    fn default() -> Self {
        Self {
            shift_tensor: Tensor::<Element>::new(vec![].into(), vec![]),
            shifted_input: Tensor::<Element>::new(vec![].into(), vec![]),
            mask: AttentionMask::<Element>::default(),
            low_range_check: Vec::default(),
            high_range_check: Vec::default(),
            exp_lookup: (Vec::default(), Vec::default()),
            zero_table_lookups: (Vec::default(), Vec::default()),
            _phantom: PhantomData::<E>,
        }
    }
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
            "Expected a single input to quantised softmax, got: {}",
            inputs.len()
        );

        // Since we have checked that quant info exists this unwrap is safe
        let QuantisedSoftmaxData {
            lut,
            number_zero_chunks,
            zero_table_vars,
            bkm,
            ..
        } = self.quant_info().unwrap();

        let input = inputs[0];
        let (shift_tensor, mask) = self.calculate_shift_data(input, &unpadded_input_shapes[0])?;

        let dim = *input.shape.last().ok_or(anyhow!(
            "Softmax input had no shape in quantised evaluation"
        ))?;
        let shifted_input_data = input
            .get_data()
            .chunks(dim)
            .zip(shift_tensor.get_data().iter())
            .flat_map(|(row, shift)| {
                // For each row we rescale the input to the correct scale factor and add the shift (its already been negated)
                row.iter()
                    .map(|elem| elem * self.scalar + shift)
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>();

        let shifted_input = Tensor::<Element>::new(input.get_shape(), shifted_input_data);
        // Apply the mask to the shifted input
        let masked_input = mask.apply(&shifted_input)?;

        // We use the mask to extract 8-bit chunks of the input, these are the smallest fractional bits
        // and so we can assume that they get mapped to 1 under `exp`
        let bit_mask: Element = 255;
        let softmax_table_vars = ceil_log2(*bkm as usize >> 16);
        let softmax_table_mask: Element = (1 << softmax_table_vars) - 1;
        let zero_table_mask: Element = (1 << *zero_table_vars) - 1;
        // Now we chunk the rescaled, masked input
        let mut low_range_check = Vec::<Element>::new();
        let mut high_range_check = Vec::<Element>::new();
        let mut lookups = Vec::<Element>::new();
        let mut outputs = Vec::<Element>::new();
        let mut zero_chunks_in: Vec<Vec<Element>> = vec![vec![]; *number_zero_chunks];
        let mut zero_chunks_out: Vec<Vec<Element>> = vec![vec![]; *number_zero_chunks];
        let mut softmax_outputs: Vec<Element> = Vec::<Element>::new();

        for input_elem in masked_input.get_data().iter() {
            // We take the absolute value as this is guaranteed to be negative or zero
            let mut rescaled = input_elem.abs();
            low_range_check.push(rescaled & bit_mask);
            rescaled >>= 8;
            high_range_check.push(rescaled & bit_mask);
            rescaled >>= 8;
            let lookup = rescaled & softmax_table_mask;
            let exp_output = lut.table_output(lookup);
            outputs.push(exp_output);
            lookups.push(lookup);
            rescaled >>= softmax_table_vars;
            // Now we iterate over the number of zero chunks, if any of these are non-zero the output of softmax should be 0 for this element.
            // We fold with initial input exp_output, at each step we append the zero chunk lookup values to their respective lists.
            let softmax_output = zero_chunks_in
                .iter_mut()
                .zip(zero_chunks_out.iter_mut())
                .fold(exp_output, |acc, (in_vec, out_vec)| {
                    let in_lookup = rescaled & zero_table_mask;
                    let out_lookup: Element = if in_lookup != 0 { 0 } else { 1 };
                    in_vec.push(in_lookup);
                    out_vec.push(out_lookup);
                    rescaled >>= *zero_table_vars;
                    acc * out_lookup
                });
            softmax_outputs.push(softmax_output);
        }

        // We store all the information that has been computed in this step that will be useful later for proving.
        let proving_data = ProvingData::Softmax(SoftmaxData {
            shift_tensor,
            shifted_input,
            mask,
            low_range_check,
            high_range_check,
            exp_lookup: (lookups, outputs),
            zero_table_lookups: (zero_chunks_in, zero_chunks_out),
            _phantom: PhantomData::<E>,
        });

        // Make the output tensor
        let output = Tensor::<Element>::new(input.get_shape(), softmax_outputs);

        Ok(LayerOut {
            outputs: vec![output],
            proving_data,
        })
    }
}

impl PadOp for Softmax<Element> {}

impl Softmax<Element> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn prove_step<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: transcript::Transcript<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        softmax_data: &SoftmaxData<E>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<(Vec<Claim<E>>, SoftmaxProof<E, PCS>)> {
        // Check we have the correct number of claims
        ensure!(
            last_claims.len() == 1,
            "Softmax only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];

        let logup_witnesses = prover.lookup_witness(node_id)?;
        // Nw we need to know if we have any zero table lookups
        let zero_table_lookups = self
            .quant_info()
            .map(|quant_info| !quant_info.number_zero_chunks.is_zero())
            .ok_or(anyhow!(
                "No Quant Info available for Softmax, unable to prove"
            ))?;
        // Check that we have the correct number of witnesses for Softmax
        let num_witnesses = if zero_table_lookups { 4 } else { 3 };
        if logup_witnesses.len() != num_witnesses {
            return Err(anyhow!(
                "There should be four lookup witnesses during Softmax proving, node: {}, number of witnesses: {}",
                node_id,
                logup_witnesses.len()
            ));
        }
        // Run the lookup protocol and return the lookup proof
        let (commits, logup_proofs): CommsAndProofs<PCS, E> = logup_witnesses
            .into_iter()
            .map(|logup_wit| {
                let logup_input = logup_wit.get_logup_input(&prover.challenge_storage)?;
                let commits = logup_wit.into_commitments();
                let logup_proof = batch_prove(&logup_input, prover.transcript)?;
                Ok((commits, logup_proof))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // We have checked that there are at least three logup proofs and we expect them to be in the order exp, range, error (zero if it exists)
        let exp_claims = logup_proofs[0].output_claims();
        let range_claims = logup_proofs[1].output_claims();
        let error_claims = logup_proofs[2].output_claims();

        let exp_point = exp_claims
            .first()
            .map(|claim| &claim.point)
            .ok_or(anyhow!("Exponential lookup in Softmax should have claims"))?;
        let range_point = range_claims
            .first()
            .map(|claim| &claim.point)
            .ok_or(anyhow!("Range lookup in Softmax should have claims"))?;
        let error_point = error_claims
            .first()
            .map(|claim| &claim.point)
            .ok_or(anyhow!("Error lookup in Softmax should have claims"))?;

        // We use the difference in point length between the error point and the exp point to work out how many variables correspond to the normalisation dimension
        let extra_vars = exp_point.len() - error_point.len();

        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();
        let two_mult = E::from_canonical_u64(1u64 << extra_vars);
        // We chain 2^-1 in all the variables that correspond to the row, that way in the sumcheck if we multiply by 2^extra_vars we end up with
        // a polynomial that has evaluations equal to the sum of the rows of exp_output (which should all be within the allowable error of quantised one).
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

        // Start to build the virtual polynomial, begin with exponential polys. This Virtual Polynomial is used because currently
        // the different polynomials that make up the input are all being evaluated at different points, in order to recombine them we need them to
        // be evaluated at the same point. To do this we use a random linear combination of the polynomials and multily each by eq(eval_point,x) so that
        // the initial sum is the same random linear combination of the evaluations we currently have (and the verifier has access to).
        let (vp, batch_challenge) = commits[0].iter().fold(
            (VirtualPolynomial::<E>::new(exp_point.len()), E::ONE),
            |(mut vp_acc, bc), (_, poly)| {
                vp_acc.add_mle_list(vec![poly.clone().into(), exp_beta.clone()], bc);
                (vp_acc, bc * alpha)
            },
        );
        // Add range polys
        let (mut vp, batch_challenge) =
            commits[1]
                .iter()
                .fold((vp, batch_challenge), |(mut vp_acc, bc), (_, poly)| {
                    vp_acc.add_mle_list(vec![poly.clone().into(), range_beta.clone()], bc);
                    (vp_acc, bc * alpha)
                });

        // Finally add the error check and the last claim, for this we need the output column of the exponential lookup
        let (_, exp_output) = commits[0]
            .last()
            .ok_or(anyhow!("Exponential lookup in Softmax had no commitments"))?;

        // If zero table lookups exists we need to extract their data and them to the sumcheck (as we want all our Claims to be on the same point)
        if logup_proofs.len() == 4 {
            let zero_table_claims = logup_proofs[3].output_claims();

            let zero_table_point =
                zero_table_claims
                    .first()
                    .map(|claim| &claim.point)
                    .ok_or(anyhow!(
                        "Zero Table lookup in Softmax should have claims as we have 4 LogUp proofs"
                    ))?;

            let zero_table_beta: ArcMultilinearExtension<E> =
                compute_betas_eval(zero_table_point).into_mle().into();

            // Now add all the zero table polys to the sumcheck
            let batch_challenge = commits[3]
                .iter()
                .fold(batch_challenge, |chal_acc, (_, poly)| {
                    vp.add_mle_list(vec![zero_table_beta.clone(), poly.clone().into()], chal_acc);
                    chal_acc * alpha
                });

            // Finally we prove that the last claim is the product of all the zero table outputs and the exp output
            let mut layer_out_prod = commits[3]
                .iter()
                .skip(1)
                .step_by(2)
                .map(|(_, poly)| ArcMultilinearExtension::<E>::from(poly.clone()))
                .collect::<Vec<_>>();
            layer_out_prod.push(exp_output.clone().into());
            let mut error_poly = layer_out_prod.clone();
            layer_out_prod.push(last_claim_beta);
            error_poly.push(error_beta);

            vp.add_mle_list(error_poly, batch_challenge * two_mult);
            vp.add_mle_list(layer_out_prod, batch_challenge * alpha);
        } else {
            // In this case the layer output is just the exp table output
            vp.add_mle_list(
                vec![exp_output.clone().into(), error_beta.clone()],
                batch_challenge * two_mult,
            );
            vp.add_mle_list(
                vec![exp_output.clone().into(), last_claim_beta],
                batch_challenge * alpha,
            );
        }

        // Run the sumcheck proof
        #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);
        // We need the point and all the poly evals (excluding beta polys)
        let sumcheck_point = &sumcheck_proof.point;
        let all_evals = state.get_mle_final_evaluations();
        let exp_evals = &[all_evals[0], all_evals[2]];
        let range_evals = &[all_evals[3], all_evals[5]];

        // Now we need to make a sumcheck proof that shows that `input_eval` is the result of applying
        // the `tril` part of the mask to the input
        let mask_eq: ArcMultilinearExtension<E> =
            compute_betas_eval(sumcheck_point).into_mle().into();
        // Pad the shift data if needed
        let mut mask = softmax_data.mask.clone();
        mask.pad()?;
        let shifted_input = softmax_data.shifted_input.pad_next_power_of_two();

        let tril_mle: ArcMultilinearExtension<E> =
            to_base::<E, _>(mask.tril.get_data()).into_mle().into();
        let bias_mle: ArcMultilinearExtension<E> =
            to_base::<E, _>(mask.bias.get_data()).into_mle().into();
        let shifted_input_mle: ArcMultilinearExtension<E> =
            to_base::<E, _>(shifted_input.get_data()).into_mle().into();

        // Make the VirtualPolynomial to prove that the mask was applied correctly. This Virtual Polynomial is
        // eq(sumcheck_point,x) * (shifted_input_mle * mask.tril_mle + mask.bias_mle).
        let mut vp = VirtualPolynomial::<E>::new(tril_mle.num_vars());
        vp.add_mle_list(vec![shifted_input_mle, tril_mle], E::ONE);
        vp.add_mle_list(vec![bias_mle], E::ONE);
        vp.mul_by_mle(mask_eq, E::BaseField::ONE);
        // Run the sumcheck proof for masking
        #[allow(deprecated)]
        let (mask_proof, mask_state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        let shifted_input_eval = mask_state.get_mle_final_evaluations()[0];

        // Work out the difference in length between the mask proof point and the number of variables for the shift commitment.
        let mask_point_len = mask_proof.point.len();
        let initial_shift = &commits[2];
        let shift_vars = initial_shift[0].1.num_vars;
        let vars_diff = mask_point_len - shift_vars;
        let shift_point = mask_proof
            .point
            .iter()
            .skip(vars_diff)
            .copied()
            .collect::<Vec<E>>();
        let shift_eval = initial_shift[0].1.evaluate(&shift_point);

        let input_claim =
            Claim::<E>::new(mask_proof.point.clone(), shifted_input_eval - shift_eval);
        // Add the commitments to be opened to the commitment prover
        let exp_commits = commits[0]
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

        let range_commits = commits[1]
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
            Claim::<E>::new(shift_point, shift_eval),
        )?;
        let shift_commit = PCS::get_pure_commitment(&initial_shift[0].0);

        // Now if we have 4 logup proofs we need to also deal with the zero table evaluations
        let (commitments, evaluations) = if logup_proofs.len() == 4 {
            // Extract the zero table lookup evaluations
            let zero_table_evals = all_evals
                .iter()
                .skip(7)
                .take(commits[3].len())
                .copied()
                .collect::<Vec<E>>();

            let zero_table_commits = commits[3]
                .iter()
                .zip(zero_table_evals.iter())
                .map(|(comm_with_wit, eval)| {
                    let comm = PCS::get_pure_commitment(&comm_with_wit.0);
                    prover.commit_prover.add_witness_claim(
                        comm_with_wit.clone(),
                        Claim::<E>::new(sumcheck_point.clone(), *eval),
                    )?;
                    Ok(comm)
                })
                .collect::<Result<Vec<PCS::Commitment>, anyhow::Error>>()?;

            let commitments = exp_commits
                .into_iter()
                .chain(range_commits)
                .chain(std::iter::once(shift_commit))
                .chain(zero_table_commits)
                .collect::<Vec<PCS::Commitment>>();

            let evaluations = exp_evals
                .iter()
                .chain(range_evals)
                .copied()
                .chain(std::iter::once(shift_eval))
                .chain(zero_table_evals)
                .collect::<Vec<E>>();
            (commitments, evaluations)
        } else {
            let commitments = exp_commits
                .into_iter()
                .chain(range_commits)
                .chain(std::iter::once(shift_commit))
                .collect::<Vec<PCS::Commitment>>();

            let evaluations = exp_evals
                .iter()
                .chain(range_evals)
                .copied()
                .chain(std::iter::once(shift_eval))
                .collect::<Vec<E>>();
            (commitments, evaluations)
        };

        let proof = SoftmaxProof::<E, PCS> {
            logup_proofs,
            commitments,
            accumulation_proof: sumcheck_proof,
            mask_proof,
            evaluations,
        };

        Ok((vec![input_claim], proof))
    }

    pub(crate) fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &Context<E, PCS>,
        output: &Tensor<Element>,
        softmax_data: &SoftmaxData<E>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        // Get the data generated during quantised evaluation
        let SoftmaxData {
            shift_tensor,
            low_range_check,
            high_range_check,
            exp_lookup: (exp_input, exp_output),
            zero_table_lookups: (zero_in, zero_out),
            ..
        } = softmax_data;
        let num_vars = ceil_log2(exp_input.len());

        // We need to work out how many chunks to split the normalisation into to be range checked.
        let QuantisedSoftmaxData {
            error_bound,
            lut,
            zero_table_vars,
            number_zero_chunks,
            ..
        } = self.quant_info().ok_or(anyhow!(
            "Could not prove Softmax because it had no quantisation data"
        ))?;
        let allowable_error = (*error_bound * OUTPUT_SCALE_FACTOR as f32).round() as Element;

        // Now we construct the polynomials used in the lookups
        // To do this we need the size of the last dimension
        let final_dim_size = *output
            .get_shape()
            .last()
            .ok_or(anyhow!("Softmax output tensor did not have a shape"))?;
        let normalisation_lookup = output
            .get_data()
            .chunks(final_dim_size)
            .map(|chunk| chunk.iter().sum::<Element>())
            .collect::<Vec<Element>>();

        let range_elements_count = count_elements(
            low_range_check
                .iter()
                .chain(high_range_check.iter())
                .cloned(),
        );
        let softman_elements_count = count_elements(
            exp_input
                .iter()
                .zip(exp_output.iter())
                .map(|(input, output)| input + output * COLUMN_SEPARATOR),
        );

        // We add zero table lookups if there are any
        let poly_evals_vec = if !number_zero_chunks.is_zero() {
            [exp_input, exp_output, low_range_check, high_range_check]
                .into_iter()
                .chain(zero_in)
                .chain(zero_out)
                .collect::<Vec<_>>()
        } else {
            vec![exp_input, exp_output, low_range_check, high_range_check]
        };

        // Make the commitments to the lookups
        let (commits, evals): CommsAndEvals<PCS, E> = poly_evals_vec
            .into_par_iter()
            .map(|vals| {
                let evaluations = to_base::<E, _>(vals);
                let mle =
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, &evaluations);
                let commit = ctx.commitment_ctx.commit(&mle)?;
                Ok(((commit, mle), evaluations))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // Split the commitments into the exp part, the range part and the zero table part
        let (exp_commits, other_commits) = commits.split_at(2);
        let (exp_evals, other_evals) = evals.split_at(2);
        let (range_commits, other_commits) = other_commits.split_at(2);
        let (range_evals, other_evals) = other_evals.split_at(2);

        // For the error we actually use the exp output table commitment so here we only need to make the evaluations
        // but we will store the `shift` polynomial and its commitment in the `LogUpWitness` that we create
        let error_evals = to_base::<E, _>(&normalisation_lookup);

        // Pad the shift data if needed
        let shift_tensor = shift_tensor.pad_next_power_of_two();

        let shift_data = shift_tensor.get_data();

        let shift_mle = DenseMultilinearExtension::<E>::from_evaluations_vec(
            ceil_log2(shift_data.len()),
            to_base::<E, _>(shift_data),
        );
        let shift_commit = ctx.commitment_ctx.commit(&shift_mle)?;

        let mut gen = LookupWitnessGen::<E, PCS>::default();

        // Add the looked up values to the generator so we can make multiplicity polys later
        gen.element_count
            .insert(TableType::Range, range_elements_count);

        // Need to recreate the parameters for the Softmax table
        gen.element_count
            .insert(TableType::Softmax(*lut), softman_elements_count);

        let quant_one = OUTPUT_SCALE_FACTOR as Element;
        gen.element_count.insert(
            TableType::ErrorTable(quant_one, allowable_error),
            count_elements(normalisation_lookup),
        );

        let mut lookup_witnesses = vec![
            LogUpWitness::<E, PCS>::new_lookup(
                exp_commits.to_vec(),
                exp_evals.to_vec(),
                2,
                TableType::Softmax(*lut),
            ),
            LogUpWitness::<E, PCS>::new_lookup(
                range_commits.to_vec(),
                range_evals.to_vec(),
                1,
                TableType::Range,
            ),
            LogUpWitness::<E, PCS>::new_lookup(
                vec![(shift_commit, shift_mle)],
                vec![error_evals],
                1,
                TableType::ErrorTable(quant_one, allowable_error),
            ),
        ];

        if !number_zero_chunks.is_zero() {
            let remaining = other_commits.len();
            let (zero_in_commits, zero_out_commits) = other_commits.split_at(remaining / 2);
            let (zero_in_evals, zero_out_evals) = other_evals.split_at(remaining / 2);

            let zero_table_lookup_commits = zero_in_commits
                .iter()
                .interleave(zero_out_commits.iter())
                .cloned()
                .collect::<Vec<_>>();
            let zero_table_lookup_evals = zero_in_evals
                .iter()
                .interleave(zero_out_evals.iter())
                .cloned()
                .collect::<Vec<_>>();

            let zero_table_elements_count = count_elements(
                zero_in
                    .iter()
                    .zip(zero_out.iter())
                    .flat_map(|(input, output)| input.iter().zip(output.iter()))
                    .map(|(input, output)| input + output * COLUMN_SEPARATOR),
            );

            gen.element_count.insert(
                TableType::ZeroTable(*zero_table_vars),
                zero_table_elements_count,
            );
            lookup_witnesses.push(LogUpWitness::<E, PCS>::new_lookup(
                zero_table_lookup_commits,
                zero_table_lookup_evals,
                2,
                TableType::ZeroTable(*zero_table_vars),
            ));
        }

        gen.logup_witnesses.insert(id, lookup_witnesses);
        Ok(gen)
    }
}

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
        step_data: &StepData<E, E>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        let softmax_data = step_data.outputs.try_softmax_data().ok_or(anyhow!(
            "Softmax LayerOut didn't have any ProvingData::Softmax"
        ))?;

        let (claims, proof) = self.prove_step(node_id, last_claims, softmax_data, prover)?;

        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<E, PCS>::Softmax(proof));

        Ok(claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        ensure!(
            step_data.inputs.len() == 1,
            "Found more than 1 input in inference step of Softmax layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output in inference step of Softmax layer"
        );
        let softmax_data = step_data.outputs.try_softmax_data().ok_or(anyhow!(
            "Softmax data not found in inference step for Sopftmax layer"
        ))?;
        self.lookup_witness(id, ctx, step_data.outputs.outputs()[0], softmax_data)
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

        // We want to keep track of the min and max output from this layer in floats. Softmax has to output values between 0.0 and 1.0
        // so we set max and min to these values. The scale is `1 / OUTPUT_SCALE_FACTOR` as this is what we multiply by to dequantise the quantised
        // outputs and the quantised domain is `(0.0 / scale, 1.0/ scale)`.
        let output_scaling = ScalingFactor::from_parts(
            1.0f32,
            0.0f32,
            1.0f32 / OUTPUT_SCALE_FACTOR as f32,
            (0, OUTPUT_SCALE_FACTOR as Element),
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
    /// The absolute value of the allowable error
    allowable_error: Element,
    /// The value that determines when we map to zero in the exp lookup
    bkm: Element,
    /// The result of calling [`f32::to_bits`] on the temperature
    temperature_bits: u32,
    /// The number of variables used for the lookup table
    size: usize,
    /// The scalar multiplier used to ensure that the inputs have the correct scale factor
    scalar: Element,
    /// The number of lookups into the zero table
    number_zero_chunks: usize,
    /// The number of bits the zero table size is
    zero_table_vars: usize,
}

impl SoftmaxCtx {
    /// Getter function to retrieve the [`TableType`] for the Softmax table.
    pub(crate) fn softmax_table(&self) -> TableType {
        TableType::Softmax(SoftmaxTableData::new(
            self.temperature_bits,
            self.size,
            self.bkm,
        ))
    }
}

impl OpInfo for SoftmaxCtx {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
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
                inv_float_temperature,
                bkm,
                number_zero_chunks,
                zero_table_vars,
                ..
            } = quant_info;

            // We convert the `f32` to bits so that the compiler doesn't complain about trait implementations
            let float_temp_bits = inv_float_temperature.to_bits();
            // Calculate the allowable error in normalisation as an Element
            let allowable_error = (*error_bound * OUTPUT_SCALE_FACTOR as f32).round() as Element;

            // Add the tables that Softmax requires
            aux.tables.insert(TableType::Range);
            aux.tables.insert(TableType::Softmax(*lut));
            aux.tables.insert(TableType::ErrorTable(
                OUTPUT_SCALE_FACTOR as Element,
                allowable_error,
            ));

            // If there is one add the ZeroTable
            if !zero_table_vars.is_zero() {
                aux.tables.insert(TableType::ZeroTable(*zero_table_vars));
            }

            // There are no common commitments for this layer
            aux.model_polys = None;
            aux.max_poly_len = aux
                .last_output_shape
                .iter()
                .fold(aux.max_poly_len, |acc, shapes| {
                    acc.max(shapes.next_power_of_two().product())
                });

            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::<E>::Softmax(SoftmaxCtx {
                    node_id: id,
                    allowable_error,
                    bkm: *bkm,
                    temperature_bits: float_temp_bits,
                    size: lut.full_table_size() as usize,
                    scalar: self.scalar,
                    number_zero_chunks: *number_zero_chunks,
                    zero_table_vars: *zero_table_vars,
                }),
                aux,
            ))
        } else {
            Err(anyhow!(
                "Softmax operation has not been quantised so no proving info available"
            ))
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
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        // First we check that we only have one claim in `last_claims`
        ensure!(
            last_claims.len() == 1,
            "Softmax only outputs 1 claim, received {} while verifying Softmax step",
            last_claims.len()
        );

        let last_claim = last_claims[0];
        // First we verify the LogUp proofs
        // Retrieve the challenges used in the logup proofs
        let logup_challenges: Vec<(E, E)> = if !self.zero_table_vars.is_zero() {
            [
                self.softmax_table(),
                TableType::Range,
                TableType::ErrorTable(OUTPUT_SCALE_FACTOR as Element, self.allowable_error),
                TableType::ZeroTable(self.zero_table_vars),
            ]
            .into_iter()
            .map(|table_type| {
                verifier
                    .challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges for LookupType: {}",
                        table_type.name()
                    ))
            })
            .collect::<Result<Vec<(E, E)>, anyhow::Error>>()?
        } else {
            [
                self.softmax_table(),
                TableType::Range,
                TableType::ErrorTable(OUTPUT_SCALE_FACTOR as Element, self.allowable_error),
            ]
            .into_iter()
            .map(|table_type| {
                verifier
                    .challenge_storage
                    .get_challenges_by_name(&table_type.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges for LookupType: {}",
                        table_type.name()
                    ))
            })
            .collect::<Result<Vec<(E, E)>, anyhow::Error>>()?
        };
        // We also need the number of instances per proof
        let instances_per_proof = if !self.number_zero_chunks.is_zero() {
            vec![1, 2, 1, self.number_zero_chunks]
        } else {
            vec![1, 2, 1]
        };

        let SoftmaxProof {
            logup_proofs,
            commitments,
            accumulation_proof,
            mask_proof,
            evaluations,
        } = proof;

        let logup_claims = izip!(
            logup_proofs.iter(),
            logup_challenges.into_iter(),
            instances_per_proof.into_iter()
        )
        .map(|(p, (const_chal, column_sep), num_instances)| {
            verify_logup_proof(
                p,
                num_instances,
                const_chal,
                column_sep,
                verifier.transcript,
            )
            .map_err(|e| e.into())
        })
        .collect::<Result<Vec<LogUpVerifierClaim<E>>, anyhow::Error>>()?;

        // We know we have at least 3 items in `logup_claims`
        let exp_claims = &logup_claims[0];
        let range_claims = &logup_claims[1];
        let error_claims = &logup_claims[2];

        // Now we squeeze the batching challenge
        let alpha = verifier
            .transcript
            .get_and_append_challenge(b"batching_challenge")
            .elements;

        // Recreate the initial evaluation of the sumcheck
        let (partial_claimed_sum, batch_challenge) = exp_claims
            .claims()
            .iter()
            .chain(range_claims.claims().iter())
            .fold((E::ZERO, E::ONE), |(acc, chal_acc), claim| {
                (acc + chal_acc * claim.eval, chal_acc * alpha)
            });

        // If we have zero table lookups add them here
        let claimed_sum = if logup_claims.len() == 4 {
            let (sum, bc) = logup_claims[3]
                .claims()
                .iter()
                .chain(error_claims.claims().iter())
                .fold(
                    (partial_claimed_sum, batch_challenge),
                    |(acc, chal_acc), claim| (acc + chal_acc * claim.eval, chal_acc * alpha),
                );
            sum + bc * last_claim.eval
        } else {
            let (sum, bc) = error_claims.claims().iter().fold(
                (partial_claimed_sum, batch_challenge),
                |(acc, chal_acc), claim| (acc + chal_acc * claim.eval, chal_acc * alpha),
            );
            sum + bc * last_claim.eval
        };

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
        let aux_info = if logup_claims.len() == 4 {
            VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                exp_point.len();
                self.number_zero_chunks + 2
            ]])
        } else {
            VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![exp_point.len(); 2]])
        };

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

        // The evaluations supplied by the prover are in the order exp_input, exp_output, low_range, high_range, shift and then pairs (zero_in, zero_out) self.number_zero_chunks times
        ensure!(
            evaluations.len() == 5 + 2 * self.number_zero_chunks,
            "Expected {} evaluations from the prover during Softmax verification, got {}",
            5 + 2 * self.number_zero_chunks,
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

        // Finally add the error check and the last claim, for this we need the output column of the exponential lookup
        let exp_output_claim = evaluations[1];

        if !self.number_zero_chunks.is_zero() {
            // Need to add in the zero table lookup related values in this case
            let zero_table_beta_eval = eq_xy_eval(logup_claims[3].point(), &sumcheck_point);

            let (new_calc_subclaim, batch_challenge) = evaluations[5..].iter().fold(
                (calc_subclaim, batch_challenge),
                |(subclaim_acc, bc), &claim| {
                    (subclaim_acc + zero_table_beta_eval * claim * bc, bc * alpha)
                },
            );
            let output_eval =
                evaluations[6..].iter().step_by(2).copied().product::<E>() * exp_output_claim;
            calc_subclaim = new_calc_subclaim
                + batch_challenge
                    * output_eval
                    * (error_beta_eval * two_mult + alpha * last_claim_beta_eval);
        } else {
            calc_subclaim += batch_challenge * error_beta_eval * two_mult * exp_output_claim;
            calc_subclaim += batch_challenge * alpha * last_claim_beta_eval * exp_output_claim;
        }

        ensure!(
            sumcheck_subclaim.expected_evaluation == calc_subclaim,
            "Sumcheck verification output claim did not match calculated claim in Softmax verification, expected: {:?}, calculated: {:?}",
            sumcheck_subclaim.expected_evaluation,
            calc_subclaim
        );

        // Now we work out the claim on the input to pass to the next layer
        let two_to_the_16 = E::from_canonical_u64(1u64 << 16);
        let two_to_the_8 = E::from_canonical_u64(1u64 << 8);

        let mask_input_eval =
            evaluations[0] * two_to_the_16 + evaluations[3] * two_to_the_8 + evaluations[2];

        let mask_input_eval = if !self.number_zero_chunks.is_zero() {
            let softmax_table_vars = ceil_log2(self.bkm as usize >> 16);
            let zero_table_init_multiplier =
                E::from_canonical_u64(1u64 << (16 + softmax_table_vars));
            let zero_table_size = E::from_canonical_u64(1u64 << self.zero_table_vars);
            evaluations[5..]
                .iter()
                .step_by(2)
                .fold(
                    (mask_input_eval, zero_table_init_multiplier),
                    |(eval_acc, mul_acc), &eval| {
                        (eval_acc + eval * mul_acc, mul_acc * zero_table_size)
                    },
                )
                .0
        } else {
            mask_input_eval
        };

        // Run verification for the masking sumcheck
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![sumcheck_point.len(); 3]]);

        let sumcheck_subclaim = IOPVerifierState::<E>::verify(
            -mask_input_eval,
            mask_proof,
            &aux_info,
            verifier.transcript,
        );

        let mask_point = sumcheck_subclaim.point_flat();
        let eq_eval = eq_xy_eval(&mask_point, &sumcheck_point);
        // Compute the tril and bias evaluation, to do this we need the number of columns and rows
        let padded_shape = &shape_step.padded_output_shape[0];
        let num_dims = padded_shape.len();
        let rows = ceil_log2(padded_shape[num_dims - 2]);
        let columns = ceil_log2(padded_shape[num_dims - 1]);
        let column_point = mask_point.iter().take(columns).copied().collect::<Vec<E>>();
        let row_point = mask_point
            .iter()
            .skip(columns)
            .take(rows)
            .copied()
            .collect::<Vec<E>>();
        let tril_eval = eval_zeroifier_mle(&column_point, &row_point);
        let negative_infinity: E = (-((self.bkm >> 16) + 1) << 16).to_field();
        let bias_eval = negative_infinity * (E::ONE - tril_eval);
        let mult_tril = eq_eval * tril_eval;
        let mult_bias = eq_eval * bias_eval;
        let mult_inv = mult_tril.inverse();

        // Now the shifted input eval is `(sumcheck_subclaim - mult_bias) * mult_inv`
        let shifted_input_eval = (sumcheck_subclaim.expected_evaluation - mult_bias) * mult_inv;
        // To get the output claim eval we subtract the shift eval and multiply by the inverse of `self.scalar`
        let field_scalar: E = self.scalar.to_field();
        let input_eval = (shifted_input_eval - evaluations[4]) * field_scalar.inverse();
        // Add the commitments to the commitment verifier, the first four relate to exp table and range checks, the fifth is for the shift
        commitments
            .iter()
            .zip(evaluations.iter())
            .take(4)
            .try_for_each(|(comm, &eval)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(comm.clone(), Claim::<E>::new(sumcheck_point.clone(), eval))
            })?;

        // Add the shift commitment claim to the verifier
        let shift_point = mask_point
            .iter()
            .skip(extra_vars)
            .copied()
            .collect::<Vec<E>>();

        verifier.commit_verifier.add_witness_claim(
            commitments[4].clone(),
            Claim::<E>::new(shift_point, evaluations[4]),
        )?;

        if !self.number_zero_chunks.is_zero() {
            // Also add the zero table commitment claims in
            commitments
                .iter()
                .zip(evaluations.iter())
                .skip(5)
                .try_for_each(|(comm, &eval)| {
                    verifier.commit_verifier.add_witness_claim(
                        comm.clone(),
                        Claim::<E>::new(sumcheck_point.clone(), eval),
                    )
                })?;
        }

        Ok(vec![Claim::<E>::new(mask_point, input_eval)])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Mask used in attention so that tokens can only see "previous" values.
pub struct AttentionMask<N> {
    /// This is the tensor we multiply elementwise to zero out the correct locations
    pub tril: Tensor<N>,
    /// This is the bias we add elementwise to ensure all zeroes are replaced with `-inf`
    pub bias: Tensor<N>,
    /// The value for negative infinity
    negative_infinity: N,
}

impl<N: Number> Default for AttentionMask<N> {
    fn default() -> Self {
        AttentionMask {
            tril: Tensor::<N>::new(vec![].into(), vec![]),
            bias: Tensor::<N>::new(vec![].into(), vec![]),
            negative_infinity: N::MIN,
        }
    }
}

impl<N: Number> AttentionMask<N> {
    /// Creates a new mask given the unpadded input shape and the value to use for `-inf`
    pub fn new(unpadded_shape: &[usize], negative_inf: N) -> Result<AttentionMask<N>> {
        // The input shape should have length either 2 or 3 and the final 2 dimensions should be equal
        let num_dims = unpadded_shape.len();

        let correct_num_dims = num_dims == 2 || num_dims == 3;
        ensure!(
            correct_num_dims,
            "In order to create an Attention Mask the input should have either 2 or 3 dimensions, got: {}",
            num_dims
        );

        // Now check that either the final two dimensions are the same or the second to last dimension is 1
        let dims_equal = unpadded_shape[num_dims - 2] == unpadded_shape[num_dims - 1];

        ensure!(
            dims_equal,
            "Final two dimensions should be equal, got: second to last: {}, last: {}",
            unpadded_shape[num_dims - 2],
            unpadded_shape[num_dims - 1]
        );

        // Now that we know all the dimensions line up make the lower triangular tensor
        let shape = if num_dims == 2 {
            let mut shape = unpadded_shape.to_vec();
            shape.insert(0, 1);
            shape
        } else {
            unpadded_shape.to_vec()
        };

        // Make the tril and bias tensor
        let tril = Tensor::<N>::tril(shape[2], shape[0], 0);

        let bias = Tensor::<N>::tri(shape[2], shape[0], 0, N::default(), negative_inf);

        Ok(AttentionMask {
            tril,
            bias,
            negative_infinity: negative_inf,
        })
    }

    /// Pads the [`AttentionMask`] for proving purposes
    fn pad(&mut self) -> Result<()> {
        // First check that the bias and tril shapes agree
        let shapes_equal = self
            .tril
            .shape
            .iter()
            .zip(self.bias.shape.iter())
            .all(|(t, b)| *t == *b);
        ensure!(
            shapes_equal,
            "Can't pad Attention Mask as tril and bias had different shapes"
        );

        // Now we check to see if everything is already a power of two
        if self.tril.shape.iter().all(|s| s.is_power_of_two()) {
            return Ok(());
        }

        // Calculate padded tensors
        // For tril and bias we just expand to a larger lower/upper triangular matrix
        let padded_shape = self
            .bias
            .get_shape()
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<usize>>();
        self.tril = Tensor::<N>::tril(padded_shape[2], padded_shape[0], 0);
        self.bias = Tensor::<N>::tri(
            padded_shape[2],
            padded_shape[0],
            0,
            N::default(),
            self.negative_infinity,
        );

        Ok(())
    }

    /// Apply the mask to an input, this method allows the input to have two or three dims and adjusts accordingly.
    /// It elementwise multiplies by `self.tril` and then adds `self.bias`.
    fn apply(&self, input: &Tensor<N>) -> Result<Tensor<N>> {
        // Check the the input has 2 or 3 dims
        let num_input_dims = input.shape.len();
        ensure!(
            num_input_dims == 2 || num_input_dims == 3,
            "To apply Attention Mask input need to have 2 or 3 dims, got: {}",
            num_input_dims
        );
        // If the input only has 2 dims reshape to have 3
        if num_input_dims == 3 {
            if !input
                .shape
                .iter()
                .zip(self.tril.shape.iter())
                .all(|(a, b)| *a == *b)
            {
                return Err(anyhow!(
                    "Cannot apply attention mask, input did not have the same shape as mask"
                ));
            }

            Ok(input.mul(&self.tril).add(&self.bias))
        } else {
            let new_shape = input.get_shape().insert(0, 1);
            let new_input = input.clone().reshape(new_shape);

            if !new_input
                .shape
                .iter()
                .zip(self.tril.shape.iter())
                .all(|(a, b)| *a == *b)
            {
                return Err(anyhow!(
                    "Cannot apply attention mask, input did not have the same shape as mask"
                ));
            }

            let output = new_input.mul(&self.tril).add(&self.bias);
            Ok(output.reshape(input.get_shape()))
        }
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
        let input = Tensor::new(
            vec![1, 3, 3].into(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![1, 3, 3].into()])
            .unwrap();
        assert_eq!(output.outputs[0].get_shape(), vec![1, 3, 3].into());

        output.outputs[0].get_data().chunks(3).for_each(|chunk| {
            assert_eq!(chunk.iter().sum::<f32>(), 1.0);
        });
    }

    #[test]
    fn test_quantise() {
        // For now we test with GPT2 like parameters
        let scale = 1.0f32 / 768.0f32.sqrt();
        let softmax = Softmax::<f32>::new_with_scale(scale, 1024);

        for num_tokens in 1015..1016 {
            // Make random q and k vectors
            let test_q = Tensor::<f32>::random(&vec![num_tokens, 768].into());
            let test_k = Tensor::<f32>::random(&vec![768, num_tokens].into());

            let q_scaling = ScalingFactor::from_tensor(&test_q, None);
            let k_scaling = ScalingFactor::from_tensor(&test_k, None);

            // Pick the quantised domain to be Some((-1i128 << 24, 1i128 << 24)) since matrix multiplication on 768 columns adds at most 10 to the bit size
            // (already at bit size 14 before this due to multiplication of two 8 bit quant integers)
            let qk_scaling = ScalingFactor::from_scale(
                q_scaling.scale() * k_scaling.scale(),
                Some((-1 << 24, 1 << 24)),
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

                    // Make sure we are always within 1/100 th of the actual value
                    assert!(
                        quant_dequant_diff < 0.01,
                        "quant dequant diff was too large got: {}",
                        quant_dequant_diff
                    );
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
        let input = Tensor::new(
            vec![3, 3].into(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![3, 3].into()])
            .unwrap();
        // Since this is a masked evaluation, each row should sum to 1 and the first row should have 1 non-zero value, the second two non-zero
        // and so on.
        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ]
        );
    }

    #[test]
    fn test_softmax_proving() {
        let input_shape = vec![12, 200, 200];

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
