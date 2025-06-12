//! This layer applies the softmax function to the last dimension of the input tensor
use std::marker::PhantomData;

use crate::{
    Element, ScalingStrategy, Tensor,
    layers::provable::{
        Evaluate, LayerOut, NodeId, OpInfo, ProvingData, QuantizeOp, QuantizeOutput,
    },
    quantization::ScalingFactor,
    tensor::Number,
};

use anyhow::{Result, anyhow, ensure};

use ff_ext::ExtensionField;

use multilinear_extensions::util::ceil_log2;
use serde::{Deserialize, Serialize};

/// The base 2 logarithm of the scale factor used in exponential lookup tables
const LOG_SCALE_FACTOR: usize = 24;
/// The scale factor for our fixed point arithmetic
const SCALE_FACTOR: usize = 1 << LOG_SCALE_FACTOR;
/// The scale factor of the outputs of the `exp` lookup
const OUTPUT_SCALE_FACTOR: usize = 1 << (LOG_SCALE_FACTOR - 1);

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax<N> {
    /// This is the factor we divide by before exponentiating, when thought of as a Boltzmann distribution this is
    /// often referred to as the "Temperature".
    scalar: N,
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
    pub fn new_with_scale(scale: N) -> Softmax<N> {
        Softmax {
            scalar: scale,
            max_size: 1024usize,
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

        let min_input_float =
            input_min - (self.max_size as f32 * (input_max * temperature).exp()).ln();
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
}

/// Calculates the error as an [`f32`] when applying softmax as described in zkLLM.
/// This functions returns the error togeter with the value `bkm` such that anything smaller
/// than `bkm` should be mapped to zero.
fn calc_softmax_error(
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
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<f32, E>> {
        let input = inputs[0];
        let chunk_size = *input.shape.last().ok_or(anyhow!(
            "Could not evaluate Softmax, Input tensor had no shape"
        ))?;
        let output = input
            .get_data()
            .chunks(chunk_size)
            .map(|vec| {
                let sum = vec.iter().map(|x| (x * self.scalar).exp()).sum::<f32>();
                vec.iter()
                    .map(|x| (x * self.scalar).exp() / sum)
                    .collect::<Vec<_>>()
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
        input_shapes: &[Vec<usize>],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Vec<usize>> {
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
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<Element, E>> {
        // First we heck that we have some quantisation info.
        ensure!(
            self.quant_info.is_some(),
            "Could not evaluate quantised softmax because the operation ahs not been quantised"
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
            ..
        } = self.quant_info().unwrap();

        let input = inputs[0];
        let chunk_size = *input.shape.last().ok_or(anyhow!(
            "Could not evaluate Softmax, Input tensor had no shape"
        ))?;

        let unpadded_chunk_size = *unpadded_input_shapes[0].last().ok_or(anyhow!(
            "Could not evaluate Softmax, unpadded input shape was empty for input"
        ))?;

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

#[cfg(test)]
mod tests {
    use goldilocks::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_softmax() {
        let softmax = Softmax::default();
        let input = Tensor::new(vec![2, 3], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3]])
            .unwrap();
        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0
            ]
        );
    }

    #[test]
    fn test_quantise() {
        // For now we test with GPT2 like parameters
        let scale = 1.0f32 / 768.0f32.sqrt();
        let softmax = Softmax::<f32>::new_with_scale(scale);

        for num_tokens in 1015..1025 {
            // Make random q and k vectors
            let test_q = Tensor::<f32>::random(&[num_tokens, 768]);
            let test_k = Tensor::<f32>::random(&[768, num_tokens]);

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
                .evaluate::<GoldilocksExt2>(&[&test_qk_quant], vec![vec![num_tokens, num_tokens]])
                .unwrap();
            // The result of running the quantised input as floats
            let dequant_output = softmax
                .evaluate::<GoldilocksExt2>(&[&test_qk_dequant], vec![vec![num_tokens, num_tokens]])
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
        let softmax = Softmax::new_with_scale(2.0);
        let input = Tensor::new(vec![2, 3], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![2, 3]])
            .unwrap();
        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0
            ]
        );
    }
}
