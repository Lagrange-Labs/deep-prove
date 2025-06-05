//! This layer applies the softmax function to the last dimension of the input tensor
use crate::{
    Element, Tensor,
    layers::provable::{Evaluate, LayerOut},
    quantization::{self, ScalingFactor},
    tensor::Number,
};

use anyhow::{Result, anyhow, ensure};

use ff_ext::ExtensionField;

use serde::{Deserialize, Serialize};

/// The base 2 logarithm of the scale factor used in exponential lookup tables
const LOG_SCALE_FACTOR: usize = 32;
/// The scale factor for our fixed point arithmetic
const SCALE_FACTOR: usize = 1 << LOG_SCALE_FACTOR;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax {
    /// The information needed to perform the quantised version of Softmax, defaults to [`None`]
    quant_info: Option<QuantisedSoftmaxInfo>,
    /// This is the factor we divide by before exponentiating, when thought of as a Boltzmann distribution this is
    /// often referred to as the "Temperature".
    scalar: f32,
    /// This is the maximum size of dimension that we will normalise over. For example in an Attention layer this would be the maximum context size.
    max_size: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
/// This struct stores information relating to performing the quantised version of the softmax operation.
pub struct QuantisedSoftmaxInfo {
    /// Scaling factor for the input, only needed for quantised version and proving
    input_scaling: ScalingFactor,
    /// Scaling factor for the output, only needed for quantised version and proving,
    output_scaling: ScalingFactor,
    /// This is the smallest possible value the input can be, we use this in the lookup tables as an indicator that a value
    /// should map to zero (so in a sense this is negative infinity)
    minimum_input: Element,
    /// The lookup table output columns, these are arranged in little endian, so the least significant chunk (that does not map to 1) is the lookup at index 0.
    luts: Vec<Vec<Element>>,
    /// The number of least significant chunks, these are chunks that get mapped to 1 by the exponential function.
    no_least_sig_chunks: usize,
    /// The size of the error as a float
    attn_error: f32,
}

impl Default for Softmax {
    fn default() -> Self {
        Softmax {
            quant_info: None,
            scalar: 1.0f32,
            max_size: 1024usize,
        }
    }
}

impl Softmax {
    pub fn new_with_scale(scale: f32) -> Softmax {
        Softmax {
            quant_info: None,
            scalar: scale,
            max_size: 1024usize,
        }
    }
    pub fn quantise(&self, input_scaling: ScalingFactor, output_scaling: ScalingFactor) -> Self {
        if self.quant_info.is_some() {
            self.clone()
        } else {
            // First we work out what we need to multiply by to get the input scale factor to be 2^32
            let input_scale_factor = input_scaling.scale();
            let multiplier = 1.0f32 / input_scale_factor;

            let log_mult = multiplier.log2();
            let diff = 16.0 - log_mult;

            // This value is the floating point number we need to multiply by to get the scale factor to be 2^16, then we
            // multiply this by 2^16 so the end result is scaled by 2^32.
            let diff_multiplier = 2.0f32.powf(diff);

            let scaled = (diff_multiplier * (1u64 << 16) as f32).round();

            // minimum_input is calculated as `(input_min - sqrt(d) * ln_n - d * input_max)/sqrt(d)` and then quantised
            let input_min = input_scaling.min();
            let input_max = input_scaling.max();

            let d = self.scalar * self.scalar;

            let ln_n = (self.max_size as f32).ln();

            let min_input_float = (input_min - self.scalar * ln_n - d * input_max) / self.scalar;

            let minimum_input_element = (min_input_float * scaled * multiplier).round() as Element;

            // The B(k)s are fixed as 1, 1 << BIT_LEN, 1 << 2*BIT_LEN, ... until we reach at least 2^32, then we have one more for the case
            // that the input is < -1.
            let no_of_blocks = (LOG_SCALE_FACTOR / *quantization::BIT_LEN) + 1;
            let bks = (0..no_of_blocks)
                .map(|i| 1i128 << i * *quantization::BIT_LEN)
                .collect::<Vec<Element>>();

            // The input is at most a 40 bit number with 32 fractional bits, anything smaller than 1/2^8 is essentially equal to 1 so we need to
            // find how many "least significant" chunks we have
            let least_sig_chunks =
                ((LOG_SCALE_FACTOR - *quantization::BIT_LEN - 8) / *quantization::BIT_LEN) + 1;

            // calculate the error in softmax
            let attn_error = calc_softmax_error(
                -bks[no_of_blocks - 1],
                -bks[least_sig_chunks],
                (1u64 << (2 * *quantization::BIT_LEN)) as f32,
                SCALE_FACTOR as f32,
                no_of_blocks as f32,
                0.0f32,
                least_sig_chunks as f32,
                self.scalar,
            );

            let table_size = 1i128 << *quantization::BIT_LEN;

            let table_columns = (least_sig_chunks..no_of_blocks)
                .map(|i| {
                    let base = 1i128 << (i * *quantization::BIT_LEN);
                    let mult = (1usize << *quantization::BIT_LEN) as f32;
                    (0i128..table_size)
                        .map(|j| {
                            let prod = base * j;
                            let float_exp = (-prod as f32 / 2.0f32.powf(32.0)).exp();
                            (float_exp * mult).round() as Element
                        })
                        .collect::<Vec<Element>>()
                })
                .collect::<Vec<Vec<Element>>>();

            let quant_info = QuantisedSoftmaxInfo {
                input_scaling,
                output_scaling,
                minimum_input: minimum_input_element,
                luts: table_columns,
                no_least_sig_chunks: least_sig_chunks,
                attn_error,
            };

            Softmax {
                quant_info: Some(quant_info),
                scalar: self.scalar,
                max_size: self.max_size,
            }
        }
    }
}

/// Calculates the error as an [`f32`] when applying softmax as described in zkLLM.
fn calc_softmax_error(
    bkm: i128,
    bl: i128,
    output_sf: f32,
    input_sf: f32,
    k: f32,
    m: f32,
    l: f32,
    temp: f32,
) -> f32 {
    let kml = k - m - l;
    let common_denom = kml * input_sf * temp;
    let first_term = (bl as f32 / common_denom).exp();
    let second_term = (bkm as f32 / common_denom).exp() / (2.0f32 * output_sf.powf(1.0 / kml));

    (first_term + second_term).powf(kml) - 1.0f32
}

impl Evaluate<f32> for Softmax {
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
                let sum = vec.iter().map(|x| (x / self.scalar).exp()).sum::<f32>();
                vec.iter()
                    .map(|x| (x / self.scalar).exp() / sum)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();
        let output_tensor = Tensor::new(input.get_shape(), output);
        Ok(LayerOut::from_vec(vec![output_tensor]))
    }
}

#[derive(Debug, Default, Clone)]
pub struct SoftmaxData<E>
where
    E: Clone + ExtensionField,
{
    shift_evals: Vec<E::BaseField>,
}

impl Evaluate<Element> for Softmax {
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
        let QuantisedSoftmaxInfo {
            input_scaling,
            output_scaling,
            minimum_input,
            luts,
            no_least_sig_chunks,
            attn_error,
        } = self.quant_info.as_ref().unwrap();

        let input = inputs[0];
        let chunk_size = *input.shape.last().ok_or(anyhow!(
            "Could not evaluate Softmax, Input tensor had no shape"
        ))?;

        let unpadded_chunk_size = *unpadded_input_shapes[0].last().ok_or(anyhow!(
            "Could not evaluate Softmax, unpadded input shape was empty for input"
        ))?;

        let shift_data = input
            .get_data()
            .chunks(chunk_size)
            .map(|vec| {
                let sum = vec
                    .iter()
                    .take(unpadded_chunk_size)
                    .map(|x| (input_scaling.dequantize(x) / self.scalar).exp())
                    .sum::<f32>();
                let log_sum = sum.ln();
                let shift = -(SCALE_FACTOR as f32 * self.scalar * log_sum).round() as Element;
                vec![shift; chunk_size]
            })
            .flatten()
            .collect::<Vec<_>>();

        todo!()
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
        let softmax = Softmax::default();
        println!("1/e: {}", (-1.0f32).exp());
        println!("ln(x) = 0.01, x = {}", 0.01f32.ln());
        let input_scaling = ScalingFactor::default();
        let output_scaling = ScalingFactor::default();

        let _ = softmax.quantise(input_scaling, output_scaling);
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
