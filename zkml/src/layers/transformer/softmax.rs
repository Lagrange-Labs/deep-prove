//! This layer applies the softmax function to the last dimension of the input tensor
use std::marker::PhantomData;

use crate::{
    Element, Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo, ProvingData, QuantizeOp},
    quantization::{self, ScalingFactor},
    tensor::{Number, Shape},
};

use anyhow::{Result, anyhow, ensure};

use ff_ext::ExtensionField;

use multilinear_extensions::util::ceil_log2;
use serde::{Deserialize, Serialize};

/// The base 2 logarithm of the scale factor used in exponential lookup tables
const LOG_SCALE_FACTOR: usize = 24;
/// The scale factor for our fixed point arithmetic
const SCALE_FACTOR: usize = 1 << LOG_SCALE_FACTOR;

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

impl<N: Number> Softmax<N> {
    pub fn new() -> Self {
        Self {
            scalar: N::unit(),
            apply_on_dim: None,
            max_size: 1024usize,
            quant_info: None,
        }
    }
    pub fn new_with_scale(scale: N) -> Softmax<N> {
        Softmax {
            scalar: scale,
            apply_on_dim: None,
            max_size: 1024usize,
            quant_info: None,
        }
    }
    pub fn quantise(&self, input_scaling: ScalingFactor) -> Result<Softmax<Element>> {
        // First we work out what we need to multiply by to get the input scale factor to be 2^32
        let input_scale_factor = input_scaling.scale();
        let temperature = self.scalar.to_f32()?;
        let multiplier =
            (SCALE_FACTOR as f32 * input_scale_factor * temperature).round() as Element;

        // minimum_input is calculated as `(input_min - sqrt(d) * ln_n - d * input_max)/sqrt(d)` and then quantised
        let input_min = input_scaling.min();
        let input_max = input_scaling.max();

        let min_input_float =
            input_min * temperature - (self.max_size as f32 * (input_max * temperature).exp()).ln();
        // Now that we have the minimum possible input as a float we need to work out how many integral bits we need to account for
        // We know that the minimum input is negative so first we take the absoloute value
        let min_input_abs = min_input_float.abs();
        let int = min_input_abs.round() as usize;
        let integral_bits = ceil_log2(int);

        let table_size = 1i128 << (integral_bits + 8);
        let base = 1i128 << (LOG_SCALE_FACTOR - 8);

        let lut = (0i128..table_size)
            .map(|j| {
                let prod = base * j;
                let float_exp = (-prod as f32 / SCALE_FACTOR as f32).exp();
                (float_exp * 256.0f32).round() as Element
            })
            .collect::<Vec<Element>>();

        let float_error = calc_softmax_error(
            base * (table_size),
            base,
            2.0f32.powf(16.0f32),
            SCALE_FACTOR as f32,
            3.0f32,
            0.0f32,
            2.0f32,
            1.0f32 / temperature,
        )
        .abs();
        println!(
            "float error: {}, 1 quant: {}, 1+e quant: {}, 1-e quant: {}",
            float_error,
            2.0f32.powf(16.0f32),
            ((1.0f32 + float_error) * 2.0f32.powf(16.0f32)).round(),
            ((1.0f32 - float_error) * 2.0f32.powf(16.0f32)).round()
        );

        let quant_info = QuantisedSoftmaxData {
            input_scale_factor: input_scaling,
            lut,
        };

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

#[derive(Debug, Clone, Serialize, Deserialize)]
/// This struct is used to store information used when evaluating the quantised version of [`Softmax`] on
/// [`Element`]s.
struct QuantisedSoftmaxData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This stores the output column of the `exp` lookup
    lut: Vec<Element>,
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

impl<N: Number> Softmax<N> {}

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
    println!("first term: {}", first_term);
    let second_term = (bkm as f32 / common_denom).exp() / (2.0f32 * output_sf.powf(1.0 / kml));
    println!("second term: {}", second_term);
    let c = (first_term + second_term).powf(kml) - 1.0f32;
    println!("c: {}", c);
    let term_one = c * (1.0f32 / (2.0f32 * input_sf * temp)).exp();
    let term_two = -1023.0f32 * ((-bkm as f32) / input_sf * temp).exp();
    println!(
        "optimal BKM: {}",
        (input_sf * temp / (kml + 1.0)) * (kml * 2048.0f32.ln() + output_sf.ln())
    );
    println!("Actual bkm: {}", bkm);
    println!("first term in error: {}", term_one);
    println!("second term in error: {}", term_two);
    term_one + term_two
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

impl QuantizeOp for Softmax<f32> {
    type QuantizedOp = Softmax<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        _input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        unimplemented!()
    }
}
#[derive(Debug, Default, Clone)]
pub struct SoftmaxData<E>
where
    E: Clone + ExtensionField,
{
    low_range_check: Vec<Element>,
    high_range_check: Vec<Element>,
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
        } = self.quant_info().unwrap();

        let input = inputs[0];
        let chunk_size = *input.shape.last().ok_or(anyhow!(
            "Could not evaluate Softmax, Input tensor had no shape"
        ))?;

        let unpadded_chunk_size = *unpadded_input_shapes[0].last().ok_or(anyhow!(
            "Could not evaluate Softmax, unpadded input shape was empty for input"
        ))?;

        // The temperature is now stored as an Element so we need to convert back to float here
        let float_temp = self.scalar as f32 / (SCALE_FACTOR as f32 * input_scale_factor.scale());

        // Calculate the shift chunk by chunk
        let shift_data = input
            .get_data()
            .chunks(chunk_size)
            .map(|vec| {
                let sum = vec
                    .iter()
                    .take(unpadded_chunk_size)
                    .map(|x| (input_scale_factor.dequantize(x) * float_temp).exp())
                    .sum::<f32>();
                let log_sum = sum.ln();
                let shift = -(SCALE_FACTOR as f32 * log_sum).round() as Element;
                vec![shift; chunk_size]
            })
            .flatten()
            .collect::<Vec<_>>();

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

        let proving_data = ProvingData::Softmax(SoftmaxData {
            low_range_check,
            high_range_check,
            exp_lookup: (lookups, outputs.clone()),
            _phantom: PhantomData::<E>,
        });

        let output = Tensor::<Element>::new(input.get_shape(), outputs);

        Ok(LayerOut {
            outputs: vec![output],
            proving_data,
        })
    }
}

#[cfg(test)]
mod tests {

    use ff_ext::GoldilocksExt2;

    use crate::Tensor;

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
        let softmax = Softmax::<f32>::new_with_scale(scale);

        for num_tokens in 1020..1024 {
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
            let test_qk = test_q.matmul(&test_k);
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
            // The full float output
            let float_output = softmax
                .evaluate::<GoldilocksExt2>(&[&test_qk], vec![vec![num_tokens, num_tokens]])
                .unwrap();

            for ((&q, f), real) in quant_output.outputs[0]
                .get_data()
                .iter()
                .zip(dequant_output.outputs[0].get_data().iter())
                .zip(float_output.outputs[0].get_data())
            {
                let float_q = q as f32 / 2.0f32.powf(8.0f32);

                let quant_dequant_diff = (float_q - f).abs();
                let quant_float_diff = (float_q - real).abs();

                // Make sure we are always withing 1/100 th of the actual value
                // assert!(quant_dequant_diff < 0.01f32);
                // assert!(quant_float_diff < 0.01f32);
                // Uncomment to see all the results
                // println!(
                //     "quantised result: {}, q dequantised: {}, dequant result: {}, real result: {}",
                //     q, float_q, f, real
                // );
            }

            quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .for_each(|chunk| {
                    let row_sum = chunk.iter().sum::<Element>();
                    let diff_from_one = (row_sum - 256).abs();
                    // assert!(diff_from_one <= 1);
                    // Uncomment to see the row sum
                    // println!("row sum: {}", row_sum)
                })
        }
    }

    #[test]
    fn test_softmax_with_scale() {
        let scale = 1.0 / 2.0;
        let softmax = Softmax::new().with_scale(scale);
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
}
