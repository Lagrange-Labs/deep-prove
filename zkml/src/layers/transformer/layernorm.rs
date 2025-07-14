use anyhow::{Result, anyhow, ensure};
use ark_std::Zero;
use multilinear_extensions::util::ceil_log2;
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::{
    Element, ScalingFactor, Tensor,
    layers::provable::{QuantizeOp, QuantizeOutput},
    padding::PaddingMode,
    parser::{gguf::FileTensorLoader, json, llm::LLMConfig},
    quantization,
    tensor::{Number, Shape},
};

use crate::layers::provable::{Evaluate, LayerOut, OpInfo};
use burn::{
    module::Param,
    nn::LayerNormConfig as BLayerNormConfig,
    tensor::{Tensor as BTensor, TensorData},
};

/// The base 2 logarithm of the scale factor used in the inverse square root lookup tables
pub(crate) const LOG_LAYERNORM_SCALE_FACTOR: usize = 16;
/// The scale factor for our fixed point arithmetic
pub(crate) const LAYERNORM_SCALE_FACTOR: usize = 1 << LOG_LAYERNORM_SCALE_FACTOR;
/// The scale factor of the outputs of the inverse square root lookup tables lookup
pub(crate) const LAYERNORM_OUTPUT_SCALE_FACTOR: usize = 1 << 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm<N> {
    pub gamma: Tensor<N>,
    pub beta: Tensor<N>,
    pub eps: f32,
    pub quant_info: Option<QuantisedLayerNormData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// This struct is used to store information used when evaluating the quantised version of [`LayerNorm`] on
/// [`Element`]s.
pub struct QuantisedLayerNormData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This is the multiplier we have to rescale the inputs with
    multiplier: Element,
    /// This stores the output column of the inverse square root lookup
    lut: Vec<Element>,
    /// The size of the dimension we average over
    dim_size: usize,
    /// This is the number of bits that get range checked
    range_check_bits: usize,
    /// The base 2 log of the value we have to multiply the most significant range check chunk by
    top_chunk_scalar_log: usize,
}

impl<N: Number> LayerNorm<N> {
    pub fn new(gamma: Tensor<N>, beta: Tensor<N>, eps: f32) -> Self {
        assert_eq!(gamma.get_shape(), beta.get_shape());
        Self {
            gamma,
            beta,
            eps,
            quant_info: None,
        }
    }

    /// Returns the size of the dimension normalisation occurs over.
    pub fn normalisation_dim_size(&self) -> usize {
        self.gamma.shape[0]
    }

    /// Quantise the layer. To do this we want to have a common scale factor so that lookup tables can be reused, so we use the
    /// constant [`LAYERNORM_SCALE_FACTOR`] as the input column scale factor. We need to work out how big the table needs to be to cover
    /// all of our possible inputs.
    ///
    /// This method reutnrs the quantised [`LayerNorm`] as well as the `intermediate_bit_size` for the following requant layer.
    pub fn quantise(
        &self,
        input_scaling: ScalingFactor,
        model_scaling: ScalingFactor,
    ) -> Result<(LayerNorm<Element>, usize)> {
        // The input to the lookup table is `N*sum2 - sum1^{2}` where `sum2 = \sum xi^{2}` and `sum1 = \sum xi`.
        // We use this value because the standard deviation can be calculated by `(N*sum2 - sum1^{2}).sqrt() / N`
        // Since each `xi` is a value between `*quantisation::MIN` and `*quantisation::MAX` it has bit-size `*quantization::BIT_LEN - 1`.
        // This means `sum1` has bit-size `ceil_log2(N) + *quantization::BIT_LEN - 1` and `sum2` has bit-size `2(*quantization::BIT_LEN - 1)`
        // Then `sum1^{2}` has bit-size `2(ceil_log2(N) + *quantization::BIT_LEN - 1)` and `Nsum2` has bit_size `ceil_log2(N) + 2(*quantization::BIT_LEN - 1)`.
        // Finally we have to multiply all of this by `multiplier = LAYERNORM_SCALE_FACTOR * input_scaling.scale() * input_scaling.scale()` so we have `ceil_log2(multiplier)`
        // additional bits on top of this.

        // Get the input scale
        let input_scale = input_scaling.scale();
        // Get the dim size (N)
        let dim_size = self.normalisation_dim_size();
        // We work out what we have to mutliply by so that everything is scaled to `LAYERNORM_SCALE_FACTOR` in quantised world
        let multiplier =
            (LAYERNORM_SCALE_FACTOR as f32 * input_scale * input_scale).round() as Element;
        // Work out the number of variables the table requires, this is likely to be far too large to actually materialise as a table
        let full_table_bit_size = 2 * (ceil_log2(dim_size) + *quantization::BIT_LEN - 1)
            + ceil_log2(multiplier as usize)
            + 1;
        // To get around this we use the fact that we should only have roughly `2*(*quantization::BIT_LEN -1)` bits of precision i.e. only the most significant `2*(*quantization::BIT_LEN -1)`
        // can actually be "trusted" the rest are essentially junk because they don't come from the actual inputs and are just guesses at the part that we have alread "rounded away" in quantisation.
        // So the actual part we perform inverse square root on is size `2*(*quantization::BIT_LEN -1)` and then we rust need the discarded part to be range checked.
        let range_checked_bits = full_table_bit_size - 2 * (*quantization::BIT_LEN - 1);

        // The final chunk might be values with fewer than *quantization::BIT_LEN bits so we work out what we need to scale the value up by in order to use our standard range check table.
        let remainder_bits = range_checked_bits % *quantization::BIT_LEN;
        let top_chunk_scalar_log = if !remainder_bits.is_zero() {
            *quantization::BIT_LEN - remainder_bits
        } else {
            0
        };
        // Calculate the lookup table
        let table_max: Element = 1 << 2 * (*quantization::BIT_LEN - 1);
        let table_min = -table_max;
        // Because we don't use the same formula for the standard deviation as LayerNorm does in float we have to rescale `self.eps` in this case to be `N^2 * self.eps`
        let rescaled_eps = (dim_size * dim_size) as f32 * self.eps;
        let lut = (table_min..table_max)
            .map(|val| {
                // First we have to shift by `range_checked_bits`
                let shifted_val = val << range_checked_bits;
                // Now we convert back to float and perform the operation
                let float_output = 1.0f32
                    / ((shifted_val as f32 / LAYERNORM_SCALE_FACTOR as f32) + rescaled_eps).sqrt();
                // Now we use the output scale factor to recover the element value
                (float_output * LAYERNORM_OUTPUT_SCALE_FACTOR as f32).round() as Element
            })
            .collect::<Vec<Element>>();

        let max_lut_value = lut.iter().map(|v| v.abs()).max().unwrap();
        // The value is positive so we just convert to usize
        let max_lut_value_bits = ceil_log2(max_lut_value as usize);

        // Make the QuantisedLayerNormData
        let quant_info = QuantisedLayerNormData {
            input_scale_factor: input_scaling,
            multiplier,
            lut,
            dim_size,
            range_check_bits: range_checked_bits,
            top_chunk_scalar_log,
        };

        let quant_gamma_data = self
            .gamma
            .get_data()
            .iter()
            .map(|v| {
                let vf32 = v.to_f32()?;
                Ok(model_scaling.quantize(&vf32))
            })
            .collect::<Result<Vec<Element>, anyhow::Error>>()?;

        let quant_gamma = Tensor::<Element>::new(self.gamma.get_shape(), quant_gamma_data);
        // Work out how to quantise the bias, it needs to have the same scale factor as the end product.
        // This will be `input_scaling.scale() * model_scaling.scale() * 1.0f32 / LAYERNORM_OUTPUT_SCALE_FACTOR as f32`
        let bias_scale = input_scale * model_scaling.scale() / LAYERNORM_OUTPUT_SCALE_FACTOR as f32;

        let bias_max = self.beta.max_abs_output().to_f32()?;

        let quant_bias_min = (-bias_max / bias_scale).round() as Element;
        let quant_bias_max = (bias_max / bias_scale).round() as Element;

        let bias_scaling = ScalingFactor::from_parts(
            bias_max,
            -bias_max,
            bias_scale,
            (quant_bias_min, quant_bias_max),
        );
        let quant_bias_data = self
            .beta
            .get_data()
            .iter()
            .map(|v| {
                let vf32 = v.to_f32()?;
                Ok(bias_scaling.quantize(&vf32))
            })
            .collect::<Result<Vec<Element>, anyhow::Error>>()?;

        let quant_beta = Tensor::<Element>::new(self.beta.get_shape(), quant_bias_data);

        // To calculate the intermediate bit size we have that the output is `self.gamma * (N * input - SUM input) * lookup_output + self.beta`
        // So lets work out the left hand bit size
        let lhs_bit_size =
            2 * (*quantization::BIT_LEN - 1) + ceil_log2(dim_size) + 1 + max_lut_value_bits;

        let intermediate_bit_size = lhs_bit_size.max(ceil_log2(quant_bias_max as usize)) + 1;

        Ok((
            LayerNorm::<Element> {
                gamma: quant_gamma,
                beta: quant_beta,
                eps: self.eps,
                quant_info: Some(quant_info),
            },
            intermediate_bit_size,
        ))
    }
}

impl LayerNorm<f32> {
    pub fn from_json(l: &json::FileTensorLoader, _c: &LLMConfig) -> anyhow::Result<Self> {
        trace!("from_json: current path: {:?}", l.prefix);
        let gamma = l.get_tensor("norm.weight")?;
        let beta = l.get_tensor("norm.bias")?;
        let eps = l.metadata_to_f32("norm_epsilon")?;
        Ok(Self::new(gamma, beta, eps))
    }
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        ensure!(
            gamma.get_shape().as_ref() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.get_shape()
        );
        ensure!(
            beta.get_shape().as_ref() == &[c.embedding_size],
            "norm_beta must have shape [{}] vs given {:?}",
            c.embedding_size,
            beta.get_shape()
        );
        let eps = loader.metadata::<f32>(c.specific_config.norm_epsilon_key());
        Ok(Self::new(gamma, beta, eps))
    }
}

impl<N: Number> OpInfo for LayerNorm<N> {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "LayerNorm({:?},{:?})",
            self.gamma.get_shape(),
            self.beta.get_shape()
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

// Type alias for the backend to use.
type Backend = burn::backend::NdArray;

impl Evaluate<f32> for LayerNorm<f32> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        assert!(inputs.len() == 1);
        let input = inputs[0];
        ensure!(
            input.get_shape().len() == 2,
            "layernorm input must have shape [seq_len, embedding_size]: found {:?}",
            input.get_shape()
        );
        let embedding_size = input.get_shape()[1];
        let device = Default::default();
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/layer.rs.html#67
        let input = BTensor::<Backend, 2>::from_data(
            TensorData::new(input.get_data().to_vec(), input.get_shape()),
            &device,
        );
        let gamma = BTensor::<Backend, 1>::from_data(
            TensorData::new(self.gamma.get_data().to_vec(), self.gamma.get_shape()),
            &device,
        );
        let beta = BTensor::<Backend, 1>::from_data(
            TensorData::new(self.beta.get_data().to_vec(), self.beta.get_shape()),
            &device,
        );
        let config = BLayerNormConfig::new(embedding_size).with_epsilon(self.eps as f64);
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Param::from_tensor(beta);
        let output = norm.forward(input);
        let Ok(data): Result<Vec<f32>, _> = output.to_data().into_vec() else {
            anyhow::bail!("failed to convert to f32");
        };
        let output_shape = Shape::new(output.shape().dims);
        Ok(LayerOut::from_tensor(Tensor::<f32>::new(
            output_shape,
            data,
        )))
    }
}

impl Evaluate<Element> for LayerNorm<Element> {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        // First we check to see if there is any quant_info, if not error
        ensure!(
            self.quant_info.is_some(),
            "Cannot perform quantised LayerNorm evaluation if self.quant_info is None"
        );
        // Ensure we have a single input
        ensure!(
            inputs.len() == 1,
            "LayerNorm should have a single input, had: {}",
            inputs.len()
        );
        let input = inputs[0];

        let QuantisedLayerNormData {
            input_scale_factor,
            multiplier,
            lut,
            dim_size,
            range_check_bits,
            top_chunk_scalar_log,
        } = self.quant_info.as_ref().unwrap();

        // So we need to take the input data and calculate `N * multiplier * SUM (xi * xi) - multiplier * (SUM xi) * (SUM xi)`
        let final_dim = *input
            .get_shape()
            .last()
            .ok_or(anyhow!("LayerNorm input didn't have a shape"))?;
        let lookup_input_data = input
            .get_data()
            .chunks(final_dim)
            .map(|chunk| {
                let sum_squares = chunk.iter().map(|x| *x * *x).sum::<Element>();
                let sum = chunk.iter().sum::<Element>();
                *dim_size as Element * multiplier * sum_squares - multiplier * sum * sum
            })
            .collect::<Vec<Element>>();

        // Now that we have the raw lookup input we split it into the part that gets shifted away and the part that gets passed to the inverse square root table
        todo!()
    }
}

impl QuantizeOp for LayerNorm<f32> {
    type QuantizedOp = LayerNorm<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: crate::layers::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<crate::layers::provable::QuantizeOutput<Self::QuantizedOp>> {
        // TODO: write the layernorm quantization rule depending on proving
        // Currently still working since we want to test quantization of layers.
        warn!("LayerNorm quantization not implemented");
        let gamma = self.gamma.quantize(&ScalingFactor::default());
        let beta = self.beta.quantize(&ScalingFactor::default());
        let quantized = LayerNorm::new(gamma, beta, self.eps);
        Ok(QuantizeOutput::new(quantized, input_scaling.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;

    use super::*;

    impl<N: Number> LayerNorm<N> {
        pub fn random(size: usize) -> Self {
            let gamma = Tensor::<N>::random(&vec![size].into());
            let beta = Tensor::<N>::random(&vec![size].into());
            let eps = 1e-5;
            Self::new(gamma, beta, eps)
        }
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_layernorm() {
        let gamma = Tensor::<f32>::new(vec![1024].into(), vec![1.0; 1024]);
        let beta = Tensor::<f32>::new(vec![1024].into(), vec![0.0; 1024]);
        let eps = 1e-5;
        let layernorm = LayerNorm {
            gamma,
            beta,
            eps,
            quant_info: None,
        };
        let input = Tensor::<f32>::new(vec![1, 1024].into(), vec![0.0; 1024]);
        let output = layernorm.evaluate::<E>(&[&input], vec![]).unwrap();
        assert_eq!(output.outputs[0].get_shape(), vec![1, 1024].into());
        assert_eq!(output.outputs[0].get_data(), vec![0.0; 1024]);
    }

    #[test]
    fn test_quantise_layernorm() {
        let gamma = Tensor::<f32>::random(&vec![1024].into());
        let beta = Tensor::<f32>::random(&vec![1024].into());
        let eps = 1e-5;
        let layernorm = LayerNorm {
            gamma,
            beta,
            eps,
            quant_info: None,
        };

        let input_scaling = ScalingFactor::default();

        let (quant_layernorm, intermediate_bit_size) =
            layernorm.quantise(input_scaling, input_scaling).unwrap();
    }
}
