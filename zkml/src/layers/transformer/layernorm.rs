use std::{collections::HashMap, marker::PhantomData};

use anyhow::{Result, anyhow, ensure};
use ark_std::Zero;
use ff_ext::ExtensionField;
use itertools::izip;
use mpcs::{PolynomialCommitmentScheme, sum_check::eq_xy_eval};
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    util::ceil_log2,
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::trace;

use crate::{
    Claim, Context, Element, ScalingFactor, ScalingStrategy, Tensor,
    commit::compute_betas_eval,
    iop::{
        context::{ContextAux, ShapeStep},
        prover::Prover,
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof, NodeId, Requant,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
    },
    lookup::{
        context::{COLUMN_SEPARATOR, CommsAndEvals, CommsAndProofs, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_prove,
            structs::{LogUpProof, LogUpVerifierClaim},
            verifier::verify_logup_proof,
        },
        witness::LogUpWitness,
    },
    model::StepData,
    padding::PaddingMode,
    parser::{gguf::FileTensorLoader, json, llm::LLMConfig},
    quantization::{self, Fieldizer},
    tensor::{Number, Shape},
};

use burn::{
    module::Param,
    nn::LayerNormConfig as BLayerNormConfig,
    tensor::{Tensor as BTensor, TensorData},
};

/// The base 2 logarithm of the scale factor used in the inverse square root lookup tables
pub(crate) const LOG_LAYERNORM_SCALE_FACTOR: usize = 24;
/// The scale factor for our fixed point arithmetic
pub(crate) const LAYERNORM_SCALE_FACTOR: usize = 1 << LOG_LAYERNORM_SCALE_FACTOR;
/// The scale factor of the outputs of the inverse square root lookup tables lookup
pub(crate) const LAYERNORM_OUTPUT_SCALE_FACTOR: usize = 1 << 10;

const GAMMA_POLY_ID: &str = "LayerNormGamma";
const BETA_POLY_ID: &str = "LayerNormBeta";

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Data obtained during quantised evaluation of [`LayerNorm`] that is used during proving
pub struct LayerNormData<E: ExtensionField> {
    /// The input of the inverse square root lookup
    lookup_input: Vec<Element>,
    /// The output of the inverse square root lookup
    lookup_output: Vec<Element>,
    /// The part of the input that need to be range checked
    range_check: Vec<Element>,
    _phantom: PhantomData<E>,
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

    /// Returns the [`QuantisedLayerNormData`] if there is any.
    pub fn quant_info(&self) -> Option<&QuantisedLayerNormData> {
        self.quant_info.as_ref()
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
    ) -> Result<(LayerNorm<Element>, usize, ScalingFactor)> {
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
                eps: rescaled_eps,
                quant_info: Some(quant_info),
            },
            intermediate_bit_size,
            bias_scaling,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormCtx {
    node_id: NodeId,
    /// The result of calling [`f32::to_bits`] on the epsilon used for normalisation purposes
    eps: u32,
    /// The number of bits that get range checked (so we can know how many instances there are in the range lookup)
    range_check_bits: usize,
    /// The size of the dimension we normalise over (unpadded)
    dim_size: usize,
    /// The multiplier used to scale up inputs to the lookup table.
    multiplier: Element,
    /// The base 2 logarithm of the multiplier for the most significant chunk we range check
    top_chunk_scalar_log: usize,
}

impl OpInfo for LayerNormCtx {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!("LayerNorm(dimension size: {})", self.dim_size,)
    }

    fn is_provable(&self) -> bool {
        true
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
    fn evaluate<E: ExtensionField>(
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
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<Element, E>> {
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
            multiplier,
            lut,
            dim_size,
            range_check_bits,
            ..
        } = self.quant_info.as_ref().unwrap();

        // So we need to take the input data and calculate `N * multiplier * SUM (xi * xi) - multiplier * (SUM xi) * (SUM xi)`
        let final_dim = *input.get_shape().last().ok_or(anyhow!(
            "Cannot evaluate LayerNorm, input didn't have a shape"
        ))?;

        let range_check_mask: Element = (1 << range_check_bits) - 1;
        let table_max: Element = 1 << 2 * (*quantization::BIT_LEN - 1);
        let ((inv_sqrt_input, inv_sqrt_output), range_check): (
            (Vec<Element>, Vec<Element>),
            Vec<Element>,
        ) = input
            .get_data()
            .chunks(final_dim)
            .map(|chunk| {
                let sum_squares = chunk.iter().map(|x| *x * *x).sum::<Element>();
                let sum = chunk.iter().sum::<Element>();
                let full_value =
                    *dim_size as Element * multiplier * sum_squares - multiplier * sum * sum;
                let range_checked = full_value & range_check_mask;
                let inv_sqrt = full_value >> range_check_bits;
                let inv_sqrt_output = lut[(inv_sqrt + table_max) as usize];
                ((inv_sqrt, inv_sqrt_output), range_checked)
            })
            .unzip();

        let output_data = input
            .get_data()
            .chunks(final_dim)
            .zip(inv_sqrt_output.iter())
            .flat_map(|(input_chunk, denominator)| {
                let sum = input_chunk.iter().sum::<Element>();
                izip!(input_chunk, self.gamma.get_data(), self.beta.get_data())
                    .map(|(&v, &gamma, &beta)| {
                        gamma * (*dim_size as Element * v - sum) * *denominator + beta
                    })
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>();

        // Make the proving data
        let layernorm_data = LayerNormData::<E> {
            lookup_input: inv_sqrt_input,
            lookup_output: inv_sqrt_output,
            range_check,
            _phantom: PhantomData::<E>,
        };

        let output_tensor = Tensor::<Element>::new(input.get_shape(), output_data);
        Ok(LayerOut::from_tensor(output_tensor)
            .with_proving_data(ProvingData::LayerNorm(layernorm_data)))
    }
}

impl Requant {
    /// We implement a special way of formulating a [`Requant`] layer here where `s1*s2/s3 = 2^-s` where
    /// s is a positive integer (so the requant layer only needs to perform a shift rather than a shift and a rescaling)
    pub(crate) fn new_shift(
        input_scaling: ScalingFactor,
        output_scaling: ScalingFactor,
        intermediate_bit_size: usize,
    ) -> Result<Requant> {
        // First we check that we can actually use this method
        let input_s = input_scaling.scale();
        let output_s = output_scaling.scale();

        let m = input_s / output_s;
        let m_log = m.log2();
        let int_part = m_log.trunc().abs();
        // We allow for a possible floating point error that results in an imperfect division
        ensure!(
            (m_log.abs() - int_part).abs() < 1e-5,
            "Cannot perform shift only Requant as the fractional part of the exponent was too large, fractional part: {}",
            (m_log.abs() - int_part).abs()
        );

        // We want the part that gets shifted away to be a multiple of the quantisation bit length (that way we can use the same range table for each chunk)
        let next_multiple = (int_part as usize).next_multiple_of(*quantization::BIT_LEN);
        let fp_scale = next_multiple - int_part as usize;
        let fixed_point_multiplier: Element = 1 << fp_scale;

        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        ensure!(
            intermediate_bit_size + fp_scale <= 63,
            "Cannot construct shift only Requant, intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );
        Ok(Requant {
            right_shift: int_part as usize,
            fixed_point_multiplier,
            fp_scale,
            multiplier: m,
            intermediate_bit_size,
        })
    }
}

impl QuantizeOp for LayerNorm<f32> {
    type QuantizedOp = LayerNorm<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        // First check we have one input_scaling
        ensure!(
            input_scaling.len() == 1,
            "Could not quantise LayerNorm, too many input scaling factors {}, expected 1",
            input_scaling.len()
        );
        let input_scaling_factor = input_scaling[0];
        // Now we construct the `model_scaling` from `self.gamma`
        let model_scaling = ScalingFactor::from_tensor(&self.gamma, None);

        let (quantised_layernorm, intermediate_bit_size, intermediate_scaling) =
            self.quantise(input_scaling_factor, model_scaling)?;
        // We will use the `intermediate_scaling` to work out a suitable `output_scaling`. Ideally `output_scaling` is 2^-s where the fractional part of `s` is the same as the fractional part of `intermediate_scaling`
        // and the integer part is such that 2^-s is as close as possible to the observed scaling factor.
        let observed_scalings = S::scaling_factors_for_node(data, node_id, 1);
        ensure!(
            observed_scalings.len() == 1,
            "Observed scaling factors for LayerNorm layer different from 1, observed {}",
            observed_scalings.len()
        );
        let observed_scaling = observed_scalings[0];
        let observed_scale = observed_scaling.scale();
        let obs_log = observed_scale.log2();
        let obs_fract = obs_log.fract().abs();
        let obs_int = obs_log.trunc().abs();
        let intermediate_scale = intermediate_scaling.scale();
        let inter_log = intermediate_scale.log2();
        let inter_fract = inter_log.fract().abs();
        // The value diff = (obs_fract - inter_fract) is between -1 and 1 and where it falls in this range defines what we should set `output_scaling` to be
        // if its positive that means `obs_fract` was larger and then we have two cases:
        //  1) diff < 0.5 => output_scaling should have scale 2^-{obs_int + inter_fract}
        //  2) diff >= 0.5 => output_scaling should have scale 2^-{obs_int + 1 + inter_fract}
        // if it is negative that means `inter_fract was larger` and then we have two cases:
        //  1) 0>= diff > -0.5 => output_scaling should have scale 2^-{obs_int + inter_fract}
        //  2) -0.5 >= diff > -1 => output_scaling should have scale 2^-{obs_int - 1 + inter_fract}
        let output_scale = match obs_fract - inter_fract {
            0.5f32..1.0f32 => 2.0f32.powf(-(obs_int + 1.0f32 + inter_fract)),
            -0.5f32..0.5f32 => 2.0f32.powf(-(obs_int + inter_fract)),
            -1.0f32..0.5f32 => 2.0f32.powf(-(obs_int - 1.0f32 + inter_fract)),
            _ => unreachable!(),
        };
        println!(
            "output scale: {output_scale}, observed scale: {observed_scale}, intermediate scale: {intermediate_scale}"
        );
        let output_scaling = ScalingFactor::from_parts(
            observed_scaling.max(),
            observed_scaling.min(),
            output_scale,
            observed_scaling.domain(),
        );
        // Make the requant layer
        let requant =
            Requant::new_shift(intermediate_scaling, output_scaling, intermediate_bit_size)?;

        Ok(QuantizeOutput::new(quantised_layernorm, vec![output_scaling]).with_requant(requant))
    }
}

impl<E> ProveInfo<E> for LayerNorm<Element>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        if let Some(quant_info) = self.quant_info() {
            let QuantisedLayerNormData {
                multiplier,
                dim_size,
                range_check_bits,
                top_chunk_scalar_log,
                ..
            } = quant_info;

            // Add the tables that LayerNorm requires
            aux.tables.insert(TableType::Range);
            aux.tables.insert(TableType::InverseSQRT(
                self.eps.to_bits(),
                *range_check_bits,
            ));

            // Add the Gamma and Beta commitments
            let gamma_evals = self.gamma.pad_next_power_of_two().get_data().to_vec();
            let beta_evals = self.beta.pad_next_power_of_two().get_data().to_vec();

            aux.model_polys = {
                let mut model_polys = HashMap::new();
                model_polys.insert(GAMMA_POLY_ID.to_string(), gamma_evals);
                model_polys.insert(BETA_POLY_ID.to_string(), beta_evals);
                Some(model_polys)
            };

            aux.max_poly_len = aux
                .last_output_shape
                .iter()
                .fold(aux.max_poly_len, |acc, shapes| {
                    acc.max(shapes.next_power_of_two().product())
                });

            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::<E>::LayerNorm(LayerNormCtx {
                    node_id: id,
                    eps: self.eps.to_bits(),
                    range_check_bits: *range_check_bits,
                    dim_size: *dim_size,
                    multiplier: *multiplier,
                    top_chunk_scalar_log: *top_chunk_scalar_log,
                }),
                aux,
            ))
        } else {
            Err(anyhow!(
                "LayerNorm operation has not been quantised so no proving info available"
            ))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`LayerNorm`] operation.
pub struct LayerNormProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The LogUp proofs for LayerNorm, they are ordered `inv_sqrt_lookup`, `range_lookup`.
    pub(crate) logup_proofs: Vec<LogUpProof<E>>,
    /// Witness commitments for this layer
    pub(crate) commitments: Vec<PCS::Commitment>,
    /// The sumcheck proof we use to make sure everything is evaluated at the same point.
    pub(crate) accumulation_proof: IOPProof<E>,
    /// The IO proof that links all claims to `last_claim` and the input
    pub(crate) io_proof: IOPProof<E>,
    /// Evaluations needed to verify the final claim of the accumulation proof and link it to the io proof
    pub(crate) acc_evals: Vec<E>,
    /// The claimed evaluations of the commitments
    pub(crate) evaluations: Vec<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LayerNormProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        let (nums, denoms): (Vec<Vec<E>>, Vec<Vec<E>>) = self
            .logup_proofs
            .iter()
            .map(|p| p.fractional_outputs())
            .unzip();

        (nums.concat(), denoms.concat())
    }
}

impl PadOp for LayerNorm<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        let LayerNorm {
            gamma,
            beta,
            eps,
            quant_info,
        } = self;
        let padded_gamma = gamma.pad_next_power_of_two();
        let padded_beta = beta.pad_next_power_of_two();

        Ok(LayerNorm::<Element> {
            gamma: padded_gamma,
            beta: padded_beta,
            eps,
            quant_info,
        })
    }
}

impl<E, PCS> ProvableOp<E, PCS> for LayerNorm<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = LayerNormCtx;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        // Check there is a single input
        ensure!(
            step_data.inputs.len() == 1,
            "LayerNorm step should only have one input, received {}",
            step_data.inputs.len()
        );
        let input_mle: ArcMultilinearExtension<E> = step_data.inputs[0]
            .get_data()
            .iter()
            .copied()
            .collect::<Vec<E>>()
            .into_mle()
            .into();
        let (claims, proof) = self.prove_step(node_id, last_claims, input_mle, prover)?;
        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<E, PCS>::LayerNorm(proof));

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
            "Found more than 1 input in inference step of LayerNorm layer"
        );
        ensure!(
            step_data.outputs.outputs().len() == 1,
            "Found more than 1 output in inference step of LayerNorm layer"
        );
        let layernorm_data = step_data.outputs.try_layernorm_data().ok_or(anyhow!(
            "LayerNorm data not found in inference step for LayerNorm layer"
        ))?;
        self.lookup_witness(id, ctx, layernorm_data)
    }
}

impl LayerNorm<Element> {
    pub(crate) fn prove_step<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: transcript::Transcript<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        input_poly: ArcMultilinearExtension<E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<(Vec<Claim<E>>, LayerNormProof<E, PCS>)> {
        // Check we have the correct number of claims
        ensure!(
            last_claims.len() == 1,
            "LayerNorm only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];

        let logup_witnesses = prover.lookup_witness(node_id)?;
        // Check that we have two LogUp witnesses
        ensure!(
            logup_witnesses.len() == 2,
            "LayerNorm requires two lookups but received {} lookup witnesses",
            logup_witnesses.len()
        );

        // Run the lookup protocol and return the lookup proof
        let (commits, logup_proofs): CommsAndProofs<PCS, E> = logup_witnesses
            .iter()
            .map(|logup_wit| {
                println!("PROVING LAYERNORM LOGUP");
                let commits = logup_wit.get_commitments();
                let logup_input = logup_wit.get_logup_input(&prover.challenge_storage)?;
                let logup_proof = batch_prove(&logup_input, prover.transcript)?;
                Ok((commits, logup_proof))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // We have checked that there are only two logup proofs and we expect them to be in the order inverse square root, range
        let inv_sqrt_claims = logup_proofs[0].output_claims();
        let range_claims = logup_proofs[1].output_claims();

        let num_vars = inv_sqrt_claims[0].point.len();
        // First perform a sumcheck so that all of our claims are evaluated at the same point
        let challenges_to_squeeze = ceil_log2(inv_sqrt_claims.len() + range_claims.len());
        let batching_challenge = (0..challenges_to_squeeze)
            .map(|_| {
                prover
                    .transcript
                    .get_and_append_challenge(b"batching")
                    .elements
            })
            .collect::<Vec<E>>();
        let rlc_terms = compute_betas_eval(&batching_challenge);

        let sqrt_eq: ArcMultilinearExtension<E> = compute_betas_eval(&inv_sqrt_claims[0].point)
            .into_mle()
            .into();
        let range_eq: ArcMultilinearExtension<E> =
            compute_betas_eval(&range_claims[0].point).into_mle().into();

        let eq_polys = std::iter::repeat_n(sqrt_eq, 2)
            .chain(std::iter::repeat_n(range_eq, range_claims.len()))
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        let vp = izip!(commits.iter().flatten().into_iter(), rlc_terms, eq_polys).fold(
            VirtualPolynomial::<E>::new(num_vars),
            |mut acc, ((_, poly), challenge, eq)| {
                acc.add_mle_list(vec![poly.clone().into(), eq], challenge);
                acc
            },
        );

        #[allow(deprecated)]
        let (accumulation_proof, accumulation_state) =
            IOPProverState::prove_parallel(vp, prover.transcript);
        let sumcheck_point = &accumulation_proof.point;
        let accumulation_evals = accumulation_state.get_mle_final_evaluations();
        let sqrt_in = accumulation_evals[0];
        let sqrt_out = accumulation_evals[2];
        let first_range = accumulation_evals[3];
        let acc_evals = [sqrt_in, sqrt_out, first_range]
            .into_iter()
            .chain(accumulation_evals.into_iter().skip(5))
            .collect::<Vec<E>>();
        // The lookups are performed over fewer variables than there are in the `last_claim` point because they sum over the final dimension before being handed to the lookup argument.
        let sum_dim_vars = ceil_log2(self.gamma.get_data().len());
        let two_inv = E::TWO.inverse();
        let two_mul = E::from_canonical_u64(1 << sum_dim_vars);

        // The input claim is multiplier * N * 2^k * eq(2^-1, ..,rk,..,rn,b) * input(b)*input(b) - 2^k * multiplier * eq(2^-1, ..,rk,..,rn,b) * input(b) * 2^k * eq(2^-1, ..,rk,..,rn,b) * input(b)
        let challenge = prover
            .transcript
            .get_and_append_challenge(b"batching")
            .elements;
        let second_challenge = prover
            .transcript
            .get_and_append_challenge(b"batching")
            .elements;
        let one_minus_challenge = E::ONE - challenge;
        let one_minus_second_challenge = E::ONE - second_challenge;

        let first_batch_chal = one_minus_challenge * one_minus_second_challenge;
        let second_batch_chal = challenge * one_minus_second_challenge;
        let third_batch_chal = one_minus_challenge * second_challenge;

        let full_point = std::iter::repeat_n(two_inv, sum_dim_vars)
            .chain(sumcheck_point.iter().copied())
            .collect::<Vec<E>>();
        let input_eq_poly: ArcMultilinearExtension<E> =
            compute_betas_eval(&full_point).into_mle().into();

        // Construct the VirtualPolynomial
        let mut vp = VirtualPolynomial::<E>::new(full_point.len());
        let (dim_size, multiplier) = self
            .quant_info
            .as_ref()
            .map(|q| (q.dim_size, q.multiplier))
            .ok_or(anyhow!(
                "No quantisation data present for LayerNorm during proving"
            ))?;
        let dim_size_field = E::from_canonical_u64(dim_size as u64);
        let multiplier_field: E = multiplier.to_field();
        vp.add_mle_list(
            vec![
                input_eq_poly.clone(),
                input_poly.clone(),
                input_poly.clone(),
            ],
            first_batch_chal * multiplier_field * dim_size_field * two_mul,
        );
        vp.add_mle_list(
            vec![
                input_eq_poly.clone(),
                input_eq_poly.clone(),
                input_poly.clone(),
                input_poly.clone(),
            ],
            -first_batch_chal * multiplier_field * two_mul * two_mul,
        );

        // `last_claim.eval` should be equal to `gamma(b) * eq(r,b)*(N * input(b))*inv_sqrt_out(b) - 2^k* gamma(b) * eq(2^-1,..,rk,..,rn,b)*input(b)*inv_qrt_out(b) + beta(b)`
        let last_claim_point = &last_claim.point;
        // We need to repeat the gamma and beta evals the correct number of times
        let number_of_repeats = 1usize << (last_claim_point.len() - sum_dim_vars);
        let gamma_poly: ArcMultilinearExtension<E> = std::iter::repeat_n(
            self.gamma
                .get_data()
                .iter()
                .map(<Element as Fieldizer<E>>::to_field)
                .collect::<Vec<E>>(),
            number_of_repeats,
        )
        .flatten()
        .collect::<Vec<E>>()
        .into_mle()
        .into();
        let beta_poly: ArcMultilinearExtension<E> = std::iter::repeat_n(
            self.beta
                .get_data()
                .iter()
                .map(<Element as Fieldizer<E>>::to_field)
                .collect::<Vec<E>>(),
            number_of_repeats,
        )
        .flatten()
        .collect::<Vec<E>>()
        .into_mle()
        .into();

        let last_claim_eq: ArcMultilinearExtension<E> =
            compute_betas_eval(last_claim_point).into_mle().into();
        let last_claim_sum_eq: ArcMultilinearExtension<E> = compute_betas_eval(
            &std::iter::repeat_n(two_inv, sum_dim_vars)
                .chain(last_claim_point.iter().skip(sum_dim_vars).copied())
                .collect::<Vec<E>>(),
        )
        .into_mle()
        .into();
        let number_repeats_inv_sqrt = 1usize << sum_dim_vars;

        let inv_sqrt_out_poly: ArcMultilinearExtension<E> = commits[0][1]
            .1
            .get_base_field_vec()
            .iter()
            .flat_map(|&eval| vec![eval; number_repeats_inv_sqrt])
            .collect::<Vec<E::BaseField>>()
            .into_mle()
            .into();

        vp.add_mle_list(
            vec![
                last_claim_eq.clone(),
                gamma_poly.clone(),
                input_poly.clone(),
                inv_sqrt_out_poly.clone(),
            ],
            second_batch_chal * dim_size_field,
        );

        let gamma_poly2: ArcMultilinearExtension<E> = self
            .gamma
            .get_data()
            .iter()
            .flat_map(|v| vec![<Element as Fieldizer<E>>::to_field(v); number_of_repeats])
            .collect::<Vec<E>>()
            .into_mle()
            .into();

        let maybe_last_claim = izip!(
            last_claim_eq.get_ext_field_vec(),
            gamma_poly.get_ext_field_vec(),
            input_poly.get_ext_field_vec(),
            inv_sqrt_out_poly.get_base_field_vec(),
            last_claim_sum_eq.get_ext_field_vec(),
            beta_poly.get_ext_field_vec(),
        )
        .enumerate()
        .fold(
            E::ZERO,
            |acc, (i, (&lceq, &gam, &input, &sqrt, &lcsum, &beta))| {
                acc + gam * E::from_base(sqrt) * input * (dim_size_field * lceq - two_mul * lcsum)
                    + lceq * beta
            },
        );
        println!(
            "Maybe last claim: {:?}, last claim: {:?}",
            maybe_last_claim, last_claim.eval
        );
        vp.add_mle_list(
            vec![
                last_claim_sum_eq,
                gamma_poly,
                input_poly,
                inv_sqrt_out_poly.clone(),
            ],
            -second_batch_chal * two_mul,
        );
        vp.add_mle_list(vec![last_claim_eq, beta_poly], second_batch_chal);

        vp.add_mle_list(
            vec![input_eq_poly, inv_sqrt_out_poly],
            third_batch_chal * two_mul,
        );
        #[allow(deprecated)]
        let (io_proof, io_state) = IOPProverState::prove_parallel(vp, prover.transcript);
        let io_point = &io_proof.point;
        let io_evaluations = io_state.get_mle_final_evaluations();
        let input_eval = io_evaluations[1];
        let gamma_eval = io_evaluations[3];
        let inv_sqrt_out_eval = io_evaluations[4];
        let beta_eval = io_evaluations[6];

        let input_claim = Claim::<E>::new(io_point.to_vec(), input_eval);
        // Add the witness claims to the commitment prover
        let inv_sqrt_out_claim = Claim::<E>::new(
            io_point
                .iter()
                .skip(sum_dim_vars)
                .copied()
                .collect::<Vec<E>>(),
            inv_sqrt_out_eval,
        );

        let (commitments, evaluations): (Vec<PCS::Commitment>, Vec<E>) =
            [inv_sqrt_claims[0].clone(), inv_sqrt_out_claim]
                .into_iter()
                .chain(range_claims.iter().cloned())
                .zip(commits.iter().flatten().into_iter())
                .map(|(claim, comm_with_wit)| {
                    let comm = PCS::get_pure_commitment(&comm_with_wit.0);
                    let eval = claim.eval;
                    prover
                        .commit_prover
                        .add_witness_claim(comm_with_wit.clone(), claim)?;
                    Ok((comm, eval))
                })
                .collect::<Result<Vec<(PCS::Commitment, E)>, anyhow::Error>>()?
                .into_iter()
                .unzip();

        // Add common commitment claims to be proven
        let common_claims = {
            let point = io_point
                .iter()
                .take(sum_dim_vars)
                .copied()
                .collect::<Vec<E>>();
            let mut claims = HashMap::new();
            claims.insert(
                GAMMA_POLY_ID.to_string(),
                Claim::<E>::new(point.clone(), gamma_eval),
            );
            claims.insert(
                BETA_POLY_ID.to_string(),
                Claim::<E>::new(point.clone(), beta_eval),
            );
            claims
        };
        prover.add_common_claims(node_id, common_claims)?;

        let proof = LayerNormProof::<E, PCS> {
            logup_proofs,
            commitments,
            accumulation_proof,
            io_proof,
            acc_evals,
            evaluations,
        };

        Ok((vec![input_claim], proof))
    }
    /// Internal method for generating the [`LogUpWitness`] for a [`LayerNorm`] step.
    fn lookup_witness<E, PCS>(
        &self,
        id: NodeId,
        ctx: &Context<E, PCS>,
        layernorm_data: &LayerNormData<E>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let mut gen = LookupWitnessGen::<E, PCS>::default();
        // Get the data generated during quantised evaluation
        let LayerNormData {
            lookup_input,
            lookup_output,
            range_check,
            ..
        } = layernorm_data;

        let num_vars = ceil_log2(lookup_input.len());

        // We need to work out how many chunks to split the shifted away part into to be range checked
        let QuantisedLayerNormData {
            range_check_bits,
            top_chunk_scalar_log,
            ..
        } = self.quant_info().ok_or(anyhow!(
            "Could not prove LayerNorm because it had no quantisation data"
        ))?;
        let number_range_checks = (range_check_bits - 1) / *quantization::BIT_LEN + 1;

        // Split `range_check` into its constituent parts
        let range_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        let top_chunk_scalar: Element = 1 << top_chunk_scalar_log;
        let range_checks = (0..number_range_checks)
            .into_par_iter()
            .map(|j| {
                if j != number_range_checks - 1 {
                    range_check
                        .iter()
                        .map(|&elem| {
                            let tmp = elem >> (j * *quantization::BIT_LEN);
                            tmp & range_mask
                        })
                        .collect::<Vec<Element>>()
                } else {
                    // In the final chunk after being shifted everything has to get multiplied by 1 << top_chunk_scalar_log
                    range_check
                        .iter()
                        .map(|&elem| {
                            let tmp = elem >> (j * *quantization::BIT_LEN);
                            (tmp & range_mask) * top_chunk_scalar
                        })
                        .collect::<Vec<Element>>()
                }
            })
            .collect::<Vec<Vec<Element>>>();
        let range_elements_count =
            range_checks
                .iter()
                .fold(HashMap::<Element, u64>::new(), |mut acc, range_check| {
                    range_check
                        .iter()
                        .for_each(|v| *acc.entry(*v).or_default() += 1);
                    acc
                });

        let inv_sqrt_element_count = lookup_input.iter().zip(lookup_output.iter()).fold(
            HashMap::<Element, u64>::new(),
            |mut acc, (&input, &output)| {
                *acc.entry(input + output * COLUMN_SEPARATOR).or_default() += 1;
                acc
            },
        );

        // Make the commitments to the lookups
        let (mut commits, mut evals): CommsAndEvals<PCS, E> = [lookup_input, lookup_output]
            .into_par_iter()
            .chain(range_checks.par_iter())
            .map(|vals| {
                let evaluations = vals
                    .iter()
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
        // Split off the commits and evaluations for the inverse square root lookup
        let inv_sqrt_commits = commits
            .drain(..2)
            .collect::<Vec<(PCS::CommitmentWithWitness, DenseMultilinearExtension<E>)>>();
        let inv_sqrt_evals = evals.drain(..2).collect::<Vec<Vec<E::BaseField>>>();

        // Add the merged columsn to the lookups lists
        gen.element_count
            .insert(TableType::Range, range_elements_count);

        gen.element_count.insert(
            TableType::InverseSQRT(self.eps.to_bits(), *range_check_bits),
            inv_sqrt_element_count,
        );

        // Insert the LogUpWitnesses
        gen.logup_witnesses.insert(
            id,
            vec![
                LogUpWitness::<E, PCS>::new_lookup(
                    inv_sqrt_commits,
                    inv_sqrt_evals.to_vec(),
                    2,
                    TableType::InverseSQRT(self.eps.to_bits(), *range_check_bits),
                ),
                LogUpWitness::<E, PCS>::new_lookup(commits, evals, 1, TableType::Range),
            ],
        );
        Ok(gen)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for LayerNormCtx
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = LayerNormProof<E, PCS>;

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
            "LayerNorm only outputs 1 claim, received {} while verifying LayerNorm step",
            last_claims.len()
        );

        let last_claim = last_claims[0];

        // First we verify the LogUp proofs
        // Retrieve the challenges used in the logup proofs
        let logup_challenges: Vec<(E, E)> = [
            TableType::InverseSQRT(self.eps, self.range_check_bits),
            TableType::Range,
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
        .collect::<Result<Vec<(E, E)>, anyhow::Error>>()?;

        let LayerNormProof {
            logup_proofs,
            commitments,
            accumulation_proof,
            io_proof,
            acc_evals,
            evaluations,
        } = proof;

        ensure!(
            logup_proofs.len() == 2,
            "Expected 2 LogUp proofs for LayerNorm verification, received: {}",
            logup_proofs.len()
        );

        // Work out how many range check instances there are
        let num_range_checks = (self.range_check_bits - 1) / *quantization::BIT_LEN + 1;
        let instances_per_proof = [1, num_range_checks];

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

        let inv_sqrt_claims = logup_claims[0].claims();
        let range_claims = logup_claims[1].claims();

        // Squeeze challenges to batch the initial claims
        let challenges_to_squeeze = ceil_log2(inv_sqrt_claims.len() + range_claims.len());
        let batching_challenge = (0..challenges_to_squeeze)
            .map(|_| {
                verifier
                    .transcript
                    .get_and_append_challenge(b"batching")
                    .elements
            })
            .collect::<Vec<E>>();
        let rlc_terms = compute_betas_eval(&batching_challenge);
        // Build the initial evaluation for the accumulatio n sumcheck
        let accumulation_inital_eval = inv_sqrt_claims
            .iter()
            .chain(range_claims.iter())
            .zip(rlc_terms.iter())
            .fold(E::ZERO, |acc, (claim, &chal)| acc + claim.eval * chal);
        let aux_info =
            VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![inv_sqrt_claims[0].point.len(); 2]]);
        let accumulation_subclaim = IOPVerifierState::verify(
            accumulation_inital_eval,
            accumulation_proof,
            &aux_info,
            verifier.transcript,
        );
        let accumulation_point = accumulation_subclaim.point_flat();

        // Make the eq poly evals
        let eq_sqrt = eq_xy_eval(&inv_sqrt_claims[0].point, &accumulation_point);
        let range_eq_eval = eq_xy_eval(&range_claims[0].point, &accumulation_point);

        // Now we check the output claim from the accumulation sumcheck
        let eq_evals = std::iter::repeat_n(eq_sqrt, 2)
            .chain(std::iter::repeat_n(range_eq_eval, num_range_checks))
            .collect::<Vec<E>>();
        let calculated_claim = izip!(acc_evals.iter(), eq_evals, rlc_terms)
            .fold(E::ZERO, |acc, (&eval, eq, chal)| acc + eval * eq * chal);

        ensure!(
            calculated_claim == accumulation_subclaim.expected_evaluation,
            "LayerNorm verifiaction failed calculated claim for accumulation proof: {:?}, did not equal the expected claim: {:?}",
            calculated_claim,
            accumulation_subclaim.expected_evaluation
        );
        // Now we build the inital evaluation for the io_sumcheck
        let challenge = verifier
            .transcript
            .get_and_append_challenge(b"batching")
            .elements;
        let second_challenge = verifier
            .transcript
            .get_and_append_challenge(b"batching")
            .elements;
        let one_minus_challenge = E::ONE - challenge;
        let one_minus_second_challenge = E::ONE - second_challenge;

        let first_batch_chal = one_minus_challenge * one_minus_second_challenge;
        let second_batch_chal = challenge * one_minus_second_challenge;
        let third_batch_chal = one_minus_challenge * second_challenge;

        let pow_two_multiplier = E::from_canonical_u64(1 << *quantization::BIT_LEN);
        let (partial_eval, power_two) = acc_evals.iter().skip(2).take(range_claims.len() - 1).fold(
            (
                acc_evals[0] * E::from_canonical_u64(1 << self.range_check_bits),
                E::ONE,
            ),
            |(acc, pow), &eval| (acc + eval * pow, pow * pow_two_multiplier),
        );
        // The last range evaluation has to be rescaled
        let top_chunk_scalar_inv = E::from_canonical_u64(1 << self.top_chunk_scalar_log).inverse();

        let first_term_eval = first_batch_chal
            * (partial_eval + *acc_evals.last().unwrap() * top_chunk_scalar_inv * power_two);
        let second_term_eval = second_batch_chal * last_claim.eval;
        let third_term_eval = acc_evals[1] * third_batch_chal;

        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![last_claim.point.len(); 4]]);
        let io_subclaim = IOPVerifierState::verify(
            first_term_eval + second_term_eval + third_term_eval,
            io_proof,
            &aux_info,
            verifier.transcript,
        );
        println!("got to end of verificaiton so far");
        Ok(vec![])
        // todo!()
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;

    use crate::{
        layers::Layer,
        model::{Model, test::prove_model},
    };

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
        let gamma = Tensor::<f32>::random(&vec![100].into());
        let beta = Tensor::<f32>::random(&vec![100].into());
        let eps = 1e-5;
        let layernorm = LayerNorm {
            gamma,
            beta,
            eps,
            quant_info: None,
        };
        // Make a random float input tensor and derive the input ScalingFactor
        let input_tensor = Tensor::<f32>::random(&vec![2, 100].into());
        let input_scaling = ScalingFactor::from_tensor(&input_tensor, None);
        // Construct the quantised LayerNorm
        let (quant_layernorm, intermediate_bit_size, output_scaling) =
            layernorm.quantise(input_scaling, input_scaling).unwrap();
        // We quantise the float input to obtain `quant_tensor` and then we dequantise to obtain `dequant_input`
        // this lets us run quantised evaluation and floating point evaluation and compare the outputs.
        let quant_tensor = input_tensor.quantize(&input_scaling);
        let dequant_input = quant_tensor.dequantize(&input_scaling);

        let dequant_output = layernorm
            .evaluate::<E>(&[&dequant_input], vec![vec![2, 100].into()])
            .unwrap()
            .outputs[0]
            .clone();

        let quant_output = quant_layernorm
            .evaluate::<E>(&[&quant_tensor], vec![vec![2, 100].into()])
            .unwrap()
            .outputs[0]
            .clone();

        let quant_output_dequant = quant_output.dequantize(&output_scaling);
        let maximum_float = quant_output_dequant.max_abs_output();
        println!("maximum float out: {maximum_float}");
        for (qdqo, dequant_o) in quant_output_dequant
            .get_data_into()
            .chunks(100)
            .into_iter()
            .zip(dequant_output.get_data_into().chunks(100))
        {
            // Check the L1 error for the row
            let sum = qdqo
                .iter()
                .zip(dequant_o.iter())
                .map(|(a, b)| a - b)
                .sum::<f32>();
            println!("row sum error: {sum}");
            // Check that the values are within 0.01 of the float output
            for (&q, &d) in qdqo.iter().zip(dequant_o.iter()) {
                assert!(
                    (q - d).abs() < 0.05f32,
                    "Values {q}, {d}, with abs diff {} which is bigger than 0.05",
                    (q - d).abs(),
                );
            }
        }
        println!("intermediate bit size: {}", intermediate_bit_size);
        println!("bias scaling log: {}", output_scaling.scale().log2());
    }

    #[test]
    fn test_layernorm_proving() {
        let gamma = Tensor::<f32>::random(&vec![100].into());
        let beta = Tensor::<f32>::random(&vec![100].into());
        let eps = 1e-5;
        let layernorm = LayerNorm {
            gamma,
            beta,
            eps,
            quant_info: None,
        };

        let mut model =
            Model::new_from_input_shapes(vec![vec![16, 100].into()], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::LayerNorm(layernorm), None)
            .unwrap();

        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }
}
