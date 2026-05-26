use crate::{
    ScalingStrategy, VectorTranscript,
    iop::{context::ShapeStep, prover::BatchFFTProof},
    layers::{hadamard, requant::Requant},
    model::StepData,
    padding::{PaddingMode, ShapeInfo, pad_conv},
    quantization::{BIT_LEN, TensorFielder},
    tensor::Shape,
};
use core::f32;
use std::collections::HashMap;

use crate::{
    Claim, Prover,
    commit::{compute_betas_eval, identity_eval},
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerProof, provable::ProvingData},
    quantization::{self, ScalingFactor},
    tensor::{ConvData, Number, get_root_of_unity},
};
use anyhow::{Context, Result, ensure};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use mpcs::PolynomialCommitmentScheme;
// use itertools::assert_equal;
use crate::{
    Element,
    quantization::Fieldizer,
    tensor::{Tensor, fft},
};
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::{info, warn};
use transcript::Transcript;

use super::{
    LayerCtx,
    provable::{
        Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
        QuantizeOutput, VerifiableCtx,
    },
};

const IS_PROVABLE: bool = true;
/// Convolution layer description (weights)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Convolution<T> {
    /// NOTE: in the case of f32, the weights are native
    /// In the case of Element (i128), the weights are already fft'd
    pub filter: Tensor<T>,
    /// Same for bias.
    pub bias: Tensor<T>,
    /// Unpadded shape of the filter. This is set to filter's shape in case of no padding.
    pub unpadded_shape: Shape,
    /// Spatial zero-padding applied to H and W dimensions of the input before convolution.
    /// Equivalent to ONNX `pads` = [pad_h, pad_w, pad_h, pad_w].
    /// Default is [0, 0] (valid / no-padding convolution).
    #[serde(default)]
    pub input_padding: [usize; 2],
    /// Spatial stride applied to H and W dimensions of the output after convolution.
    /// Equivalent to ONNX `strides` = [stride_h, stride_w].
    /// Default is [1, 1] (unstrided convolution).
    #[serde(default = "default_stride")]
    pub stride: [usize; 2],
}

fn default_stride() -> [usize; 2] {
    [1, 1]
}

/// Info about the convolution layer derived during the setup phase
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvCtx<E> {
    pub node_id: NodeId,
    pub fft_aux: VPAuxInfo<E>,
    pub fft_weights_aux: VPAuxInfo<E>,
    pub ifft_aux: VPAuxInfo<E>,
    pub delegation_fft: Vec<VPAuxInfo<E>>,
    pub delegation_fft_weights: Vec<VPAuxInfo<E>>,
    pub delegation_ifft: Vec<VPAuxInfo<E>>,
    pub hadamard: VPAuxInfo<E>,
    pub kw: usize,
    pub kx: usize,
    pub real_nw: usize,
    pub nw: usize,
    pub filter_size: usize,
    pub unpadded_filter_shape: Shape,
    pub padded_filter_shape: Shape,
    /// Spatial zero-padding applied to input before convolution; mirrors `Convolution::input_padding`.
    #[serde(default)]
    pub input_padding: [usize; 2],
    /// Spatial stride; mirrors `Convolution::stride`.
    #[serde(default = "default_stride")]
    pub stride: [usize; 2],
    /// Unpadded (pre-POT) input shape seen by this conv layer (e.g. [C, H, W] before POT padding).
    /// Used by `verify_input_claim` to reconstruct the ONNX-spatially-padded FFT input when
    /// `input_padding` is non-zero.  Defaults to empty (treated as "unknown / not needed").
    #[serde(default)]
    pub unpadded_input_shape: Shape,
}

pub fn to_bits<E: ExtensionField>(mut num: usize, bitlen: usize) -> Vec<E> {
    let mut bits = vec![E::ZERO; bitlen];
    for bit in bits.iter_mut().take(bitlen) {
        *bit = E::from_canonical_u64((num & 1) as u64);
        num >>= 1;
    }
    bits
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchoolBookConv<T>(pub(crate) Convolution<T>);

/// Contains proof material related to one step of the inference for a convolution layer
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ConvProof<E: ExtensionField> {
    // Sumcheck proof for the FFT layer
    fft_proof: IOPProof<E>,
    fft_proof_weights: IOPProof<E>,
    // Proof for the evaluation delegation of the omegas matrix
    // It consists of multiple sumcheck proofs
    fft_delegation_proof: Vec<IOPProof<E>>,
    fft_delegation_proof_weights: Vec<IOPProof<E>>,
    // Likewise for fft, we define ifft proofs
    ifft_proof: IOPProof<E>,
    ifft_delegation_proof: Vec<IOPProof<E>>,
    // Sumcheck proof for the hadamard product
    hadamard_proof: IOPProof<E>,
    // The evaluation claims produced by the corresponding sumchecks
    fft_claims: Vec<E>,
    fft_weight_claims: Vec<E>,
    ifft_claims: Vec<E>,
    fft_delegation_claims: Vec<Vec<E>>,
    fft_delegation_weights_claims: Vec<Vec<E>>,
    ifft_delegation_claims: Vec<Vec<E>>,
    partial_evals: Vec<E>,
    hadamard_clams: Vec<E>,
    bias_claim: E,
    clearing_proof: hadamard::HadamardProof<E>,
}

impl<T: Number> Convolution<T> {
    pub fn new(filter: Tensor<T>, bias: Tensor<T>) -> Self {
        assert_eq!(filter.kw(), bias.get_shape()[0]);
        assert_eq!(filter.get_shape().len(), 4);
        let filter_shape = filter.get_shape();
        Self::new_padded(filter, bias, &filter_shape)
    }

    pub fn new_with_padding(filter: Tensor<T>, bias: Tensor<T>, input_padding: [usize; 2]) -> Self {
        assert_eq!(filter.kw(), bias.get_shape()[0]);
        assert_eq!(filter.get_shape().len(), 4);
        let filter_shape = filter.get_shape();
        Self {
            filter,
            bias,
            unpadded_shape: filter_shape,
            input_padding,
            stride: [1, 1],
        }
    }

    pub fn new_with_padding_and_stride(
        filter: Tensor<T>,
        bias: Tensor<T>,
        input_padding: [usize; 2],
        stride: [usize; 2],
    ) -> Self {
        assert_eq!(filter.kw(), bias.get_shape()[0]);
        assert_eq!(filter.get_shape().len(), 4);
        let filter_shape = filter.get_shape();
        Self {
            filter,
            bias,
            unpadded_shape: filter_shape,
            input_padding,
            stride,
        }
    }

    pub(crate) fn new_without_bias(filter: Tensor<T>) -> Self {
        let bias = Tensor::zeros(Shape::new(vec![filter.kw()]));
        Self::new(filter, bias)
    }

    pub fn new_padded(filter: Tensor<T>, bias: Tensor<T>, unpadded_shape: &Shape) -> Self {
        assert_eq!(filter.kw(), bias.get_shape()[0]);
        Self {
            filter,
            bias,
            unpadded_shape: unpadded_shape.clone(),
            input_padding: [0, 0],
            stride: [1, 1],
        }
    }
    pub fn output_shape(&self, input_shape: &Shape, padding_mode: PaddingMode) -> Shape {
        let [ph, pw] = self.input_padding;
        let [sh, sw] = self.stride;
        match padding_mode {
            // unpadded shape is the shape found in onxx file for example
            PaddingMode::NoPadding => conv2d_shape_with_padding_and_stride(
                input_shape,
                &self.unpadded_shape,
                ph,
                pw,
                sh,
                sw,
            ),
            PaddingMode::Padding => padded_conv2d_shape_with_padding_and_stride(
                input_shape,
                &self.filter.real_shape(),
                ph,
                pw,
                sh,
                sw,
            ),
        }
    }

    pub fn add_bias(&self, conv_out: &Tensor<T>) -> Tensor<T> {
        let mut arr = conv_out.data.clone();
        assert_eq!(conv_out.data.len(), conv_out.kw() * conv_out.filter_size());
        for i in 0..conv_out.kw() {
            for j in 0..conv_out.filter_size() {
                arr[i * conv_out.filter_size() + j] += self.bias.data[i];
            }
        }
        Tensor::new(conv_out.get_shape(), arr)
    }

    /// Retrieves an element using (N, C, H, W) indexing
    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> T {
        assert!(self.filter.get_shape().len() <= 4);

        let (n_size, c_size, h_size, w_size) = self.filter.get4d();

        assert!(n < n_size);
        assert!(c < c_size);
        assert!(h < h_size);
        assert!(w < w_size);
        let flat_index = n * (c_size * h_size * w_size) + c * (h_size * w_size) + h * w_size + w;
        self.filter.get_data()[flat_index]
    }

    pub fn get_shape(&self) -> Shape {
        self.filter.get_shape()
    }

    pub fn kw(&self) -> usize {
        self.filter.kw()
    }

    pub fn kx(&self) -> usize {
        self.filter.kx()
    }

    pub fn nw(&self) -> usize {
        self.filter.nw()
    }

    pub fn ncols_2d(&self) -> usize {
        self.filter.ncols_2d()
    }

    pub fn nrows_2d(&self) -> usize {
        self.filter.nrows_2d()
    }
    pub fn filter_size(&self) -> usize {
        self.filter.filter_size()
    }

    fn num_outputs(num_inputs: usize) -> usize {
        assert_eq!(num_inputs, 1);
        1
    }
}

impl<T: Number> OpInfo for Convolution<T> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Self::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Conv: ({},{},{},{})",
            self.filter.kw(),
            self.filter.kx(),
            self.filter.nw(),
            self.filter.nw()
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl Evaluate<f32> for Convolution<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating convolution layer"
        );
        let input = inputs[0];
        let [ph, pw] = self.input_padding;
        let [sh, sw] = self.stride;
        assert_eq!(sh, sw, "only square strides are supported");
        let padded = input.zero_pad_spatial(ph, pw);
        Ok(LayerOut::from_vec(vec![padded.conv2d(
            &self.filter,
            &self.bias,
            sh,
        )]))
    }
}

impl Convolution<f32> {
    /// Quantizes the filter and the bias.
    /// It uses a custom scaling factor `bias_s` for the bias, if provided,
    /// otherwise the same scaling factor of the weights (i.e., `s`) is used
    pub fn quantize(self, s: &ScalingFactor, bias_s: &ScalingFactor) -> Convolution<Element> {
        let quantized_filter = self.filter.quantize(s);
        let bias = self.bias.quantize(bias_s);
        let mut q = Convolution::<Element>::new(quantized_filter, bias);
        q.input_padding = self.input_padding;
        q.stride = self.stride;
        q
    }

    pub fn op<E: ExtensionField>(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let [ph, pw] = self.input_padding;
        let [sh, sw] = self.stride;
        assert_eq!(sh, sw, "only square strides are supported");
        let padded = input.zero_pad_spatial(ph, pw);
        padded.conv2d(&self.filter, &self.bias, sh)
    }

    pub fn max_abs_weight(&self) -> f32 {
        let max_weight = self.filter.max_abs_output();
        let max_bias = self.bias.max_abs_output();
        let distance = (max_weight - max_bias).abs() / max_weight;
        if distance > 0.1 {
            warn!(
                "max_abs_weight CONV: distance between max_weight and max_bias is too large: {:.2}%",
                distance * 100.0
            );
        }
        self.filter.max_abs_output().max(self.bias.max_abs_output())
    }
}

impl Evaluate<Element> for Convolution<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<Element, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating convolution layer"
        );
        let input = inputs[0];
        ensure!(
            unpadded_input_shapes.len() == 1,
            "Found more than 1 input shape when evaluating convolution layer"
        );
        let (output, proving_data) = self.op(input, &unpadded_input_shapes[0]);
        Ok(LayerOut {
            outputs: vec![output],
            proving_data: ProvingData::Convolution(proving_data),
        })
    }
}

impl Convolution<Element> {
    /// Returns the effective input shape after applying `input_padding` to the spatial dims.
    ///
    /// The FFT-based convolution operates on the spatially-padded input, so we compute
    /// the filter's FFT with respect to this larger shape.
    pub fn effective_input_shape(&self, unpadded_input_shape: &Shape) -> Shape {
        let [ph, pw] = self.input_padding;
        if ph == 0 && pw == 0 {
            return unpadded_input_shape.clone();
        }
        // unpadded_input_shape is [C, H, W]
        assert_eq!(
            unpadded_input_shape.len(),
            3,
            "expected 3-D input shape [C,H,W]"
        );
        let c = unpadded_input_shape[0];
        let h = unpadded_input_shape[1];
        let w = unpadded_input_shape[2];
        Shape::new(vec![c, h + 2 * ph, w + 2 * pw])
    }

    /// Pads the filter and bias, and adapt the filter to the convolution fft operation.
    pub fn into_padded_and_ffted(mut self, unpadded_input_shape: &Shape) -> Self {
        self.filter = self.filter.pad_next_power_of_two();
        self.bias = self.bias.pad_next_power_of_two();
        // Use the spatially-padded input shape so the FFT is sized correctly.
        let effective_shape = self.effective_input_shape(unpadded_input_shape);
        let padded_input_shape = effective_shape
            .iter()
            .map(|&x| x.next_power_of_two())
            .collect::<Shape>();
        self.filter = self.filter.into_fft_conv(&padded_input_shape);
        self
    }

    pub fn op<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
        unpadded_input_shape: &Shape,
    ) -> (Tensor<Element>, ConvData<E>) {
        // `input` is POT-padded. When there is ONNX spatial zero-padding we must:
        //   1. Crop back to unpadded_input_shape (remove POT padding)
        //   2. Apply ONNX spatial zero-padding
        //   3. Re-apply POT padding for the FFT
        // When there is no ONNX padding the input is used as-is.
        let [ph, pw] = self.input_padding;
        let [sh, sw] = self.stride;
        assert_eq!(sh, sw, "only square strides are supported");
        let fft_input: Tensor<Element> = if ph == 0 && pw == 0 {
            input.clone()
        } else {
            // Crop `input` to the unpadded shape, then zero-pad spatially, then POT-pad.
            // `unpadded_input_shape` is [C, H, W] (3-D).
            let unpadded = input.crop_to(unpadded_input_shape);
            let spatially = unpadded.zero_pad_spatial(ph, pw);
            spatially.pad_next_power_of_two()
        };
        let (output, mut proving_data) = self.filter.fft_conv(&fft_input);
        let conv_output = self.add_bias(&output);
        // we record here the full-resolution output _after_ the bias addition.
        // For strided conv, this is the full-res (unstrided) output.
        // The prover needs these values to reconstruct conv_after_bias for the clearing hadamard proof.
        proving_data.set_output(conv_output.get_data());
        // At this stage, we're creating a "garbage clearing" tensor that sets all garbage values to 0.
        // This also zeroes non-stride-multiple positions so that the prover's claim on the
        // full-res output is consistent with the strided output delivered downstream.
        let effective_unpadded_input = self.effective_input_shape(unpadded_input_shape);
        // Full-resolution (unstrided) valid output shape
        let full_unpadded_output_shape =
            conv2d_shape(&effective_unpadded_input, &self.unpadded_shape);
        debug_assert_eq!(
            { padded_conv2d_shape(&fft_input.get_shape(), &self.filter.real_shape(),) },
            conv_output.get_shape(),
            "FFT output shape not computable"
        );

        if sh == 1 {
            // No stride: original path.
            let cleared_tensor = clear_garbage(&conv_output, &full_unpadded_output_shape);
            debug_assert!({
                let clearing_tensor =
                    new_clearing_tensor(&full_unpadded_output_shape, &conv_output.get_shape());
                let cleared_tensor2 = conv_output.flatten().mul(&clearing_tensor);
                cleared_tensor.get_data() == cleared_tensor2.get_data()
            });
            (cleared_tensor, proving_data)
        } else {
            // Strided path:
            //   1. Clear the full-res FFT output (zero out garbage AND non-stride positions)
            //   2. Compact into the strided shape [C, H/s, W/s] and POT-pad
            //
            // The full-res cleared tensor (with zeros at non-stride positions) is what the
            // conv proving circuit sees — it proves that the clearing hadamard is correct.
            // The strided compact tensor is what the downstream layers receive.
            let cleared_tensor =
                clear_garbage_strided(&conv_output, &full_unpadded_output_shape, sh);
            // Compact the strided values into shape [C, H_s, W_s] and POT-pad
            let strided_compact = compact_strided(&cleared_tensor, &full_unpadded_output_shape, sh);
            (strided_compact, proving_data)
        }
    }

    /// Returns the min and max output range of the convolution layer for a given input range.
    /// NOTE: it assumes the weights in float are NOT fft'd
    pub fn output_range(&self, _min_input: Element, _max_input: Element) -> (Element, Element) {
        // 2^{BIT_LEN + log2(k_h * k_w * k_c)}
        let (_k_n, k_c, k_h, k_w) = self.filter.get4d();
        let exp = 2 * *quantization::BIT_LEN + ceil_log2(k_h * k_w * k_c + 1);
        let min = -(2u64.pow(exp as u32) as Element);
        let max = 2u64.pow(exp as u32) as Element;
        (min, max)
    }

    /// Returns the maximum bitsize of the output of this layer
    pub fn output_bitsize(&self) -> usize {
        // 2^{BIT_LEN + log2(k_h * k_w * k_c)}
        let (_k_n, k_c, k_h, k_w) = self.filter.get4d();
        2 * (*quantization::BIT_LEN - 1) + ceil_log2(k_h * k_w * k_c + 1)
    }

    pub fn prove_batch_fft_weights<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        r: Vec<E>,
    ) -> BatchFFTWeightsProof<E>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
    {
        let padded_rows = 2 * self.filter.nw() * self.filter.nw();
        let mut w1_reduced: Vec<E> = vec![E::ZERO; self.filter.real_nw() * self.filter.real_nw()];

        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; padded_rows.ilog2() as usize];
        let mut r2 = vec![E::ZERO; r.len() - padded_rows.ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);

        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; padded_rows];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        let beta = compute_betas_eval(&r2);
        prover.phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            E::ONE,
            padded_rows.ilog2() as usize,
            false,
        );
        // compute X(i,r2)
        let filter_size = self.filter.real_nw() * self.filter.real_nw();
        (0..self.filter.kw()).for_each(|i| {
            (0..self.filter.kx()).for_each(|j| {
                (0..filter_size).for_each(|k| {
                    let index = i * filter_size * self.filter.kx() + j * filter_size + k;
                    let v: E = self.filter.data[index].to_field();
                    w1_reduced[k] += beta[i * self.filter.kx() + j] * v;
                });
            });
        });
        // for i in 0..self.filter.kw(){
        // for j in 0..self.filter.kx(){
        // for k in 0..filter_size{
        // let v: E = self.filter.data[i*filter_size*self.filter.kx() + j*filter_size + k].to_field();
        // W1_reduced[k] += beta[i*self.filter.kx()+j]*v;
        // }
        // }
        // }

        let partial_evals = w1_reduced.clone();
        w1_reduced = index_wf(
            &w1_reduced.clone(),
            self.filter.real_nw(),
            self.filter.nw(),
            padded_rows,
        )
        .collect::<Vec<E>>();
        let f_m = w1_reduced.into_mle();

        // f_m.fix_high_variables_in_place(&r2);

        // Construct the virtual polynomial and run the sumcheck prover

        let f_red = w_red.into_mle();

        let mut vp = VirtualPolynomial::<E>::new(f_m.num_vars);
        vp.add_mle_list(vec![f_m.clone().into(), f_red.clone().into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        let claims = state.get_mle_final_evaluations();

        let out_point = proof.point.clone();
        BatchFFTWeightsProof {
            proof,
            claims,
            partial_evals,
            matrix_evaluation: prover.delegate_matrix_evaluation(
                &mut f_middle,
                r1.clone(),
                out_point,
                false,
            ),
        }
    }
}

pub struct BatchFFTWeightsProof<E: ExtensionField> {
    pub proof: sumcheck::structs::IOPProof<E>,
    pub claims: Vec<E>,
    pub partial_evals: Vec<E>,
    pub matrix_evaluation: (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
}

const FILTER_POLY_ID: &str = "ConvFilter";
const BIAS_POLY_ID: &str = "ConvBias";

impl<E> ProveInfo<E> for Convolution<Element>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        let mut filter_shape = self.filter.get_shape();
        filter_shape.remove(1);
        // Capture the unpadded input shape BEFORE updating aux (it is the unpadded output of the
        // previous node, or the model's unpadded input for the first layer).
        let unpadded_input_shape = aux
            .last_unpadded_output_shape
            .first()
            .cloned()
            .unwrap_or_default();

        // For strided convolutions the downstream layers receive the compact strided output.
        // Adjust the spatial dims in last_output_shape accordingly.
        let [sh, sw] = self.stride;
        assert_eq!(sh, sw, "only square strides are supported");
        let [ph, pw] = self.input_padding;
        if sh > 1 {
            // Compute the correct compact strided output shape from the unpadded input.
            // We cannot divide the FFT spatial dim by stride because the FFT dim is
            // next_pow2(h_in + 2*ph) which may differ from h_s_valid * stride.
            // (e.g. 32×32 input, padding=1 → FFT dim=64, but strided valid out = 16×16.)
            let (strided_nw, _) = if !unpadded_input_shape.is_empty() {
                let strided_valid = conv2d_shape_with_padding_and_stride(
                    &unpadded_input_shape,
                    &self.unpadded_shape,
                    ph,
                    pw,
                    sh,
                    sw,
                );
                // strided_valid is [C, H_s, W_s]; POT-pad the spatial dims.
                let h_s = strided_valid[1];
                let w_s = strided_valid[2];
                (h_s.next_power_of_two(), w_s.next_power_of_two())
            } else {
                // Fallback if unpadded shape is unavailable — division may be wrong
                // for padded convolutions but keeps backward compat for unpadded tests.
                let nw = (filter_shape[1] / sh).next_power_of_two();
                (nw, nw)
            };
            filter_shape[1] = strided_nw;
            filter_shape[2] = strided_nw;
        }
        aux.last_output_shape
            .iter_mut()
            .for_each(|shape| *shape = filter_shape.clone());

        // Update last_unpadded_output_shape to the unpadded strided output shape so that
        // subsequent layers see the correct unpadded input shapes.
        let unpadded_output_shape = if !unpadded_input_shape.is_empty() {
            let strided_valid = conv2d_shape_with_padding_and_stride(
                &unpadded_input_shape,
                &self.unpadded_shape,
                ph,
                pw,
                sh,
                sw,
            );
            vec![strided_valid]
        } else {
            // Fallback: no unpadded tracking (shouldn't happen in normal usage).
            aux.last_unpadded_output_shape.clone()
        };
        aux.last_unpadded_output_shape = unpadded_output_shape;

        let mut delegation_fft: Vec<VPAuxInfo<E>> = Vec::new();
        let mut delegation_fft_weights: Vec<VPAuxInfo<E>> = Vec::new();
        let mut delegation_ifft: Vec<VPAuxInfo<E>> = Vec::new();
        for i in (0..(self.filter_size().ilog2() as usize)).rev() {
            delegation_fft.push(VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                i + 1,
                i + 1,
                i + 1,
            ]]));
            delegation_fft_weights.push(VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                i + 1,
                i + 1,
                i + 1,
            ]]));
            delegation_ifft.push(VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                i + 1,
                i + 1,
                i + 1,
            ]]));
        }

        let conv_info = LayerCtx::Convolution(ConvCtx {
            node_id: id,
            ifft_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                ((self.filter_size()).ilog2() as usize) + 1,
                ((self.filter_size()).ilog2() as usize) + 1,
            ]]),
            fft_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                ((self.filter_size()).ilog2() as usize) + 1,
                ((self.filter_size()).ilog2() as usize) + 1,
            ]]),
            fft_weights_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                ((self.filter_size()).ilog2() as usize) + 1,
                ((self.filter_size()).ilog2() as usize) + 1,
            ]]),
            hadamard: VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                ((self.kx() * self.filter_size()).ilog2() as usize) + 1,
                ((self.kx() * self.filter_size()).ilog2() as usize) + 1,
                ((self.kx() * self.filter_size()).ilog2() as usize) + 1,
            ]]),
            delegation_fft,
            delegation_fft_weights,
            delegation_ifft,
            kw: self.kw(),
            kx: self.kx(),
            nw: self.filter.nw(),
            real_nw: self.filter.real_nw(),
            filter_size: self.filter_size(),
            unpadded_filter_shape: self.unpadded_shape.clone(),
            padded_filter_shape: self.filter.real_shape(),
            input_padding: self.input_padding,
            stride: self.stride,
            unpadded_input_shape,
        });

        let filter_poly = self.filter.pad_next_power_of_two().get_data().to_vec();
        let bias_poly = self.bias.pad_next_power_of_two().get_data().to_vec();
        aux.model_polys = {
            let mut model_polys = HashMap::new();
            model_polys.insert(FILTER_POLY_ID.to_string(), filter_poly);
            model_polys.insert(BIAS_POLY_ID.to_string(), bias_poly);
            Some(model_polys)
        };
        Ok((conv_info, aux))
    }
}

impl Convolution<f32> {
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Convolution<Element>>> {
        let model_scaling = ScalingFactor::from_absolute_max(self.max_abs_weight(), None);
        let num_inputs = input_scaling.len();
        ensure!(
            num_inputs == 1,
            "Number of input scaling factor for convolution layer different from 1"
        );
        let input_scaling = &input_scaling[0];
        let bias_scaling = {
            // bias has to be quantized over integers with double bit length
            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
            ScalingFactor::from_scale(
                input_scaling.scale() * model_scaling.scale(),
                Some((min_quantized, max_quantized)),
            )
        };
        let quantized_conv = self.quantize(&model_scaling, &bias_scaling);
        let intermediate_bit_size = quantized_conv.output_bitsize();
        let requant = Requant::from_scaling_factors(
            *input_scaling,
            model_scaling,
            output_scaling,
            intermediate_bit_size,
        );

        Ok(QuantizeOutput::new(quantized_conv, vec![output_scaling]).with_requant(requant))
    }
}

impl QuantizeOp for Convolution<f32> {
    type QuantizedOp = Convolution<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let mut output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        let output_scaling = output_scalings.pop().unwrap();
        self.quantize_from_scalings(input_scaling, output_scaling)
    }
}

impl PadOp for Convolution<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_conv(self, si)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Convolution<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = ConvCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.prove_convolution_step(
            prover,
            last_claims[0],
            step_data.outputs.outputs()[0],
            &step_data.unpadded_output_shapes[0],
            step_data.outputs.try_convdata().unwrap(),
            ctx,
            id,
        )?])
    }
}

impl<E: ExtensionField> OpInfo for ConvCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Convolution::<Element>::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Conv Ctx: ({},{},{},{})",
            self.kw, self.kx, self.nw, self.nw,
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for ConvCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = ConvProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.verify_convolution(
            verifier,
            last_claims[0],
            proof,
            shape_step,
        )?])
    }

    /// Override the default `verify_input_claim` to handle the case where ONNX spatial padding
    /// (`input_padding != [0,0]`) was applied before convolution.
    ///
    /// The prover's final GKR claim is about `fft_input = crop(io_input) → zero_pad_spatial → POT_pad`,
    /// but the verifier receives `io_input` = the nominal POT-padded input (without ONNX spatial padding).
    /// When `input_padding` is non-zero we must apply the same transform to reconstruct `fft_input`.
    fn verify_input_claim<A: AsRef<crate::tensor::Tensor<E>>>(
        &self,
        inputs: &[A],
        claims: &[&crate::Claim<E>],
    ) -> anyhow::Result<()> {
        let [ph, pw] = self.input_padding;
        if ph == 0 && pw == 0 {
            // No ONNX spatial padding: default behavior — evaluate the raw POT-padded input.
            ensure!(
                inputs.len() == claims.len(),
                "number of input tensors and claims must be the same"
            );
            for (i, (input, claim)) in inputs.iter().zip(claims).enumerate() {
                let computed = input.as_ref().get_data().into_mle().evaluate(&claim.point);
                ensure!(
                    computed == claim.eval,
                    "input claim {} is incorrect (no ONNX padding): computed {:?}, given {:?}",
                    i,
                    computed,
                    claim.eval,
                );
            }
            return Ok(());
        }
        // ONNX spatial padding present: we must reconstruct `fft_input` from the nominal input.
        //
        // The nominal `io_input` is POT-padded but lacks the ONNX spatial zero-rows/cols.
        // We crop it back to the unpadded shape, apply ONNX spatial padding, then POT-pad again
        // to match what the prover built as `fft_input`.
        ensure!(
            inputs.len() == claims.len(),
            "number of input tensors and claims must be the same"
        );
        ensure!(
            !self.unpadded_input_shape.is_empty(),
            "ConvCtx.unpadded_input_shape is not set but ONNX padding is non-zero; \
             cannot verify input claim",
        );
        for (i, (input, claim)) in inputs.iter().zip(claims).enumerate() {
            // Reconstruct `fft_input` from the nominal POT-padded input.
            // This mirrors Convolution::op when ph > 0:
            //   1. Crop the POT-padded input to the unpadded [C, h, w] shape.
            //   2. Zero-pad spatially by ph rows/pw cols on each side → [C, h+2ph, w+2pw].
            //   3. POT-pad each dim to next power of two.
            // We do this directly with field-element arithmetic to avoid the `Number` bound.
            let nom_data = input.as_ref().get_data();
            let nom_shape = input.as_ref().get_shape();
            ensure!(
                nom_shape.len() == 3,
                "ConvCtx verify_input_claim: expected 3-D input [C,H_pot,W_pot], got {:?}",
                nom_shape,
            );
            let c = self.unpadded_input_shape[0];
            let h = self.unpadded_input_shape[1];
            let w = self.unpadded_input_shape[2];
            let h_pot = nom_shape[1];
            let w_pot = nom_shape[2];
            // Effective (ONNX-padded) shape
            let h_eff = h + 2 * ph;
            let w_eff = w + 2 * pw;
            // FFT input (POT-padded) shape
            let c_fft = c.next_power_of_two();
            let h_fft = h_eff.next_power_of_two();
            let w_fft = w_eff.next_power_of_two();
            let fft_len = c_fft * h_fft * w_fft;
            let mut fft_data = vec![E::ZERO; fft_len];
            // Fill in the valid (non-padded) elements.
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        // Source: cropped from the POT-padded nominal input.
                        let src_idx = ci * (h_pot * w_pot) + hi * w_pot + wi;
                        // Destination: in the ONNX-spatially-padded + POT-padded tensor.
                        let dst_hi = hi + ph;
                        let dst_wi = wi + pw;
                        let dst_idx = ci * (h_fft * w_fft) + dst_hi * w_fft + dst_wi;
                        fft_data[dst_idx] = nom_data[src_idx];
                    }
                }
            }
            let computed = fft_data.into_mle().evaluate(&claim.point);
            ensure!(
                computed == claim.eval,
                "input claim {} is incorrect (ONNX padding ph={} pw={}): \
                 computed {:?}, given {:?}",
                i,
                ph,
                pw,
                computed,
                claim.eval,
            );
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
impl Convolution<Element> {
    // Prove convolution of a CNN network. This is a convolution between in a 3D matrix X of dimension k_x * n_x * n_x
    // and a 4D filter matrix W of dimension k_w * k_x * n_w * n_w. The output is a 3D matrix Y of dimension k_w * n_x * n_x
    // We want to batch prove the following: Y[i] = iFFT(sum_{j \in [n_x]}(FFT(X[j]) o FFT(W[i][j])).
    #[timed::timed_instrument(name = "Prover::prove_convolution_step")]
    pub fn prove_convolution_step<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        // last random claim made
        last_claim: &Claim<E>,
        // Struct containing all necessary information
        // to generate a convolution proof
        output: &Tensor<E>,
        unpadded_output_shape: &Shape,
        proving_data: &ConvData<E>,
        info: &ConvCtx<E>,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
    {
        // First part is proving the clearing of the garbage has been done correctly.
        // For this, we create the clearing garbage tensor and just prove hadamard with the output.
        // This results in two claims: one for the non-cleared tensor and one for the clearing tensor (only 1s and 0s)
        // The non-cleared tensor claim gets passed to the main regular logic of convolution
        // The clearing tensor one gets stored in the proof and will be checked manually by the verifier (CURRENTLY)
        //
        // For stride > 1:
        //   - `output` is the compact strided tensor [C, H_s_pot, W_s_pot]
        //   - `proving_data.output_as_element` holds the full-res conv output [C*H_full_pot*W_full_pot]
        //   - `last_claim.point` is a claim about the strided output MLE
        //   We expand the claim point to the full-res space (free MLE coordinate map),
        //   then run the clearing hadamard proof on the full-res output and a stride-aware clearing tensor.
        let [sh, sw] = info.stride;
        assert_eq!(sh, sw, "only square strides supported");
        let (hadamard_claim, clearing_tensor, conv_after_bias) = if sh == 1 {
            // No stride: original path.
            let clearing_tensor = new_clearing_tensor(unpadded_output_shape, &output.get_shape());
            let conv_after_bias =
                Tensor::new(output.get_shape(), proving_data.output_as_element.clone());
            debug_assert!({
                info!(
                    "PROVE: conv_after_bias.shape(): {:?}",
                    conv_after_bias.get_shape()
                );
                info!(
                    "PROVE: conv_after_bias.data(): {:?}",
                    &conv_after_bias.get_data()[..30]
                );
                info!("PROVE: unpadded_output_shape: {unpadded_output_shape:?}");
                info!("PROVE: output.shape(): {:?}", output.get_shape());
                let cleared_out = conv_after_bias.flatten().mul(&clearing_tensor);
                let fielded: Tensor<E> = cleared_out.to_fields();
                fielded.get_data().to_vec() == output.get_data()
            });
            (last_claim.clone(), clearing_tensor, conv_after_bias)
        } else {
            // Strided path:
            //
            // The compact strided output has shape [C_pot, H_s_pot, W_s_pot].
            // The last_claim.point has log2(C_pot * H_s_pot * W_s_pot) bits.
            //
            // We expand the claim point to the full-res space [C_pot, n_x, n_x].
            // The full-res output was stored in proving_data.output_as_element before compaction.
            //
            // n_x is the actual FFT spatial dimension: next_pow2(H_in + 2*ph).
            // IMPORTANT: n_x may be > h_s_pot * stride when ONNX padding is non-zero.
            // We derive n_x directly from the stored full-res data length rather than
            // assuming n_x == h_s_pot * stride.
            let strided_shape = output.get_shape(); // [C_pot, H_s_pot, W_s_pot]
            let c_pot = strided_shape[0];
            let h_s_pot = strided_shape[1];
            let w_s_pot = strided_shape[2];
            // Derive the actual FFT spatial dimension from the stored full-res output data.
            let full_res_elements = proving_data.output_as_element.len();
            debug_assert_eq!(
                full_res_elements % c_pot,
                0,
                "full-res output length must be divisible by c_pot"
            );
            let spatial_sq = full_res_elements / c_pot;
            let n_x = (spatial_sq as f64).sqrt() as usize;
            debug_assert_eq!(
                n_x * n_x,
                spatial_sq,
                "full-res spatial dims must be square"
            );
            debug_assert!(n_x.is_power_of_two(), "n_x must be a power of two");
            let full_padded_shape = Shape::new(vec![c_pot, n_x, n_x]);

            // Expand the claim point from strided → full-res using the actual n_x.
            let r_full = expand_strided_point(&last_claim.point, c_pot, h_s_pot, w_s_pot, n_x, sh);
            let expanded_claim = Claim::new(r_full, last_claim.eval);

            // The strided valid shape tells us how many valid strided positions exist.
            // unpadded_output_shape is the strided valid shape [C, H_s, W_s].
            let clearing_tensor =
                new_clearing_tensor_strided(unpadded_output_shape, &full_padded_shape, sh);
            let conv_after_bias = Tensor::new(
                full_padded_shape.clone(),
                proving_data.output_as_element.clone(),
            );
            debug_assert_eq!(
                conv_after_bias.get_data().len(),
                full_padded_shape.product(),
                "Stride prove: full-res output size mismatch"
            );
            (expanded_claim, clearing_tensor, conv_after_bias)
        };
        let clearing_proof = hadamard::prove(
            prover.transcript,
            &hadamard_claim,
            &conv_after_bias,
            &clearing_tensor,
        );
        // since v1 is the non cleared tensor, this is what the rest of the convolution proving expects
        let last_claim = Claim::new(
            clearing_proof.random_point().to_vec(),
            clearing_proof.v1_eval(),
        );

        let filter = self;
        assert_eq!(
            filter.filter_size() * filter.kw() * 2,
            proving_data.output.len() * proving_data.output[0].len(),
            "Inconsistent output size"
        );
        assert_eq!(
            (filter.filter_size() * filter.kw()).ilog2() as usize,
            last_claim.point.len(),
            "Inconsistent random point size. Expected : {}, got: {}",
            ((filter.filter_size() * filter.kw()).ilog2()),
            last_claim.point.len()
        );
        let mut r = vec![E::ZERO; last_claim.point.len() + 1];
        let mut bias_point = vec![E::ZERO; filter.kw().ilog2() as usize];
        for (i, item) in r
            .iter_mut()
            .enumerate()
            .take(filter.filter_size().ilog2() as usize)
        {
            *item = E::ONE - last_claim.point[i];
        }
        for i in 0..(filter.kw().ilog2() as usize) {
            r[i + (filter.filter_size().ilog2() as usize) + 1] =
                last_claim.point[i + (filter.filter_size().ilog2() as usize)];
            bias_point[i] = last_claim.point[i + (filter.filter_size().ilog2() as usize)];
        }
        let mut bias_eval = E::ZERO;
        if !bias_point.is_empty() {
            bias_eval = filter
                .bias
                .evals_flat::<E>()
                .into_mle()
                .evaluate(&bias_point);
        } else if filter.bias.data.len() == 1 {
            bias_eval = filter.bias.evals_flat::<E>()[0];
        }

        debug_assert!({
            let y = proving_data
                .output
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_mle()
                .evaluate(&r);
            debug_assert_eq!(last_claim.eval - bias_eval, y, "Error in Conv 1");
            last_claim.eval - bias_eval == y
        });

        let mut temp_t = prover.transcript.clone();
        let BatchFFTProof {
            proof: ifft_proof,
            claims: ifft_claim,
            matrix_eval: ifft_del_proof,
        } = prover.prove_batch_ifft(r.clone(), &proving_data.prod);

        assert_eq!(
            filter.filter_size().ilog2() as usize + 1,
            ifft_proof.point.len(),
            "Error in ifft sumceck"
        );
        debug_assert!({
            IOPVerifierState::<E>::verify(
                last_claim.eval - bias_eval,
                &ifft_proof.clone(),
                &info.ifft_aux.clone(),
                &mut temp_t,
            );
            info!("iFFT Sumcheck Correct");
            true
        });

        // After this point, the verifier holds an evaluation claim of proving_data.prod at P1.randomness[0][i]
        // Let r' = P1.randomness[0][i] and y is the evaluation claim of prod = proving_data.prod
        // What we want to do now is to prove that prod has been correctly computed from X_fft and w (= proving_data.w)
        // In other words we want to show that prod[i] = sum_{j \in [k_x]} x[j] o w[i][j] for each i in [k_w]
        // For this let r1 be the last log(k_w) elements of r and r2 the first log(n_x^2) elements
        // Compute the arrays beta1,beta2 such that beta1[i] = beta(i,r1) and beta2[i] = beta(i,r2)

        let mut r_ifft: Vec<E> = ifft_proof.point.clone();
        for item in r.iter().skip(proving_data.output[0].len().ilog2() as usize) {
            r_ifft.push(*item);
        }

        debug_assert!({
            let eval1 = proving_data
                .prod
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_mle()
                .evaluate(&r_ifft);
            let eval2 = ifft_claim[0];
            debug_assert_eq!(
                proving_data
                    .prod
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into_mle()
                    .evaluate(&r_ifft),
                ifft_claim[0],
                "Error in Conv 1"
            );
            eval1 == eval2
        });

        let r1 = &r_ifft[(proving_data.output[0].len().ilog2() as usize)..];
        let r2 = &r_ifft[..(proving_data.output[0].len().ilog2() as usize)];
        let beta1 = compute_betas_eval(r1);
        let beta2 = compute_betas_eval(r2);
        // Given beta1,beta2 observe that :
        // \sum_{i \in [k_w]} beta1[i]prod[i] = \sum_{i \in [k_w]}sum_{j \in [k_x]} x[j] o w[i][j] =
        // = sum_{j \in [k_x]}x[j]o(\sum_{i \in [k_w]}(beta[i]*w[i][j])). We let w_reduced[j] = \sum_{i \in [k_w]}(beta[i]*w[i][j])
        // We have  \sum_{i \in [k_w]} beta1[i]prod[i] = sum_{j \in [k_x]} x[j]o w_{reduced[j]}.
        // So here we compute w_reduced

        let beta_acc = vec![beta2.clone(); filter.kx()].concat();

        // After computing w_reduced, observe that y = \sum_{k \in [n_x^2]} sum_{j \in [k_x]} beta2[k]*x[j][k]*w_reduced[j][k]
        // This is a cubic sumcheck where v1 = [x[0][0],...,x[k_x][n_x^2]], v2 = [w_reduced[0][0],...,w_reduced[k_x][n_x^2]]
        // and v3 = [beta2,..(k_x times)..,beta2]. So, first initialize v3 and then invoke the cubic sumceck.
        let mut aggregated_filter =
            vec![vec![E::ZERO; self.filter.real_nw() * self.filter.real_nw()]; self.filter.kx()];
        let filter_size = self.filter.real_nw() * self.filter.real_nw();
        // Compute aggregated_filter using iterators
        // TO DO: PARALLELIZE
        (0..self.filter.kx()).for_each(|i| {
            (0..self.filter.kw()).for_each(|j| {
                aggregated_filter[i]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(k, v)| {
                        let index = j * self.filter.kx() * filter_size + i * filter_size + k;
                        let v_field: E = self.filter.data[index].to_field();
                        *v += beta1[j] * v_field;
                    });
            });

            aggregated_filter[i] = index_wf(
                &aggregated_filter[i],
                self.filter.real_nw(),
                self.filter.nw(),
                2 * self.filter.nw() * self.filter.nw(),
            )
            .collect::<Vec<E>>();

            fft(&mut aggregated_filter[i], false);
        });

        // We need to fix the high variables in place for the filter at r1.
        let f1 = aggregated_filter
            .into_iter()
            .flatten()
            .collect::<Vec<E>>()
            .into_mle();

        let f2 = proving_data
            .input_fft
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .into_mle();
        let f3 = beta_acc.into_mle();

        let mut vp = VirtualPolynomial::<E>::new(f1.num_vars);
        vp.add_mle_list(
            vec![f1.clone().into(), f2.clone().into(), f3.clone().into()],
            E::ONE,
        );
        #[allow(deprecated)]
        let (hadamard_proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);
        let hadamard_claims = state.get_mle_final_evaluations();

        let point = [hadamard_proof.point.as_slice(), r1].concat();
        // let eval = hadamard_claims[0];

        // Finally prove the correct computation of the x_fft and get an evaluation claim of the input
        let BatchFFTProof {
            proof: fft_proof,
            claims: fft_claim,
            matrix_eval: fft_del_proof,
        } = prover.prove_batch_fft(
            hadamard_proof.point.clone(),
            &mut proving_data.input.clone(),
        );

        let BatchFFTWeightsProof {
            proof: fft_proof_weights,
            claims: fft_weight_claims,
            partial_evals,
            matrix_evaluation: fft_weights_del_proof,
        } = self.prove_batch_fft_weights(prover, point.clone());

        let weights_rand: Vec<E> = prover
            .transcript
            .read_challenges((self.filter.real_nw() * self.filter.real_nw()).ilog2() as usize);
        debug_assert!({
            let mut weights_point = fft_proof_weights.point.clone();
            let mut v_weights = weights_point.pop().unwrap();
            v_weights = (E::ONE - v_weights).inverse();

            let mut r = [
                weights_rand.clone(),
                point[(2 * self.filter.nw() * self.filter.nw()).ilog2() as usize..].to_vec(),
            ]
            .concat();
            // println!("({},{}), {}",proving_data.input.len(),proving_data.input[0].len(),p.len());
            let mut y = self.filter.get_conv_weights::<E>().into_mle().evaluate(&r);
            assert_eq!(
                y,
                partial_evals.clone().into_mle().evaluate(&weights_rand),
                "Error in fft_weights eval"
            );
            let mut indexes = vec![0_usize; self.filter.real_nw() * self.filter.real_nw()];
            for i in 0..self.filter.real_nw() {
                for j in 0..self.filter.real_nw() {
                    indexes[i * self.filter.real_nw() + j] = i * self.filter.nw() + j;
                }
            }
            r = weights_point[..(self.filter.nw() * self.filter.nw()).ilog2() as usize].to_vec();
            let mut betas = vec![E::ZERO; self.filter.real_nw() * self.filter.real_nw()];
            for i in 0..betas.len() {
                betas[i] = identity_eval(&r, &to_bits(indexes[i], r.len()));
            }
            y = E::ZERO;
            for i in 0..betas.len() {
                y += betas[i] * partial_evals[i];
            }
            assert_eq!(
                y,
                fft_weight_claims[0] * v_weights,
                "Error in padded weights eval"
            );
            y == fft_weight_claims[0] * v_weights
        });

        let bias_claim = Claim::new(bias_point, bias_eval);
        let filter_claim = Claim::new(
            [
                weights_rand.clone(),
                point[(2 * self.filter.nw() * self.filter.nw()).ilog2() as usize..].to_vec(),
            ]
            .concat(),
            partial_evals.clone().into_mle().evaluate(&weights_rand),
        );

        // Add common polynomial commitment claims to the commitment prover
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(FILTER_POLY_ID.to_string(), filter_claim);
            claims.insert(BIAS_POLY_ID.to_string(), bias_claim);
            claims
        };
        prover.add_common_claims(id, common_claims)?;

        prover.push_proof(
            id,
            LayerProof::Convolution(Box::new(ConvProof {
                fft_proof: fft_proof.clone(),
                fft_claims: fft_claim.clone(),
                fft_proof_weights,
                ifft_proof,
                fft_delegation_proof: fft_del_proof.0,
                fft_delegation_proof_weights: fft_weights_del_proof.0,
                ifft_delegation_proof: ifft_del_proof.0,
                hadamard_proof: hadamard_proof.clone(),
                ifft_claims: ifft_claim,
                fft_weight_claims,
                fft_delegation_claims: fft_del_proof.1,
                fft_delegation_weights_claims: fft_weights_del_proof.1,
                ifft_delegation_claims: ifft_del_proof.1,
                hadamard_clams: hadamard_claims,
                bias_claim: bias_eval,
                partial_evals,
                clearing_proof,
            })),
        );
        let mut input_point = fft_proof.point.clone();
        let mut v = input_point.pop().unwrap();
        v = (E::ONE - v).inverse();
        debug_assert!({
            let mut p = [
                input_point.clone(),
                hadamard_proof.point[((filter.filter_size() * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat();
            // println!("({},{}), {}",proving_data.input.len(),proving_data.input[0].len(),p.len());
            let y = proving_data
                .input
                .clone()
                .into_iter()
                .flat_map(|v| v.into_iter())
                .collect::<Vec<E>>()
                .into_mle()
                .evaluate(&p);
            assert_eq!(y, fft_claim[0] * v, "Error in input eval CONV PROVER");
            for element in p.iter_mut().take((filter.filter_size().ilog2()) as usize) {
                *element = E::ONE - *element;
            }
            assert_eq!(
                proving_data.real_input.clone().into_mle().evaluate(&p),
                fft_claim[0] * v,
                "Error in real input eval CONV PROVER"
            );
            proving_data.real_input.clone().into_mle().evaluate(&p) == fft_claim[0] * v
        });
        for ip in &mut input_point {
            *ip = E::ONE - *ip;
        }
        let final_claim = Claim {
            point: [
                input_point.clone(),
                hadamard_proof.point[((filter.filter_size() * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            eval: fft_claim[0] * v,
        };

        Ok(final_claim)
    }
}

impl<E> ConvCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    pub fn output_shape(&self, input_shape: &Shape, padding_mode: PaddingMode) -> Shape {
        let [ph, pw] = self.input_padding;
        let [sh, sw] = self.stride;
        match padding_mode {
            PaddingMode::NoPadding => conv2d_shape_with_padding_and_stride(
                input_shape,
                &self.unpadded_filter_shape,
                ph,
                pw,
                sh,
                sw,
            ),
            PaddingMode::Padding => padded_conv2d_shape_with_padding_and_stride(
                input_shape,
                &self.padded_filter_shape,
                ph,
                pw,
                sh,
                sw,
            ),
        }
    }
    pub(crate) fn verify_fft_delegation<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        mut claim: E,
        proof: &ConvProof<E>,
        delegation_proof: &[IOPProof<E>],
        delegation_claims: &[Vec<E>],
        mut prev_r: Vec<E>,
    ) {
        let iter = delegation_proof.len();
        // Verify delegation protocol of W iFFT matrix
        let exponents = pow_two_omegas(iter + 1, false);
        for i in 0..iter {
            IOPVerifierState::<E>::verify(
                claim,
                &delegation_proof[i],
                &self.delegation_fft[i],
                verifier.transcript,
            );

            assert_eq!(
                identity_eval(
                    delegation_proof[i].point.clone().as_slice(),
                    prev_r.clone().as_slice()
                ),
                delegation_claims[i][0],
                "Error in identity evaluation fft delegation iter : {i}"
            );

            assert_eq!(
                phi_eval(
                    delegation_proof[i].point.clone(),
                    proof.hadamard_proof.point[i],
                    prev_r[prev_r.len() - 1],
                    exponents.clone(),
                    i == 0
                ),
                delegation_claims[i][1],
                "Error in phi computation fft delegation iter : {i}"
            );

            claim = delegation_claims[i][2];
            prev_r = delegation_proof[i].point.clone();
        }
        assert_eq!(
            claim,
            (E::ONE - E::from_canonical_u64(2) * proof.hadamard_proof.point[iter]) * prev_r[0]
                + E::ONE
                - prev_r[0],
            "Error in final FFT delegation step"
        );
    }

    pub(crate) fn verify_convolution<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &ConvProof<E>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Claim<E>> {
        ensure!(
            shape_step.unpadded_input_shape.len() == 1,
            "More than 1 unpadded input shape found for convolution layer",
        );
        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "More than 1 padded input shape found for convolution layer",
        );
        // The first thing to do is to recreate the hadamard clearing tensor
        // Since this is only coming from public information, the verifier
        // creates the vector and evaluates it.
        // NOTE: for succinctness of verification, we could also have
        // the prover commits to the tensor product and we could skip this step.
        // OR find a closed formula
        //
        // To recreate it, we need the unpadded output shape and the real output shape.
        // Account for ONNX spatial padding: the effective input is spatially larger.
        let [ph, pw] = self.input_padding;
        let [sh, sw] = self.stride;
        assert_eq!(sh, sw, "only square strides supported");

        // For stride > 1, the last_claim refers to the compact strided output MLE.
        // Expand the claim point to the full-res space before running the hadamard proof.
        let (hadamard_claim, clearing_tensor, hctx) = if sh == 1 {
            let unpadded_output_shape = conv2d_shape_with_padding(
                &shape_step.unpadded_input_shape[0],
                &self.unpadded_filter_shape,
                ph,
                pw,
            );
            let real_output_shape = padded_conv2d_shape_with_padding(
                &shape_step.padded_input_shape[0],
                &self.padded_filter_shape,
                ph,
                pw,
            );
            let clearing_tensor = new_clearing_tensor(&unpadded_output_shape, &real_output_shape);
            let hctx = hadamard::HadamardCtx::from_len(real_output_shape.product());
            (last_claim.clone(), clearing_tensor, hctx)
        } else {
            // Strided case: the incoming claim is on the compact strided output.
            // Compute the strided valid shape [C, H_s, W_s] and the full-res padded shape.
            let strided_valid_shape = conv2d_shape_with_padding_and_stride(
                &shape_step.unpadded_input_shape[0],
                &self.unpadded_filter_shape,
                ph,
                pw,
                sh,
                sw,
            );
            // The compact POT-padded strided shape is what last_claim.point refers to.
            let strided_pot_shape = padded_conv2d_shape_with_padding_and_stride(
                &shape_step.padded_input_shape[0],
                &self.padded_filter_shape,
                ph,
                pw,
                sh,
                sw,
            );
            let c_pot = strided_pot_shape[0];
            let h_s_pot = strided_pot_shape[1];
            let w_s_pot = strided_pot_shape[2];
            // Derive the actual FFT spatial dimension n_x = next_pow2(H_in + 2*ph).
            // IMPORTANT: n_x may be > h_s_pot * stride when ONNX padding is non-zero.
            // We must NOT assume n_x == h_s_pot * stride.
            let h_unpadded = shape_step.unpadded_input_shape[0][1];
            let n_x = (h_unpadded + 2 * ph).next_power_of_two();
            let full_padded_shape = Shape::new(vec![c_pot, n_x, n_x]);

            // Expand the strided claim point to full-res using the actual n_x.
            let r_full = expand_strided_point(&last_claim.point, c_pot, h_s_pot, w_s_pot, n_x, sh);
            let expanded_claim = Claim::new(r_full, last_claim.eval);

            let clearing_tensor =
                new_clearing_tensor_strided(&strided_valid_shape, &full_padded_shape, sh);
            let hctx = hadamard::HadamardCtx::from_len(full_padded_shape.product());
            (expanded_claim, clearing_tensor, hctx)
        };

        let expected_v2_eval = clearing_tensor
            .to_mle_flat()
            .evaluate(proof.clearing_proof.random_point());
        // also set the claim to be the non-cleared output of conv. The rest of the logic is about proving the bias + fft claims.
        let last_claim = hadamard::verify(
            &hctx,
            verifier.transcript,
            &proof.clearing_proof,
            &hadamard_claim,
            expected_v2_eval,
        )
        .context("failure for hadamard proof")?;

        let conv_claim = last_claim.eval - proof.bias_claim;

        IOPVerifierState::<E>::verify(
            conv_claim,
            &proof.ifft_proof,
            &self.ifft_aux,
            verifier.transcript,
        );
        assert_eq!(
            self.delegation_ifft.len(),
            proof.ifft_delegation_proof.len(),
            "Inconsistency in iFFT delegation proofs/aux size"
        );

        let iter = proof.ifft_delegation_proof.len();
        let mut claim = proof.ifft_claims[1];
        let exponents = pow_two_omegas(iter + 1, true);
        let mut prev_r = proof.ifft_proof.point.clone();
        for i in 0..iter {
            IOPVerifierState::<E>::verify(
                claim,
                &proof.ifft_delegation_proof[i],
                &self.delegation_ifft[i],
                verifier.transcript,
            );
            assert_eq!(
                identity_eval(
                    proof.ifft_delegation_proof[i].point.clone().as_slice(),
                    prev_r.clone().as_slice()
                ),
                proof.ifft_delegation_claims[i][0],
                "Error in identity evaluation ifft delegation iter : {i}"
            );
            assert_eq!(
                phi_eval(
                    proof.ifft_delegation_proof[i].point.clone(),
                    E::ONE - last_claim.point[i],
                    prev_r[prev_r.len() - 1],
                    exponents.clone(),
                    false
                ),
                proof.ifft_delegation_claims[i][1],
                "Error in phi computation ifft delegation iter : {i}"
            );

            prev_r = proof.ifft_delegation_proof[i].point.clone();
            claim = proof.ifft_delegation_claims[i][2];
        }
        let scale = E::from_canonical_u64(1 << (iter + 1)).inverse();

        assert_eq!(
            claim,
            scale * (E::ONE) * prev_r[0] + scale * (E::ONE - prev_r[0]),
            "Error in final iFFT delegation step"
        );

        IOPVerifierState::<E>::verify(
            proof.ifft_claims[0],
            &proof.hadamard_proof,
            &self.hadamard,
            verifier.transcript,
        );
        assert_eq!(
            proof.hadamard_clams[2],
            identity_eval(&proof.ifft_proof.point, &proof.hadamard_proof.point),
            "Error in Beta evaluation"
        );

        // >>>>>> TODO : 1) Dont forget beta evaluation 2) verification of the last step of delegation <<<<<<<
        // Verify fft sumcheck
        IOPVerifierState::<E>::verify(
            proof.hadamard_clams[1],
            &proof.fft_proof,
            &self.fft_aux,
            verifier.transcript,
        );
        claim = proof.fft_claims[1];

        assert_eq!(
            self.delegation_fft.len(),
            proof.fft_delegation_proof.len(),
            "Inconsistency in FFT delegation proofs/aux size"
        );

        self.verify_fft_delegation(
            verifier,
            claim,
            proof,
            &proof.fft_delegation_proof,
            &proof.fft_delegation_claims,
            proof.fft_proof.point.clone(),
        );

        IOPVerifierState::<E>::verify(
            proof.hadamard_clams[0],
            &proof.fft_proof_weights,
            &self.fft_weights_aux,
            verifier.transcript,
        );
        claim = proof.fft_weight_claims[1];
        self.verify_fft_delegation(
            verifier,
            claim,
            proof,
            &proof.fft_delegation_proof_weights,
            &proof.fft_delegation_weights_claims,
            proof.fft_proof_weights.point.clone(),
        );

        // Validate the correctness of the padded weights claim
        // using the partial_evals provided by the prover
        let mut weights_point = proof.fft_proof_weights.point.clone();
        let mut v = weights_point.pop().unwrap();
        v = (E::ONE - v).inverse();

        let y_weights = (0..self.real_nw)
            .flat_map(|i| (0..self.real_nw).map(move |j| (i, j)))
            .fold(E::ZERO, |acc, (i, j)| {
                acc + proof.partial_evals[i * self.real_nw + j]
                    * identity_eval(
                        &to_bits(i * self.nw + j, (self.nw.ilog2() as usize) * 2),
                        &weights_point,
                    )
            });

        assert_eq!(
            proof.fft_weight_claims[0] * v,
            y_weights,
            "Error in padded_fft evaluation claim"
        );

        let weights_rand: Vec<E> = verifier
            .transcript
            .read_challenges((self.real_nw * self.real_nw).ilog2() as usize);

        let point = [
            proof.hadamard_proof.point.as_slice(),
            &last_claim.point[((self.filter_size).ilog2() as usize)..],
        ]
        .concat();

        let bias_claim = Claim::new(
            last_claim.point[(proof.ifft_delegation_proof.len())..].to_vec(),
            proof.bias_claim,
        );

        let filter_claim = Claim::new(
            [
                weights_rand.clone(),
                point[(2 * self.nw * self.nw).ilog2() as usize..].to_vec(),
            ]
            .concat(),
            proof
                .partial_evals
                .clone()
                .into_mle()
                .evaluate(&weights_rand),
        );
        // Add the common commitment claims to be verified
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(FILTER_POLY_ID.to_string(), filter_claim);
            claims.insert(BIAS_POLY_ID.to_string(), bias_claim);
            claims
        };
        verifier.add_common_claims(self.node_id, common_claims)?;

        let mut input_point = proof.fft_proof.point.clone();
        v = input_point.pop().unwrap();
        v = (E::ONE - v).inverse();
        for point in &mut input_point {
            *point = E::ONE - *point;
        }
        // the output claim for this step that is going to be verified at next step
        Ok(Claim {
            // the new randomness to fix at next layer is the randomness from the sumcheck !
            point: [
                input_point.clone(),
                proof.hadamard_proof.point[((self.filter_size * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
            // random point. 1 because vector is secondary.
            eval: proof.fft_claims[0] * v,
        })
    }
}

impl<T: Number> OpInfo for SchoolBookConv<T> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        self.0.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        self.0.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        todo!()
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl<T: Number> Evaluate<T> for SchoolBookConv<T> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<T>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<T, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating schoolbook convolution layer"
        );
        let input = inputs[0];
        let [ph, pw] = self.0.input_padding;
        let padded = input.zero_pad_spatial(ph, pw);
        Ok(LayerOut::from_vec(vec![padded.conv2d(
            &self.0.filter,
            &self.0.bias,
            1,
        )]))
    }
}

impl PadOp for SchoolBookConv<Element> {}

impl QuantizeOp for SchoolBookConv<f32> {
    type QuantizedOp = SchoolBookConv<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        Ok(QuantizeOutput {
            quantized_op: SchoolBookConv(self.0.quantize(
                // we don't care about accurate quantization for schoolbook conv
                &input_scaling[0],
                &input_scaling[0],
            )),
            output_scalings: input_scaling.to_vec(),
            requant_layer: None,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchoolBookConvCtx;

impl<E: ExtensionField> ProveInfo<E> for SchoolBookConv<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    fn step_info(&self, _id: NodeId, aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        let conv_info = LayerCtx::SchoolBookConvolution(SchoolBookConvCtx);
        Ok((conv_info, aux))
    }
}

pub fn pow_two_omegas<E: ExtensionField>(n: usize, is_fft: bool) -> Vec<E> {
    let mut pows = vec![E::ZERO; n - 1];
    let mut rou: E = get_root_of_unity(n);
    if is_fft {
        rou = rou.inverse();
    }
    pows[0] = rou;
    for i in 1..(n - 1) {
        pows[i] = pows[i - 1] * pows[i - 1];
    }
    pows
}

pub fn phi_eval<E: ExtensionField>(
    r: Vec<E>,
    rand1: E,
    rand2: E,
    exponents: Vec<E>,
    first_iter: bool,
) -> E {
    let mut eval = E::ONE;
    for i in 0..r.len() {
        eval *= E::ONE - r[i] + r[i] * exponents[exponents.len() - r.len() + i];
    }

    if first_iter {
        eval = (E::ONE - rand2) * (E::ONE - rand1 + rand1 * eval);
    } else {
        eval = E::ONE - rand1 + (E::ONE - E::from_canonical_u64(2) * rand2) * rand1 * eval;
    }

    eval
}

fn clear_garbage<T: Number>(output_tensor: &Tensor<T>, unpadded_output_shape: &Shape) -> Tensor<T> {
    let unpadded_output_shape = if unpadded_output_shape.len() == 4 {
        unpadded_output_shape.slice(1..)
    } else {
        unpadded_output_shape.clone()
    };
    let padded_shape = output_tensor.get_shape();
    let mut data = output_tensor.get_data().to_vec();
    for i in 0..padded_shape[0] {
        for j in 0..padded_shape[1] {
            for k in 0..padded_shape[2] {
                let index = i * padded_shape[1] * padded_shape[2] + j * padded_shape[2] + k;
                if !(i < unpadded_output_shape[0]
                    && j < unpadded_output_shape[1]
                    && k < unpadded_output_shape[2])
                {
                    data[index] = T::default();
                }
            }
        }
    }
    Tensor::new(padded_shape, data)
}

pub fn new_clearing_tensor(og_shape: &Shape, padded_shape: &Shape) -> Tensor<Element> {
    let og_shape = if og_shape.len() == 4 {
        og_shape.slice(1..)
    } else {
        og_shape.clone()
    };
    assert_eq!(padded_shape.len(), og_shape.len());
    assert_eq!(padded_shape.len(), 3);
    let n = padded_shape.product();
    let mut data: Vec<Element> = vec![0; n];
    for i in 0..padded_shape[0] {
        for j in 0..padded_shape[1] {
            for k in 0..padded_shape[2] {
                let index = i * padded_shape[1] * padded_shape[2] + j * padded_shape[2] + k;
                if i < og_shape[0] && j < og_shape[1] && k < og_shape[2] {
                    data[index] = 1;
                }
            }
        }
    }
    Tensor::new(Shape::new(vec![padded_shape.product()]), data)
}

/// Properly pad a filter
/// We use this function so that filter is amenable to FFT based conv2d
/// Usually vec and n are powers of 2
/// Output: [[F[0][0],…,F[0][n_w],0,…,0],[F[1][0],…,F[1][n_w],0,…,0],…]
pub fn index_wf<E: ExtensionField>(
    w: &[E],
    n_real: usize,
    n: usize,
    output_len: usize,
) -> impl ParallelIterator<Item = E> + use<'_, E> {
    (0..output_len).into_par_iter().map(move |idx| {
        let i = idx / n;
        let j = idx % n;
        if i < n_real && j < n_real {
            w[i * n_real + j]
        } else {
            E::ZERO
        }
    })
}

pub fn conv2d_shape_mode(
    input_shape: &Shape,
    filter_shape: &Shape,
    padding_mode: PaddingMode,
) -> Shape {
    match padding_mode {
        PaddingMode::NoPadding => conv2d_shape(input_shape, filter_shape),
        PaddingMode::Padding => padded_conv2d_shape(input_shape, filter_shape),
    }
}

/// Assumes stride=1, padding=0, and dilation=1
/// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
pub fn conv2d_shape(input_shape: &Shape, filter_shape: &Shape) -> Shape {
    let stride = 1usize;
    let padding = 0usize;
    let dilation = 1usize;

    let h_in = if input_shape.len() == 3 {
        input_shape[1]
    } else {
        input_shape[2]
    };
    let kernel = filter_shape[2];
    let h_out = (h_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    Shape::new(vec![filter_shape[0], h_out, h_out])
}

/// Similar to conv2d_shape but pads the output shape such that it matches what the padded inference and proving expects
pub fn padded_conv2d_shape(input_shape: &Shape, filter_shape: &Shape) -> Shape {
    conv2d_shape(input_shape, filter_shape)
        .into_vec()
        .into_iter()
        .map(|x| x.next_power_of_two())
        .collect::<Shape>()
}

/// Like `conv2d_shape` but accounts for ONNX-style spatial zero-padding applied to the input.
///
/// The effective input spatial size is `(H + 2*pad_h) × (W + 2*pad_w)`, so the output is:
/// `out = (spatial + 2*pad - kernel) / stride + 1` with stride=1.
pub fn conv2d_shape_with_padding(
    input_shape: &Shape,
    filter_shape: &Shape,
    pad_h: usize,
    pad_w: usize,
) -> Shape {
    if pad_h == 0 && pad_w == 0 {
        return conv2d_shape(input_shape, filter_shape);
    }
    let h_in = if input_shape.len() == 3 {
        input_shape[1]
    } else {
        input_shape[2]
    };
    let w_in = if input_shape.len() == 3 {
        input_shape[2]
    } else {
        input_shape[3]
    };
    let kh = filter_shape[2];
    let kw = filter_shape[3];
    let h_out = h_in + 2 * pad_h - kh + 1;
    let w_out = w_in + 2 * pad_w - kw + 1;
    Shape::new(vec![filter_shape[0], h_out, w_out])
}

/// Power-of-two padded variant of `conv2d_shape_with_padding`.
pub fn padded_conv2d_shape_with_padding(
    input_shape: &Shape,
    filter_shape: &Shape,
    pad_h: usize,
    pad_w: usize,
) -> Shape {
    conv2d_shape_with_padding(input_shape, filter_shape, pad_h, pad_w)
        .into_vec()
        .into_iter()
        .map(|x| x.next_power_of_two())
        .collect::<Shape>()
}

/// Like `conv2d_shape_with_padding` but also applies spatial stride.
///
/// `out_h = floor((h_in + 2*pad_h - kh) / stride_h) + 1`
pub fn conv2d_shape_with_padding_and_stride(
    input_shape: &Shape,
    filter_shape: &Shape,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Shape {
    if stride_h == 1 && stride_w == 1 {
        return conv2d_shape_with_padding(input_shape, filter_shape, pad_h, pad_w);
    }
    let h_in = if input_shape.len() == 3 {
        input_shape[1]
    } else {
        input_shape[2]
    };
    let w_in = if input_shape.len() == 3 {
        input_shape[2]
    } else {
        input_shape[3]
    };
    let kh = filter_shape[2];
    let kw = if filter_shape.len() > 3 {
        filter_shape[3]
    } else {
        filter_shape[2]
    };
    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    Shape::new(vec![filter_shape[0], h_out, w_out])
}

/// Power-of-two padded variant of `conv2d_shape_with_padding_and_stride`.
pub fn padded_conv2d_shape_with_padding_and_stride(
    input_shape: &Shape,
    filter_shape: &Shape,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Shape {
    conv2d_shape_with_padding_and_stride(
        input_shape,
        filter_shape,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
    .into_vec()
    .into_iter()
    .map(|x| x.next_power_of_two())
    .collect::<Shape>()
}

/// Zero out all positions in `output_tensor` that are either:
/// - outside the valid (unstrided) output region `full_unpadded_shape`, OR
/// - not at a stride-multiple position (i.e., where `h % stride != 0` or `w % stride != 0`).
///
/// The returned tensor has the **same shape** as `output_tensor` (full-res, POT-padded).
/// This is the tensor passed to `prove_convolution_step` as `conv_after_bias` for the
/// clearing hadamard proof.
fn clear_garbage_strided<T: Number>(
    output_tensor: &Tensor<T>,
    unpadded_output_shape: &Shape,
    stride: usize,
) -> Tensor<T> {
    let unpadded = if unpadded_output_shape.len() == 4 {
        unpadded_output_shape.slice(1..)
    } else {
        unpadded_output_shape.clone()
    };
    let padded_shape = output_tensor.get_shape();
    let mut data = output_tensor.get_data().to_vec();
    for i in 0..padded_shape[0] {
        for j in 0..padded_shape[1] {
            for k in 0..padded_shape[2] {
                let index = i * padded_shape[1] * padded_shape[2] + j * padded_shape[2] + k;
                let in_valid = i < unpadded[0] && j < unpadded[1] && k < unpadded[2];
                let is_stride_pos = j % stride == 0 && k % stride == 0;
                if !(in_valid && is_stride_pos) {
                    data[index] = T::default();
                }
            }
        }
    }
    Tensor::new(padded_shape, data)
}

/// Build a stride-aware clearing tensor for use in the hadamard proof.
///
/// The tensor has shape `[full_C_pot * full_H_pot * full_W_pot]` (flattened) and is 1 only at
/// `(c, j*stride, k*stride)` for valid strided output indices `(j, k)`.
///
/// This allows the verifier (who knows the public shapes and stride) to independently
/// reconstruct the clearing tensor and verify the hadamard proof.
pub fn new_clearing_tensor_strided(
    strided_valid_shape: &Shape,
    full_padded_shape: &Shape,
    stride: usize,
) -> Tensor<Element> {
    // strided_valid_shape: [C, H_s, W_s]  (no POT-padding, no batch dim)
    let sv = if strided_valid_shape.len() == 4 {
        strided_valid_shape.slice(1..)
    } else {
        strided_valid_shape.clone()
    };
    assert_eq!(full_padded_shape.len(), 3);
    assert_eq!(sv.len(), 3);
    let n = full_padded_shape.product();
    let mut data: Vec<Element> = vec![0; n];
    // sv[0] = C_valid, sv[1] = H_strided_valid, sv[2] = W_strided_valid
    for c in 0..full_padded_shape[0] {
        for j in 0..full_padded_shape[1] {
            for k in 0..full_padded_shape[2] {
                let idx =
                    c * full_padded_shape[1] * full_padded_shape[2] + j * full_padded_shape[2] + k;
                // A position is "live" iff it corresponds to a valid strided output:
                // j = s * h_s, k = s * w_s, where h_s < sv[1] and w_s < sv[2]
                if c < sv[0] && j % stride == 0 && k % stride == 0 {
                    let h_s = j / stride;
                    let w_s = k / stride;
                    if h_s < sv[1] && w_s < sv[2] {
                        data[idx] = 1;
                    }
                }
            }
        }
    }
    Tensor::new(Shape::new(vec![n]), data)
}

/// Compact the full-res stride-cleared tensor into a `[C, H_s, W_s]` tensor (POT-padded).
///
/// Extracts the `(c, j*stride, k*stride)` positions from `cleared_tensor` and packs them
/// into a contiguous array with shape `[C, H_s_pot, W_s_pot]`.
///
/// `full_unpadded_shape` is the full-res valid shape `[C, H, W]`.
fn compact_strided<T: Number>(
    cleared_tensor: &Tensor<T>,
    full_unpadded_shape: &Shape,
    stride: usize,
) -> Tensor<T> {
    let fu = if full_unpadded_shape.len() == 4 {
        full_unpadded_shape.slice(1..)
    } else {
        full_unpadded_shape.clone()
    };
    let c_valid = fu[0];
    let h_valid = fu[1];
    let w_valid = fu[2];
    // Number of strided valid output positions
    let h_s = h_valid / stride; // floor division
    let w_s = w_valid / stride;
    let h_s_pot = h_s.next_power_of_two();
    let w_s_pot = w_s.next_power_of_two();
    let c_pot = c_valid.next_power_of_two();

    let full_padded_shape = cleared_tensor.get_shape();
    let fp_h = full_padded_shape[1];
    let fp_w = full_padded_shape[2];

    let n = c_pot * h_s_pot * w_s_pot;
    let mut data: Vec<T> = vec![T::default(); n];
    for c in 0..c_valid {
        for h in 0..h_s {
            for w in 0..w_s {
                let src_idx = c * fp_h * fp_w + (h * stride) * fp_w + (w * stride);
                let dst_idx = c * h_s_pot * w_s_pot + h * w_s_pot + w;
                data[dst_idx] = cleared_tensor.get_data()[src_idx];
            }
        }
    }
    Tensor::new(Shape::new(vec![c_pot, h_s_pot, w_s_pot]), data)
}

/// Expand a strided-MLE evaluation point to a full-res-MLE evaluation point.
///
/// The MLE of a `[C_pot, H_s_pot, W_s_pot]` tensor is evaluated at a point with
/// `log2(C_pot * H_s_pot * W_s_pot)` coordinates (LSB first, packed as `r_W || r_H || r_C`).
///
/// We expand to the full-res `[C_pot, n_x, n_x]` point (where `n_x` is the actual
/// POT-padded FFT spatial dimension).
///
/// The expansion maps compact coordinate `h` (in `log_hs` bits) to full-res coordinate
/// `j = h * stride` (in `log_nx` bits), using the LSB-first bit layout:
///
///   compact h:   h_0 h_1 ... h_{log_hs-1}
///   full j=h*s:  [0]*stride_bits  h_0 ... h_{log_hs-1}  [0]*overflow_bits
///
/// where `stride_bits = log2(stride)` and `overflow_bits = log_nx - log_hs - stride_bits`.
///
/// The zero bits at the LSB come from the stride factor (stride-multiples have low bits=0).
/// The zero bits at the MSB come from POT-padding: h_s_pot * stride may be < n_x, so
/// high bits of j are always zero.
///
/// `n_x` is the actual FFT spatial dimension: `next_pow2(H_in + 2*ph)`.  This may be
/// strictly larger than `h_s_pot * stride` when ONNX spatial padding is non-zero.
///
/// Soundness: `compact.evaluate(r_s) == full_cleared.evaluate(expand(r_s))` because
/// the full-res tensor has zeros at all non-stride positions (and garbage positions).
fn expand_strided_point<E: ExtensionField>(
    r_strided: &[E],
    c_pot: usize,
    h_s_pot: usize,
    w_s_pot: usize,
    n_x: usize,
    stride: usize,
) -> Vec<E> {
    debug_assert!(n_x.is_power_of_two(), "n_x must be a power of two");
    debug_assert!(stride.is_power_of_two(), "stride must be a power of two");
    debug_assert!(
        n_x >= h_s_pot * stride,
        "n_x ({n_x}) must be >= h_s_pot * stride ({} * {} = {})",
        h_s_pot,
        stride,
        h_s_pot * stride,
    );
    debug_assert!(
        n_x >= w_s_pot * stride,
        "n_x ({n_x}) must be >= w_s_pot * stride ({} * {} = {})",
        w_s_pot,
        stride,
        w_s_pot * stride,
    );
    let log_c = c_pot.ilog2() as usize;
    let log_ws = w_s_pot.ilog2() as usize;
    let log_hs = h_s_pot.ilog2() as usize;
    let log_nx = n_x.ilog2() as usize;
    let stride_bits = stride.ilog2() as usize;
    // Overflow bits: high bits of j that are always zero because h_s_pot * stride <= n_x.
    let overflow_bits_h = log_nx - log_hs - stride_bits;
    let overflow_bits_w = log_nx - log_ws - stride_bits;

    // r_strided layout (LSB first): r_W[0..log_ws] | r_H[0..log_hs] | r_C[0..log_c]
    let r_w_s = &r_strided[..log_ws];
    let r_h_s = &r_strided[log_ws..log_ws + log_hs];
    let r_c = &r_strided[log_ws + log_hs..];
    debug_assert_eq!(r_c.len(), log_c);

    // Full-res layout: r_W_full[0..log_nx] | r_H_full[0..log_nx] | r_C[0..log_c]
    //
    // W expansion: [0]*stride_bits | r_w_s | [0]*overflow_bits_w
    // H expansion: [0]*stride_bits | r_h_s | [0]*overflow_bits_h
    let mut r_full: Vec<E> = Vec::with_capacity(log_nx + log_nx + log_c);
    // W part: stride zeros at LSB, then compact bits, then overflow zeros at MSB
    for _ in 0..stride_bits {
        r_full.push(E::ZERO);
    }
    r_full.extend_from_slice(r_w_s);
    for _ in 0..overflow_bits_w {
        r_full.push(E::ZERO);
    }
    // H part: stride zeros at LSB, then compact bits, then overflow zeros at MSB
    for _ in 0..stride_bits {
        r_full.push(E::ZERO);
    }
    r_full.extend_from_slice(r_h_s);
    for _ in 0..overflow_bits_h {
        r_full.push(E::ZERO);
    }
    // C part: unchanged
    r_full.extend_from_slice(r_c);

    r_full
}

#[cfg(test)]
mod test {
    use crate::layers::{
        activation::{Activation, Relu},
        dense::Dense,
        pooling::{Maxpool2D, Pooling, maxpool2d_shape},
        provable::evaluate_layer,
    };

    use super::*;
    use ff_ext::GoldilocksExt2;
    use p3_field::FieldAlgebra;

    fn split_garbage(
        fft_output: &Tensor<Element>,
        not_padded_shape: &Shape,
    ) -> (Vec<Element>, Vec<Element>) {
        let mut not_padded_shape = not_padded_shape.to_vec();
        not_padded_shape.remove(0);
        let mut garbage = Vec::new();
        let mut valid = Vec::new();
        for i in 0..fft_output.shape[0] {
            for j in 0..fft_output.shape[1] {
                for k in 0..fft_output.shape[2] {
                    let index =
                        i * fft_output.shape[1] * fft_output.shape[2] + j * fft_output.shape[2] + k;
                    let elem = fft_output.data[index];
                    if i < not_padded_shape[0] && j < not_padded_shape[1] && k < not_padded_shape[2]
                    {
                        valid.push(elem);
                    } else {
                        garbage.push(elem);
                    }
                }
            }
        }
        (valid, garbage)
    }
    fn subtest_clearing_methods(padded_tensor: &Tensor<Element>, unpadded_shape: &Shape) {
        let clearing_tensor = new_clearing_tensor(&unpadded_shape, &padded_tensor.get_shape());
        let cleared_tensor = padded_tensor.flatten().mul(&clearing_tensor);
        let auto_cleared_tensor = clear_garbage(padded_tensor, unpadded_shape);
        assert_eq!(cleared_tensor.get_data(), auto_cleared_tensor.get_data());
    }

    #[test]
    fn test_conv_clearing_garbage() {
        let shape: Vec<usize> = vec![5, 18, 18];
        let padded_shape = shape
            .iter()
            .map(|x| x.next_power_of_two())
            .collect::<Shape>();
        let tensor = Tensor::random(&padded_shape);
        subtest_clearing_methods(&tensor, &shape.into());
        // let clearing_tensor = new_clearing_tensor(&shape, &padded_shape);
        // let cleared_tensor = tensor.flatten().mul(&clearing_tensor);
        // let auto_cleared_tensor = clear_garbage(&tensor, &shape);
        // assert_eq!(cleared_tensor.get_data(), auto_cleared_tensor.get_data());
    }

    #[test]
    fn test_conv2d_shape() {
        let input_shape: Shape = vec![1, 23, 23].into();
        let conv_shape_og: Shape = vec![7, 1, 3, 3].into();
        let output_shape = conv2d_shape(&input_shape, &conv_shape_og);
        assert_eq!(output_shape, vec![7, 21, 21].into());
    }

    /// Test that check if just taking shapes from input and conv not padded we can manipulate input
    /// and filter to run it in padded world with FFT based convolution.
    #[test]
    fn test_conv_unpadded_to_padded() {
        let input_shape: Shape = vec![1, 23, 23].into();
        let conv_shape_og: Shape = vec![7, 1, 3, 3].into();
        // let input_shape: Shape = vec![1, 5, 5];
        // let conv_shape_og: Shape = vec![1, 1, 2, 2];
        let weight = Tensor::random(&conv_shape_og);
        let bias: Tensor<Element> = Tensor::zeros(vec![conv_shape_og[0]].into());
        let input = Tensor::random(&input_shape);
        let output = input.conv2d(&weight, &bias, 1);
        // now try to pad the input and conv and use the fft one
        let padded_input = input.pad_next_power_of_two();
        let fft_conv = Convolution::new(weight.clone(), bias).into_padded_and_ffted(&input_shape);
        let (fft_output, conv_data) = fft_conv.op::<GoldilocksExt2>(&padded_input, &input_shape);
        let (valid, _garbage) = split_garbage(&fft_output, &output.get_shape());
        assert_eq!(
            valid,
            output.get_data().to_vec(),
            "valid {:?} is not equal to {:?}",
            &valid[..40],
            &output.get_data()[..40]
        );
        // make sure the shape matches between what we can compute from unpadded and the actual fft output
        let exp_output_shape = conv2d_shape(&input_shape, &conv_shape_og);
        let mut given_output_shape = output.get_shape();
        given_output_shape.remove(0);
        assert_eq!(given_output_shape, exp_output_shape);

        // make sure we can reconstruct the fft output purely from conv_data since it's needed for proving
        let weight_padded_shape = weight
            .get_shape()
            .iter()
            .map(|x| x.next_power_of_two())
            .collect::<Shape>();
        let fft_output_shape = conv2d_shape(&padded_input.get_shape(), &weight_padded_shape);
        let fft_output_shape = fft_output_shape
            .iter()
            .map(|x| x.next_power_of_two())
            .collect::<Shape>();
        println!(
            "INSIDE TEST: fft_output.shape() : {:?}",
            fft_output.get_shape()
        );
        println!(
            "INSIDE TEST: fft_output_shape conv2d_shape(): {:?}",
            fft_output_shape
        );
        println!(
            "INSIDE TEST: padded_input shape: {:?}",
            padded_input.get_shape()
        );
        assert_eq!(fft_output.get_shape(), fft_output_shape);
        // let fft_output_data = conv_data.output_as_element(padded_input.get_shape()[1].next_power_of_two());
        let fft_output_data = conv_data.output_as_element;
        let reconstructed_fft_tensor = Tensor::new(fft_output_shape.clone(), fft_output_data);
        subtest_clearing_methods(&reconstructed_fft_tensor, &output.get_shape());
        // let cleared_reconstructed_fft_tensor = clear_garbage(&reconstructed_fft_tensor, &output.get_shape());
        let hadamard_clearing = new_clearing_tensor(&output.get_shape(), &fft_output_shape);
        let hadamard_cleared = reconstructed_fft_tensor.flatten().mul(&hadamard_clearing);
        assert_eq!(hadamard_cleared.get_data(), fft_output.get_data());
    }

    #[test]
    fn test_conv_padding_garbage() {
        let input_shape: Shape = vec![1, 23, 23].into();
        let conv_shape_og: Shape = vec![7, 1, 3, 3].into();

        // weight of the filter
        let w1 = Tensor::random(&conv_shape_og);
        let bias1: Tensor<Element> = Tensor::zeros(vec![conv_shape_og[0]].into());
        // creation of the padded and fft'd convolution
        let fft_conv =
            Convolution::new(w1.clone(), bias1.clone()).into_padded_and_ffted(&input_shape);
        let input = Tensor::random(&input_shape);
        let padded_input = input.pad_next_power_of_two();
        let (fft_output, _): (Tensor<Element>, ConvData<_>) =
            fft_conv.op::<GoldilocksExt2>(&padded_input, &input_shape);
        // just normal convolution
        let normal_output = input.conv2d(&w1, &bias1, 1);

        // Flatten for the dense layer
        let flat_fft_output = fft_output.flatten();
        let flat_normal_output = normal_output.flatten();
        // Check that the garbage and valid parts are correct
        let (valid, garbage) = split_garbage(&fft_output, &normal_output.get_shape());
        assert!(valid.len() == flat_normal_output.get_data().len());
        assert_eq!(valid, flat_normal_output.get_data().to_vec());
        assert!(!garbage.is_empty());
        // NOTE: a bit of a hack to recreate but the functione xpects the real conv shape not the flattened one
        let (valid, garbage) = split_garbage(
            &Tensor::new(fft_output.get_shape(), flat_fft_output.get_data().to_vec()),
            &normal_output.get_shape(),
        );
        // at this point the garbage should be all zeros and the valid should be the same as the non fft output as before
        assert!(garbage.iter().all(|x| *x == 0));
        assert!(valid == flat_normal_output.get_data().to_vec());

        // dense output to REMOVE garbage - even tho it is only zero now we still need to remove it to get the right shape
        // dense layer should have exactly the same number of columns as the flat normal output
        let ncols = flat_normal_output.shape[0];
        let nrows = 10;
        let dense_shape = vec![nrows, ncols];
        let dense = Dense::new(
            Tensor::new(
                dense_shape.clone().into(),
                vec![1; dense_shape.iter().product()],
            ),
            Tensor::zeros(vec![dense_shape[0]].into()),
        );
        // create the padded version:
        // take the "conv2d"input shape
        let conv_input_shape = conv2d_shape(&input_shape, &w1.get_shape());
        let conv_input_shape_padded = conv_input_shape.next_power_of_two();
        let dense_shape_padded = vec![
            nrows.next_power_of_two(),
            flat_fft_output.shape[0].next_power_of_two(),
        ];
        let mut padded_dense = dense.clone();
        padded_dense.matrix = padded_dense.matrix.pad_matrix_to_ignore_garbage(
            &conv_input_shape,
            &conv_input_shape_padded,
            &dense_shape_padded.into(),
        );
        let padded_nrows = padded_dense.nrows();
        padded_dense.bias = padded_dense.bias.pad_1d(padded_nrows);
        let no_garbage_fft_output =
            evaluate_layer::<GoldilocksExt2, _, _>(&padded_dense, &vec![&flat_fft_output], None)
                .unwrap()
                .outputs()[0]
                .clone();
        let no_garbage_normal_output =
            evaluate_layer::<GoldilocksExt2, _, _>(&dense, &vec![&flat_normal_output], None)
                .unwrap()
                .outputs()[0]
                .clone();
        let max_rows = dense.nrows();
        assert_eq!(
            &no_garbage_fft_output.get_data()[..max_rows],
            &no_garbage_normal_output.get_data()[..]
        );
        assert!(
            no_garbage_fft_output.get_data()[max_rows..]
                .iter()
                .all(|x| *x == 0)
        );
        // let ignore_garbage = create_ignore_garbage(input_shape, input_shape_padded);

        // assert_eq!(fft_output.get_shape(), normal_output.get_shape());
        // assert_eq!(fft_output.data.len(), normal_output.data.len());
        // assert!(fft_output.data == normal_output.data);
    }

    #[test]
    pub fn test_conv_fft_vs_naive() -> anyhow::Result<()> {
        let n_w = 1 << 2;
        let k_w = 1 << 0;
        let k_x = 1 << 0;

        let mut input_shape_og: Shape = vec![k_x, 256, 256].into();
        let mut input_shape_padded: Shape = input_shape_og.next_power_of_two().into();
        let filter = Tensor::random(&vec![k_w, k_x, n_w, n_w].into());
        let bias = Tensor::random(&vec![k_w].into());
        let input = Tensor::random(&input_shape_og);

        let output = input.conv2d(&filter, &bias, 1);
        let dims = filter.get_shape();
        let fft_conv =
            Convolution::new(filter.clone(), bias).into_padded_and_ffted(&input_shape_padded);
        let mut fft_input = input.clone();
        fft_input.pad_to_shape(input_shape_padded.clone());
        let (fft_output, _proving_data) =
            fft_conv.op::<GoldilocksExt2>(&fft_input, &input_shape_og);

        input_shape_og = conv2d_shape(&input_shape_og, &filter.get_shape());
        input_shape_padded = conv2d_shape(&input_shape_padded, &dims).next_power_of_two();

        // add a RELU layer
        let relu = Activation::Relu(Relu::new());
        let output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &vec![&output], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let fft_output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &vec![&fft_output], None)
            .unwrap()
            .outputs()[0]
            .clone();

        // make a pooled output
        let pool = Pooling::Maxpool2D(Maxpool2D::default());
        let output = pool.op(&output);
        let fft_output = pool.op(&fft_output);
        input_shape_og = maxpool2d_shape(&input_shape_og);
        input_shape_padded = maxpool2d_shape(&input_shape_padded);

        // again another conv
        let filter = Tensor::random(&vec![k_w, k_x, n_w, n_w].into());
        let bias = Tensor::random(&vec![k_w].into());
        println!("2AND CONV: filter.get_shape() : {:?}", filter.get_shape());
        println!("2AND CONV: bias.get_shape() : {:?}", bias.get_shape());
        println!("2AND CONV: input.get_shape() : {:?}", output.get_shape());
        let output = output.conv2d(&filter, &bias, 1);
        let dims = filter.get_shape();
        let fft_conv =
            Convolution::new(filter.clone(), bias).into_padded_and_ffted(&input_shape_padded);
        let mut fft_input = fft_output;
        fft_input.pad_to_shape(input_shape_padded.clone());
        let (fft_output, _proving_data) =
            fft_conv.op::<GoldilocksExt2>(&fft_input, &input_shape_og);

        input_shape_og = conv2d_shape(&input_shape_og, &filter.get_shape());
        input_shape_padded = conv2d_shape(&input_shape_padded, &dims).next_power_of_two();

        // Add another RELU
        let relu = Activation::Relu(Relu::new());
        let output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &vec![&output], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let fft_output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &vec![&fft_output], None)
            .unwrap()
            .outputs()[0]
            .clone();

        // make a pooled output
        let pool = Pooling::Maxpool2D(Maxpool2D::default());
        let output = pool.op(&output);
        let fft_output = pool.op(&fft_output);
        input_shape_og = maxpool2d_shape(&input_shape_og);
        input_shape_padded = maxpool2d_shape(&input_shape_padded);

        // now dense layer - first there is a "reshape" that flattens the input
        let ignore_garbage_pad = (input_shape_og.clone(), input_shape_padded.clone());
        input_shape_og = vec![input_shape_og.iter().product()].into();
        input_shape_padded = vec![input_shape_padded.iter().product()].into();

        let nrows = 10;
        let ncols = input_shape_og[0];
        let weight = Tensor::random(&vec![nrows, ncols].into());
        let bias = Tensor::random(&vec![nrows].into());
        let mut new_cols = ncols.next_power_of_two();
        let new_rows = nrows.next_power_of_two();
        if new_cols < input_shape_padded[0] {
            // must make sure that we can apply the input to this padded dense
            new_cols = input_shape_padded[0];
        }
        let conv_shape_og = ignore_garbage_pad.0.clone();
        let conv_shape_pad = ignore_garbage_pad.1.clone();
        let dense = Dense::new(weight.clone(), bias.clone());
        let dense_output = evaluate_layer::<GoldilocksExt2, _, _>(&dense, &vec![&output], None)
            .unwrap()
            .outputs()[0]
            .clone();

        let fft_weight = weight.pad_matrix_to_ignore_garbage(
            &conv_shape_og,
            &conv_shape_pad,
            &vec![new_rows, new_cols].into(),
        );
        let fft_bias = bias.clone().pad_1d(new_rows);
        let fft_dense = Dense::new(fft_weight.clone(), fft_bias.clone());
        println!("-- new_rows : {}, new_cols : {}", new_rows, new_cols);
        println!("weight.get_shape() : {:?}", weight.get_shape());
        println!("bias.get_shape() : {:?}", bias.get_shape());
        println!("fft_input.get_shape() : {:?}", fft_output.get_shape());
        println!("fft_weight.get_shape() : {:?}", fft_weight.get_shape());
        println!("fft_bias.get_shape() : {:?}", fft_bias.get_shape());
        println!(
            "output shape : {:?} - product {}",
            output.get_shape(),
            output.get_shape().iter().product::<usize>()
        );
        let fft_dense_output =
            evaluate_layer::<GoldilocksExt2, _, _>(&fft_dense, &vec![&fft_output], None)
                .unwrap()
                .outputs()[0]
                .clone();
        assert_eq!(
            dense_output.get_data()[..weight.nrows_2d()],
            fft_dense_output.get_data()[..weight.nrows_2d()]
        );
        Ok(())
    }

    /// Test that a Convolution with input_padding=[1,1] (ONNX "same" padding for kernel=3)
    /// produces an output that matches naive zero-padding + conv2d, and that output spatial
    /// dims equal input spatial dims ("same" convolution invariant).
    #[test]
    fn test_conv_with_padding() {
        // Input: 1 channel, 8x8 spatial — padded to effective 10x10 with pad=1 on each side
        let input_shape: Shape = vec![1, 8, 8].into();
        let filter_shape: Shape = vec![4, 1, 3, 3].into(); // 4 out-channels, kernel=3

        let filter: Tensor<f32> = Tensor::random(&filter_shape);
        let bias: Tensor<f32> = Tensor::zeros(vec![filter_shape[0]].into());
        let input: Tensor<f32> = Tensor::random(&input_shape);

        // --- f32 path: Convolution::new_with_padding ---
        let conv = Convolution::new_with_padding(filter.clone(), bias.clone(), [1, 1]);
        let padded_input_naive = input.zero_pad_spatial(1, 1);
        let expected_output = padded_input_naive.conv2d(&filter, &bias, 1);

        let actual_output: Tensor<f32> = conv.op::<GoldilocksExt2>(&input);

        assert_eq!(
            actual_output.get_shape(),
            expected_output.get_shape(),
            "output shapes must match"
        );
        let actual_data = actual_output.get_data();
        let expected_data = expected_output.get_data();
        for (i, (a, e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-4,
                "mismatch at index {}: actual={}, expected={}",
                i,
                a,
                e
            );
        }

        // "same" invariant: output H == input H, output W == input W
        // output_shape returns [C_out, H_out, W_out]
        let out_shape = actual_output.get_shape();
        // remove batch dim if present
        let spatial_h = if out_shape.len() == 4 {
            out_shape[2]
        } else {
            out_shape[1]
        };
        let spatial_w = if out_shape.len() == 4 {
            out_shape[3]
        } else {
            out_shape[2]
        };
        assert_eq!(spatial_h, 8, "output H must equal input H for same-conv");
        assert_eq!(spatial_w, 8, "output W must equal input W for same-conv");
    }

    /// Test the Element (quantised / ZK) path for padding=1 convolution.
    /// Verifies the FFT-based convolution with input_padding produces correct valid values.
    #[test]
    fn test_conv_with_padding_element() {
        // Small input to keep FFT test fast
        let input_shape: Shape = vec![1, 6, 6].into();
        let filter_shape: Shape = vec![2, 1, 3, 3].into(); // 2 out-channels, kernel=3

        let filter: Tensor<Element> = Tensor::random(&filter_shape);
        let bias: Tensor<Element> = Tensor::zeros(vec![filter_shape[0]].into());
        let input: Tensor<Element> = Tensor::random(&input_shape);

        // Build the padded+ffted convolution with input_padding=[1,1]
        let conv = Convolution::new_with_padding(filter.clone(), bias.clone(), [1, 1]);
        let fft_conv = conv.into_padded_and_ffted(&input_shape);

        // Run the FFT conv op
        let padded_input = input.pad_next_power_of_two();
        let (fft_output, _conv_data) = fft_conv.op::<GoldilocksExt2>(&padded_input, &input_shape);

        // Naive reference: zero-pad input then conv2d
        let padded_input_naive = input.zero_pad_spatial(1, 1);
        let expected_output = padded_input_naive.conv2d(&filter, &bias, 1);
        let expected_shape = expected_output.get_shape();

        // Extract valid region from FFT output (ignoring power-of-two garbage)
        let (valid, _garbage) = split_garbage(&fft_output, &expected_shape);
        assert_eq!(
            valid,
            expected_output.get_data().to_vec(),
            "FFT conv with input_padding=[1,1] must match naive zero-pad + conv2d"
        );

        // "same" invariant
        let out_h = if expected_shape.len() == 4 {
            expected_shape[2]
        } else {
            expected_shape[1]
        };
        let out_w = if expected_shape.len() == 4 {
            expected_shape[3]
        } else {
            expected_shape[2]
        };
        assert_eq!(out_h, 6, "output H must equal input H for same-conv");
        assert_eq!(out_w, 6, "output W must equal input W for same-conv");
    }

    /// Verify that the strided shape functions produce the correct output dimensions.
    #[test]
    fn test_strided_shape_functions() {
        // input [1, 8, 8], kernel 3x3, stride 2, no padding → out = (8-3)/2+1 = 3
        let input_shape: Shape = vec![1, 8, 8].into();
        let filter_shape: Shape = vec![4, 1, 3, 3].into();
        let out = conv2d_shape_with_padding_and_stride(&input_shape, &filter_shape, 0, 0, 2, 2);
        assert_eq!(out, Shape::new(vec![4, 3, 3]));

        // POT-padded version: 3 → next_pow2 = 4
        let out_pot =
            padded_conv2d_shape_with_padding_and_stride(&input_shape, &filter_shape, 0, 0, 2, 2);
        assert_eq!(out_pot, Shape::new(vec![4, 4, 4]));

        // stride=1 should equal no-stride version
        let out1 = conv2d_shape_with_padding_and_stride(&input_shape, &filter_shape, 0, 0, 1, 1);
        let out_ns = conv2d_shape_with_padding(&input_shape, &filter_shape, 0, 0);
        assert_eq!(out1, out_ns);
    }

    /// Verify that `clear_garbage_strided` zeroes out non-stride positions and garbage,
    /// and `compact_strided` produces a tensor whose values equal stride-indexed positions.
    #[test]
    fn test_clear_and_compact_strided() {
        // Full-res output shape: [2, 8, 8] (POT-padded; valid region [2, 6, 6])
        let full_padded_shape: Shape = vec![2, 8, 8].into();
        let full_valid_shape: Shape = vec![2, 6, 6].into();
        let tensor: Tensor<Element> = Tensor::random(&full_padded_shape);
        let stride = 2;

        let cleared = clear_garbage_strided(&tensor, &full_valid_shape, stride);

        // Verify: positions (c, j, k) with j%2!=0 or k%2!=0 or j>=6 or k>=6 must be 0
        for c in 0..2usize {
            for j in 0..8usize {
                for k in 0..8usize {
                    let idx = c * 64 + j * 8 + k;
                    let val = cleared.get_data()[idx];
                    let should_be_nonzero = c < 2 && j < 6 && k < 6 && j % 2 == 0 && k % 2 == 0;
                    if !should_be_nonzero {
                        assert_eq!(val, 0, "Expected 0 at ({c},{j},{k}) but got {val}");
                    } else {
                        // should equal the original tensor at that position
                        assert_eq!(val, tensor.get_data()[idx]);
                    }
                }
            }
        }

        // Compact: valid strided dims = 6/2=3 each, POT-padded to 4
        let compact = compact_strided(&cleared, &full_valid_shape, stride);
        assert_eq!(compact.get_shape(), Shape::new(vec![2, 4, 4]));
        // Check that compact[c, h, w] == original[c, 2h, 2w] for h,w < 3
        for c in 0..2usize {
            for h in 0..3usize {
                for w in 0..3usize {
                    let src = tensor.get_data()[c * 64 + (h * 2) * 8 + (w * 2)];
                    let dst = compact.get_data()[c * 16 + h * 4 + w];
                    assert_eq!(src, dst, "compact mismatch at ({c},{h},{w})");
                }
            }
        }
    }

    /// Verify the MLE identity: compact_output.evaluate(r_s) == full_cleared.evaluate(expand(r_s)).
    /// This is the key soundness property for the strided clearing hadamard proof.
    #[test]
    fn test_expand_strided_point_mle_identity() {
        use multilinear_extensions::mle::MultilinearExtension;
        type E = GoldilocksExt2;

        // Build a small strided output
        // full-res valid [2, 4, 4], stride=2 → compact valid [2, 2, 2], POT→ compact [2, 2, 2]
        let full_valid: Shape = vec![2, 4, 4].into();
        let full_pot: Shape = vec![2, 4, 4].into(); // already POT
        let stride = 2;

        // Create tensor with known values at stride positions
        let mut data = vec![0i64; 2 * 4 * 4];
        // Set values at stride-multiple positions
        for c in 0..2usize {
            for h in 0..2usize {
                for w in 0..2usize {
                    let val = (c * 4 + h * 2 + w + 1) as i64;
                    data[c * 16 + (h * 2) * 4 + (w * 2)] = val;
                }
            }
        }
        let full_tensor: Tensor<Element> = Tensor::new(full_pot.clone(), data);

        // Build the compact strided tensor
        let compact = compact_strided(&full_tensor, &full_valid, stride);

        // Choose a random point in the strided space
        let c_pot = compact.get_shape()[0]; // 2
        let h_s_pot = compact.get_shape()[1]; // 2
        let w_s_pot = compact.get_shape()[2]; // 2
        let log_c = c_pot.ilog2() as usize; // 1
        let log_hs = h_s_pot.ilog2() as usize; // 1
        let log_ws = w_s_pot.ilog2() as usize; // 1
        let total_bits = log_ws + log_hs + log_c; // 3

        // Use a simple deterministic "random" point
        let r_s: Vec<E> = (0..total_bits)
            .map(|i| {
                E::from_canonical_u64(match i {
                    0 => 3, // r_W bit
                    1 => 7, // r_H bit
                    2 => 2, // r_C bit
                    _ => i as u64,
                })
            })
            .collect();

        // Convert Element tensors to field tensors for MLE evaluation
        let compact_fields: Tensor<E> = compact.to_fields();
        let full_fields: Tensor<E> = full_tensor.to_fields();

        // Evaluate compact MLE at r_s
        let compact_eval: E = compact_fields.get_data().to_vec().into_mle().evaluate(&r_s);

        // Expand point and evaluate full-res MLE.
        // n_x is the actual full-res spatial dimension (= full_pot[1] = 4).
        let n_x = full_pot[1]; // 4
        let r_full = expand_strided_point::<E>(&r_s, c_pot, h_s_pot, w_s_pot, n_x, stride);
        let full_eval: E = full_fields.get_data().to_vec().into_mle().evaluate(&r_full);

        assert_eq!(
            compact_eval, full_eval,
            "MLE identity failed: compact.evaluate(r_s) != full_cleared.evaluate(expand(r_s))"
        );
    }

    /// Verify that the new_clearing_tensor_strided marks exactly the stride positions.
    #[test]
    fn test_new_clearing_tensor_strided() {
        // valid strided [2, 3, 3], full padded [2, 8, 8], stride=2
        // stride positions: (c, j*2, k*2) for j<3, k<3, c<2
        let strided_valid: Shape = vec![2, 3, 3].into();
        let full_padded: Shape = vec![2, 8, 8].into();
        let t = new_clearing_tensor_strided(&strided_valid, &full_padded, 2);
        // Should have product(full_padded) elements
        assert_eq!(t.get_data().len(), 2 * 8 * 8);
        let data = t.get_data();
        for c in 0..2usize {
            for j in 0..8usize {
                for k in 0..8usize {
                    let idx = c * 64 + j * 8 + k;
                    let expected = if c < 2 && j % 2 == 0 && k % 2 == 0 {
                        let h_s = j / 2;
                        let w_s = k / 2;
                        if h_s < 3 && w_s < 3 { 1 } else { 0 }
                    } else {
                        0
                    };
                    assert_eq!(data[idx], expected, "Wrong value at ({c},{j},{k})");
                }
            }
        }
    }

    /// f32 path: verify that a Convolution with stride=2 produces the same result as
    /// manually applying an unstrided conv and then picking every 2nd element.
    #[test]
    fn test_strided_conv_f32_path() {
        let input_shape: Shape = vec![1, 8, 8].into();
        let filter_shape: Shape = vec![2, 1, 3, 3].into(); // 2 out-channels, 3x3 kernel

        let filter: Tensor<f32> = Tensor::random(&filter_shape);
        let bias: Tensor<f32> = Tensor::zeros(vec![filter_shape[0]].into());
        let input: Tensor<f32> = Tensor::random(&input_shape);

        // Strided conv via Convolution::new_with_padding_and_stride
        let conv_strided =
            Convolution::new_with_padding_and_stride(filter.clone(), bias.clone(), [0, 0], [2, 2]);
        let out_strided = conv_strided.op::<GoldilocksExt2>(&input);

        // Reference: unstrided conv, then manually pick positions 0, 2, 4, ...
        let out_full = input.conv2d(&filter, &bias, 1);
        // out_full shape: [2, 6, 6] (8-3+1=6 for stride=1 without padding)
        // strided output should be [2, 3, 3] (6/2=3)
        // Shape may have a leading batch dimension of 1 — remove it for comparison
        let out_shape = out_strided.get_shape();
        let out_shape_nobatch = if out_shape[0] == 1 && out_shape.len() == 4 {
            out_shape.slice(1..)
        } else {
            out_shape.clone()
        };
        assert_eq!(
            out_shape_nobatch,
            Shape::new(vec![2, 3, 3]),
            "strided output shape"
        );

        // Verify values match stride-sampled positions of full-res output
        let full_data = out_full.get_data();
        let strided_data = out_strided.get_data();
        // out_full shape may have a batch dim — find (c, h, w) offsets
        let full_shape = out_full.get_shape();
        let (c_out, h_full, w_full) = if full_shape.len() == 4 {
            (full_shape[1], full_shape[2], full_shape[3])
        } else {
            (full_shape[0], full_shape[1], full_shape[2])
        };
        let strided_shape = if out_shape.len() == 4 {
            (out_shape[1], out_shape[2], out_shape[3])
        } else {
            (out_shape[0], out_shape[1], out_shape[2])
        };
        for c in 0..c_out {
            for h in 0..strided_shape.1 {
                for w in 0..strided_shape.2 {
                    let src = full_data[c * h_full * w_full + (h * 2) * w_full + (w * 2)];
                    let dst = strided_data
                        [c * strided_shape.1 * strided_shape.2 + h * strided_shape.2 + w];
                    assert!(
                        (src - dst).abs() < 1e-5,
                        "f32 stride mismatch at ({c},{h},{w}): src={src} dst={dst}"
                    );
                }
            }
        }
    }
}
