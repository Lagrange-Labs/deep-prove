use crate::{
    ScalingStrategy, VectorTranscript,
    iop::context::ShapeStep,
    layers::{hadamard, requant::Requant},
    model::StepData,
    padding::{PaddingMode, ShapeInfo, pad_conv},
    quantization::{BIT_LEN, TensorFielder},
};
use core::f32;

use crate::{
    Claim, Prover,
    commit::{compute_betas_eval, identity_eval},
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerProof, PolyID},
    quantization::{self, ScalingFactor},
    tensor::{ConvData, Number, get_root_of_unity},
};
use anyhow::{Context, Result, ensure};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
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
use tracing::{debug, warn};
use transcript::Transcript;

use super::{
    LayerCtx,
    provable::{
        Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
        QuantizeOutput, VerifiableCtx,
    },
};

pub(crate) const BIAS_POLY_ID: PolyID = 200_000;

const IS_PROVABLE: bool = true;
/// Convolution layer description (weights)
#[derive(Clone, Debug)]
pub struct Convolution<T> {
    /// NOTE: in the case of f32, the weights are native
    /// In the case of Element (i128), the weights are already fft'd
    pub filter: Tensor<T>,
    /// Same for bias.
    pub bias: Tensor<T>,
    /// Unpadded shape of the filter. This is set to filter's shape in case of no padding.
    pub unpadded_shape: Vec<usize>,
}

/// Info about the convolution layer derived during the setup phase
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvCtx<E> {
    pub poly_id: PolyID,
    pub bias_poly_id: PolyID,
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
    pub unpadded_filter_shape: Vec<usize>,
    pub padded_filter_shape: Vec<usize>,
}

pub fn to_bits<E: ExtensionField>(mut num: usize, bitlen: usize) -> Vec<E> {
    let mut bits = vec![E::ZERO; bitlen];
    for i in 0..bitlen {
        bits[i] = E::from((num & 1) as u64);
        num >>= 1;
    }
    bits
}

#[derive(Clone, Debug)]
pub struct SchoolBookConv<T>(pub(crate) Convolution<T>);

/// Contains proof material related to one step of the inference for a convolution layer
#[derive(Default, Clone, Serialize, Deserialize)]
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

    pub(crate) fn new_without_bias(filter: Tensor<T>) -> Self {
        let bias = Tensor::zeros(vec![filter.kw()]);
        Self::new(filter, bias)
    }

    pub fn new_padded(filter: Tensor<T>, bias: Tensor<T>, unpadded_shape: &[usize]) -> Self {
        assert_eq!(filter.kw(), bias.get_shape()[0]);
        Self {
            filter,
            bias,
            unpadded_shape: unpadded_shape.to_vec(),
        }
    }
    pub fn output_shape(&self, input_shape: &[usize], padding_mode: PaddingMode) -> Vec<usize> {
        match padding_mode {
            // unpadded shape is the shape found in onxx file for example
            PaddingMode::NoPadding => conv2d_shape(input_shape, &self.unpadded_shape),
            PaddingMode::Padding => padded_conv2d_shape(input_shape, &self.filter.real_shape()),
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

    pub fn get_shape(&self) -> Vec<usize> {
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
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape.as_slice(), padding_mode))
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
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating convolution layer"
        );
        let input = inputs[0];
        Ok(LayerOut::from_vec(vec![input.conv2d(
            &self.filter,
            &self.bias,
            1,
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
        Convolution::<Element>::new(quantized_filter, bias)
    }

    pub fn op<E: ExtensionField>(&self, input: &Tensor<f32>) -> Tensor<f32> {
        input.conv2d(&self.filter, &self.bias, 1)
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
        unpadded_input_shapes: Vec<Vec<usize>>,
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
        let (output, proving_data) = self.op(input, unpadded_input_shapes[0].as_slice());
        Ok(LayerOut {
            outputs: vec![output],
            proving_data: Some(proving_data),
        })
    }
}

impl Convolution<Element> {
    /// Pads the filter and bias, and adapt the filter to the convolution fft operation.
    pub fn into_padded_and_ffted(mut self, unpadded_input_shape: &[usize]) -> Self {
        self.filter = self.filter.pad_next_power_of_two();
        self.bias = self.bias.pad_next_power_of_two();
        let padded_input_shape = unpadded_input_shape
            .iter()
            .map(|&x| x.next_power_of_two())
            .collect::<Vec<usize>>();
        self.filter = self.filter.into_fft_conv(&padded_input_shape);
        self
    }

    pub fn op<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
        unpadded_input_shape: &[usize],
    ) -> (Tensor<Element>, ConvData<E>) {
        let (output, mut proving_data) = self.filter.fft_conv(input);
        let conv_output = self.add_bias(&output);
        // we record here the output _after_ the bias addition. During proving it's necessary since we're proving the clearing garbage
        // and that produces a new claim on this output.
        proving_data.set_output(conv_output.get_data());
        // At this stage, we're creating a "garbage clearing" tensor that sets all garbage values to 0. This is necessary
        // since the garbage might be of any value and we need to restrict the range of the output due to requantization proving logic.
        let unpadded_output_shape = conv2d_shape(unpadded_input_shape, &self.unpadded_shape);
        debug_assert_eq!(
            { padded_conv2d_shape(&input.get_shape(), &self.filter.real_shape()) },
            conv_output.get_shape(),
            "FFT output shape not computable"
        );
        let cleared_tensor = clear_garbage(&conv_output, &unpadded_output_shape);
        debug_assert!({
            // check that applying the clearing tensor to the conv output gives the same result - as we'd be using the clearing tensor
            // for proving
            let clearing_tensor =
                new_clearing_tensor(&unpadded_output_shape, &conv_output.get_shape());
            let cleared_tensor2 = conv_output.flatten().mul(&clearing_tensor);
            cleared_tensor.get_data() == cleared_tensor2.get_data()
        });
        (cleared_tensor, proving_data)
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

    pub fn prove_batch_fft_weights<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        r: Vec<E>,
    ) -> (
        sumcheck::structs::IOPProof<E>,
        Vec<E>,
        Vec<E>,
        (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
    )
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        let padded_rows = 2 * self.filter.nw() * self.filter.nw();
        let mut w1_reduced: Vec<E> = vec![E::ZERO; self.filter.real_nw() * self.filter.real_nw()];

        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; padded_rows.ilog2() as usize];
        let mut r2 = vec![E::ZERO; r.len() - padded_rows.ilog2() as usize];
        for i in 0..r1.len() {
            r1[i] = r[i];
        }

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
            E::from(1),
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
        (
            proof,
            claims,
            partial_evals,
            prover.delegate_matrix_evaluation(&mut f_middle, r1.clone(), out_point, false),
        )
    }
}

impl<E> ProveInfo<E> for Convolution<Element>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: PolyID, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        let mut filter_shape = self.filter.get_shape();
        filter_shape.remove(1);
        aux.last_output_shape
            .iter_mut()
            .for_each(|shape| *shape = filter_shape.clone());

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
            poly_id: id,
            bias_poly_id: BIAS_POLY_ID + id,
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
        });
        Ok((conv_info, aux))
    }

    fn commit_info(&self, id: NodeId) -> Vec<Option<(PolyID, Vec<E>)>> {
        let filter_evals = self.filter.get_conv_weights();
        let bias_evals = self.bias.evals_flat();
        let id = id as PolyID;
        debug!(
            "Commitment : conv layer ID {}: size {}",
            id,
            filter_evals.len().ilog2()
        );
        debug!(
            "Commitment : conv layer bias ID {}: size {}",
            BIAS_POLY_ID + id,
            bias_evals.len().ilog2()
        );
        vec![
            Some((id, filter_evals)),
            Some((BIAS_POLY_ID + id, bias_evals)),
        ]
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
        let shift = input_scaling.shift(&model_scaling, &output_scaling);
        let (quantized_min, _quantized_max) =
            quantized_conv.output_range(*quantization::MIN, *quantization::MAX);
        let requant = Requant::new(quantized_min.unsigned_abs() as usize, shift);
        Ok(QuantizeOutput {
            quanzited_op: quantized_conv,
            output_scalings: vec![output_scaling],
            requant_layer: Some(requant),
        })
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

impl<E> ProvableOp<E> for Convolution<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Ctx = ConvCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T>,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.prove_convolution_step(
            prover,
            last_claims[0],
            step_data.outputs.outputs()[0],
            &step_data.unpadded_output_shapes[0],
            step_data.outputs.proving_data.as_ref().unwrap(),
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
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
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

impl<E> VerifiableCtx<E> for ConvCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = ConvProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.verify_convolution(
            verifier,
            last_claims[0],
            proof,
            shape_step,
        )?])
    }
}

impl Convolution<Element> {
    // Prove convolution of a CNN network. This is a convolution between in a 3D matrix X of dimension k_x * n_x * n_x
    // and a 4D filter matrix W of dimension k_w * k_x * n_w * n_w. The output is a 3D matrix Y of dimension k_w * n_x * n_x
    // We want to batch prove the following: Y[i] = iFFT(sum_{j \in [n_x]}(FFT(X[j]) o FFT(W[i][j])).
    #[timed::timed_instrument(name = "Prover::prove_convolution_step")]
    pub fn prove_convolution_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        // last random claim made
        last_claim: &Claim<E>,
        // Struct containing all necessary information
        // to generate a convolution proof
        output: &Tensor<E>,
        unpadded_output_shape: &[usize],
        proving_data: &ConvData<E>,
        info: &ConvCtx<E>,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // First part is proving the clearing of the garbage has been done correctly.
        // For this, we create the clearing garbage tensor and just prove hadamard with the output.
        // This results in two claims: one for the non-cleared tensor and one for the clearing tensor (only 1s and 0s)
        // The non-cleared tensor claim gets passed to the main regular logic of convolution
        // The clearing tensor one gets stored in the proof and will be checked manually by the verifier (CURRENTLY)
        let clearing_tensor = new_clearing_tensor(unpadded_output_shape, &output.get_shape());
        // Take the elements BEFORE bias addition - this is what the rest of the convolution proving step expects.
        // TODO: could trade off less memory by directly recomputing it from conv data with the input shape as well.
        let conv_after_bias =
            Tensor::new(output.get_shape(), proving_data.output_as_element.clone());
        debug_assert!({
            println!(
                "PROVE: conv_after_bias.shape(): {:?}",
                conv_after_bias.get_shape()
            );
            println!(
                "PROVE: conv_after_bias.data(): {:?}",
                &conv_after_bias.get_data()[..30]
            );
            println!("PROVE: unpadded_output_shape: {unpadded_output_shape:?}");
            println!("PROVE: output.shape(): {:?}", output.get_shape());
            let cleared_out = conv_after_bias.flatten().mul(&clearing_tensor);
            let fielded: Tensor<E> = cleared_out.to_fields();
            fielded.get_data().to_vec() == output.get_data()
        });
        let clearing_proof = hadamard::prove(
            prover.transcript,
            last_claim,
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
        for i in 0..(filter.filter_size().ilog2() as usize) {
            r[i] = E::ONE - last_claim.point[i];
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
        let (ifft_proof, ifft_claim, ifft_del_proof) =
            prover.prove_batch_ifft(r.clone(), &proving_data.prod);

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
            println!("iFFT Sumcheck Correct");
            true
        });

        // After this point, the verifier holds an evaluation claim of proving_data.prod at P1.randomness[0][i]
        // Let r' = P1.randomness[0][i] and y is the evaluation claim of prod = proving_data.prod
        // What we want to do now is to prove that prod has been correctly computed from X_fft and w (= proving_data.w)
        // In other words we want to show that prod[i] = sum_{j \in [k_x]} x[j] o w[i][j] for each i in [k_w]
        // For this let r1 be the last log(k_w) elements of r and r2 the first log(n_x^2) elements
        // Compute the arrays beta1,beta2 such that beta1[i] = beta(i,r1) and beta2[i] = beta(i,r2)

        let mut r_ifft: Vec<E> = ifft_proof.point.clone();
        for i in (proving_data.output[0].len().ilog2() as usize)..r.len() {
            r_ifft.push(r[i]);
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
        // and v3 = [beta2,..(k_x times)..,beta2]. So, first initialzie v3 and then invoke the cubic sumceck.
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
        let (fft_proof, fft_claim, fft_del_proof) = prover.prove_batch_fft(
            hadamard_proof.point.clone(),
            &mut proving_data.input.clone(),
        );

        let (fft_proof_weights, fft_weight_claims, partial_evals, fft_weights_del_proof) =
            self.prove_batch_fft_weights(prover, point.clone());

        let weights_rand: Vec<E> = prover
            .transcript
            .read_challenges((self.filter.real_nw() * self.filter.real_nw()).ilog2() as usize);
        debug_assert!({
            let mut weights_point = fft_proof_weights.point.clone();
            let mut v_weights = weights_point.pop().unwrap();
            v_weights = (E::ONE - v_weights).invert().unwrap();

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

        prover
            .commit_prover
            .add_claim(
                info.poly_id,
                Claim::new(
                    [
                        weights_rand.clone(),
                        point[(2 * self.filter.nw() * self.filter.nw()).ilog2() as usize..]
                            .to_vec(),
                    ]
                    .concat(),
                    partial_evals.clone().into_mle().evaluate(&weights_rand),
                ),
            )
            .context("unable to add convolution claim")?;
        prover
            .commit_prover
            .add_claim(info.bias_poly_id, Claim::new(bias_point, bias_eval))
            .context("unable to add bias claim in convolution")?;

        prover.push_proof(
            id,
            LayerProof::Convolution(ConvProof {
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
            }),
        );
        let mut input_point = fft_proof.point.clone();
        let mut v = input_point.pop().unwrap();
        v = (E::ONE - v).invert().unwrap();
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
            for i in 0..((filter.filter_size().ilog2()) as usize) {
                p[i] = E::ONE - p[i];
            }
            assert_eq!(
                proving_data.real_input.clone().into_mle().evaluate(&p),
                fft_claim[0] * v,
                "Error in real input eval CONV PROVER"
            );
            proving_data.real_input.clone().into_mle().evaluate(&p) == fft_claim[0] * v
        });
        for i in 0..input_point.len() {
            input_point[i] = E::ONE - input_point[i];
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
    pub fn output_shape(&self, input_shape: &[usize], padding_mode: PaddingMode) -> Vec<usize> {
        match padding_mode {
            PaddingMode::NoPadding => conv2d_shape(input_shape, &self.unpadded_filter_shape),
            PaddingMode::Padding => padded_conv2d_shape(input_shape, &self.padded_filter_shape),
        }
    }
    pub(crate) fn verify_fft_delegation<T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        mut claim: E,
        proof: &ConvProof<E>,
        delegation_proof: &Vec<IOPProof<E>>,
        delegation_claims: &Vec<Vec<E>>,
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
            (E::ONE - E::from(2) * proof.hadamard_proof.point[iter]) * prev_r[0] + E::ONE
                - prev_r[0],
            "Error in final FFT delegation step"
        );
    }

    pub(crate) fn verify_convolution<T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
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
        // To recreat it, we need the unpadded output shape and the real output shape.
        let unpadded_output_shape = conv2d_shape(
            &shape_step.unpadded_input_shape[0],
            &self.unpadded_filter_shape,
        );
        let real_output_shape =
            padded_conv2d_shape(&shape_step.padded_input_shape[0], &self.padded_filter_shape);
        let clearing_tensor = new_clearing_tensor(&unpadded_output_shape, &real_output_shape);
        // now we need to verify the hadamard proof for the sumcheck part.
        let hctx = hadamard::HadamardCtx::from_len(real_output_shape.iter().product());
        let expected_v2_eval = clearing_tensor
            .to_mle_flat()
            .evaluate(proof.clearing_proof.random_point());
        // also set the claim to be the non-cleared output of conv. The rest of the logic is about proving the bias + fft claims.
        let last_claim = hadamard::verify(
            &hctx,
            verifier.transcript,
            &proof.clearing_proof,
            last_claim,
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
        let scale = E::from(1 << (iter + 1)).invert().unwrap();

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
        v = (E::ONE - v).invert().unwrap();

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

        verifier.commit_verifier.add_claim(
            self.poly_id,
            Claim::new(
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
            ),
        )?;

        verifier.commit_verifier.add_claim(
            self.bias_poly_id,
            Claim::new(
                last_claim.point[(proof.ifft_delegation_proof.len())..].to_vec(),
                proof.bias_claim,
            ),
        )?;

        let mut input_point = proof.fft_proof.point.clone();
        v = input_point.pop().unwrap();
        v = (E::ONE - v).invert().unwrap();
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
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
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
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<T, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating schoolbook convolution layer"
        );
        let input = inputs[0];
        Ok(LayerOut::from_vec(vec![input.conv2d(
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
            quanzited_op: SchoolBookConv(self.0.quantize(
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
    fn step_info(&self, _id: PolyID, aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        let conv_info = LayerCtx::SchoolBookConvolution(SchoolBookConvCtx);
        Ok((conv_info, aux))
    }
}

pub fn pow_two_omegas<E: ExtensionField>(n: usize, is_fft: bool) -> Vec<E> {
    let mut pows = vec![E::ZERO; n - 1];
    let mut rou: E = get_root_of_unity(n);
    if is_fft {
        rou = rou.invert().unwrap();
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
        eval = E::ONE - rand1 + (E::ONE - E::from(2) * rand2) * rand1 * eval;
    }

    eval
}

fn clear_garbage<T: Number>(
    output_tensor: &Tensor<T>,
    mut unpadded_output_shape: &[usize],
) -> Tensor<T> {
    if unpadded_output_shape.len() == 4 {
        unpadded_output_shape = &unpadded_output_shape[1..];
    }
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

pub fn new_clearing_tensor(mut og_shape: &[usize], padded_shape: &[usize]) -> Tensor<Element> {
    if og_shape.len() == 4 {
        og_shape = &og_shape[1..];
    }
    assert_eq!(padded_shape.len(), og_shape.len());
    assert_eq!(padded_shape.len(), 3);
    let n = padded_shape.iter().product();
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
    Tensor::new(vec![padded_shape.iter().product()], data)
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
    input_shape: &[usize],
    filter_shape: &[usize],
    padding_mode: PaddingMode,
) -> Vec<usize> {
    match padding_mode {
        PaddingMode::NoPadding => conv2d_shape(input_shape, filter_shape),
        PaddingMode::Padding => padded_conv2d_shape(input_shape, filter_shape),
    }
}

/// Assumes stride=1, padding=0, and dilation=1
/// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
pub fn conv2d_shape(input_shape: &[usize], filter_shape: &[usize]) -> Vec<usize> {
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
    vec![filter_shape[0], h_out, h_out]
}

/// Similar to conv2d_shape but pads the output shape such that it matches what the padded inference and proving expects
pub fn padded_conv2d_shape(input_shape: &[usize], filter_shape: &[usize]) -> Vec<usize> {
    conv2d_shape(input_shape, filter_shape)
        .into_iter()
        .map(|x| x.next_power_of_two())
        .collect::<Vec<usize>>()
}

#[cfg(test)]
mod test {
    use crate::{
        NextPowerOfTwo,
        layers::{
            activation::{Activation, Relu},
            dense::{self, Dense},
            pooling::{Maxpool2D, Pooling, maxpool2d_shape},
            provable::evaluate_layer,
        },
    };

    use super::*;
    use goldilocks::GoldilocksExt2;

    fn split_garbage(
        fft_output: &Tensor<Element>,
        not_padded_shape: &[usize],
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
    fn subtest_clearing_methods(padded_tensor: &Tensor<Element>, unpadded_shape: &[usize]) {
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
            .collect::<Vec<usize>>();
        let tensor = Tensor::random(&padded_shape);
        subtest_clearing_methods(&tensor, &shape);
        // let clearing_tensor = new_clearing_tensor(&shape, &padded_shape);
        // let cleared_tensor = tensor.flatten().mul(&clearing_tensor);
        // let auto_cleared_tensor = clear_garbage(&tensor, &shape);
        // assert_eq!(cleared_tensor.get_data(), auto_cleared_tensor.get_data());
    }

    #[test]
    fn test_conv2d_shape() {
        let input_shape: Vec<usize> = vec![1, 23, 23];
        let conv_shape_og: Vec<usize> = vec![7, 1, 3, 3];
        let output_shape = conv2d_shape(&input_shape, &conv_shape_og);
        assert_eq!(output_shape, vec![7, 21, 21]);
    }

    /// Test that check if just taking shapes from input and conv not padded we can manipualte input
    /// and filter to run it in padded world with FFT based convolution.
    #[test]
    fn test_conv_unpadded_to_padded() {
        let input_shape: Vec<usize> = vec![1, 23, 23];
        let conv_shape_og: Vec<usize> = vec![7, 1, 3, 3];
        // let input_shape: Vec<usize> = vec![1, 5, 5];
        // let conv_shape_og: Vec<usize> = vec![1, 1, 2, 2];
        let weight = Tensor::random(&conv_shape_og);
        let bias: Tensor<Element> = Tensor::zeros(vec![conv_shape_og[0]]);
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
            .collect::<Vec<_>>();
        let fft_output_shape = conv2d_shape(&padded_input.get_shape(), &weight_padded_shape);
        let fft_output_shape = fft_output_shape
            .into_iter()
            .map(|x| x.next_power_of_two())
            .collect::<Vec<usize>>();
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
        let input_shape: Vec<usize> = vec![1, 23, 23];
        let conv_shape_og: Vec<usize> = vec![7, 1, 3, 3];

        // wieght of the filter
        let w1 = Tensor::random(&conv_shape_og);
        let bias1: Tensor<Element> = Tensor::zeros(vec![conv_shape_og[0]]);
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
            Tensor::new(dense_shape.clone(), vec![1; dense_shape.iter().product()]),
            Tensor::zeros(vec![dense_shape[0]]),
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
            &dense_shape_padded,
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
    fn test_conv_offset_poly_id() {
        // just a large difference so we're guaranteed that the IDs won't overlap.
        // TODO: change that process by a deterministic ID depending on the position and additional info
        // not necessarily seuential
        assert!(BIAS_POLY_ID >= dense::BIAS_POLY_ID + 100_000);
    }

    #[test]
    pub fn test_conv_fft_vs_naive() -> anyhow::Result<()> {
        let n_w = 1 << 2;
        let k_w = 1 << 0;
        let k_x = 1 << 0;

        let mut input_shape_og = vec![k_x, 256, 256];
        let mut input_shape_padded = input_shape_og.next_power_of_two();
        let filter = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
        let bias = Tensor::random(&vec![k_w]);
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
        let filter = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
        let bias = Tensor::random(&vec![k_w]);
        println!("2ND CONV: filter.get_shape() : {:?}", filter.get_shape());
        println!("2ND CONV: bias.get_shape() : {:?}", bias.get_shape());
        println!("2ND CONV: input.get_shape() : {:?}", output.get_shape());
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
        input_shape_og = vec![input_shape_og.iter().product()];
        input_shape_padded = vec![input_shape_padded.iter().product()];

        let nrows = 10;
        let ncols = input_shape_og[0];
        let weight = Tensor::random(&vec![nrows, ncols]);
        let bias = Tensor::random(&vec![nrows]);
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
            &vec![new_rows, new_cols],
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
}
