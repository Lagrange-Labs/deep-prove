#![allow(clippy::needless_range_loop)]

use crate::{
    NextPowerOfTwo, ScalingFactor,
    quantization::{self, MAX_FLOAT, MIN_FLOAT},
};
use anyhow::{Result, anyhow, bail, ensure};
use ark_std::rand::Rng;
use ff_ext::{ExtensionField, GoldilocksExt2};
use itertools::Itertools;
use mpcs::util::plonky2_util::log2_ceil;
use multilinear_extensions::{mle::DenseMultilinearExtension, util::ceil_log2};
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_goldilocks::Goldilocks;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    prelude::ParallelSlice,
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, PartialEq},
    fmt::{self, Debug},
    ops::{Bound, RangeBounds},
};

use crate::{
    Element,
    layers::pooling::MAXPOOL2D_KERNEL_SIZE,
    quantization::{Fieldizer, IntoElement},
    to_bit_sequence_le,
};

pub trait Number:
    Copy
    + PartialEq
    + Clone
    + Send
    + Sync
    + Default
    + std::iter::Sum
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::Mul<Output = Self>
    + std::fmt::Debug
{
    const MIN: Self;
    const MAX: Self;
    fn unit() -> Self;
    fn random<R: Rng>(rng: &mut R) -> Self;
    /// reason abs is necessary is because f32 doesn't implement Ord trait, so to have uniform code for f32 and Element,
    /// we implement abs here.
    fn absolute_value(&self) -> Self;
    fn cmp_max(&self, other: &Self) -> Self {
        match self.compare(other) {
            Ordering::Greater => *self,
            Ordering::Equal => *self,
            Ordering::Less => *other,
        }
    }
    fn cmp_min(&self, other: &Self) -> Self {
        match self.compare(other) {
            Ordering::Greater => *other,
            Ordering::Equal => *self,
            Ordering::Less => *self,
        }
    }
    fn compare(&self, other: &Self) -> Ordering;
    fn is_negative(&self) -> bool;
    fn to_f32(&self) -> anyhow::Result<f32>;
    fn from_f32(f: f32) -> anyhow::Result<Self>;
    fn to_usize(&self) -> usize;
    fn from_usize(u: usize) -> Self;
}

impl Number for Element {
    const MIN: Element = Element::MIN;
    const MAX: Element = Element::MAX;
    fn unit() -> Self {
        1
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(*quantization::MIN..=*quantization::MAX)
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
    fn is_negative(&self) -> bool {
        *self < 0
    }
    fn to_f32(&self) -> anyhow::Result<f32> {
        ensure!(
            *self >= f32::MIN.ceil() as Element,
            "Element {self} is smaller than the minimum integer representable by f32"
        );
        ensure!(
            *self <= f32::MAX.floor() as Element,
            "Element {self} is bigger than the maximum integer representable by f32"
        );
        Ok(*self as f32)
    }
    fn from_f32(f: f32) -> anyhow::Result<Self> {
        Ok(f as Element)
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
    fn from_usize(u: usize) -> Self {
        u as Element
    }
}
impl Number for f32 {
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    fn unit() -> Self {
        1.0
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(MIN_FLOAT..=MAX_FLOAT)
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        if self < other {
            Ordering::Less
        } else if self == other {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }

    fn is_negative(&self) -> bool {
        *self < 0.0
    }
    fn to_f32(&self) -> anyhow::Result<f32> {
        Ok(*self)
    }
    fn from_f32(f: f32) -> anyhow::Result<Self> {
        Ok(f)
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
    fn from_usize(u: usize) -> Self {
        u as f32
    }
}
impl Number for GoldilocksExt2 {
    const MIN: GoldilocksExt2 = GoldilocksExt2::ZERO;
    const MAX: GoldilocksExt2 = GoldilocksExt2::ZERO;
    fn unit() -> Self {
        GoldilocksExt2::ONE
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        Element::random(rng).to_field()
    }
    fn absolute_value(&self) -> Self {
        *self
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn is_negative(&self) -> bool {
        panic!("GoldilocksExt2: is_negative is meaningless");
    }

    fn to_f32(&self) -> anyhow::Result<f32> {
        unreachable!("Called to_f32 for Goldilocks")
    }
    fn from_f32(_: f32) -> anyhow::Result<Self> {
        unreachable!("Called from_f32 for Goldilocks")
    }
    fn to_usize(&self) -> usize {
        unreachable!("Called to_usize for Goldilocks")
    }
    fn from_usize(_: usize) -> Self {
        unreachable!("Called from_usize for Goldilocks")
    }
}

/// Function testing the consistency between the actual convolution implementation and
/// the FFT one. Used for debugging purposes.
/// real_tensor is std conv2d (kw, nx-nw+1, nx-nw+1)
/// padded_tensor is results from fft conv (kw, nx, nx)
pub fn check_tensor_consistency(real_tensor: Tensor<Element>, padded_tensor: Tensor<Element>) {
    let n_x = padded_tensor.shape[1];
    for i in 0..real_tensor.shape[0] {
        for j in 0..real_tensor.shape[1] {
            for k in 0..real_tensor.shape[1] {
                // TODO: test if real_tensor.shape[2] works here
                assert!(
                    real_tensor.data[i * real_tensor.shape[1] * real_tensor.shape[1]
                        + j * real_tensor.shape[1]
                        + k]
                        == padded_tensor.data[i * n_x * n_x + j * n_x + k],
                    "Error in tensor consistency"
                );
            }
        }
    }
}

/// Returns an n-th root of unity by starting with a 32nd root of unity and squaring it (32-n) times.
/// Each squaring operation halves the order of the root of unity:
///   - For n=16: squares it 16 times (32-16) to get a 16th root of unity
///   - For n=8:  squares it 24 times (32-8) to get an 8th root of unity
///   - For n=4:  squares it 28 times (32-4) to get a 4th root of unity
///
/// The initial ROOT_OF_UNITY constant is verified to be a 32nd root of unity in the field implementation.
pub fn get_root_of_unity<E: ExtensionField>(n: usize) -> E {
    let mut rou = E::from_bases(&[
        E::BaseField::two_adic_generator(Goldilocks::TWO_ADICITY),
        E::BaseField::ZERO,
    ]);

    for _ in 0..(32 - n) {
        rou = rou * rou;
    }

    rou
}
/// Properly pad a filter
/// We use this function so that filter is amenable to FFT based conv2d
/// Usually vec and n are powers of 2
/// Output: [[F[0][0],…,F[0][n_w],0,…,0],[F[1][0],…,F[1][n_w],0,…,0],…]
pub fn index_w<E: ExtensionField>(
    w: &[Element],
    n_real: usize,
    n: usize,
    output_len: usize,
) -> impl ParallelIterator<Item = E> + use<'_, E> {
    (0..output_len).into_par_iter().map(move |idx| {
        let i = idx / n;
        let j = idx % n;
        if i < n_real && j < n_real {
            w[i * n_real + j].to_field()
        } else {
            E::ZERO
        }
    })
}
// let u = [u[1],...,u[n*n]]
// output vec = [u[n*n-1],u[n*n-2],...,u[n*n-n],....,u[0]]
// Note that y_eval =  f_vec(r) = f_u(1-r)
pub fn index_u<E: ExtensionField>(u: &[E], n: usize) -> impl Iterator<Item = E> + use<'_, E> {
    let len = n * n;
    (0..u.len() / 2).map(move |i| u[len - 1 - i])
}
/// flag: false -> FFT
/// flag: true -> iFFT
pub fn fft<E: ExtensionField + Send + Sync>(v: &mut Vec<E>, flag: bool) {
    let n = v.len();
    let logn = ark_std::log2(n);
    let mut rev: Vec<usize> = vec![0; n];
    let mut w: Vec<E> = vec![E::ZERO; n];

    rev[0] = 0;

    for i in 1..n {
        rev[i] = rev[i >> 1] >> 1 | ((i) & 1) << (logn - 1);
    }
    w[0] = E::ONE;

    let rou: E = get_root_of_unity(logn as usize);
    w[1] = rou;

    if flag {
        w[1] = w[1].inverse();
    }

    for i in 2..n {
        w[i] = w[i - 1] * w[1];
    }

    // Collect indices that need to be swapped
    let swaps: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .filter_map(|i| if rev[i] < i { Some((i, rev[i])) } else { None })
        .collect();

    // Perform swaps sequentially
    for (i, j) in swaps {
        v.swap(i, j);
    }

    let mut i: usize = 2;
    while i <= n {
        // Parallelize the FFT butterfly operations
        v.par_chunks_mut(i).for_each(|chunk| {
            let half_i = i >> 1;
            for k in 0..half_i {
                let u = chunk[k];
                let l = chunk[k + half_i] * w[n / i * k];
                chunk[k] = u + l;
                chunk[k + half_i] = u - l;
            }
        });
        i <<= 1;
    }

    if flag {
        let mut ilen = E::from_canonical_u64(n as u64);
        ilen = ilen.inverse();
        debug_assert_eq!(
            ilen * E::from_canonical_u64(n as u64),
            E::ONE,
            "Error in inv"
        );
        v.par_iter_mut().for_each(|val| {
            *val *= ilen;
        });
    }
}

#[derive(Debug, Default, Clone)]
pub struct ConvData<E>
where
    E: Clone + ExtensionField,
{
    // real_input: For debugging purposes
    pub real_input: Vec<E>, // Actual data before applying FFT and it is already padded with zeros.
    pub input: Vec<Vec<E>>, // This is the result of applying index_x to real_input
    pub input_fft: Vec<Vec<E>>, // FFT(input)
    pub prod: Vec<Vec<E>>,  // FFT(input) * FFT(weights)
    pub output: Vec<Vec<E>>, // iFFT(FFT(input) * FFT(weights)) ==> conv
    pub output_as_element: Vec<Element>, // output as element
}

impl<E> ConvData<E>
where
    E: Copy + ExtensionField,
{
    pub fn new(
        real_input: Vec<E>,
        input: Vec<Vec<E>>,
        input_fft: Vec<Vec<E>>,
        prod: Vec<Vec<E>>,
        output: Vec<Vec<E>>,
        n_x: usize,
    ) -> Self {
        let output_elems = output
            .iter()
            .flat_map(|e| {
                index_u(e.as_slice(), n_x)
                    .map(|e| e.to_element())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self {
            real_input,
            input,
            input_fft,
            prod,
            output,
            output_as_element: output_elems,
        }
    }
    pub fn set_output(&mut self, output: &[Element]) {
        self.output_as_element = output.to_vec();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Shape,
    og_shape: Shape,
}

impl<T> AsRef<Tensor<T>> for Tensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self
    }
}

impl Tensor<Element> {
    /// Returns the maximum size in bits possible if this tensor is treated as a matrix inside
    /// a matrix vector/matrix multiplication. It requires the optional inputs to specify the range
    // of the quantized values in `self` and in the other matrix being multiplied with `self`
    pub fn matmul_output_bitsize(
        &self,
        quantized_self_input_range: Option<usize>,
        quantized_other_input_range: Option<usize>,
    ) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.shape
            .matmul_output_bitsize(quantized_self_input_range, quantized_other_input_range)
    }

    pub fn dequantize(&self, s: &ScalingFactor) -> Tensor<f32> {
        let data = self
            .data
            .iter()
            .map(|e| s.dequantize(e))
            .collect::<Vec<_>>();
        Tensor::new(self.shape.clone(), data)
    }

    pub fn into_fft_conv(self, input_shape: &Shape) -> Self {
        let shape = self.shape;
        let data = self.data;
        assert!(
            shape.product() == data.len(),
            "Shape does not match data length."
        );
        assert!(shape.len() == 4, "Shape does not match data length.");
        assert!(
            shape[2].is_power_of_two(),
            "Filter dimension is not power of two"
        );
        let real_shape = shape.clone();
        let n_w = (input_shape[1] - shape[2] + 1).next_power_of_two();
        Self {
            data, // Note that field elements are back into Element
            shape: vec![shape[0], shape[1], n_w, n_w].into(), /* nw is the padded version of the input */
            og_shape: real_shape,
        }
    }
    // Create specifically a new convolution. The input shape is needed to compute the
    // output and properly arrange the weights
    #[deprecated]
    pub fn new_conv(shape: Shape, input_shape: Shape, data: Vec<Element>) -> Self {
        Tensor::new(shape, data).into_fft_conv(&input_shape)
    }
    /// Recall that weights are not plain text to the "snark". Rather it is FFT(weights).
    /// Aka there is no need to compute the FFT(input) "in-circuit".
    /// It is okay to assume the inputs to the prover is already the FFT version and the prover can commit to the FFT values.
    /// This function computes iFFT of the weights so that we can compute the scaling factors used.
    pub fn get_real_weights<F: ExtensionField>(&self) -> Vec<Vec<Vec<Element>>> {
        let mut real_weights =
            vec![vec![vec![0 as Element; self.nw() * self.nw()]; self.kx()]; self.kw()];

        let mut ctr = 0;
        for i in 0..self.kw() {
            for j in 0..self.kx() {
                for k in 0..(self.real_nw() * self.real_nw()) {
                    real_weights[i][j][k] = self.data[ctr];
                    ctr += 1;
                }
            }
        }
        real_weights
    }

    /// Convolution algorithm using FFTs.
    /// When invoking this algorithm the prover generates all witness/intermediate evaluations
    /// needed to generate a convolution proof
    pub fn fft_conv<F: ExtensionField>(
        &self,
        x: &Tensor<Element>,
    ) -> (Tensor<Element>, ConvData<F>) {
        // input to field elements
        let n_x = x.shape[1].next_power_of_two();
        let real_input = x.data.par_iter().map(|e| e.to_field()).collect::<Vec<_>>();
        let new_n = 2 * n_x * n_x;

        let (x_vec, input): (Vec<Vec<F>>, Vec<Vec<F>>) = real_input
            .par_chunks(n_x * n_x)
            .map(|chunk| {
                let xx_input = chunk.iter().cloned().rev().collect::<Vec<_>>();
                let mut xx_fft = xx_input
                    .iter()
                    .cloned()
                    .chain(std::iter::repeat(F::ZERO))
                    .take(new_n)
                    .collect::<Vec<_>>();
                fft(&mut xx_fft, false);
                (xx_fft, xx_input)
            })
            .unzip();
        // let dim1 = x_vec.len();
        // let dim2 = x_vec[0].len();

        let mut out = vec![vec![F::ZERO; 2 * self.nw() * self.nw()]; self.kw()];

        for i in 0..self.kw() {
            for j in 0..self.kx() {
                let range = (i * self.kx() * self.real_nw() * self.real_nw()
                    + j * self.real_nw() * self.real_nw())
                    ..(i * self.kx() * self.real_nw() * self.real_nw()
                        + (j + 1) * self.real_nw() * self.real_nw());
                let mut w_fft_temp = index_w(
                    &self.data[range],
                    self.real_nw(),
                    self.nw(),
                    2 * self.nw() * self.nw(),
                )
                .collect::<Vec<F>>();
                fft(&mut w_fft_temp, false);
                for k in 0..out[i].len() {
                    out[i][k] += x_vec[j][k] * w_fft_temp[k];
                }
            }
        }
        let prod = out.clone();
        for elt in out.iter_mut() {
            fft(elt, true);
        }

        // TODO: remove the requirement to keep the output value intact
        let output = out;
        let conv_data = ConvData::new(real_input, input, x_vec, prod, output, n_x);
        (
            Tensor::new(
                vec![self.shape[0], n_x, n_x].into(),
                conv_data.output_as_element.clone(),
            ),
            conv_data,
        )
    }

    /// Convolution algorithm using FFTs.
    /// When invoking this algorithm the prover generates all witness/intermediate evaluations
    /// needed to generate a convolution proof
    pub fn fft_conv_old<F: ExtensionField>(
        &self,
        x: &Tensor<Element>,
    ) -> (Tensor<Element>, ConvData<F>) {
        // input to field elements
        let n_x = x.shape[1].next_power_of_two();
        let real_input = x.data.par_iter().map(|e| e.to_field()).collect::<Vec<_>>();
        let w_fft: Vec<F> = self
            .data
            .par_iter()
            .map(|e| e.to_field())
            .collect::<Vec<_>>();
        let new_n = 2 * n_x * n_x;
        let (x_vec, input): (Vec<Vec<F>>, Vec<Vec<F>>) = real_input
            .par_iter()
            .chunks(n_x * n_x)
            .map(|chunk| {
                let xx_input = chunk.into_iter().cloned().rev().collect::<Vec<_>>();
                let mut xx_fft = xx_input
                    .iter()
                    .cloned()
                    .chain(std::iter::repeat(F::ZERO))
                    .take(new_n)
                    .collect::<Vec<_>>();
                fft(&mut xx_fft, false);
                (xx_fft, xx_input)
            })
            .unzip();
        let dim1 = x_vec.len();
        let dim2 = x_vec[0].len();
        let (out, prod): (Vec<_>, Vec<_>) = (0..self.shape[0])
            .into_par_iter()
            .map(|i| {
                let mut outi = (0..dim2)
                    .map(|k| {
                        (0..dim1)
                            .map(|j| x_vec[j][k] * w_fft[i * new_n * x_vec.len() + j * new_n + k])
                            .sum::<F>()
                    })
                    .collect::<Vec<_>>();
                // TODO: remove requirement to keep the product value intact
                let prodi = outi.clone();
                fft(&mut outi, true);
                (outi, prodi)
            })
            .unzip();
        // TODO: remove the requirement to keep the output value intact
        let output = out.clone();
        let conv_data = ConvData::new(real_input, input, x_vec, prod, output, n_x);
        (
            Tensor::new(
                vec![self.shape[0], n_x, n_x].into(),
                conv_data.output_as_element.clone(),
            ),
            conv_data,
        )
    }

    /// Returns the evaluation point, in order for (row,col) addressing
    pub fn evals_2d<F: ExtensionField>(&self) -> Vec<F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.evals_flat()
    }

    pub fn evals_flat<F: ExtensionField>(&self) -> Vec<F> {
        self.data.par_iter().map(|e| e.to_field()).collect()
    }

    pub fn to_2d_mle<F: ExtensionField>(&self) -> DenseMultilinearExtension<F> {
        Tensor::<F>::from(self).to_mle_2d()
    }

    pub fn to_mle_flat<F: ExtensionField>(&self) -> DenseMultilinearExtension<F> {
        DenseMultilinearExtension::from_evaluations_ext_vec(
            self.data.len().ilog2() as usize,
            self.evals_flat(),
        )
    }
}

impl<F: ExtensionField> From<&Tensor<Element>> for Tensor<F> {
    fn from(value: &Tensor<Element>) -> Self {
        Self {
            data: value.evals_flat(),
            shape: value.shape.clone(),
            og_shape: value.og_shape.clone(),
        }
    }
}

impl<F: ExtensionField> Tensor<F> {
    pub fn to_mle_2d(&self) -> DenseMultilinearExtension<F> {
        tensor_to_mle_2d(self, self.data.clone())
    }
}

fn tensor_to_mle_2d<T, F: ExtensionField>(
    tensor: &Tensor<T>,
    evals: Vec<F>,
) -> DenseMultilinearExtension<F> {
    assert!(tensor.is_matrix(), "Tensor is not a matrix");
    assert!(
        tensor.nrows_2d().is_power_of_two(),
        "number of rows {} is not a power of two",
        tensor.nrows_2d()
    );
    assert!(
        tensor.ncols_2d().is_power_of_two(),
        "number of columns {} is not a power of two",
        tensor.ncols_2d()
    );
    // N variable to address 2^N rows and M variables to address 2^M columns
    let num_vars = tensor.nrows_2d().ilog2() + tensor.ncols_2d().ilog2();
    DenseMultilinearExtension::from_evaluations_ext_vec(num_vars as usize, evals)
}

impl Tensor<f32> {
    pub fn quantize(self, s: &ScalingFactor) -> Tensor<Element> {
        let data = self
            .data
            .into_par_iter()
            .map(|x| s.quantize(&x))
            .collect::<Vec<_>>();
        Tensor::new(self.shape, data)
    }
}

impl<T> Tensor<T> {
    /// Create a new tensor with given shape and data
    pub fn new(shape: Shape, data: Vec<T>) -> Self {
        assert!(
            shape.product() == data.len(),
            "Shape does not match data length: shape {:?}->{} vs data.len() {}",
            shape,
            shape.product(),
            data.len()
        );
        Self {
            data,
            shape,
            og_shape: vec![0].into(),
        }
    }

    /// Is an empty tensor
    pub fn is_empty(&self) -> bool {
        self.shape.len() == 0
    }

    /// Create a new tensor with default values
    pub fn new_from_shape(shape: Shape) -> Self
    where
        T: Clone + Default,
    {
        let num_elements = shape.product();
        Self::new(shape, vec![T::default(); num_elements])
    }

    /// Is vector
    pub fn is_vector(&self) -> bool {
        self.get_shape().len() == 1 || (self.get_shape().len() == 2 && self.get_shape()[0] == 1)
    }
    /// Is matrix
    pub fn is_matrix(&self) -> bool {
        self.get_shape().len() == 2
    }

    pub fn is_convolution(&self) -> bool {
        self.get_shape().len() == 4
    }

    /// Get the number of rows from the matrix
    pub fn nrows_2d(&self) -> usize {
        let mut cols = 0;
        let dims = self.get_shape();
        if self.is_matrix() {
            cols = dims[0];
        } else if self.is_convolution() {
            cols = dims[0] * dims[2] * dims[2];
        }
        assert!(cols != 0, "Tensor is not a matrix or convolution");
        cols
    }

    /// Get the number of cols from the matrix
    pub fn ncols_2d(&self) -> usize {
        let mut cols = 0;
        let dims = self.get_shape();
        if self.is_matrix() {
            cols = dims[1];
        } else if self.is_convolution() {
            cols = dims[1] * dims[2] * dims[2];
        }
        assert!(cols != 0, "Tensor is not a matrix or convolution");
        // assert!(self.is_matrix(), "Tensor is not a matrix");
        // let dims = self.dims();

        cols
    }

    /// Returns the number of boolean variables needed to address any row, and any columns
    pub fn num_vars_2d(&self) -> (usize, usize) {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        (
            self.nrows_2d().ilog2() as usize,
            self.ncols_2d().ilog2() as usize,
        )
    }

    /// Get the dimensions of the tensor
    pub fn get_shape(&self) -> Shape {
        assert!(!self.is_empty(), "Empty tensor");
        self.shape.clone()
    }

    /// Returns the number of dimensions the [`Tensor`] has
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the input shape of the tensor
    /// TODO: Remove it
    pub fn get_input_shape(&self) -> Shape {
        assert!(!self.is_empty(), "Empty tensor");
        self.og_shape.clone()
    }
    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    pub fn get_data_into(self) -> Vec<T> {
        self.data
    }
    pub fn kx(&self) -> usize {
        self.shape[1]
    }
    pub fn kw(&self) -> usize {
        self.shape[0]
    }
    pub fn nw(&self) -> usize {
        self.shape[2]
    }
    pub fn real_nw(&self) -> usize {
        self.og_shape[2]
    }
    pub fn real_shape(&self) -> Shape {
        self.og_shape.clone()
    }
    // Returns the size of an individual filter
    pub fn filter_size(&self) -> usize {
        self.shape[2] * self.shape[2]
    }

    pub fn get_conv_weights<F: ExtensionField>(&self) -> Vec<F>
    where
        T: Fieldizer<F>,
    {
        let mut data = vec![F::ZERO; self.data.len()];
        for i in 0..data.len() {
            data[i] = self.data[i].to_field();
        }
        data
    }
}

impl<T> Tensor<T>
where
    T: Clone,
{
    pub fn flatten(&self) -> Self {
        let new_data = self.get_data().to_vec();
        let new_shape = vec![new_data.len()];
        Self {
            data: new_data,
            shape: new_shape.into(),
            og_shape: vec![0].into(),
        }
    }
    pub fn matrix_from_coeffs(data: Vec<Vec<T>>) -> anyhow::Result<Self> {
        let n_rows = data.len();
        let n_cols = data.first().expect("at least one row in a matrix").len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();
        if data.len() != n_rows * n_cols {
            bail!(
                "Number of rows and columns do not match with the total number of values in the Vec<Vec<>>"
            );
        };
        let shape = vec![n_rows, n_cols].into();
        Ok(Self {
            data,
            shape,
            og_shape: vec![0].into(),
        })
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn row_to_boolean_2d<F: ExtensionField>(&self, row: usize) -> impl Iterator<Item = F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        let (nvars_rows, _) = self.num_vars_2d();
        to_bit_sequence_le(row, nvars_rows).map(|b| F::from_canonical_u64(b as u64))
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn col_to_boolean_2d<F: ExtensionField>(&self, col: usize) -> impl Iterator<Item = F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        let (_, nvars_col) = self.num_vars_2d();
        to_bit_sequence_le(col, nvars_col).map(|b| F::from_canonical_u64(b as u64))
    }
    /// From a given row and a given column, return the vector of field elements in the right
    /// format to evaluate the MLE.
    /// little endian so we need to read cols before rows
    pub fn position_to_boolean_2d<F: ExtensionField>(&self, row: usize, col: usize) -> Vec<F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.col_to_boolean_2d(col)
            .chain(self.row_to_boolean_2d(row))
            .collect_vec()
    }

    pub(crate) fn reshape_in_place(&mut self, new_shape: Shape) {
        assert!(
            self.shape.product() == new_shape.product(),
            "Shape mismatch for reshape",
        );
        self.shape = new_shape;
    }

    pub fn reshape(mut self, new_shape: Shape) -> Tensor<T> {
        self.reshape_in_place(new_shape);
        self
    }
}

impl<T> Tensor<T>
where
    T: Number,
{
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .fold((0, T::MIN), |acc, x| match acc.1.compare(x.1) {
                Ordering::Less => (x.0, *x.1),
                _ => acc,
            })
            .0
    }

    pub fn max_abs_output(&self) -> T {
        self.data
            .iter()
            .fold(T::default(), |max, x| max.cmp_max(&x.absolute_value()))
    }
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Shape) -> Self {
        let size = shape.product();
        Self {
            // data: vec![T::zero(); size],
            data: vec![Default::default(); size],
            shape,
            og_shape: vec![0].into(),
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(
            self.shape.product() == other.shape.product(),
            "Shape mismatch for addition {:?} != {:?}",
            self.shape,
            other.shape
        );
        let mut data = vec![Default::default(); self.data.len()];
        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = self.data[i] + other.data[i];
        });

        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0].into(),
            data,
        }
    }
    /// Add a vector to each sub-tensor of the second dimension of the tensor
    /// If self is 2d, then add a vector to each row of self.
    pub fn add_dim2(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape.len() == 2, "Tensor is not a matrix");
        assert!(other.shape.len() == 1, "Tensor is not a vector");
        assert!(
            self.shape[1] == other.shape[0],
            "Shape mismatch for addition2: {:?} != {:?}",
            self.shape,
            other.shape
        );
        let data = self
            .data
            .par_chunks(self.shape[1])
            .flat_map_iter(|chunk| chunk.iter().zip(other.data.iter()).map(|(a, b)| *a + *b))
            .collect::<Vec<_>>();
        Tensor {
            shape: self.shape.clone(),
            og_shape: self.og_shape.clone(),
            data,
        }
    }
    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == other.shape, "Shape mismatch for subtraction.");
        let mut data = vec![Default::default(); self.data.len()];
        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = self.data[i] - other.data[i];
        });

        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0].into(),
            data,
        }
    }
    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(
            self.shape.numel() == other.shape.numel(),
            "Shape mismatch for multiplication: {:?} != {:?}",
            self.shape,
            other.shape
        );
        let data = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<_>>();

        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0].into(),
            data,
        }
    }
    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &T) -> Tensor<T> {
        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0].into(),
            data: self.data.par_iter().map(|x| *x * *scalar).collect(),
        }
    }
    pub fn scalar_mul_f32<N2: Number>(&self, scalar: N2) -> Tensor<T> {
        let scaled = self
            .data
            .par_iter()
            .map(|x| T::from_f32(x.to_f32()? * scalar.to_f32()?))
            .collect::<anyhow::Result<Vec<_>>>()
            .expect("Failed to scale tensor");
        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0].into(),
            data: scaled,
        }
    }
    pub fn pad_1d(mut self, new_len: usize) -> Self {
        assert!(
            self.shape.len() == 1,
            "pad_1d only works for 1d tensors, e.g. vectors"
        );
        self.data.resize(new_len, Default::default());
        self.shape[0] = new_len;
        self
    }

    /// Recursively pads the tensor so its ready to be viewed as an MLE
    pub fn pad_next_power_of_two(&self) -> Self {
        self.generic_pad_next_power_of_two(T::default())
    }

    /// Recursively pads the tensor so its ready to be viewed as an MLE, the `padding_value` is the element that is used to pad
    pub fn generic_pad_next_power_of_two(&self, padding_value: T) -> Self {
        let shape = self.get_shape();

        if shape.iter().all(|dim| dim.is_power_of_two()) {
            return self.clone();
        }

        let padded_data = Self::recursive_pad(self.get_data(), &shape, padding_value);

        let padded_shape = shape
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<usize>>();

        Tensor::<T>::new(padded_shape.into(), padded_data)
    }
    fn recursive_pad(data: &[T], remaining_dims: &[usize], padding_value: T) -> Vec<T> {
        match remaining_dims.len() {
            // If the remaining dims show we are a vector simply pad
            1 => data
                .iter()
                .cloned()
                .chain(std::iter::repeat(padding_value))
                .take(remaining_dims[0].next_power_of_two())
                .collect::<Vec<T>>(),
            // Otherwise we recurse
            _ => {
                let chunk_size = remaining_dims[1..].iter().product::<usize>();
                let mut unpadded_data = data
                    .par_chunks(chunk_size)
                    .map(|data_chunk| {
                        Self::recursive_pad(data_chunk, &remaining_dims[1..], padding_value)
                    })
                    .collect::<Vec<Vec<T>>>();
                let elem_size = unpadded_data[0].len();
                unpadded_data.resize(
                    remaining_dims[0].next_power_of_two(),
                    vec![padding_value; elem_size],
                );
                unpadded_data.concat()
            }
        }
    }

    /// Changes the shape of the current [Tensor] to `target_shape`.
    ///
    /// This method will modify the current tensor in place, extending it
    /// to comply with the new shape.
    ///
    /// # Panics
    ///
    /// If the `target_shape` differs in rank or has a dimension smaller than
    /// the current tensor.
    pub fn pad_to_shape(&mut self, target_shape: Shape) {
        assert!(
            target_shape.rank() == self.shape.rank(),
            "Target shape must have the same rank as the current tensor."
        );

        let distance = self
            .shape
            .iter()
            .zip(target_shape.iter())
            .map(|(original, new)| new.checked_sub(*original))
            .collect::<Option<Vec<usize>>>();

        assert!(
            distance.is_some(),
            "All dimensions of target shape must be greater-than-or-equal to the current tensor",
        );
        let distance = distance.unwrap();

        // First expand the underlying storage vector to the new size
        self.data.resize(target_shape.product(), T::default());

        let original_shape = &self.shape;
        let mut coord = original_shape.0.iter().map(|v| *v - 1).collect::<Vec<_>>();
        let strides = target_shape.strides();

        // Target contains the element's new position after re-shapping.
        let mut target = coord
            .iter()
            .zip(strides.iter())
            .map(|(pos, stride)| *pos * *stride)
            .sum();

        // Difference in size for a given dimension, i.e. how many empty spaces
        // are in between the dimensions after re-shaping.
        let distance = distance
            .iter()
            .zip(strides.iter())
            .map(|(distance, new)| distance * new)
            .collect::<Vec<_>>();

        // Then move the data to its new position. Data is moved from back to the front to
        // prevent overwriting.
        let mut original = original_shape.product();
        loop {
            original -= 1;
            self.data.swap(original, target);

            if original == 0 {
                break;
            }

            for (pos, el) in coord.iter_mut().enumerate().rev() {
                if *el == 0 {
                    *el = original_shape[pos] - 1;
                    target -= distance[pos];
                } else {
                    *el -= 1;
                    target -= 1;
                    break;
                }
            }
        }

        self.shape = target_shape;
    }

    /// Perform matrix-matrix multiplication
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(
            self.is_matrix() && other.is_matrix(),
            "Both tensors must be 2D for matrix multiplication."
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let (n2, p) = (other.shape[0], other.shape[1]);
        assert!(
            n == n2,
            "Matrix multiplication shape mismatch: {:?} cannot be multiplied with {:?}",
            self.shape,
            other.shape
        );

        let mut result = Tensor::zeros(vec![m, p].into());

        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, res)| {
                let i = index / p;
                let j = index % p;

                *res = (0..n)
                    .map(|k| self.data[i * n + k] * other.data[k * p + j])
                    .sum::<T>();
            });

        result
    }
    /// Perform matrix-vector multiplication
    /// TODO: actually getting the result should be done via proper tensor-like libraries
    pub fn matvec(&self, vector: &Tensor<T>) -> Tensor<T> {
        assert!(self.is_matrix(), "First argument must be a matrix.");
        assert!(vector.is_vector(), "Second argument must be a vector.");

        let (m, n) = (self.shape[0], self.shape[1]);
        let vec_len = vector.shape[0];

        assert_eq!(n, vec_len, "Matrix columns must match vector size.");

        let mut result = Tensor::zeros(vec![m].into());

        result.data.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = (0..n)
                .map(|j| self.data[i * n + j] * vector.data[j])
                .sum::<T>();
        });

        result
    }
    pub fn conv_prod(&self, x: &[Vec<T>], w: &[Vec<T>], ii: usize, jj: usize) -> T {
        w.par_iter()
            .enumerate()
            .map(|(i, w_row)| {
                w_row
                    .par_iter()
                    .enumerate()
                    .map(|(j, &w_val)| w_val * x[i + ii][j + jj])
                    .sum()
            })
            .sum()
    }
    pub fn single_naive_conv(&self, w: &[Vec<T>], x: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut out: Vec<Vec<T>> =
            vec![vec![Default::default(); x[0].len() - w[0].len() + 1]; x.len() - w.len() + 1];
        out.par_iter_mut().enumerate().for_each(|(i, out_row)| {
            out_row.par_iter_mut().enumerate().for_each(|(j, out_val)| {
                *out_val = self.conv_prod(x, w, i, j);
            });
        });
        out
    }
    pub fn add_matrix(&self, m1: &mut [Vec<T>], m2: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let mut m = vec![vec![Default::default(); m1[0].len()]; m1.len()];
        m.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, val)| {
                *val = m1[i][j] + m2[i][j];
            });
        });
        m
    }
    // Implementation of the standard convolution algorithm.
    // This is needed mostly for debugging purposes
    pub fn cnn_naive_convolution(&self, xt: &Tensor<T>) -> Tensor<T> {
        let k_w = self.shape[0];
        let k_x = self.shape[1];
        let n_w = self.shape[2];
        let n = xt.shape[0];
        let mut ctr = 0;
        assert!(n == k_x, "Inconsistency on filter/input vector");

        let mut w: Vec<Vec<Vec<Vec<T>>>> =
            vec![vec![vec![vec![Default::default(); n_w]; n_w]; k_x]; k_w];
        let mut x: Vec<Vec<Vec<T>>> =
            vec![vec![vec![Default::default(); xt.shape[1]]; xt.shape[1]]; n];
        for k in 0..k_w {
            for l in 0..k_x {
                for i in 0..n_w {
                    for j in 0..n_w {
                        w[k][l][i][j] = self.data[ctr];
                        ctr += 1;
                    }
                }
            }
        }
        ctr = 0;
        for k in 0..n {
            for i in 0..xt.shape[1] {
                for j in 0..xt.shape[1] {
                    x[k][i][j] = xt.data[ctr];
                    ctr += 1;
                }
            }
        }
        let mut conv: Vec<Vec<Vec<T>>> =
            vec![vec![vec![Default::default(); xt.shape[1] - n_w + 1]; xt.shape[1] - n_w + 1]; k_w];

        for i in 0..k_w {
            for j in 0..k_x {
                let temp = self.single_naive_conv(&w[i][j], &x[j]);
                conv[i] = self.add_matrix(&mut conv[i], temp);
            }
        }

        Tensor::new(
            vec![k_w, xt.shape[1] - n_w + 1, xt.shape[1] - n_w + 1].into(),
            conv.into_iter()
                .flat_map(|inner_vec| inner_vec.into_iter())
                .flat_map(|inner_inner_vec| inner_inner_vec.into_iter())
                .collect(),
        )
    }
    /// Transpose the matrix (2D tensor)
    pub fn transpose(&self) -> Tensor<T> {
        assert!(self.is_matrix(), "Tensor is not a matrix.");
        let (m, n) = (self.shape[0], self.shape[1]);

        let mut result = Tensor::zeros(vec![n, m].into());
        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                let i = idx % m; // Row in the result matrix
                let j = idx / m; // Column in the result matrix
                *val = self.data[i * n + j];
            });

        result
    }
    /// Concatenate a matrix (2D tensor) with a vector (1D tensor) as columns
    pub fn concat_matvec_col(&self, vector: &Tensor<T>) -> Tensor<T> {
        assert!(self.is_matrix(), "First tensor is not a matrix.");
        assert!(vector.is_vector(), "Second tensor is not a vector.");

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let vector_len = vector.shape[0];

        assert!(
            rows == vector_len,
            "Matrix row count must match vector length."
        );

        let new_cols = cols + 1;
        let mut result = Tensor::zeros(vec![rows, new_cols].into());

        result
            .data
            .par_chunks_mut(new_cols)
            .enumerate()
            .for_each(|(i, row)| {
                row[..cols].copy_from_slice(&self.data[i * cols..(i + 1) * cols]); // Copy matrix row
                row[cols] = vector.data[i]; // Append vector element as the last column
            });

        result
    }
    /// Reshapes the matrix to have at least the specified dimensions while preserving all data.
    pub fn reshape_to_fit_inplace_2d(&mut self, new_shape: Shape) {
        let old_rows = self.nrows_2d();
        let old_cols = self.ncols_2d();

        assert!(new_shape.len() == 2, "Tensor is not matrix");
        let new_rows = new_shape[0];
        let new_cols = new_shape[1];
        // Ensure we never lose information by requiring the new dimensions to be at least
        // as large as the original ones
        assert!(
            new_rows >= old_rows,
            "Cannot shrink matrix rows from {old_rows} to {new_rows} - would lose information"
        );
        assert!(
            new_cols >= old_cols,
            "Cannot shrink matrix columns from {old_cols} to {new_cols} - would lose information"
        );

        let new_data: Vec<T> = (0..new_rows * new_cols)
            .into_par_iter()
            .map(|idx| {
                let i = idx / new_cols;
                let j = idx % new_cols;
                if i < old_rows && j < old_cols {
                    self.data[i * old_cols + j]
                } else {
                    T::default() // Zero or default for padding
                }
            })
            .collect();

        *self = Tensor::new(new_shape, new_data);
    }
    pub fn maxpool2d(&self, kernel_size: usize, stride: usize) -> Tensor<T> {
        let dims = self.get_shape().len();
        assert!(dims >= 2, "Input tensor must have at least 2 dimensions.");

        let (h, w) = (self.shape[dims - 2], self.shape[dims - 1]);

        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        // Assumes dilation = 1
        assert!(
            h >= kernel_size,
            "Kernel size ({kernel_size}) is larger than input dimensions ({h}, {w})"
        );
        let out_h = (h - kernel_size) / stride + 1;
        let out_w = (w - kernel_size) / stride + 1;

        let outer_dims: usize = self.shape[..dims - 2].iter().product();
        let output: Vec<T> = (0..outer_dims * out_h * out_w)
            .into_par_iter()
            .map(|flat_idx| {
                let n = flat_idx / (out_h * out_w);
                let i = (flat_idx / out_w) % out_h;
                let j = flat_idx % out_w;

                let matrix_idx = n * (h * w);
                let src_idx = matrix_idx + (i * stride) * w + (j * stride);
                let mut max_val = self.data[src_idx];

                for ki in 0..kernel_size {
                    for kj in 0..kernel_size {
                        let src_idx = matrix_idx + (i * stride + ki) * w + (j * stride + kj);
                        let value = self.data[src_idx];
                        max_val = max_val.cmp_max(&value);
                    }
                }

                max_val
            })
            .collect();

        let mut new_shape = self.shape.clone();
        new_shape[dims - 2] = out_h;
        new_shape[dims - 1] = out_w;

        Tensor {
            data: output,
            shape: new_shape,
            og_shape: vec![0].into(),
        }
    }

    // Replaces every value of a tensor with the maxpool of its kernel
    pub fn padded_maxpool2d(&self) -> (Tensor<T>, Tensor<T>) {
        let kernel_size = MAXPOOL2D_KERNEL_SIZE;
        let stride = MAXPOOL2D_KERNEL_SIZE;

        let maxpool_result = self.maxpool2d(kernel_size, stride);

        let dims: usize = self.get_shape().len();
        assert!(dims >= 2, "Input tensor must have at least 2 dimensions.");

        let (h, w) = (self.shape[dims - 2], self.shape[dims - 1]);

        assert!(
            h % MAXPOOL2D_KERNEL_SIZE == 0,
            "Currently works only with kernel size {MAXPOOL2D_KERNEL_SIZE}"
        );
        assert!(
            w % MAXPOOL2D_KERNEL_SIZE == 0,
            "Currently works only with stride size {MAXPOOL2D_KERNEL_SIZE}"
        );

        let outer_dims: usize = self.shape[..dims - 2].iter().product();
        let maxpool_h = (h - kernel_size) / stride + 1;
        let maxpool_w = (w - kernel_size) / stride + 1;

        let padded_maxpool_data: Vec<T> = (0..outer_dims * h * w)
            .into_par_iter()
            .map(|out_idx| {
                let n = out_idx / (h * w);
                let i_full = (out_idx / w) % h;
                let j_full = out_idx % w;

                let i = i_full / stride;
                let j = j_full / stride;

                let maxpool_idx = n * maxpool_h * maxpool_w + i * maxpool_w + j;
                maxpool_result.data[maxpool_idx]
            })
            .collect();

        let padded_maxpool_tensor = Tensor {
            data: padded_maxpool_data,
            shape: self.get_shape(),
            og_shape: vec![0].into(),
        };

        (maxpool_result, padded_maxpool_tensor)
    }

    pub fn conv2d(&self, kernels: &Tensor<T>, bias: &Tensor<T>, stride: usize) -> Tensor<T> {
        let (n_size, c_size, h_size, w_size) = self.get4d();
        let (k_n, k_c, k_h, k_w) = kernels.get4d();

        assert!(
            self.get_shape().len() <= 4,
            "Supports only at most 4D input."
        );
        assert!(
            kernels.get_shape().len() <= 4,
            "Supports only at most 4D filters."
        );
        // Validate shapes
        assert_eq!(
            c_size,
            k_c,
            "Input {c_size} and kernel {k_c} channels must match! {:?} vs kernel {:?}",
            self.get_shape(),
            kernels.get_shape()
        );
        assert_eq!(
            bias.shape.as_ref(),
            &[k_n],
            "Bias shape must match number of kernels!"
        );

        let out_h = (h_size - k_h) / stride + 1;
        let out_w = (w_size - k_w) / stride + 1;
        let out_shape = vec![n_size, k_n, out_h, out_w];

        // Compute output in parallel
        let output: Vec<T> = (0..n_size * k_n * out_h * out_w)
            .into_par_iter()
            .map(|flat_idx| {
                // Decompose flat index into (n, o, oh, ow)
                let n = flat_idx / (k_n * out_h * out_w);
                let rem1 = flat_idx % (k_n * out_h * out_w);
                let o = rem1 / (out_h * out_w);
                let rem2 = rem1 % (out_h * out_w);
                let oh = rem2 / out_w;
                let ow = rem2 % out_w;

                let mut sum = T::default();

                // Convolution
                for c in 0..c_size {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let h = oh * stride + kh;
                            let w = ow * stride + kw;
                            sum += self.get_at_4d(n, c, h, w) * kernels.get_at_4d(o, c, kh, kw);
                        }
                    }
                }

                // Add bias
                sum + bias.data[o]
            })
            .collect();

        Tensor {
            data: output,
            shape: out_shape.into(),
            og_shape: vec![0].into(),
        }
    }

    pub fn to_f32(&self) -> anyhow::Result<Tensor<f32>> {
        Ok(Tensor {
            data: self
                .data
                .iter()
                .map(Number::to_f32)
                .collect::<anyhow::Result<Vec<_>>>()?,
            shape: self.shape.clone(),
            og_shape: self.og_shape.clone(),
        })
    }
    /// Makes a [`Tensor`] that is a batch of lower triangular matrices.
    /// - `matrix_dim` the number specifying the dimensions of each individual matrix (lower triangular amtrix must be square)
    /// - `num_matrices` specifies how many matrices to make
    /// - `diag` specifies the "offset" for the diagonal, an offset of `1` means we keep two `1`s on the first row instead of 1, and offset of `-1` means all `zeroes`
    pub fn tril(matrix_dim: usize, num_matrices: usize, diag: i32) -> Tensor<T> {
        Self::tri(matrix_dim, num_matrices, diag, T::unit(), T::default())
    }

    /// Makes a [`Tensor`] that is a batch of lower triangular matrices.
    /// - `matrix_dim` the number specifying the dimensions of each individual matrix (lower triangular amtrix must be square)
    /// - `num_matrices` specifies how many matrices to make
    /// - `diag` specifies the "offset" for the diagonal, an offset of `1` means we keep two `lower_val`s on the first row instead of 1, and offset of `-1` means all `upper_val`
    /// - `lower_val` specifies the value to fill the lower triangular part with
    /// - `upper_val` specifies the value to fill the upper triangular part with
    pub fn tri(
        matrix_dim: usize,
        num_matrices: usize,
        diag: i32,
        lower_val: T,
        upper_val: T,
    ) -> Tensor<T> {
        // We make one matrix and then just clone it
        let data = (0i32..matrix_dim as i32)
            .flat_map(|i| {
                if (i + diag).is_negative() {
                    vec![upper_val; matrix_dim]
                } else {
                    std::iter::repeat_n(lower_val, (i + diag + 1) as usize)
                        .chain(std::iter::repeat(upper_val))
                        .take(matrix_dim)
                        .collect::<Vec<T>>()
                }
            })
            .cycle()
            .take(num_matrices * matrix_dim * matrix_dim)
            .collect::<Vec<T>>();

        Tensor::<T>::new(vec![num_matrices, matrix_dim, matrix_dim].into(), data)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::iter::Sum,
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    /// Parse the shape as N,C,H,W
    /// if the tensor is 3d, for example the input could be 3d if there is only one batch, then
    /// it returns as if N = 1.
    pub fn get4d(&self) -> (usize, usize, usize, usize) {
        let (n_size, offset) = if self.shape.len() == 3 {
            (1, 0)
        } else {
            (self.shape.first().cloned().unwrap_or(1), 1)
        };
        let c_size = self.shape.get(offset).cloned().unwrap_or(1);
        let h_size = self.shape.get(1 + offset).cloned().unwrap_or(1);
        let w_size = self.shape.get(2 + offset).cloned().unwrap_or(1);

        (n_size, c_size, h_size, w_size)
    }

    /// Retrieves an element using (N, C, H, W) indexing
    pub fn get_at_4d(&self, n: usize, c: usize, h: usize, w: usize) -> T {
        assert!(self.shape.len() <= 4);

        let (n_size, c_size, h_size, w_size) = self.get4d();

        assert!(n < n_size);
        let flat_index = n * (c_size * h_size * w_size) + c * (h_size * w_size) + h * w_size + w;
        self.data[flat_index]
    }

    fn get_idx(&self, accessors: Vec<usize>) -> usize {
        assert!(self.shape.len() == accessors.len());
        let mut flat_index = *accessors.last().unwrap();
        let mut multiplier = *self.shape.last().unwrap();
        for (a, s) in accessors
            .iter()
            .rev()
            .skip(1)
            .zip(self.shape.iter().rev().skip(1))
        {
            assert!(
                *a < *s,
                "Index out of bounds: {a} >= {s} - 0-based indexing forbids"
            );
            flat_index += *a * multiplier;
            multiplier *= *s;
        }
        flat_index
    }

    // 0-based indexing for compatibility with other libraries
    // ex: accessors = [3,2,1] => will retrieve element at index 1 + 2 * shape[0] + 3 * shape[0] * shape[1]
    pub fn get(&self, accessors: Vec<usize>) -> T {
        let flat_index = self.get_idx(accessors);
        self.data[flat_index]
    }
}

impl<T> Tensor<T>
where
    T: Copy + Clone + Send + Sync,
    T: Copy + Default + std::ops::Mul<Output = T> + std::iter::Sum,
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    // Pads a matrix `M` to `M'` so that matrix-vector multiplication with a flattened FFT-padded convolution output `X'`
    /// matches the result of multiplying `M` with the original convolution output `X`.
    ///
    /// The real convolution output `X` has dimensions `(C, H, W)`. However, when using FFT-based convolution,
    /// the output `X'` is padded to dimensions `(C', H', W')`, where `C'`, `H'`, and `W'` are the next power of 2
    /// greater than or equal to `C`, `H`, and `W`, respectively.
    /// Given a matrix `M` designed to multiply with the flattened `X`, this function pads `M` into `M'` such that
    /// `M * X == M' * X'`, ensuring the result remains consistent despite the padding in `X'`.
    pub fn pad_matrix_to_ignore_garbage(
        &self,
        conv_shape_og: &[usize],
        conv_shape_pad: &[usize],
        mat_shp_pad: &Shape,
    ) -> Self {
        assert!(
            conv_shape_og.len() == 3 && conv_shape_pad.len() == 3,
            "Expects conv2d shape output to be 3d: conv_shape_og: {:?}, conv_shape_pad: {:?}",
            conv_shape_og.len(),
            conv_shape_pad.len()
        );
        assert!(
            mat_shp_pad.len() == 2 && self.shape.len() == 2,
            "Expects matrix to be 2d: mat_shp_pad: {:?}, self.shape: {:?}",
            mat_shp_pad.len(),
            self.shape.len()
        );
        let mat_shp_og = self.get_shape();

        let new_data: Vec<T> = (0..mat_shp_pad[0] * mat_shp_pad[1])
            .into_par_iter()
            .map(|new_loc| {
                // Decompose new_loc into (row, channel, h_in, w_in) for the padded output space
                let row = new_loc / mat_shp_pad[1];
                let channel =
                    (new_loc / (conv_shape_pad[1] * conv_shape_pad[2])) % conv_shape_pad[0];
                let h_in = (new_loc / conv_shape_pad[2]) % conv_shape_pad[1];
                let w_in = new_loc % conv_shape_pad[2];

                // Check if this position corresponds to an original data location
                if row < mat_shp_og[0]
                    && channel < conv_shape_og[0]
                    && h_in < conv_shape_og[1]
                    && w_in < conv_shape_og[2]
                {
                    let old_loc = channel * conv_shape_og[1] * conv_shape_og[2]
                        + h_in * conv_shape_og[2]
                        + w_in
                        + row * mat_shp_og[1];
                    self.data[old_loc]
                } else {
                    T::default() // Default value for non-mapped positions
                }
            })
            .collect();

        Tensor::new(mat_shp_pad.to_vec().into(), new_data)
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shape = self.shape.clone();

        while shape.len() < 4 {
            shape.reverse();
            shape.push(1);
            shape.reverse();
        }

        if shape.len() == 4 {
            let (batches, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let channel_size = height * width;
            let batch_size = channels * channel_size;

            for b in 0..batches {
                writeln!(f, "Batch {b} [{channels} channels, {height}x{width}]:")?;
                for c in 0..channels {
                    writeln!(f, "  Channel {c}:")?;
                    let offset = b * batch_size + c * channel_size;
                    for i in 0..height {
                        let row_start = offset + i * width;
                        let row_data: Vec<String> = (0..width)
                            .map(|j| format!("{:>4.2}", self.data[row_start + j]))
                            .collect();
                        writeln!(f, "    {:>3}: [{}]", i, row_data.join(", "))?;
                    }
                }
            }
            write!(f, "Shape: {:?}", self.shape)
        } else {
            write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.data) // Fallback
        }
    }
}

impl PartialEq for Tensor<Element> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl PartialEq for Tensor<GoldilocksExt2> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

pub struct TensorSlice<'a, T> {
    data: &'a [T],
    shape: Shape,
}

impl<'a, T> From<&'a Tensor<T>> for TensorSlice<'a, T> {
    fn from(value: &'a Tensor<T>) -> Self {
        Self {
            data: &value.data,
            shape: value.shape.clone(),
        }
    }
}

impl<'a, T> TensorSlice<'a, T> {
    pub(crate) fn get_shape(&self) -> Shape {
        self.shape.clone()
    }

    pub(crate) fn get_data(&self) -> &[T] {
        self.data
    }

    pub(crate) fn slice_over_first_dim(&self, dim2_start: usize, dim2_end: usize) -> Self {
        let range = dim2_start * self.shape[1]..dim2_end * self.shape[1];
        let data = &self.data[range];
        let mut new_shape = self.shape.clone();
        new_shape[0] = dim2_end - dim2_start;
        Self {
            data,
            shape: new_shape,
        }
    }
}
impl<T: Default + Clone + Copy> Tensor<T> {
    /// Permute a tensor, changing its shape according to the `order` specified as input.
    /// The `i`-th entry in the `order` vector specifies which dimension of the original
    /// tensor should become the `i`-th dimension of the output tensor.
    /// For instance, given an input tensor with shape `[7, 14, 23]` and `order = [2, 0, 1]`,
    /// then the shape of the output tensor will be `[23, 7, 14]`
    pub fn permute3d(&self, order: &[usize]) -> Self {
        assert!(self.shape.len() == 3 && order.len() == 3);
        assert!(order.iter().all(|x| *x < 3));
        let (a, b, c) = (self.shape[0], self.shape[1], self.shape[2]);
        let new_a = self.shape[order[0]];
        let new_b = self.shape[order[1]];
        let new_c = self.shape[order[2]];
        let new_shape = vec![new_a, new_b, new_c].into();
        let mut data = vec![T::default(); a * b * c];
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    let old_loc = i * b * c + j * c + k;
                    let pos = [i, j, k];
                    let new_i = pos[order[0]];
                    let new_j = pos[order[1]];
                    let new_k = pos[order[2]];
                    let new_loc = new_i * new_b * new_c + new_j * new_c + new_k;
                    assert!(
                        new_loc < new_a * new_b * new_c,
                        "Failed for {i}, {j}, {k} and {new_i}, {new_j}, {new_k}"
                    );
                    data[new_loc] = self.data[old_loc];
                }
            }
        }
        Self {
            data,
            shape: new_shape,
            og_shape: self.shape.clone(),
        }
    }

    /// Expands a [`Tensor`] to the provided shape. In order for the expansion to be valid we require that
    /// all dimensions in the provided shape are greater than or equal to the original shape. Additionally in all cases where
    /// the provided shape dimension is larger we require the original shape dimension to be 1.
    pub fn expand(&self, expansion_shape: &Shape) -> Result<Tensor<T>>
    where
        T: Clone,
    {
        // Check that the expanded shape has rank greater than or equal to self.shape and also that all dimensions are compatible
        let expansion_rank = expansion_shape.rank();
        if expansion_rank < self.rank() {
            return Err(anyhow!(
                "Cannot Expand to shape: {:?} as it has rank: {expansion_rank} which is smaller than the current rank: {}",
                expansion_shape,
                self.rank()
            ));
        }

        let rank_diff = expansion_rank - self.rank();
        let padded_shape = std::iter::repeat_n(1usize, rank_diff)
            .chain(self.shape.iter().copied())
            .collect::<Vec<usize>>();
        expansion_shape.iter().zip(padded_shape.iter()).try_for_each(|(&new_dim, &dim)|{
            // Check new_dim isn't zero 
            if new_dim == 0 {
                Err(anyhow!("Cannot expand to new shape, new shape dim: {new_dim} was zero"))
            } else if dim != 1 {
                // In this case we need both to be equal
                if dim != new_dim {
                    Err(anyhow!("Cannot expand to new shape, new shape dim: {new_dim} was not equal to current dim: {dim}"))
                } else {
                    Ok(())
                }
            } else {
                Ok(())
            }
        })?;

        // Now that we know everything checks out we expand appropriately
        let new_data = expansion_shape
            .iter()
            .zip(padded_shape.iter())
            .rev()
            .fold(
                (self.data.clone(), 1usize),
                |(data_acc, chunk_size), (&exp_dim, &current_dim)| {
                    if current_dim == exp_dim {
                        // Nothing to do in this case just update chunk_size
                        (data_acc, chunk_size * exp_dim)
                    } else {
                        // We should split the current data_acc into chunk_size pieces and then repeat each chunk exp_dim times
                        (
                            data_acc
                                .chunks(chunk_size)
                                .flat_map(|chunk| vec![chunk; exp_dim].concat())
                                .collect::<Vec<T>>(),
                            chunk_size * exp_dim,
                        )
                    }
                },
            )
            .0;

        // Now that we have the expanded data make the tensor
        Ok(Tensor::<T>::new(expansion_shape.clone(), new_data))
    }
}

impl<T: Number> Tensor<T> {
    pub fn max_value(&self) -> T {
        self.data.iter().fold(T::MIN, |max, x| max.cmp_max(x))
    }
    pub fn min_value(&self) -> T {
        self.data.iter().fold(T::MAX, |min, x| min.cmp_min(x))
    }
    #[cfg(test)]
    pub fn random(shape: &Shape) -> Self {
        Self::random_seed(shape, Some(crate::seed_from_env_or_rng()))
    }

    pub fn try_map<F: Fn(&T) -> anyhow::Result<T>>(&self, f: F) -> anyhow::Result<Self> {
        Ok(Self {
            data: self
                .data
                .iter()
                .map(f)
                .collect::<anyhow::Result<Vec<_>>>()?,
            shape: self.shape.clone(),
            og_shape: self.og_shape.clone(),
        })
    }

    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    #[cfg(test)]
    pub fn random_seed(shape: &Shape, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| crate::seed_from_env_or_rng()); // Use provided seed or default
        let mut rng = <crate::StdRng as ark_std::rand::SeedableRng>::seed_from_u64(seed);
        let size = shape.product();
        let data = (0..size).map(|_| T::random(&mut rng)).collect();
        Self {
            data,
            shape: shape.clone(),
            og_shape: vec![0].into(),
        }
    }

    // slice on the third dimension.
    // start inclusive, end exclusive
    pub fn slice_3d(&self, start: usize, end: usize) -> Self {
        assert!(self.shape.len() == 3);
        assert!(start < self.shape[0]);
        assert!(end <= self.shape[0]);
        let blocks = self.shape[1] * self.shape[2];
        let sliced = self.data[blocks * start..blocks * end].to_vec();
        Self {
            data: sliced,
            shape: vec![end - start, self.shape[1], self.shape[2]].into(),
            og_shape: vec![0].into(),
        }
    }

    // slice the tensor on the second dimension
    // dim2_start inclusive
    // dim2_end exclusive
    // TODO: refactor to take generic shape dimensions where to slice ... or just use burn API tensor
    pub fn slice_2d(&self, dim2_start: usize, dim2_end: usize) -> Self {
        assert!(self.shape.len() == 2);
        let range = dim2_start * self.shape[1]..dim2_end * self.shape[1];
        let data = self.data[range].to_vec();
        let new_shape = vec![dim2_end - dim2_start, self.shape[1]];
        Self {
            data,
            shape: new_shape.into(),
            og_shape: vec![0].into(),
        }
    }
}

impl<T> Tensor<T> {
    /// Returns an iterator that yields slices of the last dimension.
    /// For a tensor of shape [2,3,3], it will yield 6 slices (2*3) of 3 elements each.
    pub fn slice_last_dim(&self) -> impl Iterator<Item = &[T]> {
        let (it, _) = self.slice_on_dim(self.shape.len() - 2);
        it
    }

    /// Returns an iterator of slices whose length corresponds to the subspace
    /// the dimension represents. Note dim is the dimension _index_ (0-based indexing).
    /// Example: if dimension is [2,3,4], and we call `slice_on_dim(1)`,
    /// it will yield 2x3 slices of 4 elements each. If we call `slice_on_dim(0)`,
    /// it will yield 2 slices of 3x4=12 element each.
    /// If dim is the last dimension, it will simply yield a slice of the whole tensor.
    /// The shape returned is the shape of each slice. The shape is the same as the shape of the tensor
    /// if the dim is the last dimension or more
    pub fn slice_on_dim(&self, dim: usize) -> (impl Iterator<Item = &[T]>, Shape) {
        assert!(
            dim < self.shape.len(),
            "can't slice on dim {:?} if shape is {:?}",
            dim,
            self.shape
        );
        let (stride, shape) = if dim < self.shape.len() - 1 {
            let slice = self.shape.slice(dim + 1..);
            (slice.product(), slice)
        } else {
            (self.shape.product(), self.shape.clone())
        };
        (self.data.chunks(stride), shape)
    }

    // Concatenate the other tensor to the first one.
    // RESTRICTIOn: self shape is [a1,a2...,an] we
    // expect other shape to be [a2...,an] OR [1, a2...,an]
    // The new shape of self will be [a1+1,...an]
    // In other words, we only concatenate another vector if it's exactly size of the highest dimension
    // If it's 2d, then we expect other to be a vector
    pub fn concat(&mut self, other: Self) {
        // make sure that the all dimension but the highest one are the same
        let common_shape = self.shape.len().min(other.shape.len());
        let added_higher = if common_shape < self.shape.len() {
            assert!(
                self.shape
                    .iter()
                    .rev()
                    .zip(other.shape.iter().rev())
                    .take(common_shape)
                    .all(|(a, b)| a == b)
            );
            assert_eq!(common_shape + 1, self.shape.len());
            1
        } else {
            assert_eq!(common_shape, self.shape.len());
            *other.shape.first().unwrap()
        };
        // then the new shape has this higher dimension + 1 simply
        // common_shape since 0-based indexing
        *self.shape.get_mut(0).unwrap() += added_higher;
        self.data.extend(other.data);
    }
    /// Stack all the tensors in the iterator into a single tensor using `concat()`
    /// Note this naively increase the highest dimension. If you wish to stack along a new higher dimension,
    /// call `unsqueeze(0)` on the first or all tensors first.
    /// e.g. [2,3] and [2,3] will be stacked into [4,3] naively. Calling `unsqueeze(0)` on both
    /// will stack into [2,2,3].
    pub fn stack_all<I: IntoIterator<Item = Self>>(tensors: I) -> anyhow::Result<Self> {
        let mut it = tensors.into_iter();
        let mut first = it
            .next()
            .ok_or(anyhow::anyhow!("Can't concat an empty list of tensors"))?;
        for tensor in it {
            first.concat(tensor);
        }
        Ok(first)
    }

    pub fn unsqueeze(self, index: usize) -> Self {
        let new_shape = self.shape.insert(index, 1);
        Self {
            data: self.data,
            shape: new_shape,
            og_shape: self.og_shape.clone(),
        }
    }
}

/// Structure that holds a shape of a tensor.
/// NOTE: it is currently being phased in incrementally the codebase currently. There will be places where we still use Vec<usize>
#[derive(
    Debug,
    Clone,
    derive_more::From,
    derive_more::Into,
    derive_more::AsRef,
    derive_more::Index,
    derive_more::IndexMut,
    derive_more::Deref,
    derive_more::DerefMut,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn from_it<V: std::borrow::Borrow<usize>, I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self(iter.into_iter().map(|v| *v.borrow()).collect())
    }

    /// Creates a new [Shape].
    ///
    /// # Panics
    ///
    /// If `shape` is an empty vector.
    pub fn new(shape: Vec<usize>) -> Self {
        assert!(!shape.is_empty(), "Shape can not be empty");
        Self(shape)
    }

    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Shape {
        let len = self.0.len();
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => len,
        };
        Shape(self.0[start..end].to_vec())
    }

    /// Returns the size of a given dimension.
    ///
    /// ```
    /// # use zkml::tensor::Shape;
    /// let shape = Shape::new(vec![3, 5, 7]);
    /// assert_eq!(shape.dim(0), 3);
    /// assert_eq!(shape.dim(1), 5);
    /// assert_eq!(shape.dim(2), 7);
    /// ```
    pub fn dim(&self, index: usize) -> usize {
        self.0[index]
    }

    /// Adds an extra dimension with size `1` to [Shape].
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than this shape size.
    ///
    /// ```
    /// # use zkml::tensor::Shape;
    /// let shape = Shape::new(vec![3, 5]);
    /// let new_shape = shape.unsqueeze(1);
    /// assert_eq!(new_shape.dim(0), 3);
    /// assert_eq!(new_shape.dim(1), 1);
    /// assert_eq!(new_shape.dim(2), 5);
    /// ```
    pub fn unsqueeze(&self, index: usize) -> Self {
        let mut new_shape = self.0.clone();
        new_shape.insert(index, 1);
        Self(new_shape)
    }

    /// Returns the strides for this [Shape] in row major order.
    ///
    /// The values in the stride vector determine the offset
    /// needed to go to the next element of a given dimension.
    ///
    /// ```
    /// # use zkml::tensor::Shape;
    /// let shape = Shape::new(vec![3, 5, 7]);
    /// let strides = shape.strides();
    /// // row major order, inner most dimension changes the quickest
    /// assert_eq!(strides[0], 35);
    /// assert_eq!(strides[1], 7);
    /// assert_eq!(strides[2], 1);
    /// ```
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = self
            .0
            .iter()
            .rev()
            .scan(1usize, |state, item| {
                let el = Some(*state);
                *state *= item;
                el
            })
            .collect::<Vec<_>>();

        strides.reverse();
        strides
    }

    /// Inserts a new dimension at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than this shape size.
    ///
    /// ```
    /// # use zkml::tensor::Shape;
    /// let shape = Shape::new(vec![3, 5]);
    /// let new_shape = shape.insert(1, 1);
    /// assert_eq!(new_shape.dim(0), 3);
    /// assert_eq!(new_shape.dim(1), 1);
    /// assert_eq!(new_shape.dim(2), 5);
    /// ```
    pub fn insert(&self, index: usize, value: usize) -> Self {
        let mut new_shape = self.0.clone();
        new_shape.insert(index, value);
        Self(new_shape)
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        Self(permutation.iter().map(|i| self.0[*i]).collect())
    }
    pub fn next_power_of_two(&self) -> Self {
        Self(self.0.next_power_of_two())
    }
    pub fn concat(&self, other: &Self) -> Self {
        let mut new_shape = self.0.clone();
        new_shape.extend(other.0.clone());
        Self(new_shape)
    }
    pub fn into_vec(self) -> Vec<usize> {
        self.0
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn is_matrix(&self) -> bool {
        self.0.len() == 2
    }
    pub fn ncols(&self) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.0[1]
    }
    pub fn nrows(&self) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.0[0]
    }
    // Compute the bitsize of the output of the matrix multiplication of a tensor with shape `self`
    // with another matrix with a compatible shape. It requires the optional inputs to specify the range
    // of the quantized values in `self` and in the other matrix being multiplied with `self`
    pub fn matmul_output_bitsize(
        &self,
        quantized_self_input_range: Option<usize>,
        quantized_other_input_range: Option<usize>,
    ) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.ncols();
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        quantized_self_input_range
            .map(log2_ceil)
            .unwrap_or(*quantization::BIT_LEN - 1)
            + quantized_other_input_range
                .map(log2_ceil)
                .unwrap_or(*quantization::BIT_LEN - 1)
            + ceil_log2(ncols)
            + 1
    }
    pub fn is_power_of_two(&self) -> bool {
        self.0.iter().all(|x| x.is_power_of_two())
    }
    pub fn product(&self) -> usize {
        self.0.iter().product()
    }
    pub fn numel(&self) -> usize {
        self.product()
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect::<Vec<usize>>())
    }
}

// taken from https://docs.pytorch.org/docs/stable/generated/torch.isclose.html
/// Determines whether two slices of `f32` values are element-wise close within
/// the specified absolute (`atol`) and relative (`rtol`) tolerances.
///
/// The condition checked is the same as PyTorch's `torch.isclose`:
/// `|a - b| <= atol + rtol * |b|` for every corresponding element.
///
/// # Examples
///
/// ```
/// use zkml::tensor::is_close_with_tolerance;
///
/// // For 10% relative tolerance (0.1 = 10%)
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.1, 2.2, 3.3]; // 10% difference
/// assert!(is_close_with_tolerance(&a, &b, 0.0, 0.1));
///
/// // For 1e-6 absolute tolerance
/// let c = [1.0, 2.0, 3.0];
/// let d = [1.000001, 2.000001, 3.000001]; // 1e-6 difference
/// assert!(is_close_with_tolerance(&c, &d, 1e-6, 0.0));
/// ```
pub fn is_close_with_tolerance(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(|(x, y)| {
        let diff = (*x - *y).abs();
        diff <= atol + rtol * y.abs()
    })
}

/// Backwards-compatible wrapper that uses the historical default tolerances
/// (`atol = 1e-8`, `rtol = 1e-5`).
pub fn is_close(a: &[f32], b: &[f32]) -> bool {
    is_close_with_tolerance(a, b, 1e-8_f32, 1e-5_f32)
}
/// Function used to get the broadcasted shape of two [Tensors][`Tensor`].
/// To be able to broadcast two [Tensors][`Tensor`] the shapes must be compatible in each dimension, this means either:
///     1) The two dimensions are equal
///     2) One of the dimensions is 1
/// If one [`Tensor`] has fewer dimensions than the other we can prepend its [`Shape`] with `1`s. For example if we have shapes
/// `[5, 7, 2]` and `[7, 1]` then let us write `[x, y, z]` for the currently unknown broadcasted shape, the broadcasting process works as follows:
///     1) Prepend `1` to the second [`Shape`] so that we have `[5, 7, 2]` and `[1, 7, 1]`
///     2) Compare the two shape arrays from back to front and apply our rules above:
///         i) `2` and `1` are not equal, but one of them is `1` so we set the final dim (`z`) in the broadcasted shape to `2` giving us `[x, y, 2]`
///         ii) Here both dims are `7` so because they are equal the broadcasted shape at `7` will also be `7` giving us `[x, 7, 2]`
///         iii) Finally we have `5` and `1`, since one of the dims is `1` we take the larger of the two giving `x = 5`
/// This gives us a final broadcasted shape of `[5, 7, 2]`.
///
/// As another example if we have a [`Tensor`] `A` with shape `[4, 1]` and values `[1, 2, 3, 4]` and we have another [`Tensor`] `B` with shape `[3]` and values `[10, 11, 12]`
/// then brodcasting them to the same shape results in:
/// ```ignore
///       A:  [1, 1, 1,       B:  [10, 11, 12
///            2, 2, 2,            10, 11, 12
///            3, 3, 3,            10, 11, 12
///            4, 4, 4]            10, 11, 12]
/// ```
pub fn get_broadcasted_shape(shape_a: &Shape, shape_b: &Shape) -> anyhow::Result<Shape> {
    // Compare the length of both inputs and match on the result
    let rank_a = shape_a.len();
    let rank_b = shape_b.len();

    let compatibility = |a: usize, b: usize, index: usize| -> anyhow::Result<usize> {
        match (a, b) {
            // One of a or b is 1 so we return the value that is not 1
            (1, dim) | (dim, 1) => Ok(dim),
            // Both dims are the same so we return that value
            (dim_a, dim_b) if dim_a == dim_b => Ok(dim_a),
            // Any other case returns an error
            _ => Err(anyhow::anyhow!(
                "Cannot broadcast shapes as dimensions (a:{a}, b:{b}) were incompatible at index: {index}"
            )),
        }
    };

    match rank_a.cmp(&rank_b) {
        Ordering::Less => {
            // shape_a is shorter so we work out the difference and prepend with that many 1s
            let diff = rank_b - rank_a;

            let padded_shape_a = std::iter::repeat_n(1usize, diff)
                .chain(shape_a.iter().copied())
                .collect::<Vec<usize>>();

            // Now we iterate over both choosing the maximum each time
            shape_b
                .iter()
                .zip(padded_shape_a.iter())
                .enumerate()
                .map(|(index, (&b_dim, &a_dim))| compatibility(a_dim, b_dim, index))
                .collect::<Result<Vec<usize>, anyhow::Error>>()
                .map(Shape::from)
        }
        Ordering::Equal => {
            // The ranks are equal so we just iterate over both choosing the max each time
            shape_b
                .iter()
                .zip(shape_a.iter())
                .enumerate()
                .map(|(index, (&b_dim, &a_dim))| compatibility(a_dim, b_dim, index))
                .collect::<Result<Vec<usize>, anyhow::Error>>()
                .map(Shape::from)
        }
        Ordering::Greater => {
            // shape_a has larger rank so we prepend 1s to shape b
            let diff = rank_a - rank_b;

            let padded_shape_b = std::iter::repeat_n(1usize, diff)
                .chain(shape_b.iter().copied())
                .collect::<Vec<usize>>();
            // Now we iterate over both choosing the maximum each time
            shape_a
                .iter()
                .zip(padded_shape_b.iter())
                .enumerate()
                .map(|(index, (&a_dim, &b_dim))| compatibility(a_dim, b_dim, index))
                .collect::<Result<Vec<usize>, anyhow::Error>>()
                .map(Shape::from)
        }
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::Rng;
    use ff_ext::{FieldFrom, GoldilocksExt2};
    use ndarray::{Array, Ix2, Order};

    use crate::{rng_from_env_or_random, testing::random_field_vector};

    use super::*;
    use multilinear_extensions::mle::MultilinearExtension;
    #[test]
    fn test_tensor_basic_ops() {
        let tensor1 = Tensor::new(vec![2, 2].into(), vec![1, 2, 3, 4]);
        let tensor2 = Tensor::new(vec![2, 2].into(), vec![5, 6, 7, 8]);

        let result_add = tensor1.add(&tensor2);
        assert_eq!(
            result_add,
            Tensor::new(vec![2, 2].into(), vec![6, 8, 10, 12]),
            "Element-wise addition failed."
        );

        let result_sub = tensor2.sub(&tensor2);
        assert_eq!(
            result_sub,
            Tensor::zeros(vec![2, 2].into()),
            "Element-wise subtraction failed."
        );

        let result_mul = tensor1.mul(&tensor2);
        assert_eq!(
            result_mul,
            Tensor::new(vec![2, 2].into(), vec![5, 12, 21, 32]),
            "Element-wise multiplication failed."
        );

        let result_scalar = tensor1.scalar_mul(&2);
        assert_eq!(
            result_scalar,
            Tensor::new(vec![2, 2].into(), vec![2, 4, 6, 8]),
            "Element-wise scalar multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matvec() {
        let shape_m = vec![3, 3];
        let tensor_m = Tensor::new(shape_m.clone().into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let matrix = Array::from_vec(tensor_m.get_data().to_vec())
            .into_shape_with_order((shape_m, Order::RowMajor))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let tensor_v = Tensor::new(vec![3].into(), vec![10, 20, 30]);
        let vector = Array::from_vec(tensor_v.get_data().to_vec());

        let result = tensor_m.matvec(&tensor_v);
        let expected_result = matrix.dot(&vector);

        assert_eq!(
            Array::from_vec(result.get_data().to_vec()),
            expected_result,
            "Matrix-vector multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matmul() {
        let shape_a = vec![4, 3];
        let tensor_a = Tensor::new(
            shape_a.clone().into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        );
        let matrix_a = Array::from_vec(tensor_a.get_data().to_vec())
            .into_shape_with_order((shape_a, Order::RowMajor))
            .unwrap();
        let shape_b = vec![3, 3];
        let tensor_b = Tensor::new(
            shape_b.clone().into(),
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
        );
        let matrix_b = Array::from_vec(tensor_b.get_data().to_vec())
            .into_shape_with_order((shape_b, Order::RowMajor))
            .unwrap();

        let result = tensor_a.matmul(&tensor_b);

        let expected_result = matrix_a
            .into_dimensionality::<Ix2>()
            .unwrap()
            .dot(&matrix_b.into_dimensionality::<Ix2>().unwrap());

        assert_eq!(
            Array::from_vec(result.get_data().to_vec())
                .into_shape_with_order((expected_result.shape(), Order::RowMajor))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap(),
            expected_result,
            "Matrix-matrix multiplication failed."
        );
    }

    #[test]
    fn test_tensor_transpose() {
        let matrix_a = Tensor::new(
            vec![3, 4].into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        );
        let matrix_b = Tensor::new(
            vec![4, 3].into(),
            vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12],
        );

        let result = matrix_a.transpose();

        assert_eq!(result, matrix_b, "Matrix transpose failed.");
    }

    #[test]
    fn test_tensor_next_pow_of_two() {
        let shape = vec![3usize, 3].into();
        let mat = Tensor::<Element>::random_seed(&shape, Some(213));
        // println!("{}", mat);
        let new_shape = vec![shape[0].next_power_of_two(), shape[1].next_power_of_two()];
        let new_mat = mat.pad_next_power_of_two();
        assert_eq!(
            new_mat.get_shape(),
            new_shape.into(),
            "Matrix padding to next power of two failed."
        );
    }

    impl Tensor<Element> {
        pub fn get_2d(&self, i: usize, j: usize) -> Element {
            assert!(self.is_matrix() == true);
            self.data[i * self.get_shape()[1] + j]
        }

        pub fn random_eval_point(&self) -> Vec<E> {
            let mut rng = rng_from_env_or_random();
            let r = rng.gen_range(0..self.nrows_2d());
            let c = rng.gen_range(0..self.ncols_2d());
            self.position_to_boolean_2d(r, c)
        }
    }

    #[test]
    fn test_tensor_mle() {
        let mat = Tensor::random(&vec![3, 5].into());
        let shape = mat.get_shape();
        let mat = mat.pad_next_power_of_two();
        println!("matrix {}", mat);
        let mut mle = mat.to_2d_mle::<E>();
        let mut rng = rng_from_env_or_random();
        let (chosen_row, chosen_col) = (rng.gen_range(0..shape[0]), rng.gen_range(0..shape[1]));
        let elem = mat.get_2d(chosen_row, chosen_col);
        let elem_field: E = elem.to_field();
        println!("(x,y) = ({},{}) ==> {:?}", chosen_row, chosen_col, elem);
        let inputs = mat.position_to_boolean_2d(chosen_row, chosen_col);
        let output = mle.evaluate(&inputs);
        assert_eq!(elem_field, output);

        // now try to address one at a time, and starting by the row, which is the opposite order
        // of the boolean variables expected by the MLE API, given it's expecting in LE format.
        let row_input = mat.row_to_boolean_2d(chosen_row);
        mle.fix_high_variables_in_place(&row_input.collect_vec());
        let col_input = mat.col_to_boolean_2d(chosen_col);
        let output = mle.evaluate(&col_input.collect_vec());
        assert_eq!(elem_field, output);
    }

    #[test]
    fn test_tensor_matvec_concatenate() {
        let matrix = Tensor::new(vec![3, 3].into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let vector = Tensor::new(vec![3].into(), vec![10, 20, 30]);

        let result = matrix.concat_matvec_col(&vector);

        assert_eq!(
            result,
            Tensor::new(
                vec![3, 4].into(),
                vec![1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30]
            ),
            "Concatenate matrix vector as columns failed."
        );
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_conv() {
        for i in 0..3 {
            for j in 2..5 {
                for l in 0..4 {
                    for n in 1..(j - 1) {
                        let n_w = 1 << n;
                        let k_w = 1 << l;
                        let n_x = 1 << j;
                        let k_x = 1 << i;
                        let filter1 = Tensor::random(&vec![k_w, k_x, n_w, n_w].into());
                        let filter2 = filter1.clone();
                        let filter1 = filter1.into_fft_conv(&vec![k_x, n_x, n_x].into());
                        let big_x =
                            Tensor::new(vec![k_x, n_x, n_x].into(), vec![3; n_x * n_x * k_x]); //random_vector(n_x*n_x*k_x));
                        let (out_2, _) = filter1.fft_conv::<GoldilocksExt2>(&big_x);
                        let out_1 = filter2.cnn_naive_convolution(&big_x);
                        check_tensor_consistency(out_1, out_2);
                    }
                }
            }
        }
    }

    #[test]
    fn test_tensor_ext_ops() {
        let matrix_a_data = vec![1 as Element, 2, 3, 4, 5, 6, 7, 8, 9];
        let matrix_b_data = vec![10 as Element, 20, 30, 40, 50, 60, 70, 80, 90];
        let matrix_c_data = vec![300 as Element, 360, 420, 660, 810, 960, 1020, 1260, 1500];
        let vector_a_data = vec![10 as Element, 20, 30];
        let vector_b_data = vec![140 as Element, 320, 500];

        let matrix_a_data: Vec<E> = matrix_a_data.iter().map(|x| x.to_field()).collect_vec();
        let matrix_b_data: Vec<E> = matrix_b_data.iter().map(|x| x.to_field()).collect_vec();
        let matrix_c_data: Vec<E> = matrix_c_data.iter().map(|x| x.to_field()).collect_vec();
        let vector_a_data: Vec<E> = vector_a_data.iter().map(|x| x.to_field()).collect_vec();
        let vector_b_data: Vec<E> = vector_b_data.iter().map(|x| x.to_field()).collect_vec();
        let matrix = Tensor::new(vec![3usize, 3].into(), matrix_a_data.clone());
        let vector = Tensor::new(vec![3usize].into(), vector_a_data);
        let vector_expected = Tensor::new(vec![3usize].into(), vector_b_data);

        let result = matrix.matvec(&vector);

        assert_eq!(
            result, vector_expected,
            "Matrix-vector multiplication failed."
        );

        let matrix_a = Tensor::new(vec![3, 3].into(), matrix_a_data);
        let matrix_b = Tensor::new(vec![3, 3].into(), matrix_b_data);
        let matrix_c = Tensor::new(vec![3, 3].into(), matrix_c_data);

        let result = matrix_a.matmul(&matrix_b);

        assert_eq!(result, matrix_c, "Matrix-matrix multiplication failed.");
    }

    #[test]
    fn test_tensor_maxpool2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 3, 4].into(),
            vec![
                99, -35, 18, 104, -26, -48, -80, 106, 10, 8, 79, -7, -128, -45, 24, -91, -7, 88,
                -119, -37, -38, -113, -84, 86, 116, 72, -83, 100, 83, 81, 87, 58, -109, -13, -123,
                102,
            ],
        );
        let expected =
            Tensor::<Element>::new(vec![1, 3, 1, 2].into(), vec![99, 106, 88, 24, 116, 100]);

        let result = input.maxpool2d(2, 2);
        assert_eq!(result, expected, "Maxpool (Element) failed.");
    }

    #[test]
    fn test_tensor_pad_maxpool2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59,
                -64, 70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47,
                8, 112, 65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
            ],
        );
        let expected = Tensor::<Element>::new(
            vec![1, 3, 2, 2].into(),
            vec![104, -1, 74, 3, 19, 70, 79, 21, 112, 65, 69, 105],
        );

        let padded_expected = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                104, 104, -1, -1, 104, 104, -1, -1, 74, 74, 3, 3, 74, 74, 3, 3, 19, 19, 70, 70, 19,
                19, 70, 70, 79, 79, 21, 21, 79, 79, 21, 21, 112, 112, 65, 65, 112, 112, 65, 65, 69,
                69, 105, 105, 69, 69, 105, 105,
            ],
        );

        let (result, padded_result) = input.padded_maxpool2d();
        assert_eq!(result, expected, "Maxpool (Element) failed.");
        assert_eq!(
            padded_result, padded_expected,
            "Padded Maxpool (Element) failed."
        );
    }

    #[test]
    fn test_pad_tensor_for_mle() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59,
                -64, 70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47,
                8, 112, 65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
            ],
        );

        let padded = input.pad_next_power_of_two();

        padded
            .get_shape()
            .iter()
            .zip(input.get_shape().iter())
            .for_each(|(padded_dim, input_dim)| {
                assert_eq!(*padded_dim, input_dim.next_power_of_two())
            });

        let input_data = input.get_data();
        let padded_data = padded.get_data();
        for i in 0..1 {
            for j in 0..3 {
                for k in 0..4 {
                    for l in 0..4 {
                        let index = 3 * 4 * 4 * i + 4 * 4 * j + 4 * k + l;
                        assert_eq!(input_data[index], padded_data[index]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_tensor_pad() {
        let shape_a = Shape::from_it([3, 1, 1]);
        let tensor_a = Tensor::<Element>::new(shape_a.clone(), vec![1; shape_a.product()]);

        let shape_b = vec![4, 1, 1];
        let tensor_b = Tensor::<Element>::new(shape_b.clone().into(), vec![1, 1, 1, 0]);

        let tensor_c = tensor_a.pad_next_power_of_two();
        assert_eq!(tensor_b, tensor_c);

        // Here we test the generic padding
        let tensor_b = Tensor::<Element>::new(shape_b.into(), vec![1, 1, 1, 17]);

        let tensor_c = tensor_a.generic_pad_next_power_of_two(17);
        assert_eq!(tensor_b, tensor_c);
    }

    #[test]
    fn test_tensor_pad_to_shape() {
        let shape = Shape::from_it([1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]);
        let target = Shape::from_it([2]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]);
        tensor.pad_to_shape(target.clone());
        assert_eq!(tensor, res);

        let shape = Shape::from_it([2]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1, 2]);
        let target = Shape::from_it([3]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 2, 0]);
        tensor.pad_to_shape(target.clone());
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]);
        let target = Shape::from_it([2, 1]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]);
        tensor.pad_to_shape(target.clone());
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]);
        let target = Shape::from_it([1, 2]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]);
        tensor.pad_to_shape(target.clone());
        assert_eq!(tensor, res);

        let shape = Shape::from_it([2, 2]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1, 2, 3, 4]);
        let target = Shape::from_it([3, 3]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 2, 0, 3, 4, 0, 0, 0, 0]);
        tensor.pad_to_shape(target.clone());
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 1, 1]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1]);
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),

            vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        tensor.pad_to_shape(target);
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 1, 3]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),

            vec![
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        tensor.pad_to_shape(target);
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 3, 1]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),
            vec![
                1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        tensor.pad_to_shape(target);
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 2, 1, 3]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2]);
        let target = Shape::from_it([2, 3, 5, 7]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),
            vec![
                // x=0 y=0
                1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=0 y=1
                2, 2, 2, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=0 y=2
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=0
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=1
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=2
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ],
        );
        tensor.pad_to_shape(target);
        assert_eq!(tensor, res);
    }

    #[test]
    fn test_tensor_conv2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
            ],
        );

        let weights = Tensor::<Element>::new(
            vec![2, 3, 2, 2].into(),
            vec![
                1, 0, -1, 2, 0, 1, -1, 1, 1, -1, 0, 2, -1, 1, 2, 0, 1, 0, 2, -1, 0, -1, 1, 1,
            ],
        );

        let bias = Tensor::<Element>::new(vec![2].into(), vec![3, -3]);

        let expected = Tensor::<Element>::new(
            vec![1, 2, 2, 2].into(),
            vec![21, 22, 26, 27, 25, 25, 26, 26],
        );

        let result = input.conv2d(&weights, &bias, 1);
        assert_eq!(result, expected, "Conv2D (Element) failed.");
    }

    #[test]
    fn test_tensor_minimal_conv2d() {
        // k_n,k_c,k_h,k_w
        let conv_shape = vec![2, 3, 3, 3].into();
        let conv = Tensor::<Element>::random(&conv_shape);
        // minimal input shape is 1,k_c,k_h,k_w
        let input_shape = vec![1, 3, 3, 3].into();
        let input = Tensor::<Element>::random(&input_shape);
        // minimal bias shape is k_n
        let bias = Tensor::<Element>::random(&vec![2].into());
        let output = input.conv2d(&conv, &bias, 1);
        assert_eq!(output.get_shape(), vec![1, 2, 1, 1].into());
    }

    #[test]
    fn test_tensor_pad_matrix_to_ignore_garbage() {
        let old_shape: Shape = vec![2usize, 3, 3].into();
        let orows = 10usize;
        let ocols = old_shape.product();

        let new_shape: Shape = vec![3usize, 4, 4].into();
        let nrows = 12usize;
        let ncols = new_shape.product();

        let og_t = Tensor::<Element>::random(&old_shape);
        let og_flat_t = og_t.flatten(); // This is equivalent to conv2d output (flattened)

        let mut pad_t = og_t.clone();
        pad_t.pad_to_shape(new_shape.clone().into());
        let pad_flat_t = pad_t.flatten();

        let og_mat = Tensor::random(&vec![orows, ocols].into()); // This is equivalent to the first dense matrix
        let og_result = og_mat.matvec(&og_flat_t);

        let pad_mat =
            og_mat.pad_matrix_to_ignore_garbage(&old_shape, &new_shape, &vec![nrows, ncols].into());
        let pad_result = pad_mat.matvec(&pad_flat_t);

        assert_eq!(
            og_result.get_data()[..orows],
            pad_result.get_data()[..orows],
            "Unable to get rid of garbage values from conv fft."
        );
    }

    #[test]
    fn test_tensor_slice_2d() {
        let tensor = Tensor::<Element>::new(vec![3, 3].into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let sliced = tensor.slice_2d(0, 2);
        assert_eq!(sliced.get_shape(), vec![2, 3].into());
        assert_eq!(sliced.get_data(), vec![1, 2, 3, 4, 5, 6]);
        let sliced = tensor.slice_2d(2, 3);
        assert_eq!(sliced.get_shape(), vec![1, 3].into());
        assert_eq!(sliced.get_data(), vec![7, 8, 9]);
    }

    #[test]
    fn test_tensor_add_dim2() {
        let tensor = Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]);
        let vector = Tensor::<Element>::new(vec![3].into(), vec![10, 20, 30]);
        let result = tensor.add_dim2(&vector);
        assert_eq!(result.get_shape(), vec![2, 3].into());
        assert_eq!(result.get_data(), vec![11, 22, 33, 14, 25, 36]);
    }

    #[test]
    fn test_tensor_concat() {
        let mut tensor = Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]);
        let vector = Tensor::<Element>::new(vec![3].into(), vec![10, 20, 30]);
        tensor.concat(vector);

        assert_eq!(tensor.get_shape(), vec![3, 3].into());
        assert_eq!(tensor.get_data(), vec![1, 2, 3, 4, 5, 6, 10, 20, 30]);

        let vector = Tensor::<Element>::new(vec![1, 3].into(), vec![66, 77, 88]);
        tensor.concat(vector);
        assert_eq!(tensor.get_shape(), vec![4, 3].into());
        assert_eq!(
            tensor.get_data(),
            vec![1, 2, 3, 4, 5, 6, 10, 20, 30, 66, 77, 88]
        );
    }

    #[test]
    fn test_tensor_get() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        );
        // 2 + 2 * 3 + 1 * 3 * 3 = 17
        assert_eq!(tensor.get(vec![1, 2, 2]), tensor.data[17]);
    }

    #[test]
    fn test_tensor_permute3d() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        );
        // i,j,k --> j,i,k -> new shape = 3,2,3
        let permuted = tensor.permute3d(&[1, 0, 2]);
        assert_eq!(permuted.get_shape(), vec![3, 2, 3].into());
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..3 {
                    let [new_i, new_j, new_k] = [j, i, k];
                    let expected = tensor.get(vec![i, j, k]);
                    let given = permuted.get(vec![new_i, new_j, new_k]);
                    assert_eq!(expected, given);
                }
            }
        }

        let tensor = Tensor::<Element>::random(&vec![18, 5, 27].into());
        let permuted = tensor.permute3d(&vec![1, 2, 0]);
        assert_eq!(permuted.get_shape(), vec![5, 27, 18].into())
    }

    #[test]
    fn test_tensor_slice_3d() {
        let tensor = Tensor::<Element>::new(
            vec![3, 2, 2].into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        );
        let sliced = tensor.slice_3d(1, 3);
        assert_eq!(sliced.get_data(), vec![5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(sliced.get_shape(), vec![2, 2, 2].into());
    }

    #[test]
    fn test_tensor_slices_last_dim() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        );

        let mut slices = tensor.slice_last_dim();

        // First slice
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        // Second slice
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        // Third slice
        assert_eq!(slices.next().unwrap(), &[7, 8, 9]);
        // Fourth slice
        assert_eq!(slices.next().unwrap(), &[10, 11, 12]);
        // Fifth slice
        assert_eq!(slices.next().unwrap(), &[13, 14, 15]);
        // Sixth slice
        assert_eq!(slices.next().unwrap(), &[16, 17, 18]);
        // No more slices
        assert_eq!(slices.next(), None);
    }

    #[test]
    fn test_tensor_slice_on_dim() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        );
        let (mut slices, shape) = tensor.slice_on_dim(1);
        assert_eq!(shape, Shape(vec![3]));
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        assert_eq!(slices.next().unwrap(), &[7, 8, 9]);
        assert_eq!(slices.next().unwrap(), &[10, 11, 12]);
        assert_eq!(slices.next().unwrap(), &[13, 14, 15]);
        assert_eq!(slices.next().unwrap(), &[16, 17, 18]);
        assert_eq!(slices.next(), None);

        let (mut slices, shape) = tensor.slice_on_dim(0);
        assert_eq!(shape, Shape(vec![3, 3]));
        assert_eq!(slices.next().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            slices.next().unwrap(),
            &[10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        assert_eq!(slices.next(), None);

        let (slices, shape) = tensor.slice_on_dim(2);
        assert_eq!(shape, tensor.get_shape());
        let data = slices.flatten().cloned().collect::<Vec<_>>();
        assert_eq!(
            data,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
        );
    }

    #[test]
    fn test_shape() {
        let shape = Shape(vec![2, 3, 4]);
        let permuted = shape.permute(&[1, 0, 2]);
        assert_eq!(permuted.0, vec![3, 2, 4]);
    }

    #[test]
    fn test_tensor_argmax() {
        let tensor = Tensor::<Element>::new(vec![3].into(), vec![1, 2, 3]);
        let argmax = tensor.argmax();
        assert_eq!(argmax, 2);
    }

    #[test]
    fn test_tensor_stack_all() {
        let tensors = vec![
            Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]),
            Tensor::<Element>::new(vec![2, 3].into(), vec![7, 8, 9, 10, 11, 12]),
        ];
        let stacked = Tensor::<Element>::stack_all(tensors.clone()).unwrap();
        assert_eq!(stacked.get_shape(), vec![4, 3].into());
        assert_eq!(
            stacked.get_data(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        let stacked =
            Tensor::<Element>::stack_all(tensors.into_iter().map(|t| t.unsqueeze(0))).unwrap();
        assert_eq!(stacked.get_shape(), vec![2, 2, 3].into());
        assert_eq!(
            stacked.get_data(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    fn eval_lteq_poly(x_i: &[Element], y_i: &[Element]) -> Element {
        assert_eq!(x_i.len(), y_i.len());
        x_i.into_iter()
            .rev()
            .zip(y_i.into_iter().rev())
            .fold(Element::from(1), |acc, (x, y)| {
                acc * (1 - x - y + 2 * x * y) + (1 - x) * y
            })
    }

    fn eval_mle<F: ExtensionField + FieldFrom<u64>>(point: &[F]) -> F {
        let x_i = &point[..point.len() / 2];
        let y_i = &point[point.len() / 2..];
        x_i.into_iter()
            .zip(y_i.into_iter())
            .fold(F::from_v(1), |acc, (&x, &y)| {
                acc * (F::from_v(1) - x - y + F::from_v(2) * x * y) + (F::from_v(1) - x) * y
            })
    }

    fn to_be_bits<const NUM_BITS: usize>(x: Element) -> [Element; NUM_BITS] {
        (0..NUM_BITS)
            .rev()
            .map(|i| {
                let mask = 1 << i;
                let bit = (x & Element::from(mask)) >> i;
                bit
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[test]
    fn test_zeroifier_evaluation() {
        // create zeroifier matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let zeroifier_data = (0..num_columns * num_columns)
            .map(|i| {
                let r = i / num_columns;
                let c = i % num_columns;
                if r >= c {
                    Element::from(1)
                } else {
                    Element::from(0)
                }
            })
            .collect_vec();
        println!("Data: {zeroifier_data:?}");
        let zeroifier = Tensor::new(vec![num_columns, num_columns].into(), zeroifier_data);
        assert_eq!(zeroifier.get_2d(0, 0), Element::from(1));
        assert_eq!(
            zeroifier.get_2d(num_columns - 1, num_columns - 1),
            Element::from(1)
        );
        assert_eq!(zeroifier.get_2d(0, 1), Element::from(0));
        assert_eq!(zeroifier.get_2d(1, 1), Element::from(1));
        assert_eq!(zeroifier.get_2d(1, 2), Element::from(0));

        let mle = zeroifier.to_2d_mle::<GoldilocksExt2>();

        for i in 0..num_columns {
            for j in 0..num_columns {
                let x_i = to_be_bits::<NUM_BITS>(Element::from(i as u32));
                let y_i = to_be_bits::<NUM_BITS>(Element::from(j as u32));
                let cmp = eval_lteq_poly(&y_i, &x_i);
                assert_eq!(
                    zeroifier.get_2d(i, j),
                    cmp,
                    "Zeroifier evaluation failed for ({}, {})",
                    i,
                    j
                );
                // build point for MLE: first column bits in little-endiian order, then rows bits in little-endian order
                let point = y_i
                    .into_iter()
                    .rev()
                    .chain(x_i.into_iter().rev())
                    .map(|bit| GoldilocksExt2::from_v(bit as u64))
                    .collect_vec();
                let eval = mle.evaluate(&point);
                assert_eq!(eval, GoldilocksExt2::from_v(cmp as u64));
                let quick_eval = eval_mle(&point);
                assert_eq!(eval, quick_eval);
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<GoldilocksExt2>(NUM_BITS * 2);
            assert_eq!(mle.evaluate(&point), eval_mle(&point),);
        }
    }

    #[test]
    fn test_tril() {
        // Test diag = 0
        let tensor = Tensor::<Element>::tril(4, 1, 0);
        let real_value: Vec<Element> = vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1];
        assert_eq!(tensor.get_data(), real_value);
        // Test diag = 1
        let tensor = Tensor::<Element>::tril(4, 1, 1);
        let real_value: Vec<Element> = vec![1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(tensor.get_data(), real_value);
        // Test diag = -1
        let tensor = Tensor::<Element>::tril(4, 1, -1);
        let real_value: Vec<Element> = vec![0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0];
        assert_eq!(tensor.get_data(), real_value);
    }

    #[test]
    fn test_expand() {
        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let dim = rng.gen_range(1usize..11);
            let initial_shape: Shape = vec![dim].into();

            let tensor = Tensor::<f32>::random(&initial_shape);
            let data = tensor.get_data().to_vec();
            let expanded_shape: Shape = vec![7, dim].into();

            let expanded_tensor_res = tensor.expand(&expanded_shape);

            // Check the result is OK
            assert!(
                expanded_tensor_res.is_ok(),
                "error: {:?}",
                expanded_tensor_res
            );

            // Now check that the data is correct
            let expanded_tensor = expanded_tensor_res.unwrap();
            // We expect the tensor to have repeated each element 7 times
            let correct_data = vec![data; 7].concat();

            assert_eq!(expanded_tensor.data, correct_data);
        }

        // Now we check it expands along dimensions properly
        let shape: Shape = vec![3, 1, 2].into();
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0,

            3.0, 4.0, 
                                  
            5.0, 6.0];

        let expansion_shape: Shape = vec![3, 3, 2].into();
        #[rustfmt::skip]
        let expansion_data = vec![
            1.0, 2.0, 
            1.0, 2.0, 
            1.0, 2.0, 
            
            3.0, 4.0, 
            3.0, 4.0, 
            3.0, 4.0, 
            
            5.0, 6.0, 
            5.0, 6.0, 
            5.0, 6.0,
        ];

        let expanded_tensor = Tensor::<f64>::new(shape, data.clone())
            .expand(&expansion_shape)
            .unwrap();
        assert_eq!(expanded_tensor.data, expansion_data);
    }
}
