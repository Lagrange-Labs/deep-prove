use crate::{
    ScalingFactor,
    quantization::{self, MAX_FLOAT, MIN_FLOAT},
};
use anyhow::bail;
use ark_std::rand::{self, Rng, SeedableRng, rngs::StdRng};
use ff::Field;
use ff_ext::ExtensionField;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
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
    error::Error,
    fmt::{self, Debug, Display, Formatter, Result as FmtResult},
};

use crate::{
    Element,
    layers::pooling::MAXPOOL2D_KERNEL_SIZE,
    quantization::{Fieldizer, IntoElement},
    to_bit_sequence_le,
};

pub trait Number:
    Copy
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
}

impl Number for Element {
    const MIN: Element = Element::MIN;
    const MAX: Element = Element::MAX;
    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(*quantization::MIN..=*quantization::MAX)
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(&other)
    }
    fn is_negative(&self) -> bool {
        *self < 0
    }
}
impl Number for f32 {
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
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
}
impl Number for GoldilocksExt2 {
    const MIN: GoldilocksExt2 = GoldilocksExt2::ZERO;
    const MAX: GoldilocksExt2 = GoldilocksExt2::ZERO;

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
/// - For n=16: squares it 16 times (32-16) to get a 16th root of unity
/// - For n=8:  squares it 24 times (32-8) to get an 8th root of unity
/// - For n=4:  squares it 28 times (32-4) to get a 4th root of unity
/// The initial ROOT_OF_UNITY constant is verified to be a 32nd root of unity in the field implementation.
pub fn get_root_of_unity<E: ExtensionField>(n: usize) -> E {
    let mut rou = E::ROOT_OF_UNITY;

    for _ in 0..(32 - n) {
        rou = rou * rou;
    }

    return rou;
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
    (0..u.len() / 2).into_iter().map(move |i| u[len - 1 - i])
}
/// flag: false -> FFT
/// flag: true -> iFFT
pub fn fft<E: ExtensionField + Send + Sync>(v: &mut Vec<E>, flag: bool) {
    let n = v.len();
    let logn = ark_std::log2(n) as u32;
    let mut rev: Vec<usize> = vec![0; n];
    let mut w: Vec<E> = vec![E::ZERO; n];

    rev[0] = 0;

    for i in 1..n {
        rev[i] = rev[i >> 1] >> 1 | ((i) & 1) << (logn - 1);
    }
    w[0] = E::ONE;

    let rou: E = get_root_of_unity(logn as usize);
    w[1] = rou;

    if flag == true {
        w[1] = w[1].invert().unwrap();
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

    if flag == true {
        let mut ilen = E::from(n as u64);
        ilen = ilen.invert().unwrap();
        debug_assert_eq!(ilen * E::from(n as u64), E::ONE, "Error in inv");
        v.par_iter_mut().for_each(|val| {
            *val = *val * ilen;
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
            .map(|e| {
                index_u(e.as_slice(), n_x)
                    .map(|e| e.into_element())
                    .collect::<Vec<_>>()
            })
            .flatten()
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    og_shape: Vec<usize>,
}

impl Tensor<Element> {
    pub fn dequantize(&self, s: &ScalingFactor) -> Tensor<f32> {
        let data = self
            .data
            .iter()
            .map(|e| s.dequantize(e))
            .collect::<Vec<_>>();
        Tensor::new(self.shape.clone(), data)
    }
    pub fn into_fft_conv(self, input_shape: &[usize]) -> Self {
        let shape = self.shape;
        let data = self.data;
        assert!(
            shape.iter().product::<usize>() == data.len(),
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
            data,                                      /* Note that field elements are back into Element */
            shape: vec![shape[0], shape[1], n_w, n_w], // nw is the padded version of the input
            og_shape: real_shape,
        }
    }
    // Create specifically a new convolution. The input shape is needed to compute the
    // output and properly arrange the weights
    #[deprecated]
    pub fn new_conv(shape: Vec<usize>, input_shape: Vec<usize>, data: Vec<Element>) -> Self {
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
        for i in 0..out.len() {
            fft(&mut out[i], true);
        }

        // TODO: remove the requirement to keep the output value intact
        let output = out;
        let conv_data = ConvData::new(real_input, input, x_vec, prod, output, n_x);
        return (
            Tensor::new(
                vec![self.shape[0], n_x, n_x],
                conv_data.output_as_element.clone(),
            ),
            conv_data,
        );
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
                    .into_iter()
                    .map(|k| {
                        (0..dim1)
                            .into_iter()
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
        return (
            Tensor::new(
                vec![self.shape[0], n_x, n_x],
                conv_data.output_as_element.clone(),
            ),
            conv_data,
        );
    }
    /// Returns the evaluation point, in order for (row,col) addressing
    pub fn evals_2d<F: ExtensionField>(&self) -> Vec<F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.evals_flat()
    }
    pub fn evals_flat<F: ExtensionField>(&self) -> Vec<F> {
        self.data.par_iter().map(|e| e.to_field()).collect()
    }
    pub fn get_conv_weights<F: ExtensionField>(&self) -> Vec<F> {
        let mut data = vec![F::ZERO; self.data.len()];
        for i in 0..data.len() {
            data[i] = self.data[i].to_field();
        }
        data
    }
    /// Returns a MLE of the matrix that can be evaluated.
    pub fn to_mle_2d<F: ExtensionField>(&self) -> DenseMultilinearExtension<F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        assert!(
            self.nrows_2d().is_power_of_two(),
            "number of rows {} is not a power of two",
            self.nrows_2d()
        );
        assert!(
            self.ncols_2d().is_power_of_two(),
            "number of columns {} is not a power of two",
            self.ncols_2d()
        );
        // N variable to address 2^N rows and M variables to address 2^M columns
        let num_vars = self.nrows_2d().ilog2() + self.ncols_2d().ilog2();
        DenseMultilinearExtension::from_evaluations_ext_vec(num_vars as usize, self.evals_2d())
    }

    pub fn to_mle_flat<F: ExtensionField>(&self) -> DenseMultilinearExtension<F> {
        DenseMultilinearExtension::from_evaluations_ext_vec(
            self.data.len().ilog2() as usize,
            self.evals_flat(),
        )
    }
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
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert!(
            shape.iter().product::<usize>() == data.len(),
            "Shape does not match data length: shape {:?}->{}vs data.len() {}",
            shape,
            shape.iter().product::<usize>(),
            data.len()
        );
        Self {
            data,
            shape,
            og_shape: vec![0],
        }
    }

    /// Is vector
    pub fn is_vector(&self) -> bool {
        self.get_shape().len() == 1
    }
    /// Is matrix
    pub fn is_matrix(&self) -> bool {
        self.get_shape().len() == 2
    }
    ///
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

        return cols;
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
    pub fn get_shape(&self) -> Vec<usize> {
        assert!(self.shape.len() > 0, "Empty tensor");
        self.shape.clone()
    }
    /// Get the input shape of the tensor
    /// TODO: Remove it
    pub fn get_input_shape(&self) -> Vec<usize> {
        assert!(self.shape.len() > 0, "Empty tensor");
        self.og_shape.clone()
    }
    ///
    pub fn get_data(&self) -> &[T] {
        &self.data
    }
    /// Gets a mutable reference to the inner data
    pub fn get_data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Gets a reference to a specific element of the tensor data specified by a coordinate
    pub fn get_from_coord(&self, coord: &[usize]) -> Result<&T, TensorError> {
        if self.shape.len() != coord.len() {
            return Err(TensorError::ParameterError(format!(
                "Provided coordinate was not of the correct length, expected: {}, got: {}",
                self.shape.len(),
                coord.len()
            )));
        }

        let (index, _) = coord.iter().zip(self.shape.iter()).try_fold((0usize, 1usize), |(index_acc, dim_acc), (&c, &d)| {
            if c >= d {
                Err(TensorError::ParameterError(format!("Coordinate had value: {} which is greater than or equal to the size of the dimension: {} ", c, d)))
            } else {
            Ok((index_acc + c * dim_acc, dim_acc * d))
            }
        })?;

        Ok(&self.data[index])
    }

    /// Gets a mutable reference to a specific element of the tensor data specified by a coordinate
    pub fn get_from_coord_mut(&mut self, coord: &[usize]) -> Result<&mut T, TensorError> {
        if self.shape.len() != coord.len() {
            return Err(TensorError::ParameterError(format!(
                "Provided coordinate was not of the correct length, expected: {}, got: {}",
                self.shape.len(),
                coord.len()
            )));
        }

        let (index, _) = coord.iter().zip(self.shape.iter()).try_fold((0usize, 1usize), |(index_acc, dim_acc), (&c, &d)| {
            if c >= d {
                Err(TensorError::ParameterError(format!("Coordinate had value: {} which is greater than or equal to the size of the dimension: {} ", c, d)))
            } else {
            Ok((index_acc + c * dim_acc, dim_acc * d))
            }
        })?;

        Ok(&mut self.data[index])
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
    pub fn real_shape(&self) -> Vec<usize> {
        self.og_shape.clone()
    }
    // Returns the size of an individual filter
    pub fn filter_size(&self) -> usize {
        self.shape[2] * self.shape[2]
    }
}

impl<T> Tensor<T>
where
    T: Clone,
{
    ///
    pub fn flatten(&self) -> Self {
        let new_data = self.get_data().to_vec();
        let new_shape = vec![new_data.len()];
        Self {
            data: new_data,
            shape: new_shape,
            og_shape: vec![0],
        }
    }
    pub fn matix_from_coeffs(data: Vec<Vec<T>>) -> anyhow::Result<Self> {
        let n_rows = data.len();
        let n_cols = data.first().expect("at least one row in a matrix").len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();
        if data.len() != n_rows * n_cols {
            bail!(
                "Number of rows and columns do not match with the total number of values in the Vec<Vec<>>"
            );
        };
        let shape = vec![n_rows, n_cols];
        Ok(Self {
            data,
            shape,
            og_shape: vec![0],
        })
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn row_to_boolean_2d<F: ExtensionField>(&self, row: usize) -> impl Iterator<Item = F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        let (nvars_rows, _) = self.num_vars_2d();
        to_bit_sequence_le(row, nvars_rows).map(|b| F::from(b as u64))
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn col_to_boolean_2d<F: ExtensionField>(&self, col: usize) -> impl Iterator<Item = F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        let (_, nvars_col) = self.num_vars_2d();
        to_bit_sequence_le(col, nvars_col).map(|b| F::from(b as u64))
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
}

impl<T> Tensor<T>
where
    T: Number,
{
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Tensor<T> {
        assert!(
            self.shape.iter().product::<usize>() == new_shape.iter().product::<usize>(),
            "Shape mismatch for reshape"
        );
        self.shape = new_shape;
        self
    }
    pub fn max_abs_output(&self) -> T {
        self.data
            .iter()
            .fold(T::default(), |max, x| max.cmp_max(&x.absolute_value()))
    }
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            // data: vec![T::zero(); size],
            data: vec![Default::default(); size],
            shape,
            og_shape: vec![0],
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == other.shape, "Shape mismatch for addition.");
        let mut data = vec![Default::default(); self.data.len()];
        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = self.data[i] + other.data[i];
        });

        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0],
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
            og_shape: vec![0],
            data,
        }
    }
    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(
            self.shape == other.shape,
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
            og_shape: vec![0],
            data,
        }
    }
    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &T) -> Tensor<T> {
        Tensor {
            shape: self.shape.clone(),
            og_shape: vec![0],
            data: self.data.par_iter().map(|x| *x * *scalar).collect(),
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
    fn pad_next_power_of_two_2d(mut self) -> Self {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        // assume the matrix is already well formed and there is always n_rows and n_cols
        // this is because we control the creation of the matrix in the first place

        let rows = self.nrows_2d();
        let cols = self.ncols_2d();

        let new_rows = if rows.is_power_of_two() {
            rows
        } else {
            rows.next_power_of_two()
        };

        let new_cols = if cols.is_power_of_two() {
            cols
        } else {
            cols.next_power_of_two()
        };

        let mut padded = Tensor::zeros(vec![new_rows, new_cols]);

        // Copy original values into the padded matrix
        for i in 0..rows {
            for j in 0..cols {
                padded.data[i * new_cols + j] = self.data[i * cols + j];
            }
        }

        // Parallelize row-wise copying
        padded
            .data
            .par_chunks_mut(new_cols)
            .enumerate()
            .for_each(|(i, row)| {
                if i < rows {
                    row[..cols].copy_from_slice(&self.data[i * cols..(i + 1) * cols]);
                }
            });

        self = padded;

        self
    }
    /// Recursively pads the tensor so its ready to be viewed as an MLE
    pub fn pad_next_power_of_two(&self) -> Self {
        let shape = self.get_shape();

        let padded_data = Self::recursive_pad(self.get_data(), &shape);

        let padded_shape = shape
            .into_iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<usize>>();

        Tensor::<T>::new(padded_shape, padded_data)
    }
    fn recursive_pad(data: &[T], remaining_dims: &[usize]) -> Vec<T> {
        match remaining_dims.len() {
            // If the remaining dims show we are a vector simply pad
            1 => data
                .iter()
                .cloned()
                .chain(std::iter::repeat(T::default()))
                .take(remaining_dims[0].next_power_of_two())
                .collect::<Vec<T>>(),
            // If the remaining dims show that we are a matrix call the matrix method
            2 => {
                let tmp_tensor = Tensor::<T>::new(remaining_dims.to_vec(), data.to_vec())
                    .pad_next_power_of_two_2d();
                tmp_tensor.data.clone()
            }
            // Otherwise we recurse
            _ => {
                let chunk_size = remaining_dims[1..].iter().product::<usize>();
                let mut unpadded_data = data
                    .par_chunks(chunk_size)
                    .map(|data_chunk| Self::recursive_pad(data_chunk, &remaining_dims[1..]))
                    .collect::<Vec<Vec<T>>>();
                let elem_size = unpadded_data[0].len();
                unpadded_data.resize(remaining_dims[0].next_power_of_two(), vec![
                    T::default();
                    elem_size
                ]);
                unpadded_data.concat()
            }
        }
    }
    pub fn pad_to_shape(&mut self, target_shape: Vec<usize>) {
        if target_shape.len() != self.shape.len() {
            panic!("Target shape must have the same number of dimensions as the tensor.");
        }

        let current_shape = &self.shape;

        assert!(current_shape.iter().zip(&target_shape).all(|(c, t)| c <= t));
        // if current_shape.iter().zip(&target_shape).all(|(c, t)| c <= t) {
        //     // No padding is needed if all dimensions are already the correct size
        //     return;
        // }

        let mut new_data = vec![T::default(); target_shape.iter().product()];

        let mut strides: Vec<usize> = vec![1; current_shape.len()];
        for i in (0..current_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * current_shape[i + 1];
        }

        let mut target_strides: Vec<usize> = vec![1; target_shape.len()];
        for i in (0..target_shape.len() - 1).rev() {
            target_strides[i] = target_strides[i + 1] * target_shape[i + 1];
        }

        for index in 0..self.data.len() {
            let mut original_indices = vec![0; current_shape.len()];
            let mut remaining = index;

            for (j, stride) in strides.iter().enumerate() {
                original_indices[j] = remaining / stride;
                remaining %= stride;
            }

            if original_indices
                .iter()
                .zip(&target_shape)
                .all(|(idx, max)| idx < max)
            {
                let new_index: usize = original_indices
                    .iter()
                    .zip(&target_strides)
                    .map(|(idx, stride)| idx * stride)
                    .sum();
                new_data[new_index] = self.data[index].clone();
            }
        }

        self.data = new_data;
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

        let mut result = Tensor::zeros(vec![m, p]);

        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, res)| {
                let i = index / p;
                let j = index % p;

                *res = (0..n)
                    .into_par_iter()
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

        let mut result = Tensor::zeros(vec![m]);

        result.data.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = (0..n)
                .into_par_iter()
                .map(|j| self.data[i * n + j] * vector.data[j])
                .sum::<T>();
        });

        result
    }
    pub fn conv_prod(&self, x: &Vec<Vec<T>>, w: &Vec<Vec<T>>, ii: usize, jj: usize) -> T {
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
    pub fn single_naive_conv(&self, w: Vec<Vec<T>>, x: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let mut out: Vec<Vec<T>> =
            vec![vec![Default::default(); x[0].len() - w[0].len() + 1]; x.len() - w.len() + 1];
        out.par_iter_mut().enumerate().for_each(|(i, out_row)| {
            out_row.par_iter_mut().enumerate().for_each(|(j, out_val)| {
                *out_val = self.conv_prod(&x, &w, i, j);
            });
        });
        out
    }
    pub fn add_matrix(&self, m1: &mut Vec<Vec<T>>, m2: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let mut m = vec![vec![Default::default(); m1[0].len()]; m1.len()];
        m.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, val)| {
                *val = m1[i][j] + m2[i][j];
            });
        });
        return m;
    }
    // Implementation of the stadard convolution algorithm.
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
                let temp = self.single_naive_conv(w[i][j].clone(), x[j].clone());
                conv[i] = self.add_matrix(&mut conv[i], temp);
            }
        }

        return Tensor::new(
            vec![k_w, xt.shape[1] - n_w + 1, xt.shape[1] - n_w + 1],
            conv.into_iter()
                .flat_map(|inner_vec| inner_vec.into_iter())
                .flat_map(|inner_inner_vec| inner_inner_vec.into_iter())
                .collect(),
        );
    }
    /// Transpose the matrix (2D tensor)
    pub fn transpose(&self) -> Tensor<T> {
        assert!(self.is_matrix(), "Tensor is not a matrix.");
        let (m, n) = (self.shape[0], self.shape[1]);

        let mut result = Tensor::zeros(vec![n, m]);
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
        let mut result = Tensor::zeros(vec![rows, new_cols]);

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
    pub fn reshape_to_fit_inplace_2d(&mut self, new_shape: Vec<usize>) {
        let old_rows = self.nrows_2d();
        let old_cols = self.ncols_2d();

        assert!(new_shape.len() == 2, "Tensor is not matrix");
        let new_rows = new_shape[0];
        let new_cols = new_shape[1];
        // Ensure we never lose information by requiring the new dimensions to be at least
        // as large as the original ones
        assert!(
            new_rows >= old_rows,
            "Cannot shrink matrix rows from {} to {} - would lose information",
            old_rows,
            new_rows
        );
        assert!(
            new_cols >= old_cols,
            "Cannot shrink matrix columns from {} to {} - would lose information",
            old_cols,
            new_cols
        );

        let new_data: Vec<T> = (0..new_rows * new_cols)
            .into_par_iter()
            .map(|idx| {
                let i = idx / new_cols;
                let j = idx % new_cols;
                if i < old_rows && j < old_cols {
                    self.data[i * old_cols + j].clone()
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
            "Kernel size ({}) is larger than input dimensions ({}, {})",
            kernel_size,
            h,
            w
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
                let mut max_val = self.data[src_idx].clone();

                for ki in 0..kernel_size {
                    for kj in 0..kernel_size {
                        let src_idx = matrix_idx + (i * stride + ki) * w + (j * stride + kj);
                        let value = self.data[src_idx].clone();
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
            og_shape: vec![0],
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
            "Currently works only with kernel size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        assert!(
            w % MAXPOOL2D_KERNEL_SIZE == 0,
            "Currently works only with stride size {}",
            MAXPOOL2D_KERNEL_SIZE
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
                maxpool_result.data[maxpool_idx].clone()
            })
            .collect();

        let padded_maxpool_tensor = Tensor {
            data: padded_maxpool_data,
            shape: self.get_shape(),
            og_shape: vec![0],
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
            c_size, k_c,
            "Input {} and kernel {} channels must match!",
            c_size, k_c
        );
        assert_eq!(
            bias.shape,
            vec![k_n],
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
                            sum = sum + self.get(n, c, h, w) * kernels.get(o, c, kh, kw);
                        }
                    }
                }

                // Add bias
                sum + bias.data[o].clone()
            })
            .collect();

        Tensor {
            data: output,
            shape: out_shape,
            og_shape: vec![0],
        }
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
            (self.shape.get(0).cloned().unwrap_or(1), 1)
        };
        let c_size = self.shape.get(0 + offset).cloned().unwrap_or(1);
        let h_size = self.shape.get(1 + offset).cloned().unwrap_or(1);
        let w_size = self.shape.get(2 + offset).cloned().unwrap_or(1);

        (n_size, c_size, h_size, w_size)
    }

    /// Retrieves an element using (N, C, H, W) indexing
    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> T {
        assert!(self.shape.len() <= 4);

        let (n_size, c_size, h_size, w_size) = self.get4d();

        assert!(n < n_size);
        let flat_index = n * (c_size * h_size * w_size) + c * (h_size * w_size) + h * w_size + w;
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
        mat_shp_pad: &[usize],
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
                    self.data[old_loc].clone()
                } else {
                    T::default() // Default value for non-mapped positions
                }
            })
            .collect();

        Tensor::new(mat_shp_pad.to_vec(), new_data)
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shape = self.shape.clone();

        while shape.len() < 4 {
            shape = shape.into_iter().rev().collect_vec();
            shape.push(1);
            shape = shape.into_iter().rev().collect_vec();
        }

        if shape.len() == 4 {
            let (batches, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let channel_size = height * width;
            let batch_size = channels * channel_size;

            for b in 0..batches {
                writeln!(
                    f,
                    "Batch {} [{} channels, {}x{}]:",
                    b, channels, height, width
                )?;
                for c in 0..channels {
                    writeln!(f, "  Channel {}:", c)?;
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

impl<T: Number> Tensor<T> {
    pub fn max_value(&self) -> T {
        self.data.iter().fold(T::MIN, |max, x| max.cmp_max(x))
    }
    pub fn min_value(&self) -> T {
        self.data.iter().fold(T::MAX, |min, x| min.cmp_min(x))
    }
    pub fn random(shape: &[usize]) -> Self {
        Self::random_seed(shape, Some(rand::random::<u64>()))
    }

    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    pub fn random_seed(shape: &[usize], seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(rand::random::<u64>()); // Use provided seed or default
        let mut rng = StdRng::seed_from_u64(seed);
        let size = shape.iter().product();
        let data = (0..size).map(|_| T::random(&mut rng)).collect();
        Self {
            data,
            shape: shape.to_vec(),
            og_shape: vec![0],
        }
    }
}

#[derive(Debug, Clone)]
/// Enum relating to tensor errors
pub enum TensorError {
    ParameterError(String),
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TensorError::ParameterError(s) => {
                write!(f, "Parameters for Tensor operation were incorrect: {}", s)
            }
        }
    }
}

impl Error for TensorError {}

#[cfg(test)]
mod test {

    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;

    use super::*;
    use multilinear_extensions::mle::MultilinearExtension;
    #[test]
    fn test_tensor_basic_ops() {
        let tensor1 = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]);
        let tensor2 = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]);

        let result_add = tensor1.add(&tensor2);
        assert_eq!(
            result_add,
            Tensor::new(vec![2, 2], vec![6, 8, 10, 12]),
            "Element-wise addition failed."
        );

        let result_sub = tensor2.sub(&tensor2);
        assert_eq!(
            result_sub,
            Tensor::zeros(vec![2, 2]),
            "Element-wise subtraction failed."
        );

        let result_mul = tensor1.mul(&tensor2);
        assert_eq!(
            result_mul,
            Tensor::new(vec![2, 2], vec![5, 12, 21, 32]),
            "Element-wise multiplication failed."
        );

        let result_scalar = tensor1.scalar_mul(&2);
        assert_eq!(
            result_scalar,
            Tensor::new(vec![2, 2], vec![2, 4, 6, 8]),
            "Element-wise scalar multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matvec() {
        let matrix = Tensor::new(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let vector = Tensor::new(vec![3], vec![10, 20, 30]);

        let result = matrix.matvec(&vector);

        assert_eq!(
            result,
            Tensor::new(vec![3], vec![140, 320, 500]),
            "Matrix-vector multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matmul() {
        let matrix_a = Tensor::new(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let matrix_b = Tensor::new(vec![3, 3], vec![10, 20, 30, 40, 50, 60, 70, 80, 90]);

        let result = matrix_a.matmul(&matrix_b);

        assert_eq!(
            result,
            Tensor::new(vec![3, 3], vec![
                300, 360, 420, 660, 810, 960, 1020, 1260, 1500
            ]),
            "Matrix-matrix multiplication failed."
        );
    }

    #[test]
    fn test_tensor_transpose() {
        let matrix_a = Tensor::new(vec![3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let matrix_b = Tensor::new(vec![4, 3], vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);

        let result = matrix_a.transpose();

        assert_eq!(result, matrix_b, "Matrix transpose failed.");
    }

    #[test]
    fn test_tensor_next_pow_of_two() {
        let shape = vec![3usize, 3];
        let mat = Tensor::<Element>::random_seed(&shape, Some(213));
        // println!("{}", mat);
        let new_shape = vec![shape[0].next_power_of_two(), shape[1].next_power_of_two()];
        let new_mat = mat.pad_next_power_of_two();
        assert_eq!(
            new_mat.get_shape(),
            new_shape,
            "Matrix padding to next power of two failed."
        );
    }

    impl Tensor<Element> {
        pub fn get_2d(&self, i: usize, j: usize) -> Element {
            assert!(self.is_matrix() == true);
            self.data[i * self.get_shape()[1] + j]
        }

        pub fn random_eval_point(&self) -> Vec<E> {
            let mut rng = thread_rng();
            let r = rng.gen_range(0..self.nrows_2d());
            let c = rng.gen_range(0..self.ncols_2d());
            self.position_to_boolean_2d(r, c)
        }
    }

    #[test]
    fn test_tensor_mle() {
        let mat = Tensor::random(&vec![3, 5]);
        let shape = mat.get_shape();
        let mat = mat.pad_next_power_of_two();
        println!("matrix {}", mat);
        let mut mle = mat.clone().to_mle_2d::<E>();
        let (chosen_row, chosen_col) = (
            thread_rng().gen_range(0..shape[0]),
            thread_rng().gen_range(0..shape[1]),
        );
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
        let matrix = Tensor::new(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let vector = Tensor::new(vec![3], vec![10, 20, 30]);

        let result = matrix.concat_matvec_col(&vector);

        assert_eq!(
            result,
            Tensor::new(vec![3, 4], vec![1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30]),
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
                        let filter1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
                        let filter2 = filter1.clone();
                        let filter1 = filter1.into_fft_conv(&vec![k_x, n_x, n_x]);
                        let big_x = Tensor::new(vec![k_x, n_x, n_x], vec![3; n_x * n_x * k_x]); //random_vector(n_x*n_x*k_x));
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
        let matrix = Tensor::new(vec![3usize, 3], matrix_a_data.clone());
        let vector = Tensor::new(vec![3usize], vector_a_data);
        let vector_expected = Tensor::new(vec![3usize], vector_b_data);

        let result = matrix.matvec(&vector);

        assert_eq!(
            result, vector_expected,
            "Matrix-vector multiplication failed."
        );

        let matrix_a = Tensor::new(vec![3, 3], matrix_a_data);
        let matrix_b = Tensor::new(vec![3, 3], matrix_b_data);
        let matrix_c = Tensor::new(vec![3, 3], matrix_c_data);

        let result = matrix_a.matmul(&matrix_b);

        assert_eq!(result, matrix_c, "Matrix-matrix multiplication failed.");
    }

    #[test]
    fn test_tensor_maxpool2d() {
        let input = Tensor::<Element>::new(vec![1, 3, 3, 4], vec![
            99, -35, 18, 104, -26, -48, -80, 106, 10, 8, 79, -7, -128, -45, 24, -91, -7, 88, -119,
            -37, -38, -113, -84, 86, 116, 72, -83, 100, 83, 81, 87, 58, -109, -13, -123, 102,
        ]);
        let expected = Tensor::<Element>::new(vec![1, 3, 1, 2], vec![99, 106, 88, 24, 116, 100]);

        let result = input.maxpool2d(2, 2);
        assert_eq!(result, expected, "Maxpool (Element) failed.");
    }

    #[test]
    fn test_tensor_pad_maxpool2d() {
        let input = Tensor::<Element>::new(vec![1, 3, 4, 4], vec![
            93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59, -64,
            70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47, 8, 112,
            65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
        ]);
        let expected = Tensor::<Element>::new(vec![1, 3, 2, 2], vec![
            104, -1, 74, 3, 19, 70, 79, 21, 112, 65, 69, 105,
        ]);

        let padded_expected = Tensor::<Element>::new(vec![1, 3, 4, 4], vec![
            104, 104, -1, -1, 104, 104, -1, -1, 74, 74, 3, 3, 74, 74, 3, 3, 19, 19, 70, 70, 19, 19,
            70, 70, 79, 79, 21, 21, 79, 79, 21, 21, 112, 112, 65, 65, 112, 112, 65, 65, 69, 69,
            105, 105, 69, 69, 105, 105,
        ]);

        let (result, padded_result) = input.padded_maxpool2d();
        assert_eq!(result, expected, "Maxpool (Element) failed.");
        assert_eq!(
            padded_result, padded_expected,
            "Padded Maxpool (Element) failed."
        );
    }

    #[test]
    fn test_pad_tensor_for_mle() {
        let input = Tensor::<Element>::new(vec![1, 3, 4, 4], vec![
            93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59, -64,
            70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47, 8, 112,
            65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
        ]);

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
        let shape_a = vec![3, 1, 1];
        let tensor_a = Tensor::<Element>::new(shape_a.clone(), vec![1; shape_a.iter().product()]);

        let shape_b = vec![4, 1, 1];
        let tensor_b = Tensor::<Element>::new(shape_b, vec![1, 1, 1, 0]);

        let tensor_c = tensor_a.pad_next_power_of_two();
        assert_eq!(tensor_b, tensor_c);
    }

    #[test]
    fn test_tensor_pad_to_shape() {
        let shape_a = vec![3, 1, 1];
        let mut tensor_a =
            Tensor::<Element>::new(shape_a.clone(), vec![1; shape_a.iter().product()]);

        let shape_b = vec![3, 4, 4];
        let tensor_b = Tensor::<Element>::new(shape_b.clone(), vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);

        tensor_a.pad_to_shape(shape_b);
        assert_eq!(tensor_b, tensor_a);
    }

    #[test]
    fn test_tensor_conv2d() {
        let input = Tensor::<Element>::new(vec![1, 3, 3, 3], vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
        ]);

        let weights = Tensor::<Element>::new(vec![2, 3, 2, 2], vec![
            1, 0, -1, 2, 0, 1, -1, 1, 1, -1, 0, 2, -1, 1, 2, 0, 1, 0, 2, -1, 0, -1, 1, 1,
        ]);

        let bias = Tensor::<Element>::new(vec![2], vec![3, -3]);

        let expected =
            Tensor::<Element>::new(vec![1, 2, 2, 2], vec![21, 22, 26, 27, 25, 25, 26, 26]);

        let result = input.conv2d(&weights, &bias, 1);
        assert_eq!(result, expected, "Conv2D (Element) failed.");
    }

    #[test]
    fn test_tensor_minimal_conv2d() {
        // k_n,k_c,k_h,k_w
        let conv_shape = vec![2, 3, 3, 3];
        let conv = Tensor::<Element>::random(&conv_shape);
        // minimal input shape is 1,k_c,k_h,k_w
        let input_shape = vec![1, 3, 3, 3];
        let input = Tensor::<Element>::random(&input_shape);
        // minimal bias shape is k_n
        let bias = Tensor::<Element>::random(&vec![2]);
        let output = input.conv2d(&conv, &bias, 1);
        assert_eq!(output.get_shape(), vec![1, 2, 1, 1]);
    }

    #[test]
    fn test_tensor_pad_matrix_to_ignore_garbage() {
        let old_shape = vec![2usize, 3, 3];
        let orows = 10usize;
        let ocols = old_shape.iter().product::<usize>();

        let new_shape = vec![3usize, 4, 4];
        let nrows = 12usize;
        let ncols = new_shape.iter().product::<usize>();

        let og_t = Tensor::<Element>::random(&old_shape);
        let og_flat_t = og_t.flatten(); // This is equivalent to conv2d output (flattened)

        let mut pad_t = og_t.clone();
        pad_t.pad_to_shape(new_shape.clone());
        let pad_flat_t = pad_t.flatten();

        let og_mat = Tensor::random(&vec![orows, ocols]); // This is equivalent to the first dense matrix
        let og_result = og_mat.matvec(&og_flat_t);

        let pad_mat =
            og_mat.pad_matrix_to_ignore_garbage(&old_shape, &new_shape, &vec![nrows, ncols]);
        let pad_result = pad_mat.matvec(&pad_flat_t);

        assert_eq!(
            og_result.get_data()[..orows],
            pad_result.get_data()[..orows],
            "Unable to get rid of garbage values from conv fft."
        );
    }
}
