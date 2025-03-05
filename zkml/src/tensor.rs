use anyhow::bail;
use ark_std::rand::{
    self, SeedableRng, distributions::Standard, prelude::Distribution, rngs::StdRng, thread_rng,
};
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
    slice::ParallelSliceMut,
};
use std::{
    cmp::PartialEq,
    fmt::{self},
};

use crate::{
    Element,
    pooling::MAXPOOL2D_KERNEL_SIZE,
    quantization::Fieldizer,
    testing::{random_vector, random_vector_seed},
    to_bit_sequence_le,
};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> {
    /// Create a new tensor with given shape and data
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert!(
            shape.iter().product::<usize>() == data.len(),
            "Shape does not match data length."
        );
        Self { data, shape }
    }

    /// Get the dimensions of the tensor
    pub fn dims(&self) -> Vec<usize> {
        assert!(self.shape.len() > 0, "Empty tensor");
        self.shape.clone()
    }

    /// Is vector
    pub fn is_vector(&self) -> bool {
        self.dims().len() == 1
    }

    /// Is matrix
    pub fn is_matrix(&self) -> bool {
        self.dims().len() == 2
    }

    /// Get the number of rows from the matrix
    pub fn nrows_2d(&self) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        let dims = self.dims();
        return dims[0];
    }

    /// Get the number of cols from the matrix
    pub fn ncols_2d(&self) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        let dims = self.dims();
        return dims[1];
    }

    /// Returns the number of boolean variables needed to address any row, and any columns
    pub fn num_vars_2d(&self) -> (usize, usize) {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        (
            self.nrows_2d().ilog2() as usize,
            self.ncols_2d().ilog2() as usize,
        )
    }

    ///
    pub fn get_data(&self) -> &[T] {
        &self.data
    }
}

impl<T> Tensor<T>
where
    T: Copy + Clone + Send + Sync,
    T: std::iter::Sum,
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
    T: std::default::Default,
{
    ///
    pub fn flatten(&self) -> Self {
        let new_data = self.get_data().to_vec();
        let new_shape = vec![new_data.len()];
        Self {
            data: new_data,
            shape: new_shape,
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == other.shape, "Shape mismatch for addition.");
        Tensor {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == other.shape, "Shape mismatch for subtraction.");
        Tensor {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a - *b)
                .collect(),
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(
            self.shape == other.shape,
            "Shape mismatch for multiplication."
        );
        Tensor {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a * *b)
                .collect(),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &T) -> Tensor<T> {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| *x * *scalar).collect(),
        }
    }

    pub fn from_coeffs_2d(data: Vec<Vec<T>>) -> anyhow::Result<Self> {
        let n_rows = data.len();
        let n_cols = data.first().expect("at least one row in a matrix").len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();
        if data.len() != n_rows * n_cols {
            bail!(
                "Number of rows and columns do not match with the total number of values in the Vec<Vec<>>"
            );
        };
        let shape = vec![n_rows, n_cols];
        Ok(Self { data, shape })
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

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            // data: vec![T::zero(); size],
            data: vec![Default::default(); size],
            shape,
        }
    }

    pub fn pad_next_power_of_two_2d(mut self) -> Self {
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

        assert!(n == vec_len, "Matrix columns must match vector size.");

        let mut result = Tensor::zeros(vec![m]);

        result.data.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = (0..n)
                .into_par_iter()
                .map(|j| self.data[i * n + j] * vector.data[j])
                .sum::<T>();
        });

        result
    }

    /// Transpose the matrix (2D tensor)
    pub fn transpose(&self) -> Tensor<T> {
        assert!(self.is_matrix(), "Tensor is not a matrix.");
        let (m, n) = (self.shape[0], self.shape[1]);

        let mut result = Tensor::zeros(vec![n, m]);
        for i in 0..m {
            for j in 0..n {
                result.data[j * m + i] = self.data[i * n + j];
            }
        }
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

        let mut result = Tensor::<T>::zeros(new_shape);

        // Create a new matrix with expanded dimensions
        for i in 0..old_rows {
            for j in 0..old_cols {
                result.data[i * new_cols + j] = self.data[i * old_cols + j];
            }
        }
        *self = result;
    }
}

impl Tensor<Element> {
    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    pub fn random(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = random_vector(size);
        Self { data, shape }
    }

    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    pub fn random_seed(shape: Vec<usize>, seed: Option<u64>) -> Self {
        let size = shape.iter().product();
        let data = random_vector_seed(size, seed);
        Self { data, shape }
    }

    /// Returns the evaluation point, in order for (row,col) addressing
    pub fn evals_2d<F: ExtensionField>(&self) -> Vec<F> {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.data.par_iter().map(|e| e.to_field()).collect()
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
}

impl<T> Tensor<T>
where
    T: PartialOrd + Clone,
    T: std::default::Default,
{
    pub fn maxpool2d(&self, kernel_size: usize, stride: usize) -> Tensor<T> {
        let dims = self.dims().len();
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
        let mut output = vec![T::default(); outer_dims * out_h * out_w];

        for n in 0..outer_dims {
            let matrix_idx = n * (h * w);
            for i in 0..out_h {
                for j in 0..out_w {
                    let src_idx = matrix_idx + (i * stride) * w + (j * stride);
                    let mut max_val = self.data[src_idx].clone();

                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            let src_idx = matrix_idx + (i * stride + ki) * w + (j * stride + kj);
                            let value = self.data[src_idx].clone();

                            if value > max_val {
                                max_val = value;
                            }
                        }
                    }

                    let out_idx = n * out_h * out_w + i * out_w + j;
                    output[out_idx] = max_val;
                }
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[dims - 2] = out_h;
        new_shape[dims - 1] = out_w;

        Tensor {
            data: output,
            shape: new_shape,
        }
    }

    pub fn padded_maxpool2d(&self) -> (Tensor<T>, Tensor<T>) {
        let kernel_size = MAXPOOL2D_KERNEL_SIZE;
        let stride = MAXPOOL2D_KERNEL_SIZE;

        let maxpool_result = self.maxpool2d(kernel_size, stride);

        let dims: usize = self.dims().len();
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

        let mut padded_maxpool_data = vec![T::default(); self.shape.iter().product()];

        let outer_dims: usize = self.shape[..dims - 2].iter().product();
        let maxpool_h = (h - kernel_size) / stride + 1;
        let maxpool_w = (w - kernel_size) / stride + 1;

        for n in 0..outer_dims {
            let matrix_idx = n * (h * w);
            for i in 0..maxpool_h {
                for j in 0..maxpool_w {
                    let maxpool_idx = n * maxpool_h * maxpool_w + i * maxpool_w + j;
                    let maxpool_value = maxpool_result.data[maxpool_idx].clone();

                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            let out_idx = matrix_idx + (i * stride + ki) * w + (j * stride + kj);
                            padded_maxpool_data[out_idx] = maxpool_value.clone();
                        }
                    }
                }
            }
        }

        let padded_maxpool_tensor = Tensor {
            data: padded_maxpool_data,
            shape: self.dims(),
        };

        (maxpool_result, padded_maxpool_tensor)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::iter::Sum,
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    pub fn get4d(&self) -> (usize, usize, usize, usize) {
        let n_size = self.shape.get(0).cloned().unwrap_or(1);
        let c_size = self.shape.get(1).cloned().unwrap_or(1);
        let h_size = self.shape.get(2).cloned().unwrap_or(1);
        let w_size = self.shape.get(3).cloned().unwrap_or(1);

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

    pub fn conv2d(&self, kernels: &Tensor<T>, bias: &Tensor<T>, stride: usize) -> Tensor<T> {
        let (n_size, c_size, h_size, w_size) = self.get4d();
        let (k_n, k_c, k_h, k_w) = kernels.get4d();

        // Validate shapes
        assert_eq!(c_size, k_c, "Input and kernel channels must match!");
        assert_eq!(
            bias.shape,
            vec![k_n],
            "Bias shape must match number of kernels!"
        );

        let out_h = (h_size - k_h) / stride + 1;
        let out_w = (w_size - k_w) / stride + 1;
        let out_shape = vec![n_size, k_n, out_h, out_w];

        let mut output = vec![T::default(); n_size * k_n * out_h * out_w];

        for n in 0..n_size {
            for o in 0..k_n {
                for oh in 0..out_h {
                    for ow in 0..out_w {
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

                        // Add bias for this output channel (o)
                        sum = sum + bias.data[o];

                        let output_index =
                            n * (k_n * out_h * out_w) + o * (out_h * out_w) + oh * out_w + ow;
                        output[output_index] = sum;
                    }
                }
            }
        }

        Tensor {
            data: output,
            shape: out_shape,
        }
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let shape = self.shape.clone();
        let mut shape = shape.into_iter().rev().collect_vec();

        while shape.len() < 4 {
            shape.push(1);
        }

        if shape.len() == 4 {
            let (batches, channels, height, width) =
                (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
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

mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use multilinear_extensions::mle::MultilinearExtension;

    use super::*;

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
        let mat = Tensor::random_seed(shape.clone(), Some(213));
        // println!("{}", mat);
        let new_shape = vec![shape[0].next_power_of_two(), shape[1].next_power_of_two()];
        let new_mat = mat.pad_next_power_of_two_2d();
        assert_eq!(
            new_mat.dims(),
            new_shape,
            "Matrix padding to next power of two failed."
        );
    }

    impl Tensor<Element> {
        pub fn get_2d(&self, i: usize, j: usize) -> Element {
            assert!(self.is_matrix() == true);
            self.data[i * self.dims()[1] + j]
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
        let mat = Tensor::random(vec![3, 5]);
        let shape = mat.dims();
        let mat = mat.pad_next_power_of_two_2d();
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
}
