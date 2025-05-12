use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use crate::{
    Claim, Element, Prover,
    commit::{compute_betas_eval, precommit::PolyID},
    iop::{
        context::ContextAux,
        split_sumcheck::{IOPSplitProverState, SplitSumcheckError},
        verifier::Verifier,
    },
    layers::LayerProof,
    tensor::Tensor,
};

use ff_ext::ExtensionField;
use gkr::util::batch_inversion;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::VPAuxInfo,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPVerifierState};
use transcript::Transcript;

use super::LayerCtx;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Stride {
    Conv {
        height: usize,
        width: usize,
        input_rows: usize,
        input_cols: usize,
    },
}

impl Display for Stride {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Stride::Conv { height, width, .. } => write!(f, "Conv Strides ({},{})", height, width),
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct StrideProof<E: ExtensionField> {
    /// The split-sumcheck proof for the stride
    pub(crate) sumcheck_proof: IOPProof<E>,
}

#[derive(Debug, Clone)]
/// Enum for any errors that occur during stride
pub enum StrideError {
    ParameterError(String),
    ProvingError(String),
}

impl Error for StrideError {}

impl Display for StrideError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            StrideError::ParameterError(s) => {
                write!(f, "Parameters were incorrect for stride: {}", s)
            }
            StrideError::ProvingError(s) => {
                write!(f, "Error occured during stride proving: {}", s)
            }
        }
    }
}

impl From<SplitSumcheckError> for StrideError {
    fn from(e: SplitSumcheckError) -> Self {
        match e {
            SplitSumcheckError::ParameterError(s) => {
                StrideError::ProvingError(format!("Split sumcheck parameter error: {}", s))
            }
            SplitSumcheckError::ProvingError(s) => {
                StrideError::ProvingError(format!("Split sumcheck proving error: {}", s))
            }
        }
    }
}

impl Stride {
    pub fn new_conv(height: usize, width: usize, input_rows: usize, input_cols: usize) -> Stride {
        Stride::Conv {
            height,
            width,
            input_rows,
            input_cols,
        }
    }

    /// Getter for the height stride size
    pub fn height(&self) -> usize {
        match self {
            Stride::Conv { height, .. } => *height,
        }
    }

    /// Getter for the width stride size
    pub fn width(&self) -> usize {
        match self {
            Stride::Conv { width, .. } => *width,
        }
    }

    /// Getter for the number of rows of the input
    pub fn input_rows(&self) -> usize {
        match self {
            Stride::Conv { input_rows, .. } => *input_rows,
        }
    }

    /// Getter for the number of columns of the input
    pub fn input_cols(&self) -> usize {
        match self {
            Stride::Conv {
                input_cols: input_columns,
                ..
            } => *input_columns,
        }
    }

    /// Returns the number of variables the left matrix MLE has after fixing the columns
    pub fn left_variables(&self) -> usize {
        self.input_rows().next_power_of_two().ilog2() as usize
    }

    /// Returns the number of variables the right matrix MLE has after fixing the rows
    pub fn right_variables(&self) -> usize {
        self.input_cols().next_power_of_two().ilog2() as usize
    }

    pub fn stride_sample<T: Default + Clone + From<u64>>(
        &self,
        tensor: &Tensor<T>,
    ) -> Result<Tensor<T>, StrideError> {
        let height = self.height();
        let width = self.width();
        let result = tensor.subsample(height, width);
        Ok(result)
    }

    pub fn op(&self, tensor: &Tensor<Element>) -> Result<Tensor<Element>, StrideError> {
        self.stride_sample(tensor)
            .and_then(|t| Ok(t.pad_next_power_of_two()))
    }

    pub fn get_fixed_mles<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<[Vec<E>; 2], StrideError> {
        let rows = self.input_rows();
        let columns = self.input_cols();

        let left_variables = ((rows - 1) / self.height() + 1).next_power_of_two().ilog2() as usize;

        let right_variables = ((columns - 1) / self.width() + 1)
            .next_power_of_two()
            .ilog2() as usize;

        if point.len() != left_variables + right_variables {
            return Err(StrideError::ParameterError(format!(
                "Cannot fix stride polynomials, provided point has {} variables, expected: {}",
                point.len(),
                left_variables + right_variables
            )));
        }

        // Compute the left and right beta poly evals
        let left_beta_eval = compute_betas_eval(&point[left_variables..]);
        let right_beta_eval = compute_betas_eval(&point[..left_variables]);

        match *self {
            Stride::Conv { height, width, .. } => {
                let left_evals = (0..rows)
                    .map(|i| {
                        if i % height == 0 {
                            left_beta_eval[i / height]
                        } else {
                            E::ZERO
                        }
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(rows.next_power_of_two())
                    .collect::<Vec<E>>();
                let right_evals = (0..columns)
                    .map(|i| {
                        if i % width == 0 {
                            right_beta_eval[i / width]
                        } else {
                            E::ZERO
                        }
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(columns.next_power_of_two())
                    .collect::<Vec<E>>();
                Ok([left_evals, right_evals])
            }
        }
    }

    pub fn prove_step<'b, E, T>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: Claim<E>,
        input: &Tensor<E>,
    ) -> Result<Claim<E>, StrideError>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        // Store a reference to the last_claim point
        let point = &last_claim.point;

        // Get the left and right MLEs for the sumcheck.
        let [left, right]: [Vec<E>; 2] = self.get_fixed_mles::<E>(point)?;

        // We need to know how many of the input tensors high variables to fix in place
        // let input = input.pad_next_power_of_two();
        let input_shape = input.get_shape();
        let middle = if input_shape.len() < 2 {
            return Err(StrideError::ParameterError(format!(
                "Proving input had dimension {}, need at least 2 dimensions for stride",
                input_shape.len()
            )));
        } else if input_shape.len() > 2 {
            let two_dimensional_vars = input_shape
                .iter()
                .skip(input_shape.len() - 2)
                .product::<usize>()
                .ilog2() as usize;

            input
                .get_data()
                .to_vec()
                .into_mle()
                .fix_high_variables(&point[two_dimensional_vars..])
                .get_ext_field_vec()
                .to_vec()
        } else {
            input.get_data().to_vec()
        };

        // let in_cols = input_shape[1];
        // let calculated_eval = (0usize..middle.len()).fold(E::ZERO, |acc, i| {
        //     println!(
        //         "i: {} (r, c): ({}, {}), acc[i]: {:?}",
        //         i,
        //         i / in_cols,
        //         i % in_cols,
        //         left[i / in_cols] * middle[i] * right[i % in_cols],
        //     );
        //     acc + (left[i / in_cols] * middle[i] * right[i % in_cols])
        // });

        // assert_eq!(calculated_eval, last_claim.eval);
        let (proof, state) =
            IOPSplitProverState::<E>::prove_split_sumcheck(left, middle, right, prover.transcript)?;
        // Clone the point so we can use it in the out claim
        let point = proof.point.clone();

        // Push the stride proof to the proof list
        prover.push_proof(LayerProof::<E>::Stride(StrideProof::<E>::new(proof)));

        Ok(Claim {
            point,
            eval: state.middle[0],
        })
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        _id: PolyID,
        mut ctx_aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let layer_context = LayerCtx::<E>::Stride(*self);
        let shape = self.stride_shape(&ctx_aux.last_output_shape).unwrap();
        ctx_aux.last_output_shape = shape
            .into_iter()
            .map(|s| s.next_power_of_two())
            .collect::<Vec<usize>>();
        (layer_context, ctx_aux)
    }

    pub(crate) fn verify_stride<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &StrideProof<E>,
    ) -> Result<Claim<E>, StrideError> {
        let Claim::<E> { point, eval } = &last_claim;

        let claimed_sum = *eval;

        // Make the VPAuxInfo
        let vars = (self.input_cols().next_power_of_two() * self.input_rows().next_power_of_two())
            .ilog2() as usize;
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![vars, vars]]);

        println!("Verify_stride num vars: {}", vars);
        let subclaim = IOPVerifierState::<E>::verify(
            claimed_sum,
            proof.sumcheck_proof(),
            &aux_info,
            verifier.transcript,
        );

        // We reverse the subclaim point as the first challenge generated is the highest variable
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .rev()
            .collect::<Vec<E>>();

        let mut matrix_evals = self.compute_matrix_evals(point, &sumcheck_point)?;

        batch_inversion(&mut matrix_evals);

        let out_eval = matrix_evals
            .into_iter()
            .fold(subclaim.expected_evaluation, |acc, v| acc * v);

        // We need to append the correct number of elements from the last claim point to the sumcheck point
        // this is because we fixed highvariables during proving so that the sumcheck input was the MLE of a matrix
        // rather than a higher dimensional tensor.

        let left_variables = (((self.input_rows() - 1) / self.height()) + 1)
            .next_power_of_two()
            .ilog2() as usize;
        let right_variables = (((self.input_cols() - 1) / self.width()) + 1)
            .next_power_of_two()
            .ilog2() as usize;

        let point = sumcheck_point
            .into_iter()
            .chain(point.iter().skip(left_variables + right_variables).copied())
            .collect::<Vec<E>>();

        Ok(Claim {
            point,
            eval: out_eval,
        })
    }

    /// Function used by the [`Verifier`] to compute the evaluations of the stride matrices
    fn compute_matrix_evals<E: ExtensionField>(
        &self,
        last_point: &[E],
        sumcheck_point: &[E],
    ) -> Result<Vec<E>, StrideError> {
        let right_variables = self.input_cols().next_power_of_two().ilog2() as usize;
        let left_variables = self.input_rows().next_power_of_two().ilog2() as usize;

        if sumcheck_point.len() != left_variables + right_variables {
            return Err(StrideError::ParameterError(format!(
                "Cannot compute stride matrix evals, provided point has {} variables, expected: {}",
                sumcheck_point.len(),
                left_variables + right_variables
            )));
        }

        let [left, right]: [Vec<E>; 2] = self.get_fixed_mles::<E>(last_point)?;

        let left_eval = left.into_mle().evaluate(&sumcheck_point[..left_variables]);

        let right_eval = right.into_mle().evaluate(&sumcheck_point[left_variables..]);

        Ok(vec![left_eval, right_eval])
    }

    /// Given a shape returns the shape after stride is performed
    pub fn stride_shape(&self, shape: &[usize]) -> Result<Vec<usize>, StrideError> {
        let num_dims = shape.len();
        // Need to have at least two dimensions to pad
        if num_dims < 2 {
            return Err(StrideError::ParameterError(
                "Cannot calculate stride shape as input shape had fewer than two dimensions"
                    .to_string(),
            ));
        }
        // Check the provided shape agrees with the expected number of columns and rows
        if shape[num_dims - 2] < self.input_rows() {
            return Err(StrideError::ParameterError(format!(
                "Input tensor shape incorrect, expected minimum {} rows but input tensor had {} rows",
                self.input_rows(),
                shape[num_dims - 2]
            )));
        }

        if shape[num_dims - 1] < self.input_cols() {
            return Err(StrideError::ParameterError(format!(
                "Input tensor shape incorrect, expected minimum {} columns but input tensor had {} columns",
                self.input_cols(),
                shape[num_dims - 1]
            )));
        }

        let new_height = ((self.input_cols() - 1) / self.height()) + 1;
        let new_width = ((self.input_rows() - 1) / self.width()) + 1;

        Ok(shape
            .iter()
            .copied()
            .take(num_dims - 2)
            .chain([new_height, new_width])
            .collect::<Vec<usize>>())
    }
}

impl<E: ExtensionField> StrideProof<E> {
    pub fn new(sumcheck_proof: IOPProof<E>) -> StrideProof<E> {
        StrideProof { sumcheck_proof }
    }

    /// Getter for the sumcheck proof
    pub(crate) fn sumcheck_proof(&self) -> &IOPProof<E> {
        &self.sumcheck_proof
    }
}

mod tests {

    use ark_std::rand::{SeedableRng, rngs::StdRng, thread_rng};

    use goldilocks::GoldilocksExt2;

    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{
        Context, Element, default_transcript,
        layers::convolution::Convolution,
        onnx_parse::conv2d_shape,
        quantization::{TensorElementizer, TensorFielder},
        tensor::Tensor,
    };

    use super::*;

    #[test]
    fn test_conv2d_convfft() {
        let input_shape: Vec<usize> = vec![1, 1, 6, 6];
        let conv_shape_og: Vec<usize> = vec![2, 1, 2, 2];
        let weight = Tensor::random_seed(conv_shape_og.clone(), Some(0));
        let bias: Tensor<Element> = Tensor::zeros(vec![conv_shape_og[0]]);
        let input = Tensor::random_seed(input_shape.clone(), Some(0));
        let output = input.conv2d(&weight, &bias, (1, 1), (0, 0));

        let mut input = input.clone();
        input.reshape(&input_shape[1..].to_vec());

        let padded_input = input.pad_next_power_of_two();
        let weight_padded = weight.pad_next_power_of_two();
        let bias_padded = bias.pad_next_power_of_two();
        let filter_fft = Tensor::new_conv(
            weight_padded.get_shape(),
            padded_input.get_shape(),
            weight_padded.get_data().to_vec(),
        );
        let fft_conv = Convolution {
            filter: filter_fft,
            bias: bias_padded,
        };
        let (fft_output, _) = fft_conv.op::<GoldilocksExt2>(&padded_input);

        assert!(output == fft_output);
    }

    // Generate the left matrix for strides
    fn gen_left_matrix(stride_h: usize, rows: usize) -> Tensor<Element> {
        let left_h = (rows - 1) / stride_h + 1;
        let left_w = rows;
        let left_shape = vec![left_h, left_w];

        let mut data = vec![0; left_shape.iter().product()];

        // Set L[i][j] = 1 where j == i * s and j < cols_max
        for i in 0..left_h {
            let j = i * stride_h;
            if j < left_w {
                let idx = i * left_w + j;
                data[idx] = 1;
            }
        }

        Tensor::new(left_shape, data)
    }

    // Generate the right matrix for strides
    fn gen_right_matrix(stride_w: usize, cols: usize) -> Tensor<Element> {
        let right_h = cols;
        let right_w = (cols - 1) / stride_w + 1;
        let right_shape = vec![right_h, right_w];

        let mut data = vec![0; right_shape.iter().product()];

        // Set R[i][j] = 1 where i == j * s and i < cols_max
        for j in 0..right_w {
            let i = j * stride_w;
            if i < right_h {
                let idx = i * right_w + j;
                data[idx] = 1;
            }
        }

        Tensor::new(right_shape, data)
    }

    #[test]
    fn test_stride_plain() {
        let input_shape: Vec<usize> = vec![1, 1, 6, 6];
        let conv_shape_og: Vec<usize> = vec![2, 1, 2, 2];
        let weight = Tensor::random_seed(conv_shape_og.clone(), Some(0));
        let bias: Tensor<Element> = Tensor::zeros(vec![conv_shape_og[0]]);
        let input = Tensor::random_seed(input_shape.clone(), Some(0));
        let output_a = input.conv2d(&weight, &bias, (1, 1), (0, 0));

        let [ch, rows, cols] = [output_a.shape[1], output_a.shape[2], output_a.shape[3]];
        let offset = rows * cols;

        // =====================================================================

        let stride_checker = |stride: (usize, usize)| {
            let (stride_h, stride_w) = (stride.0, stride.1);

            let left_matrix = gen_left_matrix(stride_h, rows);
            let right_matrix = gen_right_matrix(stride_w, cols);

            let mut new_data = Vec::new();
            let mut new_shape = Vec::new();

            for c in 0..ch {
                let x = Tensor::new(
                    vec![rows, cols],
                    output_a.data[c * offset..(c + 1) * offset].to_vec(),
                );
                let result = left_matrix.matmul(&x).matmul(&right_matrix);
                new_data.extend(result.data);
                new_shape = result.shape;
            }

            new_shape.insert(0, ch);
            new_shape.insert(0, 1);
            let result = Tensor::new(new_shape, new_data);

            let expected = input.conv2d(&weight, &bias, (stride_h, stride_w), (0, 0));

            assert!(expected == result);
        };

        stride_checker((2, 2));
        stride_checker((2, 1));
    }

    fn stride_mle<E: ExtensionField>(
        input: Tensor<Element>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(), StrideError> {
        // println!(
        //     "Height: {:?}, Weight: {:?}, Input rows: {:?}, Input columns: {:?}",
        //     stride_h,
        //     stride_w,
        //     input.shape[input.shape.len() - 2],
        //     input.shape[input.shape.len() - 1]
        // );

        let mut rng = StdRng::seed_from_u64(0);

        let input_tensor_element = input.pad_next_power_of_two();
        let input_tensor: Tensor<E> = input_tensor_element.clone().to_fields();

        let expected_output_tensor = input.subsample(stride_h, stride_w).pad_next_power_of_two();

        let expected_output_mle = expected_output_tensor.to_mle_2d::<E>();

        // println!("Input tensor: {}", input_tensor_element);
        // println!(
        //     "Input tensor MLE: {:?}",
        //     input_tensor_element.to_mle_2d::<E>()
        // );
        // println!("Expected output tensor: {}", expected_output_tensor);
        // println!(
        //     "Expected output variables: {}",
        //     expected_output_mle.num_vars()
        // );

        let point = (0..expected_output_mle.num_vars())
            .map(|_| E::random(&mut rng))
            .collect::<Vec<E>>();

        let stride = Stride::new_conv(
            stride_h,
            stride_w,
            input.shape[input.shape.len() - 2],
            input.shape[input.shape.len() - 1],
        );
        let [left_evals, right_evals]: [Vec<E>; 2] = stride.get_fixed_mles::<E>(&point)?;

        let output_eval = expected_output_mle.evaluate(&point);

        let data_len = input_tensor.data.len();
        let in_rows = input_tensor.shape[input_tensor.shape.len() - 2];
        let in_cols = input_tensor.shape[input_tensor.shape.len() - 1];

        // println!("In rows, cols: ({}, {})", in_rows, in_cols);
        // println!("In data_len: {}", data_len);
        // println!("Lt evals: {:?}", left_evals.len());
        // println!("Rt evals: {:?}", right_evals.len());

        // println!("Lt SUM: {:?}", left_evals.iter().sum::<E>());
        // println!("Rt SUM: {:?}", right_evals.iter().sum::<E>());

        let calculated_eval = (0usize..data_len).fold(E::ZERO, |acc, i| {
            // println!(
            //     "i: {} (r, c): ({}, {}), acc[i]: {:?}",
            //     i,
            //     i / in_cols,
            //     i % in_cols,
            //     left_evals[i / in_cols] * input_tensor.get_data()[i] * right_evals[i % in_cols],
            // );
            acc + (left_evals[i / in_cols] * input_tensor.get_data()[i] * right_evals[i % in_cols])
        });

        assert_eq!(calculated_eval, output_eval);
        Ok(())
    }

    #[test]
    fn test_stride_mle() -> Result<(), StrideError> {
        // let input = Tensor::<Element>::from_contiguous(vec![3, 4]);
        // let (stride_h, stride_w) = (2, 3);
        // stride_mle::<GoldilocksExt2>(input, stride_h, stride_w)?;

        let input = Tensor::<Element>::from_contiguous(vec![3, 5]);
        let (stride_h, stride_w) = (2, 3);
        stride_mle::<GoldilocksExt2>(input, stride_h, stride_w)?;

        Ok(())
    }

    fn stride_proving<E>(
        input: Tensor<Element>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(), StrideError>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let mut rng = thread_rng();

        assert!(input.shape.len() >= 2);
        let input_rows = input.shape.len() - 2;
        let input_cols = input.shape.len() - 1;

        let stride = Stride::new_conv(
            stride_h,
            stride_w,
            input.shape[input_rows],
            input.shape[input_cols],
        );

        let input_tensor: Tensor<E> = input.pad_next_power_of_two().to_fields();
        let expected_output_tensor: Tensor<E> = input
            .subsample(stride_h, stride_w)
            .pad_next_power_of_two()
            .to_fields();

        // Create a `Context` struct
        let ctx = Context::<E> {
            steps_info: vec![LayerCtx::<E>::Stride(stride)],
            weights: crate::commit::precommit::Context::<E>::default(),
            lookup: crate::lookup::context::LookupContext::default(),
        };

        let mut prover_transcript = default_transcript::<E>();
        let mut prover = Prover::new(&ctx, &mut prover_transcript);

        let output_mle = expected_output_tensor.get_data().to_vec().into_mle();
        let point = (0..output_mle.num_vars())
            .map(|_| E::random(&mut rng))
            .collect::<Vec<E>>();
        let eval = output_mle.evaluate(&point);

        let last_claim = Claim { point, eval };

        let output_claim = stride.prove_step(&mut prover, last_claim.clone(), &input_tensor)?;

        // Now run the verifier logic
        let mut verifier_transcript = default_transcript::<E>();

        let mut verifier = Verifier::new(&mut verifier_transcript);

        let stride_proof = if let LayerProof::Stride(p) = &prover.proofs[0] {
            p
        } else {
            return Err(StrideError::ProvingError(format!(
                "Proving stride somehow eneded up with a non-stride proof",
            )));
        };

        let verifier_out_claim = stride.verify_stride(&mut verifier, last_claim, stride_proof)?;

        if verifier_out_claim.eval != output_claim.eval {
            return Err(StrideError::ProvingError(format!(
                "Prover output claim {:?} and verifier output claim {:?} were not equal",
                output_claim.eval, verifier_out_claim.eval,
            )));
        }

        let input_poly_eval = input_tensor
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&verifier_out_claim.point);

        if input_poly_eval != verifier_out_claim.eval {
            return Err(StrideError::ProvingError(format!(
                "Input tensor poly evaluation {:?} and verifier output claim {:?} were not equal",
                input_poly_eval, verifier_out_claim.eval,
            )));
        }

        Ok(())
    }

    #[test]
    fn test_stride_proving() -> Result<(), StrideError> {
        // let input = Tensor::random_seed(vec![3, 4], Some(0));
        let input = Tensor::<Element>::from_contiguous(vec![3, 4]);
        stride_proving::<GoldilocksExt2>(input, 2, 3)?;
        Ok(())
    }
}
