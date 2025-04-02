//! This module contains code for padding tensor inputs to convolution and pooling layers. Currently we support padding with either zeroes or
//! reflective padding.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use crate::{
    Claim, Prover,
    commit::{compute_betas_eval, precommit::PolyID},
    iop::{
        context::ContextAux,
        split_sumcheck::{IOPSplitProverState, SplitSumcheckError},
        verifier::Verifier,
    },
    layers::LayerProof,
    tensor::Tensor,
};
use ark_std::Zero;

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
/// Enum used to distinguish between the various types of padding.
pub enum Padding {
    /// Pads with zeroes
    Zeroes {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    },
    /// Pads in a reflective manner
    Reflective {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    },
    /// Pads by replicating input boundaries
    Replication {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    },
    /// Pads by circulating through the input
    Circular {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    },
    /// Like [`Padding::Zeroes`] but  with a constant value instead of zero
    Constant {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        constant: u64,
        input_rows: usize,
        input_columns: usize,
    },
}

impl Display for Padding {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Padding::Zeroes {
                top,
                bottom,
                left,
                right,
                ..
            } => write!(f, "Zero Padding ({},{},{},{})", top, bottom, left, right),
            Padding::Reflective {
                top,
                bottom,
                left,
                right,
                ..
            } => write!(
                f,
                "Reflective Padding ({},{},{},{})",
                top, bottom, left, right
            ),
            Padding::Replication {
                top,
                bottom,
                left,
                right,
                ..
            } => write!(
                f,
                "Replication Padding ({},{},{},{})",
                top, bottom, left, right
            ),
            Padding::Circular {
                top,
                bottom,
                left,
                right,
                ..
            } => write!(
                f,
                "Circular Padding ({},{},{},{})",
                top, bottom, left, right
            ),
            Padding::Constant {
                top,
                bottom,
                left,
                right,
                constant,
                ..
            } => write!(
                f,
                "Constant ({}) Padding, ({},{},{},{})",
                constant, top, bottom, left, right
            ),
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
/// Struct containing the proof of correct padding, used along with [`Padding`] in verification.
pub struct PaddingProof<E: ExtensionField> {
    /// The split-sumcheck proof for the padding
    pub(crate) sumcheck_proof: IOPProof<E>,
}

#[derive(Debug, Clone)]
/// Enum for any errors that occur during padding
pub enum PaddingError {
    ParameterError(String),
    ProvingError(String),
}

impl Error for PaddingError {}

impl Display for PaddingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            PaddingError::ParameterError(s) => {
                write!(f, "Parameters were incorrect for padding: {}", s)
            }
            PaddingError::ProvingError(s) => {
                write!(f, "Error occured during padding proving: {}", s)
            }
        }
    }
}

impl From<SplitSumcheckError> for PaddingError {
    fn from(e: SplitSumcheckError) -> Self {
        match e {
            SplitSumcheckError::ParameterError(s) => {
                PaddingError::ProvingError(format!("Split sumcheck parameter error: {}", s))
            }
            SplitSumcheckError::ProvingError(s) => {
                PaddingError::ProvingError(format!("Split sumcheck proving error: {}", s))
            }
        }
    }
}

impl Padding {
    /// Creates a new instance of [`Padding::Zeroes`]
    pub fn new_zeroes(
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    ) -> Padding {
        Padding::Zeroes {
            top,
            bottom,
            left,
            right,
            input_rows,
            input_columns,
        }
    }

    /// Creates a new instance of [`Padding::Reflective`]
    pub fn new_reflective(
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    ) -> Padding {
        Padding::Reflective {
            top,
            bottom,
            left,
            right,
            input_rows,
            input_columns,
        }
    }

    /// Creates a new instance of [`Padding::Replication`]
    pub fn new_replication(
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    ) -> Padding {
        Padding::Replication {
            top,
            bottom,
            left,
            right,
            input_rows,
            input_columns,
        }
    }

    /// Creates a new instance of [`Padding::Circular`]
    pub fn new_circular(
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        input_rows: usize,
        input_columns: usize,
    ) -> Padding {
        Padding::Circular {
            top,
            bottom,
            left,
            right,
            input_rows,
            input_columns,
        }
    }

    /// Creates a new instance of [`Padding::Constant`]
    pub fn new_constant(
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        constant: u64,
        input_rows: usize,
        input_columns: usize,
    ) -> Padding {
        Padding::Constant {
            top,
            bottom,
            left,
            right,
            constant,
            input_rows,
            input_columns,
        }
    }

    /// Getter for the top padding size
    pub fn top(&self) -> usize {
        match self {
            Padding::Zeroes { top, .. }
            | Padding::Reflective { top, .. }
            | Padding::Replication { top, .. }
            | Padding::Circular { top, .. }
            | Padding::Constant { top, .. } => *top,
        }
    }

    /// Getter for the bottom padding size
    pub fn bottom(&self) -> usize {
        match self {
            Padding::Zeroes { bottom, .. }
            | Padding::Reflective { bottom, .. }
            | Padding::Replication { bottom, .. }
            | Padding::Circular { bottom, .. }
            | Padding::Constant { bottom, .. } => *bottom,
        }
    }

    /// Getter for the left padding size
    pub fn left(&self) -> usize {
        match self {
            Padding::Zeroes { left, .. }
            | Padding::Reflective { left, .. }
            | Padding::Replication { left, .. }
            | Padding::Circular { left, .. }
            | Padding::Constant { left, .. } => *left,
        }
    }

    /// Getter for the right padding size
    pub fn right(&self) -> usize {
        match self {
            Padding::Zeroes { right, .. }
            | Padding::Reflective { right, .. }
            | Padding::Replication { right, .. }
            | Padding::Circular { right, .. }
            | Padding::Constant { right, .. } => *right,
        }
    }

    /// Getter for the number of rows of the input
    pub fn input_rows(&self) -> usize {
        match self {
            Padding::Zeroes { input_rows, .. }
            | Padding::Reflective { input_rows, .. }
            | Padding::Replication { input_rows, .. }
            | Padding::Circular { input_rows, .. }
            | Padding::Constant { input_rows, .. } => *input_rows,
        }
    }

    /// Getter for the number of columns of the input
    pub fn input_columns(&self) -> usize {
        match self {
            Padding::Zeroes { input_columns, .. }
            | Padding::Reflective { input_columns, .. }
            | Padding::Replication { input_columns, .. }
            | Padding::Circular { input_columns, .. }
            | Padding::Constant { input_columns, .. } => *input_columns,
        }
    }

    /// Returns the number of variables the left matrix MLE has after fixing the columns
    pub fn left_variables(&self) -> usize {
        self.input_rows().next_power_of_two().ilog2() as usize
    }

    /// Returns the number of variables the right matrix MLE has after fixing the rows
    pub fn right_variables(&self) -> usize {
        self.input_columns().next_power_of_two().ilog2() as usize
    }

    /// Returns the constant value for the padding
    pub fn constant(&self) -> Option<u64> {
        if let Padding::Constant { constant, .. } = self {
            Some(*constant)
        } else {
            None
        }
    }

    /// Pads a [`Tensor`] with a padding scheme
    pub fn pad_tensor<T: Default + Clone + From<u64>>(
        &self,
        tensor: &Tensor<T>,
    ) -> Result<Tensor<T>, PaddingError> {
        // First we check the tensor has at least two dimensions
        let shape = tensor.get_shape();
        let num_dims = shape.len();
        if num_dims < 2 {
            return Err(PaddingError::ParameterError(
                "Cannot pad a tensor that has fewer than two dimensions".to_string(),
            ));
        }
        // Check the provided shape agrees with the expected number of columns and rows
        if shape[num_dims - 2] != self.input_rows() {
            return Err(PaddingError::ParameterError(format!(
                "Input tensor shape incorrect, expected {} rows but input tensor had {} rows",
                self.input_rows(),
                shape[num_dims - 2]
            )));
        }

        if shape[num_dims - 1] != self.input_columns() {
            return Err(PaddingError::ParameterError(format!(
                "Input tensor shape incorrect, expected {} columns but input tensor had {} columns",
                self.input_columns(),
                shape[num_dims - 1]
            )));
        }

        // If the shape only has two dimensions we shortcut directly to the matrix pad method
        // We work out the size of each matrix

        let chunk_size = self.input_rows() * self.input_columns();

        let padded_data = tensor
            .get_data()
            .chunks(chunk_size)
            .map(|chunk| self.pad_matrix_data(chunk))
            .collect::<Result<Vec<Vec<T>>, PaddingError>>()?
            .concat();

        let padded_shape = self.padded_shape(&shape)?;

        Ok(Tensor::<T>::new(padded_shape, padded_data))
    }

    /// Given a shape returns the shape after padding is performed
    pub fn padded_shape(&self, shape: &[usize]) -> Result<Vec<usize>, PaddingError> {
        let num_dims = shape.len();
        // Need to have at least two dimensions to pad
        if num_dims < 2 {
            return Err(PaddingError::ParameterError(
                "Cannot calculate padded shape as input shape had fewer than two dimensions"
                    .to_string(),
            ));
        }
        // Check the provided shape agrees with the expected number of columns and rows
        if shape[num_dims - 2] != self.input_rows() {
            return Err(PaddingError::ParameterError(format!(
                "Input tensor shape incorrect, expected {} rows but input tensor had {} rows",
                self.input_rows(),
                shape[num_dims - 2]
            )));
        }

        if shape[num_dims - 1] != self.input_columns() {
            return Err(PaddingError::ParameterError(format!(
                "Input tensor shape incorrect, expected {} columns but input tensor had {} columns",
                self.input_columns(),
                shape[num_dims - 1]
            )));
        }

        let new_height = self.input_columns() + self.top() + self.bottom();
        let new_width = self.input_rows() + self.left() + self.right();

        Ok(shape
            .iter()
            .copied()
            .take(num_dims - 2)
            .chain([new_height, new_width])
            .collect::<Vec<usize>>())
    }

    /// This method returns the MLEs of the matrices we multiply by on the left and right hand side with the correct number of variables fixed
    /// depending on the claim.
    pub fn get_fixed_mles<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<[Vec<E>; 2], PaddingError> {
        let columns = self.input_columns();
        let rows = self.input_rows();

        // We also check that the point is of the correct length. It should have `(top + columns + bottom).next_power_of_two().ilog2() + (left + rows + right).next_power_of_two().ilog2()` variables.
        let left_variables = (self.top() + columns + self.bottom())
            .next_power_of_two()
            .ilog2() as usize;
        let right_variables = (self.left() + rows + self.right())
            .next_power_of_two()
            .ilog2() as usize;

        if point.len() != left_variables + right_variables {
            return Err(PaddingError::ParameterError(format!(
                "Cannot fix padding polynomials, provided point has {} variables, expected: {}",
                point.len(),
                left_variables + right_variables
            )));
        }

        // Compute the left ang right beta poly evals
        let left_beta_eval = compute_betas_eval(&point[..left_variables]);
        let right_beta_eval = compute_betas_eval(&point[left_variables..]);

        match *self {
            Padding::Zeroes { top, left, .. } | Padding::Constant { top, left, .. } => {
                let left_evals = (0..rows)
                    .map(|i| left_beta_eval[top + i])
                    .chain(std::iter::repeat(E::ZERO))
                    .take(rows.next_power_of_two())
                    .collect::<Vec<E>>();
                let right_evals = (0..columns)
                    .map(|i| right_beta_eval[left + i])
                    .chain(std::iter::repeat(E::ZERO))
                    .take(columns.next_power_of_two())
                    .collect::<Vec<E>>();
                Ok([left_evals, right_evals])
            }
            Padding::Replication {
                top,
                bottom,
                left,
                right,
                ..
            } => {
                let left_evals = (0..rows)
                    .map(|i| {
                        if i == 0 {
                            left_beta_eval.iter().take(top + 1).sum::<E>()
                        } else if i == rows - 1 {
                            left_beta_eval
                                .iter()
                                .skip(top + rows - 1)
                                .take(bottom + 1)
                                .sum::<E>()
                        } else {
                            left_beta_eval[top + i]
                        }
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(rows.next_power_of_two())
                    .collect::<Vec<E>>();

                let right_evals = (0..columns)
                    .map(|i| {
                        if i == 0 {
                            right_beta_eval.iter().take(left + 1).sum::<E>()
                        } else if i == columns - 1 {
                            right_beta_eval
                                .iter()
                                .skip(left + columns - 1)
                                .take(right + 1)
                                .sum::<E>()
                        } else {
                            right_beta_eval[left + i]
                        }
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(columns.next_power_of_two())
                    .collect::<Vec<E>>();
                Ok([left_evals, right_evals])
            }
            Padding::Reflective {
                top,
                bottom,
                left,
                right,
                ..
            } => {
                if rows <= top || rows <= bottom {
                    return Err(PaddingError::ParameterError(format!(
                        "Cannot reflective pad if top: {} or bottom: {} is geq to input rows: {}",
                        top, bottom, rows
                    )));
                }

                if columns <= left || columns <= right {
                    return Err(PaddingError::ParameterError(format!(
                        "Cannot reflective pad if left: {} or right: {} is geq to input columns: {}",
                        left, right, columns
                    )));
                }

                let left_evals = (0..rows)
                    .map(|i| {
                        let top_part = left_beta_eval[..top]
                            .get(top - i)
                            .copied()
                            .unwrap_or(E::ZERO);
                        let middle_part = left_beta_eval[top + i];
                        let bottom_part = left_beta_eval
                            .iter()
                            .skip(top + rows)
                            .take(bottom)
                            .rev()
                            .copied()
                            .collect::<Vec<E>>()
                            .get(i)
                            .copied()
                            .unwrap_or(E::ZERO);
                        top_part + middle_part + bottom_part
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(rows.next_power_of_two())
                    .collect::<Vec<E>>();

                let right_evals = (0..columns)
                    .map(|i| {
                        let left_part = right_beta_eval[..left]
                            .get(left - i)
                            .copied()
                            .unwrap_or(E::ZERO);
                        let middle_part = right_beta_eval[top + i];
                        let right_part = right_beta_eval
                            .iter()
                            .skip(left + columns)
                            .take(right)
                            .rev()
                            .copied()
                            .collect::<Vec<E>>()
                            .get(i)
                            .copied()
                            .unwrap_or(E::ZERO);
                        left_part + middle_part + right_part
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(columns.next_power_of_two())
                    .collect::<Vec<E>>();

                Ok([left_evals, right_evals])
            }
            Padding::Circular {
                top,
                bottom,
                left,
                right,
                ..
            } => {
                let left_evals = (0..rows)
                    .map(|i| {
                        let initial_skip = (top + i) % rows;
                        left_beta_eval
                            .iter()
                            .take(top + rows + bottom)
                            .skip(initial_skip)
                            .step_by(rows)
                            .sum::<E>()
                    })
                    .chain(std::iter::repeat(E::ZERO))
                    .take(rows.next_power_of_two())
                    .collect::<Vec<E>>();

                let right_evals = (0..columns)
                    .map(|i| {
                        let initial_skip = (left + i) % columns;
                        right_beta_eval
                            .iter()
                            .take(left + columns + right)
                            .skip(initial_skip)
                            .step_by(columns)
                            .sum::<E>()
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
        _output: &Tensor<E>,
    ) -> Result<Claim<E>, PaddingError>
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
        let input_shape = input.get_shape();
        let middle = if input_shape.len() < 2 {
            return Err(PaddingError::ParameterError(format!(
                "Proving input had dimension {}, need at least 2 dimensions for padding",
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

        let (proof, state) =
            IOPSplitProverState::<E>::prove_split_sumcheck(left, middle, right, prover.transcript)?;
        // Clone the point so we can use it in the out claim
        let point = proof.point.clone();

        // Push the padding proof to the proof list
        prover.push_proof(LayerProof::<E>::Padding(PaddingProof::<E>::new(proof)));

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
        let layer_context = LayerCtx::<E>::Padding(*self);
        let shape = self.padded_shape(&ctx_aux.last_output_shape).unwrap();
        ctx_aux.last_output_shape = shape
            .into_iter()
            .map(|s| s.next_power_of_two())
            .collect::<Vec<usize>>();
        (layer_context, ctx_aux)
    }

    pub(crate) fn verify_padding<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &PaddingProof<E>,
    ) -> Result<Claim<E>, PaddingError> {
        let Claim::<E> { point, eval } = &last_claim;

        // If we are in the constant padding case we need to subtract the evaluation of the padding
        let claimed_sum = if let Some(constant_eval) = self.calc_constant_eval(point) {
            *eval - constant_eval
        } else {
            *eval
        };

        // Make the VPAuxInfo
        let vars = (self.input_columns().next_power_of_two()
            * self.input_rows().next_power_of_two())
        .ilog2() as usize;
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![vars, vars]]);

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
        let sc_point_len = sumcheck_point.len();
        let point = sumcheck_point
            .into_iter()
            .chain(point.iter().skip(sc_point_len).copied())
            .collect::<Vec<E>>();

        Ok(Claim {
            point,
            eval: out_eval,
        })
    }
}

impl<E: ExtensionField> PaddingProof<E> {
    /// Create a new [`PaddingProof`]
    pub fn new(sumcheck_proof: IOPProof<E>) -> PaddingProof<E> {
        PaddingProof { sumcheck_proof }
    }

    /// Getter for the sumcheck proof
    pub(crate) fn sumcheck_proof(&self) -> &IOPProof<E> {
        &self.sumcheck_proof
    }
}

// Internal methods for the `Padding` enum
impl Padding {
    /// This method takes a slice of data that is assumed to represent a matrix together with the matrix's dimensions
    /// and pads the data, returning it as vector.
    fn pad_matrix_data<T: Default + Clone + From<u64>>(
        &self,
        data: &[T],
    ) -> Result<Vec<T>, PaddingError> {
        // If the total length of data doesn't equal the dimensions provided we should error
        let expected_size = self.input_rows() * self.input_columns();
        if data.len() != expected_size {
            return Err(PaddingError::ParameterError(format!(
                "Data length: {}, did not equal expected length from shape: {}",
                data.len(),
                expected_size
            )));
        }
        // We pad rows first then columns, that we we avoid having to do a an extra transposition at the end
        let padded_rows = (0..self.input_columns())
            .map(|i| {
                let row = data
                    .iter()
                    .skip(i)
                    .step_by(self.input_columns())
                    .cloned()
                    .collect::<Vec<T>>();
                self.pad_row(&row)
            })
            .collect::<Result<Vec<Vec<T>>, PaddingError>>()?
            .concat();

        let new_row_size = self.input_rows() + self.left() + self.right();

        // Now we pad the columns
        let padded_output = (0..new_row_size)
            .map(|i| {
                let column = padded_rows
                    .iter()
                    .skip(i)
                    .step_by(new_row_size)
                    .take(self.input_columns())
                    .cloned()
                    .collect::<Vec<T>>();
                self.pad_column(&column)
            })
            .collect::<Result<Vec<Vec<T>>, PaddingError>>()?
            .concat();

        Ok(padded_output)
    }

    /// Takes a column and pads it depending on the padding variant
    fn pad_column<T: Default + Clone + From<u64>>(
        &self,
        column: &[T],
    ) -> Result<Vec<T>, PaddingError> {
        match self {
            Padding::Zeroes { top, bottom, .. } => Ok(std::iter::repeat_n(T::default(), *top)
                .chain(column.iter().cloned())
                .chain(std::iter::repeat_n(T::default(), *bottom))
                .collect::<Vec<T>>()),
            Padding::Constant {
                top,
                bottom,
                constant,
                ..
            } => Ok(std::iter::repeat_n(T::from(*constant), *top)
                .chain(column.iter().cloned())
                .chain(std::iter::repeat_n(T::from(*constant), *bottom))
                .collect::<Vec<T>>()),
            Padding::Reflective { top, bottom, .. } => {
                // We need to check that the column size is at least as long as top and bottom
                if column.len() <= *top || column.len() <= *bottom {
                    return Err(PaddingError::ParameterError(format!(
                        "Cannot Reflective Pad the column as it is too small, column length: {}, top padding: {}, bottom padding: {}",
                        column.len(),
                        top,
                        bottom
                    )));
                }

                Ok(column
                    .iter()
                    .skip(1)
                    .take(*top)
                    .rev()
                    .chain(column.iter())
                    .chain(column.iter().rev().skip(1).take(*bottom))
                    .cloned()
                    .collect::<Vec<T>>())
            }
            Padding::Replication { top, bottom, .. } => {
                // The unwrap here is safe as this method can only be accessed through `pad_tensor` which checks that we have a non-zero number of rows and columns
                let first = column.first().cloned().unwrap();
                let last = column.last().cloned().unwrap();

                Ok(std::iter::repeat_n(first, *top)
                    .chain(column.iter().cloned())
                    .chain(std::iter::repeat_n(last, *bottom))
                    .collect::<Vec<T>>())
            }

            Padding::Circular { top, bottom, .. } => {
                let column_length = column.len();
                // For the circular case we may have to repeat the entire column a few times so we work out how many full copies of the column we need first
                let top_copies = if !top.is_zero() && *top > column_length {
                    (column_length - 1) / *top
                } else {
                    0
                };
                let bottom_copies = if !bottom.is_zero() && *bottom > column_length {
                    (column_length - 1) / *bottom
                } else {
                    0
                };

                let top_skip = column_length - (*top % column_length);

                Ok(column
                    .iter()
                    .skip(top_skip)
                    .chain(std::iter::repeat_n(column, top_copies + bottom_copies + 1).flatten())
                    .chain(column.iter().take(*bottom))
                    .cloned()
                    .collect::<Vec<T>>())
            }
        }
    }

    /// Takes a column and pads it depending on the padding variant
    fn pad_row<T: Default + Clone + From<u64>>(&self, row: &[T]) -> Result<Vec<T>, PaddingError> {
        match self {
            Padding::Zeroes { left, right, .. } => Ok(std::iter::repeat_n(T::default(), *left)
                .chain(row.iter().cloned())
                .chain(std::iter::repeat_n(T::default(), *right))
                .collect::<Vec<T>>()),
            Padding::Constant {
                left,
                right,
                constant,
                ..
            } => Ok(std::iter::repeat_n(T::from(*constant), *left)
                .chain(row.iter().cloned())
                .chain(std::iter::repeat_n(T::from(*constant), *right))
                .collect::<Vec<T>>()),
            Padding::Reflective { left, right, .. } => {
                // We need to check that the column size is at least as long as top and bottom
                if row.len() <= *left || row.len() <= *right {
                    return Err(PaddingError::ParameterError(format!(
                        "Cannot Reflective Pad the column as it is too small, column length: {}, top padding: {}, bottom padding: {}",
                        row.len(),
                        left,
                        right
                    )));
                }

                Ok(row
                    .iter()
                    .skip(1)
                    .take(*left)
                    .rev()
                    .chain(row.iter())
                    .chain(row.iter().rev().skip(1).take(*right))
                    .cloned()
                    .collect::<Vec<T>>())
            }
            Padding::Replication { left, right, .. } => {
                // The unwrap here is safe as this method can only be accessed through `pad_tensor` which checks that we have a non-zero number of rows and columns
                let first = row.first().cloned().unwrap();
                let last = row.last().cloned().unwrap();

                Ok(std::iter::repeat_n(first, *left)
                    .chain(row.iter().cloned())
                    .chain(std::iter::repeat_n(last, *right))
                    .collect::<Vec<T>>())
            }
            Padding::Circular { left, right, .. } => {
                let row_length = row.len();
                // For the circular case we may have to repeat the entire column a few times so we work out how many full copies of the column we need first
                let left_copies = if !left.is_zero() && *left > row_length {
                    (row_length - 1) / *left
                } else {
                    0
                };
                let right_copies = if !right.is_zero() && *right > row_length {
                    (row_length - 1) / *right
                } else {
                    0
                };

                let left_skip = row_length - (*left % row_length);

                Ok(row
                    .iter()
                    .skip(left_skip)
                    .chain(std::iter::repeat_n(row, left_copies + right_copies + 1).flatten())
                    .chain(row.iter().take(*right))
                    .cloned()
                    .collect::<Vec<T>>())
            }
        }
    }

    /// Function used by the [`Verifier`] to compute the evaluations of the padding matrices
    fn compute_matrix_evals<E: ExtensionField>(
        &self,
        last_point: &[E],
        sumcheck_point: &[E],
    ) -> Result<Vec<E>, PaddingError> {
        let right_variables = self.input_columns().next_power_of_two().ilog2() as usize;
        let left_variables = self.input_rows().next_power_of_two().ilog2() as usize;

        if sumcheck_point.len() != left_variables + right_variables {
            return Err(PaddingError::ParameterError(format!(
                "Cannot compute padding matrix evals, provided point has {} variables, expected: {}",
                sumcheck_point.len(),
                left_variables + right_variables
            )));
        }

        let [left, right]: [Vec<E>; 2] = self.get_fixed_mles::<E>(last_point)?;

        let left_eval = left.into_mle().evaluate(&sumcheck_point[..left_variables]);

        let right_eval = right.into_mle().evaluate(&sumcheck_point[left_variables..]);

        Ok(vec![left_eval, right_eval])
    }

    /// Function used by the [`Verifier`] to compute the extra data needed in the [`Padding::Constant`] case.
    fn calc_constant_eval<E: ExtensionField>(&self, point: &[E]) -> Option<E> {
        let constu64 = if let Some(constant) = self.constant() {
            constant
        } else {
            return None;
        };

        let betas_eval = compute_betas_eval(point);
        let const_val = E::from(constu64);

        let last_non_zero_column = self.top() + self.input_columns() + self.bottom() - 1;
        let last_non_zero_row = self.left() + self.input_rows() + self.right() - 1;

        let padded_row = (last_non_zero_row + 1).next_power_of_two();
        let padded_column = (last_non_zero_column + 1).next_power_of_two();

        let eval = betas_eval
            .into_iter()
            .enumerate()
            .fold(E::ZERO, |acc, (i, val)| {
                let column = i / padded_column;
                let row = i % padded_row;
                let zero_col = column >= self.left() && column < self.left() + self.input_columns();
                let zero_val = row >= self.top() && row < self.top() + self.input_rows();
                if (zero_val & zero_col) || column > last_non_zero_column || row > last_non_zero_row
                {
                    acc
                } else {
                    acc + val
                }
            });

        Some(eval * const_val)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::thread_rng;

    use goldilocks::GoldilocksExt2;

    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{Context, Element, default_transcript, quantization::TensorFielder};

    use super::*;

    #[test]
    fn test_padding() -> Result<(), PaddingError> {
        for case in [
            TestCase::Zeroes,
            TestCase::Reflective,
            TestCase::Replication,
            TestCase::Circular,
            TestCase::Constant,
        ] {
            padding_test_helper(case)?;
        }
        Ok(())
    }

    enum TestCase {
        Zeroes,
        Reflective,
        Replication,
        Circular,
        Constant,
    }

    impl TestCase {
        fn make_padding(
            &self,
            top: usize,
            bottom: usize,
            left: usize,
            right: usize,
            input_rows: usize,
            input_columns: usize,
        ) -> Padding {
            match self {
                TestCase::Zeroes => {
                    Padding::new_zeroes(top, bottom, left, right, input_rows, input_columns)
                }
                TestCase::Reflective => {
                    Padding::new_reflective(top, bottom, left, right, input_rows, input_columns)
                }
                TestCase::Replication => {
                    Padding::new_replication(top, bottom, left, right, input_rows, input_columns)
                }
                TestCase::Circular => {
                    Padding::new_circular(top, bottom, left, right, input_rows, input_columns)
                }
                TestCase::Constant => {
                    Padding::new_constant(top, bottom, left, right, 103, input_rows, input_columns)
                }
            }
        }

        fn expected_symmetric_result(&self) -> Vec<Element> {
            match self {
                TestCase::Zeroes => vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 1, 4, 7,
                    0, 0, 0, 0, 2, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                TestCase::Constant => vec![
                    103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
                    0, 3, 6, 103, 103, 103, 103, 1, 4, 7, 103, 103, 103, 103, 2, 5, 8, 103, 103,
                    103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
                ],
                TestCase::Reflective => vec![
                    8, 5, 2, 5, 8, 5, 2, 7, 4, 1, 4, 7, 4, 1, 6, 3, 0, 3, 6, 3, 0, 7, 4, 1, 4, 7,
                    4, 1, 8, 5, 2, 5, 8, 5, 2, 7, 4, 1, 4, 7, 4, 1, 6, 3, 0, 3, 6, 3, 0,
                ],
                TestCase::Replication => vec![
                    0, 0, 0, 3, 6, 6, 6, 0, 0, 0, 3, 6, 6, 6, 0, 0, 0, 3, 6, 6, 6, 1, 1, 1, 4, 7,
                    7, 7, 2, 2, 2, 5, 8, 8, 8, 2, 2, 2, 5, 8, 8, 8, 2, 2, 2, 5, 8, 8, 8,
                ],
                TestCase::Circular => vec![
                    4, 7, 1, 4, 7, 1, 4, 5, 8, 2, 5, 8, 2, 5, 3, 6, 0, 3, 6, 0, 3, 4, 7, 1, 4, 7,
                    1, 4, 5, 8, 2, 5, 8, 2, 5, 3, 6, 0, 3, 6, 0, 3, 4, 7, 1, 4, 7, 1, 4,
                ],
            }
        }

        fn expected_asymmetric_result(&self) -> Vec<Element> {
            match self {
                TestCase::Zeroes => vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 1, 4, 7, 0, 0, 0, 2, 5, 8, 0,
                ],
                TestCase::Constant => vec![
                    103, 103, 103, 103, 103, 103, 103, 103, 0, 3, 6, 103, 103, 103, 1, 4, 7, 103,
                    103, 103, 2, 5, 8, 103,
                ],
                TestCase::Reflective => vec![
                    7, 4, 1, 4, 7, 4, 6, 3, 0, 3, 6, 3, 7, 4, 1, 4, 7, 4, 8, 5, 2, 5, 8, 5,
                ],
                TestCase::Replication => vec![
                    0, 0, 0, 3, 6, 6, 0, 0, 0, 3, 6, 6, 1, 1, 1, 4, 7, 7, 2, 2, 2, 5, 8, 8,
                ],
                TestCase::Circular => vec![
                    5, 8, 2, 5, 8, 2, 3, 6, 0, 3, 6, 0, 4, 7, 1, 4, 7, 1, 5, 8, 2, 5, 8, 2,
                ],
            }
        }

        fn name(&self) -> String {
            match self {
                TestCase::Zeroes => "Zeroes".to_string(),
                TestCase::Reflective => "Reflective".to_string(),
                TestCase::Replication => "Replication".to_string(),
                TestCase::Circular => "Circular".to_string(),
                TestCase::Constant => "Constant 103".to_string(),
            }
        }
    }

    fn padding_test_helper(case: TestCase) -> Result<(), PaddingError> {
        let input_shape = vec![3, 3];
        let input_data: Vec<Element> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];

        let padding = case.make_padding(2, 2, 2, 2, 3, 3);

        let expected_output_shape = vec![7, 7];
        let expected_output_data: Vec<Element> = case.expected_symmetric_result();
        // First we test the padding function on a single matrix
        let input_tensor = Tensor::<Element>::new(input_shape.clone(), input_data.clone());

        let output_tensor = padding.pad_tensor(&input_tensor).unwrap();

        for (i, (output, expected)) in output_tensor
            .get_data()
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            if *output != *expected {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output {} does not match expected output {} at index {} in case {} symmetric matrix padding",
                    output,
                    expected,
                    i,
                    case.name()
                )));
            }
        }

        let output_shape = output_tensor.get_shape();

        for (i, (out, expect)) in output_shape
            .iter()
            .zip(expected_output_shape.iter())
            .enumerate()
        {
            if *out != *expect {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output shape {} does not match expected output shpae {} at index {} in case {} symmetric matrix padding",
                    out,
                    expect,
                    i,
                    case.name()
                )));
            }
        }

        // Now we test it on multiple channels
        let num_channels = 4;

        let input_shape = vec![4, 3, 3];
        let input_data = std::iter::repeat_n(input_data, num_channels)
            .flatten()
            .collect::<Vec<Element>>();

        let expected_output_shape = vec![4, 7, 7];
        let expected_output_data = std::iter::repeat_n(expected_output_data, num_channels)
            .flatten()
            .collect::<Vec<Element>>();

        let input_tensor = Tensor::<Element>::new(input_shape, input_data);

        let output_tensor = padding.pad_tensor(&input_tensor)?;

        for (i, (output, expected)) in output_tensor
            .get_data()
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            if *output != *expected {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output {} does not match expected output {} at index {} in case {} multichannel padding",
                    output,
                    expected,
                    i,
                    case.name()
                )));
            }
        }

        let output_shape = output_tensor.get_shape();

        for (i, (out, expect)) in output_shape
            .iter()
            .zip(expected_output_shape.iter())
            .enumerate()
        {
            if *out != *expect {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output shape {} does not match expected output shpae {} at index {} in case {} multichannel padding",
                    out,
                    expect,
                    i,
                    case.name()
                )));
            }
        }

        // Now test uneven padding
        let input_shape = vec![3, 3];
        let input_data: Vec<Element> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];

        let padding = case.make_padding(2, 1, 1, 0, 3, 3);

        let expected_output_shape = vec![6, 4];
        let expected_output_data: Vec<Element> = case.expected_asymmetric_result();

        let input_tensor = Tensor::<Element>::new(input_shape.clone(), input_data.clone());

        let output_tensor = padding.pad_tensor(&input_tensor)?;

        for (i, (output, expected)) in output_tensor
            .get_data()
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            if *output != *expected {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output {} does not match expected output {} at index {} in case {} asymmetric padding",
                    output,
                    expected,
                    i,
                    case.name()
                )));
            }
        }

        let output_shape = output_tensor.get_shape();

        for (i, (out, expect)) in output_shape
            .iter()
            .zip(expected_output_shape.iter())
            .enumerate()
        {
            if *out != *expect {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output shape {} does not match expected output shpae {} at index {} in case {} asymmetric padding",
                    out,
                    expect,
                    i,
                    case.name()
                )));
            }
        }

        Ok(())
    }

    #[test]
    fn test_padding_mles() -> Result<(), PaddingError> {
        for case in [
            TestCase::Zeroes,
            TestCase::Reflective,
            TestCase::Replication,
            TestCase::Circular,
            TestCase::Constant,
        ] {
            test_mle_helper::<GoldilocksExt2>(case)?;
        }
        Ok(())
    }

    fn test_mle_helper<E: ExtensionField>(case: TestCase) -> Result<(), PaddingError> {
        let mut rng = thread_rng();
        let input_shape = vec![3, 3];
        let input_data: Vec<Element> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];

        let input_tensor: Tensor<E> =
            Tensor::<Element>::new(input_shape.clone(), input_data.clone())
                .pad_next_power_of_two()
                .to_fields();

        let expected_output_shape = vec![7, 7];
        let expected_output_data: Vec<Element> = case.expected_symmetric_result();

        let expected_output_tensor: Tensor<Element> =
            Tensor::<Element>::new(expected_output_shape.clone(), expected_output_data.clone())
                .pad_next_power_of_two();

        let padding = case.make_padding(2, 2, 2, 2, 3, 3);

        let expected_output_mle = expected_output_tensor.to_mle_2d::<E>();

        let point = (0..expected_output_mle.num_vars())
            .map(|_| E::random(&mut rng))
            .collect::<Vec<E>>();

        let [left_evals, right_evals]: [Vec<E>; 2] = padding.get_fixed_mles::<E>(&point)?;

        let output_eval = expected_output_mle.evaluate(&point);

        let calculated_eval = (0usize..16).fold(E::ZERO, |acc, i| {
            acc + (left_evals[i % 4] * input_tensor.get_data()[i] * right_evals[i / 4])
        });

        let calculated_eval = if let Some(const_eval) = padding.calc_constant_eval(&point) {
            calculated_eval + const_eval
        } else {
            calculated_eval
        };

        if calculated_eval != output_eval {
            Err(PaddingError::ParameterError(format!(
                "Sum of padding MLE evaluations with input: {:?} did not equal the Padded MLE evaluation: {:?} for case {}",
                calculated_eval,
                output_eval,
                case.name()
            )))
        } else {
            Ok(())
        }
    }

    #[test]
    fn test_proving() -> Result<(), PaddingError> {
        for case in [
            TestCase::Zeroes,
            TestCase::Reflective,
            TestCase::Replication,
            TestCase::Circular,
            TestCase::Constant,
        ] {
            test_proving_helper::<GoldilocksExt2>(case)?;
        }
        Ok(())
    }

    fn test_proving_helper<E>(case: TestCase) -> Result<(), PaddingError>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let mut rng = thread_rng();
        let input_shape = vec![3, 3];
        let input_data: Vec<Element> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];

        let input_tensor: Tensor<E> =
            Tensor::<Element>::new(input_shape.clone(), input_data.clone())
                .pad_next_power_of_two()
                .to_fields();

        let expected_output_shape = vec![7, 7];
        let expected_output_data: Vec<Element> = case.expected_symmetric_result();

        let expected_output_tensor: Tensor<E> =
            Tensor::<Element>::new(expected_output_shape.clone(), expected_output_data.clone())
                .pad_next_power_of_two()
                .to_fields();

        let padding = case.make_padding(2, 2, 2, 2, 3, 3);

        // Create a `Context` struct
        let ctx = Context::<E> {
            steps_info: vec![LayerCtx::<E>::Padding(padding)],
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

        let output_claim = padding.prove_step(
            &mut prover,
            last_claim.clone(),
            &input_tensor,
            &expected_output_tensor,
        )?;

        // Now run the verifier logic
        let mut verifier_transcript = default_transcript::<E>();

        let mut verifier = Verifier::new(&mut verifier_transcript);

        let padding_proof = if let LayerProof::Padding(p) = &prover.proofs[0] {
            p
        } else {
            return Err(PaddingError::ProvingError(format!(
                "Proving padding for case {} somehow eneded up with a non-padding proof",
                case.name()
            )));
        };

        let verifier_out_claim =
            padding.verify_padding(&mut verifier, last_claim, padding_proof)?;

        if verifier_out_claim.eval != output_claim.eval {
            return Err(PaddingError::ProvingError(format!(
                "Prover output claim {:?} and verifier output claim {:?} were not equal for case {}",
                output_claim.eval,
                verifier_out_claim.eval,
                case.name()
            )));
        }

        Ok(())
    }
}
