use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, ensure};

use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::debug;
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy,
    iop::{context::ContextAux, verifier::Verifier},
    layers::LayerProof,
    model::StepData,
    padding::{PaddingMode, ShapeInfo, pad_matmul},
    quantization,
    tensor::{Number, Tensor},
};

use super::{
    LayerCtx,
    provable::{
        Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
        QuantizeOutput, VerifiableCtx,
    },
    requant::Requant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Config {
    TransposeB,
}

#[derive(Clone, Debug)]
pub struct WeightMatrix<T> {
    /// The tensor storing the matrix
    pub(crate) tensor: Tensor<T>,
    /// The unpadded shape of the matrix
    unpadded_shape: Vec<usize>,
}

/// A matrix to be multiplied in the matrix multiplication layer
#[derive(Clone, Debug)]
pub enum OperandMatrix<T> {
    /// The matrix is a constant matrix specified in the model
    Weigth(WeightMatrix<T>),
    /// The matrix is input-dependent, so there is no tensor associated to it
    Input,
}

impl<T> OperandMatrix<T> {
    pub fn new_weight_matrix(matrix: Tensor<T>) -> Self {
        let unpadded_shape = matrix.get_shape();
        OperandMatrix::Weigth(WeightMatrix {
            tensor: matrix,
            unpadded_shape,
        })
    }

    pub(crate) fn is_matrix(&self) -> bool {
        match self {
            OperandMatrix::Weigth(mat) => mat.tensor.is_matrix(),
            OperandMatrix::Input => true,
        }
    }

    pub(crate) fn get_shape(&self, padding_mode: PaddingMode) -> Option<Vec<usize>> {
        match self {
            OperandMatrix::Weigth(mat) => match padding_mode {
                PaddingMode::NoPadding => Some(mat.unpadded_shape.clone()),
                PaddingMode::Padding => Some(
                    mat.tensor
                        .get_shape()
                        .into_iter()
                        .map(|dim| dim.next_power_of_two())
                        .collect(),
                ),
            },
            OperandMatrix::Input => None,
        }
    }

    pub(crate) fn get_actual_shape(&self) -> Option<Vec<usize>> {
        match self {
            OperandMatrix::Weigth(mat) => Some(mat.tensor.get_shape()),
            OperandMatrix::Input => None,
        }
    }

    pub(crate) fn nrows(&self) -> Option<usize> {
        match self {
            OperandMatrix::Weigth(mat) => Some(mat.tensor.nrows_2d()),
            OperandMatrix::Input => None,
        }
    }

    pub(crate) fn ncols(&self) -> Option<usize> {
        match self {
            OperandMatrix::Weigth(mat) => Some(mat.tensor.ncols_2d()),
            OperandMatrix::Input => None,
        }
    }

    pub(crate) fn pad_next_power_of_two(self) -> Self
    where
        T: Number,
    {
        match self {
            OperandMatrix::Weigth(mat) => OperandMatrix::Weigth(WeightMatrix {
                tensor: mat.tensor.pad_next_power_of_two(),
                unpadded_shape: mat.unpadded_shape,
            }),
            OperandMatrix::Input => OperandMatrix::Input,
        }
    }
}

/// Description of the layer
#[derive(Clone, Debug)]
pub struct MatMul<T> {
    pub(crate) left_matrix: OperandMatrix<T>,
    pub(crate) right_matrix: OperandMatrix<T>,
    pub(crate) config: Option<Config>,
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatMulCtx<E> {
    pub(crate) node_id: NodeId,
    pub(crate) matrix_poly_aux: VPAuxInfo<E>,
    // Number of variables of the MLE polynomial for each dimension of the output matrix
    pub(crate) output_mle_num_vars: (usize, usize),
    /// Unpadded and padded shapes of the left matrix, if the left matrx is a constant matrix
    pub(crate) left_matrix_shapes: Option<(Vec<usize>, Vec<usize>)>,
    /// Unpadded and padded shapes of the right matrix, if the right matrx is a constant matrix
    pub(crate) right_matrix_shapes: Option<(Vec<usize>, Vec<usize>)>,
    pub(crate) config: Option<Config>,
}

/// Proof of the layer.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MatMulProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the matmul protocol
    pub(crate) sumcheck: IOPProof<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly".
    /// Since we only support quadratic right now it's a flat list.
    individual_claims: Vec<E>,
}

impl<T> MatMul<T> {
    pub fn new(left_matrix: OperandMatrix<T>, right_matrix: OperandMatrix<T>) -> Result<Self> {
        Self::new_internal(left_matrix, right_matrix, None)
    }

    fn new_internal(
        left_matrix: OperandMatrix<T>,
        right_matrix: OperandMatrix<T>,
        config: Option<Config>,
    ) -> Result<Self> {
        ensure!(
            left_matrix.is_matrix(),
            "left matrix for MatMul layer is not a matrix"
        );
        ensure!(
            right_matrix.is_matrix(),
            "right matrix for MatMul layer is not a matrix"
        );
        if let (Some(left_cols), Some(right_rows)) = (left_matrix.ncols(), right_matrix.nrows()) {
            ensure!(
                left_cols == right_rows,
                "Number of columns in left matrix different from number of rows of right matrix: {} != {}",
                left_cols,
                right_rows,
            );
        }
        // check that we don't have 2 weight matrix being multiplied
        match (&left_matrix, &right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) =>
                Err(anyhow!("Pointless to have a layer with 2 constant matrices, just use the product as a parameter in 
                another layer"))?,
            _ => (), // all other configurations are allowed
        }
        Ok(Self {
            left_matrix,
            right_matrix,
            config,
        })
    }

    pub fn new_with_config(
        left_matrix: OperandMatrix<T>,
        right_matrix: OperandMatrix<T>,
        config: Config,
    ) -> Result<Self> {
        Self::new_internal(left_matrix, right_matrix, Some(config))
    }

    pub fn op(&self, inputs: Vec<&Tensor<T>>) -> Result<Tensor<T>>
    where
        T: Number,
    {
        Ok(match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => panic!(
                "Found layer with 2 constant matrices, which is useless as the 
                product can be directly used instead"
            ),
            (OperandMatrix::Weigth(mat), OperandMatrix::Input) => {
                let right_matrix = inputs
                    .first()
                    .ok_or(anyhow!("No matrix provided as input to MatMul"))?;
                let transposed_matrix = if let Some(Config::TransposeB) = self.config {
                    Some(right_matrix.transpose())
                } else {
                    None
                };
                let nrows = transposed_matrix
                    .as_ref()
                    .unwrap_or(right_matrix)
                    .nrows_2d();
                ensure!(
                    nrows == mat.tensor.ncols_2d(),
                    "Incompatible shape found for input matrix: expected {:?} rows, found {:?}",
                    mat.tensor.ncols_2d(),
                    nrows,
                );
                mat.tensor
                    .matmul(transposed_matrix.as_ref().unwrap_or(&right_matrix))
            }
            (OperandMatrix::Input, OperandMatrix::Weigth(mat)) => {
                let left_matrix = inputs
                    .first()
                    .ok_or(anyhow!("No matrix provided as input to MatMul"))?;
                let transposed_matrix = if let Some(Config::TransposeB) = self.config {
                    Some(mat.tensor.transpose())
                } else {
                    None
                };
                let nrows = transposed_matrix.as_ref().unwrap_or(&mat.tensor).nrows_2d();
                ensure!(
                    left_matrix.ncols_2d() == nrows,
                    "Incompatible shape found for input matrix: expected {:?} columns, found {:?}",
                    nrows,
                    left_matrix.ncols_2d(),
                );
                left_matrix.matmul(transposed_matrix.as_ref().unwrap_or(&mat.tensor))
            }
            (OperandMatrix::Input, OperandMatrix::Input) => {
                ensure!(
                    inputs.len() == 2,
                    "Not enough inputs provided to MatMul: expected 2, found {}",
                    inputs.len()
                );
                let transposed_matrix = if let Some(Config::TransposeB) = self.config {
                    Some(inputs[1].transpose())
                } else {
                    None
                };
                let nrows = transposed_matrix.as_ref().unwrap_or(inputs[1]).nrows_2d();
                ensure!(
                    inputs[0].ncols_2d() == nrows,
                    "Incompatible shape found for input matrices: left matrix has {} columns, right matrix has {} rows",
                    inputs[0].ncols_2d(),
                    nrows,
                );
                inputs[0].matmul(transposed_matrix.as_ref().unwrap_or(inputs[1]))
            }
        })
    }

    pub fn is_right_transposed(&self) -> bool {
        matches!(self.config, Some(Config::TransposeB))
    }

    pub fn pad_next_power_of_two(self) -> Result<Self>
    where
        T: Number,
    {
        let left_matrix = self.left_matrix.pad_next_power_of_two();
        let right_matrix = self.right_matrix.pad_next_power_of_two();
        Self::new(left_matrix, right_matrix)
    }

    pub(crate) fn num_inputs(&self) -> usize {
        match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => 0,
            (OperandMatrix::Weigth(_), OperandMatrix::Input) => 1,
            (OperandMatrix::Input, OperandMatrix::Weigth(_)) => 1,
            (OperandMatrix::Input, OperandMatrix::Input) => 2,
        }
    }

    /// Method to split the point of a claim computed for the output matrix MLE among the coordinates
    /// for the left matrix and for the right matrix, which are returned as output.
    /// `output_num_vars` specifies the number of variables for each dimension of the output matrix
    fn split_claim<E: ExtensionField>(
        claim: &Claim<E>,
        output_num_vars: (usize, usize),
    ) -> (&[E], &[E]) {
        let num_vars_cols = output_num_vars.1;
        // the coordinates of `last_claim` point employed to partially evaluate the
        // left matrix MLE are the ones corresponding to the rows of the output matrix;
        // therefore, these correspond to the high variables because  the MLE is addressing
        // in little endian so (rows,cols) is actually given in (cols, rows)
        let point_for_left = &claim.point[num_vars_cols..];
        // the coordinates of `last_claim` point employed to partially evaluate the
        // right matrix MLE are the ones corresponding to the columns of the output matrix;
        // therefore, these correspond to the low variables because  the MLE is addressing
        // in little endian so (rows,cols) is actually given in (cols, rows)
        let point_for_right = &claim.point[..num_vars_cols];

        (point_for_left, point_for_right)
    }

    /// Construct the full point (i.e., with all the variables) over which the left matrix and the
    /// right matrix are evaluated in the sumcheck proof. This method requires the following inputs:
    /// - `claim`: claim computed for the output matrix MLE (input claim for the sumcheck)
    /// - `proof_point`: point employed in the sumcheck proof
    /// - `output_num_vars`: number of variables for each dimension of the output matrix
    /// - `is_right_transposed`: flag specifying whether the right matrix is transposed or not
    fn full_points<E: ExtensionField>(
        claim: &Claim<E>,
        proof_point: &[E],
        output_num_vars: (usize, usize),
        is_right_transposed: bool,
    ) -> (Vec<E>, Vec<E>) {
        let (claim_point_for_left, claim_point_for_right) =
            Self::split_claim(claim, output_num_vars);
        let point_for_right = if is_right_transposed {
            // if right matrix was transposed, we left the column variables
            // free in the sum-check, so sum-check point should correspond
            // to column variables (i.e., the low ones)
            [proof_point, claim_point_for_right]
        } else {
            [claim_point_for_right, proof_point]
        }
        .concat();
        let point_for_left = [proof_point, claim_point_for_left].concat();
        (point_for_left, point_for_right)
    }

    fn num_outputs(num_inputs: usize) -> usize {
        assert!(num_inputs < 3, "MatMul layer should have at most 2 inputs");
        1
    }
}

/// Helper method to compute output shapes for MatMul layer. It requires as input:
/// - `input_shapes`: the shapes of the input tensors
/// - `left_matrix_shape`: the shape of the left matrix, if it is a constant matrix
/// - `right_matrix_shape`: the shape of the right matrix, if it is a constant matrix
fn compute_output_shapes(
    input_shapes: &[Vec<usize>],
    left_matrix_shape: Option<&Vec<usize>>,
    right_matrix_shape: Option<&Vec<usize>>,
    config: &Option<Config>,
) -> Vec<Vec<usize>> {
    let (left_shape, right_shape) = match (left_matrix_shape, right_matrix_shape) {
        (None, None) => {
            assert_eq!(
                input_shapes.len(),
                2,
                "Expected 2 inputs for MatMul layer found {}",
                input_shapes.len()
            );
            (&input_shapes[0], &input_shapes[1])
        }
        (None, Some(shape)) => {
            assert_eq!(
                input_shapes.len(),
                1,
                "Expected 1 input for MatMul layer found {}",
                input_shapes.len()
            );
            (&input_shapes[0], shape)
        }
        (Some(shape), None) => {
            assert_eq!(
                input_shapes.len(),
                1,
                "Expected 1 input for MatMul layer found {}",
                input_shapes.len()
            );
            (shape, &input_shapes[0])
        }
        (Some(_), Some(_)) => unreachable!("Both matrices are constant, this should not happen"),
    };
    let right_shape = if let Some(Config::TransposeB) = config {
        right_shape.iter().rev().copied().collect()
    } else {
        right_shape.clone()
    };
    assert_eq!(
        left_shape[1], right_shape[0],
        "Incompatible shapes for MatMul layer: left matrix has {} columns, right matrix has {} rows",
        left_shape[1], right_shape[0],
    );
    vec![vec![left_shape[0], right_shape[1]]]
}
const IS_PROVABLE: bool = true;

impl<N: Number> OpInfo for MatMul<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        let left_matrix_shape = self.left_matrix.get_shape(padding_mode);
        let right_matrix_shape = self.right_matrix.get_shape(padding_mode);
        compute_output_shapes(
            input_shapes,
            left_matrix_shape.as_ref(),
            right_matrix_shape.as_ref(),
            &self.config,
        )
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Self::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Matrix multiplication: left = {:?}, right = {:?}",
            self.left_matrix.get_actual_shape(),
            self.right_matrix
                .get_actual_shape()
                .map(|shape| if self.is_right_transposed() {
                    shape.iter().rev().copied().collect()
                } else {
                    shape.clone()
                }),
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<N: Number> Evaluate<N> for MatMul<N> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<N, E>> {
        let output = self.op(inputs.to_vec())?;
        Ok(LayerOut::from_vec(vec![output]))
    }
}

impl<E> ProveInfo<E> for MatMul<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut ctx_aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        let info = self.ctx(id, &mut ctx_aux)?;

        // there is only one product (i.e. quadratic sumcheck)
        let info = LayerCtx::MatMul(info);

        Ok((info, ctx_aux))
    }
}

impl MatMul<f32> {
    pub fn quantize(
        self,
        left_scaling: &ScalingFactor,
        right_scaling: &ScalingFactor,
    ) -> MatMul<Element> {
        let left_matrix = match self.left_matrix {
            OperandMatrix::Weigth(mat) => OperandMatrix::Weigth(WeightMatrix {
                tensor: mat.tensor.quantize(left_scaling),
                unpadded_shape: mat.unpadded_shape,
            }),
            OperandMatrix::Input => OperandMatrix::Input, /* No need to quantize since it's an input, not a constant in the model */
        };
        let right_matrix = match self.right_matrix {
            OperandMatrix::Weigth(mat) => OperandMatrix::Weigth(WeightMatrix {
                tensor: mat.tensor.quantize(right_scaling),
                unpadded_shape: mat.unpadded_shape,
            }),
            OperandMatrix::Input => OperandMatrix::Input, /* No need to quantize since it's an input, not a constant in the model */
        };
        MatMul {
            left_matrix,
            right_matrix,
            config: self.config,
        }
    }

    // Quantize a mat mul layer using scaling factor of input and output
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<MatMul<Element>>> {
        let (left_matrix_scaling, right_matrix_scaling) =
            match (&self.left_matrix, &self.right_matrix) {
                (OperandMatrix::Weigth(mat), OperandMatrix::Input) => {
                    ensure!(
                        input_scaling.len() == 1,
                        "Expected 1 input scaling factor for MatMul layer, found {}",
                        input_scaling.len(),
                    );
                    (
                        ScalingFactor::from_absolute_max(mat.tensor.max_abs_output(), None),
                        input_scaling[0].clone(),
                    )
                }
                (OperandMatrix::Input, OperandMatrix::Weigth(mat)) => {
                    ensure!(
                        input_scaling.len() == 1,
                        "Expected 1 input scaling factor for MatMul layer, found {}",
                        input_scaling.len(),
                    );
                    (
                        input_scaling[0].clone(),
                        ScalingFactor::from_absolute_max(mat.tensor.max_abs_output(), None),
                    )
                }
                (OperandMatrix::Input, OperandMatrix::Input) => {
                    ensure!(
                        input_scaling.len() == 2,
                        "Expected 2 input scaling factors for MatMul layer, found {}",
                        input_scaling.len(),
                    );
                    (input_scaling[0].clone(), input_scaling[1].clone())
                }
                (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => Err(anyhow!(
                    "Trying to quantize a layer with 2 constant matrices"
                ))?,
            };
        let multiplier = left_matrix_scaling.m(&right_matrix_scaling, &output_scaling);
        let quantized = self.quantize(&left_matrix_scaling, &right_matrix_scaling);
        let output_bitsize = quantized.output_bitsize(*quantization::MIN, *quantization::MAX);
        let requant = Requant::from_multiplier(multiplier, output_bitsize);

        Ok(QuantizeOutput {
            quanzited_op: quantized,
            output_scalings: vec![output_scaling],
            requant_layer: Some(requant),
        })
    }
}

impl QuantizeOp for MatMul<f32> {
    type QuantizedOp = MatMul<Element>;

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

impl PadOp for MatMul<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_matmul(self, si)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for MatMul<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = MatMulCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        Ok(self.prove_step(
            node_id,
            prover,
            last_claims[0],
            step_data.inputs.iter().collect(),
            step_data.outputs.outputs()[0],
        )?)
    }
}

const MAX_BITS: u32 = 30;

const MATRIX_POLY_ID: &str = "MatMulWeight";

impl MatMul<Element> {
    /// Returns the maximum bit size of the output, given the provided bounds on the inputs
    pub fn output_bitsize(&self, min_input: Element, max_input: Element) -> usize {
        // Get either the number of columns of the left matrix or the number of rows of the right matrix,
        // which corresponds to the number of additions performed in a matrix multiplication.
        // If neither of the 2 matrices is constant in the model, then this number
        // will be undefined, and in this case the default value of MAX_BITS is used.
        let ncols = self.left_matrix.ncols().or(if self.is_right_transposed() {
            self.right_matrix.ncols()
        } else {
            self.right_matrix.nrows()
        });
        ncols
            .map(|ncols| {
                // Number of addition is defined, so we return the number of bits as
                // min_bits + max_bits + log(ncols) + 1, where `min_bit` and `max_bit` are the
                // number of inputs necessary to represent `min_input` and `max_input`, respectively.
                let min_bits = min_input.abs().ilog2() + 1;
                let max_bits = max_input.abs().ilog2() + 1;
                min_bits + max_bits + ncols.ilog2() + 1
            })
            .unwrap_or(MAX_BITS) as usize
    }

    // Return evaluations for the constant matrix employed in the layer.
    // If there is no constant matrix in the layer, `None` is returned
    pub(crate) fn eval_constant_matrix(&self) -> Option<Vec<Element>> {
        match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => panic!(
                "Found layer with 2 constant matrices, which is useless as the 
                product can be directly used instead"
            ),
            (OperandMatrix::Weigth(mat), OperandMatrix::Input) => {
                Some(mat.tensor.pad_next_power_of_two().data)
            }
            (OperandMatrix::Input, OperandMatrix::Weigth(mat)) => {
                Some(mat.tensor.pad_next_power_of_two().data)
            }
            (OperandMatrix::Input, OperandMatrix::Input) => None,
        }
    }

    /// Prove the layer
    pub fn prove_step<E, T, PCS>(
        &self,
        node_id: NodeId,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        mut inputs: Vec<&Tensor<E>>,
        output: &Tensor<E>,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let num_inputs = inputs.len();
        let (right_matrix, is_right_constant) = match &self.right_matrix {
            OperandMatrix::Weigth(mat) => (&Tensor::<E>::from(&mat.tensor), true),
            OperandMatrix::Input => {
                let matrix = inputs
                    .pop()
                    .ok_or(anyhow!("No input provided for right matrix"))?;
                (matrix, false)
            }
        };
        let transposed = self.is_right_transposed();
        let (left_matrix, is_left_constant) = match &self.left_matrix {
            OperandMatrix::Weigth(mat) => (&Tensor::<E>::from(&mat.tensor), true),
            OperandMatrix::Input => {
                let matrix = inputs
                    .pop()
                    .ok_or(anyhow!("No input provided for left matrix"))?;
                (matrix, false)
            }
        };
        let expected_num_inputs = if is_left_constant || is_right_constant {
            1
        } else {
            2
        };
        ensure!(
            inputs.is_empty(),
            "More inputs provided than necessary: expected {expected_num_inputs}, found {num_inputs}"
        );
        ensure!(
            left_matrix.is_matrix(),
            "left input matrix for MatMul layer is not a matrix"
        );
        ensure!(
            right_matrix.is_matrix(),
            "right input matrix for MatMul layer is not a matrix"
        );
        let nrows_left = left_matrix.nrows_2d();
        let ncols_right = if transposed {
            right_matrix.nrows_2d()
        } else {
            right_matrix.ncols_2d()
        };
        ensure!(
            output.is_matrix(),
            "Output tensor for MatMul layer is not a matrix"
        );
        let (nrows_out, ncols_out) = (output.nrows_2d(), output.ncols_2d());
        ensure!(
            nrows_out == nrows_left,
            "Wrong number of rows in output matrix: expected {}, found {}",
            nrows_left,
            nrows_out,
        );
        ensure!(
            ncols_out == ncols_right,
            "Wrong number of columns in output matrix: expected {}, found {}",
            ncols_right,
            ncols_out,
        );
        let num_vars_2d = output.num_vars_2d();
        let num_vars_out = num_vars_2d.0 + num_vars_2d.1;
        ensure!(
            num_vars_out == last_claim.point.len(),
            "Wrong length of last claim point: expected {}, found {}",
            num_vars_out,
            last_claim.point.len()
        );

        // construct the MLE combining the input and the matrix
        let mut right_mat_mle: DenseMultilinearExtension<E> = right_matrix.to_mle_2d();
        let mut left_mat_mle = left_matrix.to_mle_2d();
        let (point_for_left, point_for_right) = Self::split_claim(&last_claim, num_vars_2d);
        // fix the variables for the left matrix; we need to fix the variables
        // corresponding to a row, so we must fix the HIGH variables
        left_mat_mle.fix_high_variables_in_place(point_for_left);
        if transposed {
            // fix the variables for the right matrix; since it is transposed, we need to
            // fix the variables corresponding to a row, so we must fix the high variables
            right_mat_mle.fix_high_variables_in_place(point_for_right);
        } else {
            // fix the variables for the right matrix; we need to fix the variables
            // corresponding to a column, so we must fix the low variables
            right_mat_mle.fix_variables_in_place(point_for_right);
        }

        // check that after fixing the variables in both matrices the number of free
        // variables is the same
        assert_eq!(left_mat_mle.num_vars(), right_mat_mle.num_vars());

        let num_vars = left_mat_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        vp.add_mle_list(vec![left_mat_mle.into(), right_mat_mle.into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // PCS part: here we need to create an opening proof for the final evaluation of the polynomial for
        // the matrix with no input-dependent values (if any)
        // first, check that there is at most one constant matrix
        ensure!(
            !(is_left_constant && is_right_constant),
            "No need to have a layer to multiply 2 constant matrices, define a layer with the matrix product instead"
        );
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let (point_for_left, point_for_right) =
            Self::full_points(&last_claim, &proof.point, num_vars_2d, transposed);
        // collection of claims to be returned as output
        let mut output_claims = vec![];
        // claims to be bound to a committed polynomial via opening proof
        let mut common_claims = HashMap::new();
        // compute the claim for the left matrix polynomial. It will be either accumulated in the
        // evaluation claims being opened with the polynomial commitment, or returned as output,
        // depending on whether the left matrix is constant or not
        let eval = state.get_mle_final_evaluations()[0]; // The first MLE being evaluated is the left matrix poly
        let left_claim = Claim::new(point_for_left, eval);
        if is_left_constant {
            // add a claim for the constant polynomial of the left matrix
            common_claims.insert(MATRIX_POLY_ID.to_string(), left_claim);
        } else {
            // append the claim to output claims
            output_claims.push(left_claim);
        }
        // same for right matrix polynomial: compute the claim and either accumulated it in the evaluation
        // claims opened with the polynomial commitment, or return it as output
        let eval = state.get_mle_final_evaluations()[1]; // The second MLE being evaluated is the right matrix poly
        let right_claim = Claim::new(point_for_right, eval);
        if is_right_constant {
            // add a claim for the constant polynomial of the left matrix
            common_claims.insert(MATRIX_POLY_ID.to_string(), right_claim);
        } else {
            // append the claim to output claims
            output_claims.push(right_claim);
        }

        prover
            .add_common_claims(node_id, common_claims)
            .context("unable to add weight matrix claims")?;

        let proof = MatMulProof {
            sumcheck: proof,
            individual_claims: state.get_mle_final_evaluations(),
        };

        prover.push_proof(node_id, LayerProof::MatMul(proof));
        Ok(output_claims)
    }

    fn ctx<E: ExtensionField>(&self, id: NodeId, ctx_aux: &mut ContextAux) -> Result<MatMulCtx<E>>
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let (left_shape, right_shape) = match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(mat), OperandMatrix::Input) => {
                (mat.tensor.get_shape(), ctx_aux.last_output_shape[0].clone())
            }
            (OperandMatrix::Input, OperandMatrix::Weigth(mat)) => {
                (ctx_aux.last_output_shape[0].clone(), mat.tensor.get_shape())
            }
            (OperandMatrix::Input, OperandMatrix::Input) => (
                ctx_aux.last_output_shape[0].clone(),
                ctx_aux.last_output_shape[1].clone(),
            ),
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => {
                unreachable!("Found Matmul layer with 2 constant matrices, which is useless")
            }
        };
        // construct dimension of the polynomial given to the sumcheck
        let transposed_right_shape = if self.is_right_transposed() {
            Some(right_shape.iter().rev().copied().collect_vec())
        } else {
            None
        };
        let (nrows, ncols) = (
            left_shape[0],
            transposed_right_shape.as_ref().unwrap_or(&right_shape)[1],
        );
        ctx_aux.last_output_shape = vec![vec![nrows, ncols]];

        // number of variables of the MLE polynomials is the number of row
        // variables in in layer matrix
        let num_vars = Tensor::<Element>::new_from_shape(
            transposed_right_shape.unwrap_or(right_shape.clone()),
        )
        .num_vars_2d()
        .0;
        // check that the number of variables is the same as the number of
        // column variables for left matrix
        ensure!(
            num_vars
                == Tensor::<Element>::new_from_shape(left_shape.clone())
                    .num_vars_2d()
                    .1
        );

        let left_matrix_shapes = self
            .left_matrix
            .get_shape(PaddingMode::NoPadding)
            .map(|unpadded_shape| (unpadded_shape, left_shape));
        let right_matrix_shapes = self
            .right_matrix
            .get_shape(PaddingMode::NoPadding)
            .map(|unpadded_shape| (unpadded_shape, right_shape));

        // there is only one product (i.e. quadratic sumcheck)
        let info = MatMulCtx {
            node_id: id,
            matrix_poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                num_vars, num_vars,
            ]]),
            output_mle_num_vars: (nrows.ilog2() as usize, ncols.ilog2() as usize),
            left_matrix_shapes,
            right_matrix_shapes,
            config: self.config.clone(),
        };

        ctx_aux.model_polys = self.eval_constant_matrix().map(|evals| {
            debug!(
                "Commitment : mat mul layer ID {}: size {}",
                id,
                evals.len().ilog2()
            );
            let mut model_polys = HashMap::new();
            model_polys.insert(MATRIX_POLY_ID.to_string(), evals);
            model_polys
        });

        Ok(info)
    }
}

impl<E: ExtensionField> OpInfo for MatMulCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        let left_matrix_shape = self
            .left_matrix_shapes
            .as_ref()
            .map(|s| match padding_mode {
                PaddingMode::NoPadding => &s.0,
                PaddingMode::Padding => &s.1,
            });
        let right_matrix_shape = self
            .right_matrix_shapes
            .as_ref()
            .map(|s| match padding_mode {
                PaddingMode::NoPadding => &s.0,
                PaddingMode::Padding => &s.1,
            });
        compute_output_shapes(
            input_shapes,
            left_matrix_shape,
            right_matrix_shape,
            &self.config,
        )
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        MatMul::<Element>::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Matrix multiplication ctx: left = {:?}, right = {:?}",
            self.left_matrix_shapes.as_ref().map(|s| s.1.clone()),
            self.right_matrix_shapes
                .as_ref()
                .map(|s| if self.is_right_transposed() {
                    s.1.iter().rev().copied().collect()
                } else {
                    s.1.clone()
                }),
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for MatMulCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = MatMulProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &crate::iop::context::ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(self.verify_matmul(verifier, last_claims[0], proof)?)
    }
}

impl<E: ExtensionField> MatMulCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn is_right_transposed(&self) -> bool {
        matches!(self.config, Some(Config::TransposeB))
    }

    pub(crate) fn verify_matmul<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &MatMulProof<E>,
    ) -> Result<Vec<Claim<E>>> {
        let subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.sumcheck,
            &self.matrix_poly_aux,
            verifier.transcript,
        );
        let is_left_matrix_constant = self.left_matrix_shapes.is_some();
        let is_right_matrix_constant = self.right_matrix_shapes.is_some();

        // Verify claims about the matrix polynomials, for the constant input matrix (if any),
        // while claims about non-constant matrices are returned as output to be verified in
        // the next layer
        let mut output_claims = vec![];
        // claims to be verified with opening proofs
        let mut common_claims = HashMap::new();
        // check that there is at most 1 constant matrix
        ensure!(
            !(is_left_matrix_constant && is_right_matrix_constant),
            "Cannot have a MatMul layer with both constant matrices as input"
        );
        let transposed = self.is_right_transposed();
        let (point_for_left, point_for_right) = MatMul::<Element>::full_points(
            &last_claim,
            &subclaim.point_flat(),
            self.output_mle_num_vars,
            transposed,
        );
        // 0 because left matrix comes first in the product
        let eval_left = proof.individual_claims[0];
        let left_claim = Claim::new(point_for_left, eval_left);
        if is_left_matrix_constant {
            // we need to verify the polynomial commitment opening
            common_claims.insert(MATRIX_POLY_ID.to_string(), left_claim);
        } else {
            // add the claim to the output claims, to be verified in the next layer
            output_claims.push(left_claim)
        }
        // same for right matrix polynomial
        let eval_right = proof.individual_claims[1];
        let right_claim = Claim::new(point_for_right, eval_right);
        if is_right_matrix_constant {
            // we need to verify the polynomial commitment opening
            common_claims.insert(MATRIX_POLY_ID.to_string(), right_claim);
        } else {
            // add the claim to the output claims, to be verified in the next layer
            output_claims.push(right_claim)
        }

        verifier.add_common_claims(self.node_id, common_claims)?;

        // SUMCHECK verification part
        // Instead of computing the polynomial at the random point requested like this
        // let computed_point = vp.evaluate(
        //     subclaim
        //         .point
        //         .iter()
        //         .map(|c| c.elements)
        //         .collect_vec()
        //         .as_ref(),
        //
        // We compute the evaluation directly from the individual final evaluations of each polynomial
        // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
        ensure!(
            proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
            "sumcheck claim failed",
        );

        // the output claim for this step that is going to be verified at next step
        Ok(output_claims)
    }
}

impl<E: ExtensionField> MatMulProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;

    use crate::{
        Element, ScalingFactor,
        layers::{
            Layer,
            matrix_mul::{Config, MatMul, OperandMatrix},
            provable::Evaluate,
        },
        model::{Model, test::prove_model},
        padding::PaddingMode,
        tensor::Tensor,
    };

    fn test_matmul_padding(transpose: bool) {
        // Create a Mat mul layer with non-power-of-two dimensions
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .unwrap();

        let layer = if transpose {
            MatMul::new_with_config(
                OperandMatrix::Input,
                OperandMatrix::new_weight_matrix(matrix),
                Config::TransposeB,
            )
            .unwrap()
        } else {
            MatMul::new(
                OperandMatrix::Input,
                OperandMatrix::new_weight_matrix(matrix),
            )
            .unwrap()
        };

        // Pad to next power of two
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Check padded dimensions are powers of two
        let padded_dims = padded.right_matrix.get_actual_shape().unwrap();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

        // Check padded right matrix has the padded shape of right matrix in layer
        assert_eq!(
            padded_dims,
            layer.right_matrix.get_shape(PaddingMode::Padding).unwrap(),
        );

        // Check there is no shape for left matrix, since it's an input one
        assert!(padded.left_matrix.get_actual_shape().is_none());

        // Check original values are preserved
        let padded_matrix = if let OperandMatrix::Weigth(matrix) = &padded.right_matrix {
            &matrix.tensor
        } else {
            unreachable!()
        };
        assert_eq!(padded_matrix.get_data()[0], 1);
        assert_eq!(padded_matrix.get_data()[1], 2);
        assert_eq!(padded_matrix.get_data()[2], 3);
        assert_eq!(padded_matrix.get_data()[4], 4);
        assert_eq!(padded_matrix.get_data()[8], 7);

        // Check added values are zeros
        assert_eq!(padded_matrix.get_data()[3], 0);
        assert_eq!(padded_matrix.get_data()[7], 0);
        assert_eq!(padded_matrix.get_data()[15], 0);
    }

    #[test]
    fn test_matmul_pad_next_power_of_two() {
        test_matmul_padding(false);
    }

    #[test]
    fn test_matmul_pad_already_power_of_two() {
        // Create a Dense layer with power-of-two dimensions
        let matrix = Tensor::<Element>::matix_from_coeffs(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ])
        .unwrap();
        let layer = MatMul::new(
            OperandMatrix::new_weight_matrix(matrix.clone()),
            OperandMatrix::Input,
        )
        .unwrap();

        // Pad to next power of two
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Check dimensions remain the same
        assert_eq!(
            matrix.get_shape(),
            padded.left_matrix.get_actual_shape().unwrap()
        );

        // Check right matrix has no shape, since it's an input one
        assert!(padded.right_matrix.get_actual_shape().is_none());

        // Check values are preserved
        let padded_matrix = if let OperandMatrix::Weigth(matrix) = &padded.left_matrix {
            &matrix.tensor
        } else {
            unreachable!()
        };
        let left_matrix = if let OperandMatrix::Weigth(matrix) = &layer.left_matrix {
            &matrix.tensor
        } else {
            unreachable!()
        };
        assert_eq!(padded_matrix.get_data(), left_matrix.get_data());
    }

    #[test]
    fn test_matmul_pad_mixed_dimensions() {
        // Create a Dense layer with one power-of-two dimension and one non-power-of-two
        let matrix = Tensor::<Element>::matix_from_coeffs(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
        ])
        .unwrap();

        let layer = MatMul::new(
            OperandMatrix::Input,
            OperandMatrix::new_weight_matrix(matrix),
        )
        .unwrap();

        // Pad to next power of two
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Check dimensions are padded correctly
        let padded_dims = padded.right_matrix.get_actual_shape().unwrap();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Already a power of 2

        // Check padded right matrix has the padded shape of right matrix in layer
        assert_eq!(
            padded_dims,
            layer.right_matrix.get_shape(PaddingMode::Padding).unwrap(),
        );

        // Check left matrix has no shape, since it's an input one
        assert!(padded.left_matrix.get_actual_shape().is_none());

        // Check original values are preserved and padding is zeros
        let padded_matrix = if let OperandMatrix::Weigth(matrix) = &padded.right_matrix {
            &matrix.tensor
        } else {
            unreachable!()
        };
        assert_eq!(padded_matrix.get_data()[0], 1);
        assert_eq!(padded_matrix.get_data()[4], 5);
        assert_eq!(padded_matrix.get_data()[8], 9);
        assert_eq!(padded_matrix.get_data()[12], 0); // Padding
    }

    #[test]
    fn test_matmul_pad_transpose() {
        test_matmul_padding(true);
    }

    #[test]
    fn test_quantization_with_padded_matmul() {
        // Create a matrix multiplication layer
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .unwrap();

        let input_shape = vec![matrix.ncols_2d(), 5];

        // Create input data that needs quantization
        let rng = &mut thread_rng();
        let input_data = (0..input_shape[0])
            .into_iter()
            .map(|_| {
                (0..input_shape[1])
                    .into_iter()
                    .map(|_| rng.gen_range(-1.0f32..1.0f32))
                    .collect_vec()
            })
            .collect_vec();

        // Quantize the input
        let quantized_input: Vec<Element> = input_data
            .iter()
            .flat_map(|row| row.iter().map(|x| ScalingFactor::default().quantize(x)))
            .collect();

        let layer = MatMul::new(
            OperandMatrix::new_weight_matrix(matrix),
            OperandMatrix::Input,
        )
        .unwrap();

        // Pad the layer
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Create input tensor
        let input_tensor = Tensor::<Element>::new(input_shape, quantized_input);

        // Apply the layer operation on both original and padded
        let output = layer.op(vec![&input_tensor]).unwrap();
        let padded_output = padded
            .op(vec![&input_tensor.pad_next_power_of_two_2d()])
            .unwrap();

        // Check that the result is correct
        let out_shape = output.get_shape();
        let out_cols = out_shape[1];
        let padded_out_shape = padded_output.get_shape();
        let padded_out_cols = padded_output.get_shape()[1];
        for i in 0..padded_out_shape[0] {
            for j in 0..padded_out_cols {
                if i < out_shape[0] && j < out_cols {
                    // non-padded portion
                    assert_eq!(
                        output.get_data()[i * out_cols + j],
                        padded_output.get_data()[i * padded_out_cols + j]
                    );
                } else {
                    // padded portion of the output should just be zero
                    assert_eq!(padded_output.get_data()[i * padded_out_cols + j], 0,);
                }
            }
        }
    }

    #[test]
    fn test_matmul() {
        let matmul = MatMul::new(OperandMatrix::Input, OperandMatrix::Input).unwrap();
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        assert_eq!(result.outputs[0].data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_transpose_b() {
        let matmul = MatMul::new_with_config(
            OperandMatrix::Input,
            OperandMatrix::Input,
            Config::TransposeB,
        )
        .unwrap();
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose();
        let result = matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        assert_eq!(result.outputs[0].data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_proven_matmul_with_two_input_matrices() {
        let first_input_shape = vec![100, 200];
        let second_input_shape = vec![200, 300];
        let matrix_shape = vec![300, 100];
        let mut model = Model::new_from_input_shapes(
            vec![first_input_shape, second_input_shape],
            PaddingMode::NoPadding,
        );

        let matmul = MatMul::new(OperandMatrix::Input, OperandMatrix::Input).unwrap();
        let first_matmul_id = model
            .add_consecutive_layer(Layer::MatMul(matmul), None)
            .unwrap();
        let matmul = MatMul::new(
            OperandMatrix::new_weight_matrix(Tensor::random(&matrix_shape)),
            OperandMatrix::Input,
        )
        .unwrap();
        model
            .add_consecutive_layer(Layer::MatMul(matmul), Some(first_matmul_id))
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }

    #[test]
    fn test_proven_matmul_transposed() {
        let first_input_shape = vec![100, 200];
        let second_input_shape = vec![300, 200];
        let matrix_shape = vec![100, 300];
        let mut model = Model::new_from_input_shapes(
            vec![first_input_shape, second_input_shape],
            PaddingMode::NoPadding,
        );

        let matmul = MatMul::new_with_config(
            OperandMatrix::Input,
            OperandMatrix::Input,
            Config::TransposeB,
        )
        .unwrap();
        let first_matmul_id = model
            .add_consecutive_layer(Layer::MatMul(matmul), None)
            .unwrap();
        let matmul = MatMul::new_with_config(
            OperandMatrix::Input,
            OperandMatrix::new_weight_matrix(Tensor::random(&matrix_shape)),
            Config::TransposeB,
        )
        .unwrap();
        model
            .add_consecutive_layer(Layer::MatMul(matmul), Some(first_matmul_id))
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }
}
