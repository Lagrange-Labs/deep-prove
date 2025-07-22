//! Module to infere and prove statements of the form:
//! A = A_1 || A_2 || ... || A_n
//! B = B_1 || B_2 || ... || B_n
//! C = C_1 || C_2 || ... || C_n
//! where C_i = A_i @ B_i
//! Here concatenation means concatenation over the highest dimension, e.g.
//! if A_i is of shape [1, r, s] then A = [A_1, A_2, ... , A_n] is of shape [n, r, s]
//!
//! This module currently only supports the case where A_i and B_i are witnesses values.
//! Transpose: There is the option to transpose the output of the matmul. This is useful for proving to avoid
//! having to prove explicitly the transpose operation with a separate layer, as sumcheck based proving can directly
//! prove the transpose at the same time as the matmul.
use std::collections::BTreeMap;

use std::borrow::Borrow;

use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::trace;
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, Tensor,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, VerifiableCtx,
        },
        requant::Requant,
    },
    model::StepData,
    padding::{PaddingMode, ShapeInfo, pad_concat_mat_mul},
    tensor::{Number, Shape},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    pub fn new(perm: Vec<usize>) -> Self {
        assert!(
            perm.len() > 1,
            "Permutation must have at least two elements"
        );
        assert!(
            perm.iter().all(|&x| x < perm.len()),
            "Permutation indices must be less than the length of the permutation"
        );
        Self(perm)
    }

    pub fn apply(&self, shape: &Shape) -> Shape {
        shape.permute(&self.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct InputMatrixDimensions {
    /// Index in the input shape that refers to the dimension that we concatenate over.
    concat_dimension: usize,
    /// Index in the input shape that refers to the dimension over which the matrix multiplication
    /// of each chunk is performed. For example, with two matrices A and B of shape [a,b,c] and [a,c,d],
    /// the `mat_mul_dimension` for A is 2 (c) and for B is 1 (c) as well.
    mat_mul_dimension: usize,
    /// Index in the input shape that refers to the dimension which will be part of the shape of
    /// the chunks of the output matrix
    output_dimension: usize,
}

impl InputMatrixDimensions {
    /// Instantiate a new configuration for an input matrix to `ConcatMatMul`. It requires:
    /// - `concat_dimension`: dimension of the input tensor which defines how many chunks are concatenated in the input tensor
    /// - `mat_mul_dimension`: dimension of the input tensor which corresponds to the dimension over which matrix multiplication
    ///   is performed over each chunk
    /// - `output_dimension`: dimension of the input tensors which corresponds to the dimension of each chunk that is
    ///   propagated to the corresponding chunk of the output matrix
    pub fn new(concat_dimension: usize, mat_mul_dimension: usize, output_dimension: usize) -> Self {
        Self {
            concat_dimension,
            mat_mul_dimension,
            output_dimension,
        }
    }
    /// Compute the permutation, if any, to be applied to the input tensor to re-arrange the dimensions as in `expected_dimensions`
    fn compute_permutation(&self, expected_dimensions: &Self) -> Option<Permutation> {
        if self != expected_dimensions {
            // we need to permute to get to `expected_dimensions`
            let mut permute = vec![0; 3];
            permute[expected_dimensions.concat_dimension] = self.concat_dimension;
            permute[expected_dimensions.output_dimension] = self.output_dimension;
            permute[expected_dimensions.mat_mul_dimension] = self.mat_mul_dimension;
            Some(Permutation::new(permute))
        } else {
            None
        }
    }

    /// Build the point for the given input tensors over which the claim produced by sum-check is evaluated.
    /// This is necessary in case we need to permute the input. The proving happens directly over the permuted
    /// tensor, and the claim produced is then "permuted" to be consistent with the fact the input is the
    /// non permuted matrix.
    fn build_point_for_input<E: ExtensionField>(
        &self,
        point_for_concat_dim: &[E],
        point_for_mat_mul_dim: &[E],
        point_for_output_dim: &[E],
    ) -> Vec<E> {
        [
            (self.concat_dimension, point_for_concat_dim),
            (self.mat_mul_dimension, point_for_mat_mul_dim),
            (self.output_dimension, point_for_output_dim),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>() // collect in BTreeMap to sort by dimension
        .into_iter()
        .rev() // reverse since we start by higher dimensions when building points
        .flat_map(|(_, point)| point.to_vec())
        .collect()
    }

    /// Compute the MLE for the input tensor, checking if the tensor needs to be permuted for the sum-check
    /// employed in proving
    fn input_mle_for_proving<E: ExtensionField>(
        &self,
        input: &Tensor<E>,
        partial_point: &[E],
    ) -> DenseMultilinearExtension<E> {
        // determine if we need to permute the matrix for sum-check
        if self.concat_dimension > self.mat_mul_dimension || self.output_dimension == 1 {
            // we need to permute the matrix; for simplicity, we alwayes permute in order to get
            // the shape [concat_dimension, mat_mul_dimension, not_mat_mul_dimension]
            let permute = vec![
                self.concat_dimension,
                self.mat_mul_dimension,
                self.output_dimension,
            ];
            let mut mle = input.permute3d(&permute).data.into_mle();
            mle.fix_variables_in_place_parallel(partial_point);
            mle
        } else {
            // no permutation needed, just compute the MLE
            let mut mle = input.data.clone().into_mle();
            if self.output_dimension == 0 {
                // We need to fix the variables related to the first dimension, which are
                // the most significant ones
                mle.fix_high_variables_in_place(partial_point);
            } else {
                // Output dimension is the last dimension, so we need to fix the variables
                // related to the last dimension, which are the least significant ones
                mle.fix_variables_in_place_parallel(partial_point);
            }
            mle
        }
    }
}

/// Contain information about the permutations to be applied to input
/// and output matrices
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MatrixPermutations {
    left: InputMatrixDimensions,
    right: InputMatrixDimensions,
    /// If Some, it contains the permutation to apply to the output of the matmul.
    permute: Option<Permutation>,
}

impl MatrixPermutations {
    fn ensure_shape_consistency<S: Borrow<Shape>>(&self, shapes: &[S]) -> anyhow::Result<()> {
        assert!(shapes.len() == 2, "ConcatMatMul expects 2 inputs");
        ensure!(
            shapes[0].borrow().rank() == shapes[1].borrow().rank(),
            "ConcatMatMul expects input shapes with same rank: {:?} vs {:?}",
            shapes[0].borrow(),
            shapes[1].borrow()
        );
        ensure!(
            shapes[0].borrow().rank() == 3,
            "ConcatMatMul expects inputs of rank 3"
        );
        ensure!(
            shapes[0].borrow().dim(self.left.concat_dimension)
                == shapes[1].borrow().dim(self.right.concat_dimension),
            "ConcatMatMul expects inputs with same concatenation dimension"
        );
        // check consistency of matrix mul dimensions
        ensure!(
            shapes[0].borrow().dim(self.left.mat_mul_dimension)
                == shapes[1].borrow().dim(self.right.mat_mul_dimension),
            "ConcatMatMul expects submatrices dimensions to match"
        );
        Ok(())
    }

    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        let a_shape = &input_shapes[0];
        let b_shape = &input_shapes[1];
        self.ensure_shape_consistency(&[a_shape, b_shape]).unwrap();
        // inner matrix shapes
        let a_shape = if let Some(permute) = self.compute_permutation_for_left_input() {
            permute.apply(a_shape)
        } else {
            a_shape.clone()
        };
        let b_shape = if let Some(permute) = self.compute_permutation_for_right_input() {
            permute.apply(b_shape)
        } else {
            b_shape.clone()
        };

        let mut mat_result_shape: Shape =
            vec![a_shape.dim(0), a_shape.dim(1), b_shape.dim(2)].into();
        if let PaddingMode::Padding = padding_mode {
            mat_result_shape = mat_result_shape.next_power_of_two()
        }
        if let Some(ref permute) = self.permute {
            trace!("ConcatMatMul: Permute: {permute:?} over resulting shape {mat_result_shape:?}",);
            mat_result_shape = mat_result_shape.permute(&permute.0);
        }
        vec![mat_result_shape]
    }

    /// Compute permutation to be applied to the left input tensor, if any
    fn compute_permutation_for_left_input(&self) -> Option<Permutation> {
        let expected_dimensions = ConcatMatMul::expected_dimension_for_left_input();
        self.left.compute_permutation(&expected_dimensions)
    }

    /// Compute permutation to be applied to the right input tensor, if any
    fn compute_permutation_for_right_input(&self) -> Option<Permutation> {
        let expected_dimensions = ConcatMatMul::expected_dimension_for_right_input();
        self.right.compute_permutation(&expected_dimensions)
    }

    /// Split the point over which sum-check claims are evaluated in two components:
    /// - the first component refers to the concatenation dimension in the input tensors
    /// - the second components refers to the mat mul dimension in the input tensors
    fn split_sumcheck_point<'a, E: ExtensionField>(
        &self,
        point: &'a [E],
        input_shapes: &[Shape],
    ) -> Result<(&'a [E], &'a [E])> {
        let num_entries_mat_mul_dimension = input_shapes[0].dim(self.left.mat_mul_dimension);
        ensure!(
            num_entries_mat_mul_dimension == input_shapes[1].dim(self.right.mat_mul_dimension),
            "ConcatMatMul: Incompatible size of mat mul dimensions for input shapes when splitting sum-check point: expected {}, found {}",
            num_entries_mat_mul_dimension,
            input_shapes[1].dim(self.right.mat_mul_dimension),
        );
        ensure!(
            num_entries_mat_mul_dimension.is_power_of_two(),
            "Number of columns in the mat mul dimension must be a power of two, found {}",
            num_entries_mat_mul_dimension
        );
        let num_vars = num_entries_mat_mul_dimension.ilog2() as usize;

        // first set of coordinates of `point` refers to mat mul dimension
        Ok((&point[num_vars..], &point[..num_vars]))
    }

    /// Split the point over which the output claim is evaluated in three components:
    /// - the first component refers to the concatenation dimension
    /// - the second component refers to the number of rows of each chunk of the output matrix
    /// - the third component refers to the number of columns of each chunk of the output matrix
    fn split_output_claim_point<'a, E: ExtensionField>(
        &self,
        output_shape: Shape,
        point: &'a [E],
    ) -> Result<(&'a [E], &'a [E], &'a [E])> {
        // split the point according to the 3 dimensions of the output matrix
        ensure!(
            output_shape.rank() == 3,
            "Output shape must be of rank 3, found {}",
            output_shape.rank()
        );

        let num_vars = (0..3)
            .map(|i| {
                ensure!(
                    output_shape.dim(i).is_power_of_two(),
                    "Output shape dimension {} must be a power of two, found {}",
                    i,
                    output_shape.dim(i)
                );
                Ok(output_shape.dim(i).next_power_of_two().ilog2() as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        ensure!(
            point.len() == num_vars.iter().sum::<usize>(),
            "Point length {} does not match the expected number of variables {}",
            point.len(),
            num_vars.iter().sum::<usize>()
        );

        let mut coordinates_to_split = point.len();
        let points = (0..3)
            .map(|i| {
                let start_range = coordinates_to_split - num_vars[i];
                let end_range = coordinates_to_split;
                coordinates_to_split = start_range;
                &point[start_range..end_range]
            })
            .collect::<Vec<_>>();
        assert_eq!(
            coordinates_to_split, 0,
            "Not all point coordinates were split among sub-points"
        );

        // looks at whether the output matrix needs to be permuted or not
        let (concat_dimension, row_dimension, col_dimension) = self
            .permute
            .as_ref()
            .map(|p| {
                let mut new_dimensions = [0; 3];
                p.0.iter()
                    .enumerate()
                    .for_each(|(i, &source_dim)| new_dimensions[source_dim] = i);
                (new_dimensions[0], new_dimensions[1], new_dimensions[2])
            })
            .unwrap_or((0, 1, 2));

        Ok((
            points[concat_dimension],
            points[row_dimension],
            points[col_dimension],
        ))
    }
}

use super::provable::LayerOut;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatMatMul {
    permutations: MatrixPermutations,
    /// It tells what is the maximum bit size we ever expect the output of this layer to be.
    /// NOTE: This is a config item normally but we need this information during quantization.
    /// Best would be to rework quantization trait to include such config items.
    intermediate_bit_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConcatMatMulCtx<E> {
    pub(crate) node_id: NodeId,
    pub(crate) matrix_poly_aux: VPAuxInfo<E>,
    permutations: MatrixPermutations,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ConcatMatMulProof<E: ExtensionField> {
    sumcheck_proof: IOPProof<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly".
    /// Since we only support quadratic right now it's a flat list.
    individual_claims: Vec<E>,
}

impl<E: ExtensionField> ConcatMatMulProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims
            .iter()
            .fold(E::ONE, |acc, e| acc * *e)
    }
}

const DEFAULT_INTERMEDIATE_BIT_SIZE: usize = 25;

impl ConcatMatMul {
    pub fn new(left: InputMatrixDimensions, right: InputMatrixDimensions) -> Self {
        Self {
            permutations: MatrixPermutations {
                left,
                right,
                permute: None,
            },
            intermediate_bit_size: DEFAULT_INTERMEDIATE_BIT_SIZE,
        }
    }
    pub fn new_with_permute(
        left: InputMatrixDimensions,
        right: InputMatrixDimensions,
        permutation: Permutation,
    ) -> Self {
        Self {
            permutations: MatrixPermutations {
                left,
                right,
                permute: Some(permutation),
            },
            intermediate_bit_size: DEFAULT_INTERMEDIATE_BIT_SIZE,
        }
    }
    /// Update the intermediate bit size of `self`, which is necessary to properly quantize the layer,
    /// using the following information about the input matrices:
    /// - `max_shapes`: Shapes with the biggest possible dimensions for the input matrices
    /// - `quantized_left_input_range`: Range of values for the left input matrix, when quantized
    /// - `quantized_right_input_range`: Range of values for the right input matrix, when quantized
    pub fn update_intermediate_bit_size(
        self,
        max_shapes: Vec<Shape>,
        quantized_left_input_range: Option<usize>,
        quantized_right_input_range: Option<usize>,
    ) -> Self {
        self.ensure_shape_consistency(&max_shapes).unwrap();
        let matrix_shape = Shape::new(vec![
            max_shapes[0].dim(self.permutations.left.output_dimension),
            max_shapes[0].dim(self.permutations.left.mat_mul_dimension),
        ]);
        let intermediate_bit_size = matrix_shape
            .matmul_output_bitsize(quantized_left_input_range, quantized_right_input_range);
        Self {
            permutations: self.permutations,
            intermediate_bit_size,
        }
    }

    pub(crate) fn output_domain(&self) -> Element {
        (1 << self.intermediate_bit_size as Element) - 1
    }

    pub fn ensure_shape_consistency<S: Borrow<Shape>>(&self, shapes: &[S]) -> anyhow::Result<()> {
        self.permutations.ensure_shape_consistency(shapes)
    }

    /// Return the expected dimension for left input tesnor when performing `ConcatMatMul`;
    /// if the actual dimensions are different, the input tensor will be permuted
    fn expected_dimension_for_left_input() -> InputMatrixDimensions {
        // to compute `ConcatMatMul`, we need the following shape for left input:
        // [concat_dimension, output_dimension, mat_mul_dimension]
        InputMatrixDimensions {
            concat_dimension: 0,
            mat_mul_dimension: 2,
            output_dimension: 1,
        }
    }

    /// Return the expected dimension for right input tesnor when performing `ConcatMatMul`;
    /// if the actual dimensions are different, the input tensor will be permuted
    fn expected_dimension_for_right_input() -> InputMatrixDimensions {
        // to compute `ConcatMatMul`, we need the following shape for left input:
        // [concat_dimension, output_dimension, mat_mul_dimension]
        InputMatrixDimensions {
            concat_dimension: 0,
            mat_mul_dimension: 1,
            output_dimension: 2,
        }
    }

    pub(crate) fn prove_step<
        R: AsRef<Tensor<E>>,
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: Transcript<E>,
    >(
        &self,
        last_claims: Vec<&Claim<E>>,
        output: &Tensor<E>,
        inputs: &[R],
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<(Vec<crate::Claim<E>>, ConcatMatMulProof<E>)> {
        let input_shapes = inputs
            .iter()
            .map(|input| input.as_ref().get_shape())
            .collect_vec();
        self.ensure_shape_consistency(&input_shapes)?;

        let (point_for_concat, point_for_row, point_for_col) = self
            .permutations
            .split_output_claim_point(output.get_shape(), &last_claims[0].point)?;

        // determine if we need to permute the left matrix for sum-check
        let left = self
            .permutations
            .left
            .input_mle_for_proving(inputs[0].as_ref(), point_for_row);

        // determine if we need to permute the right matrix for sum-check
        let right = self
            .permutations
            .right
            .input_mle_for_proving(inputs[1].as_ref(), point_for_col);

        ensure!(
            left.num_vars() == right.num_vars(),
            "ConcatMatMul: left and right input MLEs must have the same number of variables: {} vs {}",
            left.num_vars(),
            right.num_vars()
        );

        let sum_check_num_vars = left.num_vars();

        let num_columns_left =
            inputs[0].as_ref().get_shape()[self.permutations.left.mat_mul_dimension];
        let num_rows_right =
            inputs[1].as_ref().get_shape()[self.permutations.right.mat_mul_dimension];
        ensure!(
            num_columns_left == num_rows_right,
            "ConcatMatMul: found different mat mul dimensions in left and right input matrix {} vs {}",
            num_columns_left,
            num_rows_right
        );

        // create the beta vector necessary for the batched matrix multiplication
        let beta_evals = compute_betas_eval(point_for_concat)
            .into_iter()
            .flat_map(|eval|
                // replicate it for the number of entries in the mat mul dimension
                vec![eval; num_columns_left])
            .collect_vec();

        let beta_mle = beta_evals.into_mle();

        ensure!(
            sum_check_num_vars == beta_mle.num_vars(),
            "ConcatMatMul: Beta vector MLE has an invalid number of variables: expected {}, found {}",
            sum_check_num_vars,
            beta_mle.num_vars(),
        );

        let mut vp = VirtualPolynomial::new(sum_check_num_vars);

        vp.add_mle_list(vec![beta_mle.into(), left.into(), right.into()], E::ONE);

        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        let evals = state.get_mle_final_evaluations();

        let left_eval = evals[1];
        let right_eval = evals[2];

        let proof_point = &proof.point;
        let (point_for_concat_dim, point_for_mat_mul_dim) = self
            .permutations
            .split_sumcheck_point(proof_point, &input_shapes)?;

        let left_point = self.permutations.left.build_point_for_input(
            point_for_concat_dim,
            point_for_mat_mul_dim,
            point_for_row,
        );

        let right_point = self.permutations.right.build_point_for_input(
            point_for_concat_dim,
            point_for_mat_mul_dim,
            point_for_col,
        );

        let left_claim = Claim::new(left_point, left_eval);

        let right_claim = Claim::new(right_point, right_eval);

        let proof = ConcatMatMulProof {
            sumcheck_proof: proof,
            individual_claims: evals,
        };

        Ok((vec![left_claim, right_claim], proof))
    }
}

impl<N: Number> Evaluate<N> for ConcatMatMul {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "ConcatMatMul expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        let a_shape = a.get_shape();
        let b_shape = b.get_shape();
        self.ensure_shape_consistency(&[&a_shape, &b_shape])?;
        let permuted_a = self
            .permutations
            .compute_permutation_for_left_input()
            .map(|p| a.permute3d(&p.0));
        let permuted_b = self
            .permutations
            .compute_permutation_for_right_input()
            .map(|p| b.permute3d(&p.0));
        let a = permuted_a.as_ref().unwrap_or(a);
        let b = permuted_b.as_ref().unwrap_or(b);
        let a_shape = a.get_shape();
        let b_shape = b.get_shape();
        ensure!(
            a_shape.dim(0) == b_shape.dim(0),
            "ConcatMatMul expects inputs with same batch size: {} vs {}",
            a_shape.dim(0),
            b_shape.dim(0),
        );
        let results = (0..a_shape.dim(0))
            .map(|batch| {
                let batch_a = a.slice_3d(batch, batch + 1).reshape(a_shape.slice(1..=2));
                let batch_b = b.slice_3d(batch, batch + 1).reshape(b_shape.slice(1..=2));
                batch_a.matmul(&batch_b)
            })
            .collect::<Vec<_>>();
        let mut it = results.into_iter();
        // reshape because concat expects a 3d tensor so he can accumulate in the highest dimension.
        let concat =
            it.next()
                .unwrap()
                .reshape(Shape::new(vec![1, a_shape.dim(1), b_shape.dim(2)]));
        let mut concat = it.fold(concat, |mut acc, x| {
            acc.concat(x);
            acc
        });
        if let Some(ref transpose) = self.permutations.permute {
            concat = concat.permute3d(&transpose.0);
        }
        Ok(LayerOut::from_vec(vec![concat]))
    }
}

const IS_PROVABLE: bool = true;

impl OpInfo for ConcatMatMul {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        self.permutations.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!("ConcatMatMul: {:?})", self.permutations)
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl QuantizeOp for ConcatMatMul {
    type QuantizedOp = ConcatMatMul;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: super::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<super::provable::QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scale = S::scaling_factors_for_node(data, node_id, num_outputs)[0];
        // normally it's input_scaling * model_scaling / output_scaling, except in this case, we don't have a model_scaling
        // but we have the second matrix scaling, so we use that.
        let input_scale = input_scaling[0];
        let weights_scale = input_scaling[1];
        let intermediate_bit_size = self.intermediate_bit_size;
        let requant = Requant::from_scaling_factors(
            input_scale,
            weights_scale,
            output_scale,
            intermediate_bit_size,
        );
        Ok(super::provable::QuantizeOutput::new(self, vec![output_scale]).with_requant(requant))
    }
}

impl<E: ExtensionField + DeserializeOwned> ProveInfo<E> for ConcatMatMul
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        let num_columns_left = aux.last_output_shape[0][self.permutations.left.mat_mul_dimension];

        let num_rows_right = aux.last_output_shape[1][self.permutations.right.mat_mul_dimension];

        ensure!(
            num_columns_left == num_rows_right,
            "ConcatMatMul: number of columns in left matrix chunk different from number of rows in right matrix chunk: {} vs {}",
            num_columns_left,
            num_rows_right,
        );

        let num_chunks = aux.last_output_shape[0][self.permutations.left.concat_dimension];
        ensure!(
            num_chunks == aux.last_output_shape[1][self.permutations.right.concat_dimension],
            "ConcatMatMul: number of chunk matrices in left matrix different from number of chunks in right matrix: {} vs {}",
            num_chunks,
            aux.last_output_shape[1][self.permutations.right.concat_dimension]
        );

        ensure!(num_columns_left.is_power_of_two());
        ensure!(num_chunks.is_power_of_two());

        let num_vars = (num_columns_left * num_chunks).ilog2() as usize;

        let vp_aux = VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars, num_vars]]);

        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);

        let ctx = ConcatMatMulCtx {
            node_id: id,
            matrix_poly_aux: vp_aux,
            permutations: self.permutations.clone(),
        };

        Ok((LayerCtx::ConcatMatMul(ctx), aux))
    }
}

impl PadOp for ConcatMatMul {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_concat_mat_mul(self, si)
    }
}

impl<E: ExtensionField + DeserializeOwned, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS>
    for ConcatMatMul
where
    E::BaseField: DeserializeOwned + Serialize,
{
    type Ctx = ConcatMatMulCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<crate::Claim<E>>> {
        ensure!(
            step_data.inputs.len() == 2,
            "ConcatMatMul expects 2 inputs, got {}",
            step_data.inputs.len()
        );

        ensure!(
            step_data.outputs.outputs().len() == 1,
            "ConcatMatMul expects 1 output, got {}",
            step_data.outputs.outputs().len()
        );

        let output = step_data.outputs.outputs()[0];

        let (claims, proof) = self.prove_step(last_claims, output, &step_data.inputs, prover)?;

        prover.push_proof(node_id, LayerProof::ConcatMatMul(proof));

        Ok(claims)
    }
}

impl<E: ExtensionField> OpInfo for ConcatMatMulCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        self.permutations.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "ConcatMatMulCtx: {} -> {:?}",
            self.node_id, self.permutations
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E: ExtensionField + DeserializeOwned, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS>
    for ConcatMatMulCtx<E>
where
    E::BaseField: DeserializeOwned,
{
    type Proof = ConcatMatMulProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        ensure!(
            last_claims.len() == 1,
            "Expected only one output claim for ConcatMatMul verifier, found {}",
            last_claims.len()
        );

        let last_claim = last_claims[0];

        let padded_input_shapes = &shape_step.padded_input_shape;
        self.permutations
            .ensure_shape_consistency(padded_input_shapes)?;

        let subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.sumcheck_proof,
            &self.matrix_poly_aux,
            verifier.transcript,
        );

        ensure!(
            shape_step.padded_output_shape.len() == 1,
            "Expected only one output shape for ConcatMatMul verifier, found {}",
            shape_step.padded_output_shape.len(),
        );

        let (point_for_concat, point_for_row, point_for_col) =
            self.permutations.split_output_claim_point(
                shape_step.padded_output_shape[0].clone(),
                &last_claims[0].point,
            )?;

        let sumcheck_point = subclaim.point_flat();

        let (point_for_concat_dim, point_for_mat_mul_dim) = self
            .permutations
            .split_sumcheck_point(&sumcheck_point, padded_input_shapes)?;

        // first, verify the claim about the `beta_MLE` used in the sumcheck in `prove`.
        // The MLE of  claim should be equal to \beta(x_c, point_for_concat), where x_c are the coordinates of the
        // sumcheck variables corresponding to the concatenation dimension.
        // Therefore, the claim produced by the sumcheck should be equivalent to
        // \beta(point_for_concat_dim, point_for_concat)
        let expected_beta_eval = identity_eval(point_for_concat_dim, point_for_concat);
        ensure!(
            expected_beta_eval == proof.individual_claims[0],
            "Wrong evaluation of beta_MLE found in ConcatMatMul proof: expected {}, found {}",
            expected_beta_eval,
            proof.individual_claims[0],
        );

        let left_point = self.permutations.left.build_point_for_input(
            point_for_concat_dim,
            point_for_mat_mul_dim,
            point_for_row,
        );
        let left_eval = proof.individual_claims[1];
        let left_claim = Claim::new(left_point, left_eval);

        let right_point = self.permutations.right.build_point_for_input(
            point_for_concat_dim,
            point_for_mat_mul_dim,
            point_for_col,
        );
        let right_eval = proof.individual_claims[2];
        let right_claim = Claim::new(right_point, right_eval);

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

        Ok(vec![left_claim, right_claim])
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{
        Tensor,
        layers::{
            Layer,
            provable::{Edge, Node},
        },
        model::{Model, test::prove_model},
    };

    use super::*;

    #[test]
    fn test_concat_matmul() {
        let concat_matmul = ConcatMatMul::new(
            ConcatMatMul::expected_dimension_for_left_input(),
            ConcatMatMul::expected_dimension_for_right_input(),
        );
        let a = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let b = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        assert_eq!(
            result.outputs[0].data,
            vec![7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0]
        );
    }

    #[test]
    fn test_concat_matmul_with_output_transpose() {
        let concat_matmul = ConcatMatMul::new_with_permute(
            ConcatMatMul::expected_dimension_for_left_input(),
            ConcatMatMul::expected_dimension_for_right_input(),
            Permutation::new(vec![1, 0, 2]),
        );
        let a = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let b = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        let expected = Tensor::new(
            vec![2, 2, 2].into(),
            vec![7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0],
        );
        let expected = expected.permute3d(&vec![1, 0, 2]);
        assert_eq!(result.outputs[0].data, expected.data);
        let expected_shape =
            concat_matmul.output_shapes(&[a.get_shape(), b.get_shape()], PaddingMode::NoPadding);
        assert_eq!(result.outputs[0].get_shape(), expected_shape[0]);
    }

    #[test]
    fn test_concat_matmul_with_input_transpose() {
        let a = Tensor::new(
            vec![3, 2, 2].into(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let b = Tensor::new(
            vec![2, 3, 2].into(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let concat_matmul = ConcatMatMul::new(
            InputMatrixDimensions::new(1, 2, 0),
            InputMatrixDimensions::new(0, 2, 1),
        );

        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        let expected = Tensor::new(
            vec![2, 3, 3].into(),
            vec![
                5.0, 11.0, 17.0, 17.0, 39.0, 61.0, 29.0, 67.0, 105.0, 53.0, 67.0, 81.0, 113.0,
                143.0, 173.0, 173.0, 219.0, 265.0,
            ],
        );
        assert_eq!(result.outputs[0].data, expected.data);
        let expected_shape =
            concat_matmul.output_shapes(&[a.get_shape(), b.get_shape()], PaddingMode::NoPadding);
        assert_eq!(result.outputs[0].get_shape(), expected_shape[0]);
    }

    #[test]
    fn test_proven_concat_matmul() {
        // we test over a model where concat matmul is the first layer, so we need 2 input shapes
        let input_shape_left = vec![5, 14, 27].into();
        let input_shape_right = vec![5, 27, 18].into();

        let mut model = Model::new_from_input_shapes(
            vec![input_shape_left, input_shape_right],
            PaddingMode::NoPadding,
        );
        let mat_mul = ConcatMatMul::new(
            ConcatMatMul::expected_dimension_for_left_input(),
            ConcatMatMul::expected_dimension_for_right_input(),
        );
        let _id = model
            .add_consecutive_layer(Layer::ConcatMatMul(mat_mul), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        let outputs = prove_model(model).unwrap();

        // check output shape
        assert_eq!(
            outputs[0].get_shape(),
            Shape::new(vec![5, 14, 18]).next_power_of_two()
        );
    }

    #[test]
    fn test_proven_concat_matmul_input_permutation() {
        // we test over a model where concat matmul is the first layer, so we need 2 input shapes
        let input_shape_left = vec![27, 14, 5].into(); // concat dimension is 5, mul dimension is 27
        let input_shape_right = vec![18, 5, 27].into();

        let mat_mul = ConcatMatMul::new(
            InputMatrixDimensions::new(2, 0, 1),
            InputMatrixDimensions::new(1, 2, 0),
        );

        let mut model = Model::new_from_input_shapes(
            vec![input_shape_left, input_shape_right],
            PaddingMode::NoPadding,
        );

        let _id = model
            .add_consecutive_layer(Layer::ConcatMatMul(mat_mul), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        let outputs = prove_model(model).unwrap();
        assert_eq!(
            outputs[0].get_shape(),
            Shape::new(vec![5, 14, 18]).next_power_of_two()
        );
    }

    #[test]
    fn test_model_with_chained_concat_matmul() {
        let input_shape_left = vec![17, 24, 7].into(); // concat dimension is 7, mul dimension is 24
        let input_shape_right = vec![7, 24, 45].into();

        // we have also another input, which is going to be multiplied with the output of the first
        // concat matmul operation
        let additional_input_shape = vec![21, 7, 45].into(); // concat dimension is 7, mul dimension is 45,
        // since the output shape of the previous concat matmul will be `[7, 17, 45]`

        let mut model = Model::new_from_input_shapes(
            vec![input_shape_left, input_shape_right, additional_input_shape],
            PaddingMode::NoPadding,
        );

        let first_matmul = ConcatMatMul::new(
            InputMatrixDimensions::new(2, 1, 0),
            ConcatMatMul::expected_dimension_for_right_input(),
        );

        let first_node_id = model
            .add_node(Node::new(
                vec![Edge::new_at_edge(0), Edge::new_at_edge(1)],
                Layer::ConcatMatMul(first_matmul),
            ))
            .unwrap();

        // add another concat matmul layer, multiplying the output of `first_node_id` with the additional
        // input tensor of the model
        let second_matmul = ConcatMatMul::new_with_permute(
            ConcatMatMul::expected_dimension_for_left_input(),
            InputMatrixDimensions::new(1, 2, 0),
            Permutation::new(vec![2, 0, 1]), /* we also permute the output tensor to have the concat dimension as
                                              * the middle dimension */
        );

        let _second_node_id = model
            .add_node(Node::new(
                vec![Edge::new(first_node_id, 0), Edge::new_at_edge(2)],
                Layer::ConcatMatMul(second_matmul),
            ))
            .unwrap();

        model.route_output(None).unwrap();

        let outputs = prove_model(model).unwrap();
        assert_eq!(
            outputs[0].get_shape(),
            Shape::new(vec![21, 7, 17]).next_power_of_two()
        );
    }
}
