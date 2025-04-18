use anyhow::{Context, Result, anyhow, ensure};

use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover,
    commit::precommit::PolyID,
    iop::{context::ContextAux, verifier::Verifier},
    layers::LayerProof,
    quantization,
    tensor::Tensor,
};

use super::{LayerCtx, requant::Requant};

/// A matrix to be multiplied in the matrix multiplication layer
#[derive(Clone, Debug)]
pub enum OperandMatrix {
    /// The matrix is a constant matrix specified in the modelta
    Weigth(Tensor<Element>),
    /// The matrix is input-dependent, we just need the shape
    Input(Vec<usize>),
}

impl OperandMatrix {
    pub(crate) fn is_matrix(&self) -> bool {
        match self {
            OperandMatrix::Weigth(tensor) => tensor.is_matrix(),
            OperandMatrix::Input(shape) => shape.len() == 2,
        }
    }

    pub(crate) fn get_shape(&self) -> Vec<usize> {
        match self {
            OperandMatrix::Weigth(tensor) => tensor.get_shape(),
            OperandMatrix::Input(shape) => shape.clone(),
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        match self {
            OperandMatrix::Weigth(tensor) => tensor.nrows_2d(),
            OperandMatrix::Input(shape) => shape[0],
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match self {
            OperandMatrix::Weigth(tensor) => tensor.ncols_2d(),
            OperandMatrix::Input(shape) => shape[1],
        }
    }

    pub(crate) fn pad_next_power_of_two(self) -> Self {
        match self {
            OperandMatrix::Weigth(tensor) => OperandMatrix::Weigth(tensor.pad_next_power_of_two()),
            OperandMatrix::Input(shape) => OperandMatrix::Input(
                shape
                    .into_iter()
                    .map(|dim| dim.next_power_of_two())
                    .collect(),
            ),
        }
    }

    pub(crate) fn num_vars_2d(&self) -> (usize, usize) {
        Tensor::<Element>::new_from_shape(self.get_shape()).num_vars_2d()
    }
}

/// Description of the layer
#[derive(Clone, Debug)]
pub struct MatMul {
    pub(crate) left_matrix: OperandMatrix,
    pub(crate) right_matrix: OperandMatrix,
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatMulCtx<E> {
    pub(crate) matrix_poly_id: PolyID,
    pub(crate) matrix_poly_aux: VPAuxInfo<E>,
    // Number of variables of the MLE polynomial for each dimension of the output matrix
    pub(crate) output_mle_num_vars: (usize, usize),
    pub(crate) is_left_matrix_constant: bool,
    pub(crate) is_right_matrix_constant: bool,
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

impl MatMul {
    pub fn new(left_matrix: OperandMatrix, right_matrix: OperandMatrix) -> Result<Self> {
        ensure!(
            left_matrix.is_matrix(),
            "left matrix for MatMul layer is not a matrix"
        );
        ensure!(
            right_matrix.is_matrix(),
            "right matrix for MatMul layer is not a matrix"
        );
        ensure!(
            left_matrix.ncols() == right_matrix.nrows(),
            "Number of columns in left matrix different from number of rows of right matrix: {} != {}",
            left_matrix.ncols(),
            right_matrix.nrows(),
        );
        Ok(Self {
            left_matrix,
            right_matrix,
        })
    }

    pub fn input_shape(&self) -> Vec<Vec<usize>> {
        match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => unreachable!(),
            (OperandMatrix::Weigth(_), OperandMatrix::Input(shape)) => {
                vec![shape.clone()]
            }
            (OperandMatrix::Input(shape), OperandMatrix::Weigth(_)) => vec![shape.clone()],
            (OperandMatrix::Input(left_shape), OperandMatrix::Input(right_shape)) => {
                vec![left_shape.clone(), right_shape.clone()]
            }
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "Matrix multiplication: left = {:?}, right = {:?}",
            self.left_matrix.get_shape(),
            self.right_matrix.get_shape()
        )
    }

    // Return evaluations for the constant matrix employed in the layer.
    // If there is no constant matrix in the layer, `None` is returned
    pub(crate) fn eval_constant_matrix<E: ExtensionField>(&self) -> Option<Vec<E>> {
        match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => unreachable!(),
            (OperandMatrix::Weigth(tensor), OperandMatrix::Input(_)) => Some(tensor.evals_2d()),
            (OperandMatrix::Input(_), OperandMatrix::Weigth(tensor)) => Some(tensor.evals_2d()),
            (OperandMatrix::Input(_), OperandMatrix::Input(_)) => None,
        }
    }

    pub fn op(&self, inputs: Vec<&Tensor<Element>>) -> Result<Tensor<Element>> {
        Ok(match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => unreachable!(),
            (OperandMatrix::Weigth(tensor), OperandMatrix::Input(shape)) => {
                let right_matrix = inputs
                    .first()
                    .ok_or(anyhow!("No matrix provided as input to MatMul"))?;
                ensure!(
                    right_matrix.get_shape() == *shape,
                    "Incompatible shape found for input matrix: expected {:?}, found {:?}",
                    *shape,
                    right_matrix.get_shape(),
                );
                tensor.matmul(right_matrix)
            }
            (OperandMatrix::Input(shape), OperandMatrix::Weigth(tensor)) => {
                let left_matrix = inputs
                    .first()
                    .ok_or(anyhow!("No matrix provided as input to MatMul"))?;
                ensure!(
                    left_matrix.get_shape() == *shape,
                    "Incompatible shape found for input matrix: expected {:?}, found {:?}",
                    *shape,
                    left_matrix.get_shape(),
                );
                left_matrix.matmul(tensor)
            }
            (OperandMatrix::Input(left_shape), OperandMatrix::Input(right_shape)) => {
                ensure!(
                    inputs.len() == 2,
                    "Not enough inputs provided to MatMul: expected 2, found {}",
                    inputs.len()
                );
                inputs.iter().zip(vec![left_shape, right_shape])
                    .enumerate().try_for_each(|(i, (input, shape))| {
                        ensure!(
                            input.get_shape() == *shape,
                            "Incompatible shape found for {i}-th input matrix: expected {:?}, found {:?}",
                            *shape,
                            input.get_shape(),
                        );
                        Ok(())
                    }
                )?;
                inputs[0].matmul(inputs[1])
            }
        })
    }

    pub fn pad_next_power_of_two(self) -> Result<Self> {
        let left_matrix = self.left_matrix.pad_next_power_of_two();
        let right_matrix = self.right_matrix.pad_next_power_of_two();
        Self::new(left_matrix, right_matrix)
    }

    pub fn requant_info(&self) -> Requant {
        let matrix = match (&self.left_matrix, &self.right_matrix) {
            (OperandMatrix::Weigth(_), OperandMatrix::Weigth(_)) => unreachable!(),
            (OperandMatrix::Weigth(tensor), OperandMatrix::Input(_)) => Some(tensor),
            (OperandMatrix::Input(_), OperandMatrix::Weigth(tensor)) => Some(tensor),
            (OperandMatrix::Input(_), OperandMatrix::Input(_)) => None,
        };
        if let Some(matrix) = matrix {
            let ncols = matrix.ncols_2d();
            let max_output_range = matrix
                .get_data()
                .iter()
                .chunks(ncols)
                .into_iter()
                .map(|row| {
                    let row_range = row
                        .map(|w| quantization::range_from_weight(w))
                        .fold((0, 0), |(min, max), (wmin, wmax)| (min + wmin, max + wmax));
                    // weight * MIN can be positive and higher then MAX*weight if weight's negative
                    // so we take the absolute value of the difference
                    (row_range.1 - row_range.0).unsigned_abs() as usize
                })
                .max()
                .expect("No max range found")
                .next_power_of_two();
            let shift = max_output_range.ilog2() as usize - *quantization::BIT_LEN;
            Requant {
                range: max_output_range,
                right_shift: shift,
                after_range: 1 << *quantization::BIT_LEN,
            }
        } else {
            // use a default value
            let max_output_range = 2usize; // assume range [-1, 1]
            Requant {
                range: 2, // assume range [-1, 1]
                right_shift: max_output_range.ilog2() as usize - *quantization::BIT_LEN,
                after_range: 1 << *quantization::BIT_LEN,
            }
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
    fn full_points<E: ExtensionField>(
        claim: &Claim<E>,
        proof_point: &[E],
        output_num_vars: (usize, usize),
    ) -> (Vec<E>, Vec<E>) {
        let (claim_point_for_left, claim_point_for_right) =
            Self::split_claim(claim, output_num_vars);
        let point_for_right = [claim_point_for_right, proof_point].concat();
        let point_for_left = [proof_point, claim_point_for_left].concat();
        (point_for_left, point_for_right)
    }

    pub fn prove_step<E, T>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: Claim<E>,
        mut inputs: Vec<&Tensor<E>>,
        output: &Tensor<E>,
        info: &MatMulCtx<E>,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        let num_inputs = inputs.len();
        let (right_matrix, is_right_constant) = match &self.right_matrix {
            OperandMatrix::Weigth(tensor) => (&Tensor::<E>::from(tensor), true),
            OperandMatrix::Input(shape) => {
                let matrix = inputs
                    .pop()
                    .ok_or(anyhow!("No input provided for right matrix"))?;
                ensure!(
                    matrix.get_shape() == *shape,
                    "Invalid shape found for right input matrix: expected {:?}, found {:?}",
                    shape,
                    matrix.get_shape(),
                );
                (matrix, false)
            }
        };
        let (left_matrix, is_left_constant) = match &self.left_matrix {
            OperandMatrix::Weigth(tensor) => (&Tensor::<E>::from(tensor), true),
            OperandMatrix::Input(shape) => {
                let matrix = inputs
                    .pop()
                    .ok_or(anyhow!("No input provided for left matrix"))?;
                ensure!(
                    matrix.get_shape() == *shape,
                    "Invalid shape found for left input matrix: expected {:?}, found {:?}",
                    shape,
                    matrix.get_shape(),
                );
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
        let ncols_right = right_matrix.ncols_2d();
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
        let (point_for_input, point_for_mat) = Self::split_claim(&last_claim, num_vars_2d);
        // fix the variables for the random input matrix; we need to fix the variables
        // corresponding to a row, so we must fix the HIGH variables
        left_mat_mle.fix_high_variables_in_place(point_for_input);
        // fix the variables for the layer matrix; we need to fix the variables
        // corresponding to a column, so we must fix the low variables
        right_mat_mle.fix_variables_in_place(point_for_mat);

        // check that after fixing the variables in both matrixes the number of free
        // variables is the same
        assert_eq!(left_mat_mle.num_vars(), right_mat_mle.num_vars());

        let num_vars = left_mat_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        // TODO: remove the clone once prover+verifier are working
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
            Self::full_points(&last_claim, &proof.point, num_vars_2d);
        // collection of claims to be returned as output
        let mut output_claims = vec![];
        // compute the claim for the left matrix polynomial. It will be either accumulated in the
        // evaluation claims being opened with the polynomial commitment, or returned as output,
        // depending on whether the left matrix is constant or not
        let eval = state.get_mle_final_evaluations()[0]; // The first MLE being evaluated is the left matrix poly
        let left_claim = Claim::new(point_for_left, eval);
        if is_left_constant {
            // add a claim for the constant polynomial of the left matrix
            prover
                .commit_prover
                .add_claim(info.matrix_poly_id, left_claim)
                .context("unable to add matrix claim")?;
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
            prover
                .commit_prover
                .add_claim(info.matrix_poly_id, right_claim)
                .context("unable to add matrix claim")?;
        } else {
            // append the claim to output claims
            output_claims.push(right_claim);
        }

        prover.push_proof(LayerProof::MatMul(MatMulProof {
            sumcheck: proof,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(output_claims)
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut ctx_aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // construct dimension of the polynomial given to the sumcheck
        let ncols = self.right_matrix.ncols();
        let nrows = self.left_matrix.nrows();
        ctx_aux.last_output_shape = vec![nrows, ncols];

        // number of variables of the MLE polynomials is the number of row
        // variables in in layer matrix
        let num_vars = self.right_matrix.num_vars_2d().0;
        // check that the number of variables is the same as the number of
        // column variables for left matrix
        debug_assert_eq!(num_vars, self.left_matrix.num_vars_2d().1,);

        let is_left_matrix_constant = match &self.left_matrix {
            OperandMatrix::Weigth(_) => true,
            OperandMatrix::Input(_) => false,
        };

        let is_right_matrix_constant = match &self.right_matrix {
            OperandMatrix::Weigth(_) => true,
            OperandMatrix::Input(_) => false,
        };

        // there is only one product (i.e. quadratic sumcheck)
        let info = LayerCtx::MatMul(MatMulCtx {
            matrix_poly_id: id,
            matrix_poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                num_vars, num_vars,
            ]]),
            output_mle_num_vars: (nrows.ilog2() as usize, ncols.ilog2() as usize),
            is_left_matrix_constant,
            is_right_matrix_constant,
        });

        (info, ctx_aux)
    }
}

impl<E: ExtensionField> MatMulCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub(crate) fn verify_matmul<T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &MatMulProof<E>,
    ) -> Result<Vec<Claim<E>>> {
        let subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.sumcheck,
            &self.matrix_poly_aux,
            verifier.transcript,
        );

        // Verify claims about the matrix polynomials, for the constant input matrix (if any),
        // while claims about non-constant matrices are returned as output to be verified in
        // the next layer
        let mut output_claims = vec![];
        // check that there is at most 1 constant matrix
        ensure!(
            !(self.is_left_matrix_constant && self.is_right_matrix_constant),
            "Cannot have a MatMul layer with both constant matrices as input"
        );
        let (point_for_left, point_for_right) = MatMul::full_points(
            &last_claim,
            &subclaim.point_flat(),
            self.output_mle_num_vars,
        );
        // 0 because left matrix comes first in the product
        let eval_left = proof.individual_claims[0];
        let left_claim = Claim::new(point_for_left, eval_left);
        if self.is_left_matrix_constant {
            // we need to verify the polynomial commitment opening
            verifier
                .commit_verifier
                .add_claim(self.matrix_poly_id, left_claim)?
        } else {
            // add the claim to the output claims, to be verified in the next layer
            output_claims.push(left_claim)
        }
        // same for right matrix polynomial
        let eval_right = proof.individual_claims[1];
        let right_claim = Claim::new(point_for_right, eval_right);
        if self.is_right_matrix_constant {
            // we need to verify the polynomial commitment opening
            verifier
                .commit_verifier
                .add_claim(self.matrix_poly_id, right_claim)?
        } else {
            // add the claim to the output claims, to be verified in the next layer
            output_claims.push(right_claim)
        }

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
    use itertools::Itertools;

    use crate::{
        Element,
        layers::matrix_mul::{MatMul, OperandMatrix},
        quantization::Quantizer,
        tensor::Tensor,
    };

    #[test]
    fn test_matmul_pad_next_power_of_two() {
        // Create a Mat mul layer with non-power-of-two dimensions
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .unwrap();

        let input_shape = vec![5, matrix.nrows_2d()];

        let layer = MatMul::new(
            OperandMatrix::Input(input_shape),
            OperandMatrix::Weigth(matrix),
        )
        .unwrap();

        // Pad to next power of two
        let padded = layer.pad_next_power_of_two().unwrap();

        // Check padded dimensions are powers of two
        let padded_dims = padded.right_matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

        // Check input shape is padded
        let padded_input_shape = padded.left_matrix.get_shape();
        assert_eq!(padded_input_shape[0], 8); // Next power of 2 after 5
        assert_eq!(padded_input_shape[1], 4); // Next power of 2 after 3

        // Check original values are preserved
        let padded_matrix = if let OperandMatrix::Weigth(matrix) = &padded.right_matrix {
            matrix
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
    fn test_matmul_pad_already_power_of_two() {
        // Create a Dense layer with power-of-two dimensions
        let matrix = Tensor::<Element>::matix_from_coeffs(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ])
        .unwrap();
        let input_shape = vec![matrix.ncols_2d(), 2];
        let layer = MatMul::new(
            OperandMatrix::Weigth(matrix.clone()),
            OperandMatrix::Input(input_shape.clone()),
        )
        .unwrap();

        // Pad to next power of two
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Check dimensions remain the same
        assert_eq!(matrix.get_shape(), padded.left_matrix.get_shape());

        // Check input shape remain the same
        assert_eq!(input_shape, padded.right_matrix.get_shape());

        // Check values are preserved
        let padded_matrix = if let OperandMatrix::Weigth(matrix) = &padded.left_matrix {
            matrix
        } else {
            unreachable!()
        };
        let left_matrix = if let OperandMatrix::Weigth(matrix) = &layer.left_matrix {
            matrix
        } else {
            unreachable!()
        };
        assert_eq!(padded_matrix.get_data(), left_matrix.get_data());
    }

    #[test]
    fn test_matmul_pad_mixed_dimensions() {
        // Create a Dense layer with one power-of-two dimension and one non-power-of-two
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![
                9, 10, 11, 12,
            ]])
            .unwrap();

        let input_shape = vec![5, matrix.nrows_2d()];
        let layer = MatMul::new(
            OperandMatrix::Input(input_shape),
            OperandMatrix::Weigth(matrix),
        )
        .unwrap();

        // Pad to next power of two
        let padded = layer.pad_next_power_of_two().unwrap();

        // Check dimensions are padded correctly
        let padded_dims = padded.right_matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Already a power of 2

        // Check input shape is padded correctly
        let padded_input_shape = padded.left_matrix.get_shape();
        assert_eq!(padded_input_shape[0], 8); // Next power of 2 after 5
        assert_eq!(padded_input_shape[1], 4); // Next power of 2 after 3

        // Check original values are preserved and padding is zeros
        let padded_matrix = if let OperandMatrix::Weigth(matrix) = &padded.right_matrix {
            matrix
        } else {
            unreachable!()
        };
        assert_eq!(padded_matrix.get_data()[0], 1);
        assert_eq!(padded_matrix.get_data()[4], 5);
        assert_eq!(padded_matrix.get_data()[8], 9);
        assert_eq!(padded_matrix.get_data()[12], 0); // Padding
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
            .flat_map(|row| row.iter().map(|x| Element::from_f32_unsafe(x)))
            .collect();

        let layer = MatMul::new(
            OperandMatrix::Weigth(matrix),
            OperandMatrix::Input(input_shape.clone()),
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
}
