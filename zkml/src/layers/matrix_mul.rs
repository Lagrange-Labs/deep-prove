use anyhow::{Context, Result, ensure};

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

/// Description of the layer
#[derive(Clone, Debug)]
pub struct MatMul {
    pub(crate) matrix: Tensor<Element>,
    pub(crate) input_shape: (usize, usize),
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatMulCtx<E> {
    pub(crate) matrix_poly_id: PolyID,
    pub(crate) matrix_poly_aux: VPAuxInfo<E>,
    // Number of variables of the MLE polynomial for each dimension of the output matrix
    pub(crate) output_mle_num_vars: (usize, usize),
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
    pub fn new(matrix: Tensor<Element>, input_shape: Vec<usize>) -> Result<Self> {
        ensure!(
            input_shape.len() == 2,
            "Input shape does not correspond to a matrix"
        );
        ensure!(
            matrix.is_matrix(),
            "Matrix provided for MatMul layer is not a matrix"
        );
        ensure!(
            input_shape[1] == matrix.nrows_2d(),
            "Number of columns in input different from number of rows of matrix: {} != {}",
            input_shape[1],
            matrix.nrows_2d(),
        );
        Ok(Self {
            matrix,
            input_shape: (input_shape[0], input_shape[1]),
        })
    }

    pub fn ncols(&self) -> usize {
        self.matrix.ncols_2d()
    }

    pub fn nrows(&self) -> usize {
        self.matrix.nrows_2d()
    }

    pub fn input_shape(&self) -> Vec<usize> {
        vec![self.input_shape.0, self.input_shape.1]
    }

    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        input.matmul(&self.matrix)
    }

    pub fn pad_next_power_of_two(self) -> Result<Self> {
        let matrix = self.matrix.pad_next_power_of_two();
        // Pad also the shape of the input matrix
        let input_rows = self.input_shape.0.next_power_of_two();
        let input_cols = matrix.nrows_2d();

        Self::new(matrix, vec![input_rows, input_cols])
    }

    pub fn requant_info(&self) -> Requant {
        let ncols = self.matrix.ncols_2d();
        let max_output_range = self
            .matrix
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
    }

    /// Method to split the point of a claim computed for the output matrix MLE among the coordinates
    /// for the input matrxi and for the layer matrix, which are returned as output.
    /// `output_num_vars` specifies the number of variables for each dimension of the output matrix
    fn split_claim<E: ExtensionField>(
        claim: &Claim<E>,
        output_num_vars: (usize, usize),
    ) -> (&[E], &[E]) {
        let num_vars_cols = output_num_vars.1;
        // the coordinates of `last_claim` point employed to partially evaluate the
        // input matrix MLE are the ones corresponding to the rows of the output matrix;
        // therefore, these correspond to the high variables because  the MLE is addressing
        // in little endian so (rows,cols) is actually given in (cols, rows)
        let point_for_input = &claim.point[num_vars_cols..];
        // the coordinates of `last_claim` point employed to partially evaluate the
        // layer matrix MLE are the ones corresponding to the columns of the output matrix;
        // therefore, these correspond to the low variables because  the MLE is addressing
        // in little endian so (rows,cols) is actually given in (cols, rows)
        let point_for_mat = &claim.point[..num_vars_cols];

        (point_for_input, point_for_mat)
    }

    /// Construct the full point (i.e., with all the variables) over which the input matrix and the layer
    /// matrix are evaluated in the sumcheck proof. This method requires the following inputs:
    /// - `claim`: claim computed for the output matrix MLE (input claim for the sumcheck)
    /// - `proof_point`: point employed in the sumcheck proof
    /// - `output_num_vars`: number of variables for each dimension of the output matrix
    fn full_points<E: ExtensionField>(
        claim: &Claim<E>,
        proof_point: &[E],
        output_num_vars: (usize, usize),
    ) -> (Vec<E>, Vec<E>) {
        let (claim_point_for_input, claim_point_for_mat) =
            Self::split_claim(claim, output_num_vars);
        let point_for_mat = [claim_point_for_mat, proof_point].concat();
        let point_for_input = [proof_point, claim_point_for_input].concat();
        (point_for_input, point_for_mat)
    }

    pub fn prove_step<E, T>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: Claim<E>,
        input: &Tensor<E>,
        output: &Tensor<E>,
        info: &MatMulCtx<E>,
    ) -> Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        let matrix = &self.matrix;
        let ncols_matrix = matrix.ncols_2d();
        ensure!(
            input.is_matrix(),
            "Input tensor for MatMul layer is not a matrix"
        );
        let input_shape = input.get_shape();
        ensure!(
            input_shape == vec![self.input_shape.0, self.input_shape.1],
            "Input tensor for MatMul layer has wrong shape: expected ({}, {}), found ({}, {})",
            self.input_shape.0,
            self.input_shape.1,
            input_shape[0],
            input_shape[1]
        );
        ensure!(
            output.is_matrix(),
            "Output tensor for MatMul layer is not a matrix"
        );
        let (nrows_out, ncols_out) = (output.nrows_2d(), output.ncols_2d());
        ensure!(
            nrows_out == self.input_shape.0,
            "Wrong number of rows in output matrix: expected {}, found {}",
            self.input_shape.0,
            nrows_out,
        );
        ensure!(
            ncols_out == ncols_matrix,
            "Wrong number of columns in output matrix: expected {}, found {}",
            ncols_matrix,
            ncols_out,
        );
        let (num_vars_row, num_vars_cols) = output.num_vars_2d();
        let num_vars_out = num_vars_row + num_vars_cols;
        ensure!(
            num_vars_out == last_claim.point.len(),
            "Wrong length of last claim point: expected {}, found {}",
            num_vars_out,
            last_claim.point.len()
        );

        // construct the MLE combining the input and the matrix
        let mut mat_mle: DenseMultilinearExtension<E> = matrix.to_2d_mle();
        let mut input_mle = input.to_mle_2d();
        let (point_for_input, point_for_mat) =
            Self::split_claim(&last_claim, (num_vars_row, num_vars_cols));
        // fix the variables for the random input matrix; we need to fix the variables
        // corresponding to a row, so we must fix the HIGH variables
        input_mle.fix_high_variables_in_place(point_for_input);
        // fix the variables for the layer matrix; we need to fix the variables
        // corresponding to a column, so we must fix the low variables
        mat_mle.fix_variables_in_place(point_for_mat);

        // check that after fixing the variables in both matrixes the number of free
        // variables is the same
        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());

        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let (point_for_input, point_for_mat) =
            Self::full_points(&last_claim, &proof.point, (num_vars_row, num_vars_cols));
        let eval = state.get_mle_final_evaluations()[0]; // The first MLE being evaluated is the matrix poly
        prover
            .commit_prover
            .add_claim(info.matrix_poly_id, Claim::new(point_for_mat, eval))
            .context("unable to add matrix claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: point_for_input,
            eval: state.get_mle_final_evaluations()[1],
        };
        prover.push_proof(LayerProof::MatMul(MatMulProof {
            sumcheck: proof,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
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
        let ncols = self.matrix.ncols_2d();
        let nrows = self.input_shape.0;
        ctx_aux.last_output_shape = vec![nrows, ncols];

        // number of variables of the MLE polynomials is the number of rows
        // in layer matrix
        let num_vars = self.matrix.num_vars_2d().0;

        // there is only one product (i.e. quadratic sumcheck)
        let info = LayerCtx::MatMul(MatMulCtx {
            matrix_poly_id: id,
            matrix_poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                num_vars, num_vars,
            ]]),
            output_mle_num_vars: (nrows.ilog2() as usize, ncols.ilog2() as usize),
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
    ) -> Result<Claim<E>> {
        let subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.sumcheck,
            &self.matrix_poly_aux,
            verifier.transcript,
        );

        // PCS opening for layer matrix
        let (point_for_input, point_for_matrix) = MatMul::full_points(
            &last_claim,
            &subclaim.point_flat(),
            self.output_mle_num_vars,
        );
        // 0 because Matrix comes first in Matrix x Input
        // Note we don't care about verifying that for the input matrix since it's verified at the next
        // step.
        let pcs_eval_output = proof.individual_claims[0];
        verifier.commit_verifier.add_claim(
            self.matrix_poly_id,
            Claim::new(point_for_matrix, pcs_eval_output),
        )?;

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
        Ok(Claim {
            // the new randomness to fix at next layer is the randomness from the sumcheck !
            point: point_for_input,
            // the claimed sum for the next sumcheck is MLE of the current input matrix evaluated at the
            // random point. 1 because input matrix is the right hand side of the product.
            eval: proof.individual_claims[1],
        })
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

    use crate::{Element, layers::matrix_mul::MatMul, quantization::Quantizer, tensor::Tensor};

    #[test]
    fn test_matmul_pad_next_power_of_two() {
        // Create a Mat mul layer with non-power-of-two dimensions
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .unwrap();

        let input_shape = vec![5, matrix.nrows_2d()];

        let layer = MatMul::new(matrix, input_shape).unwrap();

        // Pad to next power of two
        let padded = layer.pad_next_power_of_two().unwrap();

        // Check padded dimensions are powers of two
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

        // Check input shape is padded
        let (input_rows, input_cols) = padded.input_shape;
        assert_eq!(input_rows, 8); // Next power of 2 after 5
        assert_eq!(input_cols, 4); // Next power of 2 after 3

        // Check original values are preserved
        assert_eq!(padded.matrix.get_data()[0], 1);
        assert_eq!(padded.matrix.get_data()[1], 2);
        assert_eq!(padded.matrix.get_data()[2], 3);
        assert_eq!(padded.matrix.get_data()[4], 4);
        assert_eq!(padded.matrix.get_data()[8], 7);

        // Check added values are zeros
        assert_eq!(padded.matrix.get_data()[3], 0);
        assert_eq!(padded.matrix.get_data()[7], 0);
        assert_eq!(padded.matrix.get_data()[15], 0);
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
        let input_shape = vec![2, matrix.nrows_2d()];
        let layer = MatMul::new(matrix.clone(), input_shape.clone()).unwrap();

        // Pad to next power of two
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Check dimensions remain the same
        assert_eq!(matrix.get_shape(), padded.matrix.get_shape());

        // Check input shape remain the same
        assert_eq!((input_shape[0], input_shape[1]), padded.input_shape);

        // Check values are preserved
        for i in 0..16 {
            assert_eq!(padded.matrix.get_data()[i], layer.matrix.get_data()[i]);
        }
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
        let layer = MatMul::new(matrix, input_shape).unwrap();

        // Pad to next power of two
        let padded = layer.pad_next_power_of_two().unwrap();

        // Check dimensions are padded correctly
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Already a power of 2

        // Check input shape is padded correctly
        let (input_rows, input_cols) = padded.input_shape;
        assert_eq!(input_rows, 8); // Next power of 2 after 5
        assert_eq!(input_cols, 4); // Next power of 2 after 3

        // Check original values are preserved and padding is zeros
        assert_eq!(padded.matrix.get_data()[0], 1);
        assert_eq!(padded.matrix.get_data()[4], 5);
        assert_eq!(padded.matrix.get_data()[8], 9);
        assert_eq!(padded.matrix.get_data()[12], 0); // Padding
    }

    #[test]
    fn test_quantization_with_padded_matmul() {
        // Create input data that needs quantization
        let rng = &mut thread_rng();
        let input_data = [0; 5].map(|_| [0; 3].map(|_| rng.gen_range(-1.0f32..1.0f32)));

        // Quantize the input
        let quantized_input: Vec<Element> = input_data
            .iter()
            .flat_map(|row| row.iter().map(|x| Element::from_f32_unsafe(x)))
            .collect();

        // Create a matrix multiplication layer
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .unwrap();

        let input_shape = vec![input_data.len(), matrix.nrows_2d()];
        let layer = MatMul::new(matrix, input_shape.clone()).unwrap();

        // Pad the layer
        let padded = layer.clone().pad_next_power_of_two().unwrap();

        // Create input tensor
        let input_tensor = Tensor::<Element>::new(input_shape, quantized_input);

        // Apply the layer operation on both original and padded
        let output = layer.op(&input_tensor);
        let padded_output = padded.op(&input_tensor.pad_next_power_of_two_2d());

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
