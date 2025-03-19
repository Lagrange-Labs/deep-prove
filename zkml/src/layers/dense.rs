use crate::{
    Claim, Prover,
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerCtx, LayerProof, PolyID, requant::Requant},
    quantization,
};
use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::{debug, trace};
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

/// Bias to compute the bias ID polynomials. Since originally we take the index of each
/// layer to be the index of the layer, we need to add a bias to avoid collision with other
/// layers poly id.
pub(crate) const BIAS_POLY_ID: PolyID = 100_000;

/// Description of the layer
#[derive(Clone, Debug)]
pub struct Dense {
    pub matrix: Tensor<Element>,
    pub bias: Tensor<Element>,
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseCtx<E> {
    pub matrix_poly_id: PolyID,
    pub matrix_poly_aux: VPAuxInfo<E>,
    pub bias_poly_id: PolyID,
}

/// Proof of the layer.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct DenseProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the mat2vec protocol
    pub(crate) sumcheck: IOPProof<E>,
    /// The evaluation of the bias at the previous claims in the proving flow.
    /// The verifier substracts this from the previous claim to end up with one claim only
    /// about the matrix, without the bias.
    bias_eval: E,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

impl Dense {
    pub fn new(matrix: Tensor<Element>, bias: Tensor<Element>) -> Self {
        assert_eq!(matrix.nrows_2d(), bias.dims()[0]);
        Self { matrix, bias }
    }
    pub fn ncols(&self) -> usize {
        self.matrix.ncols_2d()
    }
    pub fn nrows(&self) -> usize {
        self.matrix.nrows_2d()
    }

    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        if input.dims().len() != 1 {
            let flat_input = input.flatten();
            self.matrix.matvec(&flat_input).add(&self.bias)
        } else {
            self.matrix.matvec(input).add(&self.bias)
        }
    }

    pub fn pad_next_power_of_two(self) -> Self {
        let matrix = self.matrix.pad_next_power_of_two_2d();
        let bias = self.bias.pad_1d(matrix.nrows_2d());
        Self::new(matrix, bias)
    }

    pub fn requant_info(&self) -> Requant {
        let ncols = self.matrix.ncols_2d();
        let max_output_range = self
            .matrix
            .get_data()
            .iter()
            .chunks(ncols)
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let row_range = row
                    .map(|w| quantization::range_from_weight(w))
                    .fold((0, 0), |(min, max), (wmin, wmax)| (min + wmin, max + wmax));
                // add the bias range - so take the weight corresponding to the row index
                let bias_weight = &self.bias.get_data()[i];
                let total_range = (row_range.0 + bias_weight, row_range.1 + bias_weight);
                // weight * MIN can be positive and higher then MAX*weight if weight's negative
                // so we take the absolute value of the difference
                (total_range.1 - total_range.0).unsigned_abs() as usize
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
    pub fn prove_step<'b, E, T>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: Claim<E>,
        input: &Tensor<E>,
        output: &Tensor<E>,
        info: &DenseCtx<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        let matrix = &self.matrix;
        let (nrows, ncols) = (matrix.nrows_2d(), matrix.ncols_2d());
        debug!("dense proving nrows: {} ncols: {}", nrows, ncols);
        assert_eq!(
            nrows,
            output.get_data().len(),
            "dense proving: nrows {} vs output {}",
            nrows,
            output.get_data().len()
        );
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(
            ncols,
            input.get_data().len(),
            "something's wrong with the input"
        );
        // Evaluates the bias at the random point so verifier can substract the evaluation
        // from the sumcheck claim that is only about the matrix2vec product.
        assert_eq!(
            self.bias.get_data().len().ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        let bias_eval = self
            .bias
            .evals_flat::<E>()
            .into_mle()
            .evaluate(&last_claim.point);
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle_2d();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = input.get_data().to_vec().into_mle();

        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let tmp_transcript = prover.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        debug_assert!({
            let mut t = tmp_transcript;
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(num_vars);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = output.get_data().to_vec().into_mle();
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            let claimed_sum_no_bias = claimed_sum - bias_eval;
            debug_assert_eq!(claimed_sum, last_claim.eval, "sumcheck eval weird");
            debug_assert_eq!(
                claimed_sum_no_bias,
                proof.extract_sum(),
                "sumcheck output weird"
            );

            trace!("prover: claimed sum: {:?}", claimed_sum);
            let subclaim =
                IOPVerifierState::<E>::verify(claimed_sum_no_bias, &proof, &vp.aux_info, &mut t);
            // now assert that the polynomial evaluated at the random point of the sumcheck proof
            // is equal to last small poly sent by prover (`subclaim.expected_evaluation`). This
            // step can be done via PCS opening proofs for all steps but first (output of
            // inference) and last (input of inference)
            let computed_point = vp.evaluate(subclaim.point_flat().as_ref());

            let final_prover_point = state
                .get_mle_final_evaluations()
                .into_iter()
                .fold(E::ONE, |acc, eval| acc * eval);
            assert_eq!(computed_point, final_prover_point);

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // is done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let point = [proof.point.as_slice(), last_claim.point.as_slice()].concat();
        let eval = state.get_mle_final_evaluations()[0];
        prover
            .commit_prover
            .add_claim(info.matrix_poly_id, Claim::new(point, eval))
            .context("unable to add matrix claim")?;
        // add the bias claim over the last claim input, since that is what is needed to "remove" the bias
        // to only verify the matrix2vec product via the sumcheck proof.
        prover
            .commit_prover
            .add_claim(info.bias_poly_id, Claim::new(last_claim.point, bias_eval))
            .context("unable to add bias claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        prover.push_proof(LayerProof::Dense(DenseProof {
            sumcheck: proof,
            bias_eval,
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
        ctx_aux.last_output_shape = vec![self.matrix.nrows_2d()];
        // each poly is only two polynomial right now: matrix and vector
        // for matrix, each time we fix the variables related to rows so we are only left
        // with the variables related to columns
        let matrix_num_vars = ncols.ilog2() as usize;
        let vector_num_vars = matrix_num_vars;
        // there is only one product (i.e. quadratic sumcheck)
        let dense_info = LayerCtx::Dense(DenseCtx {
            matrix_poly_id: id,
            matrix_poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                matrix_num_vars,
                vector_num_vars,
            ]]),
            bias_poly_id: BIAS_POLY_ID + id,
        });
        (dense_info, ctx_aux)
    }
}

impl<E: ExtensionField> DenseCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub(crate) fn verify_dense<T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &DenseProof<E>,
    ) -> anyhow::Result<Claim<E>> {
        let info = self;
        // Subtract the bias evaluation from the previous claim to remove the bias
        let eval_no_bias = last_claim.eval - proof.bias_eval;
        debug!("VERIFIER: claim {:?}", last_claim);
        // TODO: currently that API can panic - should remove panic for error
        let subclaim = IOPVerifierState::<E>::verify(
            eval_no_bias,
            &proof.sumcheck,
            &info.matrix_poly_aux,
            verifier.transcript,
        );

        // MATRIX OPENING PART
        // pcs_eval means this evaluation should come from a PCS opening proof
        let pcs_eval_input = subclaim
            .point_flat()
            .iter()
            .chain(last_claim.point.iter())
            .cloned()
            .collect_vec();
        // 0 because Matrix comes first in Matrix x Vector
        // Note we don't care about verifying that for the vector since it's verified at the next
        // step.
        let pcs_eval_output = proof.individual_claims[0];
        verifier.commit_verifier.add_claim(
            info.matrix_poly_id,
            Claim::new(pcs_eval_input, pcs_eval_output),
        )?;
        verifier.commit_verifier.add_claim(
            info.bias_poly_id,
            Claim::new(last_claim.point, proof.bias_eval),
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
            point: subclaim.point_flat(),
            // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
            // random point. 1 because vector is secondary.
            eval: proof.individual_claims[1],
        })
    }
}

impl<E: ExtensionField> DenseProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl Dense {
        pub fn random(shape: Vec<usize>) -> Self {
            assert_eq!(shape.len(), 2);
            let (nrows, ncols) = (shape[0], shape[1]);
            let matrix = Tensor::random(vec![nrows, ncols]);
            let bias = Tensor::random(vec![nrows]);
            Self::new(matrix, bias)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::quantization::Quantizer;

        #[test]
        fn test_dense_pad_next_power_of_two() {
            // Create a Dense layer with non-power-of-two dimensions
            let matrix =
                Tensor::<Element>::from_coeffs_2d(vec![vec![1, 2, 3], vec![4, 5, 6], vec![
                    7, 8, 9,
                ]])
                .unwrap();

            let bias = Tensor::<Element>::new(vec![3], vec![10, 11, 12]);

            let dense = Dense::new(matrix, bias);

            // Pad to next power of two
            let padded = dense.pad_next_power_of_two();

            // Check padded dimensions are powers of two
            let padded_dims = padded.matrix.dims();
            assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
            assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

            // Check bias is padded
            let bias_dims = padded.bias.dims();
            assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

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

            // Check bias values
            assert_eq!(padded.bias.get_data()[0], 10);
            assert_eq!(padded.bias.get_data()[1], 11);
            assert_eq!(padded.bias.get_data()[2], 12);
            assert_eq!(padded.bias.get_data()[3], 0); // Padding
        }

        #[test]
        fn test_dense_pad_already_power_of_two() {
            // Create a Dense layer with power-of-two dimensions
            let matrix = Tensor::<Element>::from_coeffs_2d(vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
            ])
            .unwrap();

            let bias = Tensor::<Element>::new(vec![4], vec![20, 21, 22, 23]);

            let dense = Dense::new(matrix, bias);

            // Pad to next power of two
            let padded = dense.clone().pad_next_power_of_two();

            // Check dimensions remain the same
            let padded_dims = padded.matrix.dims();
            assert_eq!(padded_dims[0], 4);
            assert_eq!(padded_dims[1], 4);

            // Check bias dimensions remain the same
            let bias_dims = padded.bias.dims();
            assert_eq!(bias_dims[0], 4);

            // Check values are preserved
            for i in 0..16 {
                assert_eq!(padded.matrix.get_data()[i], dense.matrix.get_data()[i]);
            }

            for i in 0..4 {
                assert_eq!(padded.bias.get_data()[i], dense.bias.get_data()[i]);
            }
        }

        #[test]
        fn test_dense_pad_mixed_dimensions() {
            // Create a Dense layer with one power-of-two dimension and one non-power-of-two
            let matrix =
                Tensor::<Element>::from_coeffs_2d(vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![
                    9, 10, 11, 12,
                ]])
                .unwrap();

            let bias = Tensor::<Element>::new(vec![3], vec![20, 21, 22]);

            let dense = Dense::new(matrix, bias);

            // Pad to next power of two
            let padded = dense.pad_next_power_of_two();

            // Check dimensions are padded correctly
            let padded_dims = padded.matrix.dims();
            assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
            assert_eq!(padded_dims[1], 4); // Already a power of 2

            // Check bias is padded
            let bias_dims = padded.bias.dims();
            assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

            // Check original values are preserved and padding is zeros
            assert_eq!(padded.matrix.get_data()[0], 1);
            assert_eq!(padded.matrix.get_data()[4], 5);
            assert_eq!(padded.matrix.get_data()[8], 9);
            assert_eq!(padded.matrix.get_data()[12], 0); // Padding

            // Check bias values
            assert_eq!(padded.bias.get_data()[0], 20);
            assert_eq!(padded.bias.get_data()[1], 21);
            assert_eq!(padded.bias.get_data()[2], 22);
            assert_eq!(padded.bias.get_data()[3], 0); // Padding
        }

        #[test]
        fn test_quantization_with_padded_dense() {
            // Create input data that needs quantization
            let input_data = vec![0.5f32, -0.3f32, 0.1f32];

            // Quantize the input
            let quantized_input: Vec<Element> = input_data
                .iter()
                .map(|x| Element::from_f32_unsafe(x))
                .collect();

            // Create a Dense layer
            let matrix =
                Tensor::<Element>::from_coeffs_2d(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

            let bias = Tensor::<Element>::new(vec![2], vec![10, 11]);

            let dense = Dense::new(matrix, bias);

            // Pad the dense layer
            let padded = dense.clone().pad_next_power_of_two();

            // Create input tensor
            let input_tensor = Tensor::<Element>::new(vec![3], quantized_input);

            // Apply the dense operation on both original and padded
            let output = dense.op(&input_tensor);
            let padded_output = padded.op(&input_tensor.pad_1d(4));

            // Check that the result is correct (for the non-padded parts)
            for i in 0..2 {
                assert_eq!(output.get_data()[i], padded_output.get_data()[i]);
            }
        }
    }
}
