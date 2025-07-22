use std::{cmp::Ordering, collections::HashMap};

use crate::{
    Claim, Prover, ScalingStrategy,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{LayerCtx, LayerProof, requant::Requant},
    model::StepData,
    padding::{PaddingMode, ShapeInfo, pad_dense},
    quantization::{self, ScalingFactor, model_scaling_factor_from_tensor_and_bias},
    tensor::{Number, Shape},
};
use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::{trace, warn};
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
    VerifiableCtx,
};

/// Description of the layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense<T> {
    pub matrix: Tensor<T>,
    pub bias: Tensor<T>,
    // set to matrix shape if the matrix is not padded
    pub unpadded_matrix_shape: Shape,
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseCtx<E> {
    pub node_id: NodeId,
    pub matrix_poly_aux: VPAuxInfo<E>,
    pub unpadded_matrix_shape: Shape,
    pub padded_matrix_shape: Shape,
}

/// Proof of the layer.
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct DenseProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the mat2vec protocol
    pub(crate) sumcheck: IOPProof<E>,
    /// The evaluation of the bias at the previous claims in the proving flow.
    /// The verifier subtracts this from the previous claim to end up with one claim only
    /// about the matrix, without the bias.
    bias_eval: E,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

fn output_shape(input_shape: &Shape, matrix_shape: &Shape) -> Shape {
    assert_eq!(
        input_shape.product(),
        matrix_shape[1],
        "matrix_shape must be 2D: input_shape {input_shape:?} vs matrix {matrix_shape:?}"
    );
    Shape::new(vec![matrix_shape[0]])
}

impl<T: Number> Dense<T> {
    pub fn new(matrix: Tensor<T>, bias: Tensor<T>) -> Self {
        assert_eq!(matrix.nrows_2d(), bias.get_shape()[0]);
        let unpadded_matrix_shape = matrix.get_shape();
        Self {
            matrix,
            bias,
            unpadded_matrix_shape,
        }
    }
    pub fn ncols(&self) -> usize {
        self.matrix.ncols_2d()
    }
    pub fn nrows(&self) -> usize {
        self.matrix.nrows_2d()
    }

    pub fn pad_next_power_of_two(self) -> Self {
        let matrix = self.matrix.pad_next_power_of_two();
        let bias = self.bias.pad_1d(matrix.nrows_2d());
        Self {
            matrix,
            bias,
            unpadded_matrix_shape: self.unpadded_matrix_shape,
        }
    }

    pub fn output_shape(&self, input_shape: &Shape, padding_mode: PaddingMode) -> Shape {
        let matrix_shape = match padding_mode {
            PaddingMode::NoPadding => self.unpadded_matrix_shape.clone(),
            PaddingMode::Padding => self.unpadded_matrix_shape.next_power_of_two(),
        };
        output_shape(input_shape, &matrix_shape)
    }

    pub fn describe(&self) -> String {
        format!(
            "Dense: ({}x{}) + bias ({})",
            self.matrix.nrows_2d(),
            self.matrix.ncols_2d(),
            !self
                .bias
                .get_data()
                .iter()
                .all(|x| x.compare(&T::default()) == Ordering::Equal)
        )
    }

    fn num_outputs(num_inputs: usize) -> usize {
        assert_eq!(num_inputs, 1);
        1
    }
}

const IS_PROVABLE: bool = true;

impl<N: Number> OpInfo for Dense<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Self::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Dense: ({},{})",
            self.matrix.nrows_2d(),
            self.matrix.ncols_2d(),
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<N: Number> Evaluate<N> for Dense<N> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<N, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating dense layer"
        );
        let input = inputs[0];
        Ok(LayerOut::from_vec(vec![if input.get_shape().len() != 1 {
            let flat_input = input.flatten();
            assert!(
                flat_input.get_data().len() == self.matrix.ncols_2d(),
                "flat input length {} (from shape {:?}) vs matrix ncols {}",
                flat_input.get_data().len(),
                input.get_shape(),
                self.matrix.ncols_2d()
            );
            let matvec = self.matrix.matvec(&flat_input);
            matvec.add(&self.bias)
        } else {
            self.matrix.matvec(input).add(&self.bias)
        }]))
    }
}

const WEIGHT_POLY_ID: &str = "DenseWeight";
const BIAS_POLY_ID: &str = "DenseBias";

impl<E> ProveInfo<E> for Dense<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        // construct dimension of the polynomial given to the sumcheck
        let ncols = self.matrix.ncols_2d();
        aux.last_output_shape
            .iter_mut()
            .for_each(|shape| *shape = Shape::new(vec![self.matrix.nrows_2d()]));
        // each poly is only two polynomial right now: matrix and vector
        // for matrix, each time we fix the variables related to rows so we are only left
        // with the variables related to columns
        let matrix_num_vars = ncols.ilog2() as usize;
        let vector_num_vars = matrix_num_vars;
        // there is only one product (i.e. quadratic sumcheck)
        let dense_info = LayerCtx::Dense(DenseCtx {
            node_id: id,
            matrix_poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![
                matrix_num_vars,
                vector_num_vars,
            ]]),
            unpadded_matrix_shape: self.unpadded_matrix_shape.clone(),
            padded_matrix_shape: self.matrix.get_shape(),
        });

        let weights_evals = self.matrix.pad_next_power_of_two().get_data().to_vec();
        let bias_evals = self.bias.pad_next_power_of_two().get_data().to_vec();

        aux.model_polys = {
            let mut model_polys = HashMap::new();
            model_polys.insert(WEIGHT_POLY_ID.to_string(), weights_evals);
            model_polys.insert(BIAS_POLY_ID.to_string(), bias_evals);
            Some(model_polys)
        };
        Ok((dense_info, aux))
    }
}

impl PadOp for Dense<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_dense(self, si)
    }
}

impl Dense<f32> {
    // Quantize a dense layer using scaling factor of input and output
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Dense<Element>>> {
        let (model_scaling, bias_scaling) =
            model_scaling_factor_from_tensor_and_bias(&input_scaling[0], &self.matrix, &self.bias);
        ensure!(
            input_scaling.len() == 1,
            "Number of input scaling factor for dense layer different from 1"
        );
        let input_scaling = &input_scaling[0];
        let quantized_dense = self.quantize(&model_scaling, &bias_scaling);
        let intermediate_bit_size = quantized_dense.output_bitsize();
        let requant = Requant::from_scaling_factors(
            *input_scaling,
            model_scaling,
            output_scaling,
            intermediate_bit_size,
        );

        Ok(QuantizeOutput::new(quantized_dense, vec![output_scaling]).with_requant(requant))
    }
}

impl QuantizeOp for Dense<f32> {
    type QuantizedOp = Dense<Element>;

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

impl<E, PCS> ProvableOp<E, PCS> for Dense<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = DenseCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.prove_step(
            prover,
            last_claims[0],
            &step_data.inputs[0],
            step_data.outputs.outputs()[0],
            ctx,
            id,
        )?])
    }
}

impl<E> OpInfo for DenseCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Dense::<Element>::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Dense: ({},{})",
            self.padded_matrix_shape[0], self.padded_matrix_shape[1],
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for DenseCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = DenseProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.verify_dense(verifier, last_claims[0], proof)?])
    }
}

impl Dense<f32> {
    /// Quantize the parameters of the dense layer. It uses a custom scaling factor `bias_s` for
    /// the bias, if provided, otherwise the same scaling factor of the weights (i.e., `s`) is used
    pub fn quantize(self, s: &ScalingFactor, bias_s: &ScalingFactor) -> Dense<Element> {
        let matrix = self.matrix.quantize(s);
        let bias = self.bias.quantize(bias_s);
        Dense::<Element> {
            matrix,
            bias,
            unpadded_matrix_shape: self.unpadded_matrix_shape,
        }
    }

    pub fn new_from_weights(weights: Tensor<f32>, bias: Tensor<f32>) -> Self {
        let unpadded_matrix_shape = weights.get_shape();
        Self {
            matrix: weights,
            bias,
            unpadded_matrix_shape,
        }
    }

    /// TODO: compute two different scaling factors for weights and bias
    pub fn max_abs_weight(&self) -> f32 {
        let max_weight = self.matrix.max_abs_output();
        let max_bias = self.bias.max_abs_output();
        let distance = (max_weight - max_bias).abs() / max_weight;
        if distance > 0.1 {
            warn!(
                "max_abs_weight DENSE: distance between max_weight and max_bias is too large: {:.2}%",
                distance * 100.0
            );
        }
        self.matrix.max_abs_output().max(self.bias.max_abs_output())
    }
}

impl Dense<Element> {
    /// Returns the (min,max) output range of the dense layer for a given input range.
    pub fn output_range(&self, _min_input: Element, _max_input: Element) -> (Element, Element) {
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.matrix.ncols_2d() as u32;
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        let power = 2 * (*quantization::BIT_LEN as u32 - 1) + ncols.ilog2() + 1;
        let min = -(2u64.pow(power) as Element);
        let max = 2u64.pow(power) as Element;
        (min, max)
    }

    /// Returns the maximum size in bits of the output
    pub fn output_bitsize(&self) -> usize {
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.matrix.ncols_2d();
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        2 * (*quantization::BIT_LEN - 1) + ceil_log2(ncols) + 1
    }

    #[timed::timed_instrument(name = "Prover::prove_dense")]
    pub fn prove_step<'b, E, T, PCS>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        input: &Tensor<E>,
        output: &Tensor<E>,
        _info: &DenseCtx<E>,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let matrix = &self.matrix;
        let (nrows, ncols) = (matrix.nrows_2d(), matrix.ncols_2d());
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
        // Evaluates the bias at the random point so verifier can subtract the evaluation
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
        // construct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_2d_mle();
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
        // add the bias claim over the last claim input, since that is what is needed to "remove" the bias
        // to only verify the matrix2vec product via the sumcheck proof.
        let bias_claim = Claim::new(last_claim.point.clone(), bias_eval);
        let weights_claim = Claim::new(point, eval);

        // Add common commitment claims to be proven
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(WEIGHT_POLY_ID.to_string(), weights_claim);
            claims.insert(BIAS_POLY_ID.to_string(), bias_claim);
            claims
        };
        prover.add_common_claims(id, common_claims)?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        prover.push_proof(
            id,
            LayerProof::Dense(DenseProof {
                sumcheck: proof,
                bias_eval,
                individual_claims: state.get_mle_final_evaluations(),
            }),
        );
        Ok(claim)
    }
}

impl<E: ExtensionField> DenseCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn output_shape(&self, input_shape: &Shape, mode: PaddingMode) -> Shape {
        let mat_shape = match mode {
            PaddingMode::NoPadding => &self.unpadded_matrix_shape,
            PaddingMode::Padding => &self.padded_matrix_shape,
        };
        output_shape(input_shape, mat_shape)
    }
    pub(crate) fn verify_dense<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &DenseProof<E>,
    ) -> anyhow::Result<Claim<E>> {
        let info = self;
        // Subtract the bias evaluation from the previous claim to remove the bias
        let eval_no_bias = last_claim.eval - proof.bias_eval;
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

        let bias_claim = Claim::new(last_claim.point.clone(), proof.bias_eval);
        let weights_claim = Claim::new(pcs_eval_input, pcs_eval_output);

        // add the common commitment claims to be verified
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(WEIGHT_POLY_ID.to_string(), weights_claim);
            claims.insert(BIAS_POLY_ID.to_string(), bias_claim);
            claims
        };
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
        self.individual_claims
            .iter()
            .fold(E::ONE, |acc, e| acc * *e)
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::layers::provable::evaluate_layer;

    use super::*;

    impl<T: Number> Dense<T> {
        pub fn random(shape: Shape) -> Self {
            assert_eq!(shape.len(), 2);
            let (nrows, ncols) = (shape[0], shape[1]);
            let matrix = Tensor::<T>::random(&vec![nrows, ncols].into());
            // let bias = Tensor::random(vec![nrows]);
            let bias = Tensor::<T>::random(&vec![nrows].into());
            Self::new(matrix, bias)
        }
    }

    #[test]
    fn test_dense_pad_next_power_of_two() {
        // Create a Dense layer with non-power-of-two dimensions
        let matrix = Tensor::<Element>::matrix_from_coeffs(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ])
        .unwrap();

        let bias = Tensor::<Element>::new(vec![3].into(), vec![10, 11, 12]);

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.pad_next_power_of_two();

        // Check padded dimensions are powers of two
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

        // Check bias is padded
        let bias_dims = padded.bias.get_shape();
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
        let matrix = Tensor::<Element>::matrix_from_coeffs(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ])
        .unwrap();

        let bias = Tensor::<Element>::new(vec![4].into(), vec![20, 21, 22, 23]);

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.clone().pad_next_power_of_two();

        // Check dimensions remain the same
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4);
        assert_eq!(padded_dims[1], 4);

        // Check bias dimensions remain the same
        let bias_dims = padded.bias.get_shape();
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
        let matrix = Tensor::<Element>::matrix_from_coeffs(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
        ])
        .unwrap();

        let bias = Tensor::<Element>::new(vec![3].into(), vec![20, 21, 22]);

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.pad_next_power_of_two();

        // Check dimensions are padded correctly
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Already a power of 2

        // Check bias is padded
        let bias_dims = padded.bias.get_shape();
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
            .map(|x| ScalingFactor::default().quantize(x))
            .collect();

        // Create a Dense layer
        let matrix =
            Tensor::<Element>::matrix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

        let bias = Tensor::<Element>::new(vec![2].into(), vec![10, 11]);

        let dense = Dense::new(matrix, bias);

        // Pad the dense layer
        let padded = dense.clone().pad_next_power_of_two();

        // Create input tensor
        let input_tensor = Tensor::<Element>::new(vec![3].into(), quantized_input);

        // Apply the dense operation on both original and padded
        let output = evaluate_layer::<GoldilocksExt2, _, _>(&dense, &vec![&input_tensor], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let padded_output =
            evaluate_layer::<GoldilocksExt2, _, _>(&padded, &vec![&input_tensor.pad_1d(4)], None)
                .unwrap()
                .outputs()[0]
                .clone();

        // Check that the result is correct (for the non-padded parts)
        for i in 0..2 {
            assert_eq!(output.get_data()[i], padded_output.get_data()[i]);
        }
    }
}
