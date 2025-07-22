use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::{Challenge, Transcript};

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Tensor,
    commit::same_poly,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
        requant::Requant,
    },
    model::StepData,
    padding::{PaddingMode, ShapeInfo, pad_qkv},
    quantization::model_scaling_factor_from_tensor_and_bias,
    tensor::{Number, Shape},
    try_unzip, try_unzip_parallel,
};

/// A layer that evaluates the tensor X against the matrices Q, K and V.
/// NOTE: it performs optimizations with the cache, so it actually
/// do the matrix mult only with the last entry of the input.
/// It also outputs only the "small" Q but with the help of caching, it outputs
/// the full K and V matrices as if they were computed using the whole input tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKV<N> {
    pub(crate) q: Tensor<N>,
    pub(crate) q_bias: Tensor<N>,
    pub(crate) k: Tensor<N>,
    pub(crate) k_bias: Tensor<N>,
    pub(crate) v: Tensor<N>,
    pub(crate) v_bias: Tensor<N>,
    unpadded_shape: Shape,       /* same shape for Q, K and V
                                  * pub cache: Option<CacheQKV<N>>, */
    pub(crate) num_heads: usize, // Needed to properly pad matrices for sub-sequent MHA layer
    pub(crate) head_dim: usize,  // Needed to properly pad matrices for sub-sequent MHA layer
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QKVCtx<E> {
    node_id: NodeId,
    sumcheck_poly_aux: VPAuxInfo<E>,
    unpadded_shape: Shape, // same shape for Q, K and V
    num_heads: usize,
    head_dim: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct QKVProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the QKV matrix multiplications
    pub(crate) sumcheck: IOPProof<E>,
    /// Proof for the aggregation of the claims about the input matrix to
    /// a single claim
    aggregation_proof: same_poly::Proof<E>,
    /// Th  evaluation of the weight MLEs over the input vector, without the bias.
    /// The verifier needs these evaluations to check the output of the sumcheck proof
    pre_bias_evals: Vec<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly".
    /// There is a pair of evaluations for each output matrix `Q`, `K` and `V`:
    /// the first evaluation in the pair refers to the input matrix MLE, while the second evaluation
    /// in the pair refers to the corresponding weight matrix `W_q`, `W_k`, `W_v`, respectively.
    /// The first pair contains the evaluations for `Q`, the second pair contains the evaluations
    /// for `K`, and the third pair contains the evaluations for `V`
    individual_claims: [(E, E); 3],
}

impl<E: ExtensionField> QKVProof<E> {
    /// Returns the aggregated sumcheck claims `y = f_1(r) * f_2(r) * f_3(r) ...` from the individual claims.
    pub fn individual_to_virtual_claim(&self, batching_challenges: &[Challenge<E>]) -> E {
        self.individual_claims
            .into_iter()
            .zip(batching_challenges)
            .fold(E::ZERO, |acc, (evals, chal)| {
                acc + evals.0 * evals.1 * chal.elements
            })
    }
}

fn padded_weight_shape(unpadded_shape: &Shape, num_heads: usize, head_dim: usize) -> Shape {
    Shape::new(vec![
        unpadded_shape[0].next_power_of_two(),
        head_dim.next_power_of_two() * num_heads.next_power_of_two(),
    ])
}

impl<N: Number> QKV<N> {
    pub fn new(
        q: Tensor<N>,
        q_bias: Tensor<N>,
        k: Tensor<N>,
        k_bias: Tensor<N>,
        v: Tensor<N>,
        v_bias: Tensor<N>,
        num_heads: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            q.get_shape() == k.get_shape(),
            "Incompatible shapes of q and k tensors in QKV layer: q = {:?}, k = {:?}",
            q.get_shape(),
            k.get_shape(),
        );
        ensure!(
            q.get_shape() == v.get_shape(),
            "Incompatible shapes of q and v tensors in QKV layer: q = {:?}, v = {:?}",
            q.get_shape(),
            v.get_shape(),
        );
        ensure!(
            q_bias.get_shape().len() == 1,
            "Bias in QKV layer is not a 1d tensor"
        );
        ensure!(
            q_bias.get_shape() == k_bias.get_shape(),
            "Incompatible shapes of q and k bias in QKV layer: q = {:?}, k = {:?}",
            q_bias.get_shape(),
            k_bias.get_shape(),
        );
        ensure!(
            q_bias.get_shape() == v_bias.get_shape(),
            "Incompatible shapes of q and v bias in QKV layer: q = {:?}, v = {:?}",
            q_bias.get_shape(),
            v_bias.get_shape(),
        );
        // mat mul : [a,b] * [b, c] -> [a, c] + [c]

        let hidden_size = q.get_shape()[1];
        assert_eq!(
            hidden_size,
            q_bias.get_shape()[0],
            "q.get_shape() {:?} != q_bias.get_shape() {:?}",
            q.get_shape(),
            q_bias.get_shape()
        );
        ensure!(
            hidden_size % num_heads == 0,
            "Expected number of heads to be a divisor of hidden size, but it's not: hidden_size = {hidden_size}, num_heads = {num_heads}"
        );
        let head_dim = hidden_size / num_heads;
        let unpadded_shape = q.get_shape();
        Ok(Self {
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            unpadded_shape,
            num_heads,
            head_dim,
            // cache: None,
        })
    }

    // Given the point of a claim referring to a 2d output tensor with `output_num_vars` variables,
    // split the point in 2 sub-points corresponding to the variables on each of the 2 dimensions (i.e., rows and columns)
    fn split_claim_point<E: ExtensionField>(
        claim_point: &[E],
        output_num_vars: (usize, usize),
    ) -> Result<(&[E], &[E])> {
        ensure!(
            claim_point.len() == output_num_vars.0 + output_num_vars.1,
            "Mismatch between size of claim point and number of variables when splitting claim point for QKV layer"
        );
        let point_for_row = &claim_point[output_num_vars.1..];
        let point_for_column = &claim_point[..output_num_vars.1];
        Ok((point_for_row, point_for_column))
    }

    // Build evaluations point for claims related to a pair (input_matrix, weight matrix) produced
    // by the sumcheck protocol in `prove`. Here, weight matrix can be one of 'self.q`, `self.k` and 'self.v`.
    // The method requires the following inputs:
    // - `claim_point`: point of the claim for the MLE of `output matrix = input_matrix*weight_matrix``
    // - `proof_point`: random point employed to prove sumcheck
    // - `output_num_vars` : number of variables of the output matrix
    fn build_points<E: ExtensionField>(
        claim_point: &[E],
        proof_point: &[E],
        output_num_vars: (usize, usize),
    ) -> Result<(Vec<E>, Vec<E>)> {
        let (point_for_row, point_for_column) =
            Self::split_claim_point(claim_point, output_num_vars)?;
        // sumcheck point is on the column variables, which are the low ones
        let input_point = [proof_point, point_for_row].concat();
        // sumcheck point is on the row variables, which are the high ones
        let weight_matrix_point = [point_for_column, proof_point].concat();
        Ok((input_point, weight_matrix_point))
    }

    // Squeeze the challenges required to batch the sumcheck equations employed to prove the layer.
    // It requires as input the output claims for the layer and the evaluations (over the same points of the output
    // claims) of the MLEs of the output tensors before bias addition, which are the claims actually used in the batched
    // sumcheck
    fn challenges_for_batched_sumcheck<E: ExtensionField, T: Transcript<E>>(
        transcript: &mut T,
        last_claims: &[&Claim<E>],
        evals_pre_bias: &[E],
    ) -> Vec<Challenge<E>> {
        // add claims about output tensors without bias to the transcript, to then squeeze the challenge necessary to batch the matrix multiplication
        // sum-check equation
        last_claims
            .iter()
            .zip(evals_pre_bias)
            .for_each(|(&claim, evals)| {
                transcript.append_field_element_exts(&claim.point);
                transcript.append_field_element_ext(evals);
            });
        // We actually need 2 random challenges, but we also return the identity element as the
        // "first challenge" to be able to batch easily with iterators
        [Challenge { elements: E::ONE }]
            .into_iter()
            .chain((0..2).map(|_| transcript.read_challenge()))
            .collect()
    }
}

const IS_PROVABLE: bool = true;

impl<N: Number> OpInfo for QKV<N> {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        assert_eq!(input_shapes.len(), 1, "Expected one input for QKV layer");
        let input_shape = input_shapes[0].clone();
        match padding_mode {
            PaddingMode::NoPadding => {
                vec![vec![input_shape[0], self.unpadded_shape[1]].into(); self.num_outputs(1)]
            }
            PaddingMode::Padding => {
                // compute head_dim from hidden_size and num_heads
                let padded_weight =
                    padded_weight_shape(&self.unpadded_shape, self.num_heads, self.head_dim);
                vec![
                    Shape::new(vec![
                        input_shape[0].next_power_of_two(),
                        padded_weight[1].next_power_of_two()
                    ]);
                    3
                ]
            }
        }
    }

    /// Compute the number of output tensors, given the number of input tensors
    /// `num_inputs`
    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs * 3
    }

    /// Textual description of the operation
    fn describe(&self) -> String {
        format!("QKV [{},{}]", self.q.get_shape()[0], self.q.get_shape()[1])
    }

    /// Specify whether the operation needs to be proven or not
    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<N: Number> Evaluate<N> for QKV<N> {
    /// Returns x[-1,..] * Q, X * K, X * V
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 1, "QKV expects 1 input");
        let shape = inputs[0].get_shape();
        let emb_size = shape[1];
        let q_emb_size = self.q.get_shape()[0];
        ensure!(
            q_emb_size == emb_size,
            "QKV: q_emb_size {} != emb_size {} (input shape {:?} vs q shape {:?})",
            q_emb_size,
            emb_size,
            shape,
            self.q.get_shape()
        );
        // if let Some(cache) = &self.cache {
        //    // make sure the size of the input match the size of the cache + 1
        //    // as we only want to do the the matmul for the new token, not for the previously generated ones
        //    ensure!(
        //        seq_len == cache.k_shape()[0] + 1,
        //        "QKV: seq_len != cache.k_shape()[0] + 1"
        //    );
        //}
        let input = inputs[0];
        // if self.cache.is_some() {
        //    &inputs[0].slice_2d(seq_len - 1, seq_len)
        //} else {
        // add row by row
        let q = input.matmul(&self.q).add_dim2(&self.q_bias);
        let k = input.matmul(&self.k).add_dim2(&self.k_bias);
        let v = input.matmul(&self.v).add_dim2(&self.v_bias);
        // if let Some(cache) = &mut self.cache {
        //    cache.stack(k, v);
        //    // vector Q, full K, full V
        //    Ok(LayerOut::from_vec(vec![q, cache.k(), cache.v()]))
        //} else {
        Ok(LayerOut::from_vec(vec![q, k, v]))
    }
}
impl QuantizeOp for QKV<f32> {
    type QuantizedOp = QKV<Element>;

    /// Convert an operation into its quantized version
    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == num_outputs,
            "Output scaling for QKV layer different from {num_outputs}"
        );
        self.quantize_from_scalings(input_scaling, &output_scalings)
    }
}

impl QKV<f32> {
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<QKV<Element>>> {
        ensure!(input_scaling.len() == 1, "QKV: input_scaling.len() != 1");
        ensure!(output_scaling.len() == 3, "QKV: output_scaling.len() != 3");
        // for each tensor, we look at the scaling factor and the scaling factor of the associated bias
        let (matrices, (biases, requants)): (Vec<_>, (Vec<_>, Vec<_>)) = output_scaling
            .iter()
            .zip(vec![
                (self.q, self.q_bias),
                (self.k, self.k_bias),
                (self.v, self.v_bias),
            ])
            .map(|(output_scaling, (tensor, bias))| {
                let (model_scaling, bias_scaling) =
                    model_scaling_factor_from_tensor_and_bias(&input_scaling[0], &tensor, &bias);
                let input_scaling = &input_scaling[0];
                let quantized_matrix = tensor.quantize(&model_scaling);
                let quantized_bias = bias.quantize(&bias_scaling);
                let intermediate_bitsize = quantized_matrix.matmul_output_bitsize(None, None);
                let requant = Requant::from_scaling_factors(
                    *input_scaling,
                    model_scaling,
                    *output_scaling,
                    intermediate_bitsize,
                );
                (quantized_matrix, (quantized_bias, requant))
            })
            .unzip();
        let mut matit = matrices.into_iter();
        let (q, k, v) = (
            matit.next().unwrap(),
            matit.next().unwrap(),
            matit.next().unwrap(),
        );
        let mut biasit = biases.into_iter();
        let (q_bias, k_bias, v_bias) = (
            biasit.next().unwrap(),
            biasit.next().unwrap(),
            biasit.next().unwrap(),
        );
        let quantized_op = QKV::new(q, q_bias, k, k_bias, v, v_bias, self.num_heads)?;
        Ok(QuantizeOutput::new(quantized_op, output_scaling.to_vec()).with_requants(requants))
    }
}

const WEIGHT_Q_POLY_ID: &str = "WeightQ";
const WEIGHT_K_POLY_ID: &str = "WeightK";
const WEIGHT_V_POLY_ID: &str = "WeightV";
const BIAS_Q_POLY_ID: &str = "BiasQ";
const BIAS_K_POLY_ID: &str = "BiasK";
const BIAS_V_POLY_ID: &str = "BiasV";

impl<E: ExtensionField + DeserializeOwned> ProveInfo<E> for QKV<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        ensure!(
            aux.last_output_shape.len() == 1,
            "expected one input shape for context of QKV layer"
        );
        ensure!(
            aux.last_output_shape[0][1] == self.q.get_shape()[0],
            "Number of columns in input matrix ({}) is different from number of rows in Q weight matrix ({})",
            aux.last_output_shape[0][1],
            self.q.get_shape()[0],
        );
        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);
        aux.model_polys = Some(
            [
                (WEIGHT_Q_POLY_ID, &self.q),
                (WEIGHT_K_POLY_ID, &self.k),
                (WEIGHT_V_POLY_ID, &self.v),
                (BIAS_Q_POLY_ID, &self.q_bias),
                (BIAS_K_POLY_ID, &self.k_bias),
                (BIAS_V_POLY_ID, &self.v_bias),
            ]
            .into_iter()
            .map(|(poly_id, matrix)| {
                let evals = matrix.pad_next_power_of_two().data;
                (poly_id.to_string(), evals)
            })
            .collect(),
        );

        // number of variables in the sum-check is equal to the number of variables corresponding to
        // the rows of weight matrices
        let num_vars = self.q.num_vars_2d().0;

        let vp_aux = VPAuxInfo::from_mle_list_dimensions(&vec![vec![num_vars, num_vars]; 3]);

        let ctx = QKVCtx {
            node_id: id,
            sumcheck_poly_aux: vp_aux,
            unpadded_shape: self.unpadded_shape.clone(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        };

        Ok((LayerCtx::QKV(ctx), aux))
    }
}

impl PadOp for QKV<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_qkv(self, si)
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS> for QKV<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Ctx = QKVCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        let expected_num_outputs = self.num_outputs(1);
        ensure!(
            last_claims.len() == expected_num_outputs,
            "Expected {expected_num_outputs} output claims for QKV layer, found {}",
            last_claims.len()
        );
        ensure!(
            step_data.inputs.len() == 1,
            "Expected 1 input tenstor in inference data for QKV layer, found {}",
            step_data.inputs.len()
        );
        let input = &step_data.inputs[0];
        ensure!(
            input.is_matrix(),
            "Input tensor for QKV layer is not a matrix"
        );
        let ncols = input.ncols_2d();
        let nrows = self.q.nrows_2d();
        ensure!(
            ncols == nrows,
            "Number of columns in input matrix ({ncols}) different from number of rows in Q weight matrix of QKV layer ({nrows})"
        );
        let expected_output_shape = vec![input.nrows_2d(), self.q.ncols_2d()].into();
        ensure!(
            step_data.outputs.outputs().len() == expected_num_outputs,
            "Expected {expected_num_outputs} output tensors in inference data for QKV layer, found {}",
            step_data.outputs.outputs().len()
        );
        step_data.outputs.outputs().into_iter().try_for_each(|out| {
                ensure!(out.get_shape() == expected_output_shape,
                    "Expected shape {expected_output_shape:?} for output of QKV layer, found shape {:?}", out.get_shape(),
                );
                Ok(())
            }
        )?;
        let output_num_vars_2d = step_data.outputs.outputs()[0].num_vars_2d(); // we can use the first one since we checked all outputs
        // have the same shape
        let output_num_vars = output_num_vars_2d.0 + output_num_vars_2d.1; // overall number of variables for the MLEs of outputs
        last_claims.iter().try_for_each(|claim| {
            ensure!(claim.point.len() == output_num_vars,
                "Unexpected length of output claim for QKV layer: expected {output_num_vars}, found {}", claim.point.len(),
            );
            Ok(())
        })?;

        // compute claims about the bias polynomials
        let (bias_claims, evals_pre_bias): (Vec<_>, Vec<_>) = try_unzip_parallel(
            last_claims
                .par_iter()
                .zip([&self.q_bias, &self.k_bias, &self.v_bias].par_iter())
                .map(|(&claim, bias_vector)| {
                    let (_, point_for_column) =
                        Self::split_claim_point(&claim.point, output_num_vars_2d)?;
                    ensure!(
                        point_for_column.len() == bias_vector.get_data().len().ilog2() as usize
                    );
                    let eval = bias_vector
                        .evals_flat::<E>()
                        .into_mle()
                        .evaluate(point_for_column);
                    let bias_claim = Claim::new(point_for_column.to_vec(), eval);
                    // subtract the bias evals from output claims to get claims about the tensors before bias addition
                    let eval_pre_bias = claim.eval - eval;
                    Ok((bias_claim, eval_pre_bias))
                }),
        )?;

        let challenges =
            Self::challenges_for_batched_sumcheck(prover.transcript, &last_claims, &evals_pre_bias);

        let input_mle = input.to_mle_2d();

        // Number of variables involved in the sum-check corresponds to the number of columns of the input matrix
        let num_vars = input.num_vars_2d().1;

        let mut vp = VirtualPolynomial::new(num_vars);

        last_claims
            .iter()
            .zip([&self.q, &self.k, &self.v])
            .zip(&challenges)
            .try_for_each(|((&claim, weight_matrix), challenge)| {
                let mut weight_mle = weight_matrix.to_2d_mle();
                let (point_for_row, point_for_column) =
                    Self::split_claim_point(&claim.point, output_num_vars_2d)?;
                let fixed_input_mle = input_mle.fix_high_variables(point_for_row);
                weight_mle.fix_variables_in_place(point_for_column);
                let coefficient = challenge.elements;
                vp.add_mle_list(vec![fixed_input_mle.into(), weight_mle.into()], coefficient);
                anyhow::Ok(())
            })?;

        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // get claims for all the MLEs involved in sum-check
        let sumcheck_evals = state
            .get_mle_final_evaluations()
            .chunks(2) // each chunk refers to a pair of (input, weight matrix) MLEs in the sumcheck
            .map(|evals| (evals[0], evals[1]))
            .collect_vec();

        // Build claims corresponding to each evaluation, splitting between claims related to the input matrix
        // and claims related to the weight matrices
        let (input_claims, weight_claims): (Vec<_>, Vec<_>) = try_unzip(
            last_claims
                .iter()
                .zip(&sumcheck_evals)
                .map(|(&claim, evals)| {
                    let (point_for_input, point_for_weight) =
                        Self::build_points(&claim.point, &proof.point, output_num_vars_2d)?;
                    anyhow::Ok((
                        Claim::new(point_for_input, evals.0),
                        Claim::new(point_for_weight, evals.1),
                    ))
                }),
        )?;

        // debug: check input claims
        debug_assert!(input_claims.iter().all(|claim| {
            let eval = input_mle.evaluate(&claim.point);
            claim.eval == eval
        }));

        // Build set of claims to be proven via polynomial commitment opening proof
        let common_claims = weight_claims
            .into_iter()
            .chain(bias_claims)
            .zip([
                WEIGHT_Q_POLY_ID,
                WEIGHT_K_POLY_ID,
                WEIGHT_V_POLY_ID,
                BIAS_Q_POLY_ID,
                BIAS_K_POLY_ID,
                BIAS_V_POLY_ID,
            ])
            .map(|(claim, id)| (id.to_string(), claim))
            .collect();

        prover.add_common_claims(node_id, common_claims)?;

        // Aggregate input claims into a single one, which will be returned as output
        let mut same_poly_prover = same_poly::Prover::new(input_mle);

        input_claims
            .into_iter()
            .try_for_each(|claim| same_poly_prover.add_claim(claim))?;

        let aggregation_proof = same_poly_prover.prove(prover.transcript)?;

        let aggregated_claim = aggregation_proof.extract_claim();

        let proof = QKVProof {
            sumcheck: proof,
            aggregation_proof,
            individual_claims: sumcheck_evals.try_into().unwrap(),
            pre_bias_evals: evals_pre_bias,
        };

        prover.push_proof(node_id, LayerProof::QKV(proof));

        Ok(vec![aggregated_claim])
    }
}

impl<E> OpInfo for QKVCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let weight_shape = match padding_mode {
            PaddingMode::NoPadding => &self.unpadded_shape,
            PaddingMode::Padding => {
                &padded_weight_shape(&self.unpadded_shape, self.num_heads, self.head_dim)
            }
        };

        assert_eq!(
            input_shapes.len(),
            1,
            "Expected only 1 input shape for QKV layer"
        );

        assert_eq!(
            input_shapes[0][1], weight_shape[0],
            "Shape mismatch for QKV ctx: number of columns in input shape different from number of rows of weight matrices {} != {}",
            input_shapes[0][1], weight_shape[0],
        );

        vec![vec![input_shapes[0][0], weight_shape[1]].into(); self.num_outputs(1)]
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs * 3
    }

    fn describe(&self) -> String {
        let padded_matrix_shape = self.unpadded_shape.next_power_of_two();
        format!(
            "QKV [{},{}]",
            padded_matrix_shape[0], padded_matrix_shape[1]
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for QKVCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = QKVProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "Expected 1 input shape for QKV verifier, found {}",
            shape_step.padded_input_shape.len(),
        );
        let padded_input_shape = &shape_step.padded_input_shape[0];
        let expected_num_outputs = self.num_outputs(1);
        ensure!(
            shape_step.padded_output_shape.len() == expected_num_outputs,
            "Expected {expected_num_outputs} shapes for QKV verifier, found {}",
            shape_step.padded_output_shape.len()
        );
        ensure!(
            last_claims.len() == expected_num_outputs,
            "Expected {expected_num_outputs} output claims for QKV verifier, found {}",
            last_claims.len()
        );

        let output_shape = &shape_step.padded_output_shape[0]; // we can just take the first one since all the output shapes
        // are expected to be the same
        let output_num_vars = (
            output_shape[0].ilog2() as usize,
            output_shape[1].ilog2() as usize,
        );

        // compute claims for the bias vector, subtracting the `pre_bias_evals` found in the proof from the output claims
        let bias_claims = last_claims
            .iter()
            .zip(&proof.pre_bias_evals)
            .map(|(&claim, eval)| {
                let bias_eval = claim.eval - *eval;
                let (_, point_for_column) =
                    QKV::<Element>::split_claim_point(&claim.point, output_num_vars)?;
                Ok(Claim::new(point_for_column.to_vec(), bias_eval))
            })
            .collect::<Result<Vec<_>>>()?;

        let challenges = QKV::<Element>::challenges_for_batched_sumcheck(
            verifier.transcript,
            last_claims,
            &proof.pre_bias_evals,
        );

        // use challenge to batch evaluations used in sumcheck
        let batched_evals = proof
            .pre_bias_evals
            .iter()
            .zip(&challenges)
            .fold(E::ZERO, |acc, (eval, chal)| acc + *eval * chal.elements);

        // verify batched sumcheck
        let subclaim = IOPVerifierState::<E>::verify(
            batched_evals,
            &proof.sumcheck,
            &self.sumcheck_poly_aux,
            verifier.transcript,
        );

        // Build claims corresponding to each evaluation of the MLEs involved in the batched sumcheck,
        // splitting between claims related to the input matrix and claims related to the weight matrices
        let (input_claims, weight_claims): (Vec<_>, Vec<_>) = try_unzip(
            last_claims
                .iter()
                .zip(
                    proof.individual_claims.iter(), // each chunk refers to a pair of (input, weight matrix) MLEs in the sumcheck
                )
                .map(|(&claim, evals)| {
                    let (point_for_input, point_for_weight) = QKV::<Element>::build_points(
                        &claim.point,
                        &subclaim.point_flat(),
                        output_num_vars,
                    )?;
                    anyhow::Ok((
                        Claim::new(point_for_input, evals.0),
                        Claim::new(point_for_weight, evals.1),
                    ))
                }),
        )?;

        // Build set of claims to be proven via polynomial commitment opening proof
        let common_claims = weight_claims
            .into_iter()
            .chain(bias_claims)
            .zip([
                WEIGHT_Q_POLY_ID,
                WEIGHT_K_POLY_ID,
                WEIGHT_V_POLY_ID,
                BIAS_Q_POLY_ID,
                BIAS_K_POLY_ID,
                BIAS_V_POLY_ID,
            ])
            .map(|(claim, id)| (id.to_string(), claim))
            .collect();

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
            proof.individual_to_virtual_claim(&challenges) == subclaim.expected_evaluation,
            "sumcheck claim failed",
        );

        let sum_check_num_vars = padded_input_shape.iter().product::<usize>().ilog2() as usize;

        let ctx = same_poly::Context::new(sum_check_num_vars);

        let mut same_poly_verifier = same_poly::Verifier::new(&ctx);

        input_claims
            .into_iter()
            .try_for_each(|claim| same_poly_verifier.add_claim(claim))?;

        let aggregated_claim =
            same_poly_verifier.verify(&proof.aggregation_proof, verifier.transcript)?;

        Ok(vec![aggregated_claim])
    }
}

#[derive(Debug, Clone)]
pub struct CacheQKV<N> {
    cache_k: Tensor<N>,
    cache_v: Tensor<N>,
    initialized: bool,
}

impl<N: Number> Default for CacheQKV<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Number> CacheQKV<N> {
    pub fn new() -> Self {
        Self {
            cache_k: Tensor::new(vec![0].into(), vec![]),
            cache_v: Tensor::new(vec![0].into(), vec![]),
            initialized: false,
        }
    }
    pub fn stack(&mut self, k: Tensor<N>, v: Tensor<N>) {
        assert!(k.is_vector(), "k is not a vector {:?}", k.get_shape());
        assert_eq!(
            k.get_shape(),
            v.get_shape(),
            "k and v have different shapes {:?} != {:?}",
            k.get_shape(),
            v.get_shape()
        );
        if self.initialized {
            assert_eq!(
                self.cache_k.get_shape()[1],
                k.get_shape()[1],
                "cache_k and k have different last dimension {:?} != {:?}",
                self.cache_k.get_shape(),
                k.get_shape()
            );
            self.cache_k.concat(k);
            self.cache_v.concat(v);
        } else {
            self.cache_k = k;
            self.cache_v = v;
            self.initialized = true;
        }
    }
    pub fn k_shape(&self) -> Shape {
        self.cache_k.get_shape()
    }
    pub fn v_shape(&self) -> Shape {
        self.cache_v.get_shape()
    }
    pub fn k(&self) -> Tensor<N> {
        self.cache_k.clone()
    }
    pub fn v(&self) -> Tensor<N> {
        self.cache_v.clone()
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;

    use crate::{
        layers::{Layer, provable::evaluate_layer},
        model::{Model, test::prove_model},
        padding::ShapeData,
        tensor::Shape,
    };

    use super::*;

    impl<N: Number> QKV<N> {
        pub(crate) fn random(
            num_heads: usize,
            emb_size: usize,
            hidden_size: usize,
        ) -> Result<Self> {
            let q = Tensor::<N>::random(&vec![emb_size, hidden_size].into());
            let q_bias = Tensor::<N>::random(&vec![hidden_size].into());
            let k = Tensor::<N>::random(&vec![emb_size, hidden_size].into());
            let k_bias = Tensor::<N>::random(&vec![hidden_size].into());
            let v = Tensor::<N>::random(&vec![emb_size, hidden_size].into());
            let v_bias = Tensor::<N>::random(&vec![hidden_size].into());
            Self::new(q, q_bias, k, k_bias, v, v_bias, num_heads)
        }
    }

    //#[test]
    // fn test_qkv_cache() {
    //    // first token
    //    let seq_len = 1;
    //    let emb_size = 2;
    //    let hidden_size = 3;
    //    let q = Tensor::<f32>::random(&[emb_size, hidden_size]);
    //    let q_bias = Tensor::<f32>::random(&[hidden_size]);
    //    let k = Tensor::<f32>::random(&[emb_size, hidden_size]);
    //    let k_bias = Tensor::<f32>::random(&[hidden_size]);
    //    let v = Tensor::<f32>::random(&[emb_size, hidden_size]);
    //    let v_bias = Tensor::<f32>::random(&[hidden_size]);
    //    let mut qkv = QKV::new(
    //        q.clone(),
    //        q_bias.clone(),
    //        k.clone(),
    //        k_bias.clone(),
    //        v.clone(),
    //        v_bias.clone(),
    //    )
    //    .with_cache();
    //    let mut input = Tensor::<f32>::random(&[1, emb_size]);
    //    let output = qkv.evaluate::<GoldilocksExt2>(&[&input]).unwrap().outputs;
    //    assert_eq!(output.len(), 3);
    //    assert_eq!(output[0].get_shape(), vec![1, hidden_size]);
    //    assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size]);
    //    let mut out_k = input.matmul(&k).add_dim2(&k_bias);
    //    assert_eq!(output[1].get_data(), out_k.get_data());
    //    let mut out_v = input.matmul(&v).add_dim2(&v_bias);
    //    assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size]);
    //    assert_eq!(output[2].get_data(), out_v.get_data());
    //    // second token
    //    let seq_len = 2;
    //    let new_token_emb = Tensor::<f32>::random(&[1, emb_size]);
    //    input.concat(new_token_emb.clone());
    //    let output = qkv.evaluate::<GoldilocksExt2>(&[&input]).unwrap().outputs;
    //    assert_eq!(output.len(), 3);
    //    assert_eq!(output[0].get_shape(), vec![1, hidden_size]);
    //    assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size]);
    //    assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size]);
    //    let out_q = new_token_emb.matmul(&q).add_dim2(&q_bias);
    //    assert_eq!(output[0].get_data(), out_q.get_data());
    //    out_k.concat(new_token_emb.matmul(&k).add_dim2(&k_bias));
    //    assert_eq!(output[1].get_data(), out_k.get_data());
    //    out_v.concat(new_token_emb.matmul(&v).add_dim2(&v_bias));
    //    assert_eq!(output[2].get_data(), out_v.get_data());
    //}

    #[test]
    fn test_qkv_no_cache() {
        // first token
        let seq_len = 3;
        let emb_size = 2;
        let hidden_size = 3;
        let num_heads = 1;
        let q = Tensor::<f32>::random(&vec![emb_size, hidden_size].into());
        let q_bias = Tensor::<f32>::random(&vec![hidden_size].into());
        let k = Tensor::<f32>::random(&vec![emb_size, hidden_size].into());
        let k_bias = Tensor::<f32>::random(&vec![hidden_size].into());
        let v = Tensor::<f32>::random(&vec![emb_size, hidden_size].into());
        let v_bias = Tensor::<f32>::random(&vec![hidden_size].into());
        let qkv = QKV::new(
            q.clone(),
            q_bias.clone(),
            k.clone(),
            k_bias.clone(),
            v.clone(),
            v_bias.clone(),
            num_heads,
        )
        .unwrap();
        let mut input = Tensor::<f32>::random(&vec![seq_len, emb_size].into());
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].get_shape(), vec![seq_len, hidden_size].into());
        assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size].into());
        let mut out_k = input.matmul(&k).add_dim2(&k_bias);
        assert_eq!(output[1].get_data(), out_k.get_data());
        let mut out_v = input.matmul(&v).add_dim2(&v_bias);
        assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size].into());
        assert_eq!(output[2].get_data(), out_v.get_data());
        // second token
        let seq_len = seq_len + 1;
        let new_token_emb = Tensor::<f32>::random(&vec![1, emb_size].into());
        input.concat(new_token_emb.clone());
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].get_shape(), vec![seq_len, hidden_size].into());
        assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size].into());
        assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size].into());
        let out_q = input.matmul(&q).add_dim2(&q_bias);
        assert_eq!(output[0].get_data(), out_q.get_data());
        out_k.concat(new_token_emb.matmul(&k).add_dim2(&k_bias));
        assert_eq!(output[1].get_data(), out_k.get_data());
        out_v.concat(new_token_emb.matmul(&v).add_dim2(&v_bias));
        assert_eq!(output[2].get_data(), out_v.get_data());
    }

    #[test]
    fn test_qkv_padding() {
        let num_inputs = 57;
        let embedding_size = 77;
        let hidden_size = 35;
        let num_heads = 7;
        let unpadded_input_shape = Shape::new(vec![num_inputs, embedding_size]);
        let weight_shape = Shape::new(vec![embedding_size, hidden_size]);
        let bias_shape = Shape::new(vec![hidden_size]);

        let layer = QKV::<Element>::random(num_heads, embedding_size, hidden_size).unwrap();
        let mut si = vec![ShapeData::new(unpadded_input_shape.clone())]
            .as_slice()
            .into();
        let padded_layer = layer.clone().pad_node(&mut si).unwrap();

        let padded_weight_shape = weight_shape.next_power_of_two();
        let padded_bias_shape = bias_shape.next_power_of_two();

        let unpadded_output_shapes =
            layer.output_shapes(&vec![unpadded_input_shape.clone()], PaddingMode::NoPadding);
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check unpadded output shapes for padded layer
        let unpadded_output_shapes =
            padded_layer.output_shapes(&vec![unpadded_input_shape.clone()], PaddingMode::NoPadding);
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check padded output shapes
        let padded_input_shape = unpadded_input_shape.next_power_of_two();
        let padded_output_shapes =
            padded_layer.output_shapes(&vec![padded_input_shape.clone()], PaddingMode::Padding);
        assert_eq!(padded_output_shapes, si.padded_input_shapes(),);

        assert_eq!(padded_layer.q.get_shape(), padded_weight_shape);
        assert_eq!(padded_layer.k.get_shape(), padded_weight_shape);
        assert_eq!(padded_layer.v.get_shape(), padded_weight_shape);
        assert_eq!(padded_layer.q_bias.get_shape(), padded_bias_shape);
        assert_eq!(padded_layer.k_bias.get_shape(), padded_bias_shape);
        assert_eq!(padded_layer.v_bias.get_shape(), padded_bias_shape);

        // check data in padded layer is the same of original layer
        let head_dim = layer.head_dim;
        assert_eq!(head_dim, hidden_size / num_heads);
        let padded_head_dim = head_dim.next_power_of_two();
        [&layer.q, &layer.k, &layer.v]
            .into_iter()
            .zip([&padded_layer.q, &padded_layer.k, &padded_layer.v])
            .for_each(|(weight, padded_weight)| {
                let padded_weight_shape = padded_weight.get_shape();
                for i in 0..padded_weight_shape[0] {
                    for j in 0..padded_weight_shape[1] {
                        if i < embedding_size
                            && j % padded_head_dim < head_dim
                            && j / padded_head_dim < num_heads
                        {
                            let original_matrix_index =
                                j / padded_head_dim * head_dim + j % padded_head_dim;
                            assert_eq!(
                                weight.get_2d(i, original_matrix_index),
                                padded_weight.get_2d(i, j)
                            );
                        } else {
                            assert_eq!(0, padded_weight.get_2d(i, j));
                        }
                    }
                }
            });

        // test also evaluation over padded layer
        let mut input = Tensor::<Element>::random(&unpadded_input_shape);
        let output = evaluate_layer::<GoldilocksExt2, _, _>(
            &layer,
            &vec![&input],
            Some(vec![unpadded_input_shape.clone()]),
        )
        .unwrap();

        assert!(
            output
                .outputs()
                .into_iter()
                .zip(&unpadded_output_shapes)
                .all(|(out, expected_shape)| out.get_shape() == *expected_shape)
        );

        input.pad_to_shape(padded_input_shape);
        let padded_output = evaluate_layer::<GoldilocksExt2, _, _>(
            &padded_layer,
            &vec![&input],
            Some(vec![unpadded_input_shape]),
        )
        .unwrap();

        assert!(
            padded_output
                .outputs()
                .into_iter()
                .zip(&padded_output_shapes)
                .all(|(out, expected_shape)| out.get_shape() == *expected_shape)
        );

        // check that padded_output has same values of output in non-padded entries
        output
            .outputs()
            .into_iter()
            .zip(padded_output.outputs())
            .zip([
                &padded_layer.q_bias,
                &padded_layer.k_bias,
                &padded_layer.v_bias,
            ]) // we need to include the bias
            // vectors for the padded rows
            .for_each(|((output, padded_out), padded_bias)| {
                let padded_out_shape = padded_out.get_shape();
                for i in 0..padded_out_shape[0] {
                    for j in 0..padded_out_shape[1] {
                        if i < num_inputs {
                            if j % padded_head_dim < head_dim && j / padded_head_dim < num_heads {
                                let original_matrix_index =
                                    j / padded_head_dim * head_dim + j % padded_head_dim;
                                assert_eq!(
                                    output.get_2d(i, original_matrix_index),
                                    padded_out.get_2d(i, j)
                                );
                            } else {
                                assert_eq!(0, padded_out.get_2d(i, j));
                            }
                        } else {
                            assert_eq!(padded_bias.get_data()[j], padded_out.get_2d(i, j));
                        }
                    }
                }
            });
    }

    #[test]
    fn test_qkv_already_padded() {
        // use power of 2 dimensions
        let num_inputs = 64;
        let embedding_size = 128;
        let hidden_size = 32;
        let num_heads = 8;
        let unpadded_input_shape = Shape::new(vec![num_inputs, embedding_size]);
        let weight_shape = Shape::new(vec![embedding_size, hidden_size]);
        let bias_shape = Shape::new(vec![hidden_size]);

        let layer = QKV::<Element>::random(num_heads, embedding_size, hidden_size).unwrap();
        let mut si = vec![ShapeData::new(unpadded_input_shape.clone())]
            .as_slice()
            .into();
        let padded_layer = layer.clone().pad_node(&mut si).unwrap();

        let unpadded_output_shapes =
            layer.output_shapes(&vec![unpadded_input_shape.clone()], PaddingMode::NoPadding);
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check unpadded output shapes for padded layer
        let unpadded_output_shapes =
            padded_layer.output_shapes(&vec![unpadded_input_shape.clone()], PaddingMode::NoPadding);
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check padded output shapes
        let padded_output_shapes =
            padded_layer.output_shapes(&vec![unpadded_input_shape], PaddingMode::Padding);
        assert_eq!(padded_output_shapes, si.padded_input_shapes(),);

        assert_eq!(padded_layer.q.get_shape(), weight_shape);
        assert_eq!(padded_layer.k.get_shape(), weight_shape);
        assert_eq!(padded_layer.v.get_shape(), weight_shape);
        assert_eq!(padded_layer.q_bias.get_shape(), bias_shape);
        assert_eq!(padded_layer.k_bias.get_shape(), bias_shape);
        assert_eq!(padded_layer.v_bias.get_shape(), bias_shape);

        // check data in padded layer is the same of original layer
        [&layer.q, &layer.k, &layer.v]
            .into_iter()
            .zip([&padded_layer.q, &padded_layer.k, &padded_layer.v])
            .for_each(|(weight, padded_weight)| {
                assert_eq!(weight.get_data(), padded_weight.get_data())
            });
    }

    #[test]
    fn test_proven_qkv_layer() {
        let num_inputs = 49;
        let embedding_size = 78;
        let hidden_size = 120;
        let num_heads = 10;

        let input_shape = vec![num_inputs, embedding_size].into();
        let mut model =
            Model::<f32>::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);

        let _qkv_node_id = model
            .add_consecutive_layer(
                Layer::QKV(QKV::random(num_heads, embedding_size, hidden_size).unwrap()),
                None,
            )
            .unwrap();

        model.route_output(None).unwrap();
        model.describe();
        prove_model(model).unwrap();
    }
}
