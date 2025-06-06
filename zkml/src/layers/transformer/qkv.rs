use std::collections::HashMap;

use anyhow::{ensure, Result};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{mle::{IntoMLE, MultilinearExtension}, virtual_poly::{VPAuxInfo, VirtualPolynomial}};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sumcheck::structs::{IOPProof, IOPProverState};
use transcript::Transcript;

use crate::{
    commit::{context::PolyId, same_poly}, iop::{context::{ContextAux, ShapeStep}, verifier::Verifier}, layers::{
        provable::{Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput, VerifiableCtx},
        requant::Requant, LayerCtx,
    }, model::StepData, padding::{PaddingMode, ShapeInfo}, quantization::model_scaling_factor_from_tensor_and_bias, tensor::Number, try_unzip, Claim, Element, NextPowerOfTwo, Prover, ScalingFactor, ScalingStrategy, Tensor
};

/// A layer that evaluates the tensor X against the matrices Q, K and V.
/// NOTE: it performs optimizations with the cache, so it actually
/// do the matrix mult only with the last entry of the input.
/// It also outputs only the "small" Q but with the help of caching, it outputs
/// the full K and V matrices as if they were computed using the whole input tensor.
#[derive(Debug, Clone)]
pub struct QKV<N> {
    pub q: Tensor<N>,
    pub q_bias: Tensor<N>,
    pub k: Tensor<N>,
    pub k_bias: Tensor<N>,
    pub v: Tensor<N>,
    pub v_bias: Tensor<N>,
    unpadded_shape: Vec<usize>, // same shape for Q, K and V
    // pub cache: Option<CacheQKV<N>>,
}
#[derive(Clone, Debug)]
pub struct QKVCtx<E> {
    node_id: NodeId,
    sumcheck_poly_aux: VPAuxInfo<E>,
    padded_matrix_shape: Vec<usize>, // same shape for Q, K and V
    unpadded_shape: Vec<usize>, // same shape for Q, K and V
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QKVProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the QKV matrix multiplications
    pub(crate) sumcheck: IOPProof<E>,
    /// Proof for the aggregation of the claims about the input matrix to 
    /// a single claim
    aggregation_proof: same_poly::Proof<E>,
    /// The evaluation of the MLEs of the outputs vectors without the bias.
    /// The verifier needs these evaluations to check the output of the sumcheck proof
    pre_bias_evals: HashMap<PolyId, E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly"
    individual_claims: Vec<E>,
}

impl<N: Number> QKV<N> {
    pub fn new(
        q: Tensor<N>,
        q_bias: Tensor<N>,
        k: Tensor<N>,
        k_bias: Tensor<N>,
        v: Tensor<N>,
        v_bias: Tensor<N>,
    ) -> Self {
        assert_eq!(q.get_shape(), k.get_shape());
        assert_eq!(q.get_shape(), v.get_shape());
        assert_eq!(q_bias.get_shape().len(), 1);
        assert_eq!(q_bias.get_shape(), k_bias.get_shape());
        assert_eq!(q_bias.get_shape(), v_bias.get_shape());
        // mat mul : [a,b] * [b, c] -> [a, c] + [c]
        assert_eq!(
            q.get_shape()[1],
            q_bias.get_shape()[0],
            "q.get_shape() {:?} != q_bias.get_shape() {:?}",
            q.get_shape(),
            q_bias.get_shape()
        );
        let unpadded_shape = q.get_shape();
        Self {
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            unpadded_shape,  
            // cache: None,
        }
    }

    fn split_claim_point<E: ExtensionField>(claim_point: &[E], output_num_vars: (usize, usize)) -> Result<(&[E], &[E])> {
        ensure!(claim_point.len() == output_num_vars.0 + output_num_vars.1, 
            "Mismatch between size of claim point and number of variables when splitting claim point for QKV layer"
        );
        let point_for_row = &claim_point[output_num_vars.1..];
        let point_for_column = &claim_point[..output_num_vars.1];
        Ok((point_for_row, point_for_column))
    }

    fn build_points<E: ExtensionField>(
        claim_point: &[E],
        proof_point: &[E],
        output_num_vars: (usize, usize),
    ) -> Result<(Vec<E>, Vec<E>)> {
        let (point_for_row, point_for_column) = Self::split_claim_point(claim_point, output_num_vars)?;
        let input_point = [point_for_row, proof_point].concat();
        let weight_matrix_point = [proof_point, point_for_column].concat();
        Ok((input_point, weight_matrix_point))
    }
}

impl<N> OpInfo for QKV<N> {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        assert_eq!(input_shapes.len(), 1, "Expected one input for QKV layer");
        let input_shape = input_shapes[0].clone();
        match padding_mode {
            PaddingMode::NoPadding => vec![
                vec![input_shape[0], self.unpadded_shape[1]];
                3
            ],
            PaddingMode::Padding => vec![
                vec![input_shape[0], self.q.get_shape()[1]],
                vec![input_shape[0], self.k.get_shape()[1]],
                vec![input_shape[0], self.v.get_shape()[1]],
            ].into_iter()
                .map(|shape| shape.next_power_of_two())
                .collect::<Vec<_>>(),
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
        true
    }
}

impl<N: Number> Evaluate<N> for QKV<N> {
    /// Returns x[-1,..] * Q, X * K, X * V
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
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
            output_scalings.len() == 1,
            "Output scaling for QKV layer different from 1"
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
            .zip(
                vec![
                    (self.q, self.q_bias),
                    (self.k, self.k_bias),
                    (self.v, self.v_bias),
                ]
                .into_iter(),
            )
            .map(|(output_scaling, (tensor, bias))| {
                let (model_scaling, bias_scaling) = model_scaling_factor_from_tensor_and_bias(
                    &input_scaling[0],
                    &output_scaling,
                    &tensor,
                    &bias,
                );
                let input_scaling = &input_scaling[0];
                let quantized_matrix = tensor.quantize(&model_scaling);
                let quantized_bias = bias.quantize(&bias_scaling);
                let intermediate_bitsize = quantized_matrix.matmul_output_bitsize();
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
        let quantized_op = QKV::new(q, q_bias, k, k_bias, v, v_bias);
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
where  E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        ensure!(aux.last_output_shape.len() == 1, "expected one input shape for context of QKV layer");
        ensure!(aux.last_output_shape[0][1] == self.q.get_shape()[0], 
            "Number of columns in input matrix ({}) is different from number of rows in Q weight matrix ({})",
            aux.last_output_shape[0][1],
            self.q.get_shape()[0],
        );
        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);
        aux.model_polys = Some([
            (WEIGHT_Q_POLY_ID, &self.q),
            (WEIGHT_K_POLY_ID, &self.k),
            (WEIGHT_V_POLY_ID, &self.v),
            (BIAS_Q_POLY_ID, &self.q_bias),
            (BIAS_K_POLY_ID, &self.k_bias),
            (BIAS_V_POLY_ID, &self.v_bias),
            
        ].into_iter().map(|(poly_id, matrix)| {
            let evals = matrix.pad_next_power_of_two().data;
            (poly_id.to_string(), evals)
        }).collect());

        // number of variables in the sum-check is equal to the number of variables corresponding to
        // the rows of weight matrices
        let num_vars = self.q.num_vars_2d().0;
        
        let vp_aux = VPAuxInfo::from_mle_list_dimensions(&vec![
            vec![num_vars, num_vars]; 3
        ]);

        let ctx = QKVCtx {
            node_id: id,
            sumcheck_poly_aux: vp_aux,
            padded_matrix_shape: self.unpadded_shape.next_power_of_two(),
            unpadded_shape: self.unpadded_shape.clone(),
        };

        todo!()
    }
}

impl PadOp for QKV<Element> {
    fn pad_node(self, _si: &mut ShapeInfo) -> Result<Self>
        where
            Self: Sized, {
        todo!()
    }
}

const BATCHING_CHALLENGE_LABEL: &str = "batching_qkv";

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
        ensure!(last_claims.len() == expected_num_outputs, "Expected {expected_num_outputs} output claims for QKV layer, found {}", last_claims.len());
        ensure!(step_data.inputs.len() == 1, "Expected 1 input tenstor in inference data for QKV layer, found {}", step_data.inputs.len());
        let input = &step_data.inputs[0];
        ensure!(input.is_matrix(), "Input tensor for QKV layer is not a matrix");
        let ncols = input.ncols_2d();
        let nrows = self.q.nrows_2d();
        ensure!(ncols == nrows, 
            "Number of columns in input matrix ({ncols}) different from number of rows in Q weight matrix of QKV layer ({nrows})" 
        );
        let expected_output_shape = vec![input.nrows_2d(), self.q.ncols_2d()];
        ensure!(step_data.outputs.outputs().len() == expected_num_outputs, 
            "Expected {expected_num_outputs} output tensors in inference data for QKV layer, found {}", step_data.outputs.outputs().len()
        );
        step_data.outputs.outputs().into_iter().try_for_each(|out| {
                ensure!(out.get_shape() == expected_output_shape, 
                    "Expected shape {expected_output_shape:?} for output of QKV layer, foudn shape {:?}", out.get_shape(),
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
        let (bias_claims, evals_pre_bias): (Vec<_>, Vec<_>) = try_unzip(last_claims.iter().zip([&self.q_bias, &self.k_bias, &self.v_bias]).map(|(&claim, bias_vector)| {
            let (_, point_for_column) = Self::split_claim_point(&claim.point, output_num_vars_2d)?;
            ensure!(point_for_column.len() == bias_vector.get_data().len().ilog2() as usize);
            let eval = bias_vector.evals_flat::<E>().into_mle().evaluate(point_for_column);
            let bias_claim = Claim::new(point_for_column.to_vec(), eval);
            // subtract the bias evals from output claims to get claims about the tensors before bias addition
            let eval_pre_bias = claim.eval - eval;
            Ok((bias_claim, eval_pre_bias)) 
        }))?;



        // add output claims to the transcript, to then squeeze the challenge necessary to batch the matrix multiplication 
        // sum-check equation
        last_claims.iter().zip(&evals_pre_bias).for_each(|(&claim, evals)| {
            prover.transcript.append_field_element_exts(&claim.point);
            prover.transcript.append_field_element_ext(evals);
        });
        

        let challenge = prover.transcript.get_and_append_challenge(BATCHING_CHALLENGE_LABEL.as_bytes());

        let input_mle = input.to_mle_2d();

        // Number of variables involved in the sum-check corresponds to the number of columns of the input matrix
        let num_vars = input.num_vars_2d().1;

        let mut vp = VirtualPolynomial::new(num_vars);

        last_claims.iter().zip([&self.q, &self.k, &self.v]).enumerate().try_for_each(|(i, (&claim, weight_matrix))| {
            let mut weight_mle = weight_matrix.to_2d_mle();
            let (point_for_row, point_for_column) = Self::split_claim_point(&claim.point, output_num_vars_2d)?;
            let fixed_input_mle = input_mle.fix_high_variables(point_for_row);
            weight_mle.fix_variables_in_place(point_for_column);
            let coefficient = challenge.elements.pow_vartime(&vec![i as u64]);
            vp.add_mle_list(vec![fixed_input_mle.into(), weight_mle.into()], coefficient);
            anyhow::Ok(())
        })?;

        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // get claims for all the MLEs involved in sum-check
        let sumcheck_evals = state.get_mle_final_evaluations();

        // Build claims corresponding to each evaluation, splitting between claims related to the input matrix 
        // and claims related to the weight matrices
        let (input_claims, weight_claims): (Vec<_>, Vec<_>) = try_unzip(last_claims.iter().zip(
            sumcheck_evals.chunks(2) // each chunk refers to a pair of (input, weight matrix) MLEs in the sumcheck
        ).map(|(&claim, evals)| {
            let (point_for_input, point_for_weight) = Self::build_points(&claim.point, &proof.point, output_num_vars_2d)?;
            anyhow::Ok((
                Claim::new(point_for_input, evals[0]),
                Claim::new(point_for_weight, evals[1]),
            ))
        }))?;

        // Build set of claims to be proven via polynomial commitment opening proof
        let common_claims = weight_claims.into_iter().chain(bias_claims).zip(
            [
                WEIGHT_Q_POLY_ID,
                WEIGHT_K_POLY_ID,
                WEIGHT_V_POLY_ID,
                BIAS_Q_POLY_ID,
                BIAS_K_POLY_ID,
                BIAS_V_POLY_ID,
            ]
        ).map(|(claim, id)| (id.to_string(), claim)).collect();

        prover.add_common_claims(node_id, common_claims)?;

        // Aggregate input claims into a single one, which will be returned as output
        let mut same_poly_prover = same_poly::Prover::new(input_mle);
        let input_num_vars = input.num_vars_2d().0 + input.num_vars_2d().1;

        let ctx = same_poly::Context::new(input_num_vars);

        input_claims.into_iter().try_for_each(|claim| 
            same_poly_prover.add_claim(claim)
        )?;

        let aggregation_proof = same_poly_prover.prove(&ctx, prover.transcript)?;

        let aggregated_claim = aggregation_proof.extract_claim(); 

        // ToDo: add `proof` to Prover
        let proof = QKVProof {
            sumcheck: proof,
            aggregation_proof,
            individual_claims: sumcheck_evals,
            pre_bias_evals: evals_pre_bias.into_iter().zip([
                BIAS_Q_POLY_ID,
                BIAS_K_POLY_ID,
                BIAS_V_POLY_ID,
            ]).map(|(eval, id)| (id.to_string(), eval)).collect(),
        };
        

        Ok(vec![aggregated_claim])

        
    }
    
}

impl<E> OpInfo for QKVCtx<E> {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        todo!()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        todo!()
    }

    fn describe(&self) -> String {
        todo!()
    }

    fn is_provable(&self) -> bool {
        todo!()
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
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct CacheQKV<N> {
    cache_k: Tensor<N>,
    cache_v: Tensor<N>,
    initialized: bool,
}

impl<N: Number> CacheQKV<N> {
    pub fn new() -> Self {
        Self {
            cache_k: Tensor::new(vec![0], vec![]),
            cache_v: Tensor::new(vec![0], vec![]),
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
    pub fn k_shape(&self) -> Vec<usize> {
        self.cache_k.get_shape()
    }
    pub fn v_shape(&self) -> Vec<usize> {
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
    use goldilocks::GoldilocksExt2;

    use super::*;

    impl<N: Number> QKV<N> {
        pub fn random(emb_size: usize, hidden_size: usize) -> Self {
            let q = Tensor::<N>::random(&[emb_size, hidden_size]);
            let q_bias = Tensor::<N>::random(&[hidden_size]);
            let k = Tensor::<N>::random(&[emb_size, hidden_size]);
            let k_bias = Tensor::<N>::random(&[hidden_size]);
            let v = Tensor::<N>::random(&[emb_size, hidden_size]);
            let v_bias = Tensor::<N>::random(&[hidden_size]);
            Self::new(q, q_bias, k, k_bias, v, v_bias)
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
        let q = Tensor::<f32>::random(&[emb_size, hidden_size]);
        let q_bias = Tensor::<f32>::random(&[hidden_size]);
        let k = Tensor::<f32>::random(&[emb_size, hidden_size]);
        let k_bias = Tensor::<f32>::random(&[hidden_size]);
        let v = Tensor::<f32>::random(&[emb_size, hidden_size]);
        let v_bias = Tensor::<f32>::random(&[hidden_size]);
        let qkv = QKV::new(
            q.clone(),
            q_bias.clone(),
            k.clone(),
            k_bias.clone(),
            v.clone(),
            v_bias.clone(),
        );
        let mut input = Tensor::<f32>::random(&[seq_len, emb_size]);
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].get_shape(), vec![seq_len, hidden_size]);
        assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size]);
        let mut out_k = input.matmul(&k).add_dim2(&k_bias);
        assert_eq!(output[1].get_data(), out_k.get_data());
        let mut out_v = input.matmul(&v).add_dim2(&v_bias);
        assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size]);
        assert_eq!(output[2].get_data(), out_v.get_data());
        // second token
        let seq_len = seq_len + 1;
        let new_token_emb = Tensor::<f32>::random(&[1, emb_size]);
        input.concat(new_token_emb.clone());
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].get_shape(), vec![seq_len, hidden_size]);
        assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size]);
        assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size]);
        let out_q = input.matmul(&q).add_dim2(&q_bias);
        assert_eq!(output[0].get_data(), out_q.get_data());
        out_k.concat(new_token_emb.matmul(&k).add_dim2(&k_bias));
        assert_eq!(output[1].get_data(), out_k.get_data());
        out_v.concat(new_token_emb.matmul(&v).add_dim2(&v_bias));
        assert_eq!(output[2].get_data(), out_v.get_data());
    }
}
