use anyhow::ensure;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Element, ScalingFactor, ScalingStrategy, Tensor,
    layers::{
        provable::{Evaluate, LayerOut, NodeId, OpInfo, QuantizeOp, QuantizeOutput},
        requant::Requant,
    },
    padding::PaddingMode,
    quantization::model_scaling_factor_from_tensor_and_bias,
    tensor::{Number, Shape},
};

/// A layer that evaluates the tensor X against the matrices Q, K and V.
/// NOTE: it performs optimizations with the cache, so it actually
/// do the matrix mult only with the last entry of the input.
/// It also outputs only the "small" Q but with the help of caching, it outputs
/// the full K and V matrices as if they were computed using the whole input tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKV<N> {
    pub q: Tensor<N>,
    pub q_bias: Tensor<N>,
    pub k: Tensor<N>,
    pub k_bias: Tensor<N>,
    pub v: Tensor<N>,
    pub v_bias: Tensor<N>,
    // pub cache: Option<CacheQKV<N>>,
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
        Self {
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            // cache: None,
        }
    }
}

impl<N> OpInfo for QKV<N> {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let input_shape = input_shapes[0].clone();
        let output_shapes = vec![
            vec![input_shape[0], self.q.get_shape()[1]].into(),
            vec![input_shape[0], self.k.get_shape()[1]].into(),
            vec![input_shape[0], self.v.get_shape()[1]].into(),
        ];
        match padding_mode {
            PaddingMode::NoPadding => output_shapes,
            PaddingMode::Padding => output_shapes
                .into_iter()
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

#[derive(Debug, Clone)]
pub struct CacheQKV<N> {
    cache_k: Tensor<N>,
    cache_v: Tensor<N>,
    initialized: bool,
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

    use super::*;

    impl<N: Number> QKV<N> {
        pub fn random(emb_size: usize, hidden_size: usize) -> Self {
            let q = Tensor::<N>::random(&vec![emb_size, hidden_size].into());
            let q_bias = Tensor::<N>::random(&vec![hidden_size].into());
            let k = Tensor::<N>::random(&vec![emb_size, hidden_size].into());
            let k_bias = Tensor::<N>::random(&vec![hidden_size].into());
            let v = Tensor::<N>::random(&vec![emb_size, hidden_size].into());
            let v_bias = Tensor::<N>::random(&vec![hidden_size].into());
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
        );
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
}
