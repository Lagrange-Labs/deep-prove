use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut},
    tensor::Number,
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
        assert_eq!(q.get_shape()[0], q_bias.get_shape()[0], "q.get_shape() {:?} != q_bias.get_shape() {:?}", q.get_shape(), q_bias.get_shape());
        Self {
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
        }
    }
    /// Returns x[-1,..] * Q, X * K, X * V
    pub fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        cache: &mut CacheQKV<N>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 1, "QKV expects 1 input");
        let shape = inputs[0].get_shape();
        let [seq_len, emb_size] = [shape[0], shape[1]];
        let q_emb_size = self.q.get_shape()[0];
        ensure!(q_emb_size == emb_size, "QKV: q_emb_size {} != emb_size {} (input shape {:?} vs q shape {:?})", q_emb_size, emb_size, shape, self.q.get_shape());
        // make sure the size of the input match the size of the cache + 1
        // as we only want to do the the matmul for the new token, not for the previously generated ones
        ensure!(
            seq_len == cache.k_shape()[0] + 1,
            "QKV: seq_len != cache.k_shape()[0] + 1"
        );
        let input = inputs[0].slice_2d(seq_len - 1, seq_len);
        // add row by row
        let q = input.matmul(&self.q).add_dim2(&self.q_bias);
        let k = input.matmul(&self.k).add_dim2(&self.k_bias);
        let v = input.matmul(&self.v).add_dim2(&self.v_bias);
        cache.stack(k, v);
        // vector Q, full K, full V
        Ok(LayerOut::from_vec(vec![q, cache.k(), cache.v()]))
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

    #[test]
    fn test_qkv() {
        // first token
        let seq_len = 1;
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
        let mut input = Tensor::<f32>::random(&[1, emb_size]);
        let mut cache = CacheQKV::new();
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input], &mut cache)
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].get_shape(), vec![1, hidden_size]);
        assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size]);
        let mut out_k = input.matmul(&k).add_dim2(&k_bias);
        assert_eq!(output[1].get_data(), out_k.get_data());
        let mut out_v = input.matmul(&v).add_dim2(&v_bias);
        assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size]);
        assert_eq!(output[2].get_data(), out_v.get_data());
        // second token
        let seq_len = 2;
        let new_token_emb = Tensor::<f32>::random(&[1, emb_size]);
        input.concat(new_token_emb.clone());
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input], &mut cache)
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].get_shape(), vec![1, hidden_size]);
        assert_eq!(output[1].get_shape(), vec![seq_len, hidden_size]);
        assert_eq!(output[2].get_shape(), vec![seq_len, hidden_size]);
        let out_q = new_token_emb.matmul(&q).add_dim2(&q_bias);
        assert_eq!(output[0].get_data(), out_q.get_data());
        out_k.concat(new_token_emb.matmul(&k).add_dim2(&k_bias));
        assert_eq!(output[1].get_data(), out_k.get_data());
        out_v.concat(new_token_emb.matmul(&v).add_dim2(&v_bias));
        assert_eq!(output[2].get_data(), out_v.get_data());
    }
}
