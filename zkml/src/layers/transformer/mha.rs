//! Multihead attention:
//! The module performs the reshape and permutation on its input and
//! finally the Q @ K.T per head.
//! The output is a vector of length num_heads where each element is a tuple (q@k^t,v) of tensors.
//! q @ k^t is of shape (1, seq_len)
//! v is of shape (seq_len, head_dim)
//! where seq_len is the length of the sequence, and num_heads is the number of heads.
//! The vector is actually flattened since LayerOut only supports a vector of Tensors, not tuple, so the length is num_heads * 2
//! NOTE: it does NOT Perform the softmax per head neither the subsequent projection with the V matrix.
//! THis is done in subsequent layers due to proving logic proving these operation separately.
use crate::{
    layers::{provable::Evaluate, transformer::qkt::QKT},
    tensor::Number,
};
use anyhow::ensure;
use ff_ext::ExtensionField;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{Tensor, layers::provable::LayerOut};
#[derive(Clone, Debug)]
pub struct MhaQK {
    num_heads: usize,
    head_dim: usize,
}

impl MhaQK {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
        }
    }

    pub fn evaluate<N: Number, E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 3, "MHA_QK expects 3 inputs");
        let head_prod = self.num_heads * self.head_dim;
        let q = inputs[0].clone();
        let k = inputs[1].clone();
        let v = inputs[2].clone();
        ensure!(q.get_shape()[0] == 1, "q should be only a vector");
        ensure!(
            q.get_shape()[1] == head_prod,
            "q should have the same number of elements as the product of the number of heads and the head dimension"
        );
        ensure!(
            k.get_shape()[1] == head_prod,
            "k should have the same number of elements as the product of the number of heads and the head dimension"
        );
        ensure!(
            v.get_shape()[1] == head_prod,
            "v should have the same number of elements as the product of the number of heads and the head dimension"
        );
        let seq_len = k.get_shape()[0];
        ensure!(
            v.get_shape()[0] == seq_len,
            "v should have the same sequence length as k"
        );
        // reshape into (seq_len, num_head, head_dim)
        let q = q.reshape(vec![1, self.num_heads, self.head_dim]);
        let k = k.reshape(vec![seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape(vec![seq_len, self.num_heads, self.head_dim]);
        let q = q.permute3d(&vec![1, 0, 2]); // (num_head, 1, head_dim)
        let k = k.permute3d(&vec![1, 0, 2]); // (num_head, seq_len, head_dim)
        let v = v.permute3d(&vec![1, 0, 2]); // (num_head, seq_len, head_dim)
        let mut qkt_heads = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                // shape is now (1, seq_len, head_dim) == [seq_len, head_dim]
                let mini_q = q.slice_3d(head, head + 1).reshape(vec![1, self.head_dim]); // [1, head_dim]
                let mini_k = k
                    .slice_3d(head, head + 1)
                    .reshape(vec![seq_len, self.head_dim]); // [seq_len, head_dim]
                let mini_v = v
                    .slice_3d(head, head + 1)
                    .reshape(vec![seq_len, self.head_dim]); // [seq_len, head_dim]
                // output Q @ K^T is of shape [1, seq_len], and v is of shape [seq_len, head_dim]
                Ok(vec![
                    QKT.evaluate::<E>(&[&mini_q, &mini_k], vec![])?
                        .outputs
                        .remove(0),
                    mini_v,
                ])
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // merge back the heads together - since proving is expecting one matrix, not a list of vectors
        let mut first_tuple = qkt_heads.remove(0).into_iter();
        // here we reshape to 3d [1, ...] such that concatenation works fine with current implementation
        let first_qk = first_tuple.next().unwrap();
        let first_v = first_tuple
            .next()
            .unwrap()
            .reshape(vec![1, seq_len, self.head_dim]);
        let (qk, v) =
            qkt_heads
                .into_iter()
                .fold((first_qk, first_v), |(mut acc_qk, mut acc_v), head| {
                    let mut head_it = head.into_iter();
                    acc_qk.concat(head_it.next().unwrap());
                    acc_v.concat(head_it.next().unwrap());
                    (acc_qk, acc_v)
                });
        // qk is now of shape [num_heads,seq_len]
        assert_eq!(qk.get_shape(), vec![self.num_heads, seq_len]);
        // v is of shape [num_heads, seq_len, head_dim].
        assert_eq!(v.get_shape(), vec![self.num_heads, seq_len, self.head_dim]);
        // The next operation in transformer is softmax row by row, and then qk @ v, "row by row" - but
        // it's actually "head by head" which is the highest dimension.
        // So for the shapes, it's [1,seq_len] @ [seq_len, head_dim] = [1, head_dim] (1 because row by row for each head)
        // This is done in separate layer in the framework since we first need to prove softmax which happens separatedly
        Ok(LayerOut::from_vec(vec![qk, v]))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_mha_qk() {
        let num_heads = 2;
        let head_dim = 4;
        let hidden_size = num_heads * head_dim;
        let mha_qk = MhaQK::new(num_heads, head_dim);
        let seq_len = 2;
        let q = Tensor::<Element>::random(&vec![1, hidden_size]);
        let k = Tensor::<Element>::random(&vec![seq_len, hidden_size]);
        let v = Tensor::<Element>::random(&vec![seq_len, hidden_size]);
        let mut output = mha_qk
            .evaluate::<_, GoldilocksExt2>(&[&q, &k, &v])
            .expect("mha_qk should not fail");
        assert_eq!(output.outputs.len(), 2);
        let (qk, v) = (output.outputs.remove(0), output.outputs.remove(0));
        // normally [1,seq_len] per head, so with all heads [num_heads, seq_len]
        assert_eq!(qk.get_shape(), vec![num_heads, seq_len]);
        // same, but on 3d
        assert_eq!(v.get_shape(), vec![num_heads, seq_len, head_dim]);
    }
}
