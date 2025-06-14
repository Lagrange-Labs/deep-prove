//! Multihead attention:
//! The module performs the reshape and permutation on its input and
//! finally the Q @ K.T per head.
//! The output is a vector of length num_heads where each element is a tuple (CAUSAL(q@k^t),v) of tensors.
//! CAUSAL(..) simply applies the causal mask such that tokens don't look at _future_ tokens.
//! q @ k^t is of shape (1, seq_len)
//! v is of shape (seq_len, head_dim)
//! where seq_len is the length of the sequence, and num_heads is the number of heads.
//! The vector is actually flattened since LayerOut only supports a vector of Tensors, not tuple, so the length is num_heads * 2
//! NOTE: it does NOT Perform the softmax per head neither the subsequent projection with the V matrix.
//! THis is done in subsequent layers due to proving logic proving these operation separately.
use crate::{
    layers::{
        matrix_mul::{self as matmul, OperandMatrix},
        provable::{Evaluate, OpInfo, QuantizeOp, QuantizeOutput},
    },
    padding::PaddingMode,
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
}

impl OpInfo for MhaQK {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        // // qk is now of shape [num_heads,q_len, seq_len]
        // // v is of shape [num_heads, seq_len, head_dim].
        let q_len = input_shapes[0][0];
        let seq_len = input_shapes[1][0];
        assert!(
            q_len == 1 || q_len == seq_len,
            "q should either be a vector OR have same seq_len as K and V"
        );
        match padding_mode {
            PaddingMode::NoPadding => {
                vec![
                    vec![self.num_heads, q_len, seq_len],
                    vec![self.num_heads, seq_len, self.head_dim],
                ]
            }
            PaddingMode::Padding => {
                vec![
                    vec![
                        self.num_heads,
                        q_len.next_power_of_two(),
                        seq_len.next_power_of_two(),
                    ],
                    vec![
                        self.num_heads,
                        seq_len.next_power_of_two(),
                        self.head_dim.next_power_of_two(),
                    ],
                ]
            }
        }
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        2
    }

    fn describe(&self) -> String {
        format!("MHA_QK({},{})", self.num_heads, self.head_dim).to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for MhaQK {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 3, "MHA_QK expects 3 inputs");
        let head_prod = self.num_heads * self.head_dim;
        let q = inputs[0].clone();
        let k = inputs[1].clone();
        let v = inputs[2].clone();
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
        let q_len = q.get_shape()[0];
        let seq_len = k.get_shape()[0];
        ensure!(
            q_len == 1 || q_len == seq_len,
            "q should either be a vector OR have same seq_len as K and V"
        );
        ensure!(
            v.get_shape()[0] == seq_len,
            "v should have the same sequence length as k"
        );
        // reshape into (seq_len, num_head, head_dim)
        let q = q.reshape(vec![q_len, self.num_heads, self.head_dim]);
        let k = k.reshape(vec![seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape(vec![seq_len, self.num_heads, self.head_dim]);
        let q = q.permute3d(&vec![1, 0, 2]); // (num_head, seq_len, head_dim)
        let k = k.permute3d(&vec![1, 0, 2]); // (num_head, seq_len, head_dim)
        let v = v.permute3d(&vec![1, 0, 2]); // (num_head, seq_len, head_dim)
        let mut qkt_heads = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                // shape is now (1, seq_len, head_dim) == [seq_len, head_dim]
                let mini_q = q
                    .slice_3d(head, head + 1)
                    .reshape(vec![q_len, self.head_dim]);
                let mini_k = k
                    .slice_3d(head, head + 1)
                    .reshape(vec![seq_len, self.head_dim]); // [seq_len, head_dim]
                let mini_v = v
                    .slice_3d(head, head + 1)
                    .reshape(vec![seq_len, self.head_dim]); // [seq_len, head_dim]
                // output Q @ K^T <=> [q_len, head_dim] x [seq_len, head_dim]^T is of shape [q_len,seq_len], and v is of shape [seq_len, head_dim]
                Ok(vec![
                    matmul::MatMul::new_with_config(
                        OperandMatrix::Input,
                        OperandMatrix::Input,
                        None, // no bias here
                        matmul::Config::TransposeB,
                    )?
                    .evaluate::<E>(&[&mini_q, &mini_k], vec![])?
                    .outputs
                    .remove(0),
                    mini_v,
                ])
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // merge back the heads together - since proving is expecting one matrix, not a list of vectors
        let mut first_tuple = qkt_heads.remove(0).into_iter();
        // here we reshape to 3d [1, ...] such that concatenation works fine with current concat implementation
        let first_qk = first_tuple.next().unwrap().reshape(vec![1, q_len, seq_len]);
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
        assert_eq!(qk.get_shape(), vec![self.num_heads, q_len, seq_len]);
        assert_eq!(v.get_shape(), vec![self.num_heads, seq_len, self.head_dim]);
        // CAUSAL MASK
        // First it sets to 0 the part that should be ignored on each Q "sequence" for each head
        // Then it adds minus infinity to the same part.
        // We do it in two steps like this because during proving, given we're in integer world, the -minus-infinity
        // would be dynamically depending on the size of Q and K^T. Also because we need to exactly fix -minus-infinity
        // to the lowest minimum value that _softmax_ can handle, so it needs to be a constant. Just "adding the causal mask"
        // would not give us these guarantees.
        let zeros = zeroifier(self.num_heads, q_len, seq_len);
        let minus_infinity = infinitizer(self.num_heads, q_len, seq_len, N::MIN);
        let qk_zeroified = qk.mul(&zeros);
        let qk_infinitized = qk_zeroified.add(&minus_infinity);

        // The next operation in transformer is softmax row by row, and then qk @ v, "row by row" - but
        // it's actually "head by head" which is the highest dimension.
        // So for the shapes, it's [q_len,seq_len] @ [seq_len, head_dim] = [q_len, head_dim]
        // This is done in separate layer in the framework since we first need to prove softmax which happens separatedly
        Ok(LayerOut::from_vec(vec![qk_infinitized, v]))
    }
}

impl QuantizeOp for MhaQK {
    type QuantizedOp = MhaQK;

    // NOTE: no requant layers after that, softmax takes care of it.
    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: crate::layers::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        // it will return a scaling factors for all heads merged together, but that's what we want since we don't want
        // to have one requant layer _per head_ it would be too costly. So we take the min/max accross all the heads concatenated.
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == 2,
            "MHA_QK should have 2 outputs scaling"
        );
        // there is no requant layers after that, softmax takes care of it.
        Ok(QuantizeOutput::new(
            MhaQK::new(self.num_heads, self.head_dim),
            output_scalings,
        ))
    }
}
pub fn zeroifier<N: Number>(num_heads: usize, q_len: usize, seq_len: usize) -> Tensor<N> {
    let zeroified = (0..num_heads)
        .into_par_iter()
        .flat_map(|_head| {
            (0..q_len)
                .into_par_iter()
                .flat_map(|q| {
                    (0..seq_len)
                        .into_iter()
                        .map(|e| if e > q { N::default() } else { N::unit() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tensor::new(vec![num_heads, q_len, seq_len], zeroified)
}

/// Sets to minus infinity the part that should be ignored on each Q "sequence" for each head
pub fn infinitizer<N: Number>(
    num_heads: usize,
    q_len: usize,
    seq_len: usize,
    minus_infinity: N,
) -> Tensor<N> {
    let zeroified = (0..num_heads)
        .into_par_iter()
        .flat_map(|_head| {
            (0..q_len)
                .into_par_iter()
                .flat_map(|q| {
                    (0..seq_len)
                        .into_iter()
                        .map(|e| if e > q { minus_infinity } else { N::default() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tensor::new(vec![num_heads, q_len, seq_len], zeroified)
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_mha_qk_vector_and_matrix() {
        struct Params {
            seq_len: usize,
            q_len: usize,
            should_fail: bool,
        }
        for params in vec![
            Params {
                seq_len: 2,
                q_len: 1,
                should_fail: false,
            },
            Params {
                seq_len: 2,
                q_len: 2,
                should_fail: false,
            },
            Params {
                seq_len: 2,
                q_len: 3,
                should_fail: true,
            },
        ] {
            let num_heads = 2;
            let head_dim = 4;
            let hidden_size = num_heads * head_dim;
            let mha_qk = MhaQK::new(num_heads, head_dim);
            let q_len = params.q_len;
            let seq_len = params.seq_len;
            let q = Tensor::<Element>::random(&vec![q_len, hidden_size]);
            let k = Tensor::<Element>::random(&vec![seq_len, hidden_size]);
            let v = Tensor::<Element>::random(&vec![seq_len, hidden_size]);
            let output = mha_qk.evaluate::<GoldilocksExt2>(&[&q, &k, &v], vec![]);
            if params.should_fail {
                assert!(output.is_err());
                continue;
            }
            let mut output = output.expect("mha_qk should not fail");
            assert_eq!(output.outputs.len(), 2);
            let (qk, v) = (output.outputs.remove(0), output.outputs.remove(0));
            // normally [1,seq_len] per head, so with all heads [num_heads, 1, seq_len]
            assert_eq!(qk.get_shape(), vec![num_heads, q_len, seq_len]);
            // same, but on 3d
            assert_eq!(v.get_shape(), vec![num_heads, seq_len, head_dim]);
            let output_shapes = mha_qk.output_shapes(
                &[q.get_shape(), k.get_shape(), v.get_shape()],
                PaddingMode::NoPadding,
            );
            assert_eq!(output_shapes, vec![qk.get_shape(), v.get_shape()]);
        }
    }

    #[test]
    fn test_zeroifier_and_infinitizer() {
        let num_heads = 2;
        let q_len = 4;
        let seq_len = 4;
        let input = Tensor::<Element>::random(&vec![num_heads, q_len, seq_len]);
        let zeros = zeroifier(num_heads, q_len, seq_len);
        let minus_infinity = infinitizer(num_heads, q_len, seq_len, Element::MIN);
        let zeroified = input.mul(&zeros);
        let infinitized = zeroified.add(&minus_infinity);
        assert_eq!(zeroified.get_shape(), input.get_shape());
        assert_eq!(infinitized.get_shape(), input.get_shape());
        infinitized
            .slice_on_dim(0)
            .enumerate()
            .all(|(head_idx, head)| {
                head.chunks(q_len).enumerate().all(|(q_idx, q)| {
                    q.iter().enumerate().all(|(i, v)| {
                        if i > 0 {
                            input.get(vec![head_idx, q_idx, i]) == *v
                        } else {
                            Element::default() == *v
                        }
                    })
                })
            });
    }
}
