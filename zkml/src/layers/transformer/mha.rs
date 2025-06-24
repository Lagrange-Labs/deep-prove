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
    Element,
    layers::{
        concat_matmul::ConcatMatMul,
        matrix_mul::{self as matmul, OperandMatrix},
        provable::{Evaluate, OpInfo, QuantizeOp, QuantizeOutput},
        reshape::Reshape,
        transformer::softmax::Softmax,
    },
    padding::PaddingMode,
    tensor::{Number, Shape},
};
use anyhow::ensure;
use ff_ext::ExtensionField;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{Tensor, layers::provable::LayerOut};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MhaLinear {
    num_heads: usize,
    head_dim: usize,
}

impl MhaLinear {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
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
                    Shape::new(vec![self.num_heads, q_len, seq_len]),
                    Shape::new(vec![self.num_heads, seq_len, self.head_dim]),
                ]
            }
            PaddingMode::Padding => {
                vec![
                    Shape::new(vec![
                        self.num_heads.next_power_of_two(),
                        q_len.next_power_of_two(),
                        seq_len.next_power_of_two(),
                    ]),
                    Shape::new(vec![
                        self.num_heads.next_power_of_two(),
                        seq_len.next_power_of_two(),
                        self.head_dim.next_power_of_two(),
                    ]),
                ]
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mha<N> {
    linear: MhaLinear,
    softmax: Softmax<N>,
    final_mul: ConcatMatMul,
    final_reshape: Reshape,
}

impl<N: Number> Mha<N> {
    pub fn new(num_heads: usize, head_dim: usize) -> anyhow::Result<Self> {
        let linear = MhaLinear {
            num_heads,
            head_dim,
        };
        let softmax = Softmax::new()
            .with_scale(N::from_f32((1.0 / (head_dim as f32)).sqrt())?)
            .on_dim(1);
        let permutation = vec![1, 0, 2];
        let final_mul = ConcatMatMul::new_with_permute(permutation);
        // reshape the output from [q_len, num_heads, head_dim] to [q_len, num_heads*head_dim]
        let final_reshape = Reshape::new_subspace(1..=2, vec![num_heads * head_dim]);
        Ok(Self {
            linear,
            softmax,
            final_mul,
            final_reshape,
        })
    }

    pub(crate) fn evaluate_with_softmax_out<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<(LayerOut<N, E>, LayerOut<N, E>)>
    where
        Softmax<N>: Evaluate<N>,
    {
        let unpadded_input_shapes = if unpadded_input_shapes.is_empty() {
            // take input shapes from inputs
            inputs.iter().map(|input| input.get_shape()).collect()
        } else {
            unpadded_input_shapes
        };

        let linear_out = self
            .linear
            .evaluate::<E>(inputs, unpadded_input_shapes.clone())?;

        let linear_out_shapes = self
            .linear
            .output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);

        // apply softmax
        let soft_out = self
            .softmax
            .evaluate::<E>(&vec![linear_out.outputs()[0]], linear_out_shapes.clone())?;

        let soft_out_shapes = self
            .softmax
            .output_shapes(&linear_out_shapes, PaddingMode::NoPadding);

        assert_eq!(
            soft_out.outputs()[0].get_shape()[0],
            linear_out.outputs()[1].get_shape()[0],
            "First dimension must match for ConcatMatMul: {:?} vs {:?}",
            soft_out.outputs()[0].get_shape(),
            linear_out.outputs()[1].get_shape()
        );

        // For each head, the matrix multiplication dimensions should align:
        // [seq_len, seq_len] @ [seq_len, head_dim] -> [seq_len, head_dim]
        assert_eq!(
            soft_out.outputs()[0].get_shape()[2],
            linear_out.outputs()[1].get_shape()[1],
            "Inner dimensions must match for matrix multiplication: {:?} vs {:?}",
            soft_out.outputs()[0].get_shape(),
            linear_out.outputs()[1].get_shape()
        );

        let out = self.final_mul.evaluate::<E>(
            &vec![soft_out.outputs()[0], linear_out.outputs()[1]],
            soft_out_shapes.clone(),
        )?;

        let out_shapes = self
            .final_mul
            .output_shapes(&soft_out_shapes, PaddingMode::NoPadding);

        let out = self.final_reshape.evaluate(&out.outputs(), out_shapes)?;

        Ok((out, soft_out))
    }
}

impl<N: Number> OpInfo for Mha<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Vec<Shape> {
        let linear_out_shapes = self.linear.output_shapes(input_shapes, padding_mode);

        let soft_out_shapes = self.softmax.output_shapes(&linear_out_shapes, padding_mode);

        let final_mul_shapes = self.final_mul.output_shapes(&soft_out_shapes, padding_mode);

        self.final_reshape
            .output_shapes(&final_mul_shapes, padding_mode)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!("MHA_QK({},{})", self.linear.num_heads, self.linear.head_dim).to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for MhaLinear {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
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
        let q = q.reshape(vec![q_len, self.num_heads, self.head_dim].into());
        let k = k.reshape(vec![seq_len, self.num_heads, self.head_dim].into());
        let v = v.reshape(vec![seq_len, self.num_heads, self.head_dim].into());
        let q = q.permute3d(&[1, 0, 2]); // (num_head, seq_len, head_dim)
        let k = k.permute3d(&[1, 0, 2]); // (num_head, seq_len, head_dim)
        let v = v.permute3d(&[1, 0, 2]); // (num_head, seq_len, head_dim)
        let mut qkt_heads = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                // shape is now (1, seq_len, head_dim) == [seq_len, head_dim]
                let mini_q = q
                    .slice_3d(head, head + 1)
                    .reshape(vec![q_len, self.head_dim].into());
                let mini_k = k
                    .slice_3d(head, head + 1)
                    .reshape(vec![seq_len, self.head_dim].into()); // [seq_len, head_dim]
                let mini_v = v
                    .slice_3d(head, head + 1)
                    .reshape(vec![seq_len, self.head_dim].into()); // [seq_len, head_dim]
                // output Q @ K^T <=> [q_len, head_dim] x [seq_len, head_dim]^T is of shape [q_len,seq_len], and v is of shape [seq_len, head_dim]
                Ok((
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
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // merge back the heads together - since proving is expecting one matrix, not a list of vectors
        let (first_qk, first_v) = qkt_heads.remove(0);
        // here we reshape to 3d [1, ...] such that concatenation works fine with current concat implementation
        let first_qk = first_qk.reshape(vec![1, q_len, seq_len].into());
        let first_v = first_v.reshape(vec![1, seq_len, self.head_dim].into());
        let (qk, v) =
            qkt_heads
                .into_iter()
                .fold((first_qk, first_v), |(mut acc_qk, mut acc_v), (qk, v)| {
                    acc_qk.concat(qk);
                    acc_v.concat(v);
                    (acc_qk, acc_v)
                });
        assert_eq!(qk.get_shape(), vec![self.num_heads, q_len, seq_len].into());
        assert_eq!(
            v.get_shape(),
            vec![self.num_heads, seq_len, self.head_dim].into()
        );
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

impl Evaluate<f32> for Mha<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let (out, _) = self.evaluate_with_softmax_out(inputs, unpadded_input_shapes)?;

        Ok(out)
    }
}

impl Evaluate<Element> for Mha<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        let (out, _) = self.evaluate_with_softmax_out(inputs, unpadded_input_shapes)?;

        Ok(out)
    }
}

impl QuantizeOp for Mha<f32> {
    type QuantizedOp = Mha<Element>;

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
            Mha::new(self.linear.num_heads, self.linear.head_dim)?,
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
                        .map(|e| if e > q { N::default() } else { N::unit() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tensor::new(vec![num_heads, q_len, seq_len].into(), zeroified)
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
                        .map(|e| if e > q { minus_infinity } else { N::default() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tensor::new(vec![num_heads, q_len, seq_len].into(), zeroified)
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

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
            let mha_qk = MhaLinear {
                num_heads,
                head_dim,
            };
            let q_len = params.q_len;
            let seq_len = params.seq_len;
            let q = Tensor::<Element>::random(&vec![q_len, hidden_size].into());
            let k = Tensor::<Element>::random(&vec![seq_len, hidden_size].into());
            let v = Tensor::<Element>::random(&vec![seq_len, hidden_size].into());
            let output = mha_qk.evaluate::<GoldilocksExt2>(&[&q, &k, &v], vec![]);
            if params.should_fail {
                assert!(output.is_err());
                continue;
            }
            let mut output = output.expect("mha_qk should not fail");
            assert_eq!(output.outputs.len(), 2);
            let (qk, v) = (output.outputs.remove(0), output.outputs.remove(0));
            // normally [1,seq_len] per head, so with all heads [num_heads, 1, seq_len]
            assert_eq!(qk.get_shape(), vec![num_heads, q_len, seq_len].into());
            // same, but on 3d
            assert_eq!(v.get_shape(), vec![num_heads, seq_len, head_dim].into());
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
        let input = Tensor::<Element>::random(&vec![num_heads, q_len, seq_len].into());
        let zeros = zeroifier(num_heads, q_len, seq_len);
        let minus_infinity = infinitizer(num_heads, q_len, seq_len, Element::MIN);
        let zeroified = input.mul(&zeros);
        let infinitized = zeroified.add(&minus_infinity);
        assert_eq!(zeroified.get_shape(), input.get_shape());
        assert_eq!(infinitized.get_shape(), input.get_shape());
        let (slice_it, _) = infinitized.slice_on_dim(0);
        slice_it.enumerate().all(|(head_idx, head)| {
            head.chunks(q_len).enumerate().all(|(q_idx, q)| {
                q.iter().enumerate().all(|(i, v)| {
                    let input_value = input.get(vec![head_idx, q_idx, i]);
                    // if we are less than the q_len, we dont have causal mask
                    if i <= q_idx {
                        input_value == *v
                    } else {
                        // otherwise we have causal mask
                        *v == Element::MIN
                    }
                })
            })
        });
    }
}
