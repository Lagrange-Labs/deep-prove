//! Multihead attention layer:
//! The module performs all the operations inside the multi-head attention layer, relying on
//! ConcatMatMul and Softmax layers as building blocks.
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
    quantization::Fieldizer,
    tensor::{Number, Shape},
};
use anyhow::ensure;
use ff_ext::{ExtensionField, FieldFrom};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{Tensor, layers::provable::LayerOut};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MhaQk {
    num_heads: usize,
    head_dim: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MhaFinalMul {
    num_heads: usize,
    head_dim: usize,
    mul: ConcatMatMul,
}

impl MhaQk {
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
                vec![vec![self.num_heads, q_len, seq_len].into()]
            }
            PaddingMode::Padding => {
                vec![
                    vec![
                        self.num_heads.next_power_of_two(),
                        q_len.next_power_of_two(),
                        seq_len.next_power_of_two(),
                    ]
                    .into(),
                ]
            }
        }
    }
}

impl MhaFinalMul {
    fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            mul: ConcatMatMul::new_with_permute(vec![1, 0, 2]),
        }
    }

    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let seq_len = input_shapes[1][0];
        let v_shape = match padding_mode {
            PaddingMode::NoPadding => {
                vec![self.num_heads, seq_len, self.head_dim]
            }
            PaddingMode::Padding => {
                vec![
                    self.num_heads.next_power_of_two(),
                    seq_len.next_power_of_two(),
                    self.head_dim.next_power_of_two(),
                ]
            }
        }
        .into();
        assert_eq!(
            input_shapes[0][2], seq_len,
            "qk should have the same sequence length as v"
        );
        let mul_input_shapes = vec![
            input_shapes[0].clone(), // QK
            v_shape,                 // V
        ];
        self.mul.output_shapes(&mul_input_shapes, padding_mode)
    }

    fn describe(&self) -> String {
        self.mul.describe()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mha<N> {
    linear: MhaQk,
    softmax: Softmax<N>,
    final_mul: MhaFinalMul,
    final_reshape: Reshape,
}

impl<N: Number> Mha<N> {
    pub fn new(num_heads: usize, head_dim: usize) -> anyhow::Result<Self> {
        let linear = MhaQk {
            num_heads,
            head_dim,
        };
        let softmax = Softmax::new()
            .with_scale(N::from_f32((1.0 / (head_dim as f32)).sqrt())?)
            .on_dim(1);
        let final_mul = MhaFinalMul::new(num_heads, head_dim);
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

        ensure!(
            inputs.len() == 3,
            "MHA layer expects 3 inputs, found {}",
            inputs.len()
        );

        let linear_out = self
            .linear
            .evaluate::<E>(&inputs[..2], unpadded_input_shapes[..2].to_vec())?;

        let linear_out_shapes = self
            .linear
            .output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);

        // apply softmax
        let soft_out = self
            .softmax
            .evaluate::<E>(&linear_out.outputs(), linear_out_shapes.clone())?;

        let soft_out_shapes = self
            .softmax
            .output_shapes(&linear_out_shapes, PaddingMode::NoPadding);

        ensure!(
            soft_out.outputs().len() == 1,
            "Softmax should return one output"
        );

        let final_mul_input_shapes =
            vec![soft_out_shapes[0].clone(), unpadded_input_shapes[2].clone()];

        let out = self.final_mul.evaluate::<E>(
            &[soft_out.outputs()[0], inputs[2]],
            final_mul_input_shapes.clone(),
        )?;

        let out_shapes = self
            .final_mul
            .output_shapes(&final_mul_input_shapes, PaddingMode::NoPadding);

        let out = self.final_reshape.evaluate(&out.outputs(), out_shapes)?;

        Ok((out, soft_out))
    }
}

impl<N: Number> OpInfo for Mha<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let linear_out_shapes = self.linear.output_shapes(&input_shapes[..2], padding_mode);

        let soft_out_shapes = self.softmax.output_shapes(&linear_out_shapes, padding_mode);

        let final_mul_input_shapes = vec![
            soft_out_shapes[0].clone(),
            input_shapes[2].clone(), // V
        ];

        let final_mul_shapes = self
            .final_mul
            .output_shapes(&final_mul_input_shapes, padding_mode);

        self.final_reshape
            .output_shapes(&final_mul_shapes, padding_mode)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "MHA({},{}): \t {}, \t {}",
            self.linear.num_heads,
            self.linear.head_dim,
            self.softmax.describe(),
            self.final_mul.describe(),
        )
        .to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for MhaQk {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "MHA_QK expects 2 inputs");
        let head_prod = self.num_heads * self.head_dim;
        let q = inputs[0].clone();
        let k = inputs[1].clone();
        ensure!(
            q.get_shape()[1] == head_prod,
            "q should have the same number of elements as the product of the number of heads and the head dimension"
        );
        ensure!(
            k.get_shape()[1] == head_prod,
            "k should have the same number of elements as the product of the number of heads and the head dimension"
        );
        let q_len = q.get_shape()[0];
        let seq_len = k.get_shape()[0];
        ensure!(
            q_len == 1 || q_len == seq_len,
            "q should either be a vector OR have same seq_len as K and V"
        );
        // reshape into (seq_len, num_head, head_dim)
        let q = q.reshape(vec![q_len, self.num_heads, self.head_dim].into());
        let k = k.reshape(vec![seq_len, self.num_heads, self.head_dim].into());
        let q = q.permute3d(&[1, 0, 2]); // (num_head, seq_len, head_dim)
        let k = k.permute3d(&[1, 0, 2]); // (num_head, seq_len, head_dim)
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
                // output Q @ K^T <=> [q_len, head_dim] x [seq_len, head_dim]^T is of shape [q_len,seq_len], and v is of shape [seq_len, head_dim]
                Ok(matmul::MatMul::new_with_config(
                    OperandMatrix::Input,
                    OperandMatrix::Input,
                    None, // no bias here
                    matmul::Config::TransposeB,
                )?
                .evaluate::<E>(&[&mini_q, &mini_k], vec![])?
                .outputs
                .remove(0))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // merge back the heads together - since proving is expecting one matrix, not a list of vectors
        let first_qk = qkt_heads.remove(0);
        // here we reshape to 3d [1, ...] such that concatenation works fine with current concat implementation
        let first_qk = first_qk.reshape(vec![1, q_len, seq_len].into());
        let qk = qkt_heads.into_iter().fold(first_qk, |mut acc_qk, qk| {
            acc_qk.concat(qk);
            acc_qk
        });
        assert_eq!(qk.get_shape(), vec![self.num_heads, q_len, seq_len].into());
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
        Ok(LayerOut::from_vec(vec![qk_infinitized]))
    }
}

impl<N: Number> Evaluate<N> for MhaFinalMul {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "MHA_FinalMul expects 2 inputs");
        let head_prod = self.num_heads * self.head_dim;
        let qk = inputs[0].clone();
        let v = inputs[1].clone();
        ensure!(
            v.get_shape()[1] == head_prod,
            "v should have the same number of elements as the product of the number of heads and the head dimension"
        );
        let seq_len = v.get_shape()[0];
        ensure!(
            qk.get_shape()[2] == seq_len,
            "qk should have the same sequence length as v"
        );
        ensure!(
            qk.get_shape()[0] == self.num_heads,
            "qk should have the same number of heads as the MHA"
        );
        let v = v.reshape(vec![seq_len, self.num_heads, self.head_dim].into());
        let v = v.permute3d(&[1, 0, 2]); // (num_head, seq_len, head_dim)
        assert_eq!(
            v.get_shape(),
            vec![self.num_heads, seq_len, self.head_dim].into()
        );

        let unpadded_seq_len = unpadded_input_shapes[1][0];

        self.mul.evaluate(
            &[&qk, &v],
            vec![
                unpadded_input_shapes[0].clone(),
                vec![self.num_heads, unpadded_seq_len, self.head_dim].into(),
            ],
        )
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

/// Method to efficienctly evaluate the MLE of the zeroifier matrix over a random
/// point. The point is provided already split between coordinates referring to the
/// columns and coordinates referring to the rows of the matrix.
/// Currently, it works only for a square zeroifier matrix
pub fn eval_zeroifier_mle<F: ExtensionField + FieldFrom<u64>>(
    column_point: &[F],
    row_point: &[F],
) -> F {
    column_point
        .iter()
        .zip(row_point)
        .fold(F::from_v(1), |acc, (&c, &r)| {
            acc * (F::from_v(1) - c - r + F::from_v(2) * c * r) + (F::from_v(1) - c) * r
        })
}

/// Method to efficienctly evaluate the MLE of the infinitizer matrix over a random
/// point. The point is provided already split between coordinates referring to the
/// columns and coordinates referring to the rows of the matrix.
/// Currently, it works only for a square infinitizer matrix
pub fn eval_infinitizer_mle<F: ExtensionField + FieldFrom<u64>>(
    column_point: &[F],
    row_point: &[F],
    minus_infinity: Element,
) -> F {
    <Element as Fieldizer<F>>::to_field(&minus_infinity)
        * (F::ONE - eval_zeroifier_mle(column_point, row_point))
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{
        Element,
        quantization::{self, Fieldizer},
        testing::random_field_vector,
    };

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
            let mha_qk = MhaQk {
                num_heads,
                head_dim,
            };
            let q_len = params.q_len;
            let seq_len = params.seq_len;
            let q = Tensor::<Element>::random(&vec![q_len, hidden_size].into());
            let k = Tensor::<Element>::random(&vec![seq_len, hidden_size].into());
            let output = mha_qk.evaluate::<GoldilocksExt2>(&[&q, &k], vec![]);
            if params.should_fail {
                assert!(output.is_err());
                continue;
            }
            let mut output = output.expect("mha_qk should not fail");
            assert_eq!(output.outputs.len(), 1);
            let qk = output.outputs.remove(0);
            // normally [1,seq_len] per head, so with all heads [num_heads, 1, seq_len]
            assert_eq!(qk.get_shape(), vec![num_heads, q_len, seq_len].into());
            let output_shapes =
                mha_qk.output_shapes(&[q.get_shape(), k.get_shape()], PaddingMode::NoPadding);
            assert_eq!(output_shapes, vec![qk.get_shape()]);
        }
    }

    #[test]
    fn test_mha_final_mul() {
        struct Params {
            seq_len: usize,
            q_len: usize,
        }
        for params in vec![
            Params {
                seq_len: 2,
                q_len: 1,
            },
            Params {
                seq_len: 2,
                q_len: 2,
            },
        ] {
            let num_heads = 2;
            let head_dim = 4;
            let hidden_size = num_heads * head_dim;
            let q_len = params.q_len;
            let seq_len = params.seq_len;
            let qk = Tensor::<Element>::random(&vec![num_heads, q_len, seq_len].into());
            let v = Tensor::<Element>::random(&vec![seq_len, hidden_size].into());
            let mha_mul = MhaFinalMul::new(num_heads, head_dim);
            let mut output = mha_mul
                .evaluate::<GoldilocksExt2>(&[&qk, &v], vec![qk.get_shape(), v.get_shape()])
                .expect("mha_final_mul should not fail");
            assert_eq!(output.outputs.len(), 1);
            let out = output.outputs.remove(0);
            assert_eq!(out.get_shape(), vec![q_len, num_heads, head_dim].into());
            let output_shapes =
                mha_mul.output_shapes(&[qk.get_shape(), v.get_shape()], PaddingMode::NoPadding);
            assert_eq!(output_shapes, vec![out.get_shape()]);
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

    // Testing method which, given as input the big-endian bit representations of 2 integers `x`, `y`,
    // returns 1 if x <= y, 0 otherwise. The output is computed through a multi-linear polynomial, which
    // should correspond to the MLE of the zeroifier matrix
    fn eval_lteq_poly(x_i: &[Element], y_i: &[Element]) -> Element {
        assert_eq!(x_i.len(), y_i.len());
        x_i.into_iter()
            .rev()
            .zip(y_i.into_iter().rev())
            .fold(Element::from(1), |acc, (x, y)| {
                acc * (1 - x - y + 2 * x * y) + (1 - x) * y
            })
    }

    fn to_be_bits<const NUM_BITS: usize>(x: Element) -> [Element; NUM_BITS] {
        (0..NUM_BITS)
            .rev()
            .map(|i| {
                let mask = 1 << i;
                let bit = (x & Element::from(mask)) >> i;
                bit
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn test_zeroifier_evaluation_for_num_heads<const NUM_HEADS_BITS: usize>() {
        // create zeroifier matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let num_heads = 1 << NUM_HEADS_BITS;

        let zeroifier = zeroifier::<Element>(num_heads, num_columns, num_columns);

        let zeroifier_heads = {
            let (it, shape) = zeroifier.slice_on_dim(0);
            it.map(|data| Tensor::new(shape.clone(), data.to_vec()))
                .collect_vec()
        };

        assert_eq!(zeroifier_heads[0].get_2d(0, 0), Element::from(1));
        assert_eq!(
            zeroifier_heads[0].get_2d(num_columns - 1, num_columns - 1),
            Element::from(1)
        );
        assert_eq!(zeroifier_heads[0].get_2d(0, 1), Element::from(0));
        assert_eq!(zeroifier_heads[0].get_2d(1, 1), Element::from(1));
        assert_eq!(zeroifier_heads[0].get_2d(1, 2), Element::from(0));

        let mle = zeroifier.to_mle_flat::<GoldilocksExt2>();

        for i in 0..num_columns {
            for j in 0..num_columns {
                for h in 0..num_heads {
                    let x_i = to_be_bits::<NUM_BITS>(Element::from(i as u64));
                    let y_i = to_be_bits::<NUM_BITS>(Element::from(j as u64));
                    let h_i = to_be_bits::<NUM_HEADS_BITS>(Element::from(h as u64));
                    // check that the zeroifier matrix is equivalent to the lteq function
                    let cmp = eval_lteq_poly(&y_i, &x_i);
                    assert_eq!(
                        zeroifier_heads[h].get_2d(i, j),
                        cmp,
                        "Zeroifier evaluation failed for ({}, {})",
                        i,
                        j
                    );
                    // build point for MLE: first column bits in little-endian order, then rows bits in little-endian order,
                    // then head bits in little-endian order
                    let point = y_i
                        .into_iter()
                        .rev()
                        .chain(x_i.into_iter().rev())
                        .chain(h_i.into_iter().rev())
                        .map(|bit| GoldilocksExt2::from_v(bit as u64))
                        .collect_vec();
                    let eval = mle.evaluate(&point);
                    assert_eq!(eval, cmp.to_field());
                    // check that the MLE evaluation with the formula is the same as `eval`.
                    // Note that the evaluation is independent from `num_heads` dimension, as the
                    // zeroifier matrix is repeated across all the heads
                    let quick_eval =
                        eval_zeroifier_mle(&point[..NUM_BITS], &point[NUM_BITS..NUM_BITS * 2]);
                    assert_eq!(eval, quick_eval);
                }
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<GoldilocksExt2>(NUM_BITS * 2 + NUM_HEADS_BITS);
            assert_eq!(
                mle.evaluate(&point),
                eval_zeroifier_mle(&point[..NUM_BITS], &point[NUM_BITS..NUM_BITS * 2],),
            );
        }
    }

    #[test]
    fn test_zeroifier_evaluation() {
        // test with a single head
        test_zeroifier_evaluation_for_num_heads::<0>();
        // test with multiple heads
        test_zeroifier_evaluation_for_num_heads::<2>();
    }

    // Testing method which, given as input the big-endian bit representations of 2 integers `x`, `y`,
    // returns `minus_infinity` if x > y, 0 otherwise. The output is computed through a multi-linear polynomial, which
    // should correspond to the MLE of the infinitizer matrix
    fn eval_gt_poly(x_i: &[Element], y_i: &[Element], minus_infinity: Element) -> Element {
        minus_infinity * (Element::unit() - eval_lteq_poly(x_i, y_i))
    }

    fn test_infinitizer_evaluation_for_num_heads<const NUM_HEADS_BITS: usize>() {
        // create infinitizer matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let num_heads = 1 << NUM_HEADS_BITS;

        let minus_infinity = *quantization::MIN;

        let infinitizer =
            infinitizer::<Element>(num_heads, num_columns, num_columns, minus_infinity);

        let infinitizer_heads = {
            let (it, shape) = infinitizer.slice_on_dim(0);
            it.map(|data| Tensor::new(shape.clone(), data.to_vec()))
                .collect_vec()
        };

        assert_eq!(infinitizer_heads[0].get_2d(0, 0), Element::from(0));
        assert_eq!(
            infinitizer_heads[0].get_2d(num_columns - 1, num_columns - 1),
            Element::from(0)
        );
        assert_eq!(
            infinitizer_heads[0].get_2d(0, 1),
            Element::from(minus_infinity)
        );
        assert_eq!(infinitizer_heads[0].get_2d(1, 1), Element::from(0));
        assert_eq!(
            infinitizer_heads[0].get_2d(1, 2),
            Element::from(minus_infinity)
        );

        let mle = infinitizer.to_mle_flat::<GoldilocksExt2>();

        for i in 0..num_columns {
            for j in 0..num_columns {
                for h in 0..num_heads {
                    let x_i = to_be_bits::<NUM_BITS>(Element::from(i as u64));
                    let y_i = to_be_bits::<NUM_BITS>(Element::from(j as u64));
                    let h_i = to_be_bits::<NUM_HEADS_BITS>(Element::from(h as u64));
                    // check that the zeroifier matrix is equivalent to the gt function with output being minus_infinity
                    let cmp = eval_gt_poly(&y_i, &x_i, minus_infinity);
                    assert_eq!(
                        infinitizer_heads[h].get_2d(i, j),
                        cmp,
                        "Zeroifier evaluation failed for ({}, {})",
                        i,
                        j
                    );
                    // build point for MLE: first column bits in little-endian order, then rows bits in little-endian order,
                    // then head bits in little-endian order
                    let point = y_i
                        .into_iter()
                        .rev()
                        .chain(x_i.into_iter().rev())
                        .chain(h_i.into_iter().rev())
                        .map(|bit| GoldilocksExt2::from_v(bit as u64))
                        .collect_vec();
                    let eval = mle.evaluate(&point);
                    println!("{cmp} {} {}", cmp as u64, u64::MAX - 2);
                    assert_eq!(eval, cmp.to_field());
                    // check that the MLE evaluation with the formula is the same as `eval`.
                    // Note that the evaluation is independent from `num_heads` dimension, as the
                    // zeroifier matrix is repeated across all the heads
                    let quick_eval = eval_infinitizer_mle(
                        &point[..NUM_BITS],
                        &point[NUM_BITS..NUM_BITS * 2],
                        minus_infinity,
                    );
                    assert_eq!(eval, quick_eval);
                }
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<GoldilocksExt2>(NUM_BITS * 2 + NUM_HEADS_BITS);
            assert_eq!(
                mle.evaluate(&point),
                eval_infinitizer_mle(
                    &point[..NUM_BITS],
                    &point[NUM_BITS..NUM_BITS * 2],
                    minus_infinity,
                ),
            );
        }
    }

    #[test]
    fn test_infinitizer_evaluation() {
        // test infinitizer with a single head
        test_infinitizer_evaluation_for_num_heads::<0>();
        // test infinitizer with multiple heads
        test_infinitizer_evaluation_for_num_heads::<3>();
    }
}
