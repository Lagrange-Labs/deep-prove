//! Module to infere and prove statements of the form:
//! A = A_1 || A_2 || ... || A_n
//! B = B_1 || B_2 || ... || B_n
//! C = C_1 || C_2 || ... || C_n
//! where C_i = A_i @ B_i
//! Here concatenation means concatenation over the highest dimension, e.g.
//! if A_i is of shape [1, r, s] then A = [A_1, A_2, ... , A_n] is of shape [n, r, s]
//!
//! This module currently only supports the case where A_i and B_i are witnesses values.
//! Transpose: There is the option to transpose the output of the matmul. This is useful for proving to avoid
//! having to prove explicitly the transpose operation with a separate layer, as sumcheck based proving can directly
//! prove the transpose at the same time as the matmul.
use std::borrow::Borrow;

use anyhow::{ensure, Result};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use serde::{de::DeserializeOwned, Serialize, Deserialize};
use transcript::Transcript;

use crate::{
    iop::{context::{ContextAux, ShapeStep}, verifier::Verifier}, layers::{
        provable::{Evaluate, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, VerifiableCtx},
        requant::Requant,
    }, model::StepData, padding::PaddingMode, tensor::{Number, Shape}, Claim, Prover, Tensor
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    pub fn new(perm: Vec<usize>) -> Self {
        assert!(perm.len() > 0, "Permutation must have at least one element");
        assert!(
            perm.iter().all(|&x| x < perm.len()),
            "Permutation indices must be less than the length of the permutation"
        );
        Self(perm)
    }

    pub fn apply(&self, shape: &Shape) -> Shape {
        shape.permute(&self.0)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputMatrix {
    /// Index in the input shape that refers the dimension that we concatenate over.
    concat_dimension: usize,
    /// Specify whether the shape of the input matrix needs to be permuted before the matmul
    permute: Option<Permutation>,
}

impl InputMatrix {
    pub fn new_with_permute(concat_dimension: usize, permute: Permutation) -> Result<Self> {
        ensure!(
            permute.0[concat_dimension] == 0,
            "The concatenation dimension must be the first dimension after the permutation is applied"
        );
        Ok(Self {
            concat_dimension,
            permute: Some(permute),
        })
    }
}

use super::provable::LayerOut;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatMatMul {
    left: InputMatrix,
    right: InputMatrix,
    /// If Some, it contains the permutation to apply to the output of the matmul.
    permute: Option<Permutation>,
    /// It tells what is the maximum bit size we ever expect the output of this layer to be.
    /// NOTE: This is a config item normally but we need this information during quantization.
    /// Best would be to rework quantization trait to include such config items.
    intermediate_bit_size: usize,
}

#[derive(Clone, Debug)]
pub struct ConcatMatMulCtx {

}

#[derive(Clone, Debug)]
pub struct ConcatMatMulProof {

}

const DEFAULT_INTERMEDIATE_BIT_SIZE: usize = 25;

impl ConcatMatMul {
    pub fn new(left: InputMatrix, right: InputMatrix) -> Self {
        Self {
            left,
            right,
            permute: None,
            intermediate_bit_size: DEFAULT_INTERMEDIATE_BIT_SIZE,
        }
    }
    pub fn new_with_permute(left: InputMatrix, right: InputMatrix, permutation: Permutation) -> Self {
        Self {
            left,
            right,
            permute: Some(permutation),
            intermediate_bit_size: DEFAULT_INTERMEDIATE_BIT_SIZE,
        }
    }
    pub fn with_max_shapes(self, max_shapes: Vec<Shape>) -> Self {
        self.ensure_shape_consistency(&max_shapes).unwrap();
        let matrix_shape = max_shapes.into_iter().next().unwrap().slice(1..2);
        let intermediate_bit_size = matrix_shape.matmul_output_bitsize();
        Self {
            left: self.left,
            right: self.right,
            permute: None,
            intermediate_bit_size,
        }
    }

    pub fn ensure_shape_consistency<S: Borrow<Shape>>(&self, shapes: &[S]) -> anyhow::Result<()> {
        assert!(shapes.len() == 2, "ConcatMatMul expects 2 inputs");
        ensure!(
            shapes[0].borrow().rank() == shapes[1].borrow().rank(),
            "ConcatMatMul expects input shapes with same rank: {:?} vs {:?}",
            shapes[0].borrow(),
            shapes[1].borrow()
        );
        ensure!(
            shapes[0].borrow().rank() == 3,
            "ConcatMatMul expects inputs of rank 3"
        );
        ensure!(
            shapes[0].borrow().dim(self.left.concat_dimension) == shapes[1].borrow().dim(self.right.concat_dimension),
            "ConcatMatMul expects inputs with same concatenation dimension"
        );
        // check consistency of permuted shapes, if any
        let left_shape = if let Some(permute) = self.left.permute.as_ref() {
            permute.apply(shapes[0].borrow())
        } else {
            shapes[0].borrow().clone()
        };
        let right_shape = if let Some(permute) = self.right.permute.as_ref() {
            permute.apply(shapes[1].borrow())
        } else {
            shapes[1].borrow().clone()
        };
        ensure!(
            left_shape.borrow().dim(2) == right_shape.borrow().dim(1),
            "ConcatMatMul expects submatrices dimensions to match"
        );
        Ok(())
    }
}

impl<N: Number> Evaluate<N> for ConcatMatMul {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 2, "ConcatMatMul expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        let a_shape = a.get_shape();
        let b_shape = b.get_shape();
        self.ensure_shape_consistency(&[&a_shape, &b_shape])?;
        let permuted_a = self.left.permute.as_ref().map(|p| a.permute3d(&p.0)); 
        let permuted_b = self.right.permute.as_ref().map(|p| b.permute3d(&p.0)); 
        let a  = permuted_a.as_ref().unwrap_or(a);
        let b = permuted_b.as_ref().unwrap_or(b);
        let a_shape = a.get_shape();
        let b_shape = b.get_shape();
        ensure!(
            a_shape.dim(0) == b_shape.dim(0),
            "ConcatMatMul expects inputs with same batch size: {} vs {}",
            a_shape.dim(0),
            b_shape.dim(0),
        );
        let results = (0..a_shape.dim(0))
            .map(|batch| {
                let batch_a = a.slice_3d(batch, batch + 1).reshape(a_shape.slice(1..=2));
                let batch_b = b.slice_3d(batch, batch + 1).reshape(b_shape.slice(1..=2));
                batch_a.matmul(&batch_b)
            })
            .collect::<Vec<_>>();
        let mut it = results.into_iter();
        // reshape because concat expects a 3d tensor so he can accumulate in the highest dimension.
        let concat = it
            .next()
            .unwrap()
            .reshape(Shape::new(vec![1, a_shape.dim(1), b_shape.dim(2)]));
        let mut concat = it.fold(concat, |mut acc, x| {
            acc.concat(x);
            acc
        });
        if let Some(ref transpose) = self.permute {
            concat = concat.permute3d(&transpose.0);
        }
        Ok(LayerOut::from_vec(vec![concat]))
    }
}

impl OpInfo for ConcatMatMul {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        let a_shape = &input_shapes[0];
        let b_shape = &input_shapes[1];
        self.ensure_shape_consistency(&[a_shape, b_shape]).unwrap();
        // inner matrix shapes
        let a_shape = if let Some(permute) = self.left.permute.as_ref() {
            permute.apply(&a_shape)
        } else {
            a_shape.clone()
        };
        let b_shape = if let Some(permute) = self.right.permute.as_ref() {
            permute.apply(&b_shape)
        } else {
            b_shape.clone()
        };
        
        let mut mat_result_shape: Shape = vec![a_shape.dim(0), a_shape.dim(1), b_shape.dim(2)].into();
        if let PaddingMode::Padding = padding_mode {
            mat_result_shape = mat_result_shape.next_power_of_two()
        }
        if let Some(ref permute) = self.permute {
            println!(
                "ConcatMatMul: Permute: {:?} over resulting shape {:?}",
                permute, mat_result_shape
            );
            mat_result_shape = mat_result_shape.permute(&permute.0);
        }
        vec![mat_result_shape]
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!("ConcatMatMul: left input {:?}, right_input {:?}, permute output {:?})", 
            self.left,
            self.right,
            self.permute
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for ConcatMatMul {
    type QuantizedOp = ConcatMatMul;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: super::provable::NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<super::provable::QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scale = S::scaling_factors_for_node(data, node_id, num_outputs)[0];
        // normally it's input_scaling * model_scaling / output_scaling, except in this case, we don't have a model_scaling
        // but we have the second matrix scaling, so we use that.
        let input_scale = input_scaling[0];
        let weights_scale = input_scaling[1];
        let intermediate_bit_size = self.intermediate_bit_size;
        let requant = Requant::from_scaling_factors(
            input_scale,
            weights_scale,
            output_scale,
            intermediate_bit_size,
        );
        Ok(super::provable::QuantizeOutput::new(self, vec![output_scale]).with_requant(requant))
    }
}

impl<E: ExtensionField + DeserializeOwned> ProveInfo<E> for ConcatMatMul 
where E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, aux: ContextAux) -> Result<(super::LayerCtx<E>, crate::iop::context::ContextAux)> {
        todo!()
    }
}

impl PadOp for ConcatMatMul {
    
}

impl<E: ExtensionField + DeserializeOwned, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS> for ConcatMatMul 
where E::BaseField: DeserializeOwned + Serialize
{
    type Ctx = ConcatMatMulCtx;

    fn prove<T: Transcript<E>>(
            &self,
            _node_id: NodeId,
            _ctx: &Self::Ctx,
            last_claims: Vec<&Claim<E>>,
            step_data: &StepData<E, E>,
            _prover: &mut Prover<E, T, PCS>,
        ) -> Result<Vec<crate::Claim<E>>> {
        ensure!(
            step_data.inputs.len() == 2,
            "ConcatMatMul expects 2 inputs, got {}",
            step_data.inputs.len()
        );
        let input_shapes = step_data.inputs.iter().map(|input| 
            input.get_shape()
        ).collect_vec();
        self.ensure_shape_consistency(&input_shapes)?;

        let left = step_data.inputs[0].to_mle_2d();
        let right = step_data.inputs[1].to_mle_2d();

        // get the dimension of the left matrix which is not the dimension over which the
        // matrix multiplication is performed on each chunk
        let left_not_mat_mul_dimension = self.left.permute.as_ref().map(|p| 
            // find the dimension that is permuted to end-up as the second dimension in the permuted left input
            p.0.iter().find_position(|x| **x == 1)
                .expect("ConcatMatMul: Permutation muts specify the second dimension")
                .0
        ).unwrap_or(1); // If there is no permutation, the default dimension not involved in matmul is the second one
        // get the dimension of the right matrix which is not the dimension over which the
        // matrix multiplication is performed on each chunk
        let right_not_mat_mul_dimension = self.right.permute.as_ref().map(|p| 
            // find the dimension that is permuted to end-up as the third dimension in the permuted right input
            p.0.iter().find_position(|x| **x == 2)
                .expect("ConcatMatMul: Permutation muts specify the third dimension")
                .0
        ).unwrap_or(2); // If there is no permutation, the default dimension not involved in matmul is the third one

        Ok(vec![last_claims[0].clone()])
    }
}

impl OpInfo for ConcatMatMulCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Vec<Shape> {
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

impl<E: ExtensionField + DeserializeOwned, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for ConcatMatMulCtx 
where E::BaseField: DeserializeOwned
{
    type Proof = ConcatMatMulProof;

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


#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_concat_matmul() {
        let concat_matmul = ConcatMatMul::new(InputMatrix::default(), InputMatrix::default());
        let a = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let b = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        assert_eq!(
            result.outputs[0].data,
            vec![7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0]
        );
    }

    #[test]
    fn test_concat_matmul_with_output_transpose() {
        let concat_matmul = ConcatMatMul::new_with_permute(
            InputMatrix::default(), 
            InputMatrix::default(), 
            Permutation::new(vec![1, 0, 2])
        );
        let a = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let b = Tensor::new(
            vec![2, 2, 2].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        let expected = Tensor::new(
            vec![2, 2, 2].into(),
            vec![7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0],
        );
        let expected = expected.permute3d(&vec![1, 0, 2]);
        assert_eq!(result.outputs[0].data, expected.data);
        let expected_shape =
            concat_matmul.output_shapes(&[a.get_shape(), b.get_shape()], PaddingMode::NoPadding);
        assert_eq!(result.outputs[0].get_shape(), expected_shape[0]);
    }

    #[test]
    fn test_concat_matmul_with_input_transpose() {
        let a = Tensor::new(vec![3, 2, 2].into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let b = Tensor::new(vec![2, 3, 2].into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let permute_a = Permutation::new(vec![1, 0, 2]);
        let permute_b = Permutation::new(vec![0, 2, 1]);
        let concat_matmul = ConcatMatMul::new(
            InputMatrix::new_with_permute(1, permute_a).unwrap(), 
            InputMatrix::new_with_permute(0, permute_b).unwrap(),
        );

        let result = concat_matmul
            .evaluate::<GoldilocksExt2>(&[&a, &b], vec![])
            .unwrap();
        let expected = Tensor::new(
            vec![2, 3, 3].into(),
            vec![
                5.0, 11.0, 17.0, 17.0, 39.0, 61.0, 29.0, 67.0, 105.0,
                53.0, 67.0, 81.0, 113.0, 143.0, 173.0, 173.0, 219.0, 265.0
            ]
        );
        assert_eq!(result.outputs[0].data, expected.data);
        let expected_shape = concat_matmul.output_shapes(
            &[a.get_shape(), b.get_shape()], 
            PaddingMode::NoPadding
        );
        assert_eq!(result.outputs[0].get_shape(), expected_shape[0]);
    }
}
