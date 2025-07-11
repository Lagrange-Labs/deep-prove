use std::collections::BTreeMap;

use anyhow::ensure;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::MultilinearExtension;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{
    iop::context::ContextAux, layers::{add::Add, provable::{Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx}, LayerCtx}, model::StepData, padding::PaddingMode, quantization::TensorFielder, tensor::{Number, Shape, TensorSlice}, Claim, Element, Prover, Tensor
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalCtx {

}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalProof {

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPositional<N> {
    positional_matrix: Tensor<N>,
    unpadded_shape: Shape,
    add_layer: Add<N>, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Positional<N> {
    Learned(LearnedPositional<N>),
    // TODO
    Rope,
}

impl<N: Number> Positional<N> {
    pub fn get_shape(&self) -> Shape {
        match self {
            Self::Learned(pos) => pos.positional_matrix.get_shape(),
            Self::Rope => unimplemented!("Rope not implemented"),
        }
    }

    pub fn new_learned(matrix: Tensor<N>) -> Self {
        let unpadded_shape = matrix.get_shape();
        Self::Learned(LearnedPositional {
            positional_matrix: matrix,
            unpadded_shape,
            add_layer: Add::new(),
        })
    }
}

impl<N: Number> Evaluate<N> for Positional<N> 
where Add<N>: Evaluate<N>
{
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> 
    {
        ensure!(
            inputs.iter().all(|x| x.get_shape().len() == 2),
            "positional embeddings only support 2d tensors"
        );

        let outputs = inputs
            .iter()
            .map(|x| {
                match self {
                    Self::Learned(pos) => {
                        let sub_pos = pos.positional_matrix.slice_2d(0, x.get_shape()[0]);
                        Ok(pos.add_layer.evaluate::<E>(
                            &[x, &sub_pos], 
                            vec![pos.unpadded_shape.clone(); 2],
                        )?
                            .outputs.pop().unwrap())
                    }
                    Self::Rope => {
                        anyhow::bail!("Rope not implemented");
                    }
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(LayerOut::from_vec(outputs))
    }
}

impl<N: Number> OpInfo for Positional<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let Self::Learned(pos) = self else {unreachable!()};
        let pos_shape = match padding_mode {
            PaddingMode::NoPadding => pos.unpadded_shape.clone(),
            PaddingMode::Padding => pos.unpadded_shape.next_power_of_two(),
        };
        input_shapes.into_iter().for_each(|s| 
            assert_eq!(s, &pos_shape)
        );
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "Positional({:?}x{:?})",
            self.get_shape()[0],
            self.get_shape()[1]
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E: ExtensionField> ProveInfo<E> for Positional<Element> {
    fn step_info(&self, 
        id: NodeId, 
        aux: ContextAux
    ) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        todo!()
    }
}

impl PadOp for Positional<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS> for Positional<Element> {
    type Ctx = PositionalCtx;

    fn prove<T: Transcript<E>>(
            &self,
            node_id: NodeId,
            _ctx: &Self::Ctx,
            last_claims: Vec<&Claim<E>>,
            step_data: &StepData<E, E>,
            prover: &mut Prover<E, T, PCS>,
        ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(last_claims.len() == step_data.inputs.len(), 
            "Found different number of inputs and outputs when proving positional layer: {} inputs, {} outputs",
            step_data.inputs.len(),
            last_claims.len(),
        );
        let Self::Learned(pos) = self else {
            unimplemented!("Proving not implemented for Positional::Rope")
        };

        let mut output_claims = vec![];
        for (output_claim, input) in last_claims.into_iter().zip(&step_data.inputs) {
            // derive sub-matrix to be added to input. ToDo: place it in proving data
            let matrix_slice = TensorSlice::from(&pos.positional_matrix);
            let sub_pos = matrix_slice
                .slice_over_first_dim(0, input.get_shape()[0])
                .to_fields()
            ;

            let (mut claims, add_proof) = pos.add_layer.prove_step(
                node_id,
                vec![output_claim],
                &[input, &sub_pos],
                prover,
            )?;

            ensure!(claims.len() == 2, "Expected 2 claims from Add proving in position layer, found {} claims",
                claims.len(),
            );

            let sub_pos_claim = claims.pop().unwrap();
            let input_claim = claims.pop().unwrap();
                      
            output_claims.push(input_claim);

            // we now need to bind the claim about the `sub_pos` tensor with a claim about `positional_matrix`
            
            // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
            // polynomial
            let num_vars = pos.positional_matrix.num_vars_2d();
            let num_vars = num_vars.0 + num_vars.1;

            let sub_pos_vars = sub_pos_claim.point.len();
            let diff_vars = num_vars - sub_pos_vars;

            ensure!(diff_vars >= 0);

            // now, we need to squeeze `diff_vars` coordinates from the transcript
            // first, we add `output_claim` and `sub_pos_claim` to the transcript
            prover.transcript.append_field_element_exts(&output_claim.point);
            prover.transcript.append_field_element_ext(&output_claim.eval);
            prover.transcript.append_field_element_exts(&sub_pos_claim.point);
            prover.transcript.append_field_element_ext(&sub_pos_claim.eval);

            // then, we get `diff_vars` challenges
            let extra_coordinates = (0..diff_vars).map(|_| 
                prover.transcript.read_challenge().elements
            ).collect_vec();

            let sub_pos_eval = sub_pos_claim.eval;

            let evaluation_point = sub_pos_claim.point.into_iter().chain(extra_coordinates).collect_vec();

            let mut slice_start = input.get_shape()[0];
            let sub_matrices = (0..diff_vars-1).map(|_| {
                let sub_matrix = matrix_slice.slice_over_first_dim(slice_start, slice_start*2);
                slice_start *= 2;
                sub_matrix
            }).collect_vec();

            // check that all the slices of `positional_matrix` have been computed
            ensure!(slice_start == pos.positional_matrix.get_shape()[0]);

            // now, evaluate the MLE of each sub-matrix
            let sub_matrix_evals = (0..diff_vars-1).into_par_iter().map(|i| {
                (i, sub_matrices[i].to_fields().to_mle_2d().evaluate(&evaluation_point[..sub_pos_vars+i]))
            }).collect::<BTreeMap<_, _>>();

        }

        Ok(output_claims)
    }
}

impl OpInfo for PositionalCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
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

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for PositionalCtx {
    type Proof = PositionalProof;

    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&crate::Claim<E>],
        verifier: &mut crate::iop::verifier::Verifier<E, T, PCS>,
        shape_step: &crate::iop::context::ShapeStep,
    ) -> anyhow::Result<Vec<crate::Claim<E>>> {
        todo!()
    }
}