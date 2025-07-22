use std::{
    collections::{BTreeMap, HashMap},
    iter::once,
};

use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::MultilinearExtension;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        add::{Add, AddCtx, AddProof},
        provable::{
            Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
    },
    model::StepData,
    padding::{PaddingMode, ShapeInfo},
    quantization::TensorFielder,
    tensor::{Number, Shape, TensorSlice},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalCtx {
    add_ctx: AddCtx,
    unpadded_shape: Shape,
    num_vars_positional_matrix: usize,
    node_id: NodeId,
}

/// Data structure containing the proof data for a single
/// input of positional encoding layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SinglePositionalProof<E> {
    // Evaluations of the sub-matrices required to compute the claim
    // about the positional matrix. Each sub-matrix is identified by
    // an incremental integer that corresponds to an extra variable to be processed
    // to get to the number of variables of the positional matrix.
    sub_matrix_evals: Vec<E>,
    // Proofs for addition of the slice of the positional matrix with an
    // input tensor
    add_proof: AddProof<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalProof<E> {
    // Proofs for all the inputs of the positional encoding layer
    proofs: Vec<SinglePositionalProof<E>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Learned<N> {
    positional: Tensor<N>,
    unpadded_shape: Shape,
    add_layer: Add<N>,
}

impl<N> Learned<N> {
    fn num_vars(&self) -> usize {
        let num_vars = self.positional.num_vars_2d();
        num_vars.0 + num_vars.1
    }

    // Sample from the transcript `t` `num_coordinates` random coordinates to be employed
    // for proving. It requires as input also the claim related to the output polynomial and
    // the claim about the sub-matrix of positional matrix being actually added to the
    // input polynomial
    fn sample_random_coordinates<E: ExtensionField, T: Transcript<E>>(
        num_coordinates: usize,
        t: &mut T,
        output_claim: &Claim<E>,
        sub_matrix_claim: &Claim<E>,
    ) -> Vec<E> {
        // first, we add `output_claim` and `sub_pos_claim` to the transcript
        t.append_field_element_exts(&output_claim.point);
        t.append_field_element_ext(&output_claim.eval);
        t.append_field_element_exts(&sub_matrix_claim.point);
        t.append_field_element_ext(&sub_matrix_claim.eval);

        // then, we get `num_coordinates` challenges
        (0..num_coordinates)
            .map(|_| t.read_challenge().elements)
            .collect()
    }

    // Compute a claim about the positional encoding matrix at point `evaluation_point`.
    // The claim is computed from the following evaluations:
    // - `sub_pos_eval`: Evaluation of the sub-matrix that is added to the input tensor
    // - `sub_matrix_evals`: Evaluations of all the sub-matrices employed to derive the claim
    // about the positional encoding matrix
    // Note that the sub-matrices the evaluations are referred to corresponds to a partition
    // of the positional encoding matrix, that is their concatenation yields the positional
    // encoding matrix itself
    fn compute_positional_matrix_claim<E: ExtensionField>(
        evaluation_point: Vec<E>,
        sub_pos_eval: E,
        sub_matrix_evals: &[E],
    ) -> Claim<E> {
        // compute the slice of the evaluation point to be used to compute the claim
        let eval_point = {
            let diff_vars = sub_matrix_evals.len();
            let start_extra_coordinates = evaluation_point.len() - diff_vars;
            &evaluation_point[start_extra_coordinates..]
        };
        let positional_matrix_eval = sub_matrix_evals
            .iter()
            .zip(eval_point)
            .fold(sub_pos_eval, |eval, (&sub_eval, &coordinate)| {
                eval * (E::ONE - coordinate) + sub_eval * coordinate
            });
        Claim::new(evaluation_point, positional_matrix_eval)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Positional<N> {
    Learned(Learned<N>),
    // TODO
    Rope,
}

impl<N: Number> Positional<N> {
    pub fn get_shape(&self) -> Shape {
        match self {
            Self::Learned(pos) => pos.positional.get_shape(),
            Self::Rope => unimplemented!("Rope not implemented"),
        }
    }

    pub fn new_learned(matrix: Tensor<N>) -> Self {
        let unpadded_shape = matrix.get_shape();
        Self::Learned(Learned {
            positional: matrix,
            unpadded_shape,
            add_layer: Add::new(),
        })
    }
}

impl<N: Number> Evaluate<N> for Positional<N>
where
    Add<N>: Evaluate<N>,
{
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(
            inputs.iter().all(|x| x.get_shape().len() == 2),
            "positional embeddings only support 2d tensors"
        );

        let outputs = inputs
            .iter()
            .map(|x| match self {
                Self::Learned(pos) => {
                    let sub_pos = pos.positional.slice_2d(0, x.get_shape()[0]);
                    pos.add_layer
                        .evaluate::<E>(&[x, &sub_pos], vec![pos.unpadded_shape.clone(); 2])?
                        .outputs
                        .pop()
                        .context("Expected at least 1 output from add in positional encoding layer")
                }
                Self::Rope => {
                    anyhow::bail!("Rope not implemented");
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(LayerOut::from_vec(outputs))
    }
}

fn output_shapes(
    input_shapes: &[Shape],
    pos_matrix_unpadded_shape: &Shape,
    padding_mode: PaddingMode,
) -> Vec<Shape> {
    let pos_shape = match padding_mode {
        PaddingMode::NoPadding => pos_matrix_unpadded_shape.clone(),
        PaddingMode::Padding => pos_matrix_unpadded_shape.next_power_of_two(),
    };
    input_shapes.iter().for_each(|s| {
        assert!(s.is_matrix());
        assert!(s[0] <= pos_shape[0]);
        assert_eq!(s[1], pos_shape[1])
    });
    input_shapes.to_vec()
}

impl<N: Number> OpInfo for Positional<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let Self::Learned(pos) = self else {
            unreachable!()
        };
        output_shapes(input_shapes, &pos.unpadded_shape, padding_mode)
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

const POSITIONAL_POLY_ID: &str = "PositionalMatrix";

impl<E: ExtensionField> ProveInfo<E> for Positional<Element> {
    fn step_info(&self, id: NodeId, aux: ContextAux) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        let Self::Learned(pos) = self else {
            unimplemented!("ProveInfo not implemented for Positional::Rope")
        };

        let (ctx, mut aux) = pos.add_layer.step_info(id, aux)?;

        let LayerCtx::<E>::Add(add_ctx) = ctx else {
            unreachable!()
        };

        aux.model_polys = Some(
            aux.model_polys
                .unwrap_or_default()
                .into_iter()
                .chain(once((
                    POSITIONAL_POLY_ID.to_string(),
                    pos.positional.pad_next_power_of_two().data,
                )))
                .collect(),
        );

        let ctx = PositionalCtx {
            add_ctx,
            unpadded_shape: pos.unpadded_shape.clone(),
            num_vars_positional_matrix: pos.num_vars(),
            node_id: id,
        };

        Ok((LayerCtx::Positional(ctx), aux))
    }
}

impl QuantizeOp for Positional<f32> {
    type QuantizedOp = Positional<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "Expected 1 input scaling factor for positional layer, found {}",
            input_scaling.len()
        );

        let Positional::Learned(pos) = self else {
            unimplemented!("Quantization not implemented for Positional::Rope")
        };

        // quantize positional matrix
        let max = pos.positional.max_abs_output();
        let pos_scaling = ScalingFactor::from_absolute_max(max, None);

        let quantized_add =
            pos.add_layer
                .quantize_op::<S>(data, node_id, &[input_scaling[0], pos_scaling])?;

        let quantized_pos = Learned {
            positional: pos.positional.quantize(&pos_scaling),
            unpadded_shape: pos.unpadded_shape,
            add_layer: quantized_add.quantized_op,
        };

        Ok(QuantizeOutput {
            quantized_op: Positional::Learned(quantized_pos),
            output_scalings: quantized_add.output_scalings,
            requant_layer: quantized_add.requant_layer,
        })
    }
}

impl PadOp for Positional<Element> {
    fn pad_node(mut self, _si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        match &mut self {
            Positional::Learned(pos) => pos.positional = pos.positional.pad_next_power_of_two(),
            Positional::Rope => (),
        }

        // no need to change `si` since the layer doesn't change the input shapes

        Ok(self)
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS>
    for Positional<Element>
{
    type Ctx = PositionalCtx;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(
            last_claims.len() == step_data.inputs.len(),
            "Found different number of inputs and outputs when proving positional layer: {} inputs, {} outputs",
            step_data.inputs.len(),
            last_claims.len(),
        );
        let Self::Learned(pos) = self else {
            unimplemented!("Proving not implemented for Positional::Rope")
        };

        let mut output_claims = vec![];
        let mut common_claims = HashMap::new();
        let proofs = last_claims
            .into_iter()
            .zip(&step_data.inputs)
            .map(|(output_claim, input)| {
                // derive sub-matrix to be added to input. ToDo: place it in proving data
                let matrix_slice = TensorSlice::from(&pos.positional);
                let sub_pos = matrix_slice
                    .slice_over_first_dim(0, input.get_shape()[0])
                    .to_fields();

                let (mut claims, add_proof) = pos.add_layer.prove_step(
                    node_id,
                    vec![output_claim],
                    &[input, &sub_pos],
                    prover,
                )?;

                ensure!(
                    claims.len() == 2,
                    "Expected 2 claims from Add proving in position layer, found {} claims",
                    claims.len(),
                );

                let sub_pos_claim = claims.pop().unwrap();
                let input_claim = claims.pop().unwrap();

                output_claims.push(input_claim);

                // we now need to bind the claim about the `sub_pos` tensor with a claim about `positional_matrix`

                // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
                // polynomial
                let num_vars = pos.num_vars();

                let sub_pos_vars = sub_pos_claim.point.len();
                let diff_vars = num_vars - sub_pos_vars;

                ensure!(diff_vars >= 0);

                // now, we need to squeeze `diff_vars` coordinates from the transcript
                let extra_coordinates = Learned::<Element>::sample_random_coordinates(
                    diff_vars,
                    prover.transcript,
                    output_claim,
                    &sub_pos_claim,
                );

                let sub_pos_eval = sub_pos_claim.eval;

                let evaluation_point = sub_pos_claim
                    .point
                    .into_iter()
                    .chain(extra_coordinates)
                    .collect_vec();

                let mut slice_start = input.get_shape()[0];
                let sub_matrices = (0..diff_vars)
                    .map(|_| {
                        let sub_matrix =
                            matrix_slice.slice_over_first_dim(slice_start, slice_start * 2);
                        slice_start *= 2;
                        sub_matrix
                    })
                    .collect_vec();

                // check that all the slices of `positional_matrix` have been computed
                ensure!(slice_start == pos.positional.get_shape()[0]);

                // now, evaluate the MLE of each sub-matrix
                let sub_matrix_evals = (0..diff_vars)
                    .into_par_iter()
                    .map(|i| {
                        (
                            i,
                            sub_matrices[i]
                                .to_fields()
                                .to_mle_2d()
                                .evaluate(&evaluation_point[..sub_pos_vars + i]),
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
                    .into_values()
                    .collect::<Vec<_>>();

                let positional_matrix_claim = Learned::<Element>::compute_positional_matrix_claim(
                    evaluation_point,
                    sub_pos_eval,
                    &sub_matrix_evals,
                );

                common_claims.insert(POSITIONAL_POLY_ID.to_string(), positional_matrix_claim);

                Ok(SinglePositionalProof {
                    sub_matrix_evals,
                    add_proof,
                })
            })
            .collect::<anyhow::Result<_>>()?;

        prover.add_common_claims(node_id, common_claims)?;

        prover.push_proof(node_id, LayerProof::Positional(PositionalProof { proofs }));

        Ok(output_claims)
    }
}

impl OpInfo for PositionalCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        output_shapes(input_shapes, &self.unpadded_shape, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "PositionalCtx:({:?}x{:?})",
            self.unpadded_shape[0], self.unpadded_shape[1]
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS>
    for PositionalCtx
{
    type Proof = PositionalProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let num_outputs = last_claims.len();

        ensure!(
            shape_step.unpadded_output_shape.len() == num_outputs,
            "Expected {num_outputs} unpadded output shapes for Positional layer verifier, found {}",
            shape_step.unpadded_output_shape.len()
        );

        ensure!(
            shape_step.padded_output_shape.len() == num_outputs,
            "Expected {num_outputs} padded output shapes for Positional layer verifier, found {}",
            shape_step.padded_output_shape.len()
        );

        ensure!(
            shape_step.unpadded_input_shape.len() == num_outputs,
            "Expected {num_outputs} unpadded input shapes for Positional layer verifier, found {}",
            shape_step.unpadded_input_shape.len()
        );

        ensure!(
            shape_step.padded_input_shape.len() == num_outputs,
            "Expected {num_outputs} padded input shapes for Positional layer verifier, found {}",
            shape_step.padded_input_shape.len()
        );

        ensure!(
            proof.proofs.len() == num_outputs,
            "Expected {num_outputs} input proofs for Positional layer verifier, found {}",
            proof.proofs.len(),
        );

        let mut output_claims = vec![];
        let mut common_claims = HashMap::new();
        for (i, (output_claim, proof)) in last_claims.iter().zip(&proof.proofs).enumerate() {
            // compute shape step for add sub-layer
            let unpadded_input_shapes = vec![shape_step.unpadded_input_shape[i].clone(); 2];
            let padded_input_shapes = vec![shape_step.padded_input_shape[i].clone(); 2];
            let shape_step = LayerCtx::<E>::Add(self.add_ctx.clone())
                .shape_step(&unpadded_input_shapes, &padded_input_shapes);

            let mut claims =
                self.add_ctx
                    .verify(&proof.add_proof, &[output_claim], verifier, &shape_step)?;

            ensure!(
                claims.len() == 2,
                "Expected 2 claims from Add verifier in position layer, found {} claims",
                claims.len(),
            );

            let sub_pos_claim = claims.pop().unwrap();

            let input_claim = claims.pop().unwrap();

            output_claims.push(input_claim);

            // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
            // polynomial
            let sub_pos_vars = sub_pos_claim.point.len();
            let diff_vars = self.num_vars_positional_matrix - sub_pos_vars;

            ensure!(diff_vars >= 0);

            // then, we sample the extra coordinates corresponding to these variables
            let extra_coordinates = Learned::<Element>::sample_random_coordinates(
                diff_vars,
                verifier.transcript,
                output_claim,
                &sub_pos_claim,
            );

            let sub_pos_eval = sub_pos_claim.eval;

            let evaluation_point = sub_pos_claim
                .point
                .into_iter()
                .chain(extra_coordinates)
                .collect_vec();

            let positional_matrix_claim = Learned::<Element>::compute_positional_matrix_claim(
                evaluation_point,
                sub_pos_eval,
                &proof.sub_matrix_evals,
            );

            common_claims.insert(POSITIONAL_POLY_ID.to_string(), positional_matrix_claim);
        }

        verifier.add_common_claims(self.node_id, common_claims)?;

        Ok(output_claims)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Element, Tensor,
        layers::{Layer, provable::PadOp, transformer::positional::Positional},
        model::{Model, test::prove_model},
        padding::{PaddingMode, ShapeData, ShapeInfo},
    };
    use rstest::rstest;

    #[test]
    fn test_positional_padding() {
        let seq_len = 14;
        let context_length = 17;
        let embedding_size = 45;
        let input_shape = vec![seq_len, embedding_size];
        let matrix_shape = vec![context_length, embedding_size];
        let positional_matrix = Tensor::<Element>::random(&matrix_shape.into());
        let pos = Layer::Positional(Positional::new_learned(positional_matrix.clone()));

        let mut si = ShapeInfo::from(vec![ShapeData::new(input_shape.into())].as_slice());

        let padded_layer = pos.pad_node(&mut si).unwrap();

        let Layer::Positional(Positional::Learned(padded_pos)) = padded_layer else {
            unreachable!()
        };

        let padded_shape = padded_pos.positional.get_shape();
        assert_eq!(padded_pos.unpadded_shape, positional_matrix.get_shape());
        assert_eq!(
            padded_shape,
            positional_matrix.get_shape().next_power_of_two()
        );

        // check that padded positional matrix has the same data of original matrix
        for i in 0..padded_shape[0] {
            for j in 0..padded_shape[1] {
                if i < padded_pos.unpadded_shape[0] && j < padded_pos.unpadded_shape[1] {
                    assert_eq!(
                        padded_pos.positional.get_2d(i, j),
                        positional_matrix.get_2d(i, j)
                    );
                } else {
                    assert_eq!(padded_pos.positional.get_2d(i, j), 0);
                }
            }
        }
    }

    #[rstest]
    #[case::less_input_than_context_length(14, 17, 31)]
    #[case::same_input_as_context_length(31, 17, 31)]
    fn test_proven_positional_layer(
        #[case] seq_len: usize,
        #[case] embedding_size: usize,
        #[case] context_length: usize,
    ) {
        let input_shape = vec![seq_len, embedding_size];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        // build positional matrix
        let matrix_shape = vec![context_length, embedding_size];
        let positional_matrix = Tensor::random(&matrix_shape.into());

        let _ = model
            .add_consecutive_layer(
                Layer::Positional(Positional::new_learned(positional_matrix)),
                None,
            )
            .unwrap();

        model.route_output(None).unwrap();

        let _ = prove_model(model).unwrap();
    }
}
