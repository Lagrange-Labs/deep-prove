use crate::{
    Claim, Context, Element, Prover, ScalingFactor, ScalingStrategy, argmax_slice,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            NodeId, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
    },
    lookup::{
        context::{LookupWitnessGen, TableType},
        logup_gkr::{prover::batch_prove, structs::LogUpProof, verifier::verify_logup_proof},
        witness::LogUpWitness,
    },
    model::StepData,
    padding::{PaddingMode, ShapeData, ShapeInfo},
    quantization::{Fieldizer, IntoElement, TensorFielder},
    tensor::Shape,
};
use anyhow::{anyhow, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::MultilinearExtension;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo},
    tensor::Number,
};
#[derive(Clone, Debug)]
pub struct ArgmaxData<E> {
    max_values: Vec<Tensor<E>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogitsCtx {}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LogitsProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    logup_proof: LogUpProof<E>,
    max_eval: E,
    max_commitment: PCS::Commitment,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LogitsProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        self.logup_proof.fractional_outputs()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Logits {
    Argmax,
}

impl Logits {
    fn evaluate_with_argmax_data<N: Number, E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<(LayerOut<N, E>, ArgmaxData<N>)> {
        ensure!(
            inputs.iter().all(|i| i.get_shape().len() >= 2),
            "Argmax is for tensors of rank >= 2"
        );

        match self {
            Logits::Argmax => {
                let (indices, maximums): (Vec<_>, Vec<_>) = inputs
                    .iter()
                    .map(|input| {
                        let (indices, maximums): (Vec<_>, Vec<_>) = input
                            .slice_last_dim()
                            .map(|row| {
                                let argmax = argmax_slice(row).unwrap();
                                let max = row[argmax];
                                (N::from_usize(argmax), max)
                            })
                            .unzip();
                        let indices = Tensor::new(Shape::new(vec![indices.len(), 1]), indices);
                        let maximums = Tensor::new(Shape::new(vec![maximums.len(), 1]), maximums);
                        (indices, maximums)
                    })
                    .unzip();
                Ok((
                    LayerOut::from_vec(indices),
                    ArgmaxData {
                        max_values: maximums,
                    },
                ))
            }
        }
    }

    fn output_shapes(input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .into_iter()
            .map(|s| vec![s.dim(0), 1].into())
            .collect()
    }

    fn split_claim_point<E: ExtensionField>(
        point: &[E],
        num_row_vars: usize,
    ) -> anyhow::Result<(&[E], &[E])> {
        // row variables are the most significant ones, so we splice between the last `num_row_vars` coordinates
        // and the other ones
        let split_item = point.len() - num_row_vars;
        let row_point = &point[split_item..];
        Ok((&point[..split_item], row_point))
    }
}

impl Evaluate<f32> for Logits {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let (output, _) = self.evaluate_with_argmax_data(inputs, unpadded_input_shapes)?;

        Ok(output)
    }
}

impl Evaluate<Element> for Logits {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<Element, E>> {
        let (output, argmax_data) =
            self.evaluate_with_argmax_data(inputs, unpadded_input_shapes)?;

        // convert argmax_data to field elements
        let argmax_data = ArgmaxData {
            max_values: argmax_data
                .max_values
                .into_iter()
                .map(|m| m.to_fields())
                .collect(),
        };

        Ok(output.with_proving_data(ProvingData::ArgMax(argmax_data)))
    }
}

impl OpInfo for Logits {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        Self::output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Logits::Argmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E: ExtensionField> ProveInfo<E> for Logits {
    fn step_info(
        &self,
        _id: NodeId,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);
        aux.tables.insert(TableType::Range);

        Ok((LayerCtx::Logits(LogitsCtx {}), aux))
    }
}

impl QuantizeOp for Logits {
    type QuantizedOp = Logits;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        // no need to quantize, we just propagate the scaling factors
        let num_inputs = input_scaling.len();
        let num_outputs = self.num_outputs(num_inputs);
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);

        Ok(QuantizeOutput::new(self, output_scalings))
    }
}

impl PadOp for Logits {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let unpadded_input_shapes = si.unpadded_input_shapes();
        let unpadded_output_shapes =
            self.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);

        let padded_input_shapes = si.padded_input_shapes();
        let padded_output_shapes = self.output_shapes(&padded_input_shapes, PaddingMode::Padding);

        ensure!(
            si.shapes.iter().all(|s| s.ignore_garbage_pad.is_none()),
            "Unexpected garbage padding to be removed in Logits layer"
        );

        si.shapes = unpadded_output_shapes
            .into_iter()
            .zip(padded_output_shapes)
            .map(|(unpadded_s, padded_s)| ShapeData {
                input_shape_padded: padded_s,
                ignore_garbage_pad: None,
                input_shape_og: unpadded_s,
            })
            .collect();

        Ok(self)
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProvableOp<E, PCS> for Logits {
    type Ctx = LogitsCtx;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        _last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(
            step_data.inputs.len() == 1,
            "Expected 1 input tensor for Logits layer, found {}",
            step_data.inputs.len()
        );
        let input = &step_data.inputs[0];

        let argmax_data = step_data
            .outputs
            .try_argmax_data()
            .ok_or(anyhow!("Argmax data not found when proving Logits layer"))?;

        ensure!(
            argmax_data.max_values.len() == 1,
            "Expected 1 tensor of max values in argmax data when proving Logits layer, found {}",
            argmax_data.max_values.len(),
        );

        let max_values = &argmax_data.max_values[0];

        let logup_witnesses = prover.lookup_witness(node_id)?;

        ensure!(
            logup_witnesses.len() == 1,
            "Expected 1 logup witness for Logits layer, found {}",
            logup_witnesses.len(),
        );

        let logup_witness = &logup_witnesses[0];

        let logup_input = logup_witness.get_logup_input(&prover.challenge_storage)?;

        let mut commits = logup_witness.get_commitments();

        ensure!(
            commits.len() == 1,
            "Expected 1 commitment in logup witness for Logits layer, found {}",
            commits.len(),
        );

        let (max_values_commit, max_mle) = commits.remove(0);

        let logup_proof = batch_prove(&logup_input, prover.transcript)?;

        // get the claim about the difference between max_values and input data
        let output_claims = logup_proof.output_claims();
        ensure!(
            output_claims.len() == 1,
            "Expected 1 claim from logup proof in Logits layer, found {}",
            output_claims.len(),
        );
        let diff_claim = &output_claims[0];

        // evaluate max_values MLE on the same point of `diff_claim`
        // we need to extract the row-related coordinates from `diff_claim.point`
        let (_, row_point) = Self::split_claim_point(&diff_claim.point, max_mle.num_vars())?;

        let max_eval = max_mle.evaluate(row_point);

        let max_commitment = PCS::get_pure_commitment(&max_values_commit);

        prover.commit_prover.add_witness_claim(
            (max_values_commit, max_mle),
            Claim::new(row_point.to_vec(), max_eval),
        )?;

        let input_claim = Claim::new(diff_claim.point.clone(), max_eval - diff_claim.eval);

        let proof = LogitsProof {
            logup_proof,
            max_eval,
            max_commitment,
        };

        prover.push_proof(node_id, LayerProof::Logits(proof));

        Ok(vec![input_claim])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &Context<E, PCS>,
        step_data: &StepData<Element, E>,
    ) -> anyhow::Result<LookupWitnessGen<E, PCS>> {
        ensure!(
            step_data.inputs.len() == 1,
            "Expected 1 input tensor for Logits witness generation, found {}",
            step_data.inputs.len()
        );

        let input = &step_data.inputs[0];

        ensure!(
            matches!(self, Logits::Argmax),
            "Only Argmax is currently supported in Logits layer"
        );

        let argmax_data = step_data.outputs.try_argmax_data().ok_or(anyhow!(
            "Argmax data not found when generating witness for Logits layer"
        ))?;

        ensure!(
            argmax_data.max_values.len() == 1,
            "Expected 1 tensor of max values for Logits argmax, found {}",
            argmax_data.max_values.len(),
        );

        let max_values = &argmax_data.max_values[0];

        let input_shape = input.get_shape();
        ensure!(
            max_values.get_shape().dim(0) == input.get_shape().dim(0),
            "Incompatible shapes between max values tensor and input tensor: {:?} vs {:?}",
            max_values.get_shape(),
            input.get_shape(),
        );

        let (merged_diff, diff_values): (Vec<Element>, Vec<E::BaseField>) = input
            .get_data()
            .into_par_iter()
            .enumerate()
            .map(|(i, input)| {
                let row_index = i / input_shape.dim(input_shape.len() - 1);
                let current_max = max_values.get_data()[row_index];

                let max_element = current_max.to_element();
                let diff = max_element - input;
                (
                    diff,
                    <Element as Fieldizer<E>>::to_field(&diff)
                        .as_base()
                        .expect("Diff element overflows field"),
                )
            })
            .unzip();

        // commit to max values
        let commits = {
            let max_mle = max_values.to_mle_2d();
            (ctx.commitment_ctx.commit(&max_mle)?, max_mle)
        };
        let mut gen = LookupWitnessGen::<E, PCS>::default();
        gen.logup_witnesses.insert(
            id,
            vec![LogUpWitness::new_lookup(
                vec![commits],
                vec![diff_values],
                1,
                TableType::Range,
            )],
        );
        gen.new_lookups.insert(TableType::Range,merged_diff);

        Ok(gen)
    }
}

impl OpInfo for LogitsCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        Logits::output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Logit::Argmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for LogitsCtx {
    type Proof = LogitsProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        _last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .get_challenges_by_name(&TableType::Range.name())
            .ok_or(anyhow!(
                "Couldn't get challenges for LookupType: {}",
                TableType::Range.name()
            ))?;
        let logup_claims = verify_logup_proof(
            &proof.logup_proof,
            1,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;

        ensure!(
            logup_claims.claims().len() == 1,
            "Expected 1 claim for logup when verifying Logis layer, found {}",
            logup_claims.claims().len(),
        );

        let claim = &logup_claims.claims()[0];
        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "Expected 1 padded input shape when verifying Logits layer, found {}",
            shape_step.padded_input_shape.len(),
        );

        let num_row_vars = shape_step.padded_input_shape[0].dim(0).ilog2() as usize;
        let (_, row_point) = Logits::split_claim_point(&claim.point, num_row_vars)?;

        let input_claim = Claim::new(claim.point.clone(), proof.max_eval - claim.eval);

        verifier.commit_verifier.add_witness_claim(
            proof.max_commitment.clone(),
            Claim::new(row_point.to_vec(), proof.max_eval),
        )?;

        Ok(vec![input_claim])
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use super::*;
    use crate::{
        layers::{Layer, provable::Evaluate},
        model::{Model, test::prove_model},
        tensor::Tensor,
    };

    #[test]
    fn test_logits_argmax() -> anyhow::Result<()> {
        let input = Tensor::new(vec![3, 2].into(), vec![0.0, 1.0, 3.0, 2.0, 4.0, 5.0]);
        let logits = Logits::Argmax;
        let out = logits.evaluate::<GoldilocksExt2>(&[&input], vec![])?;
        // first slice is [0,1] so argmax here is 1
        // second slice is [3,2] so argmax here is 0
        // the last dimension is [4,5] so argmax here is 1
        assert_eq!(out.outputs()[0].get_data(), vec![1.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_proven_logits_argmax() {
        let seq_len = 13;
        let vocab_size = 17;
        let input_shape = Shape::new(vec![seq_len, vocab_size]);
        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::Logits(Logits::Argmax), None)
            .unwrap();

        model.route_output(None).unwrap();

        prove_model(model).unwrap();
    }
}
