use crate::{
    ScalingFactor, ScalingStrategy,
    commit::compute_betas_eval,
    layers::{
        LayerProof,
        provable::{QuantizeOp, QuantizeOutput},
    },
};

use anyhow::{Context, bail, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx,
        matrix_mul::{MatMul, MatMulCtx, MatMulProof, OperandMatrix},
        provable::{
            Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx,
        },
    },
    model::StepData,
    padding::{PaddingMode, ShapeInfo},
    tensor::{Number, Shape},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embeddings<N> {
    mat: MatMul<N>,
    emb_size: usize,
    pub(crate) vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsCtx<E> {
    vocab_size: usize,
    mat_ctx: MatMulCtx<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct EmbeddingsProof<E: ExtensionField> {
    mat_proof: MatMulProof<E>,
}

impl<N: Number> Embeddings<N> {
    pub fn new(emb: Tensor<N>) -> anyhow::Result<Self> {
        let emb_size = emb.get_shape()[1];
        let vocab_size = emb.get_shape()[0];
        // left side is one hot input tensor, and right side
        // is the embedding matrix
        let left = OperandMatrix::Input;
        let right = OperandMatrix::new_weight_matrix(emb);
        let matmul = MatMul::new(left, right)?;
        Ok(Self {
            mat: matmul,
            emb_size,
            vocab_size,
        })
    }
}

impl<N: Number> OpInfo for Embeddings<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        assert!(
            input_shapes.len() == 1,
            "embeddings only support 1 input tensor"
        );
        assert_eq!(
            input_shapes[0].rank(),
            1,
            "embeddings only support 1d tensors"
        );
        let seq_len = input_shapes[0].dim(0);
        let shape = match padding_mode {
            PaddingMode::NoPadding => Shape::new(vec![seq_len, self.emb_size].into()),
            PaddingMode::Padding => Shape::new(vec![
                seq_len.next_power_of_two(),
                self.emb_size.next_power_of_two(),
            ])
            .next_power_of_two(),
        };
        vec![shape]
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "Embeddings(vocab:{:?}, hidden:{:?})",
            self.vocab_size, self.emb_size
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: Number> Evaluate<N> for Embeddings<N> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(
            inputs.iter().all(|x| {
                let shape: Shape = x.get_shape();
                shape.rank() == 1
            }),
            "embeddings only support 2d tensors with 1 value: {:?}",
            inputs.iter().map(|x| x.get_shape()).collect::<Vec<_>>()
        );
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        // we still uses this evaluation for inference as it's quicker
        // than doing the matmul with one hot encoding. Proving however will generate
        // the one hot encoding and do the matmul.
        let OperandMatrix::Weight(ref w) = self.mat.right_matrix else {
            bail!("right matrix is not a weight matrix");
        };
        let emb = &w.tensor;
        let x = inputs[0];
        let seq_len = x.get_shape()[0];
        let vocab_size = emb.get_shape()[0];
        let emb_size = emb.get_shape()[1];
        let emb_data = emb.get_data();
        let emb = x
            .get_data()
            .iter()
            .flat_map(|v| {
                let idx = v.to_usize();
                assert!(
                    idx < vocab_size,
                    "idx {idx} out of bounds for vocab size {vocab_size}"
                );
                let emd_idx = idx * emb_size;
                emb_data[emd_idx..emd_idx + emb_size].to_vec()
            })
            .collect::<Vec<_>>();
        let out_shape = Shape::new(vec![seq_len, emb_size]);
        Ok(LayerOut::from_vec(vec![Tensor::new(out_shape, emb)]))
    }
}

impl PadOp for Embeddings<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        ensure!(
            si.shapes.len() == 1,
            "embeddings only support 1 input tensor"
        );
        // we need to give the shapes that the one hot encoding will have
        let mut shape_data = si.shapes.remove(0);
        ensure!(
            shape_data.input_shape_og.rank() == 1,
            "embeddings only support 1d tensors"
        );
        shape_data.input_shape_og = one_hot_shape(
            &shape_data.input_shape_og,
            self.vocab_size,
            PaddingMode::NoPadding,
        );
        shape_data.input_shape_padded = one_hot_shape(
            &shape_data.input_shape_padded,
            self.vocab_size,
            PaddingMode::Padding,
        );
        si.shapes.push(shape_data);
        let r = self.mat.pad_node(si).map(|mat| Self { mat, ..self })?;
        Ok(r)
    }
}

impl<E> ProveInfo<E> for Embeddings<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn step_info(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        // we need to give the shapes that the one hot encoding will have
        let shape = aux.last_output_shape.remove(0);
        aux.last_output_shape
            .push(one_hot_shape(&shape, self.vocab_size, PaddingMode::Padding));
        let mat_ctx = self.mat.ctx(id, &mut aux).context("embeddings matmul: ")?;
        Ok((
            LayerCtx::Embeddings(EmbeddingsCtx {
                mat_ctx,
                vocab_size: self.vocab_size,
            }),
            aux,
        ))
    }
}

impl<E> OpInfo for EmbeddingsCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        assert!(
            input_shapes.len() == 1,
            "embeddings only support 1 input tensor"
        );
        assert_eq!(
            input_shapes[0].rank(),
            1,
            "embeddings only support 1d tensors"
        );
        // we need to give the shapes that the one hot encoding will have
        let onehot_shape = one_hot_shape(&input_shapes[0], self.vocab_size, padding_mode);
        self.mat_ctx.output_shapes(&[onehot_shape], padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        self.mat_ctx.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.mat_ctx.describe()
    }

    fn is_provable(&self) -> bool {
        self.mat_ctx.is_provable()
    }
}

impl QuantizeOp for Embeddings<f32> {
    type QuantizedOp = Embeddings<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let quantized_mat = self.mat.quantize_op::<S>(data, node_id, input_scaling)?;
        let qmatmul = quantized_mat.quantized_op;
        let OperandMatrix::Weight(w) = qmatmul.right_matrix else {
            bail!("right matrix is not a weight matrix");
        };
        let qemb = Embeddings::new(w.tensor)?;
        Ok(QuantizeOutput::new(qemb, quantized_mat.output_scalings))
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Embeddings<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = EmbeddingsCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        // we first construct the one hot encoding from the input indices and then we run
        // the matmul protocol.
        ensure!(
            step_data.inputs.len() == 1,
            "embeddings only support 1 input tensor"
        );
        ensure!(
            last_claims.len() == 1,
            "embeddings only support 1 last claim"
        );
        let last_claim = last_claims[0];
        let one_hot = one_hot_encoding(
            step_data.inputs[0].get_data(),
            self.vocab_size,
            PaddingMode::Padding,
        );
        let (output_claims, mat_proof) = self.mat.prove_step(
            node_id,
            prover,
            last_claim,
            vec![&one_hot],
            step_data.outputs.outputs()[0],
        )?;
        prover.push_proof(
            node_id,
            LayerProof::Embeddings(EmbeddingsProof { mat_proof }),
        );
        Ok(output_claims)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for EmbeddingsCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = EmbeddingsProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(
            last_claims.len() == 1,
            "embeddings only support 1 last claim"
        );
        // we verify the matmul proof first
        let mut claims = self
            .mat_ctx
            .verify_matmul(verifier, last_claims[0], &proof.mat_proof)?;
        ensure!(claims.len() == 1, "embeddings matmul should have 1 claim");
        // the first claim is the one hot encoding claim. To verify it we need to
        // efficiently evaluate the one hot encoding on it - we do this "at the end" of the verification
        // procedure to respect the framework's order of operations. The logic is in `verify_input_claim`.
        Ok(vec![claims.remove(0)])
    }

    fn verify_input_claim(
        &self,
        inputs: &Vec<Tensor<E>>,
        claims: &Vec<&Claim<E>>,
    ) -> anyhow::Result<()> {
        // TODO verify efficiently the one hot encoding claim
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        ensure!(claims.len() == 1, "embeddings only support 1 claim");
        let input = &inputs[0];
        let one_hot_claim = &claims[0];
        let vocab_nv = self.vocab_size.next_power_of_two().ilog2();
        let seq_len_nv = input.get_shape().dim(0).next_power_of_two().ilog2();
        ensure!(
            vocab_nv + seq_len_nv == one_hot_claim.point.len() as u32,
            "vocab_nv: {vocab_nv}, seq_len_nv: {seq_len_nv}, one_hot_claim.point.len(): {}",
            one_hot_claim.point.len()
        );
        let (r1, r2) = one_hot_claim.point.split_at(seq_len_nv as usize);
        let b1 = compute_betas_eval(r1);
        let b2 = compute_betas_eval(r2);
        let mut sum = E::ZERO;
        for (idx, token) in input.get_data().iter().enumerate() {
            let token_value = token.to_canonical_u64_vec()[0] as usize;
            let selector = b1[idx] * b2[token_value];
            sum += selector;
        }
        ensure!(
            sum == one_hot_claim.eval,
            "one hot encoding claim is incorrect"
        );
        Ok(())
    }
}

fn one_hot_encoding<E: ExtensionField>(indices: &[E], vb: usize, mode: PaddingMode) -> Tensor<E> {
    let mut data = Vec::new();
    let vocab_size = match mode {
        PaddingMode::NoPadding => vb,
        PaddingMode::Padding => vb.next_power_of_two(),
    };
    for idx in indices {
        let mut one_hot = vec![E::ZERO; vb];
        let idx: usize = idx.to_canonical_u64_vec()[0].try_into().unwrap();
        one_hot[idx] = E::ONE;
        data.extend_from_slice(&one_hot);
    }
    let data = match mode {
        PaddingMode::NoPadding => data,
        PaddingMode::Padding => {
            assert!(
                indices.len().is_power_of_two(),
                "indices length must be a power of two"
            );
            let target_len = indices.len() * vocab_size;
            let curr_len = data.len();
            data.into_iter()
                .chain(std::iter::repeat_n(E::ZERO, target_len - curr_len))
                .collect()
        }
    };
    Tensor::new(vec![indices.len(), vocab_size].into(), data)
}

fn one_hot_shape(input_shape: &Shape, vocab_size: usize, mode: PaddingMode) -> Shape {
    match mode {
        PaddingMode::NoPadding => input_shape.insert(1, vocab_size),
        PaddingMode::Padding => input_shape.insert(1, vocab_size.next_power_of_two()),
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{Rng, thread_rng};
    use ff_ext::GoldilocksExt2;
    use p3_field::FieldAlgebra;

    use crate::{
        Element,
        layers::Layer,
        model::{Model, test::prove_model_with},
        quantization::TensorFielder,
    };

    use super::*;

    fn generate_unique_random_indices(seq_len: usize, vocab_size: usize) -> Vec<usize> {
        let mut ctr = 0;
        while ctr < 10 {
            let d = (0..seq_len)
                .map(|_| thread_rng().gen_range(0..vocab_size))
                .collect::<Vec<_>>();
            let mut dd = d.clone();
            dd.sort();
            dd.dedup();
            if dd.len() == seq_len {
                return d;
            }
            ctr += 1;
        }
        panic!("failed to generate unique random indices");
    }

    #[test]
    fn test_one_hot_encoding_proving() -> anyhow::Result<()> {
        let seq_len: usize = 5;
        let vocab_size: usize = 200;
        let emb_size: usize = 10;
        let indices = (0..seq_len)
            .map(|_| thread_rng().gen_range(0..vocab_size) as Element)
            .collect::<Vec<_>>();
        let input_shape = Shape::from(vec![seq_len]);
        let input = Tensor::new(input_shape.clone(), indices.clone());
        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

        let embeddings_value = Tensor::random(&Shape::new(vec![vocab_size, emb_size].into()));
        let embeddings = Embeddings::new(embeddings_value.clone())?;
        let _ = model
            .add_consecutive_layer(Layer::Embeddings(embeddings), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        prove_model_with(model, Some(vec![input]))
    }

    #[test]
    fn test_one_hot_encoding_inference() -> anyhow::Result<()> {
        let seq_len: usize = 5;
        let indices_elem: Vec<Element> = (0..seq_len).map(|i| i as Element).collect::<Vec<_>>();
        let indices: Tensor<GoldilocksExt2> =
            Tensor::<Element>::new(vec![5].into(), indices_elem.clone()).to_fields();
        let vocab_size = 5;
        let emb_size = 10;
        let one_hot = one_hot_encoding(&indices.get_data(), vocab_size, PaddingMode::NoPadding);
        let expected_shape: Shape = vec![indices.get_shape().numel(), vocab_size].into();
        assert_eq!(one_hot.get_shape(), expected_shape);
        assert_eq!(
            one_hot.get_data(),
            vec![
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE
            ]
        );

        let emb = Tensor::<Element>::random(&vec![5, 10].into());
        let embeddings = Embeddings::new(emb.clone())?;
        let input = Tensor::new(vec![vocab_size].into(), indices_elem.clone());
        let out = embeddings
            .evaluate::<GoldilocksExt2>(&[&input], vec![vec![indices_elem.len(), 1].into()])?;
        let expected_shape = Shape::new(vec![seq_len, emb_size]);
        assert_eq!(out.outputs()[0].get_shape(), expected_shape);
        let onehot_result = one_hot.matmul(&emb.to_fields());
        assert_eq!(
            onehot_result.get_data(),
            out.outputs()[0].to_fields().get_data()
        );
        Ok(())
    }

    #[test]
    fn test_embeddings() -> anyhow::Result<()> {
        let seq_len = 10;
        let vocab_size = 100;
        let emb_size = 20;
        // generate the vector of embeddings for a given index
        let emb_vector = |idx: usize| -> Vec<Element> {
            (0..emb_size)
                .map(|j| Element::from((10000 * idx + j) as Element))
                .collect()
        };
        let table = (0..vocab_size).flat_map(emb_vector).collect::<Vec<_>>();
        let emb_tensor = Tensor::new(vec![vocab_size, emb_size].into(), table);
        let embeddings = Embeddings::new(emb_tensor)?;

        // generate random indices
        let input_data = generate_unique_random_indices(seq_len, vocab_size)
            .into_iter()
            .map(|x| Element::from(x as Element))
            .collect::<Vec<_>>();
        let x = Tensor::new(vec![seq_len].into(), input_data.clone());
        let out = embeddings.evaluate::<GoldilocksExt2>(&[&x], vec![vec![seq_len].into()])?;
        assert_eq!(out.outputs()[0].get_shape(), vec![seq_len, emb_size].into());
        // for each input index, check that the embedding vector is the correct one
        for (idx, table_idx) in input_data.iter().enumerate() {
            let emb = emb_vector(*table_idx as usize);
            let out_emb =
                out.outputs()[0].get_data()[idx * emb_size..(idx + 1) * emb_size].to_vec();
            assert_eq!(emb, out_emb);
        }
        Ok(())
    }
}
