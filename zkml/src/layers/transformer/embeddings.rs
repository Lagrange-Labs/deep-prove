use std::iter::once;

use crate::{
    ScalingFactor, ScalingStrategy,
    commit::{compute_betas_eval, identity_eval},
    layers::{
        LayerProof,
        provable::{QuantizeOp, QuantizeOutput},
    },
    to_bit_sequence_le,
};

use anyhow::{anyhow, bail, ensure};
use ff_ext::{ExtensionField, SmallField};
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::MultilinearExtension,
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx,
        matrix_mul::{MatMul, OperandMatrix},
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
    id: NodeId,
    vocab_size: usize,
    emb_size: usize,
    sumcheck_poly_aux: VPAuxInfo<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct EmbeddingsProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the matmul protocol
    pub(crate) sumcheck: IOPProof<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly".
    /// Since we only support quadratic right now it's a flat list.
    individual_claims: Vec<E>,
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

    pub(crate) fn embedding_matrix(&self) -> &Tensor<N> {
        let OperandMatrix::Weight(embedding_matrix) = &self.mat.right_matrix else {
            unreachable!()
        };
        &embedding_matrix.tensor
    }

    /// Split the point over which the 2d output tensor is evaluated into 2 sub-points:
    /// - The first sub-point refers to the row variables of the output tensor
    /// - The second sub-point refers to the column variables of the output tensor
    fn split_output_point<E: ExtensionField>(
        last_claim: &Claim<E>,
        emb_size: usize,
    ) -> anyhow::Result<(&[E], &[E])> {
        let num_vars = emb_size.next_power_of_two().ilog2() as usize;
        // column variables are the least significant ones
        let column_point = &last_claim.point[..num_vars];
        // row variables are the most significant ones
        let row_point = &last_claim.point[num_vars..];

        Ok((row_point, column_point))
    }

    /// Build points over which the evaluations produced by the sum-check in proving protocol are evaluated.
    /// - The first point being returned is the point for the claim about the one-hot encoded input
    /// - The second point being returned is the point for the claim about the emebdding matrix
    fn build_points_for_claims<E: ExtensionField>(
        last_claim: &Claim<E>,
        emb_size: usize,
        sumcheck_point: &[E],
    ) -> anyhow::Result<(Vec<E>, Vec<E>)> {
        let (row_point, column_point) = Self::split_output_point(last_claim, emb_size)?;
        let one_hot_claim_point = sumcheck_point
            .iter()
            .chain(row_point)
            .cloned()
            .collect_vec();
        let embedding_mat_point = column_point
            .iter()
            .chain(sumcheck_point.iter())
            .cloned()
            .collect_vec();
        Ok((one_hot_claim_point, embedding_mat_point))
    }
}

fn output_shapes(
    input_shapes: &[Shape],
    padding_mode: PaddingMode,
    embedding_size: usize,
) -> Vec<Shape> {
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
        PaddingMode::NoPadding => Shape::new(vec![seq_len, embedding_size]),
        PaddingMode::Padding => Shape::new(vec![
            seq_len.next_power_of_two(),
            embedding_size.next_power_of_two(),
        ])
        .next_power_of_two(),
    };
    vec![shape]
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
            PaddingMode::NoPadding => Shape::new(vec![seq_len, self.emb_size]),
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
            "embeddings only support 1d tensors: {:?}",
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
        let shape_data = si.shapes.get_mut(0).unwrap();
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
        let r = self.mat.pad_node(si).map(|mat| Self { mat, ..self })?;
        Ok(r)
    }
}

const EMBEDDING_POLY_ID: &str = "EmbeddingMat";

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
        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);
        aux.model_polys = Some(
            once((
                EMBEDDING_POLY_ID.to_string(),
                self.embedding_matrix().pad_next_power_of_two().data,
            ))
            .collect(),
        );
        let num_vars = self.vocab_size.next_power_of_two().ilog2() as usize;
        Ok((
            LayerCtx::Embeddings(EmbeddingsCtx {
                id,
                sumcheck_poly_aux: VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
                vocab_size: self.vocab_size,
                emb_size: self.emb_size,
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
        output_shapes(input_shapes, padding_mode, self.emb_size)
    }

    fn num_outputs(&self, _num_inputs: usize) -> usize {
        1
    }

    fn describe(&self) -> String {
        format!(
            "EmbeddingsCtx(vocab:{:?}, hidden:{:?})",
            self.vocab_size, self.emb_size
        )
    }

    fn is_provable(&self) -> bool {
        true
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
        let input = &step_data.inputs[0];
        let last_claim = last_claims[0];

        let (row_point, column_point) = Self::split_output_point(last_claim, self.emb_size)?;

        // we need to compute the vector `reduced_one_hot` whose MLE corresponds to the MLE of the one-hot
        // encoded input matrix, with row variables fixed to `row_point`.
        // Relying on the sparse structure of the one-hot encoded input matrix, this vector can be computed
        // as `reduced_one_hot[x[i]] += \beta(i, row_point)`, for all items `x[i]` in the input tensor

        // we precompute all items `\beta(i, row_point)` for all `i` between `0` and `x.len()`
        let beta_vec = compute_betas_eval(row_point);
        let vocab_size = self.vocab_size.next_power_of_two();
        let emb_size = self.emb_size.next_power_of_two();
        // we now build the `reduced_one_hot` vector as `reduced_one_hot[x[i]] += beta_vec[i]`
        let mut reduced_one_hot = vec![E::ZERO; vocab_size];

        input.get_data().iter().enumerate().try_for_each(|(i, x)| {
            let x = x
                .as_base()
                .ok_or(anyhow!("Input data at position {i} bigger than base field"))?
                .to_canonical_u64() as usize;
            reduced_one_hot[x] += beta_vec[i];
            anyhow::Ok(())
        })?;

        let reduced_one_hot = Tensor::new(vec![1, vocab_size].into(), reduced_one_hot);

        let embedding_matrix = self.embedding_matrix();

        ensure!(
            vocab_size == embedding_matrix.nrows_2d(),
            "Expected {vocab_size} rows for embedding matrix, found {}",
            embedding_matrix.nrows_2d(),
        );

        ensure!(
            emb_size == embedding_matrix.ncols_2d(),
            "Expected {emb_size} columns for embedding matrix, found {}",
            embedding_matrix.ncols_2d(),
        );

        let input_mle = reduced_one_hot.to_mle_2d();

        let mut embedding_mat_mle = embedding_matrix.to_2d_mle();

        embedding_mat_mle.fix_variables_in_place_parallel(column_point);

        // check that after fixing the variables in both matrices the number of free
        // variables is the same
        assert_eq!(input_mle.num_vars(), embedding_mat_mle.num_vars());

        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        vp.add_mle_list(vec![input_mle.into(), embedding_mat_mle.into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // sum-check will produce claims about `reduced_one_hot` vector and the `embedding_matrix`. We need
        // to commit build a claim for the one-hot encoded input tensor from the first claim, and produce an
        // opening proof for the second claim
        let one_hot_eval = state.get_mle_final_evaluations()[0];
        let embedding_mat_eval = state.get_mle_final_evaluations()[1];

        let (one_hot_claim_point, embedding_mat_point) =
            Self::build_points_for_claims(last_claim, self.emb_size, &proof.point)?;

        let output_claim = Claim::new(one_hot_claim_point, one_hot_eval);

        let embedding_claim = Claim::new(embedding_mat_point, embedding_mat_eval);

        prover.add_common_claims(
            node_id,
            once((EMBEDDING_POLY_ID.to_string(), embedding_claim)).collect(),
        )?;

        prover.push_proof(
            node_id,
            LayerProof::Embeddings(EmbeddingsProof {
                sumcheck: proof,
                individual_claims: state.get_mle_final_evaluations(),
            }),
        );
        Ok(vec![output_claim])
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
        let last_claim = &last_claims[0];
        let subclaim = IOPVerifierState::<E>::verify(
            last_claim.eval,
            &proof.sumcheck,
            &self.sumcheck_poly_aux,
            verifier.transcript,
        );

        // build claims produced by sum-check: a claim about the one-hot encoded input, and a claim about
        // the embedding matrix

        let (one_hot_claim_point, emebdding_mat_point) =
            Embeddings::<Element>::build_points_for_claims(
                last_claim,
                self.emb_size,
                &subclaim.point_flat(),
            )?;
        let one_hot_eval = proof.individual_claims[0];
        let embedding_mat_eval = proof.individual_claims[1];

        let one_hot_claim = Claim::new(one_hot_claim_point, one_hot_eval);

        let embedding_mat_claim = Claim::new(emebdding_mat_point, embedding_mat_eval);

        verifier.add_common_claims(
            self.id,
            once((EMBEDDING_POLY_ID.to_string(), embedding_mat_claim)).collect(),
        )?;

        // SUMCHECK verification part
        // Instead of computing the polynomial at the random point requested like this
        // let computed_point = vp.evaluate(
        //     subclaim
        //         .point
        //         .iter()
        //         .map(|c| c.elements)
        //         .collect_vec()
        //         .as_ref(),
        //
        // We compute the evaluation directly from the individual final evaluations of each polynomial
        // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
        ensure!(
            one_hot_eval * embedding_mat_eval == subclaim.expected_evaluation,
            "sumcheck claim failed in Logits layer verifier",
        );

        // the first claim is the one hot encoding claim. To verify it we need to
        // efficiently evaluate the one hot encoding on it - we do this "at the end" of the verification
        // procedure to respect the framework's order of operations. The logic is in `verify_input_claim`.
        Ok(vec![one_hot_claim])
    }

    fn verify_input_claim<A: AsRef<Tensor<E>>>(
        &self,
        inputs: &[A],
        claims: &[&Claim<E>],
    ) -> anyhow::Result<()> {
        // TODO verify efficiently the one hot encoding claim
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        ensure!(claims.len() == 1, "embeddings only support 1 claim");
        let input = inputs[0].as_ref();
        let one_hot_claim = &claims[0];
        let vocab_nv = self.vocab_size.next_power_of_two().ilog2();
        let seq_len_nv = input.get_shape().dim(0).next_power_of_two().ilog2();
        ensure!(
            vocab_nv + seq_len_nv == one_hot_claim.point.len() as u32,
            "vocab_nv: {vocab_nv}, seq_len_nv: {seq_len_nv}, one_hot_claim.point.len(): {}",
            one_hot_claim.point.len()
        );
        let (r1, r2) = one_hot_claim.point.split_at(vocab_nv as usize);
        let b1 = compute_betas_eval(r2);
        let sum = input
            .get_data()
            .iter()
            .zip(b1)
            .fold(E::ZERO, |sum, (token, beta)| {
                let token_value = token.to_canonical_u64_vec()[0] as usize;
                let token_le_bits = to_bit_sequence_le(token_value, r1.len())
                    .map(|b| E::from_canonical_usize(b))
                    .collect_vec();
                let selector = beta * identity_eval(r1, &token_le_bits);
                sum + selector
            });
        ensure!(
            sum == one_hot_claim.eval,
            "one hot encoding claim is incorrect"
        );
        Ok(())
    }
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
            .map(|_| thread_rng().gen_range(0..vocab_size) as f32)
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
        prove_model_with(model, vec![input])?;

        Ok(())
    }

    fn one_hot_encoding<E: ExtensionField>(indices: &[E], vocab_size: usize) -> Tensor<E> {
        let mut data = Vec::new();
        for idx in indices {
            let mut one_hot = vec![E::ZERO; vocab_size];
            let idx: usize = idx.to_canonical_u64_vec()[0].try_into().unwrap();
            one_hot[idx] = E::ONE;
            data.extend_from_slice(&one_hot);
        }
        Tensor::new(vec![indices.len(), vocab_size].into(), data)
    }

    #[test]
    fn test_one_hot_encoding_inference() -> anyhow::Result<()> {
        let seq_len: usize = 5;
        let indices_elem: Vec<Element> = (0..seq_len).map(|i| i as Element).collect::<Vec<_>>();
        let indices: Tensor<GoldilocksExt2> =
            Tensor::<Element>::new(vec![5].into(), indices_elem.clone()).to_fields();
        let vocab_size = 6;
        let emb_size = 10;
        let one_hot = one_hot_encoding(&indices.get_data(), vocab_size);
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
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
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
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ZERO,
                GoldilocksExt2::ONE,
                GoldilocksExt2::ZERO,
            ]
        );

        let emb = Tensor::<Element>::random(&vec![vocab_size, 10].into());
        let embeddings = Embeddings::new(emb.clone())?;
        let input = Tensor::new(vec![seq_len].into(), indices_elem.clone());
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
