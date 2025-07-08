use std::collections::HashMap;

use anyhow::{bail, ensure, Context};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use transcript::Transcript;

use crate::{
    iop::context::ContextAux, layers::{matrix_mul::{MatMul, MatMulCtx, MatMulProof, OperandMatrix}, provable::{Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx}, LayerCtx}, padding::{PaddingMode, ShapeInfo}, tensor::{Number, Shape}, Claim, Element, Tensor
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embeddings<N> {
    mat: MatMul<N>,
    emb_size: usize,
    pub(crate) vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsCtx<E> {
    mat_ctx: MatMulCtx<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct EmbeddingsProof<E: ExtensionField> {
    mat_proof: MatMulProof<E>,
}


impl<N: Number> Embeddings<N> {
    pub fn new(emb: Tensor<N>) -> anyhow::Result<Self> {
        let emb_size= emb.get_shape()[1];
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
        self.mat.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        self.mat.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!("Embeddings(vocab:{:?}, hidden:{:?})", self.vocab_size, self.emb_size)
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
                let shape: Shape = x.get_shape().into();
                shape.rank() == 2 && shape.dim(1) == 1
            }),
            "embeddings only support 2d tensors with 1 value: {:?}",
            inputs.iter().map(|x| x.get_shape()).collect::<Vec<_>>()
        );
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        // we still uses this evaluation for inference as it's quicker
        // than doing the matmul with one hot encoding. Proving however will generate
        // the one hot encoding and do the matmul.
        let OperandMatrix::Weight(ref w ) = self.mat.right_matrix else {
            bail!("right matrix is not a weight matrix");
        };
        let emb = &w.tensor;
        let x = inputs[0];
        let seq_len = x.get_shape()[0];
        let vocab_size = emb.get_shape()[0];
        let emb_size = emb.get_shape()[1];
        let emb_data = emb.get_data();
        let emb = x
            .slice_last_dim()
            .flat_map(|v| {
                let idx = v[0].to_usize();
                assert!(
                    idx < vocab_size,
                    "idx {} out of bounds for vocab size {}",
                    idx,
                    vocab_size
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
        self.mat.pad_node(si).map(|mat| Self { mat, ..self })
    }
}

impl<E> ProveInfo<E> for Embeddings<Element> 
where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        let mat_ctx = self.mat.ctx(id, &mut aux).context("embeddings matmul: ")?;
        Ok((LayerCtx::Embeddings(EmbeddingsCtx { mat_ctx }), aux))
    }
}

impl<E> OpInfo for EmbeddingsCtx<E>
where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
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

impl<E,PCS> ProvableOp<E,PCS> for Embeddings<Element>
where 
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = EmbeddingsCtx<E>;
    
    fn prove<T: Transcript<E>>(
        &self,
        node_id: crate::layers::provable::NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&crate::Claim<E>>,
        step_data: &crate::model::StepData<E, E>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        assert!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(std::vec![Claim::default()])
    }
    
    fn gen_lookup_witness(
        &self,
        _id: crate::layers::provable::NodeId,
        _gen: &mut crate::lookup::context::LookupWitnessGen<E, PCS>,
        _ctx: &crate::Context<E, PCS>,
        _step_data: &crate::model::StepData<Element, E>,
    ) -> anyhow::Result<()> {
        // Default implementation for nodes that don't employ a lookup table
        Ok(())
    }
}

impl<E,PCS> VerifiableCtx<E,PCS> for EmbeddingsCtx<E>
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
        verifier: &mut crate::iop::verifier::Verifier<E, T, PCS>,
        shape_step: &crate::iop::context::ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        todo!()
    }
}

fn one_hot_encoding(indices: &[usize], vocab_size: usize) -> Tensor<Element> {
    let mut data = Vec::new();
    for idx in indices {
        let mut one_hot = vec![0; vocab_size];
        one_hot[*idx] = 1;
        data.extend_from_slice(&one_hot);
    }   
    Tensor::new(vec![indices.len(), vocab_size].into(), data)
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{Rng, thread_rng};
    use ff_ext::GoldilocksExt2;

    use crate::Element;

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
    fn test_one_hot_encoding() {
        let indices = vec![0, 1, 2, 3, 4];
        let vocab_size = 5;
        let one_hot = one_hot_encoding(&indices, vocab_size);
        assert_eq!(one_hot.get_shape(), vec![indices.len(), vocab_size].into());
        assert_eq!(one_hot.get_data(), vec![1, 0, 0, 0, 0, 
                                            0, 1, 0, 0, 0, 
                                            0, 0, 1, 0, 0, 
                                            0, 0, 0, 1, 0,
                                            0, 0, 0, 0, 1]);
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
        let x = Tensor::new(vec![seq_len, 1].into(), input_data.clone());
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
