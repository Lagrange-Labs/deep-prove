use anyhow::ensure;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo},
    padding::PaddingMode,
    tensor::{Number, Shape},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embeddings<N> {
    pub emb: Tensor<N>,
}

impl<N: Number> Embeddings<N> {
    pub fn new(emb: Tensor<N>) -> Self {
        Self { emb }
    }
}

impl<N: Number> OpInfo for Embeddings<N> {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        assert_eq!(input_shapes.len(), 1);
        // for each input, we output an embedding vector
        input_shapes
            .iter()
            .map(|shape| Shape::new(vec![shape[0], self.emb.get_shape()[1]]))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        assert_eq!(num_inputs, 1, "no batch support for now");
        num_inputs
    }

    fn describe(&self) -> String {
        format!("Embeddings({:?})", self.emb.get_shape())
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
                shape.rank() == 2 && shape.dim(1) == 1
            }),
            "embeddings only support 2d tensors with 1 value: {:?}",
            inputs.iter().map(|x| x.get_shape()).collect::<Vec<_>>()
        );
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        let x = inputs[0];
        let seq_len = x.get_shape()[0];
        let vocab_size = self.emb.get_shape()[0];
        let emb_size = self.emb.get_shape()[1];
        let emb_data = self.emb.get_data();
        let emb = x
            .slice_last_dim()
            .flat_map(|v| {
                let idx = v[0].to_usize();
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
        let embeddings = Embeddings::new(emb_tensor);

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
