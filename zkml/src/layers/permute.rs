use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut},
    tensor::{Number, Shape},
};

pub struct Permute {
    args: Vec<usize>,
}

impl Permute {
    pub fn new(args: Vec<usize>) -> Self {
        Self { args }
    }
}

impl<N: Number> Evaluate<N> for Permute {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(
            inputs.iter().all(|t| t.get_shape().len() == 3),
            "Permute expects 3D tensors"
        );
        let output = inputs
            .iter()
            .map(|input| input.permute3d(&self.args))
            .collect::<Vec<_>>();
        Ok(LayerOut::from_vec(output))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{
        Element, Tensor,
        layers::{permute::Permute, provable::Evaluate},
    };

    #[test]
    fn test_permute() {
        let input = Tensor::<Element>::random(&vec![2, 3, 4].into());
        let permute = Permute::new(vec![1, 0, 2]);
        let output = permute
            .evaluate::<GoldilocksExt2>(&[&input], vec![])
            .unwrap();
        assert_eq!(output.outputs()[0].get_shape(), vec![3, 2, 4].into());
    }
}
