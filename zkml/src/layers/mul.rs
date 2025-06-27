use ff_ext::ExtensionField;

use crate::{Tensor, tensor::Number};

use super::provable::LayerOut;

pub struct ScalarMul<N: Number> {
    constant: N,
}

impl<N: Number> ScalarMul<N> {
    pub fn new(cst: N) -> Self {
        Self { constant: cst }
    }

    pub fn evaluate<N2: Number, E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N2>],
    ) -> anyhow::Result<LayerOut<N2, E>> {
        let result = inputs
            .iter()
            .map(|input| input.scalar_mul_f32(self.constant))
            .collect::<Vec<_>>();
        Ok(LayerOut::from_vec(result))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use super::*;

    #[test]
    fn test_scalar_mul() {
        let scalar_mul = ScalarMul::new(2.0);
        let input = Tensor::new(vec![1, 2, 3].into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = scalar_mul.evaluate::<_, GoldilocksExt2>(&[&input]).unwrap();
        assert_eq!(result.outputs[0].data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }
}
