use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{tensor::Number, Tensor};

use super::provable::{Evaluate, LayerOut, ProvableOpError};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add<N> {
    operand: Option<Tensor<N>>,
}

impl<N: Number> Add<N> {
    pub fn new() -> Self {
        Self { operand: None }
    }
    pub fn new_with(operand: Option<Tensor<N>>) -> Self {
        Self { operand }
    }
}

impl<N: Number> Add<N> {
    fn evaluate<E: ExtensionField>(&self, inputs: &[&Tensor<N>]) -> anyhow::Result<LayerOut<N, E>> {
        let result = if inputs.len() == 2 {
            ensure!(inputs[0].get_shape() == inputs[1].get_shape(), "Add layer expects inputs to have the same shape");
            inputs[0].add(inputs[1])
        } else if inputs.len() == 1 {
            ensure!(self.operand.is_some(), "Add operand can't be None if there is only one input");
            ensure!(inputs[0].get_shape() == self.operand.as_ref().unwrap().get_shape(), "Add layer expects input and operand to have the same shape");
            inputs[0].add(self.operand.as_ref().unwrap())
        } else {
            bail!("Add layer expects 1 or 2 inputs, got {}", inputs.len());
        };
        Ok(LayerOut::from_vec(vec![result]))
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;

    use crate::Element;

    use super::*;

    #[test]
    fn test_add() {
        let add = Add::new();
        let t1 = Tensor::<Element>::random(&vec![2, 2]);
        let t2 = Tensor::<Element>::random(&vec![2, 2]);
        let result = add.evaluate::<GoldilocksExt2>(&[&t1,&t2]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.outputs[0].get(vec![i,j]), t1.get(vec![i,j]) + t2.get(vec![i,j]));
            }
        }
        let add = Add::new_with(Some(t1.clone()));
        let result = add.evaluate::<GoldilocksExt2>(&[&t2]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.outputs[0].get(vec![i,j]), t1.get(vec![i,j]) + t2.get(vec![i,j]));
            }
        }
    }
}