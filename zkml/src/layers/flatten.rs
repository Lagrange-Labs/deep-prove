use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Tensor,
    iop::context::ContextAux,
    layers::LayerCtx,
    padding::{PaddingMode, ShapeInfo, reshape},
    tensor::{Number, Shape},
};

use super::provable::{Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProveInfo};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Flatten;
/// Even if empty, we need a context such that it implements the default
/// methods of `VerifiableCtx``
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlattenCtx;

impl OpInfo for Flatten {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|s| Shape::new(vec![s.product()]))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Reshape".to_string()
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl<N: Number> Evaluate<N> for Flatten {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: Vec<Shape>,
    ) -> Result<LayerOut<N, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating reshape layer"
        );
        let input = inputs[0];
        Ok(LayerOut::from_vec(vec![input.flatten()]))
    }
}

impl<E> ProveInfo<E> for Flatten
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, _id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        aux.last_output_shape
            .iter_mut()
            .for_each(|s| *s = s.next_power_of_two());
        Ok((LayerCtx::Flatten, aux))
    }
}

impl PadOp for Flatten {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        reshape(si)
    }
}
