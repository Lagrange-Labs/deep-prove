use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    NextPowerOfTwo, Tensor, commit::precommit::PolyID, iop::context::ContextAux, layers::LayerCtx,
    tensor::Number,
};

use super::common::{Op, ProvableOp};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reshape;

impl<N: Number> Op<N> for Reshape {
    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        vec![input_shape.iter().product::<usize>()]
    }

    fn op(&self, input: &Tensor<N>) -> Tensor<N> {
        input.flatten()
    }
    fn describe(&self) -> String {
        "Reshape".to_string()
    }
}

impl ProvableOp for Reshape {
    fn step_info<E: ExtensionField>(
        &self,
        _id: PolyID,
        mut aux: ContextAux<E>,
    ) -> (LayerCtx<E>, ContextAux<E>)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.last_output_shape = aux.last_output_shape.next_power_of_two();
        // No model polys for reshape
        aux.model_polys = vec![];

        (LayerCtx::Reshape, aux)
    }
}
