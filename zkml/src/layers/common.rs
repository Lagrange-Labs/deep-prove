use ff_ext::ExtensionField;
use serde::{Serialize, de::DeserializeOwned};

use crate::{Element, Tensor, commit::precommit::PolyID, iop::context::ContextAux, tensor::Number};

use super::LayerCtx;

pub trait Op<N: Number>:
    Clone + std::fmt::Debug + Sync + Send + Serialize + DeserializeOwned
{
    fn describe(&self) -> String;
    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize>;
    fn op(&self, input: &Tensor<N>) -> Tensor<N>;
}

#[allow(unused)]
pub(crate) trait ProvableOp: Op<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        aux: ContextAux<E>,
    ) -> (LayerCtx<E>, ContextAux<E>)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned;
}
