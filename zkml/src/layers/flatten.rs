use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{
    Element, NextPowerOfTwo, ScalingFactor, Tensor,
    commit::precommit::PolyID,
    iop::context::ContextAux,
    layers::LayerCtx,
    padding::{PaddingMode, ShapeInfo, reshape},
    tensor::Number,
};

use super::{
    Layer,
    provable::{Evaluate, LayerOut, Op, OpInfo, PadOp, ProvableOp, ProvableOpError, ProveInfo},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Flatten;

impl OpInfo for Flatten {
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        _padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>> {
        input_shapes
            .into_iter()
            .map(|s| vec![s.iter().product()])
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
        _unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<N, E>, super::provable::ProvableOpError> {
        if inputs.len() != 1 {
            return Err(super::provable::ProvableOpError::ParameterError(
                "Reshape expects exactly one input".to_string(),
            ));
        };
        let input = inputs[0];
        Ok(LayerOut::from_vec(vec![input.flatten()]))
    }
}

impl<E> ProveInfo<E> for Flatten
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(
        &self,
        _id: PolyID,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux), ProvableOpError> {
        aux.last_output_shape
            .iter_mut()
            .for_each(|s| *s = s.next_power_of_two());
        Ok((LayerCtx::Flatten, aux))
    }
}

impl PadOp for Reshape {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self, super::provable::ProvableOpError>
    where
        Self: Sized,
    {
        reshape(si).map_err(|e| ProvableOpError::GenericError(e))
    }
}

impl<E> ProvableOp<E> for Flatten
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Ctx = LayerCtx<E>; // Unused
}

impl<E, N> Op<E, N> for Flatten
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    N: Number,
{
}

impl Flatten {
    pub(crate) fn quantize(
        &self,
        _s: &ScalingFactor,
        _bias_s: Option<&ScalingFactor>,
    ) -> Layer<Element> {
        Layer::Flatten(Flatten)
    }
}
