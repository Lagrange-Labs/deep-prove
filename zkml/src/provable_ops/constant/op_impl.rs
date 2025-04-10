//! Module is the place where we define the [`InferenceOp`] implementation for [`Constant`].

use super::{
    super::{InferenceOp, error::ProvableOpError},
    Constant,
};
use crate::tensor::DeepTensor;
use crate::tensor::deep_tensor::Number;

impl<T: Number> InferenceOp<T> for Constant<T> {
    fn name(&self) -> String {
        "Constant".to_string()
    }

    fn evaluate(
        &self,
        inputs: &[DeepTensor<T>],
        const_inputs: &[DeepTensor<T>],
    ) -> Result<Vec<DeepTensor<T>>, ProvableOpError> {
        // We check that no input is provided
        if !inputs.is_empty() || !const_inputs.is_empty() {
            return Err(ProvableOpError::ParameterError(
                "Constant operation should not have any inputs".to_string(),
            ));
        }

        Ok(vec![self.tensor.clone()])
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    fn output_shapes(&self) -> Vec<Vec<usize>> {
        vec![self.tensor.shape().to_vec()]
    }
}
