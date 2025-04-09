//! Module is the place where we define the [`InferenceOp`] implementation for [`Add`].

use super::{
    super::{InferenceOp, error::ProvableOpError},
    Add,
};
use crate::tensor::DeepTensor;

// impl InferenceOp for Add {
//     fn name(&self) -> String {
//         "Add".to_string()
//     }

//     fn evaluate(
//         &self,
//         inputs: &[DeepTensor],
//         const_inputs: &[DeepTensor],
//     ) -> Result<Vec<DeepTensor>, ProvableOpError> {
//         // We check that the correct number of inputs is provided
//         if inputs.len() != self.no_inputs {
//             return Err(ProvableOpError::ParameterError(format!("Add op expected {} non-constant inputs, received {}", self.no_inputs, inputs.len())))
//         }

//         if const_inputs.len() != self.no_constant_inputs {
//             return Err(ProvableOpError::ParameterError(format!("Add op expected {} constant inputs, received {}", self.no_constant_inputs, const_inputs.len())))
//         }

//         // We check that the inputs have the correct shape
//         for (input_shape, expected_shape) in inputs.iter().chain(const_inputs.iter()).map(|t| t.shape()).zip(self.input_shapes.iter()) {
//             if input_shape != expected_shape.as_slice() {
//                 return Err(ProvableOpError::ParameterError("One of the input shapes was incorrect to Add Op".to_string()));
//             }
//         }

//         // Finally we check that all the inputs contain the same data type
//         let first_input = &inputs[0];
//         for tensor in inputs.iter().skip(1).chain(const_inputs.iter()) {
//             if !first_input.same_type(tensor) {
//                 return Err(ProvableOpError::TypeError("Inputs to Add Op did not all have the same underlying type".to_string()))
//             }
//         }

//         Ok(vec![self.tensor.clone()])
//     }

//     fn input_shapes(&self) -> Vec<Vec<usize>> {
//         vec![]
//     }

//     fn output_shapes(&self) -> Vec<Vec<usize>> {
//         vec![self.tensor.shape().to_vec()]
//     }
// }
