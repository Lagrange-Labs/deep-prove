//! Module containing the definition of the [`Add`] operation.

mod op_impl;

#[derive(Debug, Clone)]
/// Used to add tensors
pub struct Add {
    /// This is the number of non-constant inputs
    no_inputs: usize,
    /// This is the number of constant inputs
    no_constant_inputs: usize,
    /// The expected input shapes
    input_shapes: Vec<Vec<usize>>,
    /// The output shapes
    output_shapes: Vec<Vec<usize>>,
}
