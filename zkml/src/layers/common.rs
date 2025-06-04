use crate::model::trace::LayerOut;

pub trait OpInfo {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(
        &self,
        input_shapes: &[Vec<usize>],
        padding_mode: PaddingMode,
    ) -> Vec<Vec<usize>>;

    /// Compute the number of output tensors, given the number of input tensors
    /// `num_inputs`
    fn num_outputs(&self, num_inputs: usize) -> usize;

    /// Textual description of the operation
    fn describe(&self) -> String;

    /// Specify whether the operation needs to be proven or not
    fn is_provable(&self) -> bool;
}


pub trait Evaluate<T: Number> {
    /// Evaluates the operation given any inputs tensors and constant inputs.
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<T>],
        unpadded_input_shapes: Vec<Vec<usize>>,
    ) -> Result<LayerOut<T, E>>;
}
/// Helper method employed to call `Evaluate::evaluate` when there are no `unpadded_input_shapes`
/// or when the `E` type cannot be inferred automatically by the compiler
pub fn evaluate_layer<E: ExtensionField, T: Number, O: Evaluate<T>>(
    layer: &O,
    inputs: &[&Tensor<T>],
    unpadded_input_shapes: Option<Vec<Vec<usize>>>,
) -> Result<LayerOut<T, E>> {
    layer.evaluate(inputs, unpadded_input_shapes.unwrap_or_default())
}


pub trait ProveInfo<E: ExtensionField>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    /// Compute the proving context for the operation
    fn step_info(&self, id: PolyID, aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)>;

    /// Compute the data necessary to commit to the constant polynomials
    /// associated to the operation. Returns `None` if there are no
    /// constant polynomials to be committed for the given operation
    fn commit_info(&self, _id: NodeId) -> Vec<Option<(PolyID, Vec<E>)>> {
        vec![None]
    }
}
