//! Module contains definition of the [`ProvableOp`] trait, which all operations we can handle
//! should implement.

use crate::Element;
use std::{fmt::Debug, marker::PhantomData};

use crate::{model::inference::step::InferenceStep, tensor::DeepTensor};
use dyn_clone::{DynClone, clone_trait_object};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use tract_onnx::prelude::*;

pub use error::ProvableOpError;
mod add;
mod constant;
mod error;

/// Trait implemented by provable operations. All tensors passed to the [`InferenceOp`]
/// should be UNPADDED (in the sense of converting to an MLE).
///
/// Note: most implementors of [`InferenceOp`] are not expected to store any tensors.
/// If an operation has a constant input, like a convolution with weights and bias,
/// these will be found in their own `Constant` structs.
pub trait InferenceOp<T> : Debug {
    /// Returns the name of the operation, this can always be used to identify the
    /// operation even after type erasure.
    fn name(&self) -> String;

    /// Evaluates the operation given any inputs tensors and constant inputs.
    /// The `T` generic is used to determine which variant of `Self` to run
    /// i.e. floating point or quantised.
    fn evaluate(
        &self,
        inputs: &[DeepTensor<T>],
        const_inputs: &[DeepTensor<T>],
    ) -> Result<Vec<DeepTensor<T>>, ProvableOpError>;

    /// Returns the expected input shapes (in input order)
    fn input_shapes(&self) -> Vec<Vec<usize>>;

    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(&self) -> Vec<Vec<usize>>;
}


pub trait ProvableOp<PCS, E>: InferenceOp<Element>
where
    PCS: PolynomialCommitmentScheme<E>,
    E: ExtensionField,
{
    /// Produces a proof of correct execution for this operation.
    fn prove(
        &self,
        step_data: &InferenceStep<Element>,
        prover: &mut Prover<PCS, E>,
    ) -> Result<(), ProvableOpError>;

    /// Verifies a proof for this operation type
    fn verify(&self, proof: &[u8], verifier: &mut Verifier<PCS, E>) -> Result<(), ProvableOpError>;
}

pub struct Verifier<PCS, E>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    _p: PhantomData<PCS>,
    _e: PhantomData<E>,
}

pub struct Prover<PCS, E>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    _p: PhantomData<PCS>,
    _e: PhantomData<E>,
}
