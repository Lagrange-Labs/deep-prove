//! Contains the Error enum for things to do with an implementor of [`super::ProvableOp`].

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use crate::tensor::DeepTensorError;

#[derive(Debug, Clone)]
pub enum ProvableOpError {
    /// Error returned when the parameters to a function are incorrect, for instance
    /// if the wrong number of inputs are passed to a function.
    ParameterError(String),
    /// Error returned when types don't line up, for example if we cannot cast a constant tensor into the correct type.
    TypeError(String),
    /// Error variant returned when there is a problem with type conversion
    ConversionError(String),
}

impl Display for ProvableOpError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ProvableOpError::ParameterError(s) => {
                write!(f, "Incorrect parameters passed to ProvableOp: {}", s)
            }
            ProvableOpError::TypeError(s) => {
                write!(f, "Incompatible type used in ProvableOp: {}", s)
            }
            ProvableOpError::ConversionError(s) => {
                write!(f, "Provable Op conversion error: {}", s)
            }
        }
    }
}

impl Error for ProvableOpError {}

impl From<DeepTensorError> for ProvableOpError {
    fn from(value: DeepTensorError) -> Self {
        match value {
            DeepTensorError::ConversionError(s) => ProvableOpError::ConversionError(s),
            DeepTensorError::ParameterError(s) => ProvableOpError::ParameterError(s),
        }
    }
}
