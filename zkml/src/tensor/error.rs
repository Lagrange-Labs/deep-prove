//! Error enum for the [`DeepTensor`] enum and its related operations

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

#[derive(Debug, Clone)]
pub enum DeepTensorError {
    ParameterError(String),
    ConversionError(String),
}

impl Display for DeepTensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            DeepTensorError::ParameterError(s) => {
                write!(f, "Parametes to DeepTensor method were incorrect: {}", s)
            }
            DeepTensorError::ConversionError(s) => {
                write!(f, "Error when trying to convert between data types: {}", s)
            }
        }
    }
}

impl Error for DeepTensorError {}
