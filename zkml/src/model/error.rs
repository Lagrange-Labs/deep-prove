//! Module containing code definig [`ModelError`] type and conversions.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

#[derive(Debug, Clone)]
/// Errors relating to [`super::inference::inference_model::InferenceModel`].
pub enum ModelError {
    /// Error variant returned when parameters passed to a model are incorrect,
    /// i.e. input tensor dimensions are incorrect or types don't line up.
    ParameterError(String),
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ModelError::ParameterError(s) => write!(f, "Incorrect Parameters fed to Model: {}", s),
        }
    }
}

impl Error for ModelError {}
