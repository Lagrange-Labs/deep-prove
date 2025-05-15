//! Contains the Error enum for things to do with an implementor of [`super::ProvableOp`].

#[derive(thiserror::Error, Debug)]
pub enum ProvableOpError {
    /// Error returned when the parameters to a function are incorrect, for instance
    /// if the wrong number of inputs are passed to a function.
    #[error("Incorrect parameters passed to ProvableOp: {0}")]
    ParameterError(String),
    /// Error returned when types don't line up, for example if we cannot cast a constant tensor into the correct type.
    #[error("Incompatible type used in ProvableOp: {0}")]
    TypeError(String),
    /// Error variant returned when there is a problem with type conversion
    #[error("Provable Op conversion error: {0}")]
    ConversionError(String),
    /// Error variant returned when there is a problem with type conversion
    #[error("ProvableOp generic error: {0}")]
    GenericError(anyhow::Error),
    /// Error returned when trying to prove or verify a non provable layer
    #[error("Trying to prove or verify a non provable layer: {0}")]
    NotProvableLayer(String),
    /// Error returned when there is a problem with parsing the ONNX model
    #[error("Error parsing ONNX model: {0}")]
    OnnxParsingError(String),
}

impl From<anyhow::Error> for ProvableOpError {
    fn from(error: anyhow::Error) -> Self {
        ProvableOpError::GenericError(error)
    }
}
