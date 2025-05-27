//! LogUp GKR related errors

use std::error::Error;

use crate::commit::PCSError;

#[derive(Clone, Debug)]
pub enum LogUpError {
    PolynomialError(String),
    ProvingError(String),
    VerifierError(String),
    ParamterError(String),
    PCSError(PCSError),
}

impl std::fmt::Display for LogUpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogUpError::PolynomialError(s) => write!(f, "Polynomial related error: {}", s),
            LogUpError::ProvingError(s) => write!(f, "Error during LogUp proving: {}", s),
            LogUpError::VerifierError(s) => write!(f, "Error while verifying LogUp proof: {}", s),
            LogUpError::ParamterError(s) => write!(f, "Parameters were incorrect: {}", s),
            LogUpError::PCSError(e) => write!(f, "Error occurred with commitments: {:?}", e),
        }
    }
}

impl Error for LogUpError {}

impl From<PCSError> for LogUpError {
    fn from(error: PCSError) -> Self {
        LogUpError::PCSError(error)
    }
}
