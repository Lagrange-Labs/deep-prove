//! LogUp GKR related errors

use std::error::Error;

#[derive(Clone, Debug)]
pub enum LogUpError {
    PolynomialError(String),
    ProvingError(String),
    VerifierError(String),
    ParameterError(String),
}

impl std::fmt::Display for LogUpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogUpError::PolynomialError(s) => write!(f, "Polynomial related error: {s}"),
            LogUpError::ProvingError(s) => write!(f, "Error during LogUp proving: {s}"),
            LogUpError::VerifierError(s) => write!(f, "Error while verifying LogUp proof: {s}"),
            LogUpError::ParameterError(s) => write!(f, "Parameters were incorrect: {s}"),
        }
    }
}

impl Error for LogUpError {}
