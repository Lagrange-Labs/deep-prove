//! Module containing error enum for commitment related errors

use mpcs::Error as MPCSError;

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

#[derive(Clone, Debug)]
/// Encapsulates errors that can occur when dealing with commitments during [`Model`](crate::model::Model) proving.
pub enum PCSError {
    /// Error variant used when incorrect parameters are provided
    ParameterError(String),
    /// Error variant from the [`mpcs`] crate.
    MPCSError(MPCSError),
}

impl Display for PCSError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            PCSError::ParameterError(s) => {
                write!(f, "Commitment Error: Incorrect Parameters: {}", s)
            }
            PCSError::MPCSError(e) => {
                write!(f, "Error in mpcs crate method, internal error: {:?}", e)
            }
        }
    }
}

impl Error for PCSError {}

impl From<MPCSError> for PCSError {
    fn from(e: MPCSError) -> Self {
        PCSError::MPCSError(e)
    }
}
