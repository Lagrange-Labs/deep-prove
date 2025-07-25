use serde::{Deserialize, Serialize};

pub mod v1;
pub mod v2;

/// A versioned enum representing a deep prove proving request
#[derive(Serialize, Deserialize)]
pub enum DeepProveRequest {
    /// Version 1
    V1(v1::DeepProveRequest),
}

/// A versioned enum representing a deep prove proving response
#[derive(Serialize, Deserialize)]
pub enum DeepProveResponse {
    /// Version 1
    V1(v1::DeepProveResponse),
}
