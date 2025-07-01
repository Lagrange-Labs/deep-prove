//! Capture the output of layers' quantization for regression tests

use serde::Serialize;
use std::{fs, path::PathBuf};

const DIR: &'static str = "layers-quant";

// Store the output hash and the output at path containing the layer kind and input hash as dir names.
pub fn store<T>(layer_kind: &str, input_hash: &str, output_hash: &str, output: &T)
where
    T: Serialize,
{
    let output_bytes = serde_json::to_vec_pretty(output).unwrap();
    let file_dir = PathBuf::from(DIR).join(layer_kind).join(input_hash);
    fs::create_dir_all(&file_dir).unwrap();
    fs::write(file_dir.join("output_hash.txt"), output_hash).unwrap();
    fs::write(file_dir.join("data.json"), output_bytes).unwrap();
}
