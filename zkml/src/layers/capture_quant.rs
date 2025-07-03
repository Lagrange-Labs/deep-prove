//! Capture the output of layers' quantization for regression tests

use serde::Serialize;
use std::{fs, path::Path};

// Store the output hash and the output at path containing the layer kind and input hash as dir names.
pub fn store<I, O>(out_dir: &Path, input: &I, output: &O)
where
    I: Serialize,
    O: Serialize,
{
    let output_bytes = serde_json::to_vec_pretty(output).unwrap();
    let input_hash = sha256_hex(input);
    let output_hash = sha256_hex_bytes(&output_bytes);
    let dir = out_dir.join(input_hash);

    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("output_hash.txt"), output_hash).unwrap();
    fs::write(dir.join("data.json"), output_bytes).unwrap();
}

fn sha256_hex<T>(data: &T) -> String
where
    T: Serialize,
{
    let bytes = serde_json::to_vec(data).unwrap();
    sha256_hex_bytes(&bytes)
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let hash = <sha2::Sha256 as sha2::Digest>::digest(bytes);
    format!("{hash:X}")
}
