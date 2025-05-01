use std::path::PathBuf;

use crate::quantization;
use ark_std::rand::{self, Rng, SeedableRng, rngs::StdRng, thread_rng};
use ff_ext::ExtensionField;
use itertools::Itertools;
use tract_onnx::prelude::*;

use crate::Element;

pub fn _random_vector<E: ExtensionField>(n: usize) -> Vec<E> {
    let mut rng = thread_rng();
    (0..n).map(|_| E::random(&mut rng)).collect_vec()
}

pub fn random_vector(n: usize) -> Vec<Element> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| rng.gen_range(*quantization::MIN..=*quantization::MAX))
        .collect_vec()
}

pub fn random_field_vector<E: ExtensionField>(n: usize) -> Vec<E> {
    let mut rng = thread_rng();
    (0..n).map(|_| E::random(&mut rng)).collect_vec()
}

pub fn random_bool_vector<E: ExtensionField>(n: usize) -> Vec<E> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| E::from(rng.gen_bool(0.5) as u64))
        .collect_vec()
}

#[allow(unused)]
pub fn random_vector_seed(n: usize, seed: Option<u64>) -> Vec<Element> {
    let seed = seed.unwrap_or(rand::random::<u64>()); // Use provided seed or default
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| rng.gen_range(*quantization::MIN..=*quantization::MAX))
        .collect_vec()
}

pub fn load_test_onnx_model(operator_name: &str) -> anyhow::Result<TypedModel> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file = format!(
        "zkml/assets/test_scripts/{}/{}.onnx",
        operator_name, operator_name
    );
    let filepath = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .to_path_buf()
        .join(file);

    let model = tract_onnx::onnx()
        .model_for_path(filepath)
        .map_err(|e| anyhow::Error::msg(format!("Failed to load model: {:?}", e)))?;

    model.into_typed()?.into_optimized()
}
