use crate::quantization;
use ark_std::rand::{self, Rng, SeedableRng, rngs::StdRng, thread_rng};
use ff_ext::ExtensionField;
use itertools::Itertools;

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
        .map(|_| E::from_canonical_u64(rng.gen_bool(0.5) as u64))
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
