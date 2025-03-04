use crate::quantization;
use ark_std::rand::{
    self, Rng, SeedableRng,
    distributions::{Standard, uniform::SampleUniform},
    prelude::Distribution,
    rngs::StdRng,
    thread_rng,
};
use ff_ext::ExtensionField;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::Element;

pub fn random_vector(n: usize) -> Vec<Element> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| rng.gen_range(quantization::MIN..quantization::MAX))
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

pub fn random_vector_seed(n: usize, seed: Option<u64>) -> Vec<Element> {
    let seed = seed.unwrap_or(rand::random::<u64>()); // Use provided seed or default

    (0..n)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed + i as u64);
            rng.gen_range(quantization::MIN..quantization::MAX)
        })
        .collect_vec()
}

pub fn random_ranged_vector(n: usize, range: std::ops::Range<Element>) -> Vec<Element> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(range.clone())).collect_vec()
}
