use ark_std::rand::{
    Rng,
    distributions::{Standard, uniform::SampleUniform},
    prelude::Distribution,
    thread_rng,
};
use ff_ext::ExtensionField;
use itertools::Itertools;

pub fn random_vector<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen::<T>()).collect_vec()
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

pub fn random_ranged_vector<T>(n: usize, range: std::ops::Range<T>) -> Vec<T>
where
    Standard: Distribution<T>,
    T: SampleUniform + PartialOrd + Clone,
{
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(range.clone())).collect_vec()
}
