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

pub trait VecInto<U> {
    fn vec_into(self) -> Vec<U>;
}

impl<T, U> VecInto<U> for Vec<T>
where
    T: Send + Sync + Into<U>,
    U: Send + Sync,
{
    fn vec_into(self) -> Vec<U> {
        self.into_par_iter().map(Into::into).collect()
    }
}

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

pub fn random_vector_seed<T>(n: usize, seed: Option<u64>) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let seed = seed.unwrap_or(rand::random::<u64>()); // Use provided seed or default

    (0..n)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed + i as u64);
            rng.gen::<T>()
        })
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
