use core::fmt::Debug;
use ff_ext::ExtensionField;
use p3_field::{FieldExtensionAlgebra, PrimeField};
use p3_symmetric::CryptographicPermutation;
use std::ops::{Deref, DerefMut};

pub use p3_challenger::*;

use ff_ext::PoseidonField;

/// this wrap a DuplexChallenger as inner field,
/// while expose some factory method to create default permutation object with defined constant
#[derive(Clone, Debug)]
pub struct DefaultChallenger<F, P>
where
    F: PrimeField,
    P: CryptographicPermutation<[F; 8]>,
{
    inner: DuplexChallenger<F, P, 8, 4>,
}

impl<F, P> DefaultChallenger<F, P>
where
    F: PrimeField,
    P: CryptographicPermutation<[F; 8]>,
{
    pub fn new(perm: P) -> Self {
        Self {
            inner: DuplexChallenger::<F, P, 8, 4>::new(perm),
        }
    }
}

impl<F: PoseidonField> DefaultChallenger<F, F::T>
where
    F::T: CryptographicPermutation<[F; 8]>,
{
    pub fn new_poseidon_default() -> Self {
        Self {
            inner: DuplexChallenger::<F, F::T, 8, 4>::new(F::get_perm()),
        }
    }
}

impl<F, P> Deref for DefaultChallenger<F, P>
where
    F: PrimeField,
    P: CryptographicPermutation<[F; 8]>,
{
    type Target = DuplexChallenger<F, P, 8, 4>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F, P> DerefMut for DefaultChallenger<F, P>
where
    F: PrimeField,
    P: CryptographicPermutation<[F; 8]>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub trait FieldChallengerExt<F: PoseidonField>: FieldChallenger<F> {
    fn observe_ext_slice<E: ExtensionField<BaseField = F>>(&mut self, exts: &[E]) {
        exts.iter()
            .for_each(|ext| self.observe_slice(ext.as_base_slice()));
    }

    fn sample_ext_vec<EF: FieldExtensionAlgebra<F>>(&mut self, n: usize) -> Vec<EF> {
        (0..n).map(|_| self.sample_ext_element()).collect()
    }
}

impl<F, P> CanObserve<F> for DefaultChallenger<F, P>
where
    F: PoseidonField,
    P: CryptographicPermutation<[F; 8]>,
{
    fn observe(&mut self, value: F) {
        self.inner.observe(value);
    }
}

impl<F, P> CanSampleBits<usize> for DefaultChallenger<F, P>
where
    F: PoseidonField,
    P: CryptographicPermutation<[F; 8]>,
{
    fn sample_bits(&mut self, _bits: usize) -> usize {
        todo!()
    }
}

impl<F, P> CanSample<F> for DefaultChallenger<F, P>
where
    F: PoseidonField,
    P: CryptographicPermutation<[F; 8]>,
{
    fn sample(&mut self) -> F {
        self.inner.sample()
    }
}

impl<F, P> FieldChallenger<F> for DefaultChallenger<F, P>
where
    F: PoseidonField,
    P: CryptographicPermutation<[F; 8]>,
{
}

impl<F, P> FieldChallengerExt<F> for DefaultChallenger<F, P>
where
    F: PoseidonField,
    P: CryptographicPermutation<[F; 8]>,
{
}
