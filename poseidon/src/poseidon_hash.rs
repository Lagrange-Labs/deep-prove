use std::marker::PhantomData;

use crate::{
    challenger::{DefaultChallenger, FieldChallengerExt},
    constants::DIGEST_WIDTH,
    digest::Digest,
};
use ff_ext::{ExtensionField, PoseidonField};
use p3_challenger::{CanObserve, CanSample};

pub struct PoseidonHash<F> {
    _phantom: PhantomData<F>,
}

impl<F: PoseidonField> PoseidonHash<F> {}

impl<F: PoseidonField> PoseidonHash<F> {
    pub fn two_to_one(left: &Digest<F>, right: &Digest<F>) -> Digest<F> {
        compress::<F>(left, right)
    }

    pub fn hash_or_noop(inputs: &[F]) -> Digest<F> {
        if inputs.len() <= DIGEST_WIDTH {
            Digest::from_partial(inputs)
        } else {
            hash_n_to_hash_no_pad::<F>(inputs)
        }
    }

    pub fn hash_or_noop_ext<E: ExtensionField<BaseField = F>>(inputs: &[E]) -> Digest<F> {
        if inputs.len() * E::DEGREE <= DIGEST_WIDTH {
            Digest::from_iter(inputs.iter().flat_map(|v| v.as_bases()))
        } else {
            hash_n_to_hash_no_pad_ext(inputs)
        }
    }
}

pub fn hash_n_to_m_no_pad<F: PoseidonField>(inputs: &[F], num_outputs: usize) -> Vec<F> {
    let mut challenger = DefaultChallenger::<F, F::T>::new_poseidon_default();
    challenger.observe_slice(inputs);
    challenger.sample_vec(num_outputs)
}

pub fn hash_n_to_m_no_pad_ext<F: PoseidonField, E: ExtensionField<BaseField = F>>(
    inputs: &[E],
    num_outputs: usize,
) -> Vec<F> {
    let mut challenger = DefaultChallenger::<F, F::T>::new_poseidon_default();
    challenger.observe_ext_slice(inputs);
    challenger.sample_vec(num_outputs)
}

pub fn hash_n_to_hash_no_pad<F: PoseidonField>(inputs: &[F]) -> Digest<F> {
    hash_n_to_m_no_pad(inputs, DIGEST_WIDTH).try_into().unwrap()
}

pub fn hash_n_to_hash_no_pad_ext<F: PoseidonField, E: ExtensionField<BaseField = F>>(
    inputs: &[E],
) -> Digest<F> {
    hash_n_to_m_no_pad_ext(inputs, DIGEST_WIDTH)
        .try_into()
        .unwrap()
}

pub fn compress<F: PoseidonField>(x: &Digest<F>, y: &Digest<F>) -> Digest<F> {
    let mut challenger = DefaultChallenger::<F, F::T>::new_poseidon_default();
    challenger.observe_slice(x.elements());
    challenger.observe_slice(y.elements());
    Digest(challenger.sample_array::<DIGEST_WIDTH>())
}
