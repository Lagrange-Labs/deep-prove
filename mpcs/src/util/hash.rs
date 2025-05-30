use ff_ext::{ExtensionField, PoseidonField};
use poseidon::poseidon_hash::PoseidonHash;

use transcript::Transcript;

pub use poseidon::digest::Digest;

pub fn write_digest_to_transcript<E: ExtensionField>(
    digest: &Digest<E::BaseField>,
    transcript: &mut impl Transcript<E>,
) {
    digest
        .0
        .iter()
        .for_each(|x| transcript.append_field_element(x));
}

pub fn hash_two_leaves_ext<E: ExtensionField>(a: &E, b: &E) -> Digest<E::BaseField> {
    let input = [a.as_bases(), b.as_bases()].concat();
    PoseidonHash::hash_or_noop(&input)
}

pub fn hash_two_leaves_base<E: ExtensionField>(
    a: &E::BaseField,
    b: &E::BaseField,
) -> Digest<E::BaseField> {
    PoseidonHash::hash_or_noop(&[*a, *b])
}

pub fn hash_two_leaves_batch_ext<E: ExtensionField>(a: &[E], b: &[E]) -> Digest<E::BaseField> {
    let a_m_to_1_hash = PoseidonHash::hash_or_noop_ext(a);
    let b_m_to_1_hash = PoseidonHash::hash_or_noop_ext(b);
    hash_two_digests::<E::BaseField>(&a_m_to_1_hash, &b_m_to_1_hash)
}

pub fn hash_two_leaves_batch_base<E: ExtensionField>(
    a: &[E::BaseField],
    b: &[E::BaseField],
) -> Digest<E::BaseField> {
    let a_m_to_1_hash = PoseidonHash::hash_or_noop(a);
    let b_m_to_1_hash = PoseidonHash::hash_or_noop(b);
    hash_two_digests::<E::BaseField>(&a_m_to_1_hash, &b_m_to_1_hash)
}

pub fn hash_two_digests<F: PoseidonField>(a: &Digest<F>, b: &Digest<F>) -> Digest<F> {
    PoseidonHash::two_to_one(a, b)
}
