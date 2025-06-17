use ff_ext::{ExtensionField, SmallField};
use poseidon::poseidon_hash::PoseidonHash;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt::Debug;

use transcript::Transcript;

pub use poseidon::digest::Digest;

/// Trait for hashing elements into a digest and doing merkle trees
pub trait MerkleHasher<E: ExtensionField>: Debug + Clone + Send + Sync + Default {
    type Digest: Clone + Debug + Default + Send + Sync + Serialize + DeserializeOwned + PartialEq;
    fn hash_bases(bases: &[E::BaseField]) -> Self::Digest;
    fn hash_two_digests(a: &Self::Digest, b: &Self::Digest) -> Self::Digest;
    fn digest_to_transcript(digest: &Self::Digest, transcript: &mut impl Transcript<E>);

    fn hash_elems(elems: &[E]) -> Self::Digest {
        let bases = elems
            .iter()
            .flat_map(|e| e.as_bases())
            .cloned()
            .collect::<Vec<_>>();
        Self::hash_bases(&bases)
    }
    fn hash_two_leaves(a: &E, b: &E) -> Self::Digest {
        let input = [a.as_bases(), b.as_bases()].concat();
        Self::hash_bases(input.as_slice())
    }
    fn hash_two_leaves_base(a: &E::BaseField, b: &E::BaseField) -> Self::Digest {
        Self::hash_bases(&[*a, *b])
    }
    fn hash_two_leaves_batch_base(a: &[E::BaseField], b: &[E::BaseField]) -> Self::Digest {
        let a_m_to_1_hash = Self::hash_bases(a);
        let b_m_to_1_hash = Self::hash_bases(b);
        Self::hash_two_digests(&a_m_to_1_hash, &b_m_to_1_hash)
    }
    fn hash_two_leaves_batch_ext(a: &[E], b: &[E]) -> Self::Digest {
        let a_m_to_1_hash = Self::hash_elems(a);
        let b_m_to_1_hash = Self::hash_elems(b);
        Self::hash_two_digests(&a_m_to_1_hash, &b_m_to_1_hash)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PoseidonHasher;

impl<E: ExtensionField> MerkleHasher<E> for PoseidonHasher
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type Digest = Digest<E::BaseField>;
    fn hash_bases(elems: &[E::BaseField]) -> Self::Digest {
        PoseidonHash::hash_or_noop(elems)
    }
    fn hash_two_digests(a: &Self::Digest, b: &Self::Digest) -> Self::Digest {
        PoseidonHash::two_to_one(a, b)
    }
    fn digest_to_transcript(digest: &Self::Digest, transcript: &mut impl Transcript<E>) {
        digest
            .0
            .iter()
            .for_each(|x| transcript.append_field_element(x));
    }
}

#[derive(Debug, Clone, Default)]
pub struct BlakeHasher;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BlakeDigest(pub blake3::Hash);

impl Default for BlakeDigest {
    fn default() -> Self {
        BlakeDigest(blake3::Hash::from_bytes([0; 32]))
    }
}

impl<E: ExtensionField> MerkleHasher<E> for BlakeHasher {
    type Digest = BlakeDigest;
    fn hash_bases(bases: &[<E as ExtensionField>::BaseField]) -> Self::Digest {
        let mut hasher = blake3::Hasher::new();
        for elem in bases {
            hasher.update(&elem.to_canonical_u64().to_le_bytes());
        }
        BlakeDigest(hasher.finalize())
    }
    fn hash_two_digests(a: &Self::Digest, b: &Self::Digest) -> Self::Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&a.0.as_bytes()[..]);
        hasher.update(&b.0.as_bytes()[..]);
        BlakeDigest(hasher.finalize())
    }
    fn digest_to_transcript(digest: &Self::Digest, transcript: &mut impl Transcript<E>) {
        transcript.append_message(&digest.0.as_bytes()[..]);
    }
}
