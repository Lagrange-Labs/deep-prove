use ark_std::test_rng;
use criterion::{Criterion, criterion_group, criterion_main};
use ff::Field;
use ff_ext::{FromUniformBytes, GoldilocksExt2};
use mpcs::util::hash::{BlakeDigest, BlakeHasher, Digest, MerkleHasher, PoseidonHasher};
use p3_goldilocks::Goldilocks;
use poseidon::poseidon_hash::PoseidonHash;

fn random_ceno_goldy() -> Goldilocks {
    Goldilocks::random(&mut test_rng())
}
pub fn poseidon_hash(c: &mut Criterion) {
    let left = Digest(vec![random_ceno_goldy(); 4].try_into().unwrap());
    let right = Digest(vec![random_ceno_goldy(); 4].try_into().unwrap());
    c.bench_function("ceno hash 2 to 1", |bencher| {
        bencher.iter(|| {
            <PoseidonHasher as MerkleHasher<GoldilocksExt2>>::hash_two_digests(&left, &right)
        })
    });

    let values = (0..60).map(|_| random_ceno_goldy()).collect::<Vec<_>>();
    c.bench_function("ceno hash 60 to 1", |bencher| {
        bencher.iter(|| <PoseidonHasher as MerkleHasher<GoldilocksExt2>>::hash_bases(&values))
    });
}

pub fn blake_hash(c: &mut Criterion) {
    let left = BlakeDigest(blake3::hash(b"left"));
    let right = BlakeDigest(blake3::hash(b"right"));
    c.bench_function("ceno hash 2 to 1", |bencher| {
        bencher
            .iter(|| <BlakeHasher as MerkleHasher<GoldilocksExt2>>::hash_two_digests(&left, &right))
    });
    let values = (0..60).map(|_| random_ceno_goldy()).collect::<Vec<_>>();
    c.bench_function("ceno hash 60 to 1", |bencher| {
        bencher.iter(|| <BlakeHasher as MerkleHasher<GoldilocksExt2>>::hash_bases(&values))
    });
}

criterion_group!(benches, poseidon_hash, blake_hash);
criterion_main!(benches);
