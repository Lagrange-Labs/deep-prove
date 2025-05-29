use ark_std::test_rng;
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ff_ext::FromUniformBytes;
use p3_field::FieldAlgebra;
use p3_goldilocks::Goldilocks;
use plonky2::{
    field::{goldilocks_field::GoldilocksField, types::Sample},
    hash::{
        hash_types::HashOut,
        hashing::PlonkyPermutation,
        poseidon::{PoseidonHash as PlonkyPoseidonHash, PoseidonPermutation},
    },
    plonk::config::Hasher,
};
use poseidon::{challenger::DefaultChallenger, digest::Digest, poseidon_hash::PoseidonHash};

fn random_plonky_2_goldy() -> GoldilocksField {
    GoldilocksField::rand()
}

fn random_ceno_goldy() -> Goldilocks {
    Goldilocks::random(&mut test_rng())
}

fn random_ceno_hash() -> Digest<Goldilocks> {
    Digest(
        vec![Goldilocks::random(&mut test_rng()); 4]
            .try_into()
            .unwrap(),
    )
}

fn plonky_hash_single(a: GoldilocksField) {
    let _result = black_box(PlonkyPoseidonHash::hash_or_noop(&[a]));
}

fn ceno_hash_single(a: Goldilocks) {
    let _result = black_box(PoseidonHash::<Goldilocks>::hash_or_noop(&[a]));
}

fn plonky_hash_2_to_1(left: HashOut<GoldilocksField>, right: HashOut<GoldilocksField>) {
    let _result = black_box(PlonkyPoseidonHash::two_to_one(left, right));
}

fn ceno_hash_2_to_1(left: &Digest<Goldilocks>, right: &Digest<Goldilocks>) {
    let _result = black_box(PoseidonHash::<Goldilocks>::two_to_one(left, right));
}

fn plonky_hash_many_to_1(values: &[GoldilocksField]) {
    let _result = black_box(PlonkyPoseidonHash::hash_or_noop(values));
}

fn ceno_hash_many_to_1(values: &[Goldilocks]) {
    let _result = black_box(PoseidonHash::<Goldilocks>::hash_or_noop(values));
}

pub fn hashing_benchmark(c: &mut Criterion) {
    c.bench_function("plonky hash single", |bencher| {
        bencher.iter_batched(
            random_plonky_2_goldy,
            plonky_hash_single,
            BatchSize::SmallInput,
        )
    });

    c.bench_function("plonky hash 2 to 1", |bencher| {
        bencher.iter_batched(
            || {
                (
                    HashOut::<GoldilocksField>::rand(),
                    HashOut::<GoldilocksField>::rand(),
                )
            },
            |(left, right)| plonky_hash_2_to_1(left, right),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("plonky hash 60 to 1", |bencher| {
        bencher.iter_batched(
            || GoldilocksField::rand_vec(60),
            |sixty_elems| plonky_hash_many_to_1(sixty_elems.as_slice()),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("ceno hash single", |bencher| {
        bencher.iter_batched(random_ceno_goldy, ceno_hash_single, BatchSize::SmallInput)
    });

    c.bench_function("ceno hash 2 to 1", |bencher| {
        bencher.iter_batched(
            || (random_ceno_hash(), random_ceno_hash()),
            |(left, right)| ceno_hash_2_to_1(&left, &right),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("ceno hash 60 to 1", |bencher| {
        bencher.iter_batched(
            || {
                (0..60)
                    .map(|_| Goldilocks::random(&mut test_rng()))
                    .collect::<Vec<_>>()
            },
            |values| ceno_hash_many_to_1(values.as_slice()),
            BatchSize::SmallInput,
        )
    });
}

use p3_symmetric::Permutation;

// bench permutation
pub fn permutation_benchmark(c: &mut Criterion) {
    let mut plonky_permutation = PoseidonPermutation::new(core::iter::repeat(GoldilocksField(0)));
    let ceno_challenger = DefaultChallenger::<Goldilocks, _>::new_poseidon_default();

    c.bench_function("plonky permute", |bencher| {
        bencher.iter(|| plonky_permutation.permute())
    });

    c.bench_function("ceno permute", |bencher| {
        bencher.iter(|| {
            let mut state = [Goldilocks::ZERO; 8];
            ceno_challenger.permutation.permute_mut(&mut state);
        })
    });
}

criterion_group!(benches, permutation_benchmark, hashing_benchmark);
criterion_main!(benches);
