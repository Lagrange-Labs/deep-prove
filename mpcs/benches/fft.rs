use std::time::Duration;

use criterion::*;
use ff_ext::{FromUniformBytes, GoldilocksExt2};

use itertools::Itertools;
use mpcs::{coset_fft, fft_root_table};

use multilinear_extensions::mle::DenseMultilinearExtension;
use p3_field::Field;
use p3_goldilocks::Goldilocks;
use rand::{SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
type E = GoldilocksExt2;

const NUM_SAMPLES: usize = 10;
const NUM_VARS_START: usize = 15;
const NUM_VARS_END: usize = 20;
const BATCH_SIZE_LOG_START: usize = 3;
const BATCH_SIZE_LOG_END: usize = 6;

fn bench_fft(c: &mut Criterion, is_base: bool) {
    let mut group = c.benchmark_group(format!(
        "fft_{}",
        if is_base { "base" } else { "extension" }
    ));
    group.sample_size(NUM_SAMPLES);
    // Challenge is over extension field, poly over the base field
    for num_vars in NUM_VARS_START..=NUM_VARS_END {
        let root_table = fft_root_table(num_vars);
        for batch_size_log in BATCH_SIZE_LOG_START..=BATCH_SIZE_LOG_END {
            let batch_size = 1 << batch_size_log;
            let rng = ChaCha8Rng::from_seed([0u8; 32]);
            let mut polys = (0..batch_size)
                .map(|_| {
                    if is_base {
                        DenseMultilinearExtension::random(num_vars, &mut rng.clone())
                    } else {
                        DenseMultilinearExtension::from_evaluations_ext_vec(
                            num_vars,
                            (0..1 << num_vars).map(|_| E::random(&mut OsRng)).collect(),
                        )
                    }
                })
                .collect_vec();

            group.bench_function(
                BenchmarkId::new("batch_encode", format!("{}-{}", num_vars, batch_size)),
                |b| {
                    b.iter(|| {
                        polys.par_iter_mut().for_each(|poly| {
                            coset_fft::<GoldilocksExt2>(
                                &mut poly.evaluations,
                                Goldilocks::GENERATOR,
                                0,
                                &root_table,
                            );
                        });
                    })
                },
            );
        }
    }
}

fn bench_fft_goldilocks_2(c: &mut Criterion) {
    bench_fft(c, false);
}

fn bench_fft_base(c: &mut Criterion) {
    bench_fft(c, true);
}

criterion_group! {
  name = bench_basefold;
  config = Criterion::default().warm_up_time(Duration::from_millis(3000));
  targets = bench_fft_base, bench_fft_goldilocks_2
}

criterion_main!(bench_basefold);
