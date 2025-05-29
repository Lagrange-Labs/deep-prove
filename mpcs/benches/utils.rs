use std::time::Duration;

use criterion::*;

use ff_ext::{FromUniformBytes, GoldilocksExt2};
use mpcs::{one_level_eval_hc, one_level_interp_hc};

use rand::rngs::OsRng;

type E = GoldilocksExt2;

const NUM_SAMPLES: usize = 10;
const NUM_VARS_START: usize = 20;
const NUM_VARS_END: usize = 20;

fn bench_eval_hc(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_hc".to_string());
    group.sample_size(NUM_SAMPLES);
    // Challenge is over extension field, poly over the base field
    for num_vars in NUM_VARS_START..=NUM_VARS_END {
        let poly: Vec<_> = (0..1 << num_vars).map(|_| E::random(&mut OsRng)).collect();
        let challenge = E::random(&mut OsRng);

        group.bench_function(BenchmarkId::new("eval_hc", format!("{}", num_vars)), |b| {
            b.iter_batched(
                || poly.clone(),
                |mut coeffs| {
                    // Switch to coefficient form
                    one_level_eval_hc(&mut coeffs, challenge);
                },
                BatchSize::SmallInput,
            )
        });
    }
}

fn bench_interp_hc(c: &mut Criterion) {
    let mut group = c.benchmark_group("interp_hc".to_string());
    group.sample_size(NUM_SAMPLES);
    // Challenge is over extension field, poly over the base field
    for num_vars in NUM_VARS_START..=NUM_VARS_END {
        let poly: Vec<_> = (0..1 << num_vars).map(|_| E::random(&mut OsRng)).collect();

        group.bench_function(
            BenchmarkId::new("interp_hc", format!("{}", num_vars)),
            |b| {
                b.iter_batched(
                    || poly.clone(),
                    |mut coeffs| {
                        // Switch to coefficient form
                        one_level_interp_hc(&mut coeffs);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group! {
  name = bench_utils;
  config = Criterion::default().warm_up_time(Duration::from_millis(3000));
  targets = bench_eval_hc, bench_interp_hc,
}

criterion_main!(bench_utils);
