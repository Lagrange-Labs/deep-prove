#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]

use std::time::Duration;

use criterion::*;

use ff_ext::GoldilocksExt2;
use gkr::gadgets::keccak256::{keccak256_circuit, prove_keccak256, verify_keccak256};
use multilinear_extensions::util::max_usable_threads;

cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = keccak256;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
      targets = bench_keccak256
    }
  } else {
    criterion_group! {
      name = keccak256;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_keccak256
    }
  }
}

criterion_main!(keccak256);

const NUM_SAMPLES: usize = 10;

fn bench_keccak256(c: &mut Criterion) {
    println!(
        "#layers: {}",
        keccak256_circuit::<GoldilocksExt2>().layers.len()
    );

    let max_thread_id = max_usable_threads();

    let circuit = keccak256_circuit::<GoldilocksExt2>();

    let Some((proof, witness)) = prove_keccak256(1, &circuit, 1) else {
        return;
    };
    assert!(verify_keccak256(1, &witness.witness_out_ref()[0], proof, &circuit).is_ok());

    for log2_n in 0..10 {
        // expand more input size once runtime is acceptable
        let mut group = c.benchmark_group(format!("keccak256_log2_{}", log2_n));
        group.sample_size(NUM_SAMPLES);

        // Benchmark the proving time
        group.bench_function(
            BenchmarkId::new("prove_keccak256", format!("keccak256_log2_{}", log2_n)),
            |b| {
                b.iter(|| {
                    assert!(
                        prove_keccak256(log2_n, &circuit, (1 << log2_n).min(max_thread_id),)
                            .is_some()
                    );
                });
            },
        );

        group.finish();
    }
}
