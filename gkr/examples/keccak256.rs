#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]

use std::env;

use ff_ext::{ExtensionField, GoldilocksExt2};
use gkr::{
    gadgets::keccak256::{keccak256_circuit, prove_keccak256, verify_keccak256},
    structs::CircuitWitness,
};
use itertools::{Itertools, izip};
use multilinear_extensions::mle::IntoMLE;
use p3_field::FieldAlgebra;
use tracing_flame::FlameLayer;
use tracing_subscriber::{EnvFilter, Registry, fmt, layer::SubscriberExt};

fn main() {
    println!(
        "#layers: {}",
        keccak256_circuit::<GoldilocksExt2>().layers.len()
    );

    #[allow(unused_mut)]
    let mut max_thread_id: usize = env::var("RAYON_NUM_THREADS")
        .map(|v| str::parse::<usize>(&v).unwrap_or(1))
        .unwrap();

    let circuit = keccak256_circuit::<GoldilocksExt2>();
    // Sanity-check
    {
        let all_zero = vec![
            vec![<GoldilocksExt2 as ExtensionField>::BaseField::ZERO; 25 * 64],
            vec![<GoldilocksExt2 as ExtensionField>::BaseField::ZERO; 17 * 64],
        ]
        .into_iter()
        .map(|wit_in| wit_in.into_mle())
        .collect();
        let all_one = vec![
            vec![<GoldilocksExt2 as ExtensionField>::BaseField::ONE; 25 * 64],
            vec![<GoldilocksExt2 as ExtensionField>::BaseField::ZERO; 17 * 64],
        ]
        .into_iter()
        .map(|wit_in| wit_in.into_mle())
        .collect();
        let mut witness = CircuitWitness::new(&circuit, Vec::new());
        witness.add_instance(&circuit, all_zero);
        witness.add_instance(&circuit, all_one);

        izip!(
            witness.witness_out_ref()[0]
                .get_base_field_vec()
                .chunks(256),
            [[0; 25], [u64::MAX; 25]]
        )
        .for_each(|(wire_out, state)| {
            let output = wire_out[..256]
                .chunks_exact(64)
                .map(|bits| {
                    bits.iter().fold(0, |acc, bit| {
                        (acc << 1)
                            + (*bit == <GoldilocksExt2 as ExtensionField>::BaseField::ONE) as u64
                    })
                })
                .collect_vec();
            let expected = {
                let mut state = state;
                tiny_keccak::keccakf(&mut state);
                state[0..4].to_vec()
            };
            assert_eq!(output, expected)
        });
    }

    let (flame_layer, _guard) = FlameLayer::with_file("./tracing.folded").unwrap();
    let subscriber = Registry::default()
        .with(
            fmt::layer()
                .compact()
                .with_thread_ids(false)
                .with_thread_names(false),
        )
        .with(EnvFilter::from_default_env())
        .with(flame_layer.with_threads_collapsed(true));
    tracing::subscriber::set_global_default(subscriber).unwrap();

    for log2_n in 0..12 {
        let Some((proof, witness)) =
            prove_keccak256::<GoldilocksExt2>(log2_n, &circuit, (1 << log2_n).min(max_thread_id))
        else {
            return;
        };
        assert!(verify_keccak256(log2_n, &witness.witness_out_ref()[0], proof, &circuit).is_ok());
    }
}
