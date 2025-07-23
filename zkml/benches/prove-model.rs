use criterion::{Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use zkml::{
    Context, Element, FloatOnnxLoader, Prover, default_transcript,
    inputs::Input,
    model::Model,
    quantization::{AbsoluteMax, ModelMetadata},
    verify,
};

type F = GoldilocksExt2;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
type Pcs<E> = Basefold<E, BasefoldRSParams<Hasher>>;

// Choose transcript implementation at compile time
#[cfg(feature = "blake")]
type Transcript = BlakeTranscript;

#[cfg(not(feature = "blake"))]
type Transcript = transcript::basic::BasicTranscript<F>;

fn new_transcript() -> Transcript {
    #[cfg(feature = "blake")]
    {
        use transcript::blake::BlakeTranscript;
        BlakeTranscript::new(b"bench")
    }
    #[cfg(not(feature = "blake"))]
    {
        default_transcript()
    }
}

fn parse_model(model_data: &[u8]) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    FloatOnnxLoader::from_bytes_with_scaling_strategy(model_data, AbsoluteMax::new())
        .with_keep_float(true)
        .build()
}

fn run_model<T: std::io::Read>(model_data: &[u8], inputs: T) {
    let run_inputs = Input::from_reader(inputs).expect("failed to load inputs");
    let (model, md) = parse_model(model_data).expect("failed to parse model");
    let inputs = run_inputs.to_elements(&md);

    let ctx = Some(
        Context::<F, Pcs<F>>::generate(&model, None, None).expect("unable to generate context"),
    );

    for (i, input) in inputs.into_iter().enumerate() {
        let input_tensor = model
            .load_input_flat(vec![input])
            .expect("failed to call load_input_flat on the model");

        let trace = model
            .run(&input_tensor)
            .expect(&format!("input #{i} failed"));

        let mut prover_transcript = new_transcript();
        let prover = Prover::<_, _, _>::new(ctx.as_ref().unwrap(), &mut prover_transcript);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript = new_transcript();
        verify::<_, _, _>(
            ctx.as_ref().unwrap().clone(),
            proof,
            io,
            &mut verifier_transcript,
        )
        .expect("invalid proof");
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-models");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    group.bench_function("mlp", |b| {
        b.iter(|| {
            run_model(
                include_bytes!("../assets/scripts/MLP/mlp-iris-01.onnx"),
                zstd::Decoder::new(&include_bytes!("../assets/scripts/MLP/input.json.zst")[..])
                    .expect("failed to parse zstd"),
            )
        })
    });

    group.bench_function("cnn", |b| {
        b.iter(|| {
            run_model(
                include_bytes!("../assets/scripts/CNN/cnn-cifar-01.onnx"),
                zstd::Decoder::new(&include_bytes!("../assets/scripts/CNN/input.json.zst")[..])
                    .expect("failed to parse zstd"),
            )
        })
    });

    // NOTE: disabling covid model, as it is /very/ memory-intensive
    // group.bench_function("covid", |b| {
    //     b.iter(|| {
    //         run_model(
    //             include_bytes!("../assets/scripts/covid/cnn-covid.onnx"),
    //             zstd::Decoder::new(&include_bytes!("../assets/scripts/covid/input.json.zst")[..])
    //                 .expect("failed to parse zstd"),
    //         )
    //     })
    // });

    group.finish();
}

// NOTE: XXX: when running, limit RAYON_NUM_THREADS to e.g. 2 to avoid high
// concurrency resulting in measure noise.
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
