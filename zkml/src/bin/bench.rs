use zkml::model::Model;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::BufReader,
    path::Path,
    time,
};
use zkml::quantization::{AbsoluteMax, InferenceObserver, ModelMetadata, ScalingStrategy};

use anyhow::{Context as CC, ensure};
use clap::Parser;
use csv::WriterBuilder;
use goldilocks::GoldilocksExt2;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt};
use zkml::{FloatOnnxLoader, quantization::ScalingFactor};

use serde::{Deserialize, Serialize};
use zkml::{
    Context, Element, IO, Prover, argmax, default_transcript, quantization::TensorFielder, verify,
};

use rmp_serde::encode::to_vec_named;

type F = GoldilocksExt2;

#[derive(Parser, Debug)]
struct Args {
    /// onxx file to load
    #[arg(short, long)]
    onnx: String,
    /// input / output vector file in JSON. Format "{ input_data: [a,b,c], output_data: [c,d] }"
    #[arg(short, long)]
    io: String,
    /// File where to write the benchmarks
    #[arg(short,long,default_value_t = {"bench.csv".to_string()})]
    bench: String,
    /// Number of samples to process
    #[arg(short, long, default_value_t = 30)]
    num_samples: usize,
    /// Skip proving and verifying, only run inference and check accuracy
    #[arg(short, long, default_value_t = false)]
    skip_proving: bool,

    /// Quantization strategy to use
    #[arg(short, long, default_value_t = {"inference".to_string()})]
    quantization: String,
}

pub fn main() -> anyhow::Result<()> {
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
    let args = Args::parse();
    run(args).context("error running bench:")?;

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct InputJSON {
    input_data: Vec<Vec<f32>>,
    output_data: Vec<Vec<f32>>,
    pytorch_output: Vec<Vec<f32>>,
}

impl InputJSON {
    /// Returns (input,output) from the path
    pub fn from(path: &str, num_samples: usize) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut u: Self = serde_json::from_reader(reader)?;
        u.truncate(num_samples);
        u.validate()?;
        Ok(u)
    }

    fn truncate(&mut self, num_samples: usize) {
        self.input_data.truncate(num_samples);
        self.output_data.truncate(num_samples);
        self.pytorch_output.truncate(num_samples);
    }
    // poor's man validation
    fn validate(&self) -> anyhow::Result<()> {
        let rrange = zkml::quantization::MIN_FLOAT..=zkml::quantization::MAX_FLOAT;
        ensure!(self.input_data.len() > 0);
        let input_isreal = self
            .input_data
            .iter()
            .all(|v| v.iter().all(|&x| rrange.contains(&x)));
        assert_eq!(self.input_data.len(), self.output_data.len());
        assert_eq!(self.input_data.len(), self.pytorch_output.len());
        ensure!(
            input_isreal,
            "can only support real model so far (input at least)"
        );
        Ok(())
    }
    fn to_elements(self, md: &ModelMetadata) -> (Vec<Vec<Element>>, Vec<Vec<Element>>) {
        let inputs = self
            .input_data
            .into_iter()
            .map(|input| input.into_iter().map(|e| md.input.quantize(&e)).collect())
            .collect();
        let output_sf = md.output_scaling_factor();
        let outputs = self
            .output_data
            .into_iter()
            .map(|output| output.into_iter().map(|e| output_sf.quantize(&e)).collect())
            .collect();
        (inputs, outputs)
    }

    /// Computes the accuracy of pytorch outputs against the expected outputs
    pub fn compute_pytorch_accuracy(&self) -> f32 {
        let mut accuracies = Vec::new();

        for (i, (expected, pytorch_out)) in self
            .output_data
            .iter()
            .zip(self.pytorch_output.iter())
            .enumerate()
        {
            let accuracy = argmax_compare(expected, pytorch_out);
            accuracies.push(accuracy);
            info!(
                "PyTorch Run {}/{}: Accuracy: {}",
                i + 1,
                self.output_data.len(),
                if accuracy > 0 { "correct" } else { "incorrect" }
            );
        }

        let avg_accuracy = calculate_average_accuracy(&accuracies);
        avg_accuracy
    }
}

const CSV_SETUP: &str = "setup (ms)";
const CSV_INFERENCE: &str = "inference (ms)";
const CSV_PROVING: &str = "proving (ms)";
const CSV_VERIFYING: &str = "verifying (ms)";
const CSV_ACCURACY: &str = "accuracy (bool)";
const CSV_PROOF_SIZE: &str = "proof size (KB)";

/// Runs the model in float format and returns the average accuracy
fn run_float_model(raw_inputs: &InputJSON, model: &Model<f32>) -> f32 {
    let mut accuracies = Vec::new();
    info!("[+] Running model in float format");

    for (i, (input, expected)) in raw_inputs.input_data.iter()
        .zip(raw_inputs.output_data.iter())
        .enumerate() 
    {
        // Run the model in float mode
        let output = model.run_float(input.clone());
        let accuracy = argmax_compare(expected, &output.get_data());
        accuracies.push(accuracy);
        info!(
            "Float Run {}/{}: Accuracy: {}",
            i + 1,
            raw_inputs.input_data.len(),
            if accuracy > 0 { "correct" } else { "incorrect" }
        );
    }

    calculate_average_accuracy(&accuracies)
}

fn run(args: Args) -> anyhow::Result<()> {
    info!("[+] Reading raw input/output from {}", args.io);
    let raw_inputs = InputJSON::from(&args.io, args.num_samples).context("loading input:")?;
    let strategy = quantization_strategy_from(&args, &raw_inputs);
    let strat_name = strategy.name().to_string();
    info!("[+] Reading onnx model");
    let (model, md) = FloatOnnxLoader::new(&args.onnx)
        .with_scaling_strategy(strategy)
        .with_keep_float(true)
        .build()?;
    info!("[+] Model loaded");
    model.describe();

    // Get float accuracy if float model is available
    let float_accuracy = if let Some(ref float_model) = md.float_model {
        info!("[+] Running float model");
        run_float_model(&raw_inputs, float_model)
    } else {
        info!("[!] No float model available");
        0.0
    };

    info!("[+] Computing PyTorch accuracy");
    let num_samples = raw_inputs.output_data.len();
    let pytorch_accuracy = raw_inputs.compute_pytorch_accuracy();
    info!("[+] Quantizing inputs with strategy: {}", strat_name);
    let (inputs, given_outputs) = raw_inputs.to_elements(&md);

    // Generate context once and measure the time
    info!("[+] Generating context for proving");
    let now = time::Instant::now();
    let ctx = Context::<F>::generate(&model, None).expect("unable to generate context");
    let setup_time = now.elapsed().as_millis();
    info!("STEP: {} took {}ms", CSV_SETUP, setup_time);

    // Collect accuracies for final average
    let mut accuracies = Vec::new();

    for (i, (input, given_output)) in inputs
        .into_iter()
        .zip(given_outputs.into_iter())
        .enumerate()
    {
        let mut bencher = CSVBencher::from_headers(vec![
            CSV_SETUP,
            CSV_INFERENCE,
            CSV_PROVING,
            CSV_VERIFYING,
            CSV_PROOF_SIZE,
            CSV_ACCURACY,
        ]);

        // Store the setup time in the bencher (without re-running setup)
        bencher.set(CSV_SETUP, setup_time);

        let input_tensor = model.load_input_flat(input);

        info!("[+] Running inference");
        let trace = bencher.r(CSV_INFERENCE, || model.run(input_tensor.clone()));
        let output = trace.final_output().clone();
        let accuracy = argmax_compare(&given_output, &output.get_data().to_vec());
        accuracies.push(accuracy);
        bencher.set(CSV_ACCURACY, accuracy);
        // Log per-run accuracy
        info!(
            "Run {}/{}: Accuracy: {}",
            i + 1,
            args.num_samples,
            if accuracy > 0 { "correct" } else { "incorrect" }
        );
        if args.skip_proving {
            info!("[+] Skipping proving");
            bencher.set(CSV_PROVING, 0);
            bencher.set(CSV_VERIFYING, 0);
            bencher.set(CSV_PROOF_SIZE, "0.000");
            continue;
        }
        info!("[+] Running prover");
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _>::new(&ctx, &mut prover_transcript);
        let proof = bencher.r(CSV_PROVING, move || {
            prover.prove(trace).expect("unable to generate proof")
        });

        // Serialize proof using MessagePack and calculate size in KB
        let proof_bytes = to_vec_named(&proof)?;
        let proof_size_kb = proof_bytes.len() as f64 / 1024.0;
        bencher.set(CSV_PROOF_SIZE, format!("{:.3}", proof_size_kb));

        info!("[+] Running verifier");
        let mut verifier_transcript = default_transcript();
        let io = IO::new(input_tensor.to_fields(), output.to_fields());
        bencher.r(CSV_VERIFYING, || {
            verify::<_, _>(ctx.clone(), proof, io, &mut verifier_transcript).expect("invalid proof")
        });
        info!("[+] Verify proof: valid");

        bencher.flush(&args.bench)?;
        info!("[+] Benchmark results appended to {}", args.bench);
    }

    // Calculate and display average accuracy
    let avg_accuracy = calculate_average_accuracy(&accuracies);

    // Single final accuracy comparison
    info!(
        "Final accuracy comparison across {} runs:",
        num_samples
    );
    info!("ZKML float model accuracy: {:.2}%", float_accuracy * 100.0);
    info!("ZKML quantized model accuracy: {:.2}%", avg_accuracy * 100.0);
    info!("PyTorch accuracy: {:.2}%", pytorch_accuracy * 100.0);

    Ok(())
}

fn argmax_compare<A: PartialOrd, B: PartialOrd>(
    given_output: &[A],
    computed_output: &[B],
) -> usize {
    let compare_size = std::cmp::min(given_output.len(), computed_output.len());
    let a_max = argmax(&given_output[..compare_size]);
    let b_max = argmax(&computed_output[..compare_size]);
    info!("Accuracy: {}", if a_max == b_max { 1 } else { 0 });
    if a_max == b_max { 1 } else { 0 }
}

struct CSVBencher {
    data: HashMap<String, String>,
    headers: Vec<String>,
}

impl CSVBencher {
    pub fn from_headers<S: IntoIterator<Item = T>, T: Into<String>>(headers: S) -> Self {
        let strings: Vec<String> = headers.into_iter().map(Into::into).collect();
        Self {
            data: Default::default(),
            headers: strings,
        }
    }

    pub fn r<A, F: FnOnce() -> A>(&mut self, column: &str, f: F) -> A {
        self.check(column);
        let now = time::Instant::now();
        let output = f();
        let elapsed = now.elapsed().as_millis();
        info!("STEP: {} took {}ms", column, elapsed);
        self.data.insert(column.to_string(), elapsed.to_string());
        output
    }

    fn check(&self, column: &str) {
        if self.data.contains_key(column) {
            panic!(
                "CSVBencher only flushes one row at a time for now (key already registered: {})",
                column
            );
        }
        if !self.headers.contains(&column.to_string()) {
            panic!("column {} non existing", column);
        }
    }

    pub fn set<I: ToString>(&mut self, column: &str, data: I) {
        self.check(column);
        self.data.insert(column.to_string(), data.to_string());
    }

    fn flush(&self, fname: &str) -> anyhow::Result<()> {
        let file_exists = Path::new(fname).exists();
        let file = OpenOptions::new()
            .create(true)
            .append(file_exists)
            .write(true)
            .open(fname)?;
        let mut writer = WriterBuilder::new()
            .has_headers(!file_exists)
            .from_writer(file);

        let values: Vec<_> = self
            .headers
            .iter()
            .map(|k| self.data[k].to_string())
            .collect();

        if !file_exists {
            writer.write_record(&self.headers)?;
        }

        writer.write_record(&values)?;
        writer.flush()?;
        Ok(())
    }
}

fn calculate_average_accuracy(accuracies: &[usize]) -> f32 {
    if accuracies.is_empty() {
        return 0.0;
    }
    let sum: usize = accuracies.iter().sum();
    sum as f32 / accuracies.len() as f32
}

fn quantization_strategy_from(args: &Args, inputs: &InputJSON) -> Box<dyn ScalingStrategy> {
    match args.quantization.as_ref() {
        "inference" => Box::new(InferenceObserver::new_with_representative_input(
            inputs.input_data.clone(),
        )),
        //"maxabs" => Box::new(AbsoluteMax::new_with_representative_input(inputs.input_data[0].clone())),
        _ => panic!("Unsupported quantization strategy: {}", args.quantization),
    }
}
