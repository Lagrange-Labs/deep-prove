use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::BufReader,
    path::Path,
    time,
};
use timed_core::Output;
use zkml::{
    model::Model,
    quantization::{AbsoluteMax, InferenceObserver, ModelMetadata},
};

use anyhow::{Context as CC, Result, ensure};
use clap::Parser;
use csv::WriterBuilder;
use goldilocks::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use tracing::{debug, info};
use tracing_subscriber::{EnvFilter, fmt};
use zkml::FloatOnnxLoader;

use serde::{Deserialize, Serialize};
use zkml::{Context, Element, Prover, argmax, default_transcript, verify};

use rmp_serde::encode::to_vec_named;

type F = GoldilocksExt2;
type Pcs<E> = Basefold<E, BasefoldRSParams>;

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

    /// Specific input indices to run inference on (comma-separated list)
    #[arg(long, value_delimiter = ',', value_parser = parse_usize)]
    run_indices: Option<Vec<usize>>,

    /// Specific indices to use for calibration
    #[arg(long, value_delimiter = ',', value_parser = parse_usize)]
    calibration_indices: Option<Vec<usize>>,
}

// Helper function to parse a single usize
fn parse_usize(s: &str) -> Result<usize, String> {
    s.trim().parse().map_err(|e| format!("Invalid index: {e}"))
}

pub fn main() -> anyhow::Result<()> {
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
    timed_core::set_output(Output::CSV("deepprove.csv".to_string()));
    let args = Args::parse();
    run(args).context("error running bench:")?;

    Ok(())
}

#[derive(Clone, Serialize, Deserialize)]
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

    fn filter(&self, indices: Option<&Vec<usize>>) -> Self {
        if let Some(indices) = indices {
            assert!(
                indices.iter().all(|i| *i < self.input_data.len()),
                "Index {} is out of range (max: {})",
                indices.iter().max().unwrap(),
                self.input_data.len() - 1
            );
            let input_data = indices
                .iter()
                .map(|i| self.input_data[*i].clone())
                .collect();
            let output_data = indices
                .iter()
                .map(|i| self.output_data[*i].clone())
                .collect();
            let pytorch_output = indices
                .iter()
                .map(|i| self.pytorch_output[*i].clone())
                .collect();
            Self {
                input_data,
                output_data,
                pytorch_output,
            }
        } else {
            self.clone()
        }
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
        let input_sf = md.input.first().unwrap();
        let inputs = self
            .input_data
            .into_iter()
            .map(|input| input.into_iter().map(|e| input_sf.quantize(&e)).collect())
            .collect();
        let output_sf = *md.output_scaling_factor().first().unwrap();
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
            debug!(
                "PyTorch Run {}/{}: \n\t truth {:?} \n\t pytorch {:?}\n\t-> Accuracy: {}",
                i + 1,
                self.output_data.len(),
                expected,
                pytorch_out,
                if accuracy > 0 { "correct" } else { "incorrect" }
            );
        }

        calculate_average_accuracy(&accuracies)
    }
}

const CSV_SETUP: &str = "setup (ms)";
const CSV_INFERENCE: &str = "inference (ms)";
const CSV_PROVING: &str = "proving (ms)";
const CSV_VERIFYING: &str = "verifying (ms)";
const CSV_ACCURACY: &str = "accuracy (bool)";
const CSV_PROOF_SIZE: &str = "proof size (KB)";

/// Runs the model in float format and returns the average accuracy
fn run_float_model(raw_inputs: &InputJSON, model: &Model<f32>) -> Result<f32> {
    let mut accuracies = Vec::new();
    info!("[+] Running model in float format");

    for (i, (input, expected)) in raw_inputs
        .input_data
        .iter()
        .zip(raw_inputs.output_data.iter())
        .enumerate()
    {
        // Run the model in float mode
        let input = model.load_input_flat(vec![input.clone()])?;
        let output = &model.run_float(&input)?[0];
        let accuracy = argmax_compare(expected, output.get_data());
        accuracies.push(accuracy);
        debug!(
            "Float Run {}/{}: Accuracy: {}",
            i + 1,
            raw_inputs.input_data.len(),
            if accuracy > 0 { "correct" } else { "incorrect" }
        );
    }

    Ok(calculate_average_accuracy(&accuracies))
}

fn read_model(args: &Args, inputs: &InputJSON) -> Result<(Model<Element>, ModelMetadata)> {
    let calibration_inputs = inputs.filter(args.calibration_indices.as_ref());
    match args.quantization.as_ref() {
        "inference" => {
            let strategy = InferenceObserver::new_with_representative_input(
                calibration_inputs
                    .input_data
                    .iter()
                    .map(|inp| vec![inp.clone()])
                    .collect(),
            );
            FloatOnnxLoader::new_with_scaling_strategy(&args.onnx, strategy)
                .with_keep_float(true)
                .build()
        }
        "maxabs" => {
            let strategy = AbsoluteMax::new();
            FloatOnnxLoader::new_with_scaling_strategy(&args.onnx, strategy)
                .with_keep_float(true)
                .build()
        }
        _ => panic!("Unsupported quantization strategy: {}", args.quantization),
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    info!("[+] Reading raw input/output from {}", args.io);
    let run_inputs = InputJSON::from(&args.io, args.num_samples).context("loading input:")?;
    info!("[+] Found {} IO samples", run_inputs.input_data.len());
    let (model, md) = read_model(&args, &run_inputs)?;
    info!("[+] Model loaded");
    model.describe();

    let run_inputs = run_inputs.filter(args.run_indices.as_ref());

    // Get float accuracy if float model is available
    let float_accuracy = if let Some(ref float_model) = md.float_model {
        info!("[+] Running float model");
        run_float_model(&run_inputs, float_model)?
    } else {
        info!("[!] No float model available");
        0.0
    };

    info!("[+] Computing PyTorch accuracy");
    let num_samples = run_inputs.output_data.len();
    let pytorch_accuracy = run_inputs.compute_pytorch_accuracy();
    info!("[+] Quantizing inputs with strategy: {}", args.quantization);
    let (inputs, given_outputs) = run_inputs.to_elements(&md);

    // Generate context once and measure the time
    let now = time::Instant::now();
    let ctx = if !args.skip_proving {
        info!("[+] Generating context for proving");
        Some(Context::<F, Pcs<F>>::generate(&model, None).expect("unable to generate context"))
    } else {
        None
    };
    let setup_time = now.elapsed().as_millis();
    info!("STEP: {} took {}ms", CSV_SETUP, setup_time);

    // Collect accuracies for final average
    let mut accuracies = Vec::new();
    // Track failed inputs
    let mut failed_inputs = Vec::new();

    let input_iter = inputs.into_iter().zip(given_outputs).enumerate();

    for (i, (input, given_output)) in input_iter {
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

        let input_tensor = model.load_input_flat(vec![input])?;
        // let input_tensor : Tensor<Element> = Tensor::new(model.input_not_padded.clone(), input);

        info!("[+] Running inference");
        // Handle model.run failures gracefully
        let trace_result = bencher.r(CSV_INFERENCE, || model.run(&input_tensor));

        // If model.run fails, print the error and continue to the next input
        let trace = match trace_result {
            Ok(trace) => trace,
            Err(e) => {
                info!(
                    "[!] Error running inference for input {}/{}: {}",
                    i + 1,
                    args.num_samples,
                    e
                );
                failed_inputs.push(i);
                continue; // Skip to the next input without writing to CSV
            }
        };
        // TEST:
        //  This prints the min/max in f32 of the output of each layer for this run
        //  Useful to check consistency with pytorch for example
        //{
        //    let dequantized_trace = trace.dequantized(&md);
        //    for step in dequantized_trace.steps.iter() {
        //        println!(
        //            "DEQUANTIZED STEP {}: output min/max: {}/{}",
        //            step.id,
        //            step.output.min(),
        //            step.output.max()
        //        );
        //    }
        //}
        let output = trace.outputs()?[0];
        let accuracy = argmax_compare(&given_output, output.get_data());
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
            continue;
        }
        info!("[+] Running prover");
        let io = trace.to_verifier_io();
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, _>::new(ctx.as_ref().unwrap(), &mut prover_transcript);
        let proof = bencher.r(CSV_PROVING, move || {
            prover.prove(trace).expect("unable to generate proof")
        });

        // Serialize proof using MessagePack and calculate size in KB
        let proof_bytes = to_vec_named(&proof)?;
        let proof_size_kb = proof_bytes.len() as f64 / 1024.0;
        bencher.set(CSV_PROOF_SIZE, format!("{proof_size_kb:.3}"));

        info!("[+] Running verifier");
        let mut verifier_transcript = default_transcript();
        bencher.r(CSV_VERIFYING, || {
            verify::<_, _, _>(
                ctx.as_ref().unwrap().clone(),
                proof,
                io,
                &mut verifier_transcript,
            )
            .expect("invalid proof")
        });
        info!("[+] Verify proof: valid");

        bencher.flush(&args.bench)?;
        info!("[+] Benchmark results appended to {}", args.bench);
    }

    // Calculate and display average accuracy
    let avg_accuracy = calculate_average_accuracy(&accuracies);

    // Single final accuracy comparison
    info!("Final accuracy comparison across {} runs:", num_samples);
    info!("ZKML float model accuracy: {:.2}%", float_accuracy * 100.0);
    info!(
        "ZKML quantized model accuracy: {:.2}%",
        avg_accuracy * 100.0
    );
    info!("PyTorch accuracy: {:.2}%", pytorch_accuracy * 100.0);

    // Report failure statistics
    info!(
        "[!] Failed inputs: {}/{} = {:.2}% (indices: {:?})",
        failed_inputs.len(),
        num_samples,
        (failed_inputs.len() as f32 / num_samples as f32) * 100.0,
        failed_inputs
    );
    Ok(())
}

fn argmax_compare<A: PartialOrd, B: PartialOrd>(
    given_output: &[A],
    computed_output: &[B],
) -> usize {
    let compare_size = std::cmp::min(given_output.len(), computed_output.len());
    let a_max = argmax(&given_output[..compare_size]);
    let b_max = argmax(&computed_output[..compare_size]);
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
                "CSVBencher only flushes one row at a time for now (key already registered: {column})"
            );
        }
        if !self.headers.contains(&column.to_string()) {
            panic!("column {column} non existing");
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
