use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::BufReader,
    path::Path,
    time,
};

use anyhow::{Context as CC, ensure};
use clap::Parser;
use csv::WriterBuilder;
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use log::info;

use serde::{Deserialize, Serialize};
use zkml::{
    Context, Element, IO, Prover, argmax, default_transcript, load_mlp,
    lookup::LogUp,
    quantization::{Quantizer, TensorFielder},
    tensor::Tensor,
    verify,
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
}
pub fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    run(args).context("error running bench:")?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct InputJSON {
    input_data: Vec<Vec<f64>>,
    output_data: Vec<Vec<f64>>,
}

impl InputJSON {
    /// Returns (input,output) from the path
    pub fn from(path: &str) -> anyhow::Result<(Vec<Element>, Vec<Element>)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let u: Self = serde_json::from_reader(reader)?;
        u.validate()?;
        Ok(u.to_elements())
    }
    // poor's man validation
    fn validate(&self) -> anyhow::Result<()> {
        let rrange = -1.0..=1.0;
        ensure!(self.input_data.len() == 1);
        let input_isreal = self.input_data[0].iter().all(|v| rrange.contains(v));
        // let output_isreal = self.output_data[0].iter().all(|v| rrange.contains(v));
        ensure!(
            input_isreal,
            "can only support real model so far (input at least)"
        );
        Ok(())
    }
    fn to_elements(mut self) -> (Vec<Element>, Vec<Element>) {
        let inputs = self
            .input_data
            .remove(0)
            .into_iter()
            .map(|e| Element::from_f32_unsafe(&(e as f32)))
            .collect_vec();
        let outputs = self
            .output_data
            .remove(0)
            .into_iter()
            .map(|e| Element::from_f32_unsafe(&(e as f32)))
            .collect_vec();
        (inputs, outputs)
    }
}

const CSV_LOAD: &str = "load (ms)";
const CSV_SETUP: &str = "setup (ms)";
const CSV_INFERENCE: &str = "inference (ms)";
const CSV_PROVING: &str = "proving (ms)";
const CSV_VERIFYING: &str = "verifying (ms)";
const CSV_ACCURACY: &str = "accuracy (bool)";
const CSV_PROOF_SIZE: &str = "proof size (KB)";

fn run(args: Args) -> anyhow::Result<()> {
    let mut bencher = CSVBencher::from_headers(vec![
        CSV_LOAD,
        CSV_SETUP,
        CSV_INFERENCE,
        CSV_PROVING,
        CSV_VERIFYING,
        CSV_PROOF_SIZE,
        CSV_ACCURACY,
    ]);
    info!("[+] Reading onnx model");
    let model = bencher
        .r(CSV_LOAD, || load_mlp::<Element>(&args.onnx))
        .context("loading model:")?;
    model.describe();
    info!("[+] Reading input/output from pytorch");
    let (input, given_output) = InputJSON::from(&args.io).context("loading input:")?;
    let input = Tensor::<Element>::new(vec![input.len()], input);
    let input = model.prepare_input(input);
    // model.describe();
    // println!("input: {:?}",input);

    info!("[+] Generating context for proving");
    let ctx = bencher.r(CSV_SETUP, || {
        Context::<F>::generate(&model).expect("unable to generate context")
    });
    let shape = model.input_shape();
    assert_eq!(shape.len(), 1, "only support vector as input for now");

    info!("[+] Running inference");
    let trace = bencher.r(CSV_INFERENCE, || model.run(input.clone()));
    let output = trace.final_output().clone();
    bencher.set(
        CSV_ACCURACY,
        compare(&given_output, &output.get_data().to_vec()),
    );

    info!("[+] Running prover");
    let mut prover_transcript = default_transcript();
    let prover = Prover::<_, _, LogUp>::new(&ctx, &mut prover_transcript);
    let proof = bencher.r(CSV_PROVING, move || {
        prover.prove(trace).expect("unable to generate proof")
    });

    // Serialize proof using MessagePack and calculate size in KB
    let proof_bytes = to_vec_named(&proof)?;
    let proof_size_kb = proof_bytes.len() as f64 / 1024.0;
    bencher.set(CSV_PROOF_SIZE, format!("{:.3}", proof_size_kb));

    info!("[+] Running verifier");
    let mut verifier_transcript = default_transcript();
    let io = IO::new(input.to_fields(), output.to_fields());
    bencher.r(CSV_VERIFYING, || {
        verify::<_, _, LogUp>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof")
    });
    info!("[+] Verify proof: valid");

    bencher.flush(&args.bench)?;
    info!("[+] Benchmark results appended to {}", args.bench);
    Ok(())
}

fn compare<A: PartialOrd, B: PartialOrd>(given_output: &[A], computed_output: &[B]) -> usize {
    let a_max = argmax(given_output);
    let b_max = argmax(computed_output);
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
            panic!("CSVBencher only handles one row for now");
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
