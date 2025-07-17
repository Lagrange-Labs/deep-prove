use anyhow::bail;
use zkml::model::llm::{LLMContext, LLMTokenizer};
use csv::WriterBuilder;
use mpcs::{Basefold, BasefoldRSParams, Hasher};
use zkml::model::llm::LLMTokenizerObserver;
use std::path::PathBuf;
use std::{path::Path, time};
use std::collections::HashMap;
use ff_ext::GoldilocksExt2;
use timed_core::Output;
use tracing_subscriber::EnvFilter;
use std::fs::OpenOptions;
use clap::Parser;
use zkml::{
    parser::{file_cache,llm::TokenizerData},
};

type F = GoldilocksExt2;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
type Pcs<E> = Basefold<E, BasefoldRSParams<Hasher>>;

#[derive(Parser, Debug)]
struct LLMArgs {
    /// gguf file to load. It can be a local path or a URL to download.
    #[arg(short, long)]
    gguf: String,

    /// max context length (in tokens)
    #[arg(short, long, default_value_t = 1024)]
    max_context: usize,
    /// number of samples to process 
    #[arg(short, long, default_value_t = 30)]
    num_samples: usize,

    /// min user input length (in tokens)
    #[arg(short, long, default_value_t = 1)]
    min_user_len: usize,

    /// model to use
    #[arg(short, long, default_value_t = {"gpt2".to_string()})]
    model: String,

    /// output file name
    #[arg(short, long, default_value_t = {"bench-llm.csv".to_string()})]
    output: String,
}

const HEADER_MODEL: &str = "model";
const HEADER_MODEL_QUANT: &str = "model_quant";
const HEADER_MAX_CONTEXT: &str = "max_context";
const HEADER_NUM_SAMPLES: &str = "num_samples";
const HEADER_MIN_USER_LEN: &str = "min_user_len";
const HEADER_ACCURACY: &str = "accuracy";
const HEADER_INFERENCE_TIME: &str = "inference_time";
const HEADER_PROOF_TIME: &str = "proof_time";
const HEADER_VERIFY_TIME: &str = "verification";
const HEADER_CONTEXT_TIME: &str = "context_time";

fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
    timed_core::set_output(Output::CSV("bench-llm.csv".to_string()));
    let args = LLMArgs::parse();
    let model_path = if is_url(&args.gguf) {
        file_cache::ensure_downloaded(&args.gguf)?
    } else {
        PathBuf::from(args.gguf.clone())
    };

    let (driver,tokenizer) = match args.model.as_str() {
        "gpt2" => {
            let driver = Driver::load_external_model(&model_path)?.with_max_context(args.max_context);
            let tokenizer = TokenizerData::load_tokenizer_from_gguf(&model_path)?;
            (driver,tokenizer)
        }
        _ => bail!("Model {:?} not supported",args.model),
    };

    let mut bencher = CSVBencher::from_headers(vec![HEADER_MODEL, HEADER_MAX_CONTEXT, HEADER_NUM_SAMPLES, HEADER_MIN_USER_LEN, HEADER_ACCURACY,HEADER_MODEL_QUANT]);
    let driver = bencher.r(HEADER_MODEL_QUANT, || driver.into_provable_llm())?;
    let ctx: LLMContext<F, Pcs<F>> = bencher.r(HEADER_CONTEXT_TIME, || driver.context())?;

    bencher.set(HEADER_MODEL,  args.model);
    bencher.set(HEADER_MAX_CONTEXT,  args.max_context);
    bencher.set(HEADER_NUM_SAMPLES,  args.num_samples);
    bencher.set(HEADER_MIN_USER_LEN, args.min_user_len);    

    for _ in 0..args.num_samples {
        let user_tokens = driver.random_sequence(args.min_user_len);
        let sentence = tokenizer.detokenize(&user_tokens);
        let trace = bencher.r(HEADER_INFERENCE_TIME, || {
            driver.run::<GoldilocksExt2>(user_tokens.clone(),Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }))
        })?;

        let proof = bencher.r(HEADER_PROOF_TIME, || driver.prove(&ctx, trace))?;
        bencher.r(HEADER_VERIFY_TIME, || ctx.verify(proof, user_tokens).expect("invalid proof"));
        bencher.flush(&args.output)?;
    }

    Ok(())
}

use tracing::info;
use url::Url;
use zkml::model::llm::Driver;

fn is_url(s: &str) -> bool {
    // Try to parse as URL
    if let Ok(url) = Url::parse(s) {
        // Accept only common download schemes
        matches!(url.scheme(), "http" | "https" | "ftp")
    } else {
        false
    }
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