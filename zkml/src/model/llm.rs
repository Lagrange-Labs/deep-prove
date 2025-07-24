//! A LLM driver runs the model on a given input and can inspect the output of each layer
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to prepend a system model correctly.

use crate::{
    padding::PaddingMode, quantization::{InferenceObserver, ScalingStrategy}, verify, Context, Proof, Prover, IO
};
use anyhow::{Context as CC, ensure};
use ark_std::rand::{Rng, thread_rng};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::Path;
use tracing::debug;
use transcript::BasicTranscript;

use crate::{
    Element, Tensor,
    layers::{Layer, provable::Evaluate},
    model::{InferenceTrace, Model},
    padding::pad_model,
    parser::{
        gguf, json,
        llm::{LLMConfig, Token},
    },
    tensor::{Number, Shape},
};

pub trait Observer<N: Number> {
    fn observe<E: ExtensionField>(&self, step: usize, trace: &InferenceTrace<'_, E, N>);
}

/// The main struct responsible for generating the trace and the proof related
/// to LLM proving. This requires a wrapper on top of the model to drive the
/// auto regressive loop correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Driver<N: Number> {
    model: Model<N>,
    config: LLMConfig,
    max_context: Option<usize>,
}

/// The main struct responsible for verifying the proof related to the LLM proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LLMContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub ctx: Context<E, PCS>,
    pub config: LLMConfig,
    pub max_context: Option<usize>,
}

impl<E,PCS> LLMContext<E,PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LLMProof<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub proof: Proof<E, PCS>,
    /// Note the IO contains the _full_ input, e.g. the user input + the generated tokens
    pub io: IO<E>,
}

impl Driver<f32> {
    /// Loads a model from a gguf or json external file. It returns the raw model in float precision.
    pub fn load_external_model<S: AsRef<Path>>(path: S) -> anyhow::Result<Self> {
        // detect the type of the model info, either json or gguf depending on the file extension
        let (config, llm_model) = match path
            .as_ref()
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap()
        {
            "json" => {
                let loader = json::FileTensorLoader::new_from_path(path)?;
                let config = LLMConfig::from_json(&loader)?;
                let llm_model = config.model_json(&loader)?;
                (config, llm_model)
            }
            "gguf" => {
                let loader = gguf::FileTensorLoader::from_path(path)?;
                let config = LLMConfig::from_content(&loader)?;
                let llm_model = config.model(&loader)?;
                (config, llm_model)
            }
            _ => anyhow::bail!(
                "Unsupported model file extension: {}",
                path.as_ref()
                    .extension()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap()
            ),
        };

        // even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        // need to provide one.
        let init_user_shape = Shape::from(vec![1]);
        let model = llm_model.into_runnable_model(&config, init_user_shape)?;
        Ok(Self {
            model,
            config,
            max_context: None,
        })
    }

    /// Transform the model into a provable llm model with quantization and padding done.
    /// The result can be serialized and deserialized at will to serve inference+proving for this model.
    pub fn into_provable_llm(mut self) -> anyhow::Result<Driver<Element>> {
        let numel = self.config.context_length;
        let n_inputs = 1;
        let representative_inputs = (0..n_inputs)
            .map(|_| {
                self.random_sequence(numel)
                    .into_iter()
                    .map(|t| t.as_number())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.model.unpadded_input_shapes = vec![Shape::from(vec![numel])];
        self.model.input_shapes = vec![Shape::from(vec![numel, 1]).next_power_of_two()];
        let (quantized_model, _md) =
            InferenceObserver::new_with_representative_input(vec![representative_inputs])
                .quantize(self.model)?;
        let model = pad_model(quantized_model)?;
        Ok(Driver {
            model,
            config: self.config,
            max_context: self.max_context,
        })
    }
    pub fn run<E>(
        &self,
        input: Vec<Token>,
        observer: Option<impl Observer<f32>>,
    ) -> anyhow::Result<InferenceTrace<'_, E, f32>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        self.run_internal::<E>(input, observer,PaddingMode::NoPadding)
    }
}

impl<N: Number> Driver<N>
where
    Layer<N>: Evaluate<N>,
{
    pub fn new(model: Model<N>, config: LLMConfig, max_context: Option<usize>) -> Self {
        Self {
            model,
            config,
            max_context,
        }
    }
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }

    pub fn random_sequence(&self, len: usize) -> Vec<Token> {
        let mut rng = thread_rng();
        (0..len)
            .map(|_| Token::from(rng.gen_range(0..self.config.vocab_size)))
            .collect()
    }

    /// Runs take the _already_ tokenized input and run the model until the maximum sequence length is reached OR until a eos token is generated.
    /// The returned trace contains the _whole_ sequence.
    fn run_internal<E>(
        &self,
        input: Vec<Token>,
        observer: Option<impl Observer<N>>,
        mode: PaddingMode,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let eos_token: N = self.config.specific_config.eos_token().as_number();
        let mut unpadded_seq_len = input.len();
        let user_len = unpadded_seq_len;
        // -1 because we at least want to generate ONE token
        ensure!(
            unpadded_seq_len < self.config.context_length - 1,
            "Input sequence length must be less than the context length"
        );
        let tensor = Tensor::new(
            vec![input.len()].into(),
            input.into_iter().map(|t| t.as_number::<N>()).collect::<Vec<_>>(),
        );
        let mut tensor = if let PaddingMode::Padding = mode {
            tensor.pad_next_power_of_two()
        } else { 
            tensor
        };

        ensure!(
            tensor
                .get_data()
                .iter()
                .all(|t| t.to_usize() < self.config.vocab_size),
            "Input tokens must be less than the vocabulary size"
        );
        let mut trace = InferenceTrace::default();
        // convert the input to the correct number type and add a dimension to make it 2d, because the embeddings layer expects a 2d tensor
        // This means we're padding the input to the right size (e.g. next power of two)
        let max_window = self.max_context.unwrap_or(self.config.context_length);
        while unpadded_seq_len < max_window {
            trace = self.model.run::<E>(&[tensor.clone()]).context(format!(
                "running the {} iteration loop",
                unpadded_seq_len - user_len
            ))?;
            ensure!(trace.output.len() == 1, "expected 1 output, got {}", trace.output.len());
            let output = trace.output.last().unwrap();
            // We take the last token before the padding
            let last_token = output
                .get_data()
                .get(unpadded_seq_len - 1)
                .expect("last token must exist");
            if *last_token == eos_token {
                break;
            }
            // NOTE: For now, since we are NOT using any caching for the inference, we DON'T need to concat the inferences on top of each other
            // input = input.concat(last_token);
            // We simply need to take the _last_ inference trace that would contain _everything_
            unpadded_seq_len += 1;
            if tensor.get_shape().numel() <= unpadded_seq_len {
                tensor.concat(Tensor::new(vec![1, 1].into(), vec![*last_token]));
            } else {
                // here we need to insert the new token after the user input and newly generated tokens, but
                // BEFORE the padding.
                // TODO: breach of API here - tensor should do it
                tensor.data.insert(unpadded_seq_len, *last_token);
                tensor.shape.set_dim(0, tensor.shape.dim(0) + 1);
            }
            if let PaddingMode::Padding = mode {
                tensor = tensor.pad_next_power_of_two();
            }
            if let Some(ref obs) = observer {
                obs.observe(unpadded_seq_len - user_len, &trace);
            }
        }
        Ok(trace)
    }
}

impl Driver<Element> {
    pub fn run<E>(
        &self,
        input: Vec<Token>,
        observer: Option<impl Observer<Element>>,
    ) -> anyhow::Result<InferenceTrace<'_, E, Element>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        self.run_internal::<E>(input, observer, PaddingMode::Padding)
    }
    pub fn context<E, PCS>(&self) -> anyhow::Result<LLMContext<E, PCS>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let ctx = Context::<E, PCS>::generate(&self.model, None, None)?;
        Ok(LLMContext {
            ctx,
            config: self.config.clone(),
            // The verifier should put itself the max context here
            max_context: None,
        })
    }
    pub fn prove<E, PCS>(
        &self,
        ctx: &LLMContext<E, PCS>,
        trace: InferenceTrace<'_, E, Element>,
    ) -> anyhow::Result<LLMProof<E, PCS>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let mut tr: BasicTranscript<E> = BasicTranscript::new(b"model");
        let prover: Prover<'_, E, _, _> = Prover::new(&ctx.ctx, &mut tr);
        let io = trace.to_verifier_io();
        let proof = prover.prove(trace).expect("unable to generate proof");
        Ok(LLMProof { proof, io })
    }
}

impl<E, PCS> LLMContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn verify(&self, proof: LLMProof<E, PCS>, user_input: Vec<Token>) -> anyhow::Result<()>
    where
        E: ExtensionField + Serialize + DeserializeOwned + Number,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // 0. check the size of the output
        let output = proof.io.output[0].clone();
        let max_len = output.get_shape().numel();
        // in any case, the output needs to be less than the max context length
        ensure!(
            max_len <= self.config.context_length,
            "output length is greater than the context length"
        );
        // either we reached the specific may context length
        if let Some(max_context) = self.max_context {
            ensure!(
                max_len <= max_context,
                "output length is greater than the max context length"
            );
        } else {
            // OR we reached the eos token
            let eos_token = self.config.specific_config.eos_token().0;
            ensure!(
                output.get_data().last().unwrap().to_usize() == eos_token,
                "output did not end with the eos token"
            );
        }
        // 1. verify the proof it self
        let mut tr: BasicTranscript<E> = BasicTranscript::new(b"model");
        let prover_input = proof.io.input[0].clone();
        let prover_output = proof.io.output[0].clone();
        verify::<_, _, _>(self.ctx.clone(), proof.proof, proof.io, &mut tr)?;
        // 2. verify the sequentiality of the output: from the first newly generated token to the last
        // but without including the padding.
        // output is [seq_len] where []
        let seq_len = user_input.len();
        let max_len = prover_output.get_shape().numel();
        ensure!(
            prover_input.get_data()[..seq_len]
                .iter()
                .zip(user_input[..seq_len].iter())
                .all(|(a, b)| a.to_usize() == b.0),
            "user input not the same"
        );
        #[allow(clippy::needless_range_loop)]
        for i in seq_len - 1..max_len - 1 {
            // we check the next input token is the one generated by this "row" of the input
            ensure!(
                prover_input.get_data()[i + 1] == prover_output.get_data()[i],
                "next input token is not the one generated by this row"
            );
        }
        Ok(())
    }
}

pub struct LLMTokenizerObserver<'a, T: LLMTokenizer> {
    pub input: String,
    pub tokenizer: &'a T,
}

impl<'a, N: Number, T: LLMTokenizer> Observer<N> for LLMTokenizerObserver<'a, T> {
    fn observe<E: ExtensionField>(&self, step: usize, trace: &InferenceTrace<'_, E, N>) {
        let tensor = trace.output.last().unwrap();
        let new_token = tensor.get_data().last().unwrap();
        let new_token = Token::from(new_token.to_usize());
        let new_text = self.tokenizer.detokenize(
            tensor
                .get_data()
                .iter()
                .map(|t| Token::from(t.to_usize()))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        debug!(
            "seq_len {}: new token: {:?}\n\t-{}", //\n\t-{:?}",
            step,
            &new_token,
            (self.input.clone() + &new_text).trim(),
            // tensor.get_data()
        );
    }
}

pub trait LLMTokenizer {
    fn tokenize(&self, sentence: &str) -> Vec<Token>;
    fn detokenize(&self, ids: &[Token]) -> String;
}

#[cfg(test)]
mod test {
    use crate::{
        init_test_logging,
        parser::{
            file_cache,
            gguf::tests::GPT2_Q8_0_URL,
            llm::{Token, TokenizerData},
        },
        testing::Pcs,
    };

    use super::*;
    use ff_ext::GoldilocksExt2;

    #[test]
    fn test_llm_driver_prove() -> anyhow::Result<()> {
        init_test_logging("debug");
        let max_context = 10;
        let model_path = file_cache::ensure_downloaded(GPT2_Q8_0_URL)?;
        let driver = Driver::load_external_model(&model_path)?.with_max_context(max_context);
        let sentence = "The sky is";
        let tokenizer = TokenizerData::load_tokenizer_from_gguf(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let driver = driver.into_provable_llm()?;
        let ctx = driver.context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?.with_max_context(max_context);
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let proof = driver.prove(&ctx, trace)?;
        ctx.verify(proof, user_tokens)?;
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference() -> anyhow::Result<()> {
        init_test_logging("debug");
        // const PRUNED_GPT2: &str = "https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q2_K.gguf";
        const PRUNED_GPT2: &str = GPT2_Q8_0_URL;
        let model_path = file_cache::ensure_downloaded(PRUNED_GPT2)?;
        let driver = Driver::load_external_model(&model_path)?.with_max_context(6);
        let sentence = "The sky is";

        // Best to load the tokenizer from the gguf file if it's available.
        let tokenizer = TokenizerData::load_tokenizer_from_gguf(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let detokenized = tokenizer.detokenize(&user_tokens);
        assert_eq!(detokenized, sentence);
        println!("user input in tokens: {:?}", user_tokens);
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let _output = trace
            .output
            .last()
            .unwrap()
            .get_data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        // let output = detokenize(&tokenizer, &output);
        // println!("{}", output);
        Ok(())
    }
}
