//! A LLM driver runs the model on a given input and can inspect the output of each layer
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to prepend a system model correctly.

use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use serde::{Serialize, de::DeserializeOwned};
use std::path::Path;
use tracing::trace;

use crate::{
    Tensor,
    layers::{Layer, provable::Evaluate},
    model::{InferenceTrace, Model},
    parser::{
        gguf, json,
        llm::{LLMConfig, Token},
    },
    tensor::{Number, Shape},
};

pub trait Observer<N: Number> {
    fn observe<E: ExtensionField>(&self, step: usize, trace: &InferenceTrace<'_, E, N>);
}

#[derive(Debug, Clone)]
pub struct Driver<N: Number> {
    model: Model<N>,
    config: LLMConfig,
    max_context: Option<usize>,
}

impl Driver<f32> {
    pub fn load_model<S: AsRef<Path>>(path: S) -> anyhow::Result<Self> {
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
        let model = llm_model.into_provable_model(&config, init_user_shape)?;
        Ok(Self {
            model,
            config,
            max_context: None,
        })
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

    /// Runs take the _already_ tokenized input and run the model until the maximum sequence length is reached OR until a eos token is generated.
    /// The returned trace contains the _whole_ sequence.
    pub fn run_inference<E>(
        &mut self,
        input: Vec<Token>,
        observer: impl Observer<N>,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let eos_token: N = self.config.specific_config.eos_token().as_number();
        let mut seq_len = input.len();
        let user_len = seq_len;
        // -1 because we at least want to generate ONE token
        ensure!(
            seq_len < self.config.context_length - 1,
            "Input sequence length must be less than the context length"
        );
        let mut trace = InferenceTrace::default();
        // convert the input to the correct number type and add a dimension to make it 2d, because the embeddings layer expects a 2d tensor
        let mut tensor = Tensor::new(
            vec![input.len()].into(),
            input.into_iter().map(|t| t.as_number()).collect::<Vec<_>>(),
        );
        let max_window = self.max_context.unwrap_or(self.config.context_length);
        while seq_len < max_window {
            trace = self
                .model
                .run::<E>(&[tensor.clone()])
                .context(format!("running the {} iteration loop", seq_len - user_len))?;
            let output = trace.output.last().unwrap();
            let last_token = output.slice_last_dim().last().unwrap();
            ensure!(last_token.len() == 1, "Last token must be a single token");
            let last_token = last_token[0];
            if last_token == eos_token {
                break;
            }
            // NOTE: For now, since we are NOT using any caching for the inference, we DON'T need to concat the inferences on top of each other
            // input = input.concat(last_token);
            // We simply need to take the _last_ inference trace that would contain _everything_
            seq_len += 1;
            tensor.concat(Tensor::new(vec![1, 1].into(), vec![last_token]));
            debug_assert_eq!(tensor.get_shape()[0], seq_len);
            observer.observe(seq_len - user_len, &trace);
        }
        Ok(trace)
    }
}

pub struct LLMTokenizerObserver<T: LLMTokenizer> {
    input: String,
    tokenizer: T,
}

impl<N: Number, T: LLMTokenizer> Observer<N> for LLMTokenizerObserver<T> {
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
        trace!(
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
    use crate::parser::{
        file_cache,
        gguf::tests::GPT2_Q8_0_URL,
        llm::{Token, TokenizerData},
    };

    use super::*;
    use ff_ext::GoldilocksExt2;

    #[test]
    fn test_llm_driver() -> anyhow::Result<()> {
        let model_path = file_cache::ensure_downloaded(GPT2_Q8_0_URL)?;
        let mut driver = Driver::load_model(&model_path)?.with_max_context(10);
        let sentence = "The sky is";

        // Best to load the tokenizer from the gguf file if it's available.
        let tokenizer = TokenizerData::load_tokenizer_from_gguf(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let detokenized = tokenizer.detokenize(&user_tokens);
        assert_eq!(detokenized, sentence);
        println!("user input in tokens: {:?}", user_tokens);
        let trace = driver.run_inference::<GoldilocksExt2>(
            user_tokens,
            LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer,
            },
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
