//! A LLM driver runs the model on a given input and can inspect the output of each layer 
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to preprend a system model correctly.

use std::path::Path;

use anyhow::{bail, ensure, Context};
use ff_ext::ExtensionField;
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_tokenizers::tokenizer::Tokenizer as RT;
use serde::{de::DeserializeOwned, Serialize};

use crate::{layers::{provable::Evaluate, Layer}, model::{InferenceTrace, Model}, parser::{gguf, json, llm::{LLMConfig, Token}}, tensor::{Number, Shape}, Tensor};

#[derive(Debug, Clone)]
struct Driver<N: Number> {
    model: Model<N>,
    config: LLMConfig,
}

impl Driver<f32> {
    pub fn load_model<S: AsRef<Path>>(path: S) -> anyhow::Result<Self> {
        // detect the type of the model info, either json or gguf depending on the file extension
        let (config, llm_model) = match path.as_ref().extension().unwrap_or_default().to_str().unwrap() {
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
            _ => anyhow::bail!("Unsupported model file extension: {}", path.as_ref().extension().unwrap_or_default().to_str().unwrap()),
        };

        /// even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        /// need to provide one.
        let init_user_shape = Shape::from(vec![1, config.embedding_size]);
        let model = llm_model.to_provable_model(&config, init_user_shape)?;
        Ok(Self { model, config })
    }
}

impl<N: Number> Driver<N> 
where 
    Layer<N>: Evaluate<N> 
    {
    pub fn new(model: Model<N>, config: LLMConfig) -> Self {
        Self { model, config }
    }

    /// Runs take the _already_ tokenized input and run the model until the maximum sequence length is reached OR until a eos token is generated.
    /// The returned trace contains the _whole_ sequence.
    pub fn run<E>(&self, input: Vec<Token>, tokenizer: impl Tokenizer) -> anyhow::Result<InferenceTrace<'_, E, N>> 
    where 
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
    {
        let eos_token :N = self.config.specific_config.eos_token().to_number();
        let mut seq_len = input.len();
        let user_len = seq_len;
        // -1 because we at least want to generate ONE token
        ensure!(seq_len < self.config.context_length - 1, "Input sequence length must be less than the context length");
        let mut trace = InferenceTrace::default();
        // convert the input to the correct number type and add a dimension to make it 2d, because the embeddings layer expects a 2d tensor
        let mut tensor = Tensor::new(vec![input.len(), 1], input.into_iter().map(|t| t.to_number()).collect::<Vec<_>>());
        while seq_len < self.config.context_length  {
            trace = self.model.run::<E>(&[tensor.clone()]).context(format!("running the {} iteration loop", seq_len - user_len))?;
            let output = trace.output.last().unwrap();
            let last_token = output.slice_last_dim().last().unwrap();
            ensure!(last_token.len() == 1, "Last token must be a single token");
            let last_token = last_token[0];
            if last_token == eos_token {
                break;
            }
            // NOTE: For now, since we are NOT using any caching for the inference, we DON'T need to concat the inferences on top of each other
            //input = input.concat(last_token);
            // We simply need to take the _last_ inference trace that would contain _everything_
            seq_len += 1;
            // again, since we don't do caching, every further iteration includes the previous inference as well.
            let new_token = trace.output.last().unwrap().slice_last_dim().last().unwrap().clone();
            debug_assert!(new_token.len() == 1, "New token must be a single token");
            tensor.concat(Tensor::new(vec![1, 1], vec![new_token[0]]));
            debug_assert_eq!(tensor.get_shape()[0], seq_len);
            #[cfg(test)]
            {
                let sentence = tokenizer.detokenize(tensor.get_data().iter().map(|t| Token::from(t.to_usize())).collect::<Vec<_>>().as_slice());
                println!("seq_len: {}:\n\t-{}\n\t-{:?}", seq_len, sentence,tensor.get_data());
            }
        }
        Ok(trace)
    }
}

trait Tokenizer {
    fn detokenize(&self, ids: &[Token]) -> String;
}

impl Tokenizer for Gpt2Tokenizer {
    fn detokenize(&self, ids: &[Token]) -> String {
        let tokens = ids.iter().map(|i| i.into()).collect::<Vec<i64>>();
        self.decode(&tokens, true, true)
    }
}
#[cfg(test)]
mod test {
    use crate::parser::{file_cache, gguf::tests::GPT2_Q8_0_URL, llm::Token};

    use super::*;
    use goldilocks::GoldilocksExt2;
    use rust_tokenizers::{tokenizer::{Gpt2Tokenizer, Tokenizer}, vocab::Vocab};

    fn tokenize_sentence(tokenizer: &Gpt2Tokenizer, sentence: &str) -> Vec<Token> {
        let tokenized = tokenizer.tokenize_list(&[sentence]);
        tokenized.into_iter().take(1).flat_map(|s| s.into_iter().map(|t| tokenizer.vocab().token_to_id(&t).into())).collect::<Vec<_>>()
    }

    fn detokenize(tokenizer: &Gpt2Tokenizer, ids: &[Token]) -> String {
        let tokens = ids.iter().map(|i| i.into()).collect::<Vec<i64>>();
        tokenizer.decode(&tokens, true, true)
    }

    #[test]
    fn test_load_model() -> anyhow::Result<()>{
        let merge_path = file_cache::ensure_downloaded("https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt")?;
        let vocab_path = file_cache::ensure_downloaded("https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json")?;
        let model_path = file_cache::ensure_downloaded(GPT2_Q8_0_URL)?;
        let driver = Driver::load_model(&model_path)?;
        let tokenizer = Gpt2Tokenizer::from_file(vocab_path, merge_path, false)?;
        let sentence = "The sky is";
        let user_tokens = tokenize_sentence(&tokenizer, sentence);
        let detokenized = detokenize(&tokenizer, &user_tokens);
        assert_eq!(detokenized, sentence);
        println!("sentence: {}", sentence);
        println!("user_tokens: {:?}", user_tokens);
        let trace = driver.run::<GoldilocksExt2>(user_tokens,tokenizer)?;
        let output = trace.output.last().unwrap().get_data().iter().map(|t| Token::from(t.to_usize())).collect::<Vec<_>>();
        //let output = detokenize(&tokenizer, &output);
        //println!("{}", output);
        Ok(())
    }
}