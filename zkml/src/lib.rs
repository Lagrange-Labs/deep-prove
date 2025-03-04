#![feature(iter_next_chunk)]

use ff_ext::ExtensionField;
use gkr::structs::PointAndEval;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use transcript::{BasicTranscript, Transcript};
pub mod activation;
mod commit;
pub mod iop;
pub use iop::{
    Context, Proof,
    prover::Prover,
    verifier::{IO, verify},
};

pub mod lookup;
pub mod model;
mod onnx_parse;
pub mod pooling;
pub mod quantization;
pub use onnx_parse::load_mlp;

pub mod tensor;
mod testing;
mod utils;

/// We allow higher range to account for overflow. Since we do a requant after each layer, we
/// can support with i128 with 8 bits quant:
/// 16 + log(c) = 64 => c = 2^48 columns in a dense layer
pub type Element = i128;

/// Claim type to accumulate in this protocol, for a certain polynomial, known in the context.
/// f(point) = eval
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim<E> {
    point: Vec<E>,
    eval: E,
}

impl<E> Claim<E> {
    pub fn new(point: Vec<E>, eval: E) -> Self {
        Self { point, eval }
    }
}

impl<E: ExtensionField> From<PointAndEval<E>> for Claim<E> {
    fn from(value: PointAndEval<E>) -> Self {
        Claim {
            point: value.point.clone(),
            eval: value.eval,
        }
    }
}

impl<E: ExtensionField> From<&PointAndEval<E>> for Claim<E> {
    fn from(value: &PointAndEval<E>) -> Self {
        Claim {
            point: value.point.clone(),
            eval: value.eval,
        }
    }
}

impl<E: ExtensionField> Claim<E> {
    /// Pad the point to the new size given
    /// This is necessary for passing from output of padded lookups to next dense layer proving for example.
    /// NOTE: you can use it to pad or reduce size
    pub fn pad(&self, new_num_vars: usize) -> Claim<E> {
        Self {
            eval: self.eval,
            point: self
                .point
                .iter()
                .chain(std::iter::repeat(&E::ZERO))
                .take(new_num_vars)
                .cloned()
                .collect_vec(),
        }
    }
}

/// Returns the default transcript the prover and verifier must instantiate to validate a proof.
pub fn default_transcript<E: ExtensionField>() -> BasicTranscript<E> {
    BasicTranscript::new(b"m2vec")
}

pub fn pad_vector<E: ExtensionField>(mut v: Vec<E>) -> Vec<E> {
    if !v.len().is_power_of_two() {
        v.resize(v.len().next_power_of_two(), E::ZERO);
    }
    v
}
/// Returns the bit sequence of num of bit_length length.
pub(crate) fn to_bit_sequence_le(
    num: usize,
    bit_length: usize,
) -> impl DoubleEndedIterator<Item = usize> {
    assert!(
        bit_length as u32 <= usize::BITS,
        "bit_length cannot exceed usize::BITS"
    );
    (0..bit_length).map(move |i| ((num >> i) & 1) as usize)
}

pub trait VectorTranscript<E: ExtensionField> {
    fn read_challenges(&mut self, n: usize) -> Vec<E>;
}

#[cfg(not(test))]
impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| self.read_challenge().elements).collect_vec()
    }
}

#[cfg(test)]
impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| E::ONE).collect_vec()
    }
}

pub fn argmax<T: PartialOrd>(v: &[T]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // Unwrap is safe if T implements PartialOrd properly
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use crate::{
        Element, default_transcript,
        iop::{
            Context,
            prover::Prover,
            verifier::{IO, verify},
        },
        lookup::{LogUp, LookupProtocol},
        onnx_parse::load_mlp,
        quantization::{QuantInteger, TensorFielder},
        tensor::Tensor,
        to_bit_sequence_le,
    };
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;

    #[test]
    fn test_model_run() -> anyhow::Result<()> {
        test_model_run_helper::<LogUp>()?;
        Ok(())
    }

    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
    }

    fn test_model_run_helper<L: LookupProtocol<E>>() -> anyhow::Result<()> {
        let filepath = workspace_root().join("zkml/assets/model.onnx");

        let model = load_mlp::<Element>(&filepath.to_string_lossy()).unwrap();
        println!("[+] Loaded onnx file");
        let ctx = Context::<E>::generate(&model).expect("unable to generate context");
        println!("[+] Setup parameters");

        let shape = model.input_shape();
        assert_eq!(shape.len(), 1);
        let input = Tensor::random::<QuantInteger>(vec![shape[0] - 1]);
        let input = model.prepare_input(input);

        let trace = model.run(input.clone());
        let output = trace.final_output().clone();
        println!("[+] Run inference. Result: {:?}", output);

        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, L>::new(&ctx, &mut prover_transcript);
        println!("[+] Run prover");
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript = default_transcript();
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _, L>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
        println!("[+] Verify proof: valid");
        Ok(())
    }

    // TODO: move below code to a vector module

    #[test]
    fn test_vector_mle() {
        let n = (10 as usize).next_power_of_two();
        let v = (0..n).map(|_| E::random(&mut thread_rng())).collect_vec();
        let mle = v.clone().into_mle();
        let random_index = thread_rng().gen_range(0..v.len());
        let eval = to_bit_sequence_le(random_index, v.len().next_power_of_two().ilog2() as usize)
            .map(|b| E::from(b as u64))
            .collect_vec();
        let output = mle.evaluate(&eval);
        assert_eq!(output, v[random_index]);
    }
}

#[cfg(test)]
use std::sync::Once;

#[cfg(test)]
static INIT: Once = Once::new();

#[cfg(test)]
pub fn init_test_logging() {
    INIT.call_once(|| {
        // Initialize your logger only once
        env_logger::try_init().ok(); // The .ok() ignores if it's already been initialized
    });
}
