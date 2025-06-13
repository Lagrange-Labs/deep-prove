//! Deepprove library
#![feature(iter_next_chunk)]
#![feature(exact_size_is_empty)]

use ff_ext::ExtensionField;
use gkr::structs::PointAndEval;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use transcript::{BasicTranscript, Transcript};
mod commit;
pub mod iop;
pub mod quantization;
pub use iop::{
    Context, Proof,
    prover::Prover,
    verifier::{IO, verify},
};
pub use quantization::{ScalingFactor, ScalingStrategy};
pub mod layers;
pub mod lookup;
pub mod model;
pub mod padding;
mod parser;
pub use parser::{FloatOnnxLoader, ModelType};
pub mod tensor;
pub use tensor::Tensor;
#[cfg(test)]
mod testing;

/// We allow higher range to account for overflow. Since we do a requant after each layer, we
/// can support with i128 with 8 bits quant:
/// 16 + log(c) = 64 => c = 2^48 columns in a dense layer
pub type Element = i128;

/// Claim type to accumulate in this protocol, for a certain polynomial, known in the context.
/// f(point) = eval
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
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
    (0..bit_length).map(move |i| ((num >> i) & 1))
}

pub(crate) fn try_unzip<I, C, T, E>(iter: I) -> Result<C, E>
where
    I: IntoIterator<Item = Result<T, E>>,
    C: Extend<T> + Default,
{
    iter.into_iter().try_fold(C::default(), |mut c, r| {
        c.extend([r?]);
        Ok(c)
    })
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
    if v.is_empty() {
        return None;
    }

    let mut max_index = 0;
    let mut max_value = &v[0];

    for (i, value) in v.iter().enumerate().skip(1) {
        // Only update if strictly greater, ensuring we take the first maximum in ties
        if value > max_value {
            max_index = i;
            max_value = value;
        }
    }

    Some(max_index)
}

pub fn argmax_slice<N: Number>(v: &[N]) -> Option<usize> {
    if v.is_empty() {
        return None;
    }
    Some(
        v.iter()
            .enumerate()
            .fold((0, N::MIN), |acc, x| match acc.1.compare(&x.1) {
                Ordering::Less => (x.0, *x.1),
                _ => acc,
            })
            .0,
    )
}

pub trait NextPowerOfTwo {
    /// Returns a new vector where each element is the next power of two.
    fn next_power_of_two(&self) -> Self;
}
// For unsigned integer vectors
impl NextPowerOfTwo for Vec<usize> {
    fn next_power_of_two(&self) -> Self {
        self.iter().map(|&i| i.next_power_of_two()).collect()
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use crate::{
        FloatOnnxLoader, default_transcript,
        iop::{Context, prover::Prover, verifier::verify},
        parser::ModelType,
        tensor::Tensor,
        testing::Pcs,
        to_bit_sequence_le,
    };
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;

    #[test]
    fn test_model_run() -> anyhow::Result<()> {
        test_model_run_helper()?;
        Ok(())
    }

    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
    }

    fn test_model_run_helper() -> anyhow::Result<()> {
        let filepath = workspace_root().join("zkml/assets/model.onnx");
        let (model, _md) = FloatOnnxLoader::new(&filepath.to_string_lossy())
            .with_model_type(ModelType::MLP)
            .build()?;

        println!("[+] Loaded onnx file");
        let ctx = Context::<E, Pcs<E>>::generate(&model, None).expect("unable to generate context");
        println!("[+] Setup parameters");

        let shapes = model.input_shapes();
        assert_eq!(shapes.len(), 1);
        let shape = &shapes[0];
        assert_eq!(shape.len(), 1);
        let input = Tensor::random(&vec![shape[0] - 1]);
        println!("input: {:?}", input.get_data());
        let input = model.prepare_inputs(vec![input])?;

        let trace = model.run(&input).unwrap();
        let output = trace.outputs()?[0];
        println!("[+] Run inference. Result: {:?}", output);

        let io = trace.to_verifier_io();
        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _, _>::new(&ctx, &mut prover_transcript);
        println!("[+] Run prover");
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript = default_transcript();
        verify::<_, _, _>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
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

use std::cmp::Ordering;
#[cfg(test)]
use std::sync::Once;

use crate::tensor::Number;

#[cfg(test)]
static INIT: Once = Once::new();

#[cfg(test)]
pub fn init_test_logging_default() {
    use tracing_subscriber::EnvFilter;

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::fmt().with_env_filter(filter).init();
    });
}

#[cfg(test)]
pub fn init_test_logging(default_level: &str) {
    use tracing_subscriber::EnvFilter;

    INIT.call_once(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
        tracing_subscriber::fmt().with_env_filter(filter).init();
    });
}
