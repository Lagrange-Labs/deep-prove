#![feature(iter_next_chunk)]

use derive_more::{Deref, From};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::DenseMultilinearExtension;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use transcript::Transcript;
mod claims;
mod matrix;
mod model;
mod onnx_parse;
mod prover;
mod lookup;

// TODO: use a real tensor
#[derive(Debug, Clone, From, Deref)]
struct Tensor<E>(Vec<E>);

impl<E: ExtensionField> Tensor<E> {
    pub fn scale_inplace(&mut self, scaling: E) {
        self.0.par_iter_mut().for_each(|v_i| *v_i = *v_i * scaling);
    }
}

pub fn pad_vector<E: ExtensionField>(mut v: Vec<E>) -> Vec<E> {
    if !v.len().is_power_of_two() {
        v.resize(v.len().next_power_of_two(), E::ZERO);
    }
    v
}
/// Returns a MLE out of the given vector, of the right length
// TODO : make that part of tensor somehow?
pub(crate) fn vector_to_mle<E: ExtensionField>(v: Vec<E>) -> DenseMultilinearExtension<E> {
    let v = pad_vector(v);
    DenseMultilinearExtension::from_evaluation_vec_smart(v.len().ilog2() as usize, v)
}

/// Returns the bit sequence of num of bit_length length.
pub(crate) fn to_bit_sequence_le(num: usize, bit_length: usize) -> impl Iterator<Item = usize> {
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

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::MultilinearExtension;

    use crate::{model::test::random_vector, onnx_parse::load_mlp, prover::{default_transcript, verify, Context, Prover, IO}, to_bit_sequence_le, vector_to_mle, Tensor};
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;

    #[test]
    fn test_model_run() {
        let filepath = "assets/model.onnx";
        let model = load_mlp::<E>(&filepath).unwrap();
        println!("[+] Loaded onnx file");
        let ctx = Context::generate(&model).expect("unable to generate context");
        println!("[+] Setup parameters");

        let shape = model.input_shape();
        assert_eq!(shape.len(),1);
        let input = random_vector(shape[0]);

        let trace = model.run(input.clone());
        let output = trace.final_output().to_vec();
        println!("[+] Run inference. Result: {:?}",output); 

        let mut prover_transcript = default_transcript();
        let prover = Prover::new(&mut prover_transcript);
        println!("[+] Run prover");
        let proof = prover.prove(&ctx, trace).expect("unable to generate proof");

        let mut verifier_transcript = default_transcript();
        let io = IO::new(input, output.to_vec());
        verify(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
        println!("[+] Verify proof: valid");
    }


    // TODO: move below code to a vector module

    #[test]
    fn test_vector_mle() {
        let n = 10;
        let v = (0..n).map(|_| E::random(&mut thread_rng())).collect_vec();
        let mle = vector_to_mle(v.clone());
        let random_index = thread_rng().gen_range(0..v.len());
        let eval = to_bit_sequence_le(random_index, v.len().next_power_of_two().ilog2() as usize)
            .map(|b| E::from(b as u64))
            .collect_vec();
        let output = mle.evaluate(&eval);
        assert_eq!(output, v[random_index]);
    }
    #[test]
    fn test_vector_scale() {
        let v = random_vector(10);
        let e = E::random(&mut thread_rng());
        let mut scaled = Tensor(v.clone());
        scaled.scale_inplace(e);
        for (o, n) in v.iter().zip(scaled.iter()) {
            assert_eq!(*o * e, *n);
        }
    }
}
