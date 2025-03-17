use ff_ext::ExtensionField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    Element,
    quantization::{self, BIT_LEN, Fieldizer},
    tensors::Tensor,
};

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    Relu(Relu),
}

impl Activation {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        match self {
            Activation::Relu(relu) => relu.op(input),
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Relu;

impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        *BIT_LEN
    }
    pub fn poly_len() -> usize {
        1 << Self::num_vars()
    }
    pub fn shape() -> Vec<usize> {
        vec![2, Self::poly_len()]
    }
    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values
    pub fn to_mle<E: ExtensionField>() -> (Vec<E::BaseField>, Vec<E::BaseField>) {
        (*quantization::MIN..=*quantization::MAX)
            .map(|i| {
                let val: E = i.to_field();
                let op_val: E = Relu::apply(i as i128).to_field();
                (val.as_bases()[0], op_val.as_bases()[0])
            })
            .unzip()
    }

    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        Tensor::new(
            input.get_shape(),
            input
                .get_data()
                .par_iter()
                .map(|e| Self::apply(*e))
                .collect::<Vec<_>>(),
        )
    }

    #[inline(always)]
    pub fn apply(e: Element) -> Element {
        if e.is_negative() { 0 } else { e }
    }
}

#[cfg(test)]
mod test {
    use crate::to_bit_sequence_le;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};

    use super::*;

    type F = GoldilocksExt2;

    #[test]
    fn test_activation_relu_apply() {
        struct TestCase {
            input: Element,
            output: Element,
        }

        impl TestCase {
            pub fn from(input: Element, output: Element) -> Self {
                Self { input, output }
            }
        }
        for case in [
            TestCase::from(-24, 0),
            TestCase::from(0, 0),
            TestCase::from(124, 124),
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }

    #[test]
    fn test_activation_relu_mle() {
        let relu = Relu::new();
        let (input_poly, output_poly) = Relu::to_mle::<F>();

        assert_eq!(input_poly.len(), output_poly.len());
        let (input_mle, output_mle) = (
            DenseMultilinearExtension::from_evaluation_vec_smart(
                Relu::num_vars(),
                input_poly.to_vec(),
            ),
            DenseMultilinearExtension::from_evaluation_vec_smart(
                Relu::num_vars(),
                output_poly.to_vec(),
            ),
        );
        assert_eq!(input_mle.num_vars(), output_mle.num_vars());
        assert_eq!(input_mle.num_vars(), Relu::num_vars());
        let inputs = Tensor::random(vec![10]);
        let outputs = relu.op(&inputs);
        assert_eq!(inputs.get_shape(), outputs.get_shape());
        for (input, output) in inputs.get_data().iter().zip(outputs.get_data().iter()) {
            // here putting input works because every random input is a u8, so it's already within [0;256] so
            // its value "is" the index. Normally if this is not true, we should get the index of the row corresponding to that input
            let idx_vars = to_bit_sequence_le((input + 128) as usize, Relu::num_vars())
                .map(|b| F::from(b as u64))
                .collect_vec();
            let input_field = input_mle.evaluate(&idx_vars);
            let expected_ified: F = input.to_field();
            assert_eq!(input_field, expected_ified);
            let output_field = output_mle.evaluate(&idx_vars);
            let expected_ofield: F = output.to_field();
            assert_eq!(output_field, expected_ofield);
        }
        // assert_eq!(expected,given);
    }
}
