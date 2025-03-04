use crate::{Element, quantization::Fieldizer, tensor::Tensor};
use ff_ext::ExtensionField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const MAXPOOL2D_KERNEL_SIZE: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Pooling {
    Maxpool2D(Maxpool2D),
}

impl Pooling {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        match self {
            Pooling::Maxpool2D(maxpool2d) => maxpool2d.op(input),
        }
    }
}

/// Information about a maxpool2d step
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub struct Maxpool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl Maxpool2D {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        assert!(
            self.kernel_size == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for kernel size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        assert!(
            self.stride == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for stride size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        input.maxpool2d(self.kernel_size, self.stride)
    }

    pub fn compute_diff_poly<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
    ) -> Vec<E::BaseField> {
        let shape = input.dims();

        let (_, padded) = input.padded_maxpool2d();

        let diff = padded
            .get_data()
            .par_iter()
            .zip(input.get_data().par_iter())
            .map(|(a, b)| {
                let field_elem: E = (a - b).to_field();
                field_elem.as_bases()[0]
            })
            .collect::<Vec<E::BaseField>>();

        let basefield_tensor = Tensor::<E::BaseField>::new(shape, diff);

        basefield_tensor.pad_next_power_of_two().get_data().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::{Rng, thread_rng};
    use ff::Field;
    use goldilocks::{Goldilocks, GoldilocksExt2, SmallField};
    use itertools::izip;
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};

    type F = GoldilocksExt2;
    #[test]
    fn test_diff_poly() {
        println!("goldilocks modulus: {}", Goldilocks::MODULUS_U64);
        println!("18446744069423824321");
        let mut rng = thread_rng();
        for _ in 0..10 {
            let random_shape = (0..4)
                .map(|i| {
                    if i < 2 {
                        rng.gen_range(2usize..6)
                    } else {
                        2 * rng.gen_range(2usize..5)
                    }
                })
                .collect::<Vec<usize>>();
            let input_data_size = random_shape.iter().product::<usize>();
            let data = (0..input_data_size)
                .map(|_| rng.gen_range(-128i128..128))
                .collect::<Vec<Element>>();
            let input = Tensor::<Element>::new(random_shape, data);

            let info = Maxpool2D {
                kernel_size: MAXPOOL2D_KERNEL_SIZE,
                stride: MAXPOOL2D_KERNEL_SIZE,
            };

            let diff_evals = info.compute_diff_poly::<F>(&input);
            let num_vars = diff_evals.len().ilog2() as usize;

            let mle = DenseMultilinearExtension::<F>::from_evaluations_vec(num_vars, diff_evals);

            let mle_00 = mle.fix_high_variables(&[F::ZERO, F::ZERO]);
            let mle_01 = mle.fix_high_variables(&[F::ZERO, F::ONE]);
            let mle_10 = mle.fix_high_variables(&[F::ONE, F::ZERO]);
            let mle_11 = mle.fix_high_variables(&[F::ONE, F::ONE]);

            // Check that their product is zero
            izip!(
                mle_00.get_ext_field_vec(),
                mle_01.get_ext_field_vec(),
                mle_10.get_ext_field_vec(),
                mle_11.get_ext_field_vec()
            )
            .for_each(|(&m00, &m01, &m10, &m11)| assert_eq!(m00 * m01 * m10 * m11, F::ZERO));
        }
    }
}
