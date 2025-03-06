use crate::{
    Element,
    quantization::TensorFielder,
    tensor::Tensor,
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use itertools::Itertools;
use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};
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

impl Default for Maxpool2D {
    fn default() -> Self {
        Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        }
    }
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

    /// Computes MLE evaluations related to proving Maxpool function.
    /// The outputs of this function are ordered as follows
    /// 1) Maxpool output
    /// 2) four polynomials corresponding to the input to the Maxpool, each with two variables fixed
    /// so that PROD (Output - poly_i) == 0 at every evaluation point.
    pub fn compute_polys<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
    ) -> Vec<Vec<E::BaseField>> {
        let output: Tensor<E> = self.op(&input).to_fields();
        let input: Tensor<E> = input.clone().to_fields();

        self.compute_polys_field(&input, &output)
    }

    pub fn compute_polys_field<E: ExtensionField>(
        &self,
        input: &Tensor<E>,
        output: &Tensor<E>,
    ) -> Vec<Vec<E::BaseField>> {
        let padded_input = input.pad_next_power_of_two();

        let padded_output = output.pad_next_power_of_two();

        let padded_input_shape = padded_input.dims();

        let num_vars = padded_input.get_data().len().ilog2() as usize;

        let mle = DenseMultilinearExtension::<E>::from_evaluations_vec(
            num_vars,
            padded_input
                .get_data()
                .iter()
                .map(|val| val.as_bases()[0])
                .collect::<Vec<E::BaseField>>(),
        );

        // This should give all possible combinations of fixing the lowest three bits in ascending order

        let fixed_mles = (0..padded_input_shape[3] << 1)
            .map(|i| {
                let point = (0..ceil_log2(padded_input_shape[3]) + 1)
                    .map(|n| E::from((i as u64 >> n) & 1))
                    .collect::<Vec<E>>();

                mle.fix_variables(&point)
            })
            .collect::<Vec<DenseMultilinearExtension<E>>>();
        // f(0,x,0,..) = x * f(0,1,0,...) + (1 - x) * f(0,0,0,...)
        let even_mles = fixed_mles
            .iter()
            .cloned()
            .step_by(2)
            .collect::<Vec<DenseMultilinearExtension<E>>>();
        let odd_mles = fixed_mles
            .iter()
            .cloned()
            .skip(1)
            .step_by(2)
            .collect::<Vec<DenseMultilinearExtension<E>>>();

        let even_merged = even_mles
            .chunks(padded_input_shape[3] >> 1)
            .map(|mle_chunk| {
                let mut mles_vec = mle_chunk
                    .iter()
                    .map(|m| {
                        m.get_ext_field_vec()
                            .iter()
                            .map(|e| e.as_bases()[0])
                            .collect::<Vec<E::BaseField>>()
                    })
                    .collect::<Vec<Vec<E::BaseField>>>();
                while mles_vec.len() > 1 {
                    let half = mles_vec.len() / 2;

                    mles_vec = mles_vec[..half]
                        .iter()
                        .zip(mles_vec[half..].iter())
                        .map(|(l, h)| {
                            l.iter()
                                .interleave(h.iter())
                                .copied()
                                .collect::<Vec<E::BaseField>>()
                        })
                        .collect::<Vec<Vec<E::BaseField>>>();
                }

                mles_vec[0].clone()
            })
            .collect::<Vec<Vec<E::BaseField>>>();

        let odd_merged = odd_mles
            .chunks(padded_input_shape[3] >> 1)
            .map(|mle_chunk| {
                let mut mles_vec = mle_chunk
                    .iter()
                    .map(|m| {
                        m.get_ext_field_vec()
                            .iter()
                            .map(|e| e.as_bases()[0])
                            .collect::<Vec<E::BaseField>>()
                    })
                    .collect::<Vec<Vec<E::BaseField>>>();
                while mles_vec.len() > 1 {
                    let half = mles_vec.len() / 2;

                    mles_vec = mles_vec[..half]
                        .iter()
                        .zip(mles_vec[half..].iter())
                        .map(|(l, h)| {
                            l.iter()
                                .interleave(h.iter())
                                .copied()
                                .collect::<Vec<E::BaseField>>()
                        })
                        .collect::<Vec<Vec<E::BaseField>>>();
                }

                mles_vec[0].clone()
            })
            .collect::<Vec<Vec<E::BaseField>>>();

        let output_mle = padded_output
            .get_data()
            .iter()
            .map(|val| val.as_bases()[0])
            .collect::<Vec<E::BaseField>>();

        [&[output_mle], even_merged.as_slice(), odd_merged.as_slice()].concat()
    }
}

#[cfg(test)]
mod tests {
    use crate::{commit::compute_betas_eval, default_transcript};

    use super::*;
    use ark_std::rand::{Rng, thread_rng};
    use crate::quantization::Fieldizer;
    use ff::Field;
    use gkr::util::ceil_log2;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{DenseMultilinearExtension, MultilinearExtension},
        virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};

    type F = GoldilocksExt2;

    #[test]
    fn test_max_pool_zerocheck() {
        let mut rng = thread_rng();
        for _ in 0..50 {
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
            let input = Tensor::<Element>::new(random_shape.clone(), data);

            let info = Maxpool2D {
                kernel_size: MAXPOOL2D_KERNEL_SIZE,
                stride: MAXPOOL2D_KERNEL_SIZE,
            };

            let output = info.op(&input);

            let padded_input = input.pad_next_power_of_two();

            let padded_output = output.pad_next_power_of_two();

            let padded_input_shape = padded_input.dims();

            let num_vars = padded_input.get_data().len().ilog2() as usize;
            let output_num_vars = padded_output.get_data().len().ilog2() as usize;

            let mle = DenseMultilinearExtension::<F>::from_evaluations_vec(
                num_vars,
                padded_input
                    .get_data()
                    .iter()
                    .map(|val_i128| {
                        let field: F = val_i128.to_field();
                        field.as_bases()[0]
                    })
                    .collect::<Vec<Goldilocks>>(),
            );

            // This should give all possible combinations of fixing the lowest three bits in ascending order

            let fixed_mles = (0..padded_input_shape[3] << 1)
                .map(|i| {
                    let point = (0..ceil_log2(padded_input_shape[3]) + 1)
                        .map(|n| F::from((i as u64 >> n) & 1))
                        .collect::<Vec<F>>();

                    mle.fix_variables(&point)
                })
                .collect::<Vec<DenseMultilinearExtension<F>>>();
            // f(0,x,0,..) = x * f(0,1,0,...) + (1 - x) * f(0,0,0,...)
            let even_mles = fixed_mles
                .iter()
                .cloned()
                .step_by(2)
                .collect::<Vec<DenseMultilinearExtension<F>>>();
            let odd_mles = fixed_mles
                .iter()
                .cloned()
                .skip(1)
                .step_by(2)
                .collect::<Vec<DenseMultilinearExtension<F>>>();

            let even_merged = even_mles
                .chunks(padded_input_shape[3] >> 1)
                .map(|mle_chunk| {
                    let mut mles_vec = mle_chunk
                        .iter()
                        .map(|m| m.get_ext_field_vec().to_vec())
                        .collect::<Vec<Vec<F>>>();
                    while mles_vec.len() > 1 {
                        let half = mles_vec.len() / 2;

                        mles_vec = mles_vec[..half]
                            .iter()
                            .zip(mles_vec[half..].iter())
                            .map(|(l, h)| {
                                l.iter().interleave(h.iter()).copied().collect::<Vec<F>>()
                            })
                            .collect::<Vec<Vec<F>>>();
                    }

                    DenseMultilinearExtension::<F>::from_evaluations_ext_slice(
                        output_num_vars,
                        &mles_vec[0],
                    )
                })
                .collect::<Vec<DenseMultilinearExtension<F>>>();

            let odd_merged = odd_mles
                .chunks(padded_input_shape[3] >> 1)
                .map(|mle_chunk| {
                    let mut mles_vec = mle_chunk
                        .iter()
                        .map(|m| m.get_ext_field_vec().to_vec())
                        .collect::<Vec<Vec<F>>>();
                    while mles_vec.len() > 1 {
                        let half = mles_vec.len() / 2;

                        mles_vec = mles_vec[..half]
                            .iter()
                            .zip(mles_vec[half..].iter())
                            .map(|(l, h)| {
                                l.iter().interleave(h.iter()).copied().collect::<Vec<F>>()
                            })
                            .collect::<Vec<Vec<F>>>();
                    }

                    DenseMultilinearExtension::<F>::from_evaluations_ext_slice(
                        output_num_vars,
                        &mles_vec[0],
                    )
                })
                .collect::<Vec<DenseMultilinearExtension<F>>>();

            let merged_input_mles = [even_merged, odd_merged].concat();

            let output_mle = DenseMultilinearExtension::<F>::from_evaluations_ext_vec(
                output_num_vars,
                padded_output
                    .get_data()
                    .iter()
                    .map(|val_i128| {
                        let field: F = val_i128.to_field();
                        field
                    })
                    .collect::<Vec<F>>(),
            );

            let mut vp = VirtualPolynomial::<F>::new(output_num_vars);

            let diff_mles = merged_input_mles
                .iter()
                .map(|in_mle| {
                    DenseMultilinearExtension::<F>::from_evaluations_ext_vec(
                        output_num_vars,
                        in_mle
                            .get_ext_field_vec()
                            .iter()
                            .zip(output_mle.get_ext_field_vec().iter())
                            .map(|(input, output)| *output - *input)
                            .collect::<Vec<F>>(),
                    )
                    .into()
                })
                .collect::<Vec<ArcMultilinearExtension<F>>>();

            (0..1 << output_num_vars).for_each(|j| {
                let values = diff_mles
                    .iter()
                    .map(|mle| mle.get_ext_field_vec()[j])
                    .collect::<Vec<F>>();
                assert_eq!(values.iter().product::<F>(), F::ZERO)
            });

            vp.add_mle_list(diff_mles, F::ONE);

            let random_point = (0..output_num_vars)
                .map(|_| F::random(&mut rng))
                .collect::<Vec<F>>();

            let beta_evals = compute_betas_eval(&random_point);

            let beta_mle: ArcMultilinearExtension<F> =
                DenseMultilinearExtension::<F>::from_evaluations_ext_vec(
                    output_num_vars,
                    beta_evals,
                )
                .into();
            vp.mul_by_mle(beta_mle.clone(), Goldilocks::ONE);

            let aux_info = vp.aux_info.clone();

            let mut prover_transcript = default_transcript::<F>();

            #[allow(deprecated)]
            let (proof, state) = IOPProverState::<F>::prove_parallel(vp, &mut prover_transcript);

            let mut verifier_transcript = default_transcript::<F>();

            let subclaim =
                IOPVerifierState::<F>::verify(F::ZERO, &proof, &aux_info, &mut verifier_transcript);

            let point = subclaim
                .point
                .iter()
                .map(|chal| chal.elements)
                .collect::<Vec<F>>();

            let fixed_points = [[F::ZERO, F::ZERO], [F::ZERO, F::ONE], [F::ONE, F::ZERO], [
                F::ONE,
                F::ONE,
            ]]
            .map(|pair| {
                [
                    &[pair[0]],
                    &point[..ceil_log2(padded_input_shape[3]) - 1],
                    &[pair[1]],
                    &point[ceil_log2(padded_input_shape[3]) - 1..],
                ]
                .concat()
            });

            let output_eval = output_mle.evaluate(&point);
            let input_evals = fixed_points
                .iter()
                .map(|p| mle.evaluate(p))
                .collect::<Vec<F>>();

            let eq_eval = beta_mle.evaluate(&point);

            let calc_eval = input_evals
                .iter()
                .map(|ie| output_eval - *ie)
                .product::<F>()
                * eq_eval;

            assert_eq!(calc_eval, subclaim.expected_evaluation);

            // in order output - 00, output - 10, output - 01, output - 11, eq I believe
            let final_mle_evals = state.get_mle_final_evaluations();

            let [r1, r2] = [F::random(&mut rng); 2];
            let one_minus_r1 = F::ONE - r1;
            let one_minus_r2 = F::ONE - r2;

            let maybe_eval = (output_eval - final_mle_evals[0]) * one_minus_r1 * one_minus_r2
                + (output_eval - final_mle_evals[2]) * one_minus_r1 * r2
                + (output_eval - final_mle_evals[1]) * r1 * one_minus_r2
                + (output_eval - final_mle_evals[3]) * r1 * r2;

            let mle_eval = mle.evaluate(
                &[
                    &[r1],
                    &point[..ceil_log2(padded_input_shape[3]) - 1],
                    &[r2],
                    &point[ceil_log2(padded_input_shape[3]) - 1..],
                ]
                .concat(),
            );

            assert_eq!(mle_eval, maybe_eval);
        }
    }
}
