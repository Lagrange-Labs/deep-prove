use crate::{
    quantization::{self, Requant},
    tensor::ConvData,
};
use ff_ext::ExtensionField;

use crate::{Element, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Convolution {
    pub filter: Tensor<Element>,
    pub bias: Tensor<Element>,
}

impl Convolution {
    pub fn new(filter: Tensor<Element>, bias: Tensor<Element>) -> Self {
        assert_eq!(filter.kw(), bias.dims()[0]);
        Self { filter, bias }
    }
    pub fn add_bias(&self, conv_out: &Tensor<Element>) -> Tensor<Element> {
        let mut arr = conv_out.data.clone();
        assert_eq!(conv_out.data.len(), conv_out.kw() * conv_out.filter_size());
        for i in 0..conv_out.kw() {
            for j in 0..conv_out.filter_size() {
                arr[i * conv_out.filter_size() + j] += self.bias.data[i];
            }
        }
        Tensor::new(conv_out.get_shape(), arr)
    }

    pub fn op<E: ExtensionField>(&self, input: &Tensor<Element>) -> (Tensor<Element>, ConvData<E>) {
        let (output, proving_data) = self.filter.fft_conv(input);
        (self.add_bias(&output), proving_data)
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.filter.get_shape()
    }

    pub fn kw(&self) -> usize {
        self.filter.kw()
    }

    pub fn kx(&self) -> usize {
        self.filter.kx()
    }

    pub fn nw(&self) -> usize {
        self.filter.nw()
    }

    pub fn ncols_2d(&self) -> usize {
        self.filter.ncols_2d()
    }

    pub fn nrows_2d(&self) -> usize {
        self.filter.nrows_2d()
    }
    pub fn filter_size(&self) -> usize {
        self.filter.filter_size()
    }
    pub fn requant_info<E: ExtensionField>(&self) -> Requant {
        let weights = self.filter.get_real_weights::<E>();
        let min_quant = *quantization::MIN as Element;
        let max_quant = *quantization::MAX as Element;

        let mut max_range: u128 = 0;
        let max_bias = self.bias.get_data().iter().max().unwrap();

        // Keep the original iteration order: first over kernel height (i), then kernel width (j), then output channels (k)
        for i in 0..self.kw() {
            for j in 0..self.kx() {
                let mut min_temp: Element = 0;
                let mut max_temp: Element = 0;

                // Loop over output channels (k) and apply weights and bias
                for k in 0..(self.nw() * self.nw()) {
                    let weight = weights[i][j][k];
                    // PANICKING HERE
                    //let bias = self.bias.data[k]; // Bias for the current output channel
                    let bias = *max_bias;

                    if weight != 0 {
                        let (min_contrib, max_contrib) = if weight < 0 {
                            (max_quant * weight, min_quant * weight)
                        } else {
                            (min_quant * weight, max_quant * weight)
                        };

                        min_temp += min_contrib;
                        max_temp += max_contrib;
                    }

                    // Add the bias for this output channel `k`
                    min_temp += bias;
                    max_temp += bias;
                }

                // After processing all output channels for this (i, j) location, update the global min and max
                max_range = max_range.max((max_temp - min_temp).unsigned_abs().next_power_of_two());
            }
        }
        assert!(max_range.ilog2() as usize > *quantization::BIT_LEN);
        // TODO: this is wrong, as the formula comes from dense layer and not conv
        // see https://www.notion.so/lagrangelabs/sumcheck-example-19028d1c65a880a3abf6ca37318148ea?pvs=4#1ab28d1c65a88040b076e511ff98ae39
        let shift = max_range.ilog2() as usize - *quantization::BIT_LEN;

        // let ind_range = (*quantization::MAX as i64 - *quantization::MIN as i64) as usize;
        // let max_range = (2*ind_range + self.filter.ncols_2d() as usize * ind_range).next_power_of_two();
        // let shift = (max_range.ilog2() as usize) - (*quantization::BIT_LEN as usize);
        Requant {
            // range: (max_val - min_val) as usize,
            range: max_range as usize,
            right_shift: shift,
            after_range: 1 << *quantization::BIT_LEN,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;

    fn random_vector_quant(n: usize) -> Vec<Element> {
        vec![thread_rng().gen_range(-128..128); n]
    }

    #[test]
    pub fn test_quantization() {
        let n_w = 1 << 2;
        let k_w = 1 << 0;
        let n_x = 1 << 3;
        let k_x = 1 << 0;

        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        let w1 = random_vector_quant(k_w * k_x * n_w * n_w);

        let conv = Convolution::new(
            Tensor::new_conv(
                vec![k_w, k_x, n_w, n_w],
                in_dimensions[0].clone(),
                w1.clone(),
            ),
            Tensor::new(vec![k_w], random_vector_quant(k_w)),
        );
        let info = conv.requant_info::<GoldilocksExt2>();
        println!("range : {}", info.range);
        for _ in 0..100 {
            let (out, _proving_data) = conv.op::<GoldilocksExt2>(&Tensor::new(
                vec![k_x, n_x, n_x],
                random_vector_quant(k_x * n_x * n_x),
            ));
            for j in 0..out.data.len() {
                if out.data[j] < 0 {
                    assert!((-out.data[j] as usize) < info.range);
                } else {
                    assert!((out.data[j] as usize) < info.range);
                }
            }
        }
    }
}
