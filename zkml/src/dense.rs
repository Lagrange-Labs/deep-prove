use crate::{quantization, quantization::Requant};
use itertools::Itertools;

use crate::{Element, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Dense {
    pub matrix: Tensor<Element>,
    pub bias: Tensor<Element>,
}

impl Dense {
    pub fn new(matrix: Tensor<Element>, bias: Tensor<Element>) -> Self {
        assert_eq!(matrix.nrows_2d(), bias.dims()[0]);
        Self { matrix, bias }
    }
    pub fn ncols(&self) -> usize {
        self.matrix.ncols_2d()
    }
    pub fn nrows(&self) -> usize {
        self.matrix.nrows_2d()
    }

    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        self.matrix.matvec(input).add(&self.bias)
    }

    pub fn pad_next_power_of_two(self) -> Self {
        let matrix = self.matrix.pad_next_power_of_two_2d();
        let bias = self.bias.pad_1d(matrix.nrows_2d());
        Self::new(matrix, bias)
    }

    pub fn requant_info(&self) -> Requant {
        let ncols = self.matrix.ncols_2d();
        let max_output_range = self
            .matrix
            .get_data()
            .iter()
            .chunks(ncols)
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let row_range = row
                    .map(|w| quantization::range_from_weight(w))
                    .fold((0, 0), |(min, max), (wmin, wmax)| (min + wmin, max + wmax));
                // add the bias range - so take the weight corresponding to the row index
                let bias_weight = &self.bias.get_data()[i];
                let total_range = (row_range.0 + bias_weight, row_range.1 + bias_weight);
                // weight * MIN can be positive and higher then MAX*weight if weight's negative
                // so we take the absolute value of the difference
                (total_range.1 - total_range.0).unsigned_abs() as usize
            })
            .max()
            .expect("No max range found")
            .next_power_of_two();
        let shift = max_output_range.ilog2() as usize - *quantization::BIT_LEN;
        Requant {
            range: max_output_range,
            right_shift: shift,
            after_range: 1 << *quantization::BIT_LEN,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl Dense {
        pub fn random(shape: Vec<usize>) -> Self {
            assert_eq!(shape.len(), 2);
            let (nrows, ncols) = (shape[0], shape[1]);
            let matrix = Tensor::random(vec![nrows, ncols]);
            let bias = Tensor::random(vec![nrows]);
            Self::new(matrix, bias)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::quantization::Quantizer;

        #[test]
        fn test_dense_pad_next_power_of_two() {
            // Create a Dense layer with non-power-of-two dimensions
            let matrix =
                Tensor::<Element>::from_coeffs_2d(vec![vec![1, 2, 3], vec![4, 5, 6], vec![
                    7, 8, 9,
                ]])
                .unwrap();

            let bias = Tensor::<Element>::new(vec![3], vec![10, 11, 12]);

            let dense = Dense::new(matrix, bias);

            // Pad to next power of two
            let padded = dense.pad_next_power_of_two();

            // Check padded dimensions are powers of two
            let padded_dims = padded.matrix.dims();
            assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
            assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

            // Check bias is padded
            let bias_dims = padded.bias.dims();
            assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

            // Check original values are preserved
            assert_eq!(padded.matrix.get_data()[0], 1);
            assert_eq!(padded.matrix.get_data()[1], 2);
            assert_eq!(padded.matrix.get_data()[2], 3);
            assert_eq!(padded.matrix.get_data()[4], 4);
            assert_eq!(padded.matrix.get_data()[8], 7);

            // Check added values are zeros
            assert_eq!(padded.matrix.get_data()[3], 0);
            assert_eq!(padded.matrix.get_data()[7], 0);
            assert_eq!(padded.matrix.get_data()[15], 0);

            // Check bias values
            assert_eq!(padded.bias.get_data()[0], 10);
            assert_eq!(padded.bias.get_data()[1], 11);
            assert_eq!(padded.bias.get_data()[2], 12);
            assert_eq!(padded.bias.get_data()[3], 0); // Padding
        }

        #[test]
        fn test_dense_pad_already_power_of_two() {
            // Create a Dense layer with power-of-two dimensions
            let matrix = Tensor::<Element>::from_coeffs_2d(vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
            ])
            .unwrap();

            let bias = Tensor::<Element>::new(vec![4], vec![20, 21, 22, 23]);

            let dense = Dense::new(matrix, bias);

            // Pad to next power of two
            let padded = dense.clone().pad_next_power_of_two();

            // Check dimensions remain the same
            let padded_dims = padded.matrix.dims();
            assert_eq!(padded_dims[0], 4);
            assert_eq!(padded_dims[1], 4);

            // Check bias dimensions remain the same
            let bias_dims = padded.bias.dims();
            assert_eq!(bias_dims[0], 4);

            // Check values are preserved
            for i in 0..16 {
                assert_eq!(padded.matrix.get_data()[i], dense.matrix.get_data()[i]);
            }

            for i in 0..4 {
                assert_eq!(padded.bias.get_data()[i], dense.bias.get_data()[i]);
            }
        }

        #[test]
        fn test_dense_pad_mixed_dimensions() {
            // Create a Dense layer with one power-of-two dimension and one non-power-of-two
            let matrix =
                Tensor::<Element>::from_coeffs_2d(vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![
                    9, 10, 11, 12,
                ]])
                .unwrap();

            let bias = Tensor::<Element>::new(vec![3], vec![20, 21, 22]);

            let dense = Dense::new(matrix, bias);

            // Pad to next power of two
            let padded = dense.pad_next_power_of_two();

            // Check dimensions are padded correctly
            let padded_dims = padded.matrix.dims();
            assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
            assert_eq!(padded_dims[1], 4); // Already a power of 2

            // Check bias is padded
            let bias_dims = padded.bias.dims();
            assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

            // Check original values are preserved and padding is zeros
            assert_eq!(padded.matrix.get_data()[0], 1);
            assert_eq!(padded.matrix.get_data()[4], 5);
            assert_eq!(padded.matrix.get_data()[8], 9);
            assert_eq!(padded.matrix.get_data()[12], 0); // Padding

            // Check bias values
            assert_eq!(padded.bias.get_data()[0], 20);
            assert_eq!(padded.bias.get_data()[1], 21);
            assert_eq!(padded.bias.get_data()[2], 22);
            assert_eq!(padded.bias.get_data()[3], 0); // Padding
        }

        #[test]
        fn test_quantization_with_padded_dense() {
            // Create input data that needs quantization
            let input_data = vec![0.5f32, -0.3f32, 0.1f32];

            // Quantize the input
            let quantized_input: Vec<Element> = input_data
                .iter()
                .map(|x| Element::from_f32_unsafe(x))
                .collect();

            // Create a Dense layer
            let matrix =
                Tensor::<Element>::from_coeffs_2d(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

            let bias = Tensor::<Element>::new(vec![2], vec![10, 11]);

            let dense = Dense::new(matrix, bias);

            // Pad the dense layer
            let padded = dense.clone().pad_next_power_of_two();

            // Create input tensor
            let input_tensor = Tensor::<Element>::new(vec![3], quantized_input);

            // Apply the dense operation on both original and padded
            let output = dense.op(&input_tensor);
            let padded_output = padded.op(&input_tensor.pad_1d(4));

            // Check that the result is correct (for the non-padded parts)
            for i in 0..2 {
                assert_eq!(output.get_data()[i], padded_output.get_data()[i]);
            }
        }
    }
}
