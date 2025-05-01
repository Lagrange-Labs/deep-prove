//! Module containing the code for the Softmax function that maps a real vector space to the space of probability distributions.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    marker::PhantomData,
    ops::AddAssign,
};

use rayon::prelude::*;

use crate::{
    ScalingFactor,
    tensor::{Number, Tensor, TensorError},
};

#[derive(Copy, Clone, Debug)]
/// Struct that stores all the relevant information for the Softmax function.
/// For a tensor this function takes one of its dimensions and maps that dimension to a probability
/// distribution based on the current values.
///
/// For example, let `z` be a sub-tensor with `k` elements, all in the chosen dimension. Then `Softmax(z)` is defined as
///
///     `Softmax(z) = `(exp(z_{i}) / SUM_{j = 1}^{k} exp(z_{j})_{i = 1}^{k}`
///
/// Softmax is always performed on [`f32`] data types, for this reason we store an optional input and output [`ScalingFactor`] for when we
/// have to perform the operation on quantised inputs.
pub struct Softmax<T> {
    /// When calculating Softmax this is the constant value we divide inputs by before exponentiating.
    /// When viewed as a porbability distribution (specifically a Boltzmann distribution) this is referred to as the "temprature".
    pub(crate) scaling_factor: f32,
    /// The dimension that we normalise on. After applying Softmax the sum of the elements in this dimension will be `1`.
    pub(crate) dim: usize,
    /// The scaling factors that will be used to dequantise inputs before applying Softmax and requantise after application
    pub(crate) quant_params: Option<(ScalingFactor, ScalingFactor)>,
    _phantom: PhantomData<T>,
}

impl<T: Number> Softmax<T> {
    /// Create a new instance of [`Softmax`]. This method does not set any quantisation parameters.
    pub fn new(scaling_factor: f32, dim: usize) -> Softmax<T> {
        Softmax {
            scaling_factor,
            dim,
            quant_params: None,
            _phantom: PhantomData::<T>,
        }
    }
    /// Quantises the operation, if `quant_params` are already set this method does nothing.
    pub fn quantise(
        self,
        input_quant_params: ScalingFactor,
        output_quant_params: ScalingFactor,
    ) -> Softmax<T> {
        if self.is_quantised() {
            self
        } else {
            let Softmax {
                scaling_factor,
                dim,
                _phantom,
                ..
            } = self;

            Softmax {
                scaling_factor,
                dim,
                quant_params: Some((input_quant_params, output_quant_params)),
                _phantom,
            }
        }
    }
    /// Internal method used to determine if the operation has already been quantised.
    fn is_quantised(&self) -> bool {
        self.quant_params.is_some()
    }
}

impl Softmax<f32> {
    /// Method that permforms the Softmax operation on a [`Tensor`]
    pub fn op(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, SoftmaxError> {
        // Check that the tensor has enough dimensions
        let input_shape = input.get_shape();
        if self.dim >= input_shape.len() {
            return Err(SoftmaxError::ParameterError(format!(
                "Cannot perform Softmax on the {} dimension since input only has {} dimensions",
                self.dim,
                input_shape.len()
            )));
        }
        // Retrieve the maximum value in the tensor
        let max = input.max_value();
        // Now we have to group the tensor elements by the specified dimension, this is handled by the `SoftmaxFoldMap` struct
        let fold_map = input
            .get_data()
            .par_iter()
            .enumerate()
            .try_fold(
                || SoftmaxFoldMap::new(&input_shape, self.dim),
                |mut acc, (i, value)| {
                    acc.update(*value - max, i)?;
                    Result::<SoftmaxFoldMap, SoftmaxError>::Ok(acc)
                },
            )
            .try_reduce(
                || SoftmaxFoldMap::new(&input_shape, self.dim),
                |a, b| a.merge(b),
            )?;

        let (tensor, _) = fold_map.finalise()?;

        Ok(tensor)
    }
}

/// Helper function that given an index provided by the `.enumerate()` method converts it into the tensor coordinate
fn index_to_coord(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coord = shape
        .iter()
        .rev()
        .scan(1usize, |state, dim| {
            let tmp = index - (index % *state);
            let tmp = tmp / *state;
            let coord = tmp % *dim;
            *state *= *dim;
            Some(coord)
        })
        .collect::<Vec<usize>>();

    coord.reverse();

    coord
}

#[derive(Debug, Clone)]
/// Error enum for the [`Softmax`] operation.
pub enum SoftmaxError {
    /// Returned when parameters to a [`Softmax`] method are incorrect
    ParameterError(String),
    /// Returned if an error occurs in proving
    ProvingError(String),
    /// Returned if an unsupported type is passed to a method
    TypeError(String),
}

impl Display for SoftmaxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            SoftmaxError::ParameterError(s) => {
                write!(f, "Parameters for Softmax were incorrect: {}", s)
            }
            SoftmaxError::ProvingError(s) => {
                write!(f, "Proving error occured during Softmax proving: {}", s)
            }
            SoftmaxError::TypeError(s) => {
                write!(f, "Unsupported type passed to Softmax operation: {}", s)
            }
        }
    }
}

impl Error for SoftmaxError {}

impl From<TensorError> for SoftmaxError {
    fn from(e: TensorError) -> SoftmaxError {
        SoftmaxError::ParameterError(format!("Inner Tensor error: {}", e))
    }
}

#[derive(Clone, Debug)]
/// Struct used to help with iterating over [`Tensor`] data when performing the [`Softmax`] operation.
struct SoftmaxFoldMap {
    /// The shape of the input tensor to the softmax operation
    input_shape: Vec<usize>,
    /// The dimension we are normalising over
    dim: usize,
    /// This [`Tensor`] holds the running sums over each dimension so we can normalize at the end
    sums: Tensor<f32>,
    /// This vector holds the mapped input values (i.e. they have been exponentiated)
    mapped: Vec<f32>,
}

impl SoftmaxFoldMap {
    /// Creates a new [`SoftmaxFoldMap`]
    fn new(input_shape: &[usize], dim: usize) -> SoftmaxFoldMap {
        let mut sum_shape = input_shape.to_vec();
        sum_shape.remove(dim);

        let sum_data = vec![0.0f32; sum_shape.iter().product::<usize>()];
        SoftmaxFoldMap {
            input_shape: input_shape.to_vec(),
            dim,
            sums: Tensor::<f32>::new(sum_shape, sum_data),
            mapped: vec![],
        }
    }

    /// Updates the struct, here value should already have been rescaled by any constants.
    fn update(&mut self, value: f32, index: usize) -> Result<(), SoftmaxError> {
        let mut sum_coord = index_to_coord(index, &self.input_shape);
        sum_coord.remove(self.dim);

        let exp = value.exp();
        self.mapped.push(exp);
        self.sums
            .get_from_coord_mut(&sum_coord)
            .and_then(|sum| {
                sum.add_assign(exp);
                Ok(())
            })
            .map_err(SoftmaxError::from)
    }

    /// Merges two [`SoftmaxFoldMap`] instances into a single one.
    fn merge(self, other: SoftmaxFoldMap) -> Result<SoftmaxFoldMap, SoftmaxError> {
        let SoftmaxFoldMap {
            input_shape,
            dim,
            sums,
            mapped,
        } = self;
        let SoftmaxFoldMap {
            input_shape: other_input_shape,
            dim: other_dim,
            sums: other_sums,
            mapped: other_mapped,
        } = other;

        if input_shape != input_shape {
            return Err(SoftmaxError::ParameterError(format!(
                "Cannot merge SoftmaxFoldMaps with different input shapes, left: {:?}, right: {:?}",
                input_shape, other_input_shape
            )));
        }

        if dim != other_dim {
            return Err(SoftmaxError::ParameterError(format!(
                "Cannot merge SoftmaxFoldMaps the normalise over different dimensions, left: {}, right: {}",
                dim, other_dim
            )));
        }

        let Tensor { data, shape, .. } = sums;
        let Tensor {
            data: other_data, ..
        } = other_sums;

        let new_data = data
            .into_iter()
            .zip(other_data.into_iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<f32>>();

        let sums = Tensor::<f32>::new(shape, new_data);

        let mapped = mapped
            .into_iter()
            .chain(other_mapped.into_iter())
            .collect::<Vec<f32>>();

        Ok(SoftmaxFoldMap {
            input_shape,
            dim,
            sums,
            mapped,
        })
    }

    /// This method iterates over the data stored in self and divides each term of `self.mapped` by the correct value in `self.sums`.
    /// It returns both the output [`Tensor`] and `self.sums`, which will be useful in proving.
    fn finalise(self) -> Result<(Tensor<f32>, Tensor<f32>), SoftmaxError> {
        let SoftmaxFoldMap {
            input_shape,
            dim,
            sums,
            mapped,
        } = self;

        let data = mapped
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                let mut divisor_coord = index_to_coord(i, &input_shape);
                divisor_coord.remove(dim);
                let divisor = sums.get_from_coord(&divisor_coord)?;
                Ok(val / *divisor)
            })
            .collect::<Result<Vec<f32>, SoftmaxError>>()?;

        Ok((Tensor::<f32>::new(input_shape.to_vec(), data), sums))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use tract_onnx::{
        prelude::{DatumType, IntoArcTensor, tvec},
        tract_core::{tract_data::prelude::Tensor as TractTensor, value::TValue},
    };

    use crate::testing::load_test_onnx_model;

    use super::*;

    #[test]
    fn softmax_op_test_helper() -> anyhow::Result<()> {
        let shape: Vec<usize> = vec![2, 3, 4];

        for _ in 0..20 {
            let tensor = Tensor::<f32>::random(&shape);

            // First we get tract to run the test model with the random input to compare against
            let tract_tensor = TractTensor::from_shape::<f32>(&shape, tensor.get_data())?;

            let tract_input: TValue = tract_tensor.into();
            let model = load_test_onnx_model("softmax")?;

            let runnable_model = model.into_runnable()?;

            let tract_outputs = runnable_model.run(tvec!(tract_input))?;

            let expected_output_tensors = tract_outputs
                .into_iter()
                .filter_map(|value| {
                    let tract_tensor = value.into_arc_tensor();
                    let out_shape = tract_tensor.shape().to_vec();

                    let dt = tract_tensor.datum_type();

                    if let DatumType::F32 = dt {
                        let data = tract_tensor.as_slice::<f32>().unwrap().to_vec();
                        Some(Tensor::<f32>::new(out_shape, data))
                    } else {
                        None
                    }
                })
                .collect::<Vec<Tensor<f32>>>();

            if expected_output_tensors.len() != 1 {
                return Err(anyhow::anyhow!("Got more than one output"));
            }

            // Now create our `Softmax` struct and run the operation with that
            let softmax = Softmax::<f32>::new(1.0f32, 1);

            let our_output = softmax.op(&tensor)?;

            let shape_check = our_output.get_shape() == expected_output_tensors[0].get_shape();

            if !shape_check {
                return Err(anyhow!(
                    "The calculated tensor and expected tensor have different shapes"
                ));
            }

            let data_checker = our_output
                .get_data()
                .iter()
                .zip(expected_output_tensors[0].get_data().iter())
                .fold(true, |acc, (&calc, &expect)| {
                    let abs_diff = (calc - expect).abs();
                    let small = abs_diff < f32::EPSILON;
                    acc && small
                });

            if !data_checker {
                return Err(anyhow!(
                    "The calculated tensor data and expected tensor data differed by a large amount"
                ));
            }
        }
        Ok(())
    }
}
