//! Module containing the code for the Softmax function that maps a real vector space to the space of probability distributions.

use std::{
    any::TypeId,
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    marker::PhantomData,
    ops::AddAssign,
};

use rayon::prelude::*;

use crate::{
    Element, ScalingFactor,
    tensor::{Number, Tensor, TensorError, cast_tensor},
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
    pub fn quantise<Q: Number>(
        self,
        input_quant_params: ScalingFactor,
        output_quant_params: ScalingFactor,
    ) -> Softmax<Q> {
        if self.is_quantised() {
            let Softmax {
                scaling_factor,
                dim,
                quant_params,
                _phantom,
            } = self;

            Softmax {
                scaling_factor,
                dim,
                quant_params,
                _phantom: PhantomData::<Q>,
            }
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
                _phantom: PhantomData::<Q>,
            }
        }
    }
    /// Applies [`Softmax`] to the input [`Tensor`]
    pub fn op(&self, input: &Tensor<T>) -> Result<Tensor<T>, SoftmaxError>
    where
        T: 'static,
    {
        // First we have work out what type the input tensor was, if its anything other than f32 or Element we throw an error
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let float_tensor = cast_tensor::<_, f32>(input.clone());
            let (output_f32, _) = self.eval_softmax(&float_tensor)?;
            let output = cast_tensor::<f32, T>(output_f32);
            Ok(output)
        } else if TypeId::of::<Element>() == TypeId::of::<T>() {
            let quant_tensor = cast_tensor::<_, Element>(input.clone());
            let (quant_in, quant_out) = self.quant_params.ok_or(SoftmaxError::ParameterError(
                "Cannot apply Softmax to quantised input if quant params have not been set"
                    .to_string(),
            ))?;
            let float_tensor = quant_tensor.dequantize(&quant_in);
            let (float_output, _) = self.eval_softmax(&float_tensor)?;
            let quant_output = float_output.quantize(&quant_out);
            Ok(cast_tensor::<Element, T>(quant_output))
        } else {
            Err(SoftmaxError::TypeError(
                "Cannot apply Softmax to tensor with inner data type that is not f32 or Element"
                    .to_string(),
            ))
        }
    }

    /// Internal method used to determine if the operation has already been quantised.
    fn is_quantised(&self) -> bool {
        self.quant_params.is_some()
    }
}

impl<T: Number> Softmax<T> {
    /// Internal method to perform [`Softmax`] on a [`Tensor`] of [`f32`] values.
    fn eval_softmax(
        &self,
        input: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), SoftmaxError> {
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

        fold_map.finalise()
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
    use crate::testing::load_test_onnx_model;
    use anyhow::anyhow;
    use goldilocks::GoldilocksExt2 as F;
    use tract_onnx::{
        prelude::{DatumType, IntoArcTensor, tvec},
        tract_core::{tract_data::prelude::Tensor as TractTensor, value::TValue},
    };

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

            let (our_output, _) = softmax.eval_softmax(&tensor)?;

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

    #[test]
    fn test_type_error() -> anyhow::Result<()> {
        let softmax = Softmax::<F>::new(1.0f32, 1);
        let shape: Vec<usize> = vec![2, 3, 4];

        let tensor = Tensor::<F>::random(&shape);

        let res = softmax.op(&tensor);

        if let Err(e) = res {
            println!("Got error: {}", e);
            Ok(())
        } else {
            Err(anyhow!(
                "Did not get an error when trying to run Softmax on a field tensor"
            ))
        }
    }

    #[test]
    fn test_quantised_softmax() -> anyhow::Result<()> {
        let shape = vec![2, 3, 4];
        let softmax = Softmax::<f32>::new(1.0f32, 1);
        for _ in 0..20 {
            // Make a random float input and calculate its output.
            let input_tensor = Tensor::<f32>::random(&shape);
            let output_tensor = softmax.op(&input_tensor)?;

            let sums = output_tensor.get_data().iter().enumerate().fold(
                vec![0.0f32; 8],
                |mut acc, (i, val)| {
                    let mut coord = index_to_coord(i, &shape);
                    coord.remove(1);
                    let slot = coord[0] * 4 + coord[1];
                    acc[slot] += *val;
                    acc
                },
            );

            for dim_sum in sums.into_iter() {
                println!("Sum was: {}", dim_sum);
            }

            // Work out the quantisation params and create the quantised softmax operator
            let input_max = input_tensor.max_abs_output();
            let output_max = output_tensor.max_abs_output();

            let quant_in = ScalingFactor::from_absolute_max(input_max, None);
            let quant_out = ScalingFactor::from_absolute_max(output_max, None);

            let quant_softmax = softmax.clone().quantise::<Element>(quant_in, quant_out);

            // Now to test that we get an accurate result we run the quantised operation on a quantised tensor and also
            // run the floating point operation on a tensor that has been quantised and then dequantised. After dequantising
            // the quantised output it should agree with the floating point output.
            let quant_input_tensor = input_tensor.quantize(&quant_in);
            let dequant_input_tensor = quant_input_tensor.dequantize(&quant_in);

            let quant_output = quant_softmax.op(&quant_input_tensor)?;
            // Check that the quant output sums to quantised 1 along the correct dimension
            let sums = quant_output.get_data().iter().enumerate().fold(
                vec![0i128; 8],
                |mut acc, (i, val)| {
                    let mut coord = index_to_coord(i, &shape);
                    coord.remove(1);
                    let slot = coord[0] * 4 + coord[1];
                    acc[slot] += *val;
                    acc
                },
            );

            let quant_one = quant_out.quant_max();
            let one = quant_out.quantize(&1.0f32);
            println!("one: {}", one);
            println!("Quantised 1 = {}", quant_one);
            for dim_sum in sums.into_iter() {
                println!("Sum was: {}", dim_sum);
            }
            let dequant_output = quant_output.dequantize(&quant_out);
            // Check that the quant output sums to quantised 1 along the correct dimension
            let sums = dequant_output.get_data().iter().enumerate().fold(
                vec![0.0f32; 8],
                |mut acc, (i, val)| {
                    let mut coord = index_to_coord(i, &shape);
                    coord.remove(1);
                    let slot = coord[0] * 4 + coord[1];
                    acc[slot] += *val;
                    acc
                },
            );

            for dim_sum in sums.into_iter() {
                println!("Sum was: {}", dim_sum);
            }

            let expected_dequant_output = softmax.op(&dequant_input_tensor)?;

            let shape_check = dequant_output.get_shape() == expected_dequant_output.get_shape();

            if !shape_check {
                return Err(anyhow!(
                    "The calculated tensor and expected tensor have different shapes"
                ));
            }

            let data_checker = dequant_output
                .get_data()
                .iter()
                .zip(expected_dequant_output.get_data().iter())
                .fold(true, |acc, (&calc, &expect)| {
                    // println!("calc: {}, expect: {}", calc, expect);
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
