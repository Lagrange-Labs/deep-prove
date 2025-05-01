//! Module containing the code for the Softmax function that maps a real vector space to the space of probability distributions.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    marker::PhantomData,
};

use rayon::prelude::*;

use crate::{
    ScalingFactor,
    tensor::{Number, Tensor},
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

        // Now we have to group the tensor elements by the specified dimension
        let d = input_shape[self.dim];
        let right_size = input_shape[self.dim..].iter().skip(1).product::<usize>();

        let fold_map = input.get_data().iter().enumerate().try_fold(
            SoftmaxFoldMap::new(self.dim, d),
            |mut acc, (i, value)| {
                let slot = normalising_coord(i, d, right_size);
                println!("index: {}, slot: {}", i, slot);
                let exp_val = (value - max).exp();
                acc.update(exp_val, slot)?;
                Result::<SoftmaxFoldMap, SoftmaxError>::Ok(acc)
            },
        )?;
        // .try_reduce(|| SoftmaxFoldMap::new(self.dim, d), |a, b| a.merge(b))?;

        let (tensor, _) = fold_map.finalise(&input_shape);

        Ok(tensor)
    }
}

/// Helper function used in softmax to reduce a general iterator index to the coordinate in the dimension we are normalising over
fn normalising_coord(index: usize, dim_size: usize, smaller_dims_product: usize) -> usize {
    // First we subtract index % smaller_dims_product from index
    let tmp = index - (index % smaller_dims_product);

    // Now we must divide this by smaller_dims_product, note it should be a multiple of smaller_dims_product so there is no remainder
    let tmp = tmp / smaller_dims_product;

    // Now the result should be tmp % dim_size
    tmp % dim_size
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

#[derive(Clone, Debug)]
/// Struct used to help with iterating over [`Tensor`] data when performing the [`Softmax`] operation.
struct SoftmaxFoldMap {
    /// The dimension we are normalising over
    dim: usize,
    /// The size of the dimension we are normalising over
    dim_size: usize,
    /// This vector holds the running sums over each dimension so we can normalize at the end
    sums: Vec<f32>,
    /// This vector holds the mapped input values (i.e. they have been exponentiated)
    mapped: Vec<f32>,
}

impl SoftmaxFoldMap {
    /// Creates a new [`SoftmaxFoldMap`]
    fn new(dim: usize, dim_size: usize) -> SoftmaxFoldMap {
        SoftmaxFoldMap {
            dim,
            dim_size,
            sums: vec![0.0f32; dim_size],
            mapped: vec![],
        }
    }

    /// Updates the struct, here value should already have been rescaled by any constants.
    fn update(&mut self, value: f32, index: usize) -> Result<(), SoftmaxError> {
        let exp = value.exp();
        self.mapped.push(exp);
        *self
            .sums
            .get_mut(index)
            .ok_or(SoftmaxError::ParameterError(format!(
                "Provided index: {}, was greater than dimension size: {}",
                index, self.dim_size
            )))? += exp;

        Ok(())
    }

    /// Merges two [`SoftmaxFoldMap`] instances into a single one.
    fn merge(self, other: SoftmaxFoldMap) -> Result<SoftmaxFoldMap, SoftmaxError> {
        let SoftmaxFoldMap {
            dim,
            dim_size,
            sums,
            mapped,
        } = self;
        let SoftmaxFoldMap {
            dim: other_dim,
            dim_size: other_dim_size,
            sums: other_sums,
            mapped: other_mapped,
        } = other;

        if dim_size != other_dim_size {
            return Err(SoftmaxError::ParameterError(format!(
                "Cannot merge SoftmaxFoldMaps with different dim sizes, left: {}, right: {}",
                dim_size, other_dim_size
            )));
        }

        if dim != other_dim {
            return Err(SoftmaxError::ParameterError(format!(
                "Cannot merge SoftmaxFoldMaps the normalise over different dimensions, left: {}, right: {}",
                dim, other_dim
            )));
        }

        let sums = sums
            .into_iter()
            .zip(other_sums.into_iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<f32>>();
        let mapped = mapped
            .into_iter()
            .chain(other_mapped.into_iter())
            .collect::<Vec<f32>>();

        Ok(SoftmaxFoldMap {
            dim,
            dim_size,
            sums,
            mapped,
        })
    }

    /// This method iterates over the data stored in self and divides each term of `self.mapped` by the correct value in `self.sums`.
    /// It returns both the output [`Tensor`] and `self.sums`, which will be useful in proving.
    fn finalise(self, input_shape: &[usize]) -> (Tensor<f32>, Vec<f32>) {
        let SoftmaxFoldMap {
            dim,
            dim_size,
            sums,
            mapped,
        } = self;

        let right_size = input_shape[dim..].iter().skip(1).product::<usize>();

        let data = mapped
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                let slot = normalising_coord(i, dim_size, right_size);
                let divisor = sums[slot];
                val / divisor
            })
            .collect::<Vec<f32>>();

        (Tensor::<f32>::new(input_shape.to_vec(), data), sums)
    }
}

#[cfg(test)]
mod tests {
    use tract_onnx::{
        prelude::{DatumType, IntoArcTensor, Op, tvec},
        tract_core::{
            ops::nn::Softmax as TractSoftmax, tract_data::prelude::Tensor as TractTensor,
            value::TValue,
        },
    };

    use crate::testing::load_test_onnx_model;

    use super::*;

    #[test]
    fn softmax_op_test_helper() -> anyhow::Result<()> {
        let shape: Vec<usize> = vec![2, 3, 4];

        let tensor = Tensor::<f32>::random(shape.clone());

        // First we get tract to run the test model with the random input to compare against
        let tract_tensor = TractTensor::from_shape::<f32>(&shape, tensor.get_data())?;
        let tract_input: TValue = tract_tensor.into();
        let model = load_test_onnx_model("softmax")?;

        let eval_order = model.eval_order()?;

        for id in eval_order {
            let node = model.node(id);

            if let Some(op) = node.op_as::<TractSoftmax>() {
                let info = op.info()?;
                for s in info {
                    println!("{s}");
                }
            }
        }

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

        println!("Our output: {}", our_output);
        println!("Expected output: {}", expected_output_tensors[0]);

        Ok(())
    }
}
