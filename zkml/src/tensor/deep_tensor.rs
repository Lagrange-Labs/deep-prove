use std::any::TypeId;

use ark_std::Zero;
use tract_onnx::prelude::{DatumType, Tensor};

use crate::{Element, quantization::QuantisationParams};

use super::error::DeepTensorError;

macro_rules! tensor_value_constructor {
     ($(($t:ty, $var:ident)), *)  => {
        #[derive(Debug, Clone, PartialEq)]
        pub enum TensorValue {
           $( $var (Vec<$t> )),*
        }
    };
}

tensor_value_constructor!((f32, F32), (f64, F64), (i128, Element));

impl TensorValue {
    /// Method to create a new [`TensorValue`], which is a [`Vec`] of a supported data
    /// type. Currently supported types are:
    /// - [`f64`]
    /// - [`f32`]
    /// - [`i128`]
    fn new<T: Clone + 'static>(data: &[T]) -> Result<TensorValue, DeepTensorError> {
        let t_type = TypeId::of::<T>();

        match t_type {
            type_id if type_id == TypeId::of::<f32>() => {
                Ok(TensorValue::F32(cast_vec(data.to_vec())))
            }
            type_id if type_id == TypeId::of::<f64>() => {
                Ok(TensorValue::F64(cast_vec(data.to_vec())))
            }
            type_id if type_id == TypeId::of::<i128>() => {
                Ok(TensorValue::Element(cast_vec(data.to_vec())))
            }
            _ => Err(DeepTensorError::ParameterError(
                "Cannot construct DeepValue of unsupported Type".to_string(),
            )),
        }
    }

    /// Method used to determine if two [`TensorValue`] contain the same type
    fn same_type(&self, other: &TensorValue) -> bool {
        match (self, other) {
            (TensorValue::Element(..), TensorValue::Element(..))
            | (TensorValue::F32(..), TensorValue::F32(..))
            | (TensorValue::F64(..), TensorValue::F64(..)) => true,
            _ => false,
        }
    }

    /// Gets the length of the inner data
    pub fn len(&self) -> usize {
        match self {
            TensorValue::Element(v) => v.len(),
            TensorValue::F32(v) => v.len(),
            TensorValue::F64(v) => v.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Struct used to provide a type agnostic tensor for dynamic dispatch methods.
pub struct DeepTensor {
    /// The dimensions of the tensor
    shape: Vec<usize>,
    /// The actual data
    data: TensorValue,
}

impl DeepTensor {
    /// Initialise a new [`DeepTensor`], this function works out what the type of `data` is and creates the correct variant
    pub fn new<T: Clone + 'static>(
        shape: &[usize],
        data: &[T],
    ) -> Result<DeepTensor, DeepTensorError> {
        let total_size = shape.iter().product::<usize>();
        if data.len() != total_size {
            return Err(DeepTensorError::ParameterError(format!(
                "Could not construct DeepTensor, expected data length to be {}, supplied data had length: {}",
                total_size,
                data.len()
            )));
        }

        let data = TensorValue::new::<T>(data)?;
        Ok(DeepTensor {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Getter for the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Getter for the rank of the tensor (how many dimensions it has)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Quantises the [`DeepTensor`] given some [`QuantisationParams`]
    pub fn quantise(&self, params: &QuantisationParams) -> DeepTensor {
        match &self.data {
            TensorValue::Element(_) => self.clone(),
            TensorValue::F32(v) => {
                let new_data = TensorValue::Element(
                    v.iter()
                        .map(|float_val| params.quantise(*float_val))
                        .collect::<Vec<Element>>(),
                );
                DeepTensor {
                    shape: self.shape().to_vec(),
                    data: new_data,
                }
            }
            TensorValue::F64(v) => {
                let new_data = TensorValue::Element(
                    v.iter()
                        .map(|float_val| params.quantise(*float_val as f32))
                        .collect::<Vec<Element>>(),
                );
                DeepTensor {
                    shape: self.shape().to_vec(),
                    data: new_data,
                }
            }
        }
    }

    /// Used to determine whether two [`DeepTensor`] share an underlying type
    pub fn same_type(&self, other: &DeepTensor) -> bool {
        self.data.same_type(&other.data)
    }

    /// Getter for the inner data
    pub fn data(&self) -> &TensorValue {
        &self.data
    }

    /// This converts a position given in "cartesian" form to the relevant index in the `self.data` vector.
    ///
    /// ```
    /// use zkml::tensor::DeepTensor;
    ///
    /// let tensor: DeepTensor = DeepTensor::new::<i128>(vec![2,2,3].as_slice(), vec![0i128; 12].as_slice()).unwrap();
    /// let test_coord: Vec<usize> = vec![0,1,2];
    ///
    /// let expected: usize = 5;
    ///
    /// let calc_index = tensor.get_index(&test_coord).unwrap();
    ///
    /// assert_eq!(calc_index, expected);
    /// ```
    pub fn get_index(&self, coords: &[usize]) -> Result<usize, DeepTensorError> {
        let shape = self.shape();
        if shape.len() != coords.len() {
            return Err(DeepTensorError::ParameterError(format!(
                "Cannot convert coordinates to an index if coords length != shape length, coords length: {}, shape length: {}",
                coords.len(),
                shape.len()
            )));
        }

        let (index, _): (usize, usize) = coords.iter().zip(shape.iter()).rev().try_fold(
            (0usize, 1usize),
            |(index_acc, dim_acc), (&coord, &dim)| {
                if coord >= dim {
                    return Err(DeepTensorError::ParameterError(
                        "Coordinate value was larger than the axes size, cannot get index"
                            .to_string(),
                    ));
                }

                Ok((index_acc + dim_acc * coord, dim_acc * dim))
            },
        )?;

        Ok(index)
    }

    /// Moves tensor axis. The first `usize` is the axis to move and the second `usize` is the position to move to.
    pub fn move_axis(&self, from: usize, to: usize) -> Result<DeepTensor, DeepTensorError> {
        // Check that both from and to are valid
        let old_shape = self.shape();

        if from >= old_shape.len() {
            return Err(DeepTensorError::ParameterError(format!(
                "Cannot move axes, 'from' axis: {}, is outside valid range: 0 to {}",
                from,
                old_shape.len() - 1
            )));
        }

        if to >= old_shape.len() {
            return Err(DeepTensorError::ParameterError(format!(
                "Cannot move axes, 'to' axis: {}, is outside valid range: 0 to {}",
                to,
                old_shape.len() - 1
            )));
        }

        // Make the new shape vector
        let mut new_shape = old_shape.to_vec();
        let moved = new_shape.remove(from);
        new_shape.insert(to, moved);

        // Define a closure that takes a coord and the new shape and returns the new index
        let coord_func = |coords: &[usize], shape: &[usize]| -> Result<usize, DeepTensorError> {
            let mut new_coords = coords.to_vec();
            let moved = new_coords.remove(from);
            new_coords.insert(to, moved);
            let (out, _) = new_coords.iter().zip(shape.iter()).rev().try_fold(
                (0usize, 1usize),
                |(index_acc, dim_acc), (&coord, &dim)| {
                    if coord >= dim {
                        return Err(DeepTensorError::ParameterError(
                            "Coordinate value was larger than the axes size, cannot get index"
                                .to_string(),
                        ));
                    }

                    Ok((index_acc + dim_acc * coord, dim_acc * dim))
                },
            )?;
            Ok(out)
        };

        self.reindex_data(&new_shape, coord_func)
    }

    /// Swaps two axes in the tensor.
    pub fn swap_axes(&self, a: usize, b: usize) -> Result<DeepTensor, DeepTensorError> {
        // Check that both a and b are valid
        let old_shape = self.shape();

        if a >= old_shape.len() {
            return Err(DeepTensorError::ParameterError(format!(
                "Cannot swap axes, 'a' axis: {}, is outside valid range: 0 to {}",
                a,
                old_shape.len() - 1
            )));
        }

        if b >= old_shape.len() {
            return Err(DeepTensorError::ParameterError(format!(
                "Cannot swap axes, 'b' axis: {}, is outside valid range: 0 to {}",
                b,
                old_shape.len() - 1
            )));
        }

        // Make the new shape vector
        let mut new_shape = self.shape().to_vec();
        new_shape[a] = self.shape[b];
        new_shape[b] = self.shape[a];

        // Define a closure that takes a coord and the new shape and returns the new index
        let coord_func = |coords: &[usize], shape: &[usize]| -> Result<usize, DeepTensorError> {
            let mut new_coords = coords.to_vec();
            new_coords[a] = coords[b];
            new_coords[b] = coords[a];
            let (out, _) = new_coords.iter().zip(shape.iter()).rev().try_fold(
                (0usize, 1usize),
                |(index_acc, dim_acc), (&coord, &dim)| {
                    if coord >= dim {
                        return Err(DeepTensorError::ParameterError(
                            "Coordinate value was larger than the axes size, cannot get index"
                                .to_string(),
                        ));
                    }

                    Ok((index_acc + dim_acc * coord, dim_acc * dim))
                },
            )?;
            Ok(out)
        };

        self.reindex_data(&new_shape, coord_func)
    }

    /// Helper function used when moving/swapping axes.
    fn reindex_data<F>(
        &self,
        new_shape: &[usize],
        coord_func: F,
    ) -> Result<DeepTensor, DeepTensorError>
    where
        F: Fn(&[usize], &[usize]) -> Result<usize, DeepTensorError>,
    {
        let size = self.shape().iter().product::<usize>();

        let all_indices = get_all_coords(self.shape());
        // Match on the data type to produce the new tensor
        match self.data() {
            TensorValue::Element(d_slice) => {
                let mut out_data = vec![0i128; size];
                all_indices.iter().try_for_each(|coords| {
                    let old_index = self.get_index(coords)?;
                    let new_index = coord_func(coords, new_shape)?;
                    out_data[new_index] = d_slice[old_index];
                    Ok(())
                })?;
                DeepTensor::new(new_shape, &out_data)
            }
            TensorValue::F32(d_slice) => {
                let mut out_data = vec![0.0f32; size];
                all_indices.iter().try_for_each(|coords| {
                    let old_index = self.get_index(coords)?;
                    let new_index = coord_func(coords, new_shape)?;
                    out_data[new_index] = d_slice[old_index];
                    Ok(())
                })?;
                DeepTensor::new(new_shape, &out_data)
            }
            TensorValue::F64(d_slice) => {
                let mut out_data = vec![0.0f64; size];
                all_indices.iter().try_for_each(|coords| {
                    let old_index = self.get_index(coords)?;
                    let new_index = coord_func(coords, new_shape)?;
                    out_data[new_index] = d_slice[old_index];
                    Ok(())
                })?;
                DeepTensor::new(new_shape, &out_data)
            }
        }
    }
}

/// Given a tensor shape this function returns all of its coordinates
fn get_all_coords(shape: &[usize]) -> Vec<Vec<usize>> {
    let size = shape.iter().product::<usize>();
    // If size is zero we just return an empty vector
    if size.is_zero() {
        return vec![];
    }

    let mut output: Vec<Vec<usize>> = (0..shape[0]).map(|i| vec![i]).collect::<Vec<Vec<usize>>>();

    let mut round = 1;

    while round < shape.len() {
        output = output
            .into_iter()
            .flat_map(|coords| {
                (0..shape[round])
                    .map(|i| {
                        let mut round_vec = coords.clone();
                        round_vec.push(i);
                        round_vec
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect::<Vec<Vec<usize>>>();
        round += 1;
    }

    output
}

/// Internal method used for casting a vector to a known type.
/// It should only be called in situations where the [`TypeId`] of
/// `A` and `B` have already been checked to be the same.
fn cast_vec<A, B>(mut vec: Vec<A>) -> Vec<B> {
    let length = vec.len();
    let capacity = vec.capacity();
    let ptr = vec.as_mut_ptr();
    // Prevent `vec` from dropping its contents
    std::mem::forget(vec);

    // Convert the pointer to the new type
    let new_ptr = ptr as *mut B;

    // Create a new vector with the same length and capacity, but different type
    unsafe { Vec::from_raw_parts(new_ptr, length, capacity) }
}

macro_rules! tract_converter {
    ($tract:ty, $deep:ty, $tensor:expr) => {
        TensorValue::new::<$deep>(
            &Tensor::as_slice::<$tract>($tensor)
                .map_err(|e| {
                    DeepTensorError::ConversionError(format!(
                        "Could not convert tract data type: {:?} to TensorValue::{}, inner: {:?}",
                        $tensor.datum_type(),
                        stringify!($deep),
                        e
                    ))
                })?
                .into_iter()
                .map(|&val| val as $deep)
                .collect::<Vec<$deep>>(),
        )
    };
}

impl TryFrom<&Tensor> for DeepTensor {
    type Error = DeepTensorError;
    fn try_from(tract_tensor: &Tensor) -> Result<DeepTensor, DeepTensorError> {
        let data = match tract_tensor.datum_type() {
            DatumType::U8 => {
                tract_converter!(u8, Element, tract_tensor)
            }
            DatumType::U16 => {
                tract_converter!(u16, Element, tract_tensor)
            }
            DatumType::U32 => {
                tract_converter!(u32, Element, tract_tensor)
            }
            DatumType::U64 => {
                tract_converter!(u64, Element, tract_tensor)
            }
            DatumType::I8 => {
                tract_converter!(i8, Element, tract_tensor)
            }
            DatumType::I16 => {
                tract_converter!(i16, Element, tract_tensor)
            }
            DatumType::I32 => {
                tract_converter!(i32, Element, tract_tensor)
            }
            DatumType::I64 => {
                tract_converter!(i64, Element, tract_tensor)
            }
            DatumType::F32 => {
                tract_converter!(f32, f32, tract_tensor)
            }
            DatumType::F64 => {
                tract_converter!(f64, f64, tract_tensor)
            }
            dt => Err(DeepTensorError::ConversionError(format!(
                "Do not currently support conversion from: {:?}",
                dt
            ))),
        }?;

        let tract_shape = tract_tensor.shape();
        let shape = if !tract_shape.is_empty() {
            let expected_size = tract_shape.iter().product::<usize>();
            if expected_size != data.len() {
                return Err(DeepTensorError::ConversionError(format!(
                    "Converted data length: {} did not equal the expected size from the tract tensor: {}",
                    data.len(),
                    expected_size
                )));
            }
            tract_shape.to_vec()
        } else {
            match data.len() {
                0 => vec![],
                1 => vec![1],
                l => {
                    return Err(DeepTensorError::ConversionError(format!(
                        "tract tensor had empty shape, implying a scalar but converted data had length: {}",
                        l
                    )));
                }
            }
        };

        Ok(DeepTensor { shape, data })
    }
}

#[cfg(test)]
mod tests {
    use tract_onnx::prelude::tensor0;

    use super::*;

    #[test]
    fn test_coords() {
        let shape = vec![2, 4, 3];
        let size = shape.iter().product::<usize>();
        let all_coords = get_all_coords(&shape);

        let tensor = DeepTensor::new::<i128>(&shape, vec![0i128; size].as_slice()).unwrap();

        for (i, coords) in all_coords.into_iter().enumerate() {
            let calc_i = tensor.get_index(&coords).unwrap();
            assert_eq!(calc_i, i);
        }
    }

    #[test]
    fn test_move_axis() {
        let a =
            DeepTensor::new::<i128>(&[2, 3, 2], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
        let expected =
            DeepTensor::new::<i128>(&[2, 2, 3], &[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]).unwrap();
        let b = a.move_axis(1, 2).unwrap();
        assert_eq!(b, expected);
    }

    #[test]
    fn test_swap_axis() {
        let a =
            DeepTensor::new::<i128>(&[2, 3, 2], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
        let expected =
            DeepTensor::new::<i128>(&[2, 2, 3], &[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]).unwrap();
        let b = a.swap_axes(2, 1).unwrap();
        assert_eq!(b, expected);
    }

    #[test]
    fn test_tract_conversion() {
        let tract = tensor0::<f32>(1.7f32);

        let deep_tensor = DeepTensor::try_from(&tract).unwrap();
        let expected = DeepTensor::new::<f32>(&[1], &[1.7f32]).unwrap();

        assert_eq!(deep_tensor, expected)
    }
}
