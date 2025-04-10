use std::{any::TypeId, cmp::Ordering};

use ark_std::rand::Rng;
use tract_onnx::prelude::{DatumType, Tensor};

use crate::{Element, quantization::QuantisationParams};

use super::{
    error::DeepTensorError,
    utilities::{get_all_coords},
};


pub trait Number:
    Copy
    + Clone
    + Send
    + Sync
    + Default
    + std::iter::Sum
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::Mul<Output = Self>
    + std::fmt::Debug
{
    const MIN: Self;
    const MAX: Self;
    fn random<R: Rng>(rng: &mut R) -> Self;
    /// reason abs is necessary is because f32 doesn't implement Ord trait, so to have uniform code for f32 and Element,
    /// we implement abs here.
    fn absolute_value(&self) -> Self;
    fn cmp_max(&self, other: &Self) -> Self {
        match self.compare(other) {
            Ordering::Greater => *self,
            Ordering::Equal => *self,
            Ordering::Less => *other,
        }
    }
    fn cmp_min(&self, other: &Self) -> Self {
        match self.compare(other) {
            Ordering::Greater => *other,
            Ordering::Equal => *self,
            Ordering::Less => *self,
        }
    }
    fn compare(&self, other: &Self) -> Ordering;
    fn is_negative(&self) -> bool;
}

impl Number for Element {
    const MIN: Element = Element::MIN;
    const MAX: Element = Element::MAX;
    fn random<R: Rng>(rng: &mut R) -> Self {
        0
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(&other)
    }
    fn is_negative(&self) -> bool {
        *self < 0
    }
}
impl Number for f32 {
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    fn random<R: Rng>(rng: &mut R) -> Self {
        1.0
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        if self < other {
            Ordering::Less
        } else if self == other {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }

    fn is_negative(&self) -> bool {
        *self < 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Struct used to provide a type agnostic tensor for dynamic dispatch methods.
pub struct DeepTensor<T>{
    /// The dimensions of the tensor
    shape: Vec<usize>,
    /// The actual data
    data: Vec<T>,
}

impl<T: Number> DeepTensor<T> {
    /// Initialise a new [`DeepTensor`], this function works out what the type of `data` is and creates the correct variant
    pub fn new(
        shape: &[usize],
        data: &[T],
    ) -> Result<DeepTensor<T>, DeepTensorError> {
        let total_size = shape.iter().product::<usize>();
        if data.len() != total_size {
            return Err(DeepTensorError::ParameterError(format!(
                "Could not construct DeepTensor, expected data length to be {}, supplied data had length: {}",
                total_size,
                data.len()
            )));
        }

        Ok(DeepTensor {
            shape: shape.to_vec(),
            data: data.to_vec(),
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

    /// Getter for the inner data
    pub fn data(&self) -> &Vec<T> {
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
    pub fn move_axis(&self, from: usize, to: usize) -> Result<Self, DeepTensorError> {
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
    pub fn swap_axes(&self, a: usize, b: usize) -> Result<Self, DeepTensorError> {
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
    ) -> Result<Self, DeepTensorError>
    where
        F: Fn(&[usize], &[usize]) -> Result<usize, DeepTensorError>,
    {
        let size = self.shape().iter().product::<usize>();

        let all_indices = get_all_coords(self.shape());
        // Match on the data type to produce the new tensor
                let mut out_data = vec![T::default(); size];
                all_indices.iter().try_for_each(|coords| {
                    let old_index = self.get_index(coords)?;
                    let new_index = coord_func(coords, new_shape)?;
                    out_data[new_index] = self.data[old_index];
                    Ok(())
                })?;
                DeepTensor::new(new_shape, &out_data)
    }
}

impl DeepTensor<f32> {
/// Quantises the [`DeepTensor`] given some [`QuantisationParams`]
    pub fn quantise(&self, params: &QuantisationParams) -> DeepTensor<Element> {
        DeepTensor::<Element> {
            shape: self.shape.clone(),
            data: self.data.iter().map(|float_val| params.quantise(*float_val)).collect::<Vec<Element>>(),
        }
    }
}

macro_rules! tract_converter {
    ($tract:ty, $deep:ty, $tensor:expr) => {
        {
            Tensor::as_slice::<$tract>($tensor)
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
                .collect::<Vec<$deep>>()
            }
    };
}

impl TryFrom<&Tensor> for DeepTensor<f32> {
    type Error = DeepTensorError;
    fn try_from(tract_tensor: &Tensor) -> Result<DeepTensor<f32>, DeepTensorError> {
        let data = match tract_tensor.datum_type() {
            DatumType::F32 => {
                Ok(tract_converter!(f32, f32, tract_tensor))
            }
            //DatumType::F64 => {
            //    Ok(tract_converter!(f64, f64, tract_tensor))
            //}
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
