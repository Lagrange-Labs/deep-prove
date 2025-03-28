//! This module contains code for padding tensor inputs to convolution and pooling layers. Currently we support padding with either zeroes or
//! reflective padding.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use ark_std::Zero;

use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Enum used to distinguish between the various types of padding.
pub enum Padding {
    /// Pads with zeroes
    Zeroes {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
    /// Pads in a reflective manner
    Reflective {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
    /// Pads by replicating input boundaries
    Replication {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
    /// Pads by circulating through the input
    Circular {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone)]
/// Enum for any errors that occur during padding
pub enum PaddingError {
    ParameterError(String),
}

impl Error for PaddingError {}

impl Display for PaddingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            PaddingError::ParameterError(s) => {
                write!(f, "Parameters were incorrect for padding: {}", s)
            }
        }
    }
}

impl Padding {
    /// Creates a new instance of [`Padding::Zeroes`]
    pub fn new_zeroes(top: usize, bottom: usize, left: usize, right: usize) -> Padding {
        Padding::Zeroes {
            top,
            bottom,
            left,
            right,
        }
    }

    /// Creates a new instance of [`Padding::Reflective`]
    pub fn new_reflective(top: usize, bottom: usize, left: usize, right: usize) -> Padding {
        Padding::Reflective {
            top,
            bottom,
            left,
            right,
        }
    }

    /// Creates a new instance of [`Padding::Replication`]
    pub fn new_replication(top: usize, bottom: usize, left: usize, right: usize) -> Padding {
        Padding::Replication {
            top,
            bottom,
            left,
            right,
        }
    }

    /// Creates a new instance of [`Padding::Circular`]
    pub fn new_circular(top: usize, bottom: usize, left: usize, right: usize) -> Padding {
        Padding::Circular {
            top,
            bottom,
            left,
            right,
        }
    }

    /// Getter for the top padding size
    pub fn top(&self) -> usize {
        match self {
            Padding::Zeroes { top, .. }
            | Padding::Reflective { top, .. }
            | Padding::Replication { top, .. }
            | Padding::Circular { top, .. } => *top,
        }
    }

    /// Getter for the bottom padding size
    pub fn bottom(&self) -> usize {
        match self {
            Padding::Zeroes { bottom, .. }
            | Padding::Reflective { bottom, .. }
            | Padding::Replication { bottom, .. }
            | Padding::Circular { bottom, .. } => *bottom,
        }
    }

    /// Getter for the left padding size
    pub fn left(&self) -> usize {
        match self {
            Padding::Zeroes { left, .. }
            | Padding::Reflective { left, .. }
            | Padding::Replication { left, .. }
            | Padding::Circular { left, .. } => *left,
        }
    }

    /// Getter for the right padding size
    pub fn right(&self) -> usize {
        match self {
            Padding::Zeroes { right, .. }
            | Padding::Reflective { right, .. }
            | Padding::Replication { right, .. }
            | Padding::Circular { right, .. } => *right,
        }
    }

    /// Pads a [`Tensor`] with a padding scheme
    pub fn pad_tensor<T: Default + Clone + std::fmt::Debug>(
        &self,
        tensor: &Tensor<T>,
    ) -> Result<Tensor<T>, PaddingError> {
        // First we check the tensor has at least two dimensions
        let shape = tensor.get_shape();
        let num_dims = shape.len();
        if num_dims < 2 {
            return Err(PaddingError::ParameterError(
                "Cannot pad a tensor that has fewer than two dimensions".to_string(),
            ));
        }

        // If the shape only has two dimensions we shortcut directly to the matrix pad method
        // We work out the size of each matrix

        let chunk_size = shape.iter().skip(num_dims - 2).product::<usize>();
        // We check that the height and width isn't zero
        if chunk_size == 0 {
            return Err(PaddingError::ParameterError(
                "One of height or width is zero so cannot pad".to_string(),
            ));
        }

        let padded_data = tensor
            .get_data()
            .chunks(chunk_size)
            .map(|chunk| self.pad_matrix_data(chunk, &[shape[num_dims - 2], shape[num_dims - 1]]))
            .collect::<Result<Vec<Vec<T>>, PaddingError>>()?
            .concat();

        let padded_shape = self.padded_shape(&shape)?;

        Ok(Tensor::<T>::new(padded_shape, padded_data))
    }

    /// Given a shape returns the shape after padding is performed
    pub fn padded_shape(&self, shape: &[usize]) -> Result<Vec<usize>, PaddingError> {
        let num_dims = shape.len();
        // Need to have at least two dimensions to pad
        if num_dims < 2 {
            return Err(PaddingError::ParameterError(
                "Cannot calculate padded shape as input shape had fewer than two dimensions"
                    .to_string(),
            ));
        }

        let new_height = shape[num_dims - 2] + self.top() + self.bottom();
        let new_width = shape[num_dims - 1] + self.left() + self.right();

        Ok(shape
            .iter()
            .copied()
            .take(num_dims - 2)
            .chain([new_height, new_width])
            .collect::<Vec<usize>>())
    }
}

// Internal methods for the `Padding` enum
impl Padding {
    /// This method takes a slice of data that is assumed to represent a matrix together with the matrix's dimensions
    /// and pads the data, returning it as vector.
    fn pad_matrix_data<T: Default + Clone + std::fmt::Debug>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> Result<Vec<T>, PaddingError> {
        // If the shape doesn't have exactly 2 elements its not a matrix so we error
        if shape.len() != 2 {
            return Err(PaddingError::ParameterError(format!(
                "Provided shape did not have length 2 (it had length: {})",
                shape.len()
            )));
        }
        // If the total length of data doesn't equal the dimensions provided we should error
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(PaddingError::ParameterError(format!(
                "Data length: {}, did not equal expected length from shape: {}",
                data.len(),
                expected_size
            )));
        }
        // We pad rows first then columns, that we we avoid having to do a an extra transposition at the end
        let padded_rows = (0..shape[1])
            .map(|i| {
                let row = data
                    .iter()
                    .skip(i)
                    .step_by(shape[1])
                    .cloned()
                    .collect::<Vec<T>>();
                self.pad_row(&row)
            })
            .collect::<Result<Vec<Vec<T>>, PaddingError>>()?
            .concat();

        let new_row_size = shape[0] + self.left() + self.right();

        // Now we pad the columns
        let padded_output = (0..new_row_size)
            .map(|i| {
                let column = padded_rows
                    .iter()
                    .skip(i)
                    .step_by(new_row_size)
                    .take(shape[1])
                    .cloned()
                    .collect::<Vec<T>>();
                self.pad_column(&column)
            })
            .collect::<Result<Vec<Vec<T>>, PaddingError>>()?
            .concat();

        Ok(padded_output)
    }

    /// Takes a column and pads it depending on the padding variant
    fn pad_column<T: Default + Clone>(&self, column: &[T]) -> Result<Vec<T>, PaddingError> {
        match self {
            Padding::Zeroes { top, bottom, .. } => Ok(std::iter::repeat_n(T::default(), *top)
                .chain(column.iter().cloned())
                .chain(std::iter::repeat_n(T::default(), *bottom))
                .collect::<Vec<T>>()),
            Padding::Reflective { top, bottom, .. } => {
                // We need to check that the column size is at least as long as top and bottom
                if column.len() <= *top || column.len() <= *bottom {
                    return Err(PaddingError::ParameterError(format!(
                        "Cannot Reflective Pad the column as it is too small, column length: {}, top padding: {}, bottom padding: {}",
                        column.len(),
                        top,
                        bottom
                    )));
                }

                Ok(column
                    .iter()
                    .skip(1)
                    .take(*top)
                    .rev()
                    .chain(column.iter())
                    .chain(column.iter().rev().skip(1).take(*bottom))
                    .cloned()
                    .collect::<Vec<T>>())
            }
            Padding::Replication { top, bottom, .. } => {
                // The unwrap here is safe as this method can only be accessed through `pad_tensor` which checks that we have a non-zero number of rows and columns
                let first = column.first().cloned().unwrap();
                let last = column.last().cloned().unwrap();

                Ok(std::iter::repeat_n(first, *top)
                    .chain(column.iter().cloned())
                    .chain(std::iter::repeat_n(last, *bottom))
                    .collect::<Vec<T>>())
            }

            Padding::Circular { top, bottom, .. } => {
                let column_length = column.len();
                // For the circular case we may have to repeat the entire column a few times so we work out how many full copies of the column we need first
                let top_copies = if !top.is_zero() && *top > column_length {
                    (column_length - 1) / *top
                } else {
                    0
                };
                let bottom_copies = if !bottom.is_zero() && *bottom > column_length {
                    (column_length - 1) / *bottom
                } else {
                    0
                };

                let top_skip = column_length - (*top % column_length);

                Ok(column
                    .iter()
                    .skip(top_skip)
                    .chain(std::iter::repeat_n(column, top_copies + bottom_copies + 1).flatten())
                    .chain(column.iter().take(*bottom))
                    .cloned()
                    .collect::<Vec<T>>())
            }
        }
    }

    /// Takes a column and pads it depending on the padding variant
    fn pad_row<T: Default + Clone>(&self, row: &[T]) -> Result<Vec<T>, PaddingError> {
        match self {
            Padding::Zeroes { left, right, .. } => Ok(std::iter::repeat_n(T::default(), *left)
                .chain(row.iter().cloned())
                .chain(std::iter::repeat_n(T::default(), *right))
                .collect::<Vec<T>>()),
            Padding::Reflective { left, right, .. } => {
                // We need to check that the column size is at least as long as top and bottom
                if row.len() <= *left || row.len() <= *right {
                    return Err(PaddingError::ParameterError(format!(
                        "Cannot Reflective Pad the column as it is too small, column length: {}, top padding: {}, bottom padding: {}",
                        row.len(),
                        left,
                        right
                    )));
                }

                Ok(row
                    .iter()
                    .skip(1)
                    .take(*left)
                    .rev()
                    .chain(row.iter())
                    .chain(row.iter().rev().skip(1).take(*right))
                    .cloned()
                    .collect::<Vec<T>>())
            }
            Padding::Replication { left, right, .. } => {
                // The unwrap here is safe as this method can only be accessed through `pad_tensor` which checks that we have a non-zero number of rows and columns
                let first = row.first().cloned().unwrap();
                let last = row.last().cloned().unwrap();

                Ok(std::iter::repeat_n(first, *left)
                    .chain(row.iter().cloned())
                    .chain(std::iter::repeat_n(last, *right))
                    .collect::<Vec<T>>())
            }
            Padding::Circular { left, right, .. } => {
                let row_length = row.len();
                // For the circular case we may have to repeat the entire column a few times so we work out how many full copies of the column we need first
                let left_copies = if !left.is_zero() && *left > row_length {
                    (row_length - 1) / *left
                } else {
                    0
                };
                let right_copies = if !right.is_zero() && *right > row_length {
                    (row_length - 1) / *right
                } else {
                    0
                };

                let left_skip = row_length - (*left % row_length);

                Ok(row
                    .iter()
                    .skip(left_skip)
                    .chain(std::iter::repeat_n(row, left_copies + right_copies + 1).flatten())
                    .chain(row.iter().take(*right))
                    .cloned()
                    .collect::<Vec<T>>())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Element;

    use super::*;

    #[test]
    fn test_padding() -> Result<(), PaddingError> {
        for case in [
            TestCase::Zeroes,
            TestCase::Reflective,
            TestCase::Replication,
            TestCase::Circular,
        ] {
            padding_test_helper(case)?;
        }
        Ok(())
    }

    enum TestCase {
        Zeroes,
        Reflective,
        Replication,
        Circular,
    }

    impl TestCase {
        fn make_padding(&self, top: usize, bottom: usize, left: usize, right: usize) -> Padding {
            match self {
                TestCase::Zeroes => Padding::new_zeroes(top, bottom, left, right),
                TestCase::Reflective => Padding::new_reflective(top, bottom, left, right),
                TestCase::Replication => Padding::new_replication(top, bottom, left, right),
                TestCase::Circular => Padding::new_circular(top, bottom, left, right),
            }
        }

        fn expected_symmetric_result(&self) -> Vec<Element> {
            match self {
                TestCase::Zeroes => vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 1, 4, 7,
                    0, 0, 0, 0, 2, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                TestCase::Reflective => vec![
                    8, 5, 2, 5, 8, 5, 2, 7, 4, 1, 4, 7, 4, 1, 6, 3, 0, 3, 6, 3, 0, 7, 4, 1, 4, 7,
                    4, 1, 8, 5, 2, 5, 8, 5, 2, 7, 4, 1, 4, 7, 4, 1, 6, 3, 0, 3, 6, 3, 0,
                ],
                TestCase::Replication => vec![
                    0, 0, 0, 3, 6, 6, 6, 0, 0, 0, 3, 6, 6, 6, 0, 0, 0, 3, 6, 6, 6, 1, 1, 1, 4, 7,
                    7, 7, 2, 2, 2, 5, 8, 8, 8, 2, 2, 2, 5, 8, 8, 8, 2, 2, 2, 5, 8, 8, 8,
                ],
                TestCase::Circular => vec![
                    4, 7, 1, 4, 7, 1, 4, 5, 8, 2, 5, 8, 2, 5, 3, 6, 0, 3, 6, 0, 3, 4, 7, 1, 4, 7,
                    1, 4, 5, 8, 2, 5, 8, 2, 5, 3, 6, 0, 3, 6, 0, 3, 4, 7, 1, 4, 7, 1, 4,
                ],
            }
        }

        fn expected_asymmetric_result(&self) -> Vec<Element> {
            match self {
                TestCase::Zeroes => vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 1, 4, 7, 0, 0, 0, 2, 5, 8, 0,
                ],
                TestCase::Reflective => vec![
                    7, 4, 1, 4, 7, 4, 6, 3, 0, 3, 6, 3, 7, 4, 1, 4, 7, 4, 8, 5, 2, 5, 8, 5,
                ],
                TestCase::Replication => vec![
                    0, 0, 0, 3, 6, 6, 0, 0, 0, 3, 6, 6, 1, 1, 1, 4, 7, 7, 2, 2, 2, 5, 8, 8,
                ],
                TestCase::Circular => vec![
                    5, 8, 2, 5, 8, 2, 3, 6, 0, 3, 6, 0, 4, 7, 1, 4, 7, 1, 5, 8, 2, 5, 8, 2,
                ],
            }
        }

        fn name(&self) -> String {
            match self {
                TestCase::Zeroes => "Zeroes".to_string(),
                TestCase::Reflective => "Reflective".to_string(),
                TestCase::Replication => "Replication".to_string(),
                TestCase::Circular => "Circular".to_string(),
            }
        }
    }

    fn padding_test_helper(case: TestCase) -> Result<(), PaddingError> {
        let input_shape = vec![3, 3];
        let input_data: Vec<Element> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];

        let padding = case.make_padding(2, 2, 2, 2);

        let expected_output_shape = vec![7, 7];
        let expected_output_data: Vec<Element> = case.expected_symmetric_result();
        // First we test the padding function on a single matrix
        let input_tensor = Tensor::<Element>::new(input_shape.clone(), input_data.clone());

        let output_tensor = padding.pad_tensor(&input_tensor).unwrap();

        for (i, (output, expected)) in output_tensor
            .get_data()
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            if *output != *expected {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output {} does not match expected output {} at index {} in case {} symmetric matrix padding",
                    output,
                    expected,
                    i,
                    case.name()
                )));
            }
        }

        let output_shape = output_tensor.get_shape();

        for (i, (out, expect)) in output_shape
            .iter()
            .zip(expected_output_shape.iter())
            .enumerate()
        {
            if *out != *expect {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output shape {} does not match expected output shpae {} at index {} in case {} symmetric matrix padding",
                    out,
                    expect,
                    i,
                    case.name()
                )));
            }
        }

        // Now we test it on multiple channels
        let num_channels = 4;

        let input_shape = vec![4, 3, 3];
        let input_data = std::iter::repeat_n(input_data, num_channels)
            .flatten()
            .collect::<Vec<Element>>();

        let expected_output_shape = vec![4, 7, 7];
        let expected_output_data = std::iter::repeat_n(expected_output_data, num_channels)
            .flatten()
            .collect::<Vec<Element>>();

        let input_tensor = Tensor::<Element>::new(input_shape, input_data);

        let output_tensor = padding.pad_tensor(&input_tensor)?;

        for (i, (output, expected)) in output_tensor
            .get_data()
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            if *output != *expected {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output {} does not match expected output {} at index {} in case {} multichannel padding",
                    output,
                    expected,
                    i,
                    case.name()
                )));
            }
        }

        let output_shape = output_tensor.get_shape();

        for (i, (out, expect)) in output_shape
            .iter()
            .zip(expected_output_shape.iter())
            .enumerate()
        {
            if *out != *expect {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output shape {} does not match expected output shpae {} at index {} in case {} multichannel padding",
                    out,
                    expect,
                    i,
                    case.name()
                )));
            }
        }

        // Now test uneven padding
        let input_shape = vec![3, 3];
        let input_data: Vec<Element> = vec![0, 3, 6, 1, 4, 7, 2, 5, 8];

        let padding = case.make_padding(2, 1, 1, 0);

        let expected_output_shape = vec![6, 4];
        let expected_output_data: Vec<Element> = case.expected_asymmetric_result();

        let input_tensor = Tensor::<Element>::new(input_shape.clone(), input_data.clone());

        let output_tensor = padding.pad_tensor(&input_tensor)?;

        for (i, (output, expected)) in output_tensor
            .get_data()
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            if *output != *expected {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output {} does not match expected output {} at index {} in case {} asymmetric padding",
                    output,
                    expected,
                    i,
                    case.name()
                )));
            }
        }

        let output_shape = output_tensor.get_shape();

        for (i, (out, expect)) in output_shape
            .iter()
            .zip(expected_output_shape.iter())
            .enumerate()
        {
            if *out != *expect {
                return Err(PaddingError::ParameterError(format!(
                    "Calculated output shape {} does not match expected output shpae {} at index {} in case {} asymmetric padding",
                    out,
                    expect,
                    i,
                    case.name()
                )));
            }
        }

        Ok(())
    }
}
