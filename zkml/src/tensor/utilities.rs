//! Module containing utility functions when dealing with tensors

use ark_std::Zero;

/// Internal method used for casting a vector to a known type.
/// It should only be called in situations where the [`TypeId`] of
/// `A` and `B` have already been checked to be the same.
pub(crate) fn cast_vec<A, B>(mut vec: Vec<A>) -> Vec<B> {
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

/// Given a tensor shape this function returns all of its coordinates
pub(crate) fn get_all_coords(shape: &[usize]) -> Vec<Vec<usize>> {
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

/// Helper function that given two tensor shapes returns the broadcasted shape
pub(crate) fn get_broadcasted_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let a_rank = a.len();
    let b_rank = b.len();

    if a_rank == b_rank {
        a.iter()
            .zip(b.iter())
            .map(|(a_dim, b_dim)| *a_dim.max(b_dim))
            .collect::<Vec<usize>>()
    } else if a_rank < b_rank {
        b.to_vec()
    } else {
        a.to_vec()
    }
}
