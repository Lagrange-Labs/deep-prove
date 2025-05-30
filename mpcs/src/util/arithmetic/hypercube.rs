use ff_ext::ExtensionField;
use multilinear_extensions::mle::FieldType;
use p3_field::Field;
use rayon::prelude::{ParallelIterator, ParallelSliceMut};

use crate::util::log2_strict;

pub fn interpolate_field_type_over_boolean_hypercube<E: ExtensionField>(evals: &mut FieldType<E>) {
    match evals {
        FieldType::Ext(evals) => interpolate_over_boolean_hypercube(evals),
        FieldType::Base(evals) => interpolate_over_boolean_hypercube(evals),
        _ => unreachable!(),
    };
}

pub fn interpolate_over_boolean_hypercube<F: Field>(evals: &mut [F]) {
    // let timer = start_timer!(|| "interpolate_over_hypercube");
    // iterate over array, replacing even indices with (evals[i] - evals[(i+1)])
    let n = log2_strict(evals.len());

    evals.par_chunks_mut(2).for_each(|chunk| {
        chunk[1] -= chunk[0];
    });

    // This code implicitly assumes that coeffs has size at least 1 << n,
    // that means the size of evals should be a power of two
    for i in 2..n + 1 {
        let chunk_size = 1 << i;
        evals.par_chunks_mut(chunk_size).for_each(|chunk| {
            let half_chunk = chunk_size >> 1;
            for j in half_chunk..chunk_size {
                chunk[j] -= chunk[j - half_chunk];
            }
        });
    }
    // end_timer!(timer);
}
