use p3_field::{FieldAlgebra, PrimeField};
use p3_symmetric::CryptographicPermutation;

use crate::SmallField;

pub trait PoseidonField: PrimeField + SmallField {
    type T: CryptographicPermutation<[Self; 8]>;
    fn get_perm() -> Self::T;
}

pub(crate) fn new_array<const N: usize, F: FieldAlgebra>(input: [u64; N]) -> [F; N] {
    let mut output = [F::ZERO; N];
    let mut i = 0;
    while i < N {
        output[i] = F::from_canonical_u64(input[i]);
        i += 1;
    }
    output
}
