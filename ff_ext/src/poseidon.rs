use p3_field::PrimeField;
use p3_symmetric::CryptographicPermutation;

use crate::SmallField;

pub trait PoseidonField: PrimeField + SmallField {
    type T: CryptographicPermutation<[Self; 8]>;
    fn get_perm() -> Self::T;
}
