use crate::{Element, tensor::Tensor};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

pub const MAXPOOL2D_KERNEL_SIZE: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Pooling {
    Maxpool2D(Maxpool2D),
}

impl Pooling {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        match self {
            Pooling::Maxpool2D(maxpool2d) => maxpool2d.op(input),
        }
    }
}

/// Information about a maxpool2d step
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub struct Maxpool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl Maxpool2D {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        assert!(
            self.kernel_size == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for kernel size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        assert!(
            self.stride == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for stride size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        input.maxpool2d(self.kernel_size, self.stride)
    }
}
