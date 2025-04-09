//! Module contain the [`TryFrom`] implementation from [`Const`] to [`Constant`]

use super::Constant;
use crate::{provable_ops::ProvableOpError, tensor::DeepTensor};
use tract_onnx::tract_hir::ops::konst::Const;

impl TryFrom<Const> for Constant {
    type Error = ProvableOpError;

    fn try_from(tract_op: Const) -> Result<Constant, ProvableOpError> {
        // First we will check we can handle the data type
        let tract_tensor = &tract_op.0;

        // Now we can just use the DeepTensor TryFrom implementation
        let tensor: DeepTensor = tract_tensor.as_ref().try_into()?;

        Ok(Constant {
            tensor,
            quant_params: None,
        })
    }
}
