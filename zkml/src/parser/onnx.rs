use crate::{
    layers::{
        Layer,
        activation::Activation,
        convolution::Convolution,
        pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
        provable::{
            Edge, NodeId, Op as DOP, ProvableModel, ProvableNode, ProvableOp, ProvableOpError,
        },
    },
    model::Model,
    quantization::Fieldizer,
};
use anyhow::{Context, bail, ensure};
use ff_ext::ExtensionField;
use goldilocks::GoldilocksExt2;
use once_cell::sync::Lazy;
use tracing::debug;
use std::{collections::HashMap, sync::Mutex};
use tract_onnx::{
    prelude::*,
    tract_core::{
        self,
        ops::{
            binary::TypedBinOp,
            cnn::{Conv, MaxPool},
            einsum::EinSum,
            source::TypedSource,
        },
    },
    tract_hir::{
        internal::Expansion,
        ops::{cnn::PaddingSpec, konst::Const, math::Add},
    },
};
use transcript::Transcript;

type OnnxModel = Graph<TypedFact, Box<dyn TypedOp + 'static>>;
type OnnxNode = Node<TypedFact, Box<dyn TypedOp + 'static>>;
type CustomNode = crate::layers::provable::ProvableNode<f32>;

macro_rules! ensure_onnx {
        // Match with format args
        ($cond:expr, $err_fmt:literal, $($args:expr),+ $(,)?) => {
            if !$cond {
                return Err(ProvableOpError::OnnxParsingError(format!($err_fmt, $($args),+)).into());
            }
        };
        // Match with plain string (no args)
        ($cond:expr, $err_msg:literal $(,)?) => {
            if !$cond {
                return Err(ProvableOpError::OnnxParsingError($err_msg.to_string()).into());
            }
        };
    }

pub fn from_path(path: &str) -> Result<ProvableModel<f32>, ProvableOpError> {
    let pmodel = tract_onnx::onnx()
        .model_for_path(path)?
        .into_typed()?
        .into_decluttered()?;
    // so far we dont support batching
    let mut values = SymbolValues::default();
    let symbol = pmodel.sym("batch_size");
    values.set(&symbol, 1);
    let model = pmodel.concretize_dims(&values)?;
    drop(pmodel);
    

    let plan = SimplePlan::new(model)?;
    let onnx_model = plan.model();
    let inference_order = plan.order_without_consts();
    let input_node = onnx_model.node(inference_order[0]);
    let input_source = downcast_to::<TypedSource>(input_node)?;
        debug!("onnx input_source: {:?}", input_source.fact.shape.to_tvec());
    let input_shape = input_source
        .fact
        .shape
        .to_tvec()
        .into_iter()
        .map(|x| match x {
            TDim::Val(v) => Ok(v as usize),
            _ => err(format!("Input {} has unknown input shape: {:?}", input_node.name, x)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut model = ProvableModel::new_from_input_shapes(vec![input_shape.to_vec()]);
    let mut it = inference_order[1..].iter();
    let mut first_node = true;
    let mut last_node_id = 0;
    while !it.is_empty() {
        let (id, zkml_node) = parse_node(onnx_model, &mut it, first_node).map_err(|e| {
            ProvableOpError::OnnxParsingError(format!("Error parsing node: {:?}", e.to_string()))
        })?;
        // let (id, zkml_node) = parse_node(onnx_model, &mut it, first_node)?;
        let desc = zkml_node.operation.describe();
        model.add_node_with_id(id, zkml_node).map_err(|e| {
            ProvableOpError::OnnxParsingError(
                format!("adding node {} -> {:?}", desc, e.to_string()).into(),
            )
        })?;
        first_node = false;
        last_node_id = id;
    }
    model.route_output(vec![Edge::new(last_node_id, 0)])?;
    Ok(model)
}

type LoadFn = fn(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut dyn Iterator<Item = &usize>,
) -> Result<(NodeId, CustomNode), ProvableOpError>;

static PARSER_FACTORY: Lazy<HashMap<&'static str, LoadFn>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("Conv", load_conv as LoadFn);
    m.insert("Gemm.ab", load_gemm as LoadFn);
    m.insert("Relu", load_relu as LoadFn);
    m.insert("Flatten", load_flatten as LoadFn);
    m.insert("Pool", load_maxpool as LoadFn);
    m
});

fn parse_node(
    model: &OnnxModel,
    iter: &mut dyn Iterator<Item = &usize>,
    first_node: bool,
) -> Result<(NodeId, CustomNode), ProvableOpError> {
    let curr_node_id = iter.next().ok_or(anyhow::anyhow!("No nodes left"))?;
    let curr_node = model.node(*curr_node_id);
    debug!(
        "curr_node id {}: {:?} : {:?} <- inputs: {:?}",
        curr_node_id, curr_node.name, curr_node.name, curr_node.inputs
    );
    #[allow(unused_variables)]
    let op_name = &curr_node.name;
    if let Some(layer_name) = PARSER_FACTORY
        .keys()
        .find(|&&layer_name| op_name.contains(layer_name))
    {
        let parser = PARSER_FACTORY.get(layer_name).unwrap();
        let (node_id, mut node) = parser(model, *curr_node_id, curr_node, iter)?;
        if first_node {
            // if the node is the first one, we need to add the input edge as an input to the node
            node.inputs = node
                .inputs
                .into_iter()
                .map(|x| Edge::new_at_edge(x.index))
                .collect();
        }
        debug!(
            "parsed node id: {:?} : {:?} <- inputs: {:?}",
            curr_node_id,
            node.operation.describe(),
            node.inputs
        );
        Ok((node_id, node))
    } else {
        err(format!("Unknown node type: {}", op_name))
    }
}

fn load_flatten(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut dyn Iterator<Item = &usize>,
) -> Result<(NodeId, CustomNode), ProvableOpError> {
    ensure_onnx!(
        node.inputs.len() == 1,
        "Flatten {} must have 1 input",
        node.name
    );
    let node = ProvableNode::new(
        vec![Edge::new(node.inputs[0].node, node.inputs[0].slot)],
        Layer::Flatten(crate::layers::flatten::Flatten),
    );
    Ok((node_id, node))
}

fn load_maxpool(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut dyn Iterator<Item = &usize>,
) -> Result<(NodeId, CustomNode), ProvableOpError> {
    ensure_onnx!(
        node.inputs.len() == 1,
        "MaxPool {} must have 1 input",
        node.name
    );
    let max_node = downcast_to::<MaxPool>(node)?;
    let expected_value: usize = MAXPOOL2D_KERNEL_SIZE;
    if let Some(ref strides) = max_node.pool_spec.strides {
        ensure_onnx!(
            strides.iter().all(|&x| x == expected_value),
            "Strides must be {}",
            expected_value
        );
    }
    match max_node.pool_spec.padding {
        PaddingSpec::Explicit(ref pad0, ref pad1) => {
            ensure_onnx!(
                pad0.iter().all(|&x| x == 0) && pad1.iter().all(|&x| x == 0),
                "Padding must be 0s"
            );
        }
        PaddingSpec::ExplicitOnnxPool(ref pad0, ref pad1, _) => {
            ensure_onnx!(
                pad0.iter().all(|&x| x == 0) && pad1.iter().all(|&x| x == 0),
                "Padding must be 0s"
            );
        }
        PaddingSpec::Valid => (),
        _ => {
            return err(format!(
                "Padding for {} must have valid padding {:?}",
                node.name, max_node.pool_spec.padding
            ));
        }
    }
    ensure_onnx!(
        max_node
            .pool_spec
            .kernel_shape
            .iter()
            .all(|&x| x == expected_value),
        "Kernel shape must be {}",
        expected_value
    );
    if let Some(ref dil) = max_node.pool_spec.dilations {
        ensure_onnx!(dil.iter().all(|&x| x == 1), "Dilations must be 1");
    }
    let zkml_maxpool = Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default()));
    let node = ProvableNode::new(
        vec![Edge::new(node.inputs[0].node, node.inputs[0].slot)],
        zkml_maxpool,
    );
    Ok((node_id, node))
}

fn load_relu(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut dyn Iterator<Item = &usize>,
) -> Result<(NodeId, CustomNode), ProvableOpError> {
    let relu = crate::layers::activation::Relu::new();
    // find the input node that corresponds to the const input of Relu - since tract_onnx transforms
    // a relu operation into Max(input, Const(0))
    // the input node would be the other one.
    ensure_onnx!(
        node.inputs.len() == 2,
        "Relu {} must have 2 inputs",
        node.name
    );
    let real_input_id = match model.node(node.inputs[1].node).op_as::<Const>() {
        Some(_) => {
            node.inputs[0]
        }
        None => {
            ensure_onnx!(
                model.node(node.inputs[0].node).op_as::<Const>().is_some(),
                "Relu {} has no constant input",
                node.name
            );
            node.inputs[1]
        }
    };
    let provable_node = crate::layers::provable::ProvableNode::new(
        vec![Edge::new(real_input_id.node, real_input_id.slot)],
        Layer::Activation(Activation::Relu(relu)),
    );
    Ok((node_id, provable_node))
}

fn load_gemm(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut dyn Iterator<Item = &usize>,
) -> Result<(NodeId, CustomNode), ProvableOpError> {
    let _matrix = downcast_to::<EinSum>(node)?;
    // TODO: we only support matvec for now for onnx models
    // Fetch the input which is constant (e.g. the weights)
    ensure_onnx!(
        node.inputs.len() == 2,
        "Gemm {} must have 2 inputs",
        node.name
    );
    let Some(weight_link) = node
        .inputs
        .iter()
        .rev()
        .find(|&x| is_const(model.node(x.node)))
    else {
        return err(format!("Gemm {} has no constant input", node.name));
    };
    let weight = extract_const_tensor(model.node(weight_link.node))?;
    // find the input node
    let Some(input_link) = node.inputs.iter().find(|&x| x.node != weight_link.node) else {
        return err(format!("Gemm {} has no input", node.name));
    };

    // also extract the bias if any. If there is one, that means the next node is a Add node and we
    // must make sure one of the inputs is the current matrix node. Otherwise that's just a normal add that we don't support.
    // TODO: support general case with Add layer.
    let mut piter = iter.peekable();
    let next_node_id = **piter
        .peek()
        .ok_or(ProvableOpError::OnnxParsingError(format!(
            "Gemm {} has no next node",
            node.name
        )))?;

    let next_node = model.node(next_node_id);
    // if there's a bias, the next op is a TypedBinOp( Add ) node
    let bias_node_id = match downcast_to::<TypedBinOp>(next_node) {
        // safety net
        _ if next_node.inputs.len() != 2 => {
            // no bias, just return the matrix node
            None
        }
        // the operation must be an Add
        Ok(binop) if binop.0.is::<tract_core::ops::math::Add>() => {
            // now on this node, we need to ensure one of the inputs is the current matrix node
            match next_node
                .inputs
                .iter()
                .enumerate()
                .find(|(_i, &x)| x.node == node_id)
            {
                Some((idx, ..)) => {
                    // Now we need to find the bias node, which is the other input to the Add node
                    // and we can extract it as a constant tensor afterwards
                    // since only two elements, we can just do 1 - idx
                    let bias_input = next_node.inputs[1 - idx];
                    // let bias_node = model.node(bias_input.node);
                    // in that case, we move on the iterator, since we already saw the bias node and the Add is part of the dense layer
                    // unwrap is safe here since we peeked already
                    piter.next().unwrap();
                    Some(bias_input.node)
                }
                None => {
                    // no bias, just return the matrix node
                    None
                }
            }
        }
        _ => {
            // no bias, just return the matrix node
            None
        }
    };
    let mut bias_tensor = match bias_node_id {
        Some(bias) => {
            let bias_node = model.node(bias);
            extract_const_tensor(bias_node)?
        }
        None => crate::Tensor::zeros(vec![weight.shape[0]]),
    };
    ensure_onnx!(
        bias_tensor.shape.len() == 1 || bias_tensor.shape.len() == 2,
        "Bias tensor must be 1D or 2D with batch: {:?}",
        bias_tensor.shape
    );
    if bias_tensor.shape.len() == 2 {
        ensure_onnx!(
            bias_tensor.shape[0] == 1,
            "Bias tensor must be 1D with batch: {:?}",
            bias_tensor.shape
        );
        bias_tensor.shape = bias_tensor.shape[1..].to_vec();
    }
    ensure_onnx!(
        bias_tensor.shape[0] == weight.shape[0],
        "Bias tensor must have same size as filter's rows"
    );
    let dense = crate::layers::dense::Dense::new(weight, bias_tensor);
    let provable_node = crate::layers::provable::ProvableNode::new(
        vec![Edge::new(input_link.node, input_link.slot)],
        Layer::Dense(dense),
    );
    // here since the bias addition is the _last_ operation, the next layers are gonna refer
    // to the id of the add node and not the gemm node.
    let provable_id = if bias_node_id.is_some() {
        next_node_id
    } else {
        node_id
    };
    Ok((provable_id, provable_node))
}

fn load_conv(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut dyn Iterator<Item = &usize>,
) -> Result<(NodeId, CustomNode), ProvableOpError> {
    let conv_node = downcast_to::<Conv>(node)?;
    // TODO: once we support different padding and strides, extract the data in this function
    check_conv2d_attributes(conv_node)?;
    // TODO: support for conv without bias
    ensure_onnx!(
        node.inputs.len() == 3,
        "ONNX Conv {} must have at least 3 inputs: {}",
        node.name,
        node.inputs.len()
    );
    let input_link = node.inputs[0];
    let filter_link = node.inputs[1];
    let bias_link = node.inputs[2];
    let filter_node = model.node(filter_link.node);
    let bias_node = model.node(bias_link.node);
    let filter_const = extract_const_tensor(filter_node)?;
    let bias_const = extract_const_tensor(bias_node)?;
    let conv = Convolution::new(filter_const, bias_const);
    let provable_node = crate::layers::provable::ProvableNode::new(
        vec![Edge::new(input_link.node, input_link.slot)],
        Layer::Convolution(conv),
    );
    Ok((node_id, provable_node))
}

fn is_const(node: &OnnxNode) -> bool {
    downcast_to::<Const>(node).is_ok()
}

fn extract_const_tensor(node: &OnnxNode) -> Result<crate::Tensor<f32>, ProvableOpError> {
    let tensor = downcast_to::<Const>(node)?;
    let slice = tensor.0.as_slice::<f32>()?;
    ensure_onnx!(node.outputs.len() == 1, "constant output shape len == 1");
    let Some(shape) = node.outputs[0].fact.shape.as_concrete() else {
        return err(format!("Filter shape {} is not concrete", node.name));
    };
    Ok(crate::Tensor::new(shape.to_vec(), slice.to_vec()))
}

/// Get the conv2d attributes and assert if supported by DeepProve
fn check_conv2d_attributes(node: &Conv) -> Result<(), ProvableOpError> {
    let Some(ref strides) = node.pool_spec.strides else {
        return err(format!("Conv has no strides: {}", node.name()));
    };
    ensure_onnx!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    ensure_onnx!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    let PaddingSpec::Explicit(ref pad0, ref pad1) = &node.pool_spec.padding else {
        return err(format!("Conv has no pads: {}", node.name()));
    };
    ensure_onnx!(
        pad0.iter().all(|&x| x == 0),
        "Padding for {}must be 0s: {:?}",
        node.name(),
        pad0,
    );
    ensure_onnx!(
        pad1.iter().all(|&x| x == 0),
        "Padding for {}must be 0s: {:?}",
        node.name(),
        pad1,
    );
    let Some(ref dilations) = node.pool_spec.dilations else {
        return Err(ProvableOpError::OnnxParsingError(format!(
            "Conv has no dilations: {}",
            node.name()
        ))
        .into());
    };
    ensure_onnx!(
        dilations.iter().all(|&x| x == 1),
        "Dilations for {} must be 1: {:?}",
        node.name(),
        dilations
    );
    let kernel_shape = &node.pool_spec.kernel_shape;
    ensure_onnx!(
        kernel_shape.iter().all(|&x| x > 1),
        "Kernel shape for {} must be > 1: {:?}",
        node.name(),
        kernel_shape
    );
    ensure_onnx!(
        kernel_shape.len() == 2,
        "Kernel shape for {} must be 2D: {:?}",
        node.name(),
        kernel_shape
    );
    ensure_onnx!(
        kernel_shape[0] == kernel_shape[1],
        "Kernel shape for {} must be square: {:?}",
        node.name(),
        kernel_shape
    );
    Ok(())
}

fn err<T>(msg: String) -> Result<T, ProvableOpError> {
    Err(ProvableOpError::OnnxParsingError(msg).into())
}

fn downcast_to<T: Op>(node: &OnnxNode) -> Result<&T, ProvableOpError> {
    match node.op_as::<T>() {
        Some(b) => Ok(b),
        None => {
            return Err(ProvableOpError::OnnxParsingError(format!(
                "Node {} is not a {}",
                node.name,
                std::any::type_name::<T>()
            ))
            .into());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::padding::PaddingMode;

    use super::*;

    #[test]
    fn test_parser_load_conv() {
        let model = from_path("bench-cnn/model.onnx").unwrap();
        let input_shape = model.input_shapes(PaddingMode::NoPadding)[0].clone();

        let input_tensor = crate::tensor::Tensor::random(&input_shape);
        let trace = model.run::<GoldilocksExt2>(&[input_tensor]).unwrap();
        assert!(trace.steps.len() >= 1);
    }
}
