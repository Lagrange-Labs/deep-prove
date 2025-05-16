use crate::{
    ModelType,
    layers::{
        Layer,
        activation::Activation,
        convolution::Convolution,
        pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
        provable::{Edge, Node as ProvableNode, NodeId, OpInfo},
    },
    model::Model,
    padding::PaddingMode,
};
use anyhow::{Context, Result, bail, ensure};
use std::{collections::HashMap, iter::Peekable};
use tracing::debug;
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
    tract_hir::ops::{cnn::PaddingSpec, konst::Const},
};

type OnnxModel = Graph<TypedFact, Box<dyn TypedOp + 'static>>;
type OnnxNode = Node<TypedFact, Box<dyn TypedOp + 'static>>;
type CustomNode = crate::layers::provable::Node<f32>;

macro_rules! ensure_onnx {
        // Match with format args
        ($cond:expr, $err_fmt:literal, $($args:expr),+ $(,)?) => {
            ensure!($cond,
                "when parsing onnx model: {}",
                format!($err_fmt, $($args),+),
            );
        };
        // Match with plain string (no args)
        ($cond:expr, $err_msg:literal $(,)?) => {
            ensure!($cond,
                "when parsing onnx model: {}",
                $err_msg,
            );
        };
    }

pub fn from_path(path: &str) -> Result<Model<f32>> {
    let model_type = ModelType::from_onnx(path).context("can't prove unknown model:")?;
    let model = {
        let pmodel = tract_onnx::onnx()
            .model_for_path(path)?
            .into_typed()?
            .into_decluttered()?;
        // so far we dont support batching
        let mut values = SymbolValues::default();
        let symbol = pmodel.sym("batch_size");
        values.set(&symbol, 1);
        pmodel.concretize_dims(&values)?
    };

    let plan = SimplePlan::new(model)?;
    let onnx_model = plan.model();
    let inference_order = plan.order_without_consts();
    let input_node = onnx_model.node(inference_order[0]);
    let input_source = downcast_to::<TypedSource>(input_node)?;
    debug!("onnx input_source: {:?}", input_source.fact.shape.to_tvec());
    let mut input_shape = input_source
        .fact
        .shape
        .to_tvec()
        .into_iter()
        .map(|x| match x {
            TDim::Val(v) => Ok(v as usize),
            _ => err(format!(
                "Input {} has unknown input shape: {:?}",
                input_node.name, x
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if model_type == ModelType::CNN || model_type == ModelType::MLP {
        if input_shape[0] == 1 {
            input_shape.remove(0);
        }
    }

    let mut pmodel =
        Model::new_from_input_shapes(vec![input_shape.to_vec()], PaddingMode::NoPadding);
    let mut it = inference_order[1..].iter().peekable();
    let mut first_node = true;
    let mut last_node_id = 0;
    let parser = ParserFactory::init();
    while !it.is_empty() {
        let (id, zkml_node) = parser
            .parse_node(onnx_model, &mut it, first_node)
            .context("Error parsing node")?;
        // let (id, zkml_node) = parse_node(onnx_model, &mut it, first_node)?;
        let desc = zkml_node.operation.describe();
        pmodel
            .add_node_with_id(id, zkml_node)
            .context(format!("adding node {desc}:"))?;
        first_node = false;
        last_node_id = id;
    }
    let outputs = onnx_model
        .output_outlets()?
        .iter()
        .map(|outlet| Edge::new(outlet.node, outlet.slot))
        .collect::<Vec<_>>();
    assert!(
        outputs
            .iter()
            .any(|edge| edge.node.unwrap() == last_node_id)
    );
    pmodel.route_output(Some(outputs))?;
    Ok(pmodel)
}

type LoadFn<'a, I> = fn(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut Peekable<I>,
) -> Result<(NodeId, CustomNode)>;

// static PARSER_FACTORY: Lazy<HashMap<&'static str, LoadFn>> = Lazy::new(|| {
// let mut m = HashMap::new();
// m.insert("Conv", load_conv as LoadFn);
// m.insert("Gemm.ab", load_gemm as LoadFn);
// m.insert("MatMul", load_gemm as LoadFn);
// m.insert("Relu", load_relu as LoadFn);
// m.insert("Flatten", load_flatten as LoadFn);
// m.insert("Pool", load_maxpool as LoadFn);
// m
// });

struct ParserFactory<'a, I: Iterator<Item = &'a usize> + Sized>(
    HashMap<&'static str, LoadFn<'a, I>>,
);

impl<'a, I: Iterator<Item = &'a usize> + Sized> ParserFactory<'a, I> {
    fn init() -> Self {
        let mut m = HashMap::new();
        m.insert("Conv", load_conv as LoadFn<'a, I>);
        m.insert("Gemm.ab", load_gemm as LoadFn<'a, I>);
        m.insert("MatMul", load_gemm as LoadFn<'a, I>); //ToDo: currently MatMul is only used for dense layers without bias;
        // we would probably need an ad-hoc method when introducing general purpose matrix multiplication layer
        m.insert("Relu", load_relu as LoadFn<'a, I>);
        m.insert("Flatten", load_flatten as LoadFn<'a, I>);
        m.insert("Pool", load_maxpool as LoadFn<'a, I>);
        ParserFactory(m)
    }

    fn parse_node(
        &self,
        model: &OnnxModel,
        iter: &mut Peekable<I>,
        first_node: bool,
    ) -> Result<(NodeId, CustomNode)> {
        let curr_node_id = iter.next().ok_or(anyhow::anyhow!("No nodes left"))?;
        let curr_node = model.node(*curr_node_id);
        debug!(
            "curr_node id {}: {:?} : {:?} <- inputs: {:?}",
            curr_node_id, curr_node.name, curr_node.name, curr_node.inputs
        );
        #[allow(unused_variables)]
        let op_name = &curr_node.name;
        if let Some(layer_name) = self
            .0
            .keys()
            .find(|&&layer_name| op_name.contains(layer_name))
        {
            let parser = self.0.get(layer_name).unwrap();
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
            err(format!("Unknown node type: {op_name}"))
        }
    }
}

fn load_flatten<'a, I: Iterator<Item = &'a usize> + Sized>(
    _model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<(NodeId, CustomNode)> {
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

fn load_maxpool<'a, I: Iterator<Item = &'a usize> + Sized>(
    _model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<(NodeId, CustomNode)> {
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
        "Kernel shape must be square with size {}",
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

fn load_relu<'a, I: Iterator<Item = &'a usize> + Sized>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<(NodeId, CustomNode)> {
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
        Some(_) => node.inputs[0],
        None => {
            ensure_onnx!(
                model.node(node.inputs[0].node).op_as::<Const>().is_some(),
                "Relu {} has no constant input",
                node.name
            );
            node.inputs[1]
        }
    };
    let provable_node = crate::layers::provable::Node::new(
        vec![Edge::new(real_input_id.node, real_input_id.slot)],
        Layer::Activation(Activation::Relu(relu)),
    );
    Ok((node_id, provable_node))
}

fn load_gemm<'a, I: Iterator<Item = &'a usize> + Sized>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    iter: &mut Peekable<I>,
) -> Result<(NodeId, CustomNode)> {
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
    let mut weight = extract_const_tensor(model.node(weight_link.node))?;
    // here maybe the weights are still in 3d shape, so we need to flatten the input portion
    // since it's always [out, in...], we just take everything after the first dimension and flatten it
    let mut weight_shape = weight.get_shape();
    if weight_shape.len() > 2 {
        let input_flattened = weight_shape[1..].iter().product::<usize>();
        weight_shape = vec![weight_shape[0], input_flattened];
        weight.shape = weight_shape.clone();
    }
    ensure_onnx!(
        weight.is_matrix(),
        "Weight for Gemm must be a matrix: {:?}",
        weight.get_shape()
    );
    // find the input node
    let Some(input_link) = node.inputs.iter().find(|&x| x.node != weight_link.node) else {
        return err(format!("Gemm {} has no input", node.name));
    };

    // check if the weight matrix needs to be transposed
    let input_node = model.node(input_link.node);
    let input_shape = get_node_output_shape(input_node, input_link.slot)?;
    // NOTE: flatten the input shape always because tract_onnx can skip the Flatten layer
    let mut input_shape = vec![input_shape.iter().product::<usize>()];

    if input_shape.len() != 1 {
        assert!(
            input_shape[0] == 1,
            "First dimension of Gemm layer input should be 1. Input shape was: {:?}",
            input_shape
        );
        input_shape.remove(0);
    }
    ensure_onnx!(
        input_shape.len() == 1,
        "Input shape for Gemm must be a vector, found {:?}",
        input_shape
    );
    let mut weight_shape = weight.get_shape();
    // If the weights are a 1D vector we insert a 1 in the shape after checking everything lines up
    if weight_shape.len() == 1 {
        ensure_onnx!(
            weight_shape[0] == input_shape[0],
            "Incompatible shapes found for Gemm node: input shape is {:?}, weight shape is {:?}",
            input_shape,
            weight_shape,
        );
        weight.shape.insert(0, 1);
    } else {
        if weight_shape[1] != input_shape[0] {
            weight = weight.transpose();
            weight_shape = weight.get_shape();
        }
        ensure_onnx!(
            *weight_shape.last().unwrap() == input_shape[0],
            "Incompatible shapes found for Gemm node: input shape is {:?}, weight shape is {:?}",
            input_shape,
            weight_shape,
        );
    }

    // also extract the bias if any. If there is one, that means the next node is a Add node and we
    // must make sure one of the inputs is the current matrix node. Otherwise that's just a normal add that we don't support.
    // TODO: support general case with Add layer.
    let (edge_id, bias_node_id) = match iter.peek() {
        // no next node, no bias
        None => (node_id, None),
        Some(&&next_node_id) => {
            let next_node = model.node(next_node_id);
            // if there's a bias, the next op is a TypedBinOp( Add ) node
            match downcast_to::<TypedBinOp>(next_node) {
                // safety net
                _ if next_node.inputs.len() != 2 => {
                    // no bias, just return the matrix node
                    (node_id, None)
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
                            iter.next().unwrap();
                            (next_node_id, Some(bias_input.node))
                        }
                        None => {
                            // no bias, just return the matrix node
                            (node_id, None)
                        }
                    }
                }
                _ => {
                    // no bias, just return the matrix node
                    (node_id, None)
                }
            }
        }
    };
    let mut bias_tensor = match bias_node_id {
        Some(bias) => {
            let bias_node = model.node(bias);
            extract_const_tensor(bias_node)?
        }
        // we always require a bias tensor in current proving logic
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
    let provable_node = crate::layers::provable::Node::new(
        vec![Edge::new(input_link.node, input_link.slot)],
        Layer::Dense(dense),
    );
    // here since the bias addition is the _last_ operation, the next layers are gonna refer
    // to the id of the add node and not the gemm node.
    Ok((edge_id, provable_node))
}

fn load_conv<'a, I: Iterator<Item = &'a usize> + Sized>(
    model: &OnnxModel,
    node_id: NodeId,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<(NodeId, CustomNode)> {
    let conv_node = downcast_to::<Conv>(node)?;
    // TODO: once we support different padding and strides, extract the data in this function
    check_conv2d_attributes(conv_node)?;
    // TODO: support for conv without bias
    ensure_onnx!(
        node.inputs.len() == 3,
        "ONNX Conv {} must have exactly 3 inputs: {}",
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
    let conv = if bias_const.is_empty() {
        // it's a convolution layer without bias
        Convolution::new_without_bias(filter_const)
    } else {
        Convolution::new(filter_const, bias_const)
    };
    let provable_node = crate::layers::provable::Node::new(
        vec![Edge::new(input_link.node, input_link.slot)],
        Layer::Convolution(conv),
    );
    Ok((node_id, provable_node))
}

fn is_const(node: &OnnxNode) -> bool {
    downcast_to::<Const>(node).is_ok()
}

fn extract_const_tensor(node: &OnnxNode) -> Result<crate::Tensor<f32>> {
    let tensor = downcast_to::<Const>(node)?;
    let slice = tensor.0.as_slice::<f32>()?;
    ensure_onnx!(node.outputs.len() == 1, "constant output shape len == 1");
    let Some(shape) = node.outputs[0].fact.shape.as_concrete() else {
        return err(format!("Filter shape {} is not concrete", node.name));
    };
    Ok(crate::Tensor::new(shape.to_vec(), slice.to_vec()))
}

fn get_node_output_shape(node: &OnnxNode, output_idx: usize) -> Result<Vec<usize>> {
    ensure_onnx!(
        output_idx < node.outputs.len(),
        "Trying to get output {} of node {}, but there are only {} outputs",
        output_idx,
        node.name,
        node.outputs.len(),
    );
    let Some(shape) = node.outputs[output_idx].fact.shape.as_concrete() else {
        return err(format!("shape of node {} is not concrete", node.name));
    };
    Ok(shape.to_vec())
}

/// Get the conv2d attributes and assert if supported by DeepProve
fn check_conv2d_attributes(node: &Conv) -> Result<()> {
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
        return err(format!("Conv has no dilations: {}", node.name()));
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

fn err<T>(msg: String) -> Result<T> {
    bail!("Onnx parsing: {msg}")
}

fn downcast_to<T: Op>(node: &OnnxNode) -> Result<&T> {
    match node.op_as::<T>() {
        Some(b) => Ok(b),
        None => err(format!(
            "Node {} is not a {}",
            node.name,
            std::any::type_name::<T>()
        )),
    }
}

#[cfg(test)]
mod tests {

    use goldilocks::GoldilocksExt2;

    use super::*;

    #[test]
    fn test_parser_load_conv() {
        let model = from_path("assets/scripts/CNN/cnn-cifar-01.onnx").unwrap();
        let input_shape = model.input_shapes()[0].clone();

        let input_tensor = crate::tensor::Tensor::random(&input_shape);
        let trace = model.run::<GoldilocksExt2>(&[input_tensor]).unwrap();
        assert!(trace.steps.len() >= 1);
    }
}
