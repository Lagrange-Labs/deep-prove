use ff_ext::ExtensionField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::debug;

use crate::{
    Element,
    activation::{Activation, Relu},
    convolution::Convolution,
    dense::Dense,
    pooling::Pooling,
    quantization::{Requant, TensorFielder},
    tensor::{ConvData, Tensor},
};

// The index of the step, starting from the input layer. (proving is done in the opposite flow)
pub type StepIdx = usize;

#[derive(Clone, Debug)]
pub enum Layer {
    Dense(Dense),
    // TODO: replace this with a Tensor based implementation
    Convolution(Convolution),
    // Traditional convolution is used for debug purposes. That is because the actual convolution
    // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
    SchoolBookConvolution(Convolution),
    Activation(Activation),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}

pub enum LayerOutput<F>
where
    F: ExtensionField,
{
    NormalOut(Tensor<Element>),
    ConvOut((Tensor<Element>, ConvData<F>)),
}

impl Layer {
    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op<F: ExtensionField>(&self, input: &Tensor<Element>) -> LayerOutput<F> {
        match &self {
            Layer::Dense(ref dense) => LayerOutput::NormalOut(dense.op(input)),
            Layer::Activation(activation) => LayerOutput::NormalOut(activation.op(input)),

            Layer::Convolution(ref filter) => LayerOutput::ConvOut(filter.op(input)),
            // Traditional convolution is used for debug purposes. That is because the actual convolution
            // we use relies on the FFT algorithm. This convolution does not have a snark implementation.
            Layer::SchoolBookConvolution(ref conv_pair) => {
                // LayerOutput::NormalOut(filter.cnn_naive_convolution(input))
                LayerOutput::NormalOut(input.conv2d(&conv_pair.filter, &conv_pair.bias, 1))
            }

            Layer::Requant(info) => {
                // NOTE: we assume we have default quant structure as input
                LayerOutput::NormalOut(info.op(input))
            }
            Layer::Pooling(info) => LayerOutput::NormalOut(info.op(input)),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match &self {
            Layer::Dense(ref dense) => vec![dense.matrix.nrows_2d(), dense.matrix.ncols_2d()],

            Layer::Convolution(ref filter) => filter.get_shape(),
            Layer::SchoolBookConvolution(ref filter) => filter.get_shape(),

            Layer::Activation(Activation::Relu(_)) => Relu::shape(),
            Layer::Requant(info) => info.shape(),
            Layer::Pooling(Pooling::Maxpool2D(info)) => vec![info.kernel_size, info.kernel_size],
        }
    }

    pub fn describe(&self) -> String {
        match &self {
            Layer::Dense(ref dense) => {
                format!(
                    "Dense: ({},{})",
                    dense.matrix.nrows_2d(),
                    dense.matrix.ncols_2d(),
                    // matrix.fmt_integer()
                )
            }
            Layer::Convolution(ref filter) => {
                format!(
                    "Conv: ({},{},{},{})",
                    filter.kw(),
                    filter.kx(),
                    filter.nw(),
                    filter.nw()
                )
            }
            Layer::SchoolBookConvolution(ref _filter) => {
                format!(
                    "Conv: Traditional convolution for debug purposes" /* matrix.fmt_integer() */
                )
            }
            Layer::Activation(Activation::Relu(_)) => {
                format!("RELU: {}", 1 << Relu::num_vars())
            }
            Layer::Requant(info) => {
                format!("Requant: {}", info.shape()[1])
            }
            Layer::Pooling(Pooling::Maxpool2D(info)) => format!(
                "MaxPool2D{{ kernel size: {}, stride: {} }}",
                info.kernel_size, info.stride
            ),
        }
    }
}

/// NOTE: this doesn't handle dynamism in the model with loops for example for LLMs where it
/// produces each token one by one.
#[derive(Clone, Debug)]
pub struct Model {
    padded_in_shape: Vec<usize>,
    layers: Vec<Layer>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            padded_in_shape: Vec::new(),
            layers: Default::default(),
        }
    }
    pub fn add_layer<F: ExtensionField>(&mut self, l: Layer) {
        let after_layer = match l {
            Layer::Dense(ref dense) => {
                // append a requantization layer after
                // NOTE: since we requantize at each dense step currently, we assume
                // default quantization inputs for matrix and vector
                Some(Layer::Requant(dense.requant_info()))
            }
            Layer::Convolution(ref filter) => Some(Layer::Requant(filter.requant_info::<F>())),
            // Layer::Traditional_Convolution(ref filter) => {
            // Some(Layer::Requant(Requant::from_matrix_default(filter)))
            // }
            _ => None,
        };
        self.layers.push(l);
        if let Some(ll) = after_layer {
            self.layers.push(ll);
        }
    }

    pub fn set_input_shape(&mut self, padded_in_shape: Vec<usize>) {
        self.padded_in_shape = padded_in_shape;
    }

    pub fn prepare_input(&self, input: Tensor<Element>) -> Tensor<Element> {
        match self.layers[0] {
            Layer::Dense(ref dense) => input.pad_1d(dense.ncols()),
            Layer::Convolution(_) | Layer::SchoolBookConvolution(_) => {
                assert!(
                    self.padded_in_shape.len() > 0,
                    "Set the input shape using `set_input_shape`"
                );
                let mut input = input;
                input.pad_to_shape(self.padded_in_shape.clone());
                input
            }
            _ => {
                panic!("unable to deal with non-vector input yet");
            }
        }
    }

    pub fn run<'a, E: ExtensionField>(
        &'a self,
        input: Tensor<Element>,
    ) -> InferenceTrace<'a, Element, E> {
        let mut trace = InferenceTrace::<Element, E>::new(input);
        for (id, layer) in self.layers() {
            let input = trace.last_input();
            let output = layer.op(input);
            match output {
                LayerOutput::NormalOut(output) => {
                    debug!("step: {}: output: {:?}", id, output);
                    let empty_matrix: Vec<Vec<E>> = vec![vec![Default::default(); 0]; 0];
                    let conv_data = ConvData::<E>::new(
                        vec![Default::default(); 0],
                        empty_matrix.clone(),
                        empty_matrix.clone(),
                        empty_matrix.clone(),
                        empty_matrix.clone(),
                    );
                    let step = InferenceStep {
                        layer,
                        output,
                        id,
                        conv_data,
                    };
                    trace.push_step(step);
                }
                LayerOutput::ConvOut((output, conv_data)) => {
                    debug!("step: {}: output: {:?}", id, output);
                    let step = InferenceStep {
                        layer,
                        output,
                        id,
                        conv_data,
                    };
                    trace.push_step(step);
                }
            }
        }
        trace
    }

    pub fn layers(&self) -> impl DoubleEndedIterator<Item = (StepIdx, &Layer)> {
        self.layers.iter().enumerate()
    }

    pub fn input_shape(&self) -> Vec<usize> {
        if let Layer::Dense(mat) = &self.layers[0] {
            vec![mat.matrix.nrows_2d()]
        } else if matches!(
            &self.layers[0],
            Layer::Convolution(_) | Layer::SchoolBookConvolution(_)
        ) {
            assert!(
                self.padded_in_shape.len() > 0,
                "Set the input shape using `set_input_shape`"
            );
            self.padded_in_shape.clone()
        } else {
            panic!("layer is not starting with a dense or conv layer?")
        }
    }

    pub fn first_output_shape(&self) -> Vec<usize> {
        if let Layer::Dense(mat) = &self.layers[0] {
            vec![mat.matrix.nrows_2d()]
        } else if let Layer::Convolution(filter) = &self.layers[0] {
            vec![filter.nrows_2d()]
        } else {
            panic!("layer is not starting with a dense layer?")
        }
    }
    /// Prints to stdout
    pub fn describe(&self) {
        println!("Model description:");
        for (idx, layer) in self.layers() {
            println!("\t- {}: {}", idx, layer.describe());
        }
        println!("\n");
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Keeps track of all input and outputs of each layer, with a reference to the layer.
pub struct InferenceTrace<'a, E, F: ExtensionField> {
    pub steps: Vec<InferenceStep<'a, E, F>>,
    /// The initial input to the model
    input: Tensor<E>,
}

impl<'a, F: ExtensionField> InferenceTrace<'a, Element, F> {
    pub fn to_field(self) -> InferenceTrace<'a, F, F> {
        let input = self.input.to_fields();
        let field_steps = self
            .steps
            .par_iter()
            .map(|step| InferenceStep {
                id: step.id,
                layer: step.layer,
                output: step.output.clone().to_fields(),
                conv_data: step.conv_data.clone(),
            })
            .collect::<Vec<_>>();
        InferenceTrace {
            steps: field_steps,
            input,
        }
    }
}

impl<'a, E, F: ExtensionField> InferenceTrace<'a, E, F> {
    fn new(input: Tensor<E>) -> Self {
        Self {
            steps: Default::default(),
            input,
        }
    }

    pub fn last_step(&self) -> &InferenceStep<'a, E, F> {
        self.steps
            .last()
            .expect("can't call last_step on empty inferece")
    }

    /// Useful when building the trace. The next input is either the first input or the last
    /// output.
    fn last_input(&self) -> &Tensor<E> {
        if self.steps.is_empty() {
            &self.input
        } else {
            // safe unwrap since it's not empty
            &self.steps.last().unwrap().output
        }
    }

    /// Returns the final output of the whole trace
    pub fn final_output(&self) -> &Tensor<E> {
        &self
            .steps
            .last()
            .expect("can't call final_output on empty trace")
            .output
    }

    fn push_step(&mut self, step: InferenceStep<'a, E, F>) {
        self.steps.push(step);
    }

    /// Returns an iterator over (input, step) pairs
    pub fn iter(&self) -> InferenceTraceIterator<'_, 'a, E, F> {
        InferenceTraceIterator {
            trace: self,
            current_idx: 0,
            end_idx: self.steps.len(),
        }
    }
}

/// Iterator that yields (input, step) pairs for each inference step
pub struct InferenceTraceIterator<'t, 'a, E, F: ExtensionField> {
    trace: &'t InferenceTrace<'a, E, F>,
    current_idx: usize,
    /// For double-ended iteration
    end_idx: usize,
}

impl<'t, 'a, E, F: ExtensionField> Iterator for InferenceTraceIterator<'t, 'a, E, F> {
    type Item = (&'t Tensor<E>, &'t InferenceStep<'a, E, F>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.end_idx {
            return None;
        }

        let step = &self.trace.steps[self.current_idx];
        let input = if self.current_idx == 0 {
            &self.trace.input
        } else {
            &self.trace.steps[self.current_idx - 1].output
        };

        self.current_idx += 1;
        Some((input, step))
    }
}

impl<'t, 'a, E, F: ExtensionField> DoubleEndedIterator for InferenceTraceIterator<'t, 'a, E, F> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.end_idx {
            return None;
        }

        self.end_idx -= 1;
        let step = &self.trace.steps[self.end_idx];
        let input = if self.end_idx == 0 {
            &self.trace.input
        } else {
            &self.trace.steps[self.end_idx - 1].output
        };

        Some((input, step))
    }
}

pub struct InferenceStep<'a, E, F: ExtensionField> {
    pub id: StepIdx,
    /// Reference to the layer that produced this step
    pub layer: &'a Layer,
    /// Output produced by this layer
    pub output: Tensor<E>,
    pub conv_data: ConvData<F>,
}

#[cfg(test)]
pub(crate) mod test {
    use ark_std::rand::{Rng, thread_rng};
    use ff_ext::ExtensionField;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};

    use crate::{
        Element,
        activation::{Activation, Relu},
        convolution::Convolution,
        default_transcript,
        dense::Dense,
        model::Layer,
        pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
        quantization::TensorFielder,
        tensor::Tensor,
        testing::{NextPowerOfTwo, random_bool_vector, random_vector},
    };

    use super::Model;

    type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const SELECTOR_POOLING: usize = 2;
    // TODO: change to be 3 when `Model` is updated to work with higher dimensional tensors as input.
    const MOD_SELECTOR: usize = 2;

    impl Model {
        /// Returns a random model with specified number of dense layers and a matching input.
        pub fn random(num_dense_layers: usize) -> (Self, Tensor<Element>) {
            let mut model = Model::new();
            let mut rng = thread_rng();
            let mut last_row = rng.gen_range(3..15);
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // if true {
                    // last row becomes new column
                    let (nrows, ncols) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    model.add_layer::<F>(Layer::Dense(
                        Dense::random(vec![nrows, ncols]).pad_next_power_of_two(),
                    ));
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    model.add_layer::<F>(Layer::Activation(Activation::Relu(Relu::new())));
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else if selector % MOD_SELECTOR == SELECTOR_POOLING {
                    // Currently unreachable until Model is updated to work with higher dimensional tensors
                    // TODO: Implement higher dimensional tensor functionality.
                    model.add_layer::<F>(Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())));
                    last_row -= MAXPOOL2D_KERNEL_SIZE - 1;
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            let input_dims = model.layers.first().unwrap().shape();
            // ncols since matrix2vector is summing over the columns
            let input = Tensor::random(vec![input_dims[1]]);
            (model, input)
        }

        /// Returns a model that only contains pooling and relu layers.
        /// The output [`Model`] will contain `num_layers` [`Maxpool2D`] layers and a [`Dense`] layer as well.
        pub fn random_pooling(num_layers: usize) -> (Model, Tensor<Element>) {
            let mut model = Model::new();
            let mut rng = thread_rng();
            // Since Maxpool reduces the size of the output based on the kernel size and the stride we need to ensure that
            // Our starting input size is large enough for the number of layers.

            // If maxpool input matrix has dimensions w x h then output has width and height
            // out_w = (w - kernel_size) / stride + 1
            // out_h = (h - kenrel_size) / stride + 1
            // Hence to make sure we have a large enough tensor for the last step
            // we need to have that w_first > 2^{num_layers + 1} + 2^{num_layers}
            // and likewise for h_first.

            let minimum_initial_size = (1 << num_layers) * (3usize);

            let mut input_shape = (0..3)
                .map(|i| {
                    if i < 1 {
                        rng.gen_range(1..5usize).next_power_of_two()
                    } else {
                        (minimum_initial_size + rng.gen_range(1..4usize)).next_power_of_two()
                    }
                })
                .collect::<Vec<usize>>();

            let input = Tensor::<Element>::random(input_shape.clone());

            let info = Maxpool2D::default();
            for _ in 0..num_layers {
                input_shape
                    .iter_mut()
                    .skip(1)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                model.add_layer::<F>(Layer::Pooling(Pooling::Maxpool2D(info)));
            }

            let (nrows, ncols) = (rng.gen_range(3..15), input_shape.iter().product::<usize>());

            model.add_layer::<F>(Layer::Dense(
                Dense::random(vec![nrows, ncols]).pad_next_power_of_two(),
            ));

            (model, input)
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::random(3);
        model.run::<F>(input);
    }

    pub fn check_tensor_consistency_field<E: ExtensionField>(
        real_tensor: Tensor<E>,
        padded_tensor: Tensor<E>,
    ) {
        let n_x = padded_tensor.shape[1];
        for i in 0..real_tensor.shape[0] {
            for j in 0..real_tensor.shape[1] {
                for k in 0..real_tensor.shape[1] {
                    // if(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] > 0){
                    assert!(
                        real_tensor.data[i * real_tensor.shape[1] * real_tensor.shape[1]
                            + j * real_tensor.shape[1]
                            + k]
                            == padded_tensor.data[i * n_x * n_x + j * n_x + k],
                        "Error in tensor consistency"
                    );
                    //}else{
                    //   assert!(-E::from(-real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] as u64) == E::from(padded_tensor.data[i*n_x*n_x + j*n_x + k] as u64) ,"Error in tensor consistency");
                    //}
                }

                // assert!(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j ] == padded_tensor.data[i*n_x*n_x + j],"Error in tensor consistency");
            }
        }
    }

    fn random_vector_quant(n: usize) -> Vec<Element> {
        // vec![thread_rng().gen_range(-128..128); n]
        random_vector(n)
    }

    #[test]
    fn test_cnn() {
        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![1, 32, 32], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        // println!("in_dimensions: {:?}", in_dimensions);
        let w1 = random_vector_quant(16 * 16);
        let w2 = random_vector_quant(16 * 4 * 16);
        let w3 = random_vector_quant(16 * 8);

        let shape1 = vec![1 << 4, 1 << 0, 1 << 2, 1 << 2]; // [16, 1, 4, 4]
        let shape2 = vec![1 << 2, 1 << 4, 1 << 2, 1 << 2]; // [4, 16, 4, 4]
        let shape3 = vec![1 << 1, 1 << 2, 1 << 2, 1 << 2]; // [2, 4, 4, 4]
        let conv1 = Tensor::new_conv(
            shape1.clone(),
            // [1, 32, 32]
            in_dimensions[0].clone(),
            w1.clone(),
        );
        let conv2 = Tensor::new_conv(
            shape2.clone(),
            // [16, 32, 32]
            in_dimensions[1].clone(),
            w2.clone(),
        );
        let conv3 = Tensor::new_conv(
            shape3.clone(),
            // [4, 32, 32]
            in_dimensions[2].clone(),
            w3.clone(),
        );

        let bias1 = Tensor::zeros(vec![shape1[0]]);
        let bias2 = Tensor::zeros(vec![shape2[0]]);
        let bias3 = Tensor::zeros(vec![shape3[0]]);

        let trad_conv1 = Tensor::new(shape1.clone(), w1.clone());
        let trad_conv2 = Tensor::new(shape2.clone(), w2.clone());
        let trad_conv3 = Tensor::new(shape3.clone(), w3.clone());

        let mut model = Model::new();
        model.add_layer::<F>(Layer::Convolution(Convolution::new(
            conv1.clone(),
            bias1.clone(),
        )));
        model.add_layer::<F>(Layer::Convolution(Convolution::new(
            conv2.clone(),
            bias2.clone(),
        )));
        model.add_layer::<F>(Layer::Convolution(Convolution::new(
            conv3.clone(),
            bias3.clone(),
        )));

        let input = Tensor::new(vec![1, 32, 32], random_vector_quant(1024));
        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input.clone());

        let mut model2 = Model::new();
        model2.add_layer::<F>(Layer::SchoolBookConvolution(Convolution::new(
            trad_conv1, bias1,
        )));
        model2.add_layer::<F>(Layer::SchoolBookConvolution(Convolution::new(
            trad_conv2, bias2,
        )));
        model2.add_layer::<F>(Layer::SchoolBookConvolution(Convolution::new(
            trad_conv3, bias3,
        )));
        let trace2 = model.run::<F>(input.clone());

        check_tensor_consistency_field::<GoldilocksExt2>(
            trace2.final_output().clone().to_fields(),
            trace.final_output().clone().to_fields(),
        );

        let _out1: &Tensor<i128> = trace.final_output();
    }

    #[test]
    fn test_conv_maxpool() {
        let input_shape_padded = vec![3usize, 32, 32].next_power_of_two();
        let shape1 = vec![6, 3, 5, 5].next_power_of_two();

        let filter1 = Tensor::new_conv(
            shape1.clone(),
            input_shape_padded.clone(),
            random_vector_quant(shape1.prod()),
        );

        let bias1 = Tensor::random(vec![shape1[0]]);

        let mut model = Model::new();
        model.add_layer::<F>(Layer::Convolution(Convolution::new(
            filter1.clone(),
            bias1.clone(),
        )));
        model.add_layer::<F>(Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())));

        let input = Tensor::random(input_shape_padded.clone());
        let _: crate::model::InferenceTrace<'_, _, GoldilocksExt2> = model.run::<F>(input.clone());
    }

    #[test]
    fn test_model_manual_run() {
        let dense1 = Dense::random(vec![10, 11]).pad_next_power_of_two();
        let dense2 = Dense::random(vec![7, dense1.ncols()]).pad_next_power_of_two();
        let input = Tensor::random(vec![dense1.ncols()]);
        let output1 = dense1.op(&input);
        let requant = dense1.requant_info();
        let requantized_output1 = requant.op(&output1);
        let final_output = dense2.op(&requantized_output1);

        let mut model = Model::new();
        model.add_layer::<F>(Layer::Dense(dense1.clone()));
        model.add_layer::<F>(Layer::Dense(dense2.clone()));

        let trace = model.run::<F>(input.clone());
        // 4 steps because we requant after each dense layer
        assert_eq!(trace.steps.len(), 4);

        // Verify first step
        assert_eq!(trace.steps[0].output, output1);

        // Verify second step
        assert_eq!(trace.steps[2].output, final_output.clone());
        let (nrow, _) = (dense2.nrows(), dense2.ncols());
        assert_eq!(final_output.get_data().len(), nrow);
    }

    #[test]
    fn test_inference_trace_iterator() {
        let dense1 = Dense::random(vec![10, 11]).pad_next_power_of_two();
        // let relu1 = Activation::Relu(Relu);
        let dense2 = Dense::random(vec![7, dense1.ncols()]).pad_next_power_of_two();
        // let relu2 = Activation::Relu(Relu);
        let input = Tensor::random(vec![dense1.ncols()]);

        let mut model = Model::new();
        model.add_layer::<F>(Layer::Dense(dense1));
        model.add_layer::<F>(Layer::Dense(dense2));

        let trace = model.run::<F>(input.clone());

        // Verify iterator yields correct input/output pairs
        let mut iter = trace.iter();

        // First step should have original input
        let (first_input, first_step) = iter.next().unwrap();
        assert_eq!(*first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Second step should have first step's output as input
        let (second_input, second_step) = iter.next().unwrap();
        assert_eq!(*second_input, trace.steps[0].output);
        assert_eq!(second_step.output, trace.steps[1].output);

        // Third step should have second step's output as input
        let (third_input, third_step) = iter.next().unwrap();
        assert_eq!(*third_input, trace.steps[1].output);
        assert_eq!(third_step.output, trace.steps[2].output);

        // Fourth step should have third step's output as input
        let (fourth_input, fourth_step) = iter.next().unwrap();
        assert_eq!(*fourth_input, trace.steps[2].output);
        assert_eq!(fourth_step.output, trace.steps[3].output);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_inference_trace_reverse_iterator() {
        let dense1 = Dense::random(vec![10, 11]).pad_next_power_of_two();

        let input = Tensor::random(vec![dense1.ncols()]);

        let mut model = Model::new();
        model.add_layer::<F>(Layer::Dense(dense1));

        let trace = model.run::<F>(input.clone());

        // Test reverse iteration
        let mut rev_iter = trace.iter().rev();

        // Last step should come first in reverse
        let (last_input, last_step) = rev_iter.next().unwrap();
        assert_eq!(*last_input, trace.steps[0].output);
        assert_eq!(last_step.output, trace.steps[1].output);

        // First step should come last in reverse
        let (first_input, first_step) = rev_iter.next().unwrap();
        assert_eq!(*first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Iterator should be exhausted
        assert!(rev_iter.next().is_none());
    }

    use ff::Field;
    #[test]
    fn test_model_sequential() {
        let (model, input) = Model::random(1);
        model.describe();
        println!("INPUT: {:?}", input);
        let bb = model.clone();
        let trace = bb.run::<F>(input.clone()).to_field();
        let dense_layers = model
            .layers()
            .flat_map(|(_id, l)| match l {
                Layer::Dense(ref dense) => Some(dense.clone()),
                _ => None,
            })
            .collect_vec();
        let matrices_mle = dense_layers
            .iter()
            .map(|d| d.matrix.to_mle_2d::<F>())
            .collect_vec();
        let point1 = random_bool_vector(dense_layers[0].matrix.nrows_2d().ilog2() as usize);
        println!("point1: {:?}", point1);
        // -2 because there is always requant after each dense layer
        let computed_eval1 = trace.steps[trace.steps.len() - 2]
            .output
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&point1);
        let flatten_mat1 = matrices_mle[0].fix_high_variables(&point1);
        let bias_eval = dense_layers[0]
            .bias
            .evals_flat::<F>()
            .into_mle()
            .evaluate(&point1);
        let computed_eval1_no_bias = computed_eval1 - bias_eval;
        let input_vector = trace.input.clone();
        // since y = SUM M(j,i) x(i) + B(j)
        // then
        // y(r) - B(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![
            flatten_mat1.clone().into(),
            input_vector.get_data().to_vec().into_mle().into(),
        ];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, _state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, _s2) =
            IOPProverState::prove_batch_polys(1, vec![vp.clone()], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1_no_bias, given_eval1);

        let _subclaim = IOPVerifierState::<F>::verify(
            computed_eval1_no_bias,
            &proof,
            &vp.aux_info,
            &mut default_transcript(),
        );
    }

    use crate::{Context, IO, Prover, lookup, verify};
    use transcript::BasicTranscript;

    #[test]
    fn test_single_matvec_prover() {
        let w1 = random_vector_quant(1024 * 1024);
        let conv1 = Tensor::new(vec![1024, 1024], w1.clone());
        let w2 = random_vector_quant(1024);
        let conv2 = Tensor::new(vec![1024], w2.clone());

        let mut model = Model::new();
        model.add_layer::<F>(Layer::Dense(Dense::new(conv1, conv2)));
        model.describe();
        let input = Tensor::new(vec![1024], random_vector_quant(1024));
        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input.clone());
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        let output = trace.final_output().clone();
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, lookup::LogUp> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _, lookup::LogUp>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_cnn_prover() {
        let n_w = 1 << 2;
        let k_w = 1 << 4;
        let n_x = 1 << 5;
        let k_x = 1 << 1;

        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        let w1 = random_vector_quant(k_w * k_x * n_w * n_w);
        let conv1 = Tensor::new_conv(
            vec![k_w, k_x, n_w, n_w],
            in_dimensions[0].clone(),
            w1.clone(),
        );
        let mut model = Model::new();
        model.add_layer::<F>(Layer::Convolution(Convolution::new(
            conv1.clone(),
            Tensor::new(vec![conv1.kw()], random_vector_quant(conv1.kw())),
        )));
        model.describe();
        let input = Tensor::new(vec![k_x, n_x, n_x], random_vector_quant(n_x * n_x * k_x));
        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input.clone());
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2>::generate(&model, Some(input.dims()))
            .expect("Unable to generate context");
        let output = trace.final_output().clone();

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, lookup::LogUp> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _, lookup::LogUp>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_cnn_prover() {
        for i in 0..3 {
            for j in 2..5 {
                for l in 0..4 {
                    for n in 1..(j - 1) {
                        let n_w = 1 << n;
                        let k_w = 1 << l;
                        let n_x = 1 << j;
                        let k_x = 1 << i;

                        let mut in_dimensions: Vec<Vec<usize>> =
                            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

                        for i in 0..in_dimensions.len() {
                            for j in 0..in_dimensions[0].len() {
                                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
                            }
                        }
                        let w1 = random_vector_quant(k_w * k_x * n_w * n_w);
                        let conv1 = Tensor::new_conv(
                            vec![k_w, k_x, n_w, n_w],
                            in_dimensions[0].clone(),
                            w1.clone(),
                        );

                        let mut model = Model::new();
                        model.add_layer::<F>(Layer::Convolution(Convolution::new(
                            conv1.clone(),
                            Tensor::new(vec![conv1.kw()], random_vector_quant(conv1.kw())),
                        )));
                        model.describe();
                        let input =
                            Tensor::new(vec![k_x, n_x, n_x], random_vector_quant(n_x * n_x * k_x));
                        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
                            model.run::<F>(input.clone());
                        let mut tr: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        let ctx = Context::<GoldilocksExt2>::generate(&model, Some(input.dims()))
                            .expect("Unable to generate context");
                        let output = trace.final_output().clone();
                        let prover: Prover<
                            '_,
                            GoldilocksExt2,
                            BasicTranscript<GoldilocksExt2>,
                            lookup::LogUp,
                        > = Prover::new(&ctx, &mut tr);
                        let proof = prover.prove(trace).expect("unable to generate proof");
                        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        let io = IO::new(input.to_fields(), output.to_fields());
                        verify::<_, _, lookup::LogUp>(ctx, proof, io, &mut verifier_transcript)
                            .unwrap();
                    }
                }
            }
        }
    }
}
