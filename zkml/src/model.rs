use ff_ext::ExtensionField;
use itertools::Itertools;
use log::debug;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    Element,
    activation::{Activation, Relu},
    quantization::{Requant, TensorFielder},
    tensor::Tensor,
};

// The index of the step, starting from the input layer. (proving is done in the opposite flow)
pub type StepIdx = usize;

#[derive(Clone, Debug)]
pub enum Layer {
    // TODO: replace this with a Tensor based implementation
    Dense(Tensor<Element>),
    Activation(Activation),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}

impl Layer {
    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        match self {
            Layer::Dense(ref matrix) => matrix.matvec(input),
            Layer::Activation(activation) => activation.op(input),
            Layer::Requant(info) => {
                // NOTE: we assume we have default quant structure as input
                info.op(input)
            }
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Layer::Dense(ref matrix) => vec![matrix.nrows_2d(), matrix.ncols_2d()],
            Layer::Activation(Activation::Relu(_)) => Relu::shape(),
            Layer::Requant(info) => info.shape(),
        }
    }
    pub fn describe(&self) -> String {
        match self {
            Layer::Dense(ref matrix) => {
                format!(
                    "Dense: ({},{})",
                    matrix.nrows_2d(),
                    matrix.ncols_2d(),
                    // matrix.fmt_integer()
                )
            }
            Layer::Activation(Activation::Relu(_)) => {
                format!("RELU: {}", 1 << Relu::num_vars())
            }
            Layer::Requant(info) => {
                format!("Requant: {}", info.shape()[1])
            }
        }
    }
    /// Prepare the input to return it in the right format expected for the first layer.
    /// for the bias
    pub fn prepare_input(&self, input: Tensor<Element>) -> Tensor<Element> {
        match self {
            Layer::Dense(ref matrix) => {
                if input.get_data().len() == matrix.ncols_2d() {
                    // no need to do anything if it's already at the right format
                    input
                } else {
                    // append 1 for the bias factor and pad to right size
                    let data = input
                        .get_data()
                        .to_vec()
                        .into_iter()
                        .chain(std::iter::once(1))
                        .chain(std::iter::repeat(0))
                        .take(matrix.ncols_2d())
                        .collect_vec();
                    Tensor::new(vec![matrix.ncols_2d()], data)
                }
            }
            _ => panic!("Layer {:?} should not be a first layer", self.describe()),
        }
    }
}

/// NOTE: this doesn't handle dynamism in the model with loops for example for LLMs where it
/// produces each token one by one.
#[derive(Clone, Debug)]
pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            layers: Default::default(),
        }
    }
    pub fn add_layer(&mut self, l: Layer) {
        let after_layer = match l {
            Layer::Dense(ref matrix) => {
                // append a requantization layer after
                // NOTE: since we requantize at each dense step currently, we assume
                // default quantization inputs for matrix and vector
                Some(Layer::Requant(Requant::from_matrix_default(matrix)))
            }
            _ => None,
        };
        self.layers.push(l);
        if let Some(ll) = after_layer {
            self.layers.push(ll);
        }
    }

    /// Prepare the input for the first layer. For example if it's a dense layer, input will be padded correctly
    /// to handle the bias factor.
    pub fn prepare_input(&self, input: Tensor<Element>) -> Tensor<Element> {
        self.layers[0].prepare_input(input)
    }

    pub fn run<'a>(&'a self, input: Tensor<Element>) -> InferenceTrace<'a, Element> {
        let mut trace = InferenceTrace::<Element>::new(input);
        for (id, layer) in self.layers() {
            let input = trace.last_input();
            let output = layer.op(input);
            debug!("step: {}: output: {:?}", id, output);
            let step = InferenceStep { layer, output, id };
            trace.push_step(step);
        }
        trace
    }

    pub fn layers(&self) -> impl DoubleEndedIterator<Item = (StepIdx, &Layer)> {
        self.layers.iter().enumerate()
    }

    pub fn input_shape(&self) -> Vec<usize> {
        let Layer::Dense(mat) = &self.layers[0] else {
            panic!("layer is not starting with a dense layer?");
        };
        vec![mat.ncols_2d()]
    }

    pub fn first_output_shape(&self) -> Vec<usize> {
        let Layer::Dense(mat) = &self.layers[0] else {
            panic!("layer is not starting with a dense layer?");
        };
        vec![mat.nrows_2d()]
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
pub struct InferenceTrace<'a, E> {
    steps: Vec<InferenceStep<'a, E>>,
    /// The initial input to the model
    input: Tensor<E>,
}

impl<'a> InferenceTrace<'a, Element> {
    pub fn to_field<E: ExtensionField>(self) -> InferenceTrace<'a, E> {
        let input = self.input.to_fields();
        let field_steps = self
            .steps
            .par_iter()
            .map(|step| InferenceStep {
                id: step.id,
                layer: step.layer,
                output: step.output.clone().to_fields(),
            })
            .collect::<Vec<_>>();
        InferenceTrace {
            steps: field_steps,
            input,
        }
    }
}

impl<'a, E> InferenceTrace<'a, E> {
    fn new(input: Tensor<E>) -> Self {
        Self {
            steps: Default::default(),
            input,
        }
    }

    pub fn last_step(&self) -> &InferenceStep<'a, E> {
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

    fn push_step(&mut self, step: InferenceStep<'a, E>) {
        self.steps.push(step);
    }

    /// Returns an iterator over (input, step) pairs
    pub fn iter(&self) -> InferenceTraceIterator<'_, 'a, E> {
        InferenceTraceIterator {
            trace: self,
            current_idx: 0,
            end_idx: self.steps.len(),
        }
    }
}

/// Iterator that yields (input, step) pairs for each inference step
pub struct InferenceTraceIterator<'t, 'a, E> {
    trace: &'t InferenceTrace<'a, E>,
    current_idx: usize,
    /// For double-ended iteration
    end_idx: usize,
}

impl<'t, 'a, E> Iterator for InferenceTraceIterator<'t, 'a, E> {
    type Item = (&'t Tensor<E>, &'t InferenceStep<'a, E>);

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

impl<'t, 'a, E> DoubleEndedIterator for InferenceTraceIterator<'t, 'a, E> {
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

pub struct InferenceStep<'a, E> {
    pub id: StepIdx,
    /// Reference to the layer that produced this step
    pub layer: &'a Layer,
    /// Output produced by this layer
    pub output: Tensor<E>,
}

#[cfg(test)]
pub(crate) mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::IOPProverState;

    use crate::{
        Element,
        activation::{Activation, Relu},
        default_transcript,
        model::Layer,
        quantization::{QuantInteger, TensorFielder},
        tensor::Tensor,
        testing::random_bool_vector,
    };

    use super::Model;

    type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const MOD_SELECTOR: usize = 2;

    impl Model {
        /// Returns a random model with specified number of dense layers and a matching input.
        pub fn random(num_dense_layers: usize) -> (Self, Tensor<Element>) {
            let mut model = Model::new();
            let mut rng = thread_rng();
            let mut last_row = rng.gen_range(3..15);
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // last row becomes new column
                    let (nrows, ncols) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let mat = Tensor::random::<QuantInteger>(vec![nrows, ncols])
                        .pad_next_power_of_two_2d();
                    model.add_layer(Layer::Dense(mat));
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            let input_dims = model.layers.first().unwrap().shape();
            // ncols since matrix2vector is summing over the columns
            let input = Tensor::random::<QuantInteger>(vec![input_dims[1]]);
            (model, input)
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::random(3);
        model.run(input);
    }

    #[test]
    fn test_model_run() {
        let mat1 = Tensor::random::<QuantInteger>(vec![10, 11]).pad_next_power_of_two_2d();
        let mat2 =
            Tensor::random::<QuantInteger>(vec![7, mat1.ncols_2d()]).pad_next_power_of_two_2d();
        let input = Tensor::random::<QuantInteger>(vec![mat1.ncols_2d()]);
        let output1 = mat1.matvec(&input);
        let final_output = mat2.matvec(&output1);

        let mut model = Model::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2.clone()));

        let trace = model.run(input.clone()).to_field::<F>();
        assert_eq!(trace.steps.len(), 2);

        // Verify first step
        assert_eq!(trace.steps[0].output, output1.to_fields());

        // Verify second step
        assert_eq!(trace.steps[1].output, final_output.clone().to_fields());
        let (nrow, _) = (mat2.nrows_2d(), mat2.ncols_2d());
        assert_eq!(final_output.get_data().len(), nrow);
    }

    #[test]
    fn test_inference_trace_iterator() {
        let mat1 = Tensor::random::<QuantInteger>(vec![10, 11]).pad_next_power_of_two_2d();
        // let relu1 = Activation::Relu(Relu);
        let mat2 =
            Tensor::random::<QuantInteger>(vec![7, mat1.ncols_2d()]).pad_next_power_of_two_2d();
        // let relu2 = Activation::Relu(Relu);
        let input = Tensor::random::<QuantInteger>(vec![mat1.ncols_2d()]);

        let mut model = Model::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run(input.clone());

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

        // Iterator should be exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_inference_trace_reverse_iterator() {
        let mat1 = Tensor::random::<QuantInteger>(vec![10, 11]).pad_next_power_of_two_2d();
        let mat2 =
            Tensor::random::<QuantInteger>(vec![7, mat1.ncols_2d()]).pad_next_power_of_two_2d();
        let input = Tensor::random::<QuantInteger>(vec![mat1.ncols_2d()]);

        let mut model = Model::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run(input.clone());

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
        let (model, input) = Model::random(2);
        model.describe();
        println!("INPUT: {:?}", input);
        let bb = model.clone();
        let trace = bb.run(input.clone()).to_field::<F>();
        let matrices = model
            .layers()
            .flat_map(|(_id, l)| match l {
                Layer::Dense(ref matrix) => Some(matrix.clone()),
                _ => None,
            })
            .collect_vec();
        let matrices_mle = matrices.iter().map(|m| m.to_mle_2d::<F>()).collect_vec();
        let point1 = random_bool_vector(matrices[1].nrows_2d().ilog2() as usize);
        println!("point1: {:?}", point1);
        let computed_eval1 = trace
            .final_output()
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&point1);
        let flatten_mat1 = matrices_mle[1].fix_high_variables(&point1);
        let input_vector = trace.steps[trace.steps.len() - 2].output.clone();
        // y(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![
            flatten_mat1.clone().into(),
            input_vector.get_data().to_vec().into_mle().into(),
        ];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, s2) = IOPProverState::prove_batch_polys(1, vec![vp], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1, given_eval1);
    }
}
