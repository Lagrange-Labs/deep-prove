use crate::dense::Dense;
use ff_ext::ExtensionField;
use itertools::Itertools;
use log::debug;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    Element,
    activation::{Activation, Relu},
    quantization::{Quantizer, Requant, TensorFielder},
    tensor::Tensor,
};

// The index of the step, starting from the input layer. (proving is done in the opposite flow)
pub type StepIdx = usize;

#[derive(Clone, Debug)]
pub enum Layer {
    Dense(Dense),
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
            Layer::Dense(ref dense) => dense.op(input),
            Layer::Activation(activation) => activation.op(input),
            Layer::Requant(info) => {
                // NOTE: we assume we have default quant structure as input
                info.op(input)
            }
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Layer::Dense(ref dense) => vec![dense.matrix.nrows_2d(), dense.matrix.ncols_2d()],
            Layer::Activation(Activation::Relu(_)) => Relu::shape(),
            Layer::Requant(info) => info.shape(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Layer::Dense(ref dense) => {
                format!(
                    "Dense: ({},{})",
                    dense.matrix.nrows_2d(),
                    dense.matrix.ncols_2d(),
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
            Layer::Dense(ref dense) => {
                // append a requantization layer after
                // NOTE: since we requantize at each dense step currently, we assume
                // default quantization inputs for matrix and vector
                Some(Layer::Requant(dense.requant_info()))
            }
            _ => None,
        };
        self.layers.push(l);
        if let Some(ll) = after_layer {
            self.layers.push(ll);
        }
    }

    pub fn prepare_input(&self, input: Tensor<Element>) -> Tensor<Element> {
        match self.layers[0] {
            Layer::Dense(ref dense) => input.pad_1d(dense.nrows()),
            _ => {
                panic!("unable to deal with non-vector input yet");
            }
        }
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
        vec![mat.matrix.ncols_2d()]
    }

    pub fn first_output_shape(&self) -> Vec<usize> {
        let Layer::Dense(mat) = &self.layers[0] else {
            panic!("layer is not starting with a dense layer?");
        };
        vec![mat.matrix.nrows_2d()]
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
        dense::Dense,
        model::Layer,
        quantization::{Requant, TensorFielder},
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
                // if selector % MOD_SELECTOR == SELECTOR_DENSE {
                if true {
                    // last row becomes new column
                    let (nrows, ncols) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    model.add_layer(Layer::Dense(
                        Dense::random(vec![nrows, ncols]).pad_next_power_of_two(),
                    ));
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
            let input = Tensor::random(vec![input_dims[1]]);
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
        let dense1 = Dense::random(vec![10, 11]).pad_next_power_of_two();
        let dense2 = Dense::random(vec![7, dense1.ncols()]).pad_next_power_of_two();
        let input = Tensor::random(vec![dense1.ncols()]);
        let output1 = dense1.op(&input);
        let requant = dense1.requant_info();
        let requantized_output1 = requant.op(&output1);
        let final_output = dense2.op(&requantized_output1);

        let mut model = Model::new();
        model.add_layer(Layer::Dense(dense1.clone()));
        model.add_layer(Layer::Dense(dense2.clone()));

        let trace = model.run(input.clone()).to_field::<F>();
        // 4 steps because we requant after each dense layer
        assert_eq!(trace.steps.len(), 4);

        // Verify first step
        assert_eq!(trace.steps[0].output, output1.to_fields());

        // Verify second step
        assert_eq!(trace.steps[2].output, final_output.clone().to_fields());
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
        model.add_layer(Layer::Dense(dense1));
        model.add_layer(Layer::Dense(dense2));

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
        model.add_layer(Layer::Dense(dense1));

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
        let (model, input) = Model::random(1);
        model.describe();
        println!("INPUT: {:?}", input);
        let bb = model.clone();
        let trace = bb.run(input.clone()).to_field::<F>();
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
        // y(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![
            flatten_mat1.clone().into(),
            input_vector.get_data().to_vec().into_mle().into(),
        ];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, _state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, _s2) = IOPProverState::prove_batch_polys(1, vec![vp], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1_no_bias, given_eval1);
    }
}
