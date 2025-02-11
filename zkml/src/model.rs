use ff_ext::ExtensionField;
use multilinear_extensions::mle::DenseMultilinearExtension;

use crate::matrix::Matrix;

/// A layer has a unique ID associated to it in the model
pub type PolyID = usize;

#[derive(Clone, Debug)]
pub enum Layer<E> {
    // TODO: replace this with a Tensor based implementation
    Dense(Matrix<E>),
}

impl<E: ExtensionField> Layer<E> {
    pub fn dim(&self) -> (usize, usize) {
        match self {
            Layer::Dense(ref matrix) => (matrix.nrows(), matrix.ncols()),
        }
    }

    /// Run the operation associated with that layer with the given input
    // TODO: move to tensor library : right now it works because we assume there is only Dense
    // layer which is matmul
    pub fn op(&self, input: &[E]) -> Vec<E> {
        match self {
            Layer::Dense(ref matrix) => matrix.matmul(input),
        }
    }

    pub fn mle(&self) -> DenseMultilinearExtension<E> {
        match self {
            Layer::Dense(ref matrix) => matrix.to_mle(),
        }
    }

    pub fn evals(&self) -> Vec<E> {
        match self {
            Layer::Dense(ref matrix) => matrix.evals(),
        }
    }
}

/// NOTE: this doesn't handle dynamism in the model with loops for example for LLMs where it
/// produces each token one by one.
#[derive(Clone, Debug)]
pub struct Model<E> {
    layers: Vec<Layer<E>>,
}

impl<E: ExtensionField> Model<E> {
    pub fn new() -> Self {
        Self {
            layers: Default::default(),
        }
    }
    fn add_layer(&mut self, l: Layer<E>) {
        self.layers.push(l);
    }

    pub fn run<'a>(&'a self, input: Vec<E>) -> InferenceTrace<'a, E> {
        let mut trace = InferenceTrace::new(input);
        for (id,layer) in self.layers() {
            let input = trace.last_input();
            let output = layer.op(input);
            let step = InferenceStep { 
                layer, 
                output,
                id,
            };
            trace.push_step(step);
        }
        trace
    }

    pub fn layers(&self) -> impl DoubleEndedIterator<Item = (PolyID, &Layer<E>)> {
        self.layers.iter().enumerate()
    }

    pub fn nlayers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, idx: usize) -> Option<(PolyID, &Layer<E>)> {
        self.layers.get(idx).map(|l| (idx, l))
    }
}

/// Keeps track of all input and outputs of each layer, with a reference to the layer.
pub struct InferenceTrace<'a, E> {
    steps: Vec<InferenceStep<'a, E>>,
    /// The initial input to the model
    input: Vec<E>,
}

impl<'a, E> InferenceTrace<'a, E> {
    fn new(input: Vec<E>) -> Self {
        Self {
            steps: Default::default(),
            input,
        }
    }

    /// Useful when building the trace. The next input is either the first input or the last
    /// output.
    fn last_input(&self) -> &[E] {
        if self.steps.is_empty() {
            &self.input
        } else {
            // safe unwrap since it's not empty
            &self.steps.last().unwrap().output
        }
    }

    /// Returns the input that led to this inference trace
    pub fn input(&self) -> &[E] {
        &self.input
    }

    /// Returns the final output of the whole trace
    pub fn final_output(&self) -> &[E] {
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
    type Item = (&'t [E], &'t InferenceStep<'a, E>);

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
    pub id: PolyID,
    /// Reference to the layer that produced this step
    pub layer: &'a Layer<E>,
    /// Output produced by this layer
    pub output: Vec<E>,
}

#[cfg(test)]
pub(crate) mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;

    use crate::{matrix::Matrix, model::Layer};

    use super::Model;
    use ff_ext::ExtensionField;

    type F = GoldilocksExt2;

    pub fn random_vector<E: ExtensionField>(n: usize) -> Vec<E> {
        let mut rng = thread_rng();
        (0..n).map(|_| E::random(&mut rng)).collect_vec()
    }

    pub fn random_bool_vector<E: ExtensionField>(n: usize) -> Vec<E> {
        let mut rng = thread_rng();
        (0..n)
            .map(|_| E::from(rng.gen_bool(0.5) as u64))
            .collect_vec()
    }

    impl<E: ExtensionField> Model<E> {
        /// Returns a random model with specified number of dense layers and a matching input.
        pub fn random(num_dense_layers: usize) -> (Self, Vec<E>) {
            let mut model = Model::new();
            let mut rng = thread_rng();
            let mut last_row = rng.gen_range(3..15);
            for _ in 0..num_dense_layers {
                // last row becomes new column
                let (nrows, ncols) = (rng.gen_range(3..15), last_row);
                last_row = nrows;
                let mat = Matrix::<E>::random((nrows, ncols)).pad_next_power_of_two();
                model.add_layer(Layer::Dense(mat));
            }
            let input_dims = model.layers.first().unwrap().dim();
            // ncols since matrix2vector is summing over the columns
            let input = random_vector(input_dims.1);
            (model, input)
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::<F>::random(15);
        model.run(input);
    }

    #[test]
    fn test_model_run() {
        let mat1 = Matrix::<F>::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::<F>::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = random_vector(mat1.ncols());
        let output1 = mat1.matmul(&input);
        let final_output = mat2.matmul(&output1);

        let mut model = Model::<F>::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2.clone()));

        let trace = model.run(input.clone());
        assert_eq!(trace.steps.len(), 2);

        // Verify first step
        assert_eq!(trace.steps[0].output, output1);

        // Verify second step
        assert_eq!(trace.steps[1].output, final_output);
        let (nrow, _) = (mat2.nrows(), mat2.ncols());
        assert_eq!(final_output.len(), nrow);
    }

    #[test]
    fn test_inference_trace_iterator() {
        let mat1 = Matrix::<F>::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::<F>::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = random_vector(mat1.ncols());

        let mut model = Model::<F>::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run(input.clone());

        // Verify iterator yields correct input/output pairs
        let mut iter = trace.iter();

        // First step should have original input
        let (first_input, first_step) = iter.next().unwrap();
        assert_eq!(first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Second step should have first step's output as input
        let (second_input, second_step) = iter.next().unwrap();
        assert_eq!(second_input, trace.steps[0].output);
        assert_eq!(second_step.output, trace.steps[1].output);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_inference_trace_reverse_iterator() {
        let mat1 = Matrix::<F>::random((10, 11)).pad_next_power_of_two();
        let mat2 = Matrix::<F>::random((7, mat1.ncols())).pad_next_power_of_two();
        let input = random_vector(mat1.ncols());

        let mut model = Model::<F>::new();
        model.add_layer(Layer::Dense(mat1));
        model.add_layer(Layer::Dense(mat2));

        let trace = model.run(input.clone());

        // Test reverse iteration
        let mut rev_iter = trace.iter().rev();

        // Last step should come first in reverse
        let (last_input, last_step) = rev_iter.next().unwrap();
        assert_eq!(last_input, trace.steps[0].output);
        assert_eq!(last_step.output, trace.steps[1].output);

        // First step should come last in reverse
        let (first_input, first_step) = rev_iter.next().unwrap();
        assert_eq!(first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Iterator should be exhausted
        assert!(rev_iter.next().is_none());
    }
}
