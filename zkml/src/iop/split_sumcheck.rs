//! This module provides and adapted version of the sumcheck protocol described in https://eprint.iacr.org/2024/1210.pdf
//! we use this to prove linear operators of the the form
//!     L * M * R
//! with L, M and R all matrices.

use ff_ext::ExtensionField;
use rayon::prelude::*;
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};
use sumcheck::{
    structs::{IOPProof, IOPProverMessage},
    util::AdditiveArray,
};
use transcript::{Challenge, Transcript};

/// Prover State of a PolyIOP.
#[derive(Default)]
pub struct IOPSplitProverState<E: ExtensionField> {
    /// The middle matirces evaluations on the full boolean hypercube for this proof
    pub(crate) middle: Vec<E>,
    /// The left matrices evaluations on its unfixed variables
    pub(crate) left: Vec<E>,
    /// The right matrices evaluations on its unfixed variables
    pub(crate) right: Vec<E>,
    /// The number of unfixed variables the left matrix has
    pub(crate) left_vars: usize,
    /// The number of unfixed variables the right matrix has
    pub(crate) right_vars: usize,
}

#[derive(Debug, Clone)]
pub enum SplitSumcheckError {
    ParameterError(String),
    ProvingError(String),
}

impl Display for SplitSumcheckError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            SplitSumcheckError::ParameterError(s) => {
                write!(f, "Split sumcheck parameter error: {}", s)
            }
            Self::ProvingError(s) => {
                write!(f, "Split sumcheck proving error: {}", s)
            }
        }
    }
}

impl Error for SplitSumcheckError {}

impl<E: ExtensionField> IOPSplitProverState<E> {
    pub fn prove_split_sumcheck<T: Transcript<E>>(
        left: Vec<E>,
        middle: Vec<E>,
        right: Vec<E>,
        transcript: &mut T,
    ) -> Result<(IOPProof<E>, Self), SplitSumcheckError> {
        // Initialise the split sumcheck prover state
        let mut prover_state = Self::initialise_prover(left, middle, right)?;

        let num_variables = prover_state.left_vars + prover_state.right_vars;

        let mut challenge = None;
        let mut challenges = Vec::<E>::with_capacity(num_variables);
        let mut prover_msgs = Vec::with_capacity(num_variables);

        transcript.append_message(&num_variables.to_le_bytes());
        // Although there are three polynomials because of the nature of the split sumcheck we can treat all the round polynomials as degree 2
        transcript.append_message(&2usize.to_le_bytes());

        // First we perform the rounds that only effect `middle` and `right`
        let right_rounds = prover_state.right_vars;
        for _ in 0..right_rounds {
            let prover_msg = prover_state.prove_right_round_and_update(&challenge)?;

            prover_msg
                .evaluations()
                .iter()
                .for_each(|e| transcript.append_field_element_ext(e));

            prover_msgs.push(prover_msg);

            let c = transcript.get_and_append_challenge(b"Internal round");
            challenges.push(c.elements);
            challenge = Some(c);
        }

        // Now we perform the left rounds
        let left_rounds = prover_state.left_vars;
        for j in 0..left_rounds {
            let prover_msg = prover_state.prove_left_round_and_update(&challenge, j == 0)?;

            prover_msg
                .evaluations()
                .iter()
                .for_each(|e| transcript.append_field_element_ext(e));

            prover_msgs.push(prover_msg);

            let c = transcript.get_and_append_challenge(b"Internal round");
            challenges.push(c.elements);
            challenge = Some(c);
        }

        // Update `left` and `middle` one last time (after checking they have length 2)
        if let Some(challenge) = challenge {
            let challenge_elem = challenge.elements;

            if prover_state.middle.len() != 2 {
                return Err(SplitSumcheckError::ProvingError(format!(
                    "After all rounds `middle` poly should have length 2, it has length {}",
                    prover_state.middle.len()
                )));
            }
            prover_state.middle = vec![
                challenge_elem * (prover_state.middle[1] - prover_state.middle[0])
                    + prover_state.middle[0],
            ];

            if prover_state.left.len() != 2 {
                return Err(SplitSumcheckError::ProvingError(format!(
                    "After all rounds `left` poly should have length 2, it has length {}",
                    prover_state.left.len()
                )));
            }
            prover_state.left = vec![
                challenge_elem * (prover_state.left[1] - prover_state.left[0])
                    + prover_state.left[0],
            ];
        }
        // We have to reverse the order of challenges
        challenges.reverse();
        Ok((IOPProof::new(challenges, prover_msgs), prover_state))
    }

    /// Initialises the prover state fromt he supplied evaluations.
    fn initialise_prover(
        left: Vec<E>,
        middle: Vec<E>,
        right: Vec<E>,
    ) -> Result<Self, SplitSumcheckError> {
        let left_len = left.len();
        let right_len = right.len();
        let mid_len = middle.len();

        // mid_len needs to be equal to left_len * right_len
        if mid_len != left_len * right_len {
            return Err(SplitSumcheckError::ParameterError(format!(
                "Middle matrix should have {} * {} total evaluations, supplied one has {}",
                left_len, right_len, mid_len
            )));
        }
        // If any of the evaluations don't have power of 2 length we error
        for (len, name) in [
            (mid_len, "middle"),
            (left_len, "left"),
            (right_len, "right"),
        ] {
            if !len.is_power_of_two() {
                return Err(SplitSumcheckError::ParameterError(format!(
                    "The evaluations for the {} matrix were not a power of two, got {}",
                    name, len
                )));
            }
        }

        Ok(IOPSplitProverState {
            middle,
            left,
            right,
            left_vars: left_len.ilog2() as usize,
            right_vars: right_len.ilog2() as usize,
        })
    }

    /// Performs the sumcheck provers work in a round where only `middle` and `right` polynomials are involved.
    pub(crate) fn prove_right_round_and_update(
        &mut self,
        challenge: &Option<Challenge<E>>,
    ) -> Result<IOPProverMessage<E>, SplitSumcheckError> {
        // Update the evaluations of `middle` and `right`
        if let Some(challenge) = challenge {
            let challenge_elem = challenge.elements;

            // The midpoint of `right` and `middle`
            let right_half_size = self.right.len() >> 1;
            let middle_half_size = self.middle.len() >> 1;

            // Fix the high variable for `middle`
            let (lo, hi) = self.middle.split_at(middle_half_size);
            self.middle = lo
                .par_iter()
                .zip(hi)
                .with_min_len(64)
                .map(|(lo, hi)| challenge_elem * (*hi - *lo) + *lo)
                .collect::<Vec<E>>();

            // Fix the high variable for `right`
            let (lo, hi) = self.right.split_at(right_half_size);
            self.right = lo
                .par_iter()
                .zip(hi)
                .with_min_len(64)
                .map(|(lo, hi)| challenge_elem * (*hi - *lo) + *lo)
                .collect::<Vec<E>>();
        }

        // We need to iterate over half of `right`
        let right_half_size = self.right.len() >> 1;
        let field_two = E::from(2);

        // Make the S_{i}(x, j) polynomials for this round
        let (right_low, right_high) = self.right.split_at(right_half_size);
        let round_evals: AdditiveArray<E, 3> = right_low
            .par_iter()
            .zip(right_high.par_iter())
            .enumerate()
            .fold(
                AdditiveArray::<E, 3>::default,
                |acc, (x1, (r_low, r_high))| {
                    let low_index = x1 << self.left_vars;
                    let high_index = (right_half_size + x1) << self.left_vars;

                    // We have to sum over all the low variables `x2`
                    let AdditiveArray([zero_eval, one_eval, two_eval]) = self
                        .left
                        .iter()
                        .enumerate()
                        .fold(AdditiveArray::<E, 3>::default(), |acc, (x2, &left_eval)| {
                            let zero_eval = left_eval * self.middle[low_index + x2];
                            let one_eval = left_eval * self.middle[high_index + x2];
                            let two_eval = field_two * one_eval - zero_eval;
                            acc + AdditiveArray([zero_eval, one_eval, two_eval])
                        });

                    let right_zero_eval = *r_low;
                    let right_one_eval = *r_high;
                    let right_two_eval = field_two * right_one_eval - right_zero_eval;

                    acc + AdditiveArray([
                        zero_eval * right_zero_eval,
                        one_eval * right_one_eval,
                        two_eval * right_two_eval,
                    ])
                },
            )
            .reduce(AdditiveArray::<E, 3>::default, |a, b| a + b);

        Ok(IOPProverMessage::<E>::new(round_evals.0.to_vec()))
    }

    /// Performs the sumcheck provers work in a round where only `middle` and `left` polynomials are involved.
    pub(crate) fn prove_left_round_and_update(
        &mut self,
        challenge: &Option<Challenge<E>>,
        is_first_round: bool,
    ) -> Result<IOPProverMessage<E>, SplitSumcheckError> {
        // Update the evaluations of `middle` and `left`
        if let Some(challenge) = challenge {
            let challenge_elem = challenge.elements;

            // The midpoint of `right` and `middle`
            let left_half_size = self.left.len() >> 1;
            let middle_half_size = self.middle.len() >> 1;

            // Fix the high variable for `middle`
            let (lo, hi) = self.middle.split_at(middle_half_size);
            self.middle = lo
                .par_iter()
                .zip(hi)
                .with_min_len(64)
                .map(|(lo, hi)| challenge_elem * (*hi - *lo) + *lo)
                .collect::<Vec<E>>();

            // The first round involving `left` we actually want to fold the `right` for a final time
            if !is_first_round {
                // Fix the high variable for `left`
                let (lo, hi) = self.left.split_at(left_half_size);
                self.left = lo
                    .par_iter()
                    .zip(hi)
                    .with_min_len(64)
                    .map(|(lo, hi)| challenge_elem * (*hi - *lo) + *lo)
                    .collect::<Vec<E>>();
            } else {
                // The length of `right` in the first round involving `left` should be 2 so we chec that here
                if self.right.len() != 2 {
                    return Err(SplitSumcheckError::ProvingError(format!(
                        "In the first left poly round right should have length 2, it has length {}",
                        self.right.len()
                    )));
                }
                self.right = vec![challenge_elem * (self.right[1] - self.right[0]) + self.right[0]];
            }
        }

        // We need to iterate over half of `right`
        let left_half_size = self.left.len() >> 1;
        let field_two = E::from(2);

        // Make the S_{i}(x, j) polynomials for this round
        let (left_low, left_high) = self.left.split_at(left_half_size);
        let (middle_low, middle_high) = self.middle.split_at(left_half_size);
        let AdditiveArray([zero_eval, one_eval, two_eval]) = left_low
            .par_iter()
            .zip(left_high.par_iter())
            .zip(middle_low.par_iter())
            .zip(middle_high.par_iter())
            .fold(
                AdditiveArray::<E, 3>::default,
                |acc, (((left_low, left_high), mid_low), mid_high)| {
                    let left_zero_eval = *left_low;
                    let left_one_eval = *left_high;
                    let left_two_eval = field_two * left_one_eval - left_zero_eval;

                    let middle_zero_eval = *mid_low;
                    let middle_one_eval = *mid_high;
                    let middle_two_eval = field_two * middle_one_eval - middle_zero_eval;

                    acc + AdditiveArray([
                        middle_zero_eval * left_zero_eval,
                        middle_one_eval * left_one_eval,
                        middle_two_eval * left_two_eval,
                    ])
                },
            )
            .reduce(AdditiveArray::<E, 3>::default, |a, b| a + b);

        let right_value = self.right[0];
        Ok(IOPProverMessage::<E>::new(vec![
            zero_eval * right_value,
            one_eval * right_value,
            two_eval * right_value,
        ]))
    }
}

#[cfg(test)]
mod tests {
    use crate::default_transcript;
    use ark_std::rand::{Rng, thread_rng};

    use goldilocks::GoldilocksExt2;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VPAuxInfo,
    };
    use sumcheck::structs::IOPVerifierState;

    use super::*;

    type F = GoldilocksExt2;

    /// Generates three random vectors of evaulations, output is ordered (`left`, `middle`, `right`)
    fn generate_split_sumcheck_inputs<E: ExtensionField>(
        num_vars: usize,
    ) -> (Vec<E>, Vec<E>, Vec<E>) {
        let mut rng = thread_rng();
        // need num_vars to be at least 4
        let (left_vars, middle_vars, right_vars) = if num_vars < 4 {
            (2usize, 4, 2)
        } else {
            let left_vars = rng.gen_range(2..num_vars - 1);
            let right_vars = num_vars - left_vars;
            (left_vars, num_vars, right_vars)
        };

        let left = (0..1 << left_vars)
            .map(|_| E::random(&mut rng))
            .collect::<Vec<E>>();
        let middle = (0..1 << middle_vars)
            .map(|_| E::random(&mut rng))
            .collect::<Vec<E>>();
        let right = (0..1 << right_vars)
            .map(|_| E::random(&mut rng))
            .collect::<Vec<E>>();

        (left, middle, right)
    }

    /// Calculates the expected sum that is input to the verifier for the sumcheck
    fn expected_sum<E: ExtensionField>(left: &[E], middle: &[E], right: &[E]) -> E {
        let middle_len = middle.len();
        let left_len = left.len();
        let left_vars = left_len.ilog2() as usize;

        (0..middle_len).fold(E::ZERO, |acc, i| {
            let left_index = i % left_len;
            let right_index = i >> left_vars;

            acc + (left[left_index] * middle[i] * right[right_index])
        })
    }

    #[test]
    fn test_split_sumcheck() -> Result<(), SplitSumcheckError> {
        for num_vars in 4..15 {
            println!("TEST SPLIT SUMCHECK WITH {} VARIABLES", num_vars);
            let (left, middle, right) = generate_split_sumcheck_inputs::<F>(num_vars);
            let left_vars = left.len().ilog2() as usize;
            let middle_vars = middle.len().ilog2() as usize;
            let right_vars = right.len().ilog2() as usize;
            println!(
                "LEFT VARS: {}, MIDDLE VARS: {}, RIGHT VARS: {}",
                left_vars, middle_vars, right_vars
            );
            let mut prover_transcript = default_transcript::<F>();

            let expected_sum = expected_sum(&left, &middle, &right);

            let (proof, state) = IOPSplitProverState::<F>::prove_split_sumcheck(
                left.clone(),
                middle.clone(),
                right.clone(),
                &mut prover_transcript,
            )?;

            // Now we check that the evals output by state correspond to the correct evaluations
            let left_eval = state.left[0];
            let middle_eval = state.middle[0];
            let right_eval = state.right[0];

            let expected_left = left.into_mle().evaluate(&proof.point[..left_vars]);
            let expected_middle = middle.into_mle().evaluate(&proof.point);
            let expected_right = right.into_mle().evaluate(&proof.point[left_vars..]);

            assert_eq!(left_eval, expected_left, "left evals not correct");
            assert_eq!(middle_eval, expected_middle, "middle evals not correct");
            assert_eq!(right_eval, expected_right, "right evals not correct");

            let mut verifier_transcript = default_transcript::<F>();
            let aux_info = VPAuxInfo::<F>::from_mle_list_dimensions(&[vec![num_vars, num_vars]]);
            let _ = IOPVerifierState::<F>::verify(
                expected_sum,
                &proof,
                &aux_info,
                &mut verifier_transcript,
            );
        }

        Ok(())
    }
}
