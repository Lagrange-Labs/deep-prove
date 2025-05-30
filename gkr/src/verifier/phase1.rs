use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::{Itertools, chain, izip};
use multilinear_extensions::virtual_poly::{VPAuxInfo, build_eq_x_r_vec, eq_eval};
use std::marker::PhantomData;
use transcript::Transcript;

use crate::{
    error::GKRError,
    structs::{Circuit, IOPProverStepMessage, IOPVerifierState, PointAndEval},
    utils::MatrixMLERowFirst,
};

use super::SumcheckState;

impl<E: ExtensionField> IOPVerifierState<E> {
    pub(super) fn verify_and_update_state_phase1_step1(
        &mut self,
        circuit: &Circuit<E>,
        step_msg: IOPProverStepMessage<E>,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(), GKRError> {
        let timer = start_timer!(|| "Verifier sumcheck phase 1 step 1");
        let alpha = transcript
            .get_and_append_challenge(b"combine subset evals")
            .elements;
        let total_length = self.to_next_phase_point_and_evals.len()
            + self.subset_point_and_evals[self.layer_id as usize].len()
            + 1;
        let alpha_pows = {
            let mut alpha_pows = vec![E::ONE; total_length];
            for i in 0..total_length.saturating_sub(1) {
                alpha_pows[i + 1] = alpha_pows[i] * alpha;
            }
            alpha_pows
        };

        let lo_num_vars = circuit.layers[self.layer_id as usize].num_vars;
        let hi_num_vars = self.instance_num_vars;

        // sigma = \sum_j( \alpha^j * subset[i][j](rt_j || ry_j) )
        let mut sigma_1 = izip!(self.to_next_phase_point_and_evals.iter(), alpha_pows.iter())
            .fold(E::ZERO, |acc, (point_and_eval, alpha_pow)| {
                acc + point_and_eval.eval * *alpha_pow
            });
        sigma_1 += izip!(
            self.subset_point_and_evals[self.layer_id as usize].iter(),
            alpha_pows
                .iter()
                .skip(self.to_next_phase_point_and_evals.len())
        )
        .fold(E::ZERO, |acc, ((_, point_and_eval), alpha_pow)| {
            acc + point_and_eval.eval * *alpha_pow
        });

        // Sumcheck: sigma = \sum_{t || y}(f1({t || y}) * (\sum_j g1^{(j)}({t || y})))
        // f1^{(j)}(y) = layers[i](t || y)
        // g1^{(j)}(t || y) = \alpha^j * eq(rt_j, t) * eq(ry_j, y)
        // g1^{(j)}(t || y) = \alpha^j * eq(rt_j, t) * copy_to[j](ry_j, y)
        let claim_1 = SumcheckState::verify(
            sigma_1,
            &step_msg.sumcheck_proof,
            &VPAuxInfo {
                max_degree: 2,
                max_num_variables: lo_num_vars + hi_num_vars,
                phantom: PhantomData,
            },
            transcript,
        );

        let claim1_point = claim_1.point.iter().map(|x| x.elements).collect_vec();
        let claim1_point_lo_num_vars = claim1_point.len() - hi_num_vars;
        let eq_y_ry = build_eq_x_r_vec(&claim1_point[..claim1_point_lo_num_vars]);

        assert_eq!(step_msg.sumcheck_eval_values.len(), 1);
        let f_value = step_msg.sumcheck_eval_values[0];

        let g_value: E = chain![
            izip!(self.to_next_phase_point_and_evals.iter(), alpha_pows.iter()).map(
                |(point_and_eval, alpha_pow)| {
                    let point_lo_num_vars = point_and_eval.point.len() - hi_num_vars;
                    let eq_t = eq_eval(
                        &point_and_eval.point[point_lo_num_vars..],
                        &claim1_point[(claim1_point.len() - hi_num_vars)..],
                    );
                    let eq_y = eq_eval(
                        &point_and_eval.point[..point_lo_num_vars],
                        &claim1_point[..point_lo_num_vars],
                    );
                    eq_t * eq_y * *alpha_pow
                }
            ),
            izip!(
                self.subset_point_and_evals[self.layer_id as usize].iter(),
                alpha_pows
                    .iter()
                    .skip(self.to_next_phase_point_and_evals.len())
            )
            .map(|((new_layer_id, point_and_eval), alpha_pow)| {
                let point_lo_num_vars = point_and_eval.point.len() - hi_num_vars;
                let eq_t = eq_eval(
                    &point_and_eval.point[point_lo_num_vars..],
                    &claim1_point[(claim1_point.len() - hi_num_vars)..],
                );
                let eq_yj_ryj = build_eq_x_r_vec(&point_and_eval.point[..point_lo_num_vars]);
                eq_t * circuit.layers[self.layer_id as usize].copy_to[new_layer_id]
                    .as_slice()
                    .eval_row_first(&eq_yj_ryj, &eq_y_ry)
                    * *alpha_pow
            }),
        ]
        .sum();

        let got_value = f_value * g_value;

        end_timer!(timer);
        if claim_1.expected_evaluation != got_value {
            return Err(GKRError::VerifyError("phase1 step1 failed"));
        }
        self.to_next_step_point_and_eval = PointAndEval::new_from_ref(&claim1_point, &f_value);
        self.to_next_phase_point_and_evals = vec![self.to_next_step_point_and_eval.clone()];

        self.subset_point_and_evals[self.layer_id as usize].clear();

        Ok(())
    }
}
