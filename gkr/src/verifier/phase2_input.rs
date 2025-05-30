use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::{Itertools, chain, izip};
use multilinear_extensions::virtual_poly::{VPAuxInfo, build_eq_x_r_vec};
use std::mem;
use sumcheck::util::ceil_log2;
use transcript::Transcript;

use crate::{
    circuit::EvaluateGateCIn,
    error::GKRError,
    structs::{Circuit, IOPProverStepMessage, IOPVerifierState, PointAndEval},
    utils::{counter_eval, i64_to_field, segment_eval_greater_than},
};

use super::SumcheckState;

impl<E: ExtensionField> IOPVerifierState<E> {
    pub(super) fn verify_and_update_state_input_phase2_step1(
        &mut self,
        circuit: &Circuit<E>,
        step_msg: IOPProverStepMessage<E>,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(), GKRError> {
        let timer = start_timer!(|| "Verifier sumcheck phase 2 step 1");
        let layer = &circuit.layers[self.layer_id as usize];
        let lo_out_num_vars = layer.num_vars;
        let lo_in_num_vars = circuit.max_wit_in_num_vars;
        let hi_num_vars = self.instance_num_vars;

        self.out_point = mem::take(&mut self.to_next_step_point_and_eval.point);
        let lo_point = &self.out_point[..lo_out_num_vars];
        let hi_point = &self.out_point[lo_out_num_vars..];

        self.eq_y_ry = build_eq_x_r_vec(lo_point);

        let g_value_const = circuit
            .paste_from_consts_in
            .iter()
            .map(|(c, (l, r))| {
                let c = i64_to_field::<E::BaseField>(*c);
                let segment_greater_than_l_1 = if *l == 0 {
                    E::ONE
                } else {
                    segment_eval_greater_than(l - 1, lo_point)
                };
                let segment_greater_than_r_1 = segment_eval_greater_than(r - 1, lo_point);
                (segment_greater_than_l_1 - segment_greater_than_r_1) * c
            })
            .sum::<E>();

        let mut sumcheck_sigma = self.to_next_step_point_and_eval.eval - g_value_const;
        if !layer.add_consts.is_empty() {
            sumcheck_sigma -= layer
                .add_consts
                .as_slice()
                .eval(&self.eq_y_ry, &self.challenges);
        }

        if lo_in_num_vars.is_none() {
            if sumcheck_sigma != E::ZERO {
                return Err(GKRError::VerifyError("input phase2 step1 failed"));
            }
            return Ok(());
        }

        let lo_in_num_vars = lo_in_num_vars.unwrap();

        let claim = SumcheckState::verify(
            sumcheck_sigma,
            &step_msg.sumcheck_proof,
            &VPAuxInfo {
                max_degree: 2,
                max_num_variables: lo_in_num_vars,
                phantom: std::marker::PhantomData,
            },
            transcript,
        );

        let claim_point = claim.point.iter().map(|x| x.elements).collect_vec();

        self.eq_x1_rx1 = build_eq_x_r_vec(&claim_point);
        let g_values_iter = chain![
            circuit.paste_from_wits_in.iter().cloned(),
            circuit
                .paste_from_counter_in
                .iter()
                .map(|(_, (l, r))| (*l, *r))
        ]
        .map(|(l, r)| {
            (l..r)
                .map(|i| self.eq_y_ry[i] * self.eq_x1_rx1[i - l])
                .sum::<E>()
        });

        // TODO: Double check here.
        let f_counter_values = circuit
            .paste_from_counter_in
            .iter()
            .map(|(num_vars, _)| {
                let point = [&claim_point[..*num_vars], hi_point].concat();
                counter_eval(num_vars + hi_num_vars, &point)
                    * claim_point[*num_vars..]
                        .iter()
                        .map(|x| E::ONE - *x)
                        .product::<E>()
            })
            .collect_vec();
        let got_value = izip!(
            chain![
                step_msg.sumcheck_eval_values.iter(),
                f_counter_values.iter()
            ],
            g_values_iter
        )
        .map(|(f, g)| *f * g)
        .sum::<E>();

        self.to_next_phase_point_and_evals = izip!(
            circuit.paste_from_wits_in.iter(),
            step_msg.sumcheck_eval_values.into_iter()
        )
        .map(|((l, r), eval)| {
            let num_vars = ceil_log2(*r - *l);
            let point = [&claim_point[..num_vars], hi_point].concat();
            let wit_in_eval = eval
                * claim_point[num_vars..]
                    .iter()
                    .map(|x| E::ONE - *x)
                    .product::<E>()
                    .inverse();
            PointAndEval::new_from_ref(&point, &wit_in_eval)
        })
        .collect_vec();
        self.to_next_step_point_and_eval =
            PointAndEval::new([&claim_point, hi_point].concat(), E::ZERO);

        end_timer!(timer);
        if claim.expected_evaluation != got_value {
            return Err(GKRError::VerifyError("input phase2 step1 failed"));
        }

        Ok(())
    }
}
