use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::DenseMultilinearExtension,
    virtual_poly::{ArcMultilinearExtension, VirtualPolynomial, build_eq_x_r_vec_sequential},
};
use p3_field::FieldAlgebra;
use simple_frontend::structs::LayerId;
use std::sync::Arc;
use sumcheck::{macros::entered_span, util::ceil_log2};

use crate::{
    exit_span,
    structs::{
        Circuit, CircuitWitness, IOPProverState, IOPProverStepMessage, PointAndEval, SumcheckProof,
    },
    utils::{MatrixMLERowFirst, tensor_product},
};

// Prove the items copied from the current layer to later layers for data parallel circuits.
impl<E: ExtensionField> IOPProverState<E> {
    /// Sumcheck 1: sigma = \sum_{t || y}(f1({t || y}) * (\sum_j g1^{(j)}({t || y})))
    ///     sigma = \sum_j( \alpha^j * subset[i][j](rt_j || ry_j) )
    ///     f1^{(j)}(y) = layers[i](t || y)
    ///     g1^{(j)}(y) = \alpha^j * eq(rt_j, t) * eq(ry_j, y)
    ///     g1^{(j)}(y) = \alpha^j * eq(rt_j, t) * copy_to[j](ry_j, y)
    #[tracing::instrument(skip_all, name = "build_phase1_step1_sumcheck_poly")]
    pub(super) fn build_phase1_step1_sumcheck_poly<'a>(
        &self,
        layer_id: LayerId,
        alpha: E,
        eq_t: &Vec<Vec<E>>,
        circuit: &Circuit<E>,
        circuit_witness: &'a CircuitWitness<E>,
        multi_threads_meta: (usize, usize),
    ) -> VirtualPolynomial<'a, E> {
        let span = entered_span!("preparation");
        let timer = start_timer!(|| "Prover sumcheck phase 1 step 1");

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
        let hi_num_vars = circuit_witness.instance_num_vars();

        // parallel unit logic handling
        let (thread_id, max_thread_id) = multi_threads_meta;
        let log2_max_thread_id = ceil_log2(max_thread_id);

        exit_span!(span);

        // f1^{(j)}(y) = layers[i](t || y)
        let f1: ArcMultilinearExtension<E> = Arc::new(
            circuit_witness.layers_ref()[layer_id as usize]
                .get_ranged_mle(multi_threads_meta.1, multi_threads_meta.0),
        );

        assert_eq!(
            f1.num_vars(),
            hi_num_vars + lo_num_vars - log2_max_thread_id
        );

        let span = entered_span!("g1");
        // g1^{(j)}(y) = \alpha^j * eq(rt_j, t) * eq(ry_j, y)
        // g1^{(j)}(y) = \alpha^j * eq(rt_j, t) * copy_to[j](ry_j, y)
        let copy_to_matrices = &circuit.layers[self.layer_id as usize].copy_to;
        let g1: ArcMultilinearExtension<'a, E> = {
            let gs = izip!(&self.to_next_phase_point_and_evals, &alpha_pows, eq_t)
                .map(|(point_and_eval, alpha_pow, eq_t)| {
                    // g1^{(j)}(y) = \alpha^j * eq(rt_j, t) * eq(ry_j, y)
                    let point_lo_num_vars = point_and_eval.point.len() - hi_num_vars;

                    let eq_y =
                        build_eq_x_r_vec_sequential(&point_and_eval.point[..point_lo_num_vars])
                            .into_iter()
                            .take(1 << lo_num_vars)
                            .map(|eq| *alpha_pow * eq)
                            .collect_vec();

                    let eq_t_unit_len = eq_t.len() / max_thread_id;
                    let start_index = thread_id * eq_t_unit_len;
                    let g1_j =
                        tensor_product(&eq_t[start_index..(start_index + eq_t_unit_len)], &eq_y);

                    assert_eq!(
                        g1_j.len(),
                        (1 << (hi_num_vars + lo_num_vars - log2_max_thread_id))
                    );

                    g1_j
                })
                .chain(
                    izip!(
                        &self.subset_point_and_evals[self.layer_id as usize],
                        &alpha_pows[self.to_next_phase_point_and_evals.len()..],
                        eq_t.iter().skip(self.to_next_phase_point_and_evals.len())
                    )
                    .map(
                        |((new_layer_id, point_and_eval), alpha_pow, eq_t)| {
                            let point_lo_num_vars = point_and_eval.point.len() - hi_num_vars;
                            let copy_to = &copy_to_matrices[new_layer_id];
                            let lo_eq_w_p = build_eq_x_r_vec_sequential(
                                &point_and_eval.point[..point_lo_num_vars],
                            );

                            // g2^{(j)}(y) = \alpha^j * eq(rt_j, t) * copy_to[j](ry_j, y)
                            let eq_t_unit_len = eq_t.len() / max_thread_id;
                            let start_index = thread_id * eq_t_unit_len;
                            let g2_j = tensor_product(
                                &eq_t[start_index..(start_index + eq_t_unit_len)],
                                &copy_to.as_slice().fix_row_row_first_with_scalar(
                                    &lo_eq_w_p,
                                    lo_num_vars,
                                    alpha_pow,
                                ),
                            );

                            assert_eq!(
                                g2_j.len(),
                                (1 << (hi_num_vars + lo_num_vars - log2_max_thread_id))
                            );
                            g2_j
                        },
                    ),
                )
                .collect::<Vec<Vec<_>>>();

            DenseMultilinearExtension::from_evaluations_ext_vec(
                hi_num_vars + lo_num_vars - log2_max_thread_id,
                gs.into_iter()
                    .fold(vec![E::ZERO; 1 << f1.num_vars()], |mut acc, g| {
                        assert_eq!(1 << f1.num_vars(), g.len());
                        acc.iter_mut().enumerate().for_each(|(i, v)| *v += g[i]);
                        acc
                    }),
            )
            .into()
        };
        exit_span!(span);

        // sumcheck: sigma = \sum_{s || y}(f1({s || y}) * (\sum_j g1^{(j)}({s || y})))
        let span = entered_span!("virtual_poly");
        let mut virtual_poly_1: VirtualPolynomial<E> = VirtualPolynomial::new_from_mle(f1, E::ONE);
        virtual_poly_1.mul_by_mle(g1, E::BaseField::ONE);
        exit_span!(span);
        end_timer!(timer);

        virtual_poly_1
    }

    pub(super) fn combine_phase1_step1_evals(
        &mut self,
        sumcheck_proof_1: SumcheckProof<E>,
        prover_state: sumcheck::structs::IOPProverState<E>,
    ) -> IOPProverStepMessage<E> {
        let (mut f1, _): (Vec<_>, Vec<_>) = prover_state
            .get_mle_final_evaluations()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let eval_value_1 = f1.remove(0).1;

        self.to_next_step_point.clone_from(&sumcheck_proof_1.point);
        self.to_next_phase_point_and_evals = vec![PointAndEval::new_from_ref(
            &self.to_next_step_point,
            &eval_value_1,
        )];
        self.subset_point_and_evals[self.layer_id as usize].clear();

        IOPProverStepMessage {
            sumcheck_proof: sumcheck_proof_1,
            sumcheck_eval_values: vec![eval_value_1],
        }
    }
}
