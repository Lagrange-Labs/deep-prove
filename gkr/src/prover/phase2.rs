use ark_std::{end_timer, iterable::Iterable, start_timer};
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
};
use p3_field::FieldAlgebra;
use simple_frontend::structs::LayerId;
use std::sync::Arc;

use sumcheck::{
    macros::{entered_span, exit_span},
    util::ceil_log2,
};

use crate::structs::{
    CircuitWitness, IOPProverState,
    Step::{Step1, Step2, Step3},
};
use multilinear_extensions::mle::MultilinearExtension;

use crate::{
    circuit::EvaluateConstant,
    structs::{Circuit, IOPProverStepMessage, PointAndEval, SumcheckProof},
};

macro_rules! prepare_stepx_g_fn {
    (&mut $a1:ident, $s_in:ident, $s_out:ident, $d:ident $(,$c:ident, |$f_s_in:ident, $f_s_out:ident, $g:ident| $op:expr)* $(,)?) => {
        $a1.chunks_mut(1 << $s_in)
            // enumerated index is the instance index
            .fold([$d << $s_in, $d << $s_out], |mut s_acc, evals_vec| {
                // prefix s with global thread id $d
                let (s_in, s_out) = (&s_acc[0], &s_acc[1]);
                $(
                    $c.iter().for_each(|(fanin_cellid, gates)| {
                        let eval = gates.iter().map(|$g| {
                            let $f_s_in = s_in;
                            let $f_s_out = s_out;
                            $op
                        }).fold(E::ZERO, |acc, item| acc + item);
                        evals_vec[*fanin_cellid] += eval;
                    });
                )*
                s_acc[0] += (1 << $s_in);
                s_acc[1] += (1 << $s_out);
                s_acc
            });
    };
}

// Prove the computation in the current layer for data parallel circuits.
// The number of terms depends on the gate.
// Here is an example of degree 3:
// layers[i](rt || ry) = \sum_{s1}( \sum_{s2}( \sum_{s3}( \sum_{x1}( \sum_{x2}( \sum_{x3}(
//     eq(rt, s1, s2, s3) * mul3(ry, x1, x2, x3) * layers[i + 1](s1 || x1) * layers[i + 1](s2 || x2) * layers[i + 1](s3 || x3)
// ) ) ) ) ) ) + sum_s1( sum_s2( sum_{x1}( sum_{x2}(
//     eq(rt, s1, s2) * mul2(ry, x1, x2) * layers[i + 1](s1 || x1) * layers[i + 1](s2 || x2)
// ) ) ) ) + \sum_{s1}( \sum_{x1}(
//     eq(rt, s1) * add(ry, x1) * layers[i + 1](s1 || x1)
// ) ) + \sum_{s1}( \sum_{x1}(
//      \sum_j eq(rt, s1) paste_from[j](ry, x1) * subset[j][i](s1 || x1)
// ) ) + add_const(ry)
impl<E: ExtensionField> IOPProverState<E> {
    /// Sumcheck 1: sigma = \sum_{s1 || x1} f1(s1 || x1) * g1(s1 || x1) + \sum_j f1'_j(s1 || x1) * g1'_j(s1 || x1)
    ///     sigma = layers[i](rt || ry) - add_const(ry),
    ///     f1(s1 || x1) = layers[i + 1](s1 || x1)
    ///     g1(s1 || x1) = \sum_{s2}( \sum_{s3}( \sum_{x2}( \sum_{x3}(
    ///         eq(rt, s1, s2, s3) * mul3(ry, x1, x2, x3) * layers[i + 1](s2 || x2) * layers[i + 1](s3 || x3)
    ///     ) ) ) ) + \sum_{s2}( \sum_{x2}(
    ///         eq(rt, s1, s2) * mul2(ry, x1, x2) * layers[i + 1](s2 || x2)
    ///     ) ) + eq(rt, s1) * add(ry, x1)
    ///     f1'^{(j)}(s1 || x1) = subset[j][i](s1 || x1)
    ///     g1'^{(j)}(s1 || x1) = eq(rt, s1) paste_from[j](ry, x1)
    #[tracing::instrument(skip_all, name = "build_phase2_step1_sumcheck_poly")]
    pub(super) fn build_phase2_step1_sumcheck_poly<'a>(
        eq: &[Vec<E>; 1],
        layer_id: LayerId,
        circuit: &Circuit<E>,
        circuit_witness: &'a CircuitWitness<E>,
        multi_threads_meta: (usize, usize),
    ) -> VirtualPolynomial<'a, E> {
        let timer = start_timer!(|| "Prover sumcheck phase 2 step 1");
        let layer = &circuit.layers[layer_id as usize];
        let lo_out_num_vars = layer.num_vars;
        let lo_in_num_vars = layer.max_previous_num_vars;
        let hi_num_vars = circuit_witness.instance_num_vars();
        let eq = &eq[0];

        // parallel unit logic handling
        let (thread_id, max_thread_id) = multi_threads_meta;
        let log2_max_thread_id = ceil_log2(max_thread_id);
        let threads_num_vars = hi_num_vars - log2_max_thread_id;
        let thread_s = thread_id << threads_num_vars;

        let challenges = &circuit_witness.challenges;

        let span = entered_span!("f1_g1");
        // merge next_layer_vec with next_layer_poly
        let next_layer_vec =
            circuit_witness.layers_ref()[layer_id as usize + 1].get_base_field_vec();

        let next_layer_poly: ArcMultilinearExtension<'a, E> =
            if circuit_witness.layers_ref()[layer_id as usize + 1].num_vars() - hi_num_vars
                < lo_in_num_vars
            {
                Arc::new(
                    // TODO(Matthias, by 2024-11-01): review whether we can redesign this API to avoid the deprecated resize_ranged
                    #[allow(deprecated)]
                    circuit_witness.layers_ref()[layer_id as usize + 1].resize_ranged(
                        1 << hi_num_vars,
                        1 << lo_in_num_vars,
                        multi_threads_meta.1,
                        multi_threads_meta.0,
                    ),
                )
            } else {
                Arc::new(
                    circuit_witness.layers_ref()[layer_id as usize + 1]
                        .get_ranged_mle(multi_threads_meta.1, multi_threads_meta.0),
                )
            };
        // f1(s1 || x1) = layers[i + 1](s1 || x1)
        let f1: ArcMultilinearExtension<'a, E> = next_layer_poly.clone();

        // g1(s1 || x1) = \sum_{s2}( \sum_{s3}( \sum_{x2}( \sum_{x3}(
        //     eq(rt, s1, s2, s3) * mul3(ry, x1, x2, x3) * layers[i + 1](s2 || x2) * layers[i + 1](s3 || x3)
        // ) ) ) ) + \sum_{s2}( \sum_{x2}(
        //     eq(rt, s1, s2) * mul2(ry, x1, x2) * layers[i + 1](s2 || x2)
        // ) ) + eq(rt, s1) * add(ry, x1)
        let mut g1 = vec![E::ZERO; 1 << f1.num_vars()];
        let mul3s_fanin_mapping = &layer.mul3s_fanin_mapping[Step1 as usize];
        let mul2s_fanin_mapping = &layer.mul2s_fanin_mapping[Step1 as usize];
        let adds_fanin_mapping = &layer.adds_fanin_mapping[Step1 as usize];

        prepare_stepx_g_fn!(
            &mut g1,
            lo_in_num_vars,
            lo_out_num_vars,
            thread_s,
            mul3s_fanin_mapping,
            |s_in, s_out, gate| {
                eq[s_out ^ gate.idx_out]
                    * next_layer_vec[s_in + gate.idx_in[1]]
                    * next_layer_vec[s_in + gate.idx_in[2]]
                    * gate.scalar.eval(challenges)
            },
            mul2s_fanin_mapping,
            |s_in, s_out, gate| {
                eq[s_out ^ gate.idx_out]
                    * next_layer_vec[s_in + gate.idx_in[1]]
                    * gate.scalar.eval(challenges)
            },
            adds_fanin_mapping,
            |_s_in, s_out, gate| eq[s_out ^ gate.idx_out] * gate.scalar.eval(challenges)
        );
        let g1 = DenseMultilinearExtension::from_evaluations_ext_vec(f1.num_vars(), g1).into();
        exit_span!(span);

        // f1'^{(j)}(s1 || x1) = subset[j][i](s1 || x1)
        // g1'^{(j)}(s1 || x1) = eq(rt, s1) paste_from[j](ry, x1)
        let span = entered_span!("f1j_g1j");
        let (f1_j, g1_j): (
            Vec<ArcMultilinearExtension<'a, E>>,
            Vec<ArcMultilinearExtension<'a, E>>,
        ) = izip!(&layer.paste_from)
            .map(|(j, paste_from)| {
                let paste_from_sources =
                    circuit_witness.layers_ref()[*j as usize].get_base_field_vec();
                let layer_per_instance_size = circuit_witness.layers_ref()[*j as usize]
                    .evaluations()
                    .len()
                    / circuit_witness.n_instances();

                let old_wire_id = |old_layer_id: usize, subset_wire_id: usize| -> usize {
                    circuit.layers[old_layer_id].copy_to[&{ layer_id }][subset_wire_id]
                };

                let mut f1_j = vec![E::BaseField::ZERO; 1 << f1.num_vars()];
                let mut g1_j = vec![E::ZERO; 1 << f1.num_vars()];

                for s in 0..(1 << (hi_num_vars - log2_max_thread_id)) {
                    let global_s = thread_s + s;
                    let instance_start_index = layer_per_instance_size * global_s;
                    // TODO find max consecutive subset_wire_ids and optimize by copy_from_slice
                    paste_from
                        .iter()
                        .enumerate()
                        .for_each(|(subset_wire_id, &new_wire_id)| {
                            f1_j[(s << lo_in_num_vars) ^ subset_wire_id] = paste_from_sources
                                [instance_start_index + old_wire_id(*j as usize, subset_wire_id)];
                            g1_j[(s << lo_in_num_vars) ^ subset_wire_id] +=
                                eq[(global_s << lo_out_num_vars) ^ new_wire_id];
                        });
                }
                let f1_j: ArcMultilinearExtension<'a, E> = Arc::new(
                    DenseMultilinearExtension::from_evaluations_vec(f1.num_vars(), f1_j),
                );
                let g1_j: ArcMultilinearExtension<'a, E> = Arc::new(
                    DenseMultilinearExtension::from_evaluations_ext_vec(f1.num_vars(), g1_j),
                );
                (f1_j, g1_j)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let (f, g): (
            Vec<ArcMultilinearExtension<'a, E>>,
            Vec<ArcMultilinearExtension<'a, E>>,
        ) = ([vec![f1], f1_j].concat(), [vec![g1], g1_j].concat());

        // sumcheck: sigma = \sum_{s1 || x1} f1(s1 || x1) * g1(s1 || x1) + \sum_j f1'_j(s1 || x1) * g1'_j(s1 || x1)
        let mut virtual_poly_1 = VirtualPolynomial::new(f[0].num_vars());
        for (f, g) in f.into_iter().zip(g.into_iter()) {
            let mut tmp = VirtualPolynomial::new_from_mle(f, E::ONE);
            tmp.mul_by_mle(g, E::BaseField::ONE);
            virtual_poly_1.merge(&tmp);
        }
        exit_span!(span);
        end_timer!(timer);

        virtual_poly_1
    }

    pub(super) fn combine_phase2_step1_evals(
        &mut self,
        circuit: &Circuit<E>,
        sumcheck_proof_1: SumcheckProof<E>,
        prover_state: sumcheck::structs::IOPProverState<E>,
    ) -> IOPProverStepMessage<E> {
        let layer = &circuit.layers[self.layer_id as usize];
        let eval_point_1 = sumcheck_proof_1.point.clone();
        let (f1_vec, g1_vec): (Vec<_>, Vec<_>) = prover_state
            .get_mle_final_evaluations()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let f1_vec_len = f1_vec.len();
        // eval_values_f1
        let mut eval_values_1 = f1_vec.into_iter().map(|(_, f1_j)| f1_j).collect_vec();

        // eval_values_g1[0]
        eval_values_1.push(g1_vec[0].1);

        self.to_next_phase_point_and_evals =
            vec![PointAndEval::new_from_ref(&eval_point_1, &eval_values_1[0])];
        izip!(
            layer.paste_from.iter(),
            eval_values_1[..f1_vec_len].iter().skip(1)
        )
        .for_each(|((&old_layer_id, _), &subset_value)| {
            self.subset_point_and_evals[old_layer_id as usize].push((
                self.layer_id,
                PointAndEval::new_from_ref(&eval_point_1, &subset_value),
            ));
        });
        self.to_next_step_point = eval_point_1;

        IOPProverStepMessage {
            sumcheck_proof: sumcheck_proof_1,
            sumcheck_eval_values: eval_values_1,
        }
    }

    /// Sumcheck 2 sigma = \sum_{s2 || x2} f2(s2 || x2) * g2(s2 || x2)
    ///     sigma = g1(rs1 || rx1) - eq(rt, rs1) * add(ry, rx1)
    ///     f2(s2 || x2) = layers[i + 1](s2 || x2)
    ///     g2(s2 || x2) = \sum_{s3}( \sum_{x3}(
    ///         eq(rt, rs1, s2, s3) * mul3(ry, rx1, x2, x3) * layers[i + 1](s3 || x3)
    ///     ) ) + eq(rt, rs1, s2) * mul2(ry, rx1, x2)
    #[tracing::instrument(skip_all, name = "build_phase2_step2_sumcheck_poly")]
    pub(super) fn build_phase2_step2_sumcheck_poly<'a>(
        layer_id: LayerId,
        eqs: &[Vec<E>; 2],
        circuit: &Circuit<E>,
        circuit_witness: &'a CircuitWitness<E>,
        multi_threads_meta: (usize, usize),
    ) -> VirtualPolynomial<'a, E> {
        let timer = start_timer!(|| "Prover sumcheck phase 2 step 2");
        let layer = &circuit.layers[layer_id as usize];
        let lo_out_num_vars = layer.num_vars;
        let lo_in_num_vars = layer.max_previous_num_vars;
        let (eq0, eq1) = (&eqs[0], &eqs[1]);

        // parallel unit logic handling
        let hi_num_vars = circuit_witness.instance_num_vars();
        let (thread_id, max_thread_id) = multi_threads_meta;
        let log2_max_thread_id = ceil_log2(max_thread_id);
        let threads_num_vars = hi_num_vars - log2_max_thread_id;
        let thread_s = thread_id << threads_num_vars;

        let next_layer_vec = circuit_witness.layers[layer_id as usize + 1].get_base_field_vec();

        let challenges = &circuit_witness.challenges;

        let span = entered_span!("f2_g2");
        // f2(s2 || x2) = layers[i + 1](s2 || x2)
        let f2 = Arc::new(
            circuit_witness.layers_ref()[layer_id as usize + 1]
                .get_ranged_mle(multi_threads_meta.1, multi_threads_meta.0),
        );

        // g2(s2 || x2) = \sum_{s3}( \sum_{x3}(
        //     eq(rt, rs1, s2, s3) * mul3(ry, rx1, x2, x3) * layers[i + 1](s3 || x3)
        // ) ) + eq(rt, rs1, s2) * mul2(ry, rx1, x2)
        let g2: ArcDenseMultilinearExtension<E> = {
            let mut g2 = vec![E::ZERO; 1 << (f2.num_vars())];
            let mul3s_fanin_mapping = &layer.mul3s_fanin_mapping[Step2 as usize];
            let mul2s_fanin_mapping = &layer.mul2s_fanin_mapping[Step2 as usize];
            prepare_stepx_g_fn!(
                &mut g2,
                lo_in_num_vars,
                lo_out_num_vars,
                thread_s,
                mul3s_fanin_mapping,
                |s_in, s_out, gate| {
                    eq0[s_out ^ gate.idx_out]
                        * eq1[s_in ^ gate.idx_in[0]]
                        * next_layer_vec[s_in + gate.idx_in[2]]
                        * gate.scalar.eval(challenges)
                },
                mul2s_fanin_mapping,
                |s_in, s_out, gate| {
                    eq0[s_out ^ gate.idx_out]
                        * eq1[s_in ^ gate.idx_in[0]]
                        * gate.scalar.eval(challenges)
                },
            );
            DenseMultilinearExtension::from_evaluations_ext_vec(f2.num_vars(), g2).into()
        };
        exit_span!(span);
        end_timer!(timer);

        // sumcheck: sigma = \sum_{s2 || x2} f2(s2 || x2) * g2(s2 || x2)
        let mut virtual_poly_2 = VirtualPolynomial::new_from_mle(f2, E::ONE);
        virtual_poly_2.mul_by_mle(g2, E::BaseField::ONE);

        virtual_poly_2
    }

    pub(super) fn combine_phase2_step2_evals(
        &mut self,
        _circuit: &Circuit<E>,
        sumcheck_proof_2: SumcheckProof<E>,
        prover_state: sumcheck::structs::IOPProverState<E>,
        no_step3: bool,
    ) -> IOPProverStepMessage<E> {
        let eval_point_2 = sumcheck_proof_2.point.clone();
        let (f2, g2): (Vec<_>, Vec<_>) = prover_state
            .get_mle_final_evaluations()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let (eval_value_f2, eval_value_g2) = (f2[0].1, g2[0].1);

        self.to_next_phase_point_and_evals
            .push(PointAndEval::new_from_ref(&eval_point_2, &eval_value_f2));
        self.to_next_step_point = eval_point_2;
        if no_step3 {
            IOPProverStepMessage {
                sumcheck_proof: sumcheck_proof_2,
                sumcheck_eval_values: vec![eval_value_f2],
            }
        } else {
            IOPProverStepMessage {
                sumcheck_proof: sumcheck_proof_2,
                sumcheck_eval_values: vec![eval_value_f2, eval_value_g2],
            }
        }
    }

    /// Sumcheck 3 sigma = \sum_{s3 || x3} f3(s3 || x3) * g3(s3 || x3)
    ///     sigma = g2(rs2 || rx2) - eq(rt, rs1, rs2) * mul2(ry, rx1, rx2)
    ///     f3(s3 || x3) = layers[i + 1](s3 || x3)
    ///     g3(s3 || x3) = eq(rt, rs1, rs2, s3) * mul3(ry, rx1, rx2, x3)
    #[tracing::instrument(skip_all, name = "build_phase2_step3_sumcheck_poly")]
    pub(super) fn build_phase2_step3_sumcheck_poly<'a>(
        layer_id: LayerId,
        eqs: &[Vec<E>; 3],
        circuit: &Circuit<E>,
        circuit_witness: &'a CircuitWitness<E>,
        multi_threads_meta: (usize, usize),
    ) -> VirtualPolynomial<'a, E> {
        let timer = start_timer!(|| "Prover sumcheck phase 2 step 3");
        let layer = &circuit.layers[layer_id as usize];
        let lo_out_num_vars = layer.num_vars;
        let lo_in_num_vars = layer.max_previous_num_vars;
        let (eq0, eq1, eq2) = (&eqs[0], &eqs[1], &eqs[2]);

        // parallel unit logic handling
        let hi_num_vars = circuit_witness.instance_num_vars();
        let (thread_id, max_thread_id) = multi_threads_meta;
        let log2_max_thread_id = ceil_log2(max_thread_id);
        let threads_num_vars = hi_num_vars - log2_max_thread_id;
        let thread_s = thread_id << threads_num_vars;

        let challenges = &circuit_witness.challenges;

        let span = entered_span!("f3_g3");
        // f3(s3 || x3) = layers[i + 1](s3 || x3)
        let f3 = Arc::new(
            circuit_witness.layers_ref()[layer_id as usize + 1]
                .get_ranged_mle(multi_threads_meta.1, multi_threads_meta.0),
        );
        // g3(s3 || x3) = eq(rt, rs1, rs2, s3) * mul3(ry, rx1, rx2, x3)
        let g3 = {
            let mut g3 = vec![E::ZERO; 1 << (f3.num_vars())];
            let fanin_mapping = &layer.mul3s_fanin_mapping[Step3 as usize];
            prepare_stepx_g_fn!(
                &mut g3,
                lo_in_num_vars,
                lo_out_num_vars,
                thread_s,
                fanin_mapping,
                |s_in, s_out, gate| {
                    eq0[s_out ^ gate.idx_out]
                        * eq1[s_in ^ gate.idx_in[0]]
                        * eq2[s_in ^ gate.idx_in[1]]
                        * gate.scalar.eval(challenges)
                }
            );
            DenseMultilinearExtension::from_evaluations_ext_vec(f3.num_vars(), g3).into()
        };

        let mut virtual_poly_3 = VirtualPolynomial::new_from_mle(f3, E::ONE);
        virtual_poly_3.mul_by_mle(g3, E::BaseField::ONE);

        exit_span!(span);
        end_timer!(timer);
        virtual_poly_3
    }

    pub(super) fn combine_phase2_step3_evals(
        &mut self,
        _circuit: &Circuit<E>,
        sumcheck_proof_3: SumcheckProof<E>,
        prover_state: sumcheck::structs::IOPProverState<E>,
    ) -> IOPProverStepMessage<E> {
        let eval_point_3 = sumcheck_proof_3.point.clone();
        let (f3, _): (Vec<_>, Vec<_>) = prover_state
            .get_mle_final_evaluations()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let eval_values_3 = vec![f3[0].1];
        self.to_next_phase_point_and_evals
            .push(PointAndEval::new_from_ref(&eval_point_3, &eval_values_3[0]));
        self.to_next_step_point = eval_point_3;
        IOPProverStepMessage {
            sumcheck_proof: sumcheck_proof_3,
            sumcheck_eval_values: eval_values_3,
        }
    }
}
