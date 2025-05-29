use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension, MultilinearExtension},
    virtual_poly::{VirtualPolynomial, build_eq_x_r_vec},
};
use p3_field::FieldAlgebra;
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use sumcheck::util::ceil_log2;
use transcript::Transcript;

use crate::{
    izip_parallelizable,
    prover::SumcheckStateV2,
    structs::{Circuit, CircuitWitness, IOPProverState, IOPProverStepMessage, PointAndEval},
};

// Prove the computation in the current layer for data parallel circuits.
// The number of terms depends on the gate.
// Here is an example of degree 3:
// layers[i](rt || ry) = \sum_x(
//     \sum_j  paste_from[j](ry, x) * subset[j][i](rt || x)
// ) + add_const(ry)
impl<E: ExtensionField> IOPProverState<E> {
    /// Sumcheck 1: sigma = \sum_j f1'_j(x1) * g1'_j(x1)
    ///     sigma = layers[i](rt || ry) - add_const(ry),
    ///     f1'^{(j)}(x1) = subset[j][i](rt || x1)
    ///     g1'^{(j)}(x1) = paste_from[j](ry, x1)
    #[tracing::instrument(skip_all, name = "prove_and_update_state_input_phase2_step1")]
    pub(super) fn prove_and_update_state_input_phase2_step1(
        &mut self,
        circuit: &Circuit<E>,
        circuit_witness: &CircuitWitness<E>,
        transcript: &mut impl Transcript<E>,
    ) -> IOPProverStepMessage<E> {
        let timer = start_timer!(|| "Prover sumcheck input phase 2 step 1");
        let layer = &circuit.layers[self.layer_id as usize];
        let lo_out_num_vars = layer.num_vars;
        let max_lo_in_num_vars = circuit.max_wit_in_num_vars.unwrap_or(0);
        let hi_num_vars = circuit_witness.instance_num_vars();
        let hi_point = &self.to_next_step_point[lo_out_num_vars..];

        let eq_y_ry = build_eq_x_r_vec(&self.to_next_step_point[..lo_out_num_vars]);

        let paste_from_wit_in = &circuit.paste_from_wits_in;
        let wits_in = circuit_witness.witness_in_ref();

        let (mut f_vec, mut g_vec): (
            Vec<ArcDenseMultilinearExtension<E>>,
            Vec<ArcDenseMultilinearExtension<E>>,
        ) = izip_parallelizable!(paste_from_wit_in)
            .enumerate()
            .map(|(j, (l, r))| {
                let wit_in = circuit_witness.witness_in_ref()[j].get_base_field_vec();
                let per_instance_size = wit_in.len() / circuit_witness.n_instances();
                let mut f = vec![E::BaseField::ZERO; 1 << (max_lo_in_num_vars + hi_num_vars)];
                let mut g = vec![E::ZERO; 1 << max_lo_in_num_vars];

                for (subset_wire_id, &eq_val) in eq_y_ry[*l..*r].iter().enumerate() {
                    for s in 0..(1 << hi_num_vars) {
                        let instance_start_index = s * per_instance_size;
                        f[(s << max_lo_in_num_vars) ^ subset_wire_id] =
                            wit_in[instance_start_index + subset_wire_id];
                    }
                    g[subset_wire_id] = eq_val;
                }

                (
                    {
                        let mut f = DenseMultilinearExtension::from_evaluations_vec(
                            max_lo_in_num_vars + hi_num_vars,
                            f,
                        );
                        f.fix_high_variables_in_place(hi_point);
                        f.into()
                    },
                    DenseMultilinearExtension::from_evaluations_ext_vec(max_lo_in_num_vars, g)
                        .into(),
                )
            })
            .unzip();

        let paste_from_counter_in = &circuit.paste_from_counter_in;
        let (f_vec_counter_in, g_vec_counter_in): (
            Vec<ArcDenseMultilinearExtension<E>>,
            Vec<ArcDenseMultilinearExtension<E>>,
        ) = izip_parallelizable!(paste_from_counter_in)
            .map(|(num_vars, (l, r))| {
                let mut f = vec![E::BaseField::ZERO; 1 << (max_lo_in_num_vars + hi_num_vars)];
                let mut g = vec![E::ZERO; 1 << max_lo_in_num_vars];

                for (subset_wire_id, &eq_val) in eq_y_ry[*l..*r].iter().enumerate() {
                    for s in 0..(1 << hi_num_vars) {
                        f[(s << max_lo_in_num_vars) ^ subset_wire_id] =
                            E::BaseField::from_canonical_u64(
                                ((s << num_vars) ^ subset_wire_id) as u64,
                            );
                    }
                    g[subset_wire_id] = eq_val;
                }
                (
                    {
                        let mut f = DenseMultilinearExtension::from_evaluations_vec(
                            max_lo_in_num_vars + hi_num_vars,
                            f,
                        );
                        f.fix_high_variables_in_place(hi_point);
                        f.into()
                    },
                    DenseMultilinearExtension::from_evaluations_ext_vec(max_lo_in_num_vars, g)
                        .into(),
                )
            })
            .unzip();

        f_vec.extend(f_vec_counter_in);
        g_vec.extend(g_vec_counter_in);

        let mut virtual_poly = VirtualPolynomial::new(max_lo_in_num_vars);
        for (f, g) in f_vec.into_iter().zip(g_vec.into_iter()) {
            let mut tmp = VirtualPolynomial::new_from_mle(f, E::ONE);
            tmp.mul_by_mle(g, E::BaseField::ONE);
            virtual_poly.merge(&tmp);
        }

        #[allow(deprecated)]
        let (sumcheck_proofs, prover_state) =
            SumcheckStateV2::prove_parallel(virtual_poly, transcript);
        let eval_point = sumcheck_proofs.point.clone();
        let (f_vec, _): (Vec<_>, Vec<_>) = prover_state
            .get_mle_final_evaluations()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let eval_values_f = f_vec
            .into_iter()
            .take(wits_in.len())
            .map(|(_, f)| f)
            .collect_vec();

        self.to_next_phase_point_and_evals = izip!(paste_from_wit_in.iter(), eval_values_f.iter())
            .map(|((l, r), eval)| {
                let num_vars = ceil_log2(*r - *l);
                let point = [&eval_point[..num_vars], hi_point].concat();
                let wit_in_eval = *eval
                    * eval_point[num_vars..]
                        .iter()
                        .map(|x| E::ONE - *x)
                        .product::<E>()
                        .inverse();
                PointAndEval::new_from_ref(&point, &wit_in_eval)
            })
            .collect_vec();
        self.to_next_step_point = [&eval_point, hi_point].concat();

        end_timer!(timer);

        IOPProverStepMessage {
            sumcheck_proof: sumcheck_proofs,
            sumcheck_eval_values: eval_values_f,
        }
    }
}
