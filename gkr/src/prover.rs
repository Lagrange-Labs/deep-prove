use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::virtual_poly::{VirtualPolynomial, build_eq_x_r_vec};

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use simple_frontend::structs::LayerId;
use transcript::Transcript;

use crate::{
    entered_span, exit_span,
    structs::{
        Circuit, CircuitWitness, GKRInputClaims, IOPProof, IOPProverState, PointAndEval,
        SumcheckStepType,
    },
    tracing_span,
};

mod phase1;
mod phase1_output;
mod phase2;
mod phase2_input;
mod phase2_linear;

#[cfg(test)]
mod test;

type SumcheckStateV2<'a, F> = sumcheck::structs::IOPProverState<'a, F>;

impl<E: ExtensionField> IOPProverState<E> {
    /// Prove process for data parallel circuits.
    #[tracing::instrument(skip_all, name = "gkr::prove_parallel")]
    pub fn prove_parallel(
        circuit: &Circuit<E>,
        circuit_witness: &CircuitWitness<E>,
        output_evals: Vec<PointAndEval<E>>,
        wires_out_evals: Vec<PointAndEval<E>>,
        max_thread_id: usize,
        transcript: &mut impl Transcript<E>,
    ) -> (IOPProof<E>, GKRInputClaims<E>) {
        let timer = start_timer!(|| "Proving");
        let span = entered_span!("Proving");
        // TODO: Currently haven't support non-power-of-two number of instances.
        assert!(circuit_witness.n_instances == 1 << circuit_witness.instance_num_vars());

        let mut prover_state = tracing_span!("prover_init_parallel").in_scope(|| {
            Self::prover_init_parallel(
                circuit,
                circuit_witness.instance_num_vars(),
                output_evals,
                wires_out_evals,
                transcript,
            )
        });

        let sumcheck_proofs = (0..circuit.layers.len() as LayerId)
            .flat_map(|layer_id| {
                let timer = start_timer!(|| format!("Prove layer {layer_id}"));

                prover_state.layer_id = layer_id;

                let dummy_step = SumcheckStepType::Undefined;
                let proofs = circuit.layers[layer_id as usize]
                    .sumcheck_steps
                    .iter()
                    .chain(vec![&dummy_step, &dummy_step])
                    .tuple_windows()
                    .flat_map(|steps| match steps {
                        (SumcheckStepType::OutputPhase1Step1, _, _) => {
                            let alpha = transcript
                                .get_and_append_challenge(b"combine subset evals")
                                .elements;
                            let hi_num_vars = circuit_witness.instance_num_vars();
                            let eq_t = prover_state
                                .to_next_phase_point_and_evals
                                .par_iter()
                                .chain(
                                    prover_state.subset_point_and_evals[layer_id as usize]
                                        .par_iter()
                                        .map(|(_, point_and_eval)| point_and_eval),
                                )
                                .chain(
                                    vec![PointAndEval {
                                        point: prover_state.assert_point.clone(),
                                        eval: E::ZERO, // evaluation value doesn't matter
                                    }]
                                    .par_iter(),
                                )
                                .map(|point_and_eval| {
                                    let point_lo_num_vars =
                                        point_and_eval.point.len() - hi_num_vars;
                                    build_eq_x_r_vec(&point_and_eval.point[point_lo_num_vars..])
                                })
                                .collect::<Vec<Vec<_>>>();

                            let virtual_polys: Vec<VirtualPolynomial<E>> = (0..max_thread_id)
                                .into_par_iter()
                                .map(|thread_id| {
                                    let span = entered_span!("build_poly");
                                    let virtual_poly =
                                        Self::build_state_output_phase1_step1_sumcheck_poly(
                                            &prover_state,
                                            &eq_t,
                                            alpha,
                                            circuit,
                                            circuit_witness,
                                            (thread_id, max_thread_id),
                                        );
                                    exit_span!(span);
                                    virtual_poly
                                })
                                .collect();

                            let (sumcheck_proof, sumcheck_prover_state) =
                                sumcheck::structs::IOPProverState::<E>::prove_batch_polys(
                                    max_thread_id,
                                    virtual_polys,
                                    transcript,
                                );

                            let prover_msg = prover_state.combine_output_phase1_step1_evals(
                                sumcheck_proof,
                                sumcheck_prover_state,
                            );

                            vec![prover_msg]
                        }
                        (SumcheckStepType::Phase1Step1, _, _) => {
                            let alpha = transcript
                                .get_and_append_challenge(b"combine subset evals")
                                .elements;
                            let hi_num_vars = circuit_witness.instance_num_vars();
                            let eq_t = prover_state
                                .to_next_phase_point_and_evals
                                .par_iter()
                                .chain(
                                    prover_state.subset_point_and_evals[layer_id as usize]
                                        .par_iter()
                                        .map(|(_, point_and_eval)| point_and_eval),
                                )
                                .map(|point_and_eval| {
                                    let point_lo_num_vars =
                                        point_and_eval.point.len() - hi_num_vars;
                                    build_eq_x_r_vec(&point_and_eval.point[point_lo_num_vars..])
                                })
                                .collect::<Vec<Vec<_>>>();

                            let virtual_polys: Vec<VirtualPolynomial<E>> = (0..max_thread_id)
                                .into_par_iter()
                                .map(|thread_id| {
                                    let span = entered_span!("build_poly");
                                    let virtual_poly = Self::build_phase1_step1_sumcheck_poly(
                                        &prover_state,
                                        layer_id,
                                        alpha,
                                        &eq_t,
                                        circuit,
                                        circuit_witness,
                                        (thread_id, max_thread_id),
                                    );
                                    exit_span!(span);
                                    virtual_poly
                                })
                                .collect();

                            let (sumcheck_proof, sumcheck_prover_state) =
                                sumcheck::structs::IOPProverState::<E>::prove_batch_polys(
                                    max_thread_id,
                                    virtual_polys,
                                    transcript,
                                );

                            let prover_msg = prover_state
                                .combine_phase1_step1_evals(sumcheck_proof, sumcheck_prover_state);

                            vec![prover_msg]
                        }
                        (SumcheckStepType::Phase2Step1, step2, _) => {
                            let span = entered_span!("phase2_gkr");
                            let max_steps = match step2 {
                                SumcheckStepType::Phase2Step2 => 3,
                                SumcheckStepType::Phase2Step2NoStep3 => 2,
                                _ => unreachable!(),
                            };

                            let mut eqs = vec![];
                            let mut res = vec![];
                            for step in 0..max_steps {
                                let bounded_eval_point = prover_state.to_next_step_point.clone();
                                eqs.push(build_eq_x_r_vec(&bounded_eval_point));
                                // build step round poly
                                let virtual_polys: Vec<VirtualPolynomial<E>> = (0..max_thread_id)
                                    .into_par_iter()
                                    .map(|thread_id| {
                                        let span = entered_span!("build_poly");
                                        let virtual_poly = match step {
                                            0 => {
                                                let virtual_poly =
                                                    Self::build_phase2_step1_sumcheck_poly(
                                                        eqs.as_slice().try_into().unwrap(),
                                                        layer_id,
                                                        circuit,
                                                        circuit_witness,
                                                        (thread_id, max_thread_id),
                                                    );
                                                virtual_poly
                                            }
                                            1 => {
                                                let virtual_poly =
                                                    Self::build_phase2_step2_sumcheck_poly(
                                                        layer_id,
                                                        eqs.as_slice().try_into().unwrap(),
                                                        circuit,
                                                        circuit_witness,
                                                        (thread_id, max_thread_id),
                                                    );
                                                virtual_poly
                                            }
                                            2 => {
                                                let virtual_poly =
                                                    Self::build_phase2_step3_sumcheck_poly(
                                                        layer_id,
                                                        eqs.as_slice().try_into().unwrap(),
                                                        circuit,
                                                        circuit_witness,
                                                        (thread_id, max_thread_id),
                                                    );
                                                virtual_poly
                                            }
                                            _ => unimplemented!(),
                                        };
                                        exit_span!(span);
                                        virtual_poly
                                    })
                                    .collect();

                                let (sumcheck_proof, sumcheck_prover_state) =
                                    sumcheck::structs::IOPProverState::<E>::prove_batch_polys(
                                        max_thread_id,
                                        virtual_polys,
                                        transcript,
                                    );

                                let iop_prover_step = match step {
                                    0 => prover_state.combine_phase2_step1_evals(
                                        circuit,
                                        sumcheck_proof,
                                        sumcheck_prover_state,
                                    ),
                                    1 => {
                                        let no_step3: bool = max_steps == 2;
                                        prover_state.combine_phase2_step2_evals(
                                            circuit,
                                            sumcheck_proof,
                                            sumcheck_prover_state,
                                            no_step3,
                                        )
                                    }
                                    2 => prover_state.combine_phase2_step3_evals(
                                        circuit,
                                        sumcheck_proof,
                                        sumcheck_prover_state,
                                    ),
                                    _ => unimplemented!(),
                                };

                                res.push(iop_prover_step);
                            }
                            exit_span!(span);
                            res
                        }
                        (SumcheckStepType::LinearPhase2Step1, _, _) => [prover_state
                            .prove_and_update_state_linear_phase2_step1(
                                circuit,
                                circuit_witness,
                                transcript,
                            )]
                        .to_vec(),
                        (SumcheckStepType::InputPhase2Step1, _, _) => [prover_state
                            .prove_and_update_state_input_phase2_step1(
                                circuit,
                                circuit_witness,
                                transcript,
                            )]
                        .to_vec(),
                        _ => {
                            vec![]
                        }
                    })
                    .collect_vec();
                end_timer!(timer);

                proofs
            })
            .collect_vec();
        end_timer!(timer);
        exit_span!(span);

        (
            IOPProof { sumcheck_proofs },
            GKRInputClaims {
                point_and_evals: prover_state.to_next_phase_point_and_evals,
            },
        )
    }

    /// Initialize proving state for data parallel circuits.
    fn prover_init_parallel(
        circuit: &Circuit<E>,
        instance_num_vars: usize,
        output_evals: Vec<PointAndEval<E>>,
        wires_out_evals: Vec<PointAndEval<E>>,
        transcript: &mut impl Transcript<E>,
    ) -> Self {
        let n_layers = circuit.layers.len();
        let output_wit_num_vars = circuit.layers[0].num_vars + instance_num_vars;
        let mut subset_point_and_evals = vec![vec![]; n_layers];
        let to_next_step_point = if output_evals.is_empty() {
            wires_out_evals.last().unwrap().point.clone()
        } else {
            output_evals.last().unwrap().point.clone()
        };
        let assert_point = (0..output_wit_num_vars)
            .map(|_| {
                transcript
                    .get_and_append_challenge(b"assert_point")
                    .elements
            })
            .collect_vec();
        let to_next_phase_point_and_evals = output_evals;
        subset_point_and_evals[0] = wires_out_evals
            .into_iter()
            .map(|p| (0 as LayerId, p))
            .collect();

        Self {
            to_next_phase_point_and_evals,
            subset_point_and_evals,
            to_next_step_point,

            assert_point,
            // Default
            layer_id: 0,
        }
    }
}
