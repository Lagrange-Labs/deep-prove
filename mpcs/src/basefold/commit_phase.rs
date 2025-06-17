use super::{
    encoding::EncodingScheme,
    structure::{BasefoldCommitPhaseProof, BasefoldSpec},
    sumcheck::{
        sum_check_challenge_round, sum_check_first_round, sum_check_first_round_field_type,
        sum_check_last_round,
    },
};
use crate::util::{
    arithmetic::{interpolate_over_boolean_hypercube, interpolate2_weights},
    field_type_index_ext, field_type_iter_ext, field_type_par_iter_ext,
    hash::MerkleHasher,
    log2_strict,
    merkle_tree::MerkleTree,
};
use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::Itertools;
use serde::{Serialize, de::DeserializeOwned};
use transcript::Transcript;

use multilinear_extensions::{mle::FieldType, virtual_poly::build_eq_x_r_vec};

use crate::util::plonky2_util::reverse_index_bits_in_place;
use rayon::prelude::*;

use super::structure::BasefoldCommitmentWithWitness;

// outputs (trees, sumcheck_oracles, oracles, bh_evals, eq, eval)
pub fn commit_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    point: &[E],
    comm: &BasefoldCommitmentWithWitness<E, Spec::MerkleHasher>,
    transcript: &mut impl Transcript<E>,
    num_vars: usize,
    num_rounds: usize,
) -> (
    Vec<MerkleTree<E, Spec::MerkleHasher>>,
    BasefoldCommitPhaseProof<E, <Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
)
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let timer = start_timer!(|| "Commit phase");
    #[cfg(feature = "sanity-check")]
    assert_eq!(point.len(), num_vars);
    let mut trees = Vec::with_capacity(num_vars);
    let mut running_oracle = field_type_iter_ext(&comm.get_codewords()[0]).collect_vec();
    let mut running_evals = comm.polynomials_bh_evals[0].clone();

    #[cfg(feature = "sanity-check")]
    assert_eq!(
        running_oracle.len(),
        running_evals.len() << Spec::get_rate_log()
    );
    #[cfg(feature = "sanity-check")]
    assert_eq!(running_evals.len(), 1 << num_vars);

    // eq is the evaluation representation of the eq(X,r) polynomial over the hypercube
    let build_eq_timer = start_timer!(|| "Basefold::open");
    let mut eq = build_eq_x_r_vec(point);
    end_timer!(build_eq_timer);
    reverse_index_bits_in_place(&mut eq);

    let sumcheck_timer = start_timer!(|| "Basefold sumcheck first round");
    let mut last_sumcheck_message = sum_check_first_round_field_type(&mut eq, &mut running_evals);
    end_timer!(sumcheck_timer);

    #[cfg(feature = "sanity-check")]
    assert_eq!(last_sumcheck_message.len(), 3);

    let mut running_evals = match running_evals {
        FieldType::Ext(evals) => evals,
        FieldType::Base(evals) => evals.iter().map(|x| E::from(*x)).collect_vec(),
        _ => unreachable!(),
    };

    let mut sumcheck_messages = Vec::with_capacity(num_rounds);
    let mut roots = Vec::with_capacity(num_rounds - 1);
    let mut final_message = Vec::new();
    let mut running_tree_inner = Vec::new();
    for i in 0..num_rounds {
        let sumcheck_timer = start_timer!(|| format!("Basefold round {i}"));
        // For the first round, no need to send the running root, because this root is
        // committing to a vector that can be recovered from linearly combining other
        // already-committed vectors.
        transcript.append_field_element_exts(&last_sumcheck_message);
        sumcheck_messages.push(last_sumcheck_message);

        let challenge = transcript.get_and_append_challenge(b"commit round");

        // Fold the current oracle for FRI
        let new_running_oracle = basefold_one_round_by_interpolation_weights::<E, Spec>(
            pp,
            log2_strict(running_oracle.len()) - 1,
            &running_oracle,
            challenge.elements,
        );

        if i > 0 {
            let running_tree = MerkleTree::<E, Spec::MerkleHasher>::from_inner_leaves(
                running_tree_inner,
                FieldType::Ext(running_oracle),
            );
            trees.push(running_tree);
        }

        if i < num_rounds - 1 {
            last_sumcheck_message =
                sum_check_challenge_round(&mut eq, &mut running_evals, challenge.elements);

            // To avoid cloning the running oracle, explicitly separate the
            // computation of Merkle tree inner nodes and the building of
            // entire Merkle tree. First compute the inner nodes without
            // consuming the leaves, so that we can get the challenge.
            // Then the oracle will be used to fold to the next oracle in the next
            // round. After that, this oracle is free to be moved to build the
            // complete Merkle tree.
            running_tree_inner =
                MerkleTree::<E, Spec::MerkleHasher>::compute_inner_ext(&new_running_oracle);
            let running_root =
                MerkleTree::<E, Spec::MerkleHasher>::root_from_inner(&running_tree_inner);
            <Spec::MerkleHasher as MerkleHasher<E>>::digest_to_transcript(
                &running_root,
                transcript,
            );
            roots.push(running_root.clone());

            running_oracle = new_running_oracle;
        } else {
            // Clear this so the compiler knows the old value is safe to move.
            last_sumcheck_message = Vec::new();
            running_oracle = Vec::new();
            running_tree_inner = Vec::new();
            // The difference of the last round is that we don't need to compute the message,
            // and we don't interpolate the small polynomials. So after the last round,
            // running_evals is exactly the evaluation representation of the
            // folded polynomial so far.
            sum_check_last_round(&mut eq, &mut running_evals, challenge.elements);
            // For the FRI part, we send the current polynomial as the message.
            // Transform it back into little endiean before sending it
            reverse_index_bits_in_place(&mut running_evals);
            transcript.append_field_element_exts(&running_evals);
            final_message = running_evals;
            // To prevent the compiler from complaining that the value is moved
            running_evals = Vec::new();

            if cfg!(feature = "sanity-check") {
                // If the prover is honest, in the last round, the running oracle
                // on the prover side should be exactly the encoding of the folded polynomial.

                let mut coeffs = final_message.clone();
                interpolate_over_boolean_hypercube(&mut coeffs);
                if <Spec::EncodingScheme as EncodingScheme<E>>::message_is_even_and_odd_folding() {
                    reverse_index_bits_in_place(&mut coeffs);
                }
                let basecode = <Spec::EncodingScheme as EncodingScheme<E>>::encode(
                    pp,
                    &FieldType::Ext(coeffs),
                );
                let basecode = match basecode {
                    FieldType::Ext(b) => b,
                    _ => panic!("Should be ext field"),
                };

                let mut new_running_oracle = new_running_oracle;
                reverse_index_bits_in_place(&mut new_running_oracle);
                assert_eq!(basecode, new_running_oracle);
            }
        }
        end_timer!(sumcheck_timer);
    }
    end_timer!(timer);

    (
        trees,
        BasefoldCommitPhaseProof {
            sumcheck_messages,
            roots,
            final_message,
        },
    )
}

// outputs (trees, sumcheck_oracles, oracles, bh_evals, eq, eval)
#[allow(clippy::too_many_arguments)]
pub fn batch_commit_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    point: &[E],
    comms: &[BasefoldCommitmentWithWitness<E, Spec::MerkleHasher>],
    transcript: &mut impl Transcript<E>,
    num_vars: usize,
    num_rounds: usize,
    coeffs: &[E],
) -> (
    Vec<MerkleTree<E, Spec::MerkleHasher>>,
    BasefoldCommitPhaseProof<E, <Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
)
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let timer = start_timer!(|| "Batch Commit phase");
    assert_eq!(point.len(), num_vars);
    let mut trees = Vec::with_capacity(num_vars);
    let mut running_oracle = vec![E::ZERO; 1 << (num_vars + Spec::get_rate_log())];

    let build_oracle_timer = start_timer!(|| "Basefold build initial oracle");
    // Before the interaction, collect all the polynomials whose num variables match the
    // max num variables
    let running_oracle_len = running_oracle.len();
    comms
        .iter()
        .zip(coeffs.iter())
        .filter(|(comm, _)| comm.codeword_size() == running_oracle_len)
        .for_each(|(comm, coeff)| {
            running_oracle
                .par_iter_mut()
                .zip_eq(field_type_par_iter_ext(&comm.get_codewords()[0]))
                .for_each(|(r, a)| *r += a * *coeff);
        });
    end_timer!(build_oracle_timer);

    let build_oracle_timer = start_timer!(|| "Basefold build initial sumcheck evals");
    // Unlike the FRI part, the sum-check part still follows the original procedure,
    // and linearly combine all the polynomials once for all
    let mut sum_of_all_evals_for_sumcheck = vec![E::ZERO; 1 << num_vars];
    comms.iter().zip(coeffs.iter()).for_each(|(comm, coeff)| {
        sum_of_all_evals_for_sumcheck
            .par_chunks_mut(1 << (num_vars - log2_strict(comm.polynomials_bh_evals[0].len())))
            .zip(field_type_par_iter_ext(&comm.polynomials_bh_evals[0]))
            .for_each(|(rs, val)| {
                let mul = val * *coeff;
                rs.iter_mut().for_each(|r| *r += mul);
            });
    });
    end_timer!(build_oracle_timer);

    // eq is the evaluation representation of the eq(X,r) polynomial over the hypercube
    let mut eq = build_eq_x_r_vec(point);
    reverse_index_bits_in_place(&mut eq);

    let sumcheck_timer = start_timer!(|| "Basefold first round");
    let mut sumcheck_messages = Vec::with_capacity(num_rounds + 1);
    let mut last_sumcheck_message =
        sum_check_first_round(&mut eq, &mut sum_of_all_evals_for_sumcheck);
    sumcheck_messages.push(last_sumcheck_message.clone());
    end_timer!(sumcheck_timer);

    let mut roots = Vec::with_capacity(num_rounds - 1);
    let mut final_message = Vec::new();
    let mut running_tree_inner = Vec::new();
    let mut new_running_oracle: Vec<E> = Vec::new();
    for i in 0..num_rounds {
        let sumcheck_timer = start_timer!(|| format!("Batch basefold round {i}"));
        // For the first round, no need to send the running root, because this root is
        // committing to a vector that can be recovered from linearly combining other
        // already-committed vectors.
        transcript.append_field_element_exts(&last_sumcheck_message);

        let challenge = transcript
            .get_and_append_challenge(b"commit round")
            .elements;

        if i > 0 {
            let running_tree = MerkleTree::<E, Spec::MerkleHasher>::from_inner_leaves(
                running_tree_inner,
                FieldType::Ext(new_running_oracle.clone()),
            );
            trees.push(running_tree);

            // Then merge the rest polynomials whose sizes match the current running oracle
            let running_oracle_len = new_running_oracle.len();
            comms
                .iter()
                .zip(coeffs.iter())
                .filter(|(comm, _)| comm.codeword_size() == running_oracle_len)
                .for_each(|(comm, coeff)| {
                    new_running_oracle
                        .par_iter_mut()
                        .zip_eq(field_type_par_iter_ext(&comm.get_codewords()[0]))
                        .for_each(|(r, a)| *r += a * *coeff);
                });

            running_oracle = new_running_oracle;
        }

        // Fold the current oracle for FRI
        new_running_oracle = basefold_one_round_by_interpolation_weights::<E, Spec>(
            pp,
            log2_strict(running_oracle.len()) - 1,
            &running_oracle,
            challenge,
        );

        if i < num_rounds - 1 {
            last_sumcheck_message =
                sum_check_challenge_round(&mut eq, &mut sum_of_all_evals_for_sumcheck, challenge);
            sumcheck_messages.push(last_sumcheck_message.clone());

            running_tree_inner =
                MerkleTree::<E, Spec::MerkleHasher>::compute_inner_ext(&new_running_oracle);
            let running_root =
                MerkleTree::<E, Spec::MerkleHasher>::root_from_inner(&running_tree_inner);
            <Spec::MerkleHasher as MerkleHasher<E>>::digest_to_transcript(
                &running_root,
                transcript,
            );
            roots.push(running_root);
        } else {
            // Clear the value so the compiler does not think they are moved
            running_oracle = Vec::new();
            running_tree_inner = Vec::new();
            // The difference of the last round is that we don't need to compute the message,
            // and we don't interpolate the small polynomials. So after the last round,
            // sum_of_all_evals_for_sumcheck is exactly the evaluation representation of the
            // folded polynomial so far.
            sum_check_last_round(&mut eq, &mut sum_of_all_evals_for_sumcheck, challenge);
            // For the FRI part, we send the current polynomial as the message.
            // Transform it back into little endiean before sending it
            reverse_index_bits_in_place(&mut sum_of_all_evals_for_sumcheck);
            transcript.append_field_element_exts(&sum_of_all_evals_for_sumcheck);
            final_message = sum_of_all_evals_for_sumcheck;
            // To prevent the compiler from complaining that the value is moved
            sum_of_all_evals_for_sumcheck = Vec::new();

            if cfg!(feature = "sanity-check") {
                // If the prover is honest, in the last round, the running oracle
                // on the prover side should be exactly the encoding of the folded polynomial.

                let mut coeffs = final_message.clone();
                if <Spec::EncodingScheme as EncodingScheme<E>>::message_is_even_and_odd_folding() {
                    reverse_index_bits_in_place(&mut coeffs);
                }
                interpolate_over_boolean_hypercube(&mut coeffs);
                let basecode = <Spec::EncodingScheme as EncodingScheme<E>>::encode(
                    pp,
                    &FieldType::Ext(coeffs),
                );
                let basecode = match basecode {
                    FieldType::Ext(x) => x,
                    _ => panic!("Expected ext field"),
                };

                reverse_index_bits_in_place(&mut new_running_oracle);
                assert_eq!(basecode, new_running_oracle);
            }
        }
        end_timer!(sumcheck_timer);
    }
    end_timer!(timer);
    (
        trees,
        BasefoldCommitPhaseProof {
            sumcheck_messages,
            roots,
            final_message,
        },
    )
}

// outputs (trees, sumcheck_oracles, oracles, bh_evals, eq, eval)
#[allow(clippy::too_many_arguments)]
pub fn simple_batch_commit_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    point: &[E],
    batch_coeffs: &[E],
    comm: &BasefoldCommitmentWithWitness<E, Spec::MerkleHasher>,
    transcript: &mut impl Transcript<E>,
    num_vars: usize,
    num_rounds: usize,
) -> (
    Vec<MerkleTree<E, Spec::MerkleHasher>>,
    BasefoldCommitPhaseProof<E, <Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
)
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let timer = start_timer!(|| "Simple batch commit phase");
    assert_eq!(point.len(), num_vars);
    assert_eq!(comm.num_polys, batch_coeffs.len());
    let prepare_timer = start_timer!(|| "Prepare");
    let mut trees = Vec::with_capacity(num_vars);
    let batch_codewords_timer = start_timer!(|| "Batch codewords");
    let mut running_oracle = comm.batch_codewords(batch_coeffs);
    end_timer!(batch_codewords_timer);
    let mut running_evals = (0..(1 << num_vars))
        .into_par_iter()
        .map(|i| {
            comm.polynomials_bh_evals
                .iter()
                .zip(batch_coeffs)
                .map(|(eval, coeff)| field_type_index_ext(eval, i) * *coeff)
                .sum()
        })
        .collect::<Vec<_>>();
    end_timer!(prepare_timer);

    // eq is the evaluation representation of the eq(X,r) polynomial over the hypercube
    let build_eq_timer = start_timer!(|| "Basefold::build eq");
    let mut eq = build_eq_x_r_vec(point);
    end_timer!(build_eq_timer);

    let reverse_bits_timer = start_timer!(|| "Basefold::reverse bits");
    reverse_index_bits_in_place(&mut eq);
    end_timer!(reverse_bits_timer);

    let sumcheck_timer = start_timer!(|| "Basefold sumcheck first round");
    let mut last_sumcheck_message = sum_check_first_round(&mut eq, &mut running_evals);
    end_timer!(sumcheck_timer);

    let mut sumcheck_messages = Vec::with_capacity(num_rounds);
    let mut roots = Vec::with_capacity(num_rounds - 1);
    let mut final_message = Vec::new();
    let mut running_tree_inner = Vec::new();
    for i in 0..num_rounds {
        let sumcheck_timer = start_timer!(|| format!("Basefold round {i}"));
        // For the first round, no need to send the running root, because this root is
        // committing to a vector that can be recovered from linearly combining other
        // already-committed vectors.
        transcript.append_field_element_exts(&last_sumcheck_message);
        sumcheck_messages.push(last_sumcheck_message);

        let challenge = transcript
            .get_and_append_challenge(b"commit round")
            .elements;

        // Fold the current oracle for FRI
        let new_running_oracle = basefold_one_round_by_interpolation_weights::<E, Spec>(
            pp,
            log2_strict(running_oracle.len()) - 1,
            &running_oracle,
            challenge,
        );

        if i > 0 {
            let running_tree = MerkleTree::<E, Spec::MerkleHasher>::from_inner_leaves(
                running_tree_inner,
                FieldType::Ext(running_oracle),
            );
            trees.push(running_tree);
        }

        if i < num_rounds - 1 {
            last_sumcheck_message =
                sum_check_challenge_round(&mut eq, &mut running_evals, challenge);
            running_tree_inner =
                MerkleTree::<E, Spec::MerkleHasher>::compute_inner_ext(&new_running_oracle);
            let running_root =
                MerkleTree::<E, Spec::MerkleHasher>::root_from_inner(&running_tree_inner);
            <Spec::MerkleHasher as MerkleHasher<E>>::digest_to_transcript(
                &running_root,
                transcript,
            );
            roots.push(running_root);
            running_oracle = new_running_oracle;
        } else {
            // Assign a new value to the old running vars so that the compiler
            // knows the old value is safe to move.
            last_sumcheck_message = Vec::new();
            running_oracle = Vec::new();
            running_tree_inner = Vec::new();
            // The difference of the last round is that we don't need to compute the message,
            // and we don't interpolate the small polynomials. So after the last round,
            // running_evals is exactly the evaluation representation of the
            // folded polynomial so far.
            sum_check_last_round(&mut eq, &mut running_evals, challenge);
            // For the FRI part, we send the current polynomial as the message.
            // Transform it back into little endiean before sending it
            reverse_index_bits_in_place(&mut running_evals);
            transcript.append_field_element_exts(&running_evals);
            final_message = running_evals;
            // To avoid the compiler complaining that running_evals is moved.
            running_evals = Vec::new();

            if cfg!(feature = "sanity-check") {
                // If the prover is honest, in the last round, the running oracle
                // on the prover side should be exactly the encoding of the folded polynomial.

                let mut coeffs = final_message.clone();
                if <Spec::EncodingScheme as EncodingScheme<E>>::message_is_even_and_odd_folding() {
                    reverse_index_bits_in_place(&mut coeffs);
                }
                interpolate_over_boolean_hypercube(&mut coeffs);
                let basecode = <Spec::EncodingScheme as EncodingScheme<E>>::encode(
                    pp,
                    &FieldType::Ext(coeffs),
                );
                let basecode = match basecode {
                    FieldType::Ext(basecode) => basecode,
                    _ => panic!("Should be ext field"),
                };

                let mut new_running_oracle = new_running_oracle;
                reverse_index_bits_in_place(&mut new_running_oracle);
                assert_eq!(basecode, new_running_oracle);
            }
        }
        end_timer!(sumcheck_timer);
    }
    end_timer!(timer);
    (
        trees,
        BasefoldCommitPhaseProof {
            sumcheck_messages,
            roots,
            final_message,
        },
    )
}

fn basefold_one_round_by_interpolation_weights<E: ExtensionField, Spec: BasefoldSpec<E>>(
    pp: &<Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
    level: usize,
    values: &[E],
    challenge: E,
) -> Vec<E> {
    values
        .par_chunks_exact(2)
        .enumerate()
        .map(|(i, ys)| {
            let (x0, x1, w) =
                <Spec::EncodingScheme as EncodingScheme<E>>::prover_folding_coeffs(pp, level, i);
            interpolate2_weights([(x0, ys[0]), (x1, ys[1])], w, challenge)
        })
        .collect::<Vec<_>>()
}
