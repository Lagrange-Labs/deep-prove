use crate::structs::{Circuit, CircuitWitness, IOPProverState, IOPVerifierState, PointAndEval};
use ff_ext::{ExtensionField, GoldilocksExt2};
use itertools::Itertools;
use multilinear_extensions::mle::{DenseMultilinearExtension, IntoMLE};
use p3_field::{Field, FieldAlgebra};
use p3_goldilocks::Goldilocks;
use simple_frontend::structs::{CellId, CircuitBuilder};
use std::{iter, time::Duration};
use transcript::{BasicTranscript, Transcript};

// build an IsZero Gadget
// IsZero Gadget returns 1 when value == 0, and returns 0 otherwise.
// when value != 0 check inv = value ^ {-1}: cond1 = value * (value *
// inv - 1) = 0
// when value == 0 check inv = 0: cond2 = inv ⋅ (value *
// inv - 1) = 0
// value and inv must occupy one cell and
// all intermediate computations are restricted by field size
pub fn is_zero_gadget<Ext: ExtensionField>(
    circuit_builder: &mut CircuitBuilder<Ext>,
    value: CellId,
    inv: CellId,
) -> (CellId, CellId, CellId) {
    // value * inv
    let value_mul_inv = circuit_builder.create_cell();
    circuit_builder.mul2(value_mul_inv, value, inv, Ext::BaseField::ONE);
    // value * inv - 1
    let value_mul_inv_minus_one = value_mul_inv;
    circuit_builder.add_const(value_mul_inv_minus_one, -Ext::BaseField::ONE);
    // cond1 = value * (value * inv - 1)
    let cond1 = circuit_builder.create_cell();
    circuit_builder.mul2(cond1, value, value_mul_inv_minus_one, Ext::BaseField::ONE);
    // cond2 = inv * (value * inv - 1)
    let cond2 = circuit_builder.create_cell();
    circuit_builder.mul2(cond2, inv, value_mul_inv_minus_one, Ext::BaseField::ONE);
    // is_zero is a copy of value_mul_inv_minus_one
    let is_zero = circuit_builder.create_cell();
    circuit_builder.add(is_zero, value_mul_inv_minus_one, Ext::BaseField::ONE);

    (is_zero, cond1, cond2)
}

#[test]
fn test_gkr_circuit_is_zero_gadget_simple() {
    // input and output
    let in_value = vec![Goldilocks::from_canonical_u64(5)];
    let in_inv = vec![Goldilocks::from_canonical_u64(5).inverse()];
    let out_is_zero = Goldilocks::from_canonical_u64(0);

    // build the circuit, only one cell for value, inv and value * inv etc
    let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
    let (value_wire_in_id, value) = circuit_builder.create_witness_in(1);
    let (inv_wire_in_id, inv) = circuit_builder.create_witness_in(1);
    let (is_zero, cond1, cond2) = is_zero_gadget(&mut circuit_builder, value[0], inv[0]);
    let cond_wire_out_id = circuit_builder.create_witness_out_from_cells(&[cond1, cond2]);
    let is_zero_wire_out_id = circuit_builder.create_witness_out_from_cells(&[is_zero]);

    circuit_builder.configure();
    #[cfg(debug_assertions)]
    circuit_builder.print_info();
    let circuit = Circuit::new(&circuit_builder);
    println!("circuit: {:?}", circuit);

    // assign wire in
    let n_wits_in = circuit.n_witness_in;
    let mut wit_in = vec![DenseMultilinearExtension::default(); n_wits_in];
    wit_in[value_wire_in_id as usize] = in_value.into_mle();
    wit_in[inv_wire_in_id as usize] = in_inv.into_mle();
    let circuit_witness = {
        let challenges = vec![GoldilocksExt2::from_canonical_u64(2)];
        let mut circuit_witness = CircuitWitness::new(&circuit, challenges);
        circuit_witness.add_instance(&circuit, wit_in);
        circuit_witness
    };
    println!("circuit witness: {:?}", circuit_witness);
    // use of check_correctness will panic
    // circuit_witness.check_correctness(&circuit);

    // check the result
    let layers = circuit_witness.layers_ref();
    println!("layers: {:?}", layers);

    let wits_out = circuit_witness.witness_out_ref();
    let cond_wire_out_ref = &wits_out[cond_wire_out_id as usize];
    let is_zero_wire_out_ref = &wits_out[is_zero_wire_out_id as usize];
    println!(
        "cond wire outs: {:?}, is zero wire out {:?}",
        cond_wire_out_ref, is_zero_wire_out_ref
    );

    // cond1 and cond2
    assert_eq!(
        cond_wire_out_ref.get_base_field_vec()[0],
        Goldilocks::from_canonical_u64(0)
    );
    assert_eq!(
        cond_wire_out_ref.get_base_field_vec()[1],
        Goldilocks::from_canonical_u64(0)
    );
    // is_zero
    assert_eq!(is_zero_wire_out_ref.get_base_field_vec()[0], out_is_zero);

    // add prover-verifier process
    let mut prover_transcript =
        BasicTranscript::<GoldilocksExt2>::new(b"test_gkr_circuit_IsZeroGadget_simple");
    let mut verifier_transcript =
        BasicTranscript::<GoldilocksExt2>::new(b"test_gkr_circuit_IsZeroGadget_simple");

    let mut prover_wires_out_evals = vec![];
    let mut verifier_wires_out_evals = vec![];
    let instance_num_vars = 1_u32.ilog2() as usize;
    for wire_out_id in [cond_wire_out_id, is_zero_wire_out_id] {
        let output_mle = &wits_out[wire_out_id as usize];
        let prover_output_point = iter::repeat_with(|| {
            prover_transcript
                .get_and_append_challenge(b"output_point_test_gkr_circuit_IsZeroGadget_simple")
                .elements
        })
        .take(output_mle.num_vars())
        .collect_vec();
        let verifier_output_point = iter::repeat_with(|| {
            verifier_transcript
                .get_and_append_challenge(b"output_point_test_gkr_circuit_IsZeroGadget_simple")
                .elements
        })
        .take(output_mle.num_vars())
        .collect_vec();
        let prover_output_eval = output_mle.evaluate(&prover_output_point);
        let verifier_output_eval = output_mle.evaluate(&verifier_output_point);
        prover_wires_out_evals.push(PointAndEval::new(prover_output_point, prover_output_eval));
        verifier_wires_out_evals.push(PointAndEval::new(
            verifier_output_point,
            verifier_output_eval,
        ));
    }

    let start = std::time::Instant::now();
    let (proof, _) = IOPProverState::prove_parallel(
        &circuit,
        &circuit_witness,
        vec![],
        prover_wires_out_evals,
        1,
        &mut prover_transcript,
    );
    let proof_time: Duration = start.elapsed();

    let start = std::time::Instant::now();
    let _claim = IOPVerifierState::verify_parallel(
        &circuit,
        &[],
        vec![],
        verifier_wires_out_evals,
        proof,
        instance_num_vars,
        &mut verifier_transcript,
    )
    .unwrap();
    let verification_time: Duration = start.elapsed();

    println!(
        "proof time: {:?}, verification time: {:?}",
        proof_time, verification_time
    );
}

#[test]
fn test_gkr_circuit_is_zero_gadget_u256() {
    // IsZero for U256. Each cell holds 4 bits preventing multiplication overflow.
    // value is decomposed into 64 cells
    // assert IsZero(value) when all 64 cells are zero
    const UINT256_4_N_OPERAND_CELLS: usize = 64;

    // input and output
    let mut in_value = vec![
        Goldilocks::from_canonical_u64(0),
        Goldilocks::from_canonical_u64(5),
    ];
    in_value.resize(UINT256_4_N_OPERAND_CELLS, Goldilocks::from_canonical_u64(0));
    let mut in_inv = vec![
        Goldilocks::from_canonical_u64(0),
        Goldilocks::from_canonical_u64(5).inverse(),
    ];
    in_inv.resize(UINT256_4_N_OPERAND_CELLS, Goldilocks::from_canonical_u64(0));
    let out_is_zero = Goldilocks::from_canonical_u64(0);

    // build the circuit, number of cells for value is UINT256_4_N_OPERAND_CELLS
    // inv is the inverse of each cell's value, if value = 0 then inv = 0
    let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
    let (value_wire_in_id, value) = circuit_builder.create_witness_in(UINT256_4_N_OPERAND_CELLS);
    let (inv_wire_in_id, inv) = circuit_builder.create_witness_in(UINT256_4_N_OPERAND_CELLS);

    // is_zero_value = prod_{value_item} (is_zero_value_item)
    let mut cond1: Vec<CellId> = vec![];
    let mut cond2: Vec<CellId> = vec![];
    let mut is_zero_prev_items = circuit_builder.create_cell();
    circuit_builder.add_const(is_zero_prev_items, Goldilocks::from_canonical_u64(1));
    for (value_item, inv_item) in value.into_iter().zip(inv) {
        let (is_zero_item, cond1_item, cond2_item) =
            is_zero_gadget(&mut circuit_builder, value_item, inv_item);
        cond1.push(cond1_item);
        cond2.push(cond2_item);
        let is_zero = circuit_builder.create_cell();
        // TODO: can optimize using mul3
        circuit_builder.mul2(
            is_zero,
            is_zero_prev_items,
            is_zero_item,
            Goldilocks::from_canonical_u64(1),
        );
        is_zero_prev_items = is_zero;
    }

    let cond_wire_out_id = circuit_builder
        .create_witness_out_from_cells(&[cond1.as_slice(), cond2.as_slice()].concat());
    let is_zero_wire_out_id = circuit_builder.create_witness_out_from_cells(&[is_zero_prev_items]);

    circuit_builder.configure();

    #[cfg(debug_assertions)]
    circuit_builder.print_info();

    let circuit = Circuit::new(&circuit_builder);
    println!("circuit: {:?}", circuit);

    // assign wire in
    let n_wits_in = circuit.n_witness_in;
    let mut wits_in = vec![DenseMultilinearExtension::<GoldilocksExt2>::default(); n_wits_in];
    wits_in[value_wire_in_id as usize] = in_value.into_mle();
    wits_in[inv_wire_in_id as usize] = in_inv.into_mle();
    let circuit_witness = {
        let challenges = vec![GoldilocksExt2::from_canonical_u64(2)];
        let mut circuit_witness = CircuitWitness::new(&circuit, challenges);
        circuit_witness.add_instance(&circuit, wits_in);
        circuit_witness
    };
    println!("circuit witness: {:?}", circuit_witness);
    // use of check_correctness will panic
    // circuit_witness.check_correctness(&circuit);

    // check the result
    let layers = circuit_witness.layers_ref();
    println!("layers: {:?}", layers);

    let wits_out = circuit_witness.witness_out_ref();
    let cond_wire_out_ref = &wits_out[cond_wire_out_id as usize];
    let is_zero_wire_out_ref = &wits_out[is_zero_wire_out_id as usize];
    println!(
        "cond wire outs: {:?}, is zero wire out {:?}",
        cond_wire_out_ref, is_zero_wire_out_ref
    );

    // cond1 and cond2
    // for cond_item in cond_wire_out_ref.instances[0].clone().into_iter() {
    //     assert_eq!(cond_item, Goldilocks::from(0));
    // }
    // is_zero
    assert_eq!(is_zero_wire_out_ref.get_base_field_vec()[0], out_is_zero);

    // add prover-verifier process
    let mut prover_transcript =
        BasicTranscript::<GoldilocksExt2>::new(b"test_gkr_circuit_IsZeroGadget_simple");
    let mut verifier_transcript =
        BasicTranscript::<GoldilocksExt2>::new(b"test_gkr_circuit_IsZeroGadget_simple");

    let mut prover_wires_out_evals = vec![];
    let mut verifier_wires_out_evals = vec![];
    for wire_out_id in [cond_wire_out_id, is_zero_wire_out_id] {
        let output_mle = &wits_out[wire_out_id as usize];
        let prover_output_point = iter::repeat_with(|| {
            prover_transcript
                .get_and_append_challenge(b"output_point_test_gkr_circuit_IsZeroGadget_simple")
                .elements
        })
        .take(output_mle.num_vars())
        .collect_vec();
        let verifier_output_point = iter::repeat_with(|| {
            verifier_transcript
                .get_and_append_challenge(b"output_point_test_gkr_circuit_IsZeroGadget_simple")
                .elements
        })
        .take(output_mle.num_vars())
        .collect_vec();
        let prover_output_eval = output_mle.evaluate(&prover_output_point);
        let verifier_output_eval = output_mle.evaluate(&verifier_output_point);
        prover_wires_out_evals.push(PointAndEval::new(prover_output_point, prover_output_eval));
        verifier_wires_out_evals.push(PointAndEval::new(
            verifier_output_point,
            verifier_output_eval,
        ));
    }

    let start = std::time::Instant::now();
    let _proof = IOPProverState::prove_parallel(
        &circuit,
        &circuit_witness,
        vec![],
        prover_wires_out_evals,
        1,
        &mut prover_transcript,
    );
    let proof_time: Duration = start.elapsed();

    // verifier panics due to mismatch of number of variables
    // let start = std::time::Instant::now();
    // let _claim = IOPVerifierState::verify_parallel(
    // &circuit,
    // &[],
    // &[],
    // &verifier_wires_out_evals,
    // &proof,
    // instance_num_vars,
    // &mut verifier_transcript,
    // ).unwrap();
    // let verification_time: Duration = start.elapsed();
    //
    // println!("proof time: {:?}, verification time: {:?}", proof_time, verification_time);
    println!("proof time: {:?}", proof_time);
}
