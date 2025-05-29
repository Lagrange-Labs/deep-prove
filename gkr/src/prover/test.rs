use std::collections::HashMap;

use ark_std::test_rng;
use ff_ext::{ExtensionField, GoldilocksExt2};
use itertools::{Itertools, izip};
use multilinear_extensions::mle::DenseMultilinearExtension;
use p3_field::FieldAlgebra;
use simple_frontend::structs::{ChallengeConst, ChallengeId, CircuitBuilder, MixedCell};
use transcript::BasicTranscript;

use crate::{
    structs::{Circuit, CircuitWitness, IOPProverState, IOPVerifierState, PointAndEval},
    utils::i64_to_field,
};

fn copy_and_paste_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();
    // Layer 3
    let (_, input) = circuit_builder.create_witness_in(4);

    // Layer 2
    let mul_01 = circuit_builder.create_cell();
    circuit_builder.mul2(mul_01, input[0], input[1], Ext::BaseField::ONE);

    // Layer 1
    let mul_012 = circuit_builder.create_cell();
    circuit_builder.mul2(mul_012, mul_01, input[2], Ext::BaseField::ONE);

    // Layer 0
    let (_, mul_001123) = circuit_builder.create_witness_out(1);
    circuit_builder.mul3(
        mul_001123[0],
        mul_01,
        mul_012,
        input[3],
        Ext::BaseField::ONE,
    );

    circuit_builder.configure();

    Circuit::new(&circuit_builder)
}

fn copy_and_paste_witness<'a, Ext: ExtensionField>()
-> (Vec<DenseMultilinearExtension<Ext>>, CircuitWitness<'a, Ext>) {
    // witness_in, single instance
    let inputs = vec![vec![
        i64_to_field(5),
        i64_to_field(7),
        i64_to_field(11),
        i64_to_field(13),
    ]];
    let witness_in: Vec<DenseMultilinearExtension<Ext>> = vec![inputs.into()];

    let layers: Vec<DenseMultilinearExtension<Ext>> = vec![
        vec![vec![i64_to_field(175175)]].into(),
        vec![vec![
            i64_to_field(385),
            i64_to_field(35),
            i64_to_field(13),
            i64_to_field(0), // pad
        ]]
        .into(),
        vec![vec![i64_to_field(35), i64_to_field(11)]].into(),
        vec![vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
        ]]
        .into(),
    ];

    let outputs = vec![vec![i64_to_field(175175)]];
    let witness_out: Vec<DenseMultilinearExtension<Ext>> = vec![outputs.into()];

    (
        witness_in.clone(),
        CircuitWitness {
            layers: layers.into_iter().map(|w| w.into()).collect(),
            witness_in: witness_in.into_iter().map(|w| w.into()).collect(),
            witness_out: witness_out.into_iter().map(|w| w.into()).collect(),
            n_instances: 1,
            challenges: HashMap::new(),
        },
    )
}

fn paste_from_wit_in_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();

    // Layer 2
    let (_leaf_id1, leaves1) = circuit_builder.create_witness_in(3);
    let (_leaf_id2, leaves2) = circuit_builder.create_witness_in(3);
    // Unused input elements should also be in the circuit.
    let (_dummy_id, _) = circuit_builder.create_witness_in(3);
    let _ = circuit_builder.create_counter_in(1);
    let _ = circuit_builder.create_constant_in(2, 1);

    // Layer 1
    let (_, inners) = circuit_builder.create_witness_out(2);
    circuit_builder.mul2(inners[0], leaves1[0], leaves1[1], Ext::BaseField::ONE);
    circuit_builder.mul2(inners[1], leaves1[2], leaves2[0], Ext::BaseField::ONE);

    // Layer 0
    let (_, root) = circuit_builder.create_witness_out(1);
    circuit_builder.mul2(root[0], inners[0], inners[1], Ext::BaseField::ONE);

    circuit_builder.configure();

    Circuit::new(&circuit_builder)
}

fn paste_from_wit_in_witness<'a, Ext: ExtensionField>()
-> (Vec<DenseMultilinearExtension<Ext>>, CircuitWitness<'a, Ext>) {
    // witness_in, single instance
    let leaves1 = vec![vec![i64_to_field(5), i64_to_field(7), i64_to_field(11)]];
    let leaves2 = vec![vec![i64_to_field(13), i64_to_field(17), i64_to_field(19)]];
    let dummy = vec![vec![i64_to_field(13), i64_to_field(17), i64_to_field(19)]];
    let witness_in = vec![leaves1.into(), leaves2.into(), dummy.into()];

    let layers: Vec<DenseMultilinearExtension<_>> = vec![
        vec![vec![
            i64_to_field(5005),
            i64_to_field(35),
            i64_to_field(143),
            i64_to_field(0), // pad
        ]]
        .into(),
        vec![vec![i64_to_field(35), i64_to_field(143)]].into(),
        vec![vec![
            i64_to_field(5), // leaves1
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13), // leaves2
            i64_to_field(17),
            i64_to_field(19),
            i64_to_field(13), // dummy
            i64_to_field(17),
            i64_to_field(19),
            i64_to_field(0), // counter
            i64_to_field(1),
            i64_to_field(1), // constant
            i64_to_field(1),
            i64_to_field(0), // pad
            i64_to_field(0),
            i64_to_field(0),
        ]]
        .into(),
    ];

    let outputs1 = vec![vec![i64_to_field(35), i64_to_field(143)]];
    let outputs2 = vec![vec![i64_to_field(5005)]];
    let witness_out: Vec<DenseMultilinearExtension<Ext>> = vec![outputs1.into(), outputs2.into()];

    (
        witness_in.clone(),
        CircuitWitness {
            layers: layers.into_iter().map(|w| w.into()).collect(),
            witness_in: witness_in.into_iter().map(|w| w.into()).collect(),
            witness_out: witness_out.into_iter().map(|w| w.into()).collect(),
            n_instances: 1,
            challenges: HashMap::new(),
        },
    )
}

fn copy_to_wit_out_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();
    // Layer 2
    let (_, leaves) = circuit_builder.create_witness_in(4);

    // Layer 1
    let (_inner_id, inners) = circuit_builder.create_witness_out(2);
    circuit_builder.mul2(inners[0], leaves[0], leaves[1], Ext::BaseField::ONE);
    circuit_builder.mul2(inners[1], leaves[2], leaves[3], Ext::BaseField::ONE);

    // Layer 0
    let root = circuit_builder.create_cell();
    circuit_builder.mul2(root, inners[0], inners[1], Ext::BaseField::ONE);
    circuit_builder.assert_const(root, 5005);

    circuit_builder.configure();

    Circuit::new(&circuit_builder)
}

fn copy_to_wit_out_witness<'a, Ext: ExtensionField>()
-> (Vec<DenseMultilinearExtension<Ext>>, CircuitWitness<'a, Ext>) {
    // witness_in, single instance
    let leaves = vec![vec![
        i64_to_field(5),
        i64_to_field(7),
        i64_to_field(11),
        i64_to_field(13),
    ]]
    .into();
    let witness_in = vec![leaves];

    let layers: Vec<DenseMultilinearExtension<Ext>> = vec![
        vec![vec![
            i64_to_field(5005),
            i64_to_field(35),
            i64_to_field(143),
            i64_to_field(0), // pad
        ]]
        .into(),
        vec![vec![i64_to_field(35), i64_to_field(143)]].into(),
        vec![vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
        ]]
        .into(),
    ];

    let outputs = vec![vec![i64_to_field(35), i64_to_field(143)]];
    let witness_out: Vec<DenseMultilinearExtension<Ext>> = vec![outputs.into()];

    (
        witness_in.clone(),
        CircuitWitness {
            layers: layers.into_iter().map(|w| w.into()).collect(),
            witness_in: witness_in.into_iter().map(|w| w.into()).collect(),
            witness_out: witness_out.into_iter().map(|w| w.into()).collect(),
            n_instances: 1,
            challenges: HashMap::new(),
        },
    )
}

fn copy_to_wit_out_witness_2<'a, Ext: ExtensionField>()
-> (Vec<DenseMultilinearExtension<Ext>>, CircuitWitness<'a, Ext>) {
    // witness_in, 2 instances
    let leaves = vec![
        vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
        ],
        vec![
            i64_to_field(5),
            i64_to_field(13),
            i64_to_field(11),
            i64_to_field(7),
        ],
    ];
    let witness_in = vec![leaves.into()];

    let layers: Vec<DenseMultilinearExtension<Ext>> = vec![
        vec![
            vec![
                i64_to_field(5005),
                i64_to_field(35),
                i64_to_field(143),
                i64_to_field(0), // pad
            ],
            vec![
                i64_to_field(5005),
                i64_to_field(65),
                i64_to_field(77),
                i64_to_field(0), // pad
            ],
        ]
        .into(),
        vec![
            vec![i64_to_field(35), i64_to_field(143)],
            vec![i64_to_field(65), i64_to_field(77)],
        ]
        .into(),
        vec![
            vec![
                i64_to_field(5),
                i64_to_field(7),
                i64_to_field(11),
                i64_to_field(13),
            ],
            vec![
                i64_to_field(5),
                i64_to_field(13),
                i64_to_field(11),
                i64_to_field(7),
            ],
        ]
        .into(),
    ];

    let outputs = vec![
        vec![i64_to_field(35), i64_to_field(143)],
        vec![i64_to_field(65), i64_to_field(77)],
    ];
    let witness_out: Vec<DenseMultilinearExtension<Ext>> = vec![outputs.into()];

    (
        witness_in.clone(),
        CircuitWitness {
            layers: layers.into_iter().map(|w| w.into()).collect(),
            witness_in: witness_in.into_iter().map(|w| w.into()).collect(),
            witness_out: witness_out.into_iter().map(|w| w.into()).collect(),
            n_instances: 2,
            challenges: HashMap::new(),
        },
    )
}

fn rlc_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();
    // Layer 2
    let (_, leaves) = circuit_builder.create_witness_in(4);

    // Layer 1
    let inners = circuit_builder.create_ext_cells(2);
    circuit_builder.rlc(&inners[0], &[leaves[0], leaves[1]], 0 as ChallengeId);
    circuit_builder.rlc(&inners[1], &[leaves[2], leaves[3]], 1 as ChallengeId);

    // Layer 0
    let (_root_id, roots) = circuit_builder.create_ext_witness_out(1);
    circuit_builder.mul2_ext(&roots[0], &inners[0], &inners[1], Ext::BaseField::ONE);

    circuit_builder.configure();

    Circuit::new(&circuit_builder)
}

fn rlc_witness<'a, Ext>() -> (
    Vec<DenseMultilinearExtension<Ext>>,
    CircuitWitness<'a, Ext>,
    Vec<Ext>,
)
where
    Ext: ExtensionField<DEGREE = 2>,
{
    let challenges = vec![
        Ext::from_bases(&[i64_to_field(31), i64_to_field(37)]),
        Ext::from_bases(&[i64_to_field(97), i64_to_field(23)]),
    ];
    let challenge_pows = challenges
        .iter()
        .enumerate()
        .map(|(i, x)| {
            (0..3)
                .map(|j| {
                    (
                        ChallengeConst {
                            challenge: i as u8,
                            exp: j as u64,
                        },
                        x.exp_u64(j as u64),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    // witness_in, double instances
    let leaves = vec![
        vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
        ],
        vec![
            i64_to_field(5),
            i64_to_field(13),
            i64_to_field(11),
            i64_to_field(7),
        ],
    ];
    let witness_in = vec![leaves.clone().into()];

    let inner00: Ext = challenge_pows[0][0].1 * leaves[0][0]
        + challenge_pows[0][1].1 * leaves[0][1]
        + challenge_pows[0][2].1;
    let inner01: Ext = challenge_pows[1][0].1 * leaves[0][2]
        + challenge_pows[1][1].1 * leaves[0][3]
        + challenge_pows[1][2].1;
    let inner10: Ext = challenge_pows[0][0].1 * leaves[1][0]
        + challenge_pows[0][1].1 * leaves[1][1]
        + challenge_pows[0][2].1;
    let inner11: Ext = challenge_pows[1][0].1 * leaves[1][2]
        + challenge_pows[1][1].1 * leaves[1][3]
        + challenge_pows[1][2].1;

    let inners = vec![
        [inner00.clone().as_bases(), inner01.clone().as_bases()].concat(),
        [inner10.clone().as_bases(), inner11.clone().as_bases()].concat(),
    ];

    let root_tmp0 = vec![
        inners[0][0] * inners[0][2],
        inners[0][0] * inners[0][3],
        inners[0][1] * inners[0][2],
        inners[0][1] * inners[0][3],
    ];
    let root_tmp1 = vec![
        inners[1][0] * inners[1][2],
        inners[1][0] * inners[1][3],
        inners[1][1] * inners[1][2],
        inners[1][1] * inners[1][3],
    ];
    let root_tmps = vec![root_tmp0, root_tmp1];

    let root0 = inner00 * inner01;
    let root1 = inner10 * inner11;
    let roots = vec![
        root0.as_bases().iter().cloned().collect_vec(),
        root1.as_bases().iter().cloned().collect_vec(),
    ];

    let layers: Vec<DenseMultilinearExtension<Ext>> = vec![
        roots.clone().into(),
        root_tmps.into(),
        inners.into(),
        leaves.into(),
    ];

    let outputs = roots;
    let witness_out: Vec<DenseMultilinearExtension<Ext>> = vec![outputs.into()];

    (
        witness_in.clone(),
        CircuitWitness {
            layers: layers.into_iter().map(|w| w.into()).collect(),
            witness_in: witness_in.into_iter().map(|w| w.into()).collect(),
            witness_out: witness_out.into_iter().map(|w| w.into()).collect(),
            n_instances: 2,
            challenges: challenge_pows
                .iter()
                .flatten()
                .cloned()
                .map(|(k, v)| (k, v.as_bases().to_vec()))
                .collect::<HashMap<_, _>>(),
        },
        challenges,
    )
}

fn inv_sum_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();
    let (_input_id, input) = circuit_builder.create_ext_witness_in(2);
    let (_cond_id, cond) = circuit_builder.create_witness_in(2);
    let (_, output) = circuit_builder.create_ext_witness_out(2);
    // selector denominator 1 or input[0] or input[0] * input[1]
    let den_mul = circuit_builder.create_ext_cell();
    circuit_builder.mul2_ext(&den_mul, &input[0], &input[1], Ext::BaseField::ONE);
    let tmp = circuit_builder.create_ext_cell();
    circuit_builder.sel_mixed_and_ext(
        &tmp,
        &MixedCell::Constant(Ext::BaseField::ONE),
        &input[0],
        cond[0],
    );
    circuit_builder.sel_ext(&output[0], &tmp, &den_mul, cond[1]);

    // select the numerator 0 or 1 or input[0] + input[1]
    let den_add = circuit_builder.create_ext_cell();
    circuit_builder.add_ext(&den_add, &input[0], Ext::BaseField::ONE);
    circuit_builder.add_ext(&den_add, &input[1], Ext::BaseField::ONE);
    circuit_builder.sel_mixed_and_ext(&output[1], &cond[0].into(), &den_add, cond[1]);

    circuit_builder.configure();
    Circuit::new(&circuit_builder)
}

fn inv_sum_witness_4_instances<'a, Ext: ExtensionField>() -> CircuitWitness<'a, Ext> {
    let circuit = inv_sum_circuit::<Ext>();
    // witness_in, double instances
    let leaves = vec![
        vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
        ],
        vec![
            i64_to_field(5),
            i64_to_field(13),
            i64_to_field(11),
            i64_to_field(7),
        ],
        vec![
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(17),
            i64_to_field(19),
        ],
        vec![
            i64_to_field(29),
            i64_to_field(17),
            i64_to_field(19),
            i64_to_field(23),
        ],
    ];
    let cond: Vec<Vec<<Ext as ExtensionField>::BaseField>> = vec![
        vec![i64_to_field(1), i64_to_field(1)],
        vec![i64_to_field(1), i64_to_field(1)],
        vec![i64_to_field(1), i64_to_field(1)],
        vec![i64_to_field(0), i64_to_field(0)],
    ];
    let witness_in = vec![leaves.into(), cond.into()];
    let mut circuit_wits = CircuitWitness::new(&circuit, vec![]);
    circuit_wits.add_instances(&circuit, witness_in, 4);
    circuit_wits
}

fn lookup_inner_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();

    // Layer 2
    let (_, input) = circuit_builder.create_ext_witness_in(4);
    // Layer 0
    let output = circuit_builder.create_ext_cells(2);
    // denominator
    circuit_builder.mul2_ext(
        &output[0], // output_den
        &input[0],  // input_den[0]
        &input[2],  // input_den[1]
        Ext::BaseField::ONE,
    );

    // numerator
    circuit_builder.mul2_ext(
        &output[1], // output_num
        &input[0],  // input_den[0]
        &input[3],  // input_num[1]
        Ext::BaseField::ONE,
    );
    circuit_builder.mul2_ext(
        &output[1], // output_num
        &input[2],  // input_den[1]
        &input[1],  // input_num[0]
        Ext::BaseField::ONE,
    );

    circuit_builder.configure();
    Circuit::new(&circuit_builder)
}

fn lookup_inner_witness_4_instances<'a, Ext: ExtensionField>() -> CircuitWitness<'a, Ext> {
    let circuit = lookup_inner_circuit::<Ext>();
    // witness_in, double instances
    let leaves = vec![
        vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
            i64_to_field(17),
            i64_to_field(19),
            i64_to_field(23),
            i64_to_field(29),
        ],
        vec![
            i64_to_field(5),
            i64_to_field(13),
            i64_to_field(11),
            i64_to_field(7),
            i64_to_field(19),
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(17),
        ],
        vec![
            i64_to_field(13),
            i64_to_field(11),
            i64_to_field(7),
            i64_to_field(5),
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(17),
            i64_to_field(19),
        ],
        vec![
            i64_to_field(11),
            i64_to_field(7),
            i64_to_field(13),
            i64_to_field(5),
            i64_to_field(29),
            i64_to_field(17),
            i64_to_field(19),
            i64_to_field(23),
        ],
    ];
    let witness_in = vec![leaves.into()];
    let mut circuit_wits = CircuitWitness::new(&circuit, vec![]);
    circuit_wits.add_instances(&circuit, witness_in, 4);
    circuit_wits
}

fn mixed_in_circuit<Ext: ExtensionField>() -> Circuit<Ext> {
    let mut circuit_builder = CircuitBuilder::<Ext>::default();

    // Layer 1
    let (_, _input) = circuit_builder.create_witness_in(5);
    let (_, _input_ext) = circuit_builder.create_ext_witness_in(3);
    let _input_const1 = circuit_builder.create_constant_in(2, 11);
    let _input_counter = circuit_builder.create_counter_in(1);
    let _input_const2 = circuit_builder.create_constant_in(2, 17);

    circuit_builder.configure();
    Circuit::new(&circuit_builder)
}

fn mixed_in_witness_4_instances<'a, Ext: ExtensionField>() -> CircuitWitness<'a, Ext> {
    let circuit = mixed_in_circuit::<Ext>();
    // witness_in, double instances
    let input = vec![
        vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(13),
            i64_to_field(17),
        ],
        vec![
            i64_to_field(13),
            i64_to_field(11),
            i64_to_field(7),
            i64_to_field(19),
            i64_to_field(11),
        ],
        vec![
            i64_to_field(7),
            i64_to_field(5),
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(41),
        ],
        vec![
            i64_to_field(29),
            i64_to_field(17),
            i64_to_field(97),
            i64_to_field(19),
            i64_to_field(23),
        ],
    ];
    let input_ext = vec![
        vec![
            i64_to_field(5),
            i64_to_field(7),
            i64_to_field(11),
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(31),
        ],
        vec![
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(31),
            i64_to_field(13),
            i64_to_field(17),
            i64_to_field(19),
        ],
        vec![
            i64_to_field(31),
            i64_to_field(13),
            i64_to_field(17),
            i64_to_field(23),
            i64_to_field(29),
            i64_to_field(31),
        ],
        vec![
            i64_to_field(13),
            i64_to_field(17),
            i64_to_field(19),
            i64_to_field(37),
            i64_to_field(41),
            i64_to_field(43),
        ],
    ];
    let witness_in = vec![input.into(), input_ext.into()];
    let mut circuit_wits = CircuitWitness::new(&circuit, vec![]);
    circuit_wits.add_instances(&circuit, witness_in, 4);
    circuit_wits
}

fn prove_and_verify<Ext: ExtensionField>(
    circuit: Circuit<Ext>,
    circuit_wits: CircuitWitness<'_, Ext>,
    challenges: Vec<Ext>,
) {
    let mut rng = test_rng();
    println!(
        "circuit_wits.instance_num_vars() {}, circuit.output_num_vars() {}",
        circuit_wits.instance_num_vars(),
        circuit.output_num_vars()
    );
    let out_num_vars = circuit.output_num_vars() + circuit_wits.instance_num_vars();
    let out_point = (0..out_num_vars)
        .map(|_| Ext::random(&mut rng))
        .collect_vec();

    let out_point_and_evals = if circuit.n_witness_out == 0 {
        vec![PointAndEval::new(
            out_point.clone(),
            circuit_wits.output_layer_witness_ref().evaluate(&out_point),
        )]
    } else {
        vec![]
    };
    let wit_out_point_and_evals = circuit_wits
        .witness_out_ref()
        .iter()
        .map(|wit| {
            println!("wit {:?}", wit.evaluations());
            PointAndEval::new(
                out_point[..wit.num_vars()].to_vec(),
                wit.evaluate(&out_point[..wit.num_vars()]),
            )
        })
        .collect_vec();

    let mut prover_transcript = BasicTranscript::new(b"transcript");
    let (proof, prover_input_claim) = IOPProverState::prove_parallel(
        &circuit,
        &circuit_wits,
        out_point_and_evals.clone(),
        wit_out_point_and_evals.clone(),
        1,
        &mut prover_transcript,
    );

    let mut verifier_transcript = BasicTranscript::new(b"transcript");
    let verifier_input_claim = IOPVerifierState::verify_parallel(
        &circuit,
        &challenges,
        out_point_and_evals,
        wit_out_point_and_evals,
        proof,
        circuit_wits.instance_num_vars(),
        &mut verifier_transcript,
    )
    .expect("Verification failed");

    assert!(
        !izip!(
            prover_input_claim.point_and_evals.iter(),
            verifier_input_claim.point_and_evals.iter()
        )
        .any(|(p, v)| p.point != v.point || p.eval != v.eval)
    );
    assert!(
        !izip!(
            circuit_wits.witness_in.iter(),
            prover_input_claim.point_and_evals.iter()
        )
        .any(|(wit, p)| wit.evaluate(&p.point) != p.eval)
    );
}

#[test]
fn test_copy_and_paste() {
    let circuit = copy_and_paste_circuit::<GoldilocksExt2>();
    let (_, circuit_wits) = copy_and_paste_witness::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}

#[test]
fn test_paste_from_wit_in() {
    let circuit = paste_from_wit_in_circuit::<GoldilocksExt2>();
    let (_, circuit_wits) = paste_from_wit_in_witness::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}

#[test]
fn test_copy_to_wit_out() {
    let circuit = copy_to_wit_out_circuit::<GoldilocksExt2>();
    let (_, circuit_wits) = copy_to_wit_out_witness::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}

#[test]
fn test_copy_to_wit_out_2_instances() {
    let circuit = copy_to_wit_out_circuit::<GoldilocksExt2>();
    let (_, circuit_wits) = copy_to_wit_out_witness_2::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}

#[test]
fn test_challenges() {
    let circuit = rlc_circuit::<GoldilocksExt2>();
    let (_, circuit_wits, challenges) = rlc_witness::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, challenges);
}

#[test]
fn test_inv_sum() {
    let circuit = inv_sum_circuit::<GoldilocksExt2>();
    let circuit_wits = inv_sum_witness_4_instances::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}

#[test]
fn test_lookup_inner_output_eval() {
    let circuit = lookup_inner_circuit::<GoldilocksExt2>();
    let circuit_wits = lookup_inner_witness_4_instances::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}

#[test]
fn test_mixed_in() {
    let circuit = mixed_in_circuit::<GoldilocksExt2>();
    let circuit_wits = mixed_in_witness_4_instances::<GoldilocksExt2>();
    prove_and_verify(circuit, circuit_wits, vec![]);
}
