use core::{fmt, panic};
use std::collections::{BTreeMap, HashMap};

use ff_ext::ExtensionField;
use itertools::Itertools;
use simple_frontend::structs::{
    CellId, CellType, ChallengeConst, CircuitBuilder, ConstantType, InType, LayerId, OutType,
};
use sumcheck::util::ceil_log2;

use crate::{
    structs::{Circuit, Gate1In, Gate2In, Gate3In, GateCIn, Layer, SumcheckStepType},
    utils::{MatrixMLEColumnFirst, MatrixMLERowFirst, i64_to_field},
};

struct LayerSubsets {
    subsets: BTreeMap<(u32, usize), usize>,
    layer_id: LayerId,
    wire_id_assigner: CellId,
}

impl LayerSubsets {
    fn new(layer_id: LayerId, layer_size: usize) -> Self {
        Self {
            subsets: BTreeMap::new(),
            layer_id,
            wire_id_assigner: layer_size,
        }
    }

    /// Compute the new `wire_id` after copy the cell from the old layer to the
    /// new layer. If old layer == new layer, return the old wire_id.
    fn update_wire_id(&mut self, old_layer_id: LayerId, old_wire_id: CellId) -> CellId {
        if old_layer_id == self.layer_id {
            return old_wire_id;
        }
        if let std::collections::btree_map::Entry::Vacant(e) =
            self.subsets.entry((old_layer_id, old_wire_id))
        {
            e.insert(self.wire_id_assigner);
            self.wire_id_assigner += 1;
        }
        self.subsets[&(old_layer_id, old_wire_id)]
    }

    /// Compute `paste_from` matrix and `max_previous_num_vars` for
    /// `self.layer_id`, as well as `copy_to` for old layers.
    fn update_layer_info<Ext: ExtensionField>(&self, layers: &mut [Layer<Ext>]) {
        let mut paste_from = BTreeMap::new();
        for ((old_layer_id, old_wire_id), new_wire_id) in self.subsets.iter() {
            paste_from
                .entry(*old_layer_id)
                .or_insert(vec![])
                .push(*new_wire_id);
            layers[*old_layer_id as usize]
                .copy_to
                .entry(self.layer_id)
                .or_default()
                .push(*old_wire_id);
        }
        layers[self.layer_id as usize].paste_from = paste_from;

        layers[self.layer_id as usize].num_vars = ceil_log2(self.wire_id_assigner);
        layers[self.layer_id as usize].max_previous_num_vars = layers[self.layer_id as usize]
            .max_previous_num_vars
            .max(ceil_log2(
                layers[self.layer_id as usize]
                    .paste_from
                    .values()
                    .map(|old_wire_ids| old_wire_ids.len())
                    .max()
                    .unwrap_or(1),
            ));
    }
}

impl<E: ExtensionField> Circuit<E> {
    /// Generate the circuit from circuit builder.
    pub fn new(circuit_builder: &CircuitBuilder<E>) -> Self {
        assert!(circuit_builder.n_layers.is_some());
        let n_layers = circuit_builder.n_layers.unwrap();

        // ==================================
        // Put cells into layers. Maintain two vectors:
        // - `layers_of_cell_id` stores all cell ids in each layer;
        // - `wire_ids_in_layer` stores the wire id of each cell in its layer.
        // ==================================
        let (layers_of_cell_id, wire_ids_in_layer) = {
            let mut layers_of_cell_id = vec![vec![]; n_layers as usize];
            let mut wire_ids_in_layer = vec![0; circuit_builder.cells.len()];
            for (i, cell) in circuit_builder.cells.iter().enumerate() {
                // If layer isn't assigned, then the cell is not in the circuit.
                if let Some(layer) = cell.layer {
                    wire_ids_in_layer[i] = layers_of_cell_id[layer as usize].len();
                    layers_of_cell_id[layer as usize].push(i);
                }
            }
            (layers_of_cell_id, wire_ids_in_layer)
        };

        let mut layers = vec![Layer::default(); n_layers as usize];

        // ==================================
        // From the input layer to the output layer, construct the gates. If a
        // gate has the input from multiple previous layers, then we need to
        // copy them to the current layer.
        // ==================================

        // Input layer if pasted from wires_in and constant.
        let in_cell_ids = {
            let mut in_cell_ids = BTreeMap::new();
            for (cell_id, cell) in circuit_builder.cells.iter().enumerate() {
                if let Some(CellType::In(in_type)) = cell.cell_type {
                    in_cell_ids.entry(in_type).or_insert(vec![]).push(cell_id);
                }
            }
            in_cell_ids
        };

        let mut input_paste_from_wits_in = vec![(0, 0); circuit_builder.n_witness_in()];
        let mut input_paste_from_counter_in = Vec::new();
        let mut input_paste_from_consts_in = Vec::new();
        let mut max_in_wit_num_vars: Option<usize> = None;
        for (ty, in_cell_ids) in in_cell_ids.iter() {
            let segment = (
                wire_ids_in_layer[in_cell_ids[0]],
                wire_ids_in_layer[in_cell_ids[in_cell_ids.len() - 1]] + 1, /* + 1 for exclusive
                                                                            *   last index */
            );
            match ty {
                InType::Witness(wit_id) => {
                    input_paste_from_wits_in[*wit_id as usize] = segment;
                    max_in_wit_num_vars = max_in_wit_num_vars
                        .map_or(Some(ceil_log2(in_cell_ids.len())), |x| {
                            Some(x.max(ceil_log2(in_cell_ids.len())))
                        });
                }
                InType::Counter(num_vars) => {
                    input_paste_from_counter_in.push((*num_vars, segment));
                    max_in_wit_num_vars = max_in_wit_num_vars
                        .map_or(Some(ceil_log2(in_cell_ids.len())), |x| {
                            Some(x.max(ceil_log2(in_cell_ids.len())))
                        });
                }
                InType::Constant(constant) => {
                    input_paste_from_consts_in.push((*constant, segment));
                }
            }
        }

        layers[n_layers as usize - 1].layer_id = n_layers - 1;
        for layer_id in (0..n_layers - 1).rev() {
            layers[layer_id as usize].layer_id = layer_id;
            // current_subsets: (old_layer_id, old_wire_id) -> new_wire_id
            // It only stores the wires not in the current layer.
            let new_layer_id = layer_id + 1;
            let mut subsets = LayerSubsets::new(
                new_layer_id,
                layers_of_cell_id[new_layer_id as usize]
                    .len()
                    .next_power_of_two(),
            );

            for (i, cell_id) in layers_of_cell_id[layer_id as usize].iter().enumerate() {
                let cell = &circuit_builder.cells[*cell_id];
                for gate in cell.gates.iter() {
                    let idx_in = gate
                        .idx_in
                        .iter()
                        .map(|&cell_id| {
                            let old_layer_id = circuit_builder.cells[cell_id].layer.unwrap();
                            let old_wire_id = wire_ids_in_layer[cell_id];
                            subsets.update_wire_id(old_layer_id, old_wire_id)
                        })
                        .collect_vec();
                    match idx_in.len() {
                        0 => layers[layer_id as usize].add_consts.push(GateCIn {
                            idx_in: [],
                            idx_out: i,
                            scalar: gate.scalar,
                        }),
                        1 => {
                            let gate = Gate1In {
                                idx_in: idx_in.clone().try_into().unwrap(),
                                idx_out: i,
                                scalar: gate.scalar,
                            };
                            layers[layer_id as usize].adds.push(gate.clone());
                            for (i, idx_in) in idx_in.iter().enumerate() {
                                layers[layer_id as usize].adds_fanin_mapping[i]
                                    .entry(*idx_in)
                                    .or_insert(vec![])
                                    .push(gate.clone());
                            }
                        }
                        2 => {
                            let gate = Gate2In {
                                idx_in: idx_in.clone().try_into().unwrap(),
                                idx_out: i,
                                scalar: gate.scalar,
                            };
                            layers[layer_id as usize].mul2s.push(gate.clone());
                            for (i, idx_in) in idx_in.iter().enumerate() {
                                layers[layer_id as usize].mul2s_fanin_mapping[i]
                                    .entry(*idx_in)
                                    .or_insert(vec![])
                                    .push(gate.clone());
                            }
                        }
                        3 => {
                            let gate = Gate3In {
                                idx_in: idx_in.clone().try_into().unwrap(),
                                idx_out: i,
                                scalar: gate.scalar,
                            };
                            layers[layer_id as usize].mul3s.push(gate.clone());
                            for (i, idx_in) in idx_in.iter().enumerate() {
                                layers[layer_id as usize].mul3s_fanin_mapping[i]
                                    .entry(*idx_in)
                                    .or_insert(vec![])
                                    .push(gate.clone());
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }

            subsets.update_layer_info(&mut layers);
            // Initialize the next layer `max_previous_num_vars` equals that of the `self.layer_id`.
            layers[layer_id as usize].max_previous_num_vars =
                layers[new_layer_id as usize].num_vars;
        }

        // Compute the copy_to from the output layer to the wires_out. Notice
        // that we don't pad the original output layer elements to the power of
        // two.
        let mut output_subsets = LayerSubsets::new(0, layers_of_cell_id[0].len());
        let mut output_copy_to_wits_out = vec![vec![]; circuit_builder.n_witness_out()];
        let mut output_assert_const = vec![];
        for (cell_id, cell) in circuit_builder.cells.iter().enumerate() {
            if let Some(CellType::Out(out)) = cell.cell_type {
                if cell.layer.is_none() {
                    panic!("Output cell: {:?} should have a layer.", (cell_id, cell));
                }
                let old_layer_id = cell.layer.unwrap();
                let old_wire_id = wire_ids_in_layer[cell_id];
                match out {
                    OutType::Witness(wit_id) => {
                        output_copy_to_wits_out[wit_id as usize]
                            .push(output_subsets.update_wire_id(old_layer_id, old_wire_id));
                    }
                    OutType::AssertConst(constant) => {
                        let new_wire_id = output_subsets.update_wire_id(old_layer_id, old_wire_id);
                        output_assert_const.push(GateCIn {
                            idx_in: [],
                            idx_out: new_wire_id,
                            scalar: ConstantType::Field(i64_to_field(constant)),
                        });
                    }
                }
            }
        }
        let output_copy_to = output_copy_to_wits_out.into_iter().collect_vec();
        output_subsets.update_layer_info(&mut layers);

        // Update sumcheck steps
        (0..n_layers).for_each(|layer_id| {
            let mut curr_sc_steps = vec![];
            let layer = &layers[layer_id as usize];
            if layer.layer_id == 0 {
                let seg = (0..1 << layer.num_vars).collect_vec();
                if circuit_builder.n_witness_out() > 1
                    || circuit_builder.n_witness_out() == 1 && output_copy_to[0] != seg
                    || !output_assert_const.is_empty()
                {
                    curr_sc_steps.extend([SumcheckStepType::OutputPhase1Step1]);
                }
            } else {
                let last_layer = &layers[(layer_id - 1) as usize];
                if !last_layer.is_linear() || !layer.copy_to.is_empty() {
                    curr_sc_steps.extend([SumcheckStepType::Phase1Step1]);
                }
            }

            if layer.layer_id == n_layers - 1 {
                if input_paste_from_wits_in.len() > 1
                    || input_paste_from_wits_in.len() == 1
                        && input_paste_from_wits_in[0] != (0, 1 << layer.num_vars)
                    || !input_paste_from_counter_in.is_empty()
                    || !input_paste_from_consts_in.is_empty()
                {
                    curr_sc_steps.push(SumcheckStepType::InputPhase2Step1);
                }
            } else if layer.is_linear() {
                curr_sc_steps.push(SumcheckStepType::LinearPhase2Step1);
            } else {
                curr_sc_steps.push(SumcheckStepType::Phase2Step1);
                if !layer.mul2s.is_empty() || !layer.mul3s.is_empty() {
                    if layer.mul3s.is_empty() {
                        curr_sc_steps.push(SumcheckStepType::Phase2Step2NoStep3);
                    } else {
                        curr_sc_steps.push(SumcheckStepType::Phase2Step2);
                    }
                }
                if !layer.mul3s.is_empty() {
                    curr_sc_steps.push(SumcheckStepType::Phase2Step3);
                }
            }
            layers[layer_id as usize].sumcheck_steps = curr_sc_steps;
        });

        Self {
            layers,
            n_witness_in: circuit_builder.n_witness_in(),
            n_witness_out: circuit_builder.n_witness_out(),
            paste_from_wits_in: input_paste_from_wits_in,
            paste_from_counter_in: input_paste_from_counter_in,
            paste_from_consts_in: input_paste_from_consts_in,
            copy_to_wits_out: output_copy_to,
            assert_consts: output_assert_const,
            max_wit_in_num_vars: max_in_wit_num_vars,
        }
    }

    pub(crate) fn generate_basefield_challenges(
        &self,
        challenges: &[E],
    ) -> HashMap<ChallengeConst, Vec<E::BaseField>> {
        let mut challenge_exps = HashMap::<ChallengeConst, E>::new();
        let mut update_const = |constant| match constant {
            ConstantType::Challenge(c, _) => {
                challenge_exps
                    .entry(c)
                    .or_insert(challenges[c.challenge as usize].exp_u64(c.exp));
            }
            ConstantType::ChallengeScaled(c, _, _) => {
                challenge_exps
                    .entry(c)
                    .or_insert(challenges[c.challenge as usize].exp_u64(c.exp));
            }
            _ => {}
        };
        self.layers.iter().for_each(|layer| {
            layer
                .add_consts
                .iter()
                .for_each(|gate| update_const(gate.scalar));
            layer.adds.iter().for_each(|gate| update_const(gate.scalar));
            layer
                .mul2s
                .iter()
                .for_each(|gate| update_const(gate.scalar));
            layer
                .mul3s
                .iter()
                .for_each(|gate| update_const(gate.scalar));
        });
        challenge_exps
            .into_iter()
            .map(|(k, v)| (k, v.as_bases().to_vec()))
            .collect()
    }

    pub fn output_layer_ref(&self) -> &Layer<E> {
        self.layers.first().unwrap()
    }

    pub fn first_layer_ref(&self) -> &Layer<E> {
        self.layers.last().unwrap()
    }

    pub fn output_num_vars(&self) -> usize {
        self.output_layer_ref().num_vars
    }

    pub fn output_size(&self) -> usize {
        1 << self.output_layer_ref().num_vars
    }

    pub fn is_input_layer(&self, layer_id: LayerId) -> bool {
        layer_id as usize == self.layers.len() - 1
    }

    pub fn is_output_layer(&self, layer_id: LayerId) -> bool {
        layer_id == 0
    }
}

impl<E: ExtensionField> Layer<E> {
    pub fn is_linear(&self) -> bool {
        self.mul2s.is_empty() && self.mul3s.is_empty()
    }

    pub fn size(&self) -> usize {
        1 << self.num_vars
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn max_previous_num_vars(&self) -> usize {
        self.max_previous_num_vars
    }

    pub fn max_previous_size(&self) -> usize {
        1 << self.max_previous_num_vars
    }

    pub fn paste_from_fix_variables_eq(
        &self,
        old_layer_id: LayerId,
        current_point_eq: &[E],
    ) -> Vec<E> {
        assert_eq!(current_point_eq.len(), self.size());
        self.paste_from
            .get(&old_layer_id)
            .unwrap()
            .as_slice()
            .fix_row_col_first(current_point_eq, self.max_previous_num_vars)
    }

    pub fn paste_from_eval_eq(
        &self,
        old_layer_id: LayerId,
        current_point_eq: &[E],
        subset_point_eq: &[E],
    ) -> E {
        assert_eq!(current_point_eq.len(), self.size());
        assert_eq!(subset_point_eq.len(), self.max_previous_size());
        self.paste_from
            .get(&old_layer_id)
            .unwrap()
            .as_slice()
            .eval_col_first(current_point_eq, subset_point_eq)
    }

    pub fn copy_to_fix_variables(&self, new_layer_id: LayerId, subset_point_eq: &[E]) -> Vec<E> {
        let old_wire_ids = self.copy_to.get(&new_layer_id).unwrap();
        old_wire_ids
            .as_slice()
            .fix_row_row_first(subset_point_eq, self.num_vars)
    }

    pub fn copy_to_eval_eq(
        &self,
        new_layer_id: LayerId,
        subset_point_eq: &[E],
        current_point_eq: &[E],
    ) -> E {
        self.copy_to
            .get(&new_layer_id)
            .unwrap()
            .as_slice()
            .eval_row_first(subset_point_eq, current_point_eq)
    }
}

impl<E: ExtensionField> fmt::Debug for Layer<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Layer {{")?;
        writeln!(f, "  layer_id: {}", self.layer_id)?;
        writeln!(f, "  num_vars: {}", self.num_vars)?;
        writeln!(f, "  max_previous_num_vars: {}", self.max_previous_num_vars)?;
        writeln!(f, "  add_consts: ")?;
        for add_const in self.add_consts.iter() {
            writeln!(f, "    {add_const:?}")?;
        }
        writeln!(f, "  adds: ")?;
        for add in self.adds.iter() {
            writeln!(f, "    {add:?}")?;
        }
        writeln!(f, "  mul2s: ")?;
        for mul2 in self.mul2s.iter() {
            writeln!(f, "    {mul2:?}")?;
        }
        writeln!(f, "  mul3s: ")?;
        for mul3 in self.mul3s.iter() {
            writeln!(f, "    {mul3:?}")?;
        }
        writeln!(f, "  copy_to: {:?}", self.copy_to)?;
        writeln!(f, "  paste_from: {:?}", self.paste_from)?;
        writeln!(f, "}}")
    }
}

impl<E: ExtensionField> fmt::Debug for Circuit<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Circuit {{")?;
        writeln!(f, "  layers: ")?;
        for layer in self.layers.iter() {
            writeln!(f, "    {layer:?}")?;
        }
        writeln!(f, "  n_witness_in: {}", self.n_witness_in)?;
        writeln!(f, "  paste_from_wits_in: {:?}", self.paste_from_wits_in)?;
        writeln!(
            f,
            "  paste_from_counter_in: {:?}",
            self.paste_from_counter_in
        )?;
        writeln!(f, "  paste_from_consts_in: {:?}", self.paste_from_consts_in)?;
        writeln!(f, "  copy_to_wits_out: {:?}", self.copy_to_wits_out)?;
        writeln!(f, "  assert_const: {:?}", self.assert_consts)?;
        writeln!(f, "  max_wires_in_num_vars: {:?}", self.max_wit_in_num_vars)?;
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ff_ext::GoldilocksExt2;
    use p3_field::FieldAlgebra;
    use p3_goldilocks::Goldilocks;
    use simple_frontend::structs::{ChallengeConst, ChallengeId, CircuitBuilder, ConstantType};

    use crate::structs::{Circuit, Gate, GateCIn, SumcheckStepType};

    #[test]
    fn test_copy_and_paste() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        // Layer 3
        let (_, input) = circuit_builder.create_witness_in(4);

        // Layer 2
        let mul_01 = circuit_builder.create_cell();
        circuit_builder.mul2(mul_01, input[0], input[1], Goldilocks::ONE);

        // Layer 1
        let mul_012 = circuit_builder.create_cell();
        circuit_builder.mul2(mul_012, mul_01, input[2], Goldilocks::ONE);

        // Layer 0
        let (_, mul_001123) = circuit_builder.create_witness_out(1);
        circuit_builder.mul3(mul_001123[0], mul_01, mul_012, input[3], Goldilocks::ONE);

        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);

        assert_eq!(circuit.layers.len(), 4);
        assert_eq!(circuit.layers[3].num_vars, 2);
        assert_eq!(circuit.layers[2].num_vars, 1);
        assert_eq!(circuit.layers[1].num_vars, 2);
        assert_eq!(circuit.layers[0].num_vars, 0);

        // layers[3][2] is copied to layers[2][1], layers[3][3] is copied to layers[1][2]
        // layers[2][0] is copied to layers[1][1]

        let mut expected_paste_from_2 = BTreeMap::new();
        expected_paste_from_2.insert(3, vec![1]);
        assert_eq!(circuit.layers[2].paste_from, expected_paste_from_2);

        let mut expected_paste_from_1 = BTreeMap::new();
        expected_paste_from_1.insert(2, vec![1]);
        expected_paste_from_1.insert(3, vec![2]);
        assert_eq!(circuit.layers[1].paste_from, expected_paste_from_1);

        let mut expected_copy_to_3 = BTreeMap::new();
        expected_copy_to_3.insert(2, vec![2]);
        expected_copy_to_3.insert(1, vec![3]);
        assert_eq!(circuit.layers[3].copy_to, expected_copy_to_3);

        let mut expected_copy_to_2 = BTreeMap::new();
        expected_copy_to_2.insert(1, vec![0]);
        assert_eq!(circuit.layers[2].copy_to, expected_copy_to_2);
    }

    #[test]
    fn test_paste_from_wit_in() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();

        // Layer 2
        let (leaf_id, leaves) = circuit_builder.create_witness_in(6);
        // Unused input elements should also be in the circuit.
        let (dummy_id, _) = circuit_builder.create_witness_in(3);
        let _ = circuit_builder.create_counter_in(1);
        let _ = circuit_builder.create_constant_in(2, 1);

        // Layer 1
        let (_, inners) = circuit_builder.create_witness_out(2);
        circuit_builder.mul2(inners[0], leaves[0], leaves[1], Goldilocks::ONE);
        circuit_builder.mul2(inners[1], leaves[2], leaves[3], Goldilocks::ONE);

        // Layer 0
        let (_, root) = circuit_builder.create_witness_out(1);
        circuit_builder.mul2(root[0], inners[0], inners[1], Goldilocks::ONE);

        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);

        assert_eq!(circuit.layers.len(), 3);
        assert_eq!(circuit.layers[2].num_vars, 4);
        assert_eq!(circuit.layers[1].num_vars, 1);
        // Layers[1][0] -> Layers[0][1], Layers[1][1] -> Layers[0][2]
        assert_eq!(circuit.layers[0].num_vars, 2);

        let mut expected_paste_from_wits_in = vec![(0, 0); 2];
        expected_paste_from_wits_in[leaf_id as usize] = (0usize, 6usize);
        expected_paste_from_wits_in[dummy_id as usize] = (6, 9);
        let expected_paste_from_counter_in = vec![(1, (9, 11))];
        let expected_paste_from_consts_in = vec![(1, (11, 13))];
        assert_eq!(circuit.paste_from_wits_in, expected_paste_from_wits_in);
        assert_eq!(
            circuit.paste_from_counter_in,
            expected_paste_from_counter_in
        );
        assert_eq!(circuit.paste_from_consts_in, expected_paste_from_consts_in);
    }

    #[test]
    fn test_copy_to_wit_out() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        // Layer 2
        let (_, leaves) = circuit_builder.create_witness_in(4);

        // Layer 1
        let (_inner_id, inners) = circuit_builder.create_witness_out(2);
        circuit_builder.mul2(inners[0], leaves[0], leaves[1], Goldilocks::ONE);
        circuit_builder.mul2(inners[1], leaves[2], leaves[3], Goldilocks::ONE);

        // Layer 0
        let root = circuit_builder.create_cell();
        circuit_builder.mul2(root, inners[0], inners[1], Goldilocks::ONE);
        circuit_builder.assert_const(root, 1);

        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);

        assert_eq!(circuit.layers.len(), 3);
        assert_eq!(circuit.layers[2].num_vars, 2);
        assert_eq!(circuit.layers[1].num_vars, 1);
        assert_eq!(circuit.layers[0].num_vars, 2);
        // Layers[1][0] -> Layers[0][1], Layers[1][1] -> Layers[0][2]
        let mut expected_copy_to_1 = BTreeMap::new();
        expected_copy_to_1.insert(0, vec![0, 1]);
        let mut expected_paste_from_0 = BTreeMap::new();
        expected_paste_from_0.insert(1, vec![1, 2]);

        assert_eq!(circuit.layers[1].copy_to, expected_copy_to_1);
        assert_eq!(circuit.layers[0].paste_from, expected_paste_from_0);

        let expected_copy_to_wits_out = vec![vec![1, 2]];
        let expected_assert_const = vec![GateCIn {
            idx_in: [],
            idx_out: 0,
            scalar: ConstantType::Field(Goldilocks::ONE),
        }];

        assert_eq!(circuit.copy_to_wits_out, expected_copy_to_wits_out);
        assert_eq!(circuit.assert_consts, expected_assert_const);
    }

    #[test]
    fn test_rlc_circuit() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        // Layer 2
        let (_, leaves) = circuit_builder.create_witness_in(4);

        // Layer 1
        let inners = circuit_builder.create_ext_cells(2);
        circuit_builder.rlc(&inners[0], &[leaves[0], leaves[1]], 0 as ChallengeId);
        circuit_builder.rlc(&inners[1], &[leaves[2], leaves[3]], 1 as ChallengeId);

        // Layer 0
        let (_, roots) = circuit_builder.create_ext_witness_out(1);
        circuit_builder.mul2_ext(&roots[0], &inners[0], &inners[1], Goldilocks::ONE);

        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);

        let expected_layer2_add_consts = vec![
            Gate {
                idx_in: [],
                idx_out: 0,
                scalar: ConstantType::<GoldilocksExt2>::Challenge(
                    ChallengeConst {
                        challenge: 0,
                        exp: 2,
                    },
                    0,
                ),
            },
            Gate {
                idx_in: [],
                idx_out: 1,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 0,
                        exp: 2,
                    },
                    1,
                ),
            },
            Gate {
                idx_in: [],
                idx_out: 2,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 1,
                        exp: 2,
                    },
                    0,
                ),
            },
            Gate {
                idx_in: [],
                idx_out: 3,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 1,
                        exp: 2,
                    },
                    1,
                ),
            },
        ];

        let expected_layer2_adds = vec![
            Gate {
                idx_in: [0],
                idx_out: 0,
                scalar: ConstantType::<GoldilocksExt2>::Challenge(
                    ChallengeConst {
                        challenge: 0,
                        exp: 0,
                    },
                    0,
                ),
            },
            Gate {
                idx_in: [1],
                idx_out: 0,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 0,
                        exp: 1,
                    },
                    0,
                ),
            },
            Gate {
                idx_in: [0],
                idx_out: 1,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 0,
                        exp: 0,
                    },
                    1,
                ),
            },
            Gate {
                idx_in: [1],
                idx_out: 1,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 0,
                        exp: 1,
                    },
                    1,
                ),
            },
            Gate {
                idx_in: [2],
                idx_out: 2,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 1,
                        exp: 0,
                    },
                    0,
                ),
            },
            Gate {
                idx_in: [3],
                idx_out: 2,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 1,
                        exp: 1,
                    },
                    0,
                ),
            },
            Gate {
                idx_in: [2],
                idx_out: 3,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 1,
                        exp: 0,
                    },
                    1,
                ),
            },
            Gate {
                idx_in: [3],
                idx_out: 3,
                scalar: ConstantType::Challenge(
                    ChallengeConst {
                        challenge: 1,
                        exp: 1,
                    },
                    1,
                ),
            },
        ];

        let expected_layer1_mul2s = vec![
            Gate {
                idx_in: [0, 2],
                idx_out: 0,
                scalar: ConstantType::<GoldilocksExt2>::Field(Goldilocks::ONE),
            },
            Gate {
                idx_in: [0, 3],
                idx_out: 1,
                scalar: ConstantType::Field(Goldilocks::ONE),
            },
            Gate {
                idx_in: [1, 2],
                idx_out: 2,
                scalar: ConstantType::Field(Goldilocks::ONE),
            },
            Gate {
                idx_in: [1, 3],
                idx_out: 3,
                scalar: ConstantType::Field(Goldilocks::ONE),
            },
        ];

        assert_eq!(circuit.layers[2].add_consts, expected_layer2_add_consts);
        assert_eq!(circuit.layers[2].adds, expected_layer2_adds);
        assert_eq!(circuit.layers[1].mul2s, expected_layer1_mul2s);
    }

    #[test]
    fn test_selector_sumcheck_steps() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let _ = circuit_builder.create_constant_in(6, 1);
        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);
        assert_eq!(circuit.layers.len(), 1);
        assert_eq!(
            circuit.layers[0].sumcheck_steps,
            vec![SumcheckStepType::InputPhase2Step1]
        );
    }

    #[test]
    fn test_lookup_inner_sumcheck_steps() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();

        // Layer 2
        let (_, input) = circuit_builder.create_ext_witness_in(4);
        // Layer 0
        let output = circuit_builder.create_ext_cells(2);
        // denominator
        circuit_builder.mul2_ext(
            &output[0], // output_den
            &input[0],  // input_den[0]
            &input[2],  // input_den[1]
            Goldilocks::ONE,
        );

        // numerator
        circuit_builder.mul2_ext(
            &output[1], // output_num
            &input[0],  // input_den[0]
            &input[3],  // input_num[1]
            Goldilocks::ONE,
        );
        circuit_builder.mul2_ext(
            &output[1], // output_num
            &input[2],  // input_den[1]
            &input[1],  // input_num[0]
            Goldilocks::ONE,
        );

        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);

        assert_eq!(circuit.layers.len(), 3);
        // Single input witness, therefore no input phase 2 steps.
        assert_eq!(
            circuit.layers[2].sumcheck_steps,
            vec![SumcheckStepType::Phase1Step1]
        );
        // There are only one incoming evals since the last layer is linear, and
        // no subset evals. Therefore, there are no phase1 steps.
        assert_eq!(
            circuit.layers[1].sumcheck_steps,
            vec![
                SumcheckStepType::Phase2Step1,
                SumcheckStepType::Phase2Step2NoStep3,
            ]
        );
        // Output layer, single output witness, therefore no output phase 1 steps.
        assert_eq!(
            circuit.layers[0].sumcheck_steps,
            vec![SumcheckStepType::LinearPhase2Step1]
        );
    }

    #[test]
    fn test_product_sumcheck_steps() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let (_, input) = circuit_builder.create_witness_in(2);
        let (_, output) = circuit_builder.create_witness_out(1);
        circuit_builder.mul2(output[0], input[0], input[1], Goldilocks::ONE);
        circuit_builder.configure();
        let circuit = Circuit::new(&circuit_builder);

        assert_eq!(circuit.layers.len(), 2);
        // Single input witness, therefore no input phase 2 steps.
        assert_eq!(
            circuit.layers[1].sumcheck_steps,
            vec![SumcheckStepType::Phase1Step1]
        );
        // Output layer, single output witness, therefore no output phase 1 steps.
        assert_eq!(
            circuit.layers[0].sumcheck_steps,
            vec![
                SumcheckStepType::Phase2Step1,
                SumcheckStepType::Phase2Step2NoStep3
            ]
        );
    }
}
