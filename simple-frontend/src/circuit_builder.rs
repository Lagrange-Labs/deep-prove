use std::collections::VecDeque;

use ff_ext::ExtensionField;

use crate::structs::{CellId, CellType, CircuitBuilder, ConstantType, GateType, LayerId};

mod base_opt;
mod derives;
mod ext_opt;

impl<Ext: ExtensionField> GateType<Ext> {
    pub(crate) fn add_const(constant: ConstantType<Ext>) -> Self {
        Self {
            idx_in: vec![],
            scalar: constant,
        }
    }

    pub(crate) fn add(in0: CellId, scalar: ConstantType<Ext>) -> Self {
        Self {
            idx_in: vec![in0],
            scalar,
        }
    }

    pub(crate) fn mul2(in0: CellId, in1: CellId, scalar: ConstantType<Ext>) -> Self {
        Self {
            idx_in: vec![in0, in1],
            scalar,
        }
    }

    pub(crate) fn mul3(in0: CellId, in1: CellId, in2: CellId, scalar: ConstantType<Ext>) -> Self {
        Self {
            idx_in: vec![in0, in1, in2],
            scalar,
        }
    }
}

impl<Ext: ExtensionField> CircuitBuilder<Ext> {
    /// Prepare the circuit. This is to assign the layers and challenge levels
    /// to the cells.
    pub fn configure(&mut self) {
        // Topological sort.
        // Initialize the out degree.
        let mut degrees = vec![0; self.cells.len()];
        for cell in self.cells.iter() {
            for gate in cell.gates.iter() {
                for input in gate.idx_in.iter() {
                    degrees[*input] += 1;
                }
            }
        }

        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.cells.len()];
        let mut layers = vec![0 as LayerId; self.cells.len()];
        for cell_id in 0..self.cells.len() {
            if degrees[cell_id] == 0 {
                queue.push_front(cell_id);
                visited[cell_id] = true;
                layers[cell_id] = 0;
            }
        }

        // Assign layers to all cells:
        let mut max_layer_id = 0;
        while !queue.is_empty() {
            let curr = queue.pop_back().unwrap();
            let cell = &mut self.cells[curr];

            let curr_layer = layers[curr];
            max_layer_id = max_layer_id.max(curr_layer);
            cell.layer = Some(curr_layer);

            for gate in cell.gates.iter() {
                for input in gate.idx_in.iter() {
                    degrees[*input] -= 1;
                    layers[*input] = layers[*input].max(curr_layer + 1);
                    if degrees[*input] == 0 && !visited[*input] {
                        queue.push_front(*input);
                        visited[*input] = true;
                    }
                }
            }
        }

        // Force input cells to be in the input layer.
        for cell_id in 0..self.cells.len() {
            if matches!(self.cells[cell_id].cell_type, Some(CellType::In(_))) {
                self.cells[cell_id].layer = Some(max_layer_id);
            }
        }

        self.n_layers = Some(max_layer_id + 1);
    }

    pub fn n_witness_in(&self) -> usize {
        self.n_witness_in
    }

    pub fn n_witness_out(&self) -> usize {
        self.n_witness_out
    }

    #[cfg(debug_assertions)]
    pub fn print_info(&self) {
        println!("The number of layers: {}", self.n_layers.as_ref().unwrap());
        println!("The number of cells: {}", self.cells.len());

        for (i, cell) in self.cells.iter().enumerate() {
            println!("Cell {i}: {cell:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::enumerate;

    use crate::structs::CircuitBuilder;
    use ff_ext::GoldilocksExt2;
    use p3_field::FieldAlgebra;
    use p3_goldilocks::Goldilocks;

    #[test]
    fn test_cb_empty_cells() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let (_, input_cells) = circuit_builder.create_witness_in(4);
        let zero_cells = circuit_builder.create_cells(2);
        let leaves = input_cells
            .iter()
            .chain(zero_cells.iter())
            .cloned()
            .collect::<Vec<_>>();
        let inners = circuit_builder.create_cells(2);
        circuit_builder.mul3(inners[0], leaves[0], leaves[1], leaves[2], Goldilocks::ONE);
        circuit_builder.mul3(inners[1], leaves[3], leaves[4], leaves[5], Goldilocks::ONE);
        circuit_builder.create_witness_out_from_cells(&inners);

        circuit_builder.configure();

        assert_eq!(circuit_builder.cells.len(), 8);
        let layers = [1, 1, 1, 1, 1, 1, 0, 0];
        for (cell_id, layer) in enumerate(layers).take(8) {
            assert_eq!(circuit_builder.cells[cell_id].layer, Some(layer));
        }
    }

    #[test]
    fn test_cb_const_cells() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let (_, input_cells) = circuit_builder.create_witness_in(4);
        let const_cells = circuit_builder.create_cells(2);
        circuit_builder.add_const(const_cells[0], Goldilocks::ONE);
        circuit_builder.add_const(const_cells[1], Goldilocks::ONE);

        let leaves = input_cells
            .iter()
            .chain(const_cells.iter())
            .cloned()
            .collect::<Vec<_>>();
        let inners = circuit_builder.create_cells(2);
        circuit_builder.mul3(inners[0], leaves[0], leaves[1], leaves[2], Goldilocks::ONE);
        circuit_builder.mul3(inners[1], leaves[3], leaves[4], leaves[5], Goldilocks::ONE);
        circuit_builder.create_witness_out_from_cells(&inners);

        circuit_builder.configure();

        assert_eq!(circuit_builder.cells.len(), 8);
        let layers = [1, 1, 1, 1, 1, 1, 0, 0];
        for (cell_id, &layer) in layers.iter().enumerate().take(8) {
            assert_eq!(circuit_builder.cells[cell_id].layer, Some(layer));
        }
    }

    #[test]
    fn test_assert_cells() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let (_, leaves) = circuit_builder.create_witness_in(4);

        let inners = circuit_builder.create_cells(2);
        circuit_builder.mul2(inners[0], leaves[0], leaves[1], Goldilocks::ONE);
        circuit_builder.mul2(inners[1], leaves[2], leaves[3], Goldilocks::ONE);
        circuit_builder.assert_const(inners[0], 1);
        circuit_builder.assert_const(inners[1], 1);

        circuit_builder.configure();

        assert_eq!(circuit_builder.cells.len(), 6);
        let layers = [1, 1, 1, 1, 0, 0];
        for (cell_id, &layer) in layers.iter().enumerate().take(6) {
            assert_eq!(
                circuit_builder.cells[cell_id].layer,
                Some(layer),
                "cell_id: {}",
                cell_id
            );
        }
    }

    #[test]
    fn test_inner_output_cells() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let (_, leaves) = circuit_builder.create_witness_in(4);

        let (_, inners) = circuit_builder.create_witness_out(2);
        circuit_builder.mul2(inners[0], leaves[0], leaves[1], Goldilocks::ONE);
        circuit_builder.mul2(inners[1], leaves[2], leaves[3], Goldilocks::ONE);

        let (_, root) = circuit_builder.create_witness_out(1);
        circuit_builder.mul2(root[0], inners[0], inners[1], Goldilocks::ONE);

        circuit_builder.configure();

        assert_eq!(circuit_builder.cells.len(), 7);
        let layers = [2, 2, 2, 2, 1, 1, 0];
        for (cell_id, &layer) in layers.iter().enumerate().take(7) {
            assert_eq!(circuit_builder.cells[cell_id].layer, Some(layer));
        }
    }

    #[test]
    fn test_force_input_layer() {
        let mut circuit_builder = CircuitBuilder::<GoldilocksExt2>::default();
        let (_, leaves) = circuit_builder.create_witness_in(6);

        // Unused input elements should also be in the circuit.
        let _ = circuit_builder.create_witness_in(3);
        let _ = circuit_builder.create_counter_in(1);
        let _ = circuit_builder.create_constant_in(2, 1);

        let (_, inners) = circuit_builder.create_witness_out(2);
        circuit_builder.mul2(inners[0], leaves[0], leaves[1], Goldilocks::ONE);
        circuit_builder.mul2(inners[1], leaves[2], leaves[3], Goldilocks::ONE);

        let (_, root) = circuit_builder.create_witness_out(1);
        circuit_builder.mul2(root[0], inners[0], inners[1], Goldilocks::ONE);

        circuit_builder.configure();

        assert_eq!(circuit_builder.cells.len(), 16);
        let layers = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0];
        for (cell_id, &layer) in layers.iter().enumerate() {
            assert_eq!(circuit_builder.cells[cell_id].layer, Some(layer));
        }
    }
}
