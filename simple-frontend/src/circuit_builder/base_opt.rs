use ff_ext::ExtensionField;
use p3_field::FieldAlgebra;

use crate::structs::{
    Cell, CellId, CellType, CircuitBuilder, ConstantType, GateType, InType, MixedCell, OutType,
    WitnessId,
};

impl<Ext: ExtensionField> CircuitBuilder<Ext> {
    pub fn create_cell(&mut self) -> CellId {
        self.cells.push(Cell::default());
        self.cells.len() - 1
    }

    pub fn create_cells(&mut self, num: usize) -> Vec<CellId> {
        self.cells.extend((0..num).map(|_| Cell::default()));
        (self.cells.len() - num..self.cells.len()).collect()
    }

    /// This is to mark the cells with special functionality.
    pub(crate) fn mark_cells(&mut self, cell_type: CellType, cells: &[CellId]) {
        cells.iter().for_each(|cell| {
            self.cells[*cell].cell_type = Some(cell_type);
        });
    }

    pub fn create_witness_in(&mut self, num: usize) -> (WitnessId, Vec<CellId>) {
        let cell = self.create_cells(num);
        self.mark_cells(
            CellType::In(InType::Witness(self.n_witness_in as WitnessId)),
            &cell,
        );
        self.n_witness_in += 1;
        ((self.n_witness_in - 1) as WitnessId, cell)
    }

    /// Create input cells and assign it to be constant.
    pub fn create_constant_in(&mut self, num: usize, constant: i64) -> Vec<CellId> {
        let cell = self.create_cells(num);
        self.mark_cells(CellType::In(InType::Constant(constant)), &cell);
        cell
    }

    /// Create input cells as a counter. It should count from 0 to n_instance *
    /// num through the whole circuit.
    pub fn create_counter_in(&mut self, num_vars: usize) -> Vec<CellId> {
        let cell = self.create_cells(1 << num_vars);
        self.mark_cells(CellType::In(InType::Counter(num_vars)), &cell);
        cell
    }

    pub fn create_witness_out(&mut self, num: usize) -> (WitnessId, Vec<CellId>) {
        let cell = self.create_cells(num);
        self.mark_cells(
            CellType::Out(OutType::Witness(self.n_witness_out as WitnessId)),
            &cell,
        );
        self.n_witness_out += 1;
        ((self.n_witness_out - 1) as WitnessId, cell)
    }

    pub fn create_witness_out_from_cells(&mut self, cells: &[CellId]) -> WitnessId {
        self.mark_cells(
            CellType::Out(OutType::Witness(self.n_witness_out as WitnessId)),
            cells,
        );
        self.n_witness_out += 1;
        (self.n_witness_out - 1) as WitnessId
    }

    pub fn add_const(&mut self, out: CellId, constant: Ext::BaseField) {
        if constant == Ext::BaseField::ZERO {
            return;
        }
        self.add_const_internal(out, ConstantType::Field(constant));
    }

    pub fn add_const_type(&mut self, out: CellId, constant_type: ConstantType<Ext>) {
        self.add_const_internal(out, constant_type);
    }

    pub(crate) fn add_const_internal(&mut self, out: CellId, constant: ConstantType<Ext>) {
        let out_cell = &mut self.cells[out];
        out_cell.gates.push(GateType::add_const(constant));
    }

    pub fn add(&mut self, out: CellId, in_0: CellId, scalar: Ext::BaseField) {
        if scalar == Ext::BaseField::ZERO {
            return;
        }
        self.add_internal(out, in_0, ConstantType::Field(scalar));
    }

    pub(crate) fn add_internal(&mut self, out: CellId, in_0: CellId, scalar: ConstantType<Ext>) {
        let out_cell = &mut self.cells[out];
        out_cell.gates.push(GateType::add(in_0, scalar));
    }

    pub fn mul2(&mut self, out: CellId, in_0: CellId, in_1: CellId, scalar: Ext::BaseField) {
        if scalar == Ext::BaseField::ZERO {
            return;
        }
        self.mul2_internal(out, in_0, in_1, ConstantType::Field(scalar));
    }

    pub(crate) fn mul2_internal(
        &mut self,
        out: CellId,
        in_0: CellId,
        in_1: CellId,
        scalar: ConstantType<Ext>,
    ) {
        let out_cell = &mut self.cells[out];
        out_cell.gates.push(GateType::mul2(in_0, in_1, scalar));
    }

    pub fn mul3(
        &mut self,
        out: CellId,
        in_0: CellId,
        in_1: CellId,
        in_2: CellId,
        scalar: Ext::BaseField,
    ) {
        if scalar == Ext::BaseField::ZERO {
            return;
        }
        self.mul3_internal(out, in_0, in_1, in_2, ConstantType::Field(scalar));
    }

    pub(crate) fn mul3_internal(
        &mut self,
        out: CellId,
        in_0: CellId,
        in_1: CellId,
        in_2: CellId,
        scalar: ConstantType<Ext>,
    ) {
        let out_cell = &mut self.cells[out];
        out_cell
            .gates
            .push(GateType::mul3(in_0, in_1, in_2, scalar));
    }

    pub fn assert_const(&mut self, out: CellId, constant: i64) {
        // check cell and if it's belong to input cell, we must create an intermediate cell
        // and avoid edit input layer cell_type, otherwise it will be orphan cell with no fan-in.
        let out_cell = if let Some(CellType::In(_)) = self.cells[out].cell_type {
            let out_cell = self.create_cell();
            self.add(out_cell, out, Ext::BaseField::ONE);
            out_cell
        } else {
            out
        };
        self.mark_cells(CellType::Out(OutType::AssertConst(constant)), &[out_cell]);
    }

    pub fn add_cell_expr(&mut self, out: CellId, in_0: MixedCell<Ext>) {
        match in_0 {
            MixedCell::Constant(constant) => {
                self.add_const(out, constant);
            }
            MixedCell::Cell(cell_id) => {
                self.add(out, cell_id, Ext::BaseField::ONE);
            }
            MixedCell::CellExpr(cell_id, a, b) => {
                self.add(out, cell_id, a);
                self.add_const(out, b);
            }
        };
    }

    /// IMHO This is `enforce_selection` gate rather than `select` gate.
    pub fn select(&mut self, out: CellId, in_0: CellId, in_1: CellId, cond: CellId) {
        // (1 - cond) * in_0 + cond * in_1 = (in_1 - in_0) * cond + in_0
        let diff = self.create_cell();
        self.add(diff, in_1, Ext::BaseField::ONE);
        self.add(diff, in_0, -Ext::BaseField::ONE);
        self.mul2(out, diff, cond, Ext::BaseField::ONE);
        self.add(out, in_0, Ext::BaseField::ONE);
    }

    pub fn sel_mixed(
        &mut self,
        out: CellId,
        in_0: MixedCell<Ext>,
        in_1: MixedCell<Ext>,
        cond: CellId,
    ) {
        // (1 - cond) * in_0 + cond * in_1 = (in_1 - in_0) * cond + in_0
        match (in_0, in_1) {
            (MixedCell::Constant(in_0), MixedCell::Constant(in_1)) => {
                self.add(out, cond, in_1 - in_0);
            }
            (MixedCell::Constant(in_0), in_1) => {
                let diff = in_1.expr(Ext::BaseField::ONE, -in_0);
                let diff_cell = self.create_cell();
                self.add_cell_expr(diff_cell, diff);
                self.mul2(out, diff_cell, cond, Ext::BaseField::ONE);
                self.add_const(out, in_0);
            }
            (in_0, MixedCell::Constant(in_1)) => {
                self.add_cell_expr(out, in_0);
                let diff = in_0.expr(-Ext::BaseField::ONE, in_1);
                let diff_cell = self.create_cell();
                self.add_cell_expr(diff_cell, diff);
                self.mul2(out, diff_cell, cond, Ext::BaseField::ONE);
            }
            (in_0, in_1) => {
                self.add_cell_expr(out, in_0);
                let diff = self.create_cell();
                self.add_cell_expr(diff, in_1);
                self.add_cell_expr(diff, in_0.expr(-Ext::BaseField::ONE, Ext::BaseField::ZERO));
                self.mul2(out, diff, cond, Ext::BaseField::ONE);
            }
        }
    }
}
