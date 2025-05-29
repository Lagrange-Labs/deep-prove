use ff_ext::ExtensionField;
use itertools::Itertools;
use p3_field::FieldAlgebra;
use std::marker::PhantomData;

use crate::{
    rlc_base_term, rlc_const_term,
    structs::{
        CellId, CellType, ChallengeConst, ChallengeId, CircuitBuilder, ConstantType, ExtCellId,
        InType, MixedCell, OutType, WitnessId,
    },
};

impl<Ext: ExtensionField> From<Vec<CellId>> for ExtCellId<Ext> {
    /// converting a vector of CellIds into an ext cell
    fn from(cells: Vec<CellId>) -> Self {
        Self {
            cells,
            phantom: PhantomData,
        }
    }
}

impl<Ext: ExtensionField> From<ExtCellId<Ext>> for Vec<CellId> {
    /// converting an ext cell into a vector of CellIds
    fn from(val: ExtCellId<Ext>) -> Self {
        val.cells
    }
}

impl<Ext: ExtensionField> AsRef<[CellId]> for ExtCellId<Ext> {
    fn as_ref(&self) -> &[CellId] {
        &self.cells
    }
}

impl<Ext: ExtensionField> ExtCellId<Ext> {
    /// converting a vector of ext cells into a vector of CellIds
    pub fn exts_to_cells(exts: &[Self]) -> Vec<CellId> {
        exts.iter().flat_map(|ext| ext.cells.clone()).collect()
    }

    /// degree of the ext cell
    pub fn degree(&self) -> usize {
        self.cells.len()
    }
}

// Public APIs
impl<Ext: ExtensionField> CircuitBuilder<Ext> {
    // ======================================
    // Cell creations
    // ======================================

    /// Create an ExtCellId for an extension field element.
    /// Note: an extension cell already consists of multiple cells.
    pub fn create_ext_cell(&mut self) -> ExtCellId<Ext> {
        self.create_cells(<Ext as ExtensionField>::DEGREE).into()
    }

    /// Create a vector of ExtCells for a vector of extension field elements.
    /// Note: an extension cell already consists of multiple cells.
    pub fn create_ext_cells(&mut self, num: usize) -> Vec<ExtCellId<Ext>> {
        let cells = self.create_cells(num * <Ext as ExtensionField>::DEGREE);
        cells
            .chunks_exact(<Ext as ExtensionField>::DEGREE)
            .map(|x| x.to_vec().into())
            .collect()
    }

    pub fn create_ext_witness_in(&mut self, num: usize) -> (WitnessId, Vec<ExtCellId<Ext>>) {
        let cells = self.create_cells(num * <Ext as ExtensionField>::DEGREE);
        self.mark_cells(
            CellType::In(InType::Witness(self.n_witness_in as WitnessId)),
            &cells,
        );
        self.n_witness_in += 1;
        (
            (self.n_witness_in - 1) as WitnessId,
            cells
                .chunks_exact(<Ext as ExtensionField>::DEGREE)
                .map(|x| x.to_vec().into())
                .collect(),
        )
    }

    /// Create input cells and assign it to be constant.
    pub fn create_ext_constant_in(&mut self, num: usize, constant: i64) -> Vec<ExtCellId<Ext>> {
        let cells = self.create_ext_cells(num);
        cells.iter().for_each(|ext_cell| {
            // first base field is the constant
            self.mark_cells(
                CellType::In(InType::Constant(constant)),
                &[ext_cell.cells[0]],
            );
            // the rest fields are 0s
            self.mark_cells(CellType::In(InType::Constant(0)), &ext_cell.cells[1..]);
        });
        cells
    }

    pub fn create_ext_witness_out(&mut self, num: usize) -> (WitnessId, Vec<ExtCellId<Ext>>) {
        let cells = self.create_cells(num * <Ext as ExtensionField>::DEGREE);
        self.mark_cells(
            CellType::Out(OutType::Witness(self.n_witness_out as WitnessId)),
            &cells,
        );
        self.n_witness_out += 1;
        (
            (self.n_witness_out - 1) as WitnessId,
            cells
                .chunks_exact(<Ext as ExtensionField>::DEGREE)
                .map(|x| x.to_vec().into())
                .collect(),
        )
    }

    pub fn create_witness_out_from_exts(&mut self, exts: &[ExtCellId<Ext>]) -> WitnessId {
        for ext in exts {
            self.mark_cells(
                CellType::Out(OutType::Witness(self.n_witness_out as WitnessId)),
                ext.as_ref(),
            );
        }
        self.n_witness_out += 1;
        (self.n_witness_out - 1) as WitnessId
    }

    // ======================================
    // Cell selections
    // ======================================
    /// Base on the condition, select
    /// - either extension cell in_0,
    /// - or a new extension cell from [in_1, 0, 0, 0 ...]
    pub fn sel_ext_and_mixed(
        &mut self,
        out: &ExtCellId<Ext>,
        in_0: &ExtCellId<Ext>,
        in_1: &MixedCell<Ext>,
        cond: CellId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        assert_eq!(in_0.degree(), <Ext as ExtensionField>::DEGREE);

        out.cells
            .iter()
            .zip_eq(
                in_0.cells
                    .iter()
                    .zip_eq([*in_1].iter().chain(std::iter::repeat_n(
                        &MixedCell::Constant(Ext::BaseField::ZERO),
                        <Ext as ExtensionField>::DEGREE - 1,
                    ))),
            )
            .for_each(|(&out, (&in0, &in1))| self.sel_mixed(out, in0.into(), in1, cond));
    }

    /// Base on the condition, select
    /// - either a new extension cell from [in_0, 0, 0, 0 ...]
    /// - or extension cell in_1,
    pub fn sel_mixed_and_ext(
        &mut self,
        out: &ExtCellId<Ext>,
        in_0: &MixedCell<Ext>,
        in_1: &ExtCellId<Ext>,
        cond: CellId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        assert_eq!(in_1.degree(), <Ext as ExtensionField>::DEGREE);

        out.cells
            .iter()
            .zip_eq(
                in_1.cells
                    .iter()
                    .zip_eq([*in_0].iter().chain(std::iter::repeat_n(
                        &MixedCell::Constant(Ext::BaseField::ZERO),
                        <Ext as ExtensionField>::DEGREE - 1,
                    ))),
            )
            .for_each(|(&out, (&in1, &in0))| self.sel_mixed(out, in0, in1.into(), cond));
    }

    /// Base on the condition, select extension cells in_0 or in_1
    pub fn sel_ext(
        &mut self,
        out: &ExtCellId<Ext>,
        in_0: &ExtCellId<Ext>,
        in_1: &ExtCellId<Ext>,
        cond: CellId,
    ) {
        // we only need to check one degree since the rest are
        // enforced by zip_eq
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);

        out.cells
            .iter()
            .zip_eq(in_0.cells.iter().zip_eq(in_1.cells.iter()))
            .for_each(|(&out, (&in0, &in1))| self.select(out, in0, in1, cond));
    }

    // ======================================
    // Cell arithmetics
    // ======================================
    /// Constrain out += in_0*scalar
    pub fn add_ext(&mut self, out: &ExtCellId<Ext>, in_0: &ExtCellId<Ext>, scalar: Ext::BaseField) {
        // we only need to check one degree since the rest are
        // enforced by zip_eq
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        out.cells
            .iter()
            .zip_eq(in_0.cells.iter())
            .for_each(|(&o, &i)| self.add(o, i, scalar));
    }

    /// Constrain
    /// - out[i] += in_0[i] * in_1 * scalar for i in 0..DEGREE-1
    pub fn mul_ext_base(
        &mut self,
        out: &ExtCellId<Ext>,
        in_0: &ExtCellId<Ext>,
        in_1: CellId,
        scalar: Ext::BaseField,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        out.cells
            .iter()
            .zip_eq(in_0.cells.iter())
            .for_each(|(&o, &i)| self.mul2(o, i, in_1, scalar));
    }

    /// Constrain Extension field multiplications.
    /// In the case of DEGREE = 2, it is
    /// - out[0] += (in_0[0] * in_1[0] + 7 * in_0[1] * in_1[1]) * scalar
    /// - out[1] += (in_0[0] * in_1[1] +     in_0[1] * in_1[0]) * scalar
    pub fn mul2_ext(
        &mut self,
        out: &ExtCellId<Ext>,
        in_0: &ExtCellId<Ext>,
        in_1: &ExtCellId<Ext>,
        scalar: Ext::BaseField,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        match <Ext as ExtensionField>::DEGREE {
            2 => self.mul2_degree_2_ext_internal(&out.cells, &in_0.cells, &in_1.cells, scalar),
            3 => self.mul2_degree_3_ext_internal(&out.cells, &in_0.cells, &in_1.cells, scalar),
            // we do not support extension field beyond 3 at the moment
            _ => unimplemented!(),
        }
    }

    /// Constrain out += in_0 * c
    pub fn add_product_of_ext_and_challenge(
        &mut self,
        out: &ExtCellId<Ext>,
        in_0: &ExtCellId<Ext>,
        c: ChallengeConst,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        assert_eq!(in_0.degree(), <Ext as ExtensionField>::DEGREE);
        match <Ext as ExtensionField>::DEGREE {
            2 => self.add_ext_mul_challenge_2(&out.cells, &in_0.cells, c),
            3 => self.add_ext_mul_challenge_3(&out.cells, &in_0.cells, c),
            _ => unimplemented!(),
        }
    }

    // ======================================
    // Cell random linear combinations
    // ======================================

    /// Compute the random linear combination of `in_array` by challenge.
    /// out = (\sum_{i = 0}^{in_array.len()} challenge^i * in_array[i]) + challenge^{in_array.len()}.
    pub fn rlc(&mut self, out: &ExtCellId<Ext>, in_array: &[CellId], challenge: ChallengeId) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        for (i, item) in in_array.iter().enumerate() {
            let c = ChallengeConst {
                challenge,
                exp: i as u64,
            };
            rlc_base_term!(self, <Ext as ExtensionField>::DEGREE, out.cells, *item; c);
        }
        let c = ChallengeConst {
            challenge,
            exp: in_array.len() as u64,
        };
        rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c);
    }

    /// Computes a combination of a number of columns, multiplying each column by successive powers of the provided challenge
    /// useful in LogUp proving.
    pub fn combine_columns(
        &mut self,
        out: &ExtCellId<Ext>,
        in_array: &[CellId],
        challenge: ChallengeId,
        const_challenge: ChallengeId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        for (i, item) in in_array.iter().enumerate() {
            let c = ChallengeConst {
                challenge,
                exp: i as u64 + 1,
            };
            rlc_base_term!(self, <Ext as ExtensionField>::DEGREE, out.cells, *item; c);
        }
        let c = ChallengeConst {
            challenge: const_challenge,
            exp: 1u64,
        };
        rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c);
    }

    /// Computes a combination of a number of columns, multiplying each column by successive powers of the provided challenge
    /// useful in LogUp proving.
    pub fn calculate_table_column(
        &mut self,
        out: &ExtCellId<Ext>,
        in_array: &[CellId],
        partition: &[usize],
        challenges: &[ChallengeId],
        const_challenge: ChallengeId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        let partition_sum = partition.iter().sum::<usize>();
        assert_eq!(in_array.len(), partition_sum);

        let mut array_offset = 0usize;
        for (i, &num) in partition.iter().enumerate() {
            in_array
                .iter()
                .skip(array_offset)
                .take(num)
                .enumerate()
                .for_each(|(j, &item)| {
                    let c = ChallengeConst {
                        challenge: challenges[i],
                        exp: j as u64 + 1,
                    };
                    rlc_base_term!(self, <Ext as ExtensionField>::DEGREE, out.cells, item; c);
                });
            array_offset += num;
        }

        let c = ChallengeConst {
            challenge: const_challenge,
            exp: 1u64,
        };
        rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c);
    }

    /// Compute the random linear combination of `in_array` by challenge.
    /// out = \sum_{i = 0}^{in_array.len()} challenge^i * in_array[i] + challenge^{in_array.len()}.
    pub fn rlc_ext(
        &mut self,
        out: &ExtCellId<Ext>,
        in_array: &[ExtCellId<Ext>],
        challenge: ChallengeId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        match <Ext as ExtensionField>::DEGREE {
            2 => self.rlc_ext_2(out, in_array, challenge),
            3 => self.rlc_ext_3(out, in_array, challenge),
            _ => unimplemented!(),
        }
    }

    /// Compute the random linear combination of `in_array` with mixed types by challenge.
    /// out = (\sum_{i = 0}^{in_array.len()} challenge^i * (\sum_j in_array[i][j])) + challenge^{in_array.len()}.
    pub fn rlc_mixed(
        &mut self,
        out: &ExtCellId<Ext>,
        in_array: &[MixedCell<Ext>],
        challenge: ChallengeId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        for (i, item) in in_array.iter().enumerate() {
            let c: ChallengeConst = ChallengeConst {
                challenge,
                exp: i as u64,
            };
            match item {
                MixedCell::Constant(constant) => {
                    rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c, *constant)
                }
                MixedCell::Cell(cell_id) => {
                    rlc_base_term!(self, <Ext as ExtensionField>::DEGREE, out.cells, *cell_id; c)
                }
                MixedCell::CellExpr(cell_id, a, b) => {
                    rlc_base_term!(self, <Ext as ExtensionField>::DEGREE, out.cells, *cell_id; c, *a);
                    rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c, *b);
                }
            }
        }
        let c: ChallengeConst = ChallengeConst {
            challenge,
            exp: in_array.len() as u64,
        };
        rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c);
    }
}

// Internal APIs
impl<Ext: ExtensionField> CircuitBuilder<Ext> {
    /// let a1b1 = a.0[0] * b.0[0];
    /// let a1b2 = a.0[0] * b.0[1];
    /// let a2b1 = a.0[1] * b.0[0];
    /// let a2b2 = a.0[1] * b.0[1];
    /// let c1 = a1b1 + Goldilocks(7) * a2b2;
    /// let c2 = a2b1 + a1b2;
    fn mul2_degree_2_ext_internal(
        &mut self,
        out: &[CellId],
        in_0: &[CellId],
        in_1: &[CellId],
        scalar: Ext::BaseField,
    ) {
        let a0b0 = self.create_cell();
        self.mul2(a0b0, in_0[0], in_1[0], Ext::BaseField::ONE);
        let a0b1 = self.create_cell();
        self.mul2(a0b1, in_0[0], in_1[1], Ext::BaseField::ONE);
        let a1b0 = self.create_cell();
        self.mul2(a1b0, in_0[1], in_1[0], Ext::BaseField::ONE);
        let a1b1 = self.create_cell();
        self.mul2(a1b1, in_0[1], in_1[1], Ext::BaseField::ONE);
        self.add(out[0], a0b0, scalar);
        self.add(out[0], a1b1, Ext::BaseField::from_canonical_u64(7) * scalar);
        self.add(out[1], a1b0, scalar);
        self.add(out[1], a0b1, scalar);
    }

    fn add_ext_mul_challenge_2(&mut self, out: &[CellId], in_0: &[CellId], c: ChallengeConst) {
        let a0b0 = self.create_cell();
        let in_1 = [ConstantType::Challenge(c, 0), ConstantType::Challenge(c, 1)];
        self.add_internal(a0b0, in_0[0], in_1[0]);
        let a0b1 = self.create_cell();
        self.add_internal(a0b1, in_0[0], in_1[1]);
        let a1b0 = self.create_cell();
        self.add_internal(a1b0, in_0[1], in_1[0]);
        let a1b1 = self.create_cell();
        self.add_internal(a1b1, in_0[1], in_1[1]);
        self.add(out[0], a0b0, Ext::BaseField::ONE);
        self.add(out[0], a1b1, Ext::BaseField::from_canonical_u64(7));
        self.add(out[1], a1b0, Ext::BaseField::ONE);
        self.add(out[1], a0b1, Ext::BaseField::ONE);
    }

    /// Random linear combinations for extension cells with degree = 2
    fn rlc_ext_2(
        &mut self,
        out: &ExtCellId<Ext>,
        in_array: &[ExtCellId<Ext>],
        challenge: ChallengeId,
    ) {
        assert_eq!(out.degree(), <Ext as ExtensionField>::DEGREE);
        for (i, item) in in_array.iter().enumerate() {
            let c = ChallengeConst {
                challenge,
                exp: i as u64,
            };
            self.add_ext_mul_challenge_2(&out.cells, &item.cells, c);
        }
        let c = ChallengeConst {
            challenge,
            exp: in_array.len() as u64,
        };
        rlc_const_term!(self, <Ext as ExtensionField>::DEGREE, out.cells; c);
    }

    /// Random linear combinations for extension cells with degree = 3
    fn rlc_ext_3(
        &mut self,
        out: &ExtCellId<Ext>,
        in_array: &[ExtCellId<Ext>],
        challenge: ChallengeId,
    ) {
        assert_eq!(out.degree(), 3);
        for (i, item) in in_array.iter().enumerate() {
            let c = ChallengeConst {
                challenge,
                exp: i as u64,
            };
            self.add_ext_mul_challenge_3(&out.cells, &item.cells, c);
        }
        let c = ChallengeConst {
            challenge,
            exp: in_array.len() as u64,
        };
        rlc_const_term!(self, 3, out.cells; c);
    }

    /// let a1b1 = a.0[0] * b.0[0];
    /// let a1b2 = a.0[0] * b.0[1];
    /// let a1b3 = a.0[0] * b.0[2];
    /// let a2b1 = a.0[1] * b.0[0];
    /// let a2b2 = a.0[1] * b.0[1];
    /// let a2b3 = a.0[1] * b.0[2];
    /// let a3b1 = a.0[2] * b.0[0];
    /// let a3b2 = a.0[2] * b.0[1];
    /// let a3b3 = a.0[2] * b.0[2];
    /// let c1 = a1b1 + a3b2 + a2b3;
    /// let c2 = a2b1 + a1b2 + a2b3 + a3b2 + a3b3;
    /// let c3 = a3b1 + a2b2 + a1b3 + a3b3;
    /// GoldilocksExt3([c1, c2, c3])
    fn mul2_degree_3_ext_internal(
        &mut self,
        out: &[CellId],
        in_0: &[CellId],
        in_1: &[CellId],
        scalar: Ext::BaseField,
    ) {
        let a0b0 = self.create_cell();
        self.mul2(a0b0, in_0[0], in_1[0], Ext::BaseField::ONE);
        let a0b1 = self.create_cell();
        self.mul2(a0b1, in_0[0], in_1[1], Ext::BaseField::ONE);
        let a0b2 = self.create_cell();
        self.mul2(a0b2, in_0[0], in_1[2], Ext::BaseField::ONE);
        let a1b0 = self.create_cell();
        self.mul2(a1b0, in_0[1], in_1[0], Ext::BaseField::ONE);
        let a1b1 = self.create_cell();
        self.mul2(a1b1, in_0[1], in_1[1], Ext::BaseField::ONE);
        let a1b2 = self.create_cell();
        self.mul2(a1b2, in_0[1], in_1[2], Ext::BaseField::ONE);
        let a2b0 = self.create_cell();
        self.mul2(a2b0, in_0[2], in_1[0], Ext::BaseField::ONE);
        let a2b1 = self.create_cell();
        self.mul2(a2b1, in_0[2], in_1[1], Ext::BaseField::ONE);
        let a2b2 = self.create_cell();
        self.mul2(a2b2, in_0[2], in_1[2], Ext::BaseField::ONE);
        self.add(out[0], a0b0, scalar);
        self.add(out[0], a2b1, scalar);
        self.add(out[0], a1b2, scalar);
        self.add(out[1], a1b0, scalar);
        self.add(out[1], a0b1, scalar);
        self.add(out[1], a2b1, scalar);
        self.add(out[1], a1b2, scalar);
        self.add(out[1], a2b2, scalar);
        self.add(out[2], a2b0, scalar);
        self.add(out[2], a1b1, scalar);
        self.add(out[2], a0b2, scalar);
        self.add(out[2], a2b2, scalar);
    }

    fn add_ext_mul_challenge_3(&mut self, out: &[CellId], in_0: &[CellId], c: ChallengeConst) {
        let in_1 = [
            ConstantType::Challenge(c, 0),
            ConstantType::Challenge(c, 1),
            ConstantType::Challenge(c, 2),
        ];
        let a0b0 = self.create_cell();
        self.add_internal(a0b0, in_0[0], in_1[0]);
        let a0b1 = self.create_cell();
        self.add_internal(a0b1, in_0[0], in_1[1]);
        let a0b2 = self.create_cell();
        self.add_internal(a0b2, in_0[0], in_1[2]);
        let a1b0 = self.create_cell();
        self.add_internal(a1b0, in_0[1], in_1[0]);
        let a1b1 = self.create_cell();
        self.add_internal(a1b1, in_0[1], in_1[1]);
        let a1b2 = self.create_cell();
        self.add_internal(a1b2, in_0[1], in_1[2]);
        let a2b0 = self.create_cell();
        self.add_internal(a2b0, in_0[2], in_1[0]);
        let a2b1 = self.create_cell();
        self.add_internal(a2b1, in_0[2], in_1[1]);
        let a2b2 = self.create_cell();
        self.add_internal(a2b2, in_0[2], in_1[2]);
        self.add(out[0], a0b0, Ext::BaseField::ONE);
        self.add(out[0], a2b1, Ext::BaseField::ONE);
        self.add(out[0], a1b2, Ext::BaseField::ONE);
        self.add(out[1], a1b0, Ext::BaseField::ONE);
        self.add(out[1], a0b1, Ext::BaseField::ONE);
        self.add(out[1], a2b1, Ext::BaseField::ONE);
        self.add(out[1], a1b2, Ext::BaseField::ONE);
        self.add(out[1], a2b2, Ext::BaseField::ONE);
        self.add(out[2], a2b0, Ext::BaseField::ONE);
        self.add(out[2], a1b1, Ext::BaseField::ONE);
        self.add(out[2], a0b2, Ext::BaseField::ONE);
        self.add(out[2], a2b2, Ext::BaseField::ONE);
    }
}
