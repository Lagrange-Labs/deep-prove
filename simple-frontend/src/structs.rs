extern crate p3_field;
use ff_ext::ExtensionField;
use p3_field::FieldAlgebra;
use serde::{Deserialize, Serialize};
use std::{hash::Hash, marker::PhantomData};

// We make use of the following identifiers.
// For type safety we want different alias for those identifiers; while disallow arithmetics cross different identifiers.
// We achieve this via setting them to different primitive types.
// This works better/simpler than struct-wrapping
pub type ChallengeId = u8;
pub type TableType = u16;
pub type WitnessId = u16;
pub type LayerId = u32;
pub type CellId = usize;

#[derive(Clone, Copy, Debug, Serialize, Eq, PartialEq, Hash, Deserialize)]
pub struct ChallengeConst {
    pub challenge: ChallengeId,
    pub exp: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConstantType<Ext: ExtensionField> {
    Field(#[serde(skip)] Ext::BaseField),
    /// ChallengeConst is an extension field element represents a power of ChallengeId.
    /// The usize here denotes the `usize`-th base field element of the const.
    Challenge(ChallengeConst, usize),
    ChallengeScaled(ChallengeConst, usize, #[serde(skip)] Ext::BaseField),
}

/// Represent a gate in the circuit. The inner variables denote the input
/// indices and scalar.
#[derive(Clone, Debug)]
pub struct GateType<Ext: ExtensionField> {
    pub idx_in: Vec<CellId>,
    pub scalar: ConstantType<Ext>,
}

/// Store wire structure of the circuit.
#[derive(Clone, Debug, Default)]
pub struct Cell<Ext: ExtensionField> {
    /// The layer of the cell.
    pub layer: Option<LayerId>,
    /// The value of the cell is the sum of all gates.
    pub gates: Vec<GateType<Ext>>,
    /// The type of the cell, e.g., public input, witness, challenge, etc.
    pub cell_type: Option<CellType>,
}

/// An ExtCell consists DEGREE number of cells.
#[derive(Clone, Debug)]
pub struct ExtCellId<Ext: ExtensionField> {
    pub cells: Vec<CellId>,
    pub phantom: PhantomData<Ext>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Serialize, Ord, PartialOrd)]
pub enum InType {
    Witness(WitnessId),
    /// Constant(num_vars) acts like a counter (0, 1, 2, ...) through all
    /// instances. Each instance holds 1 << num_vars of them.
    Counter(usize),
    /// Constant keeps the same for all instances.
    Constant(i64),
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Serialize, Ord, PartialOrd)]
pub enum OutType {
    Witness(WitnessId),
    AssertConst(i64),
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Serialize, Ord, PartialOrd)]
pub enum CellType {
    In(InType),
    Out(OutType),
}

/// A MixedCell can be a constant, a cell, or a Cell Expression
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub enum MixedCell<Ext: ExtensionField> {
    Constant(Ext::BaseField),
    Cell(CellId),
    CellExpr(CellId, Ext::BaseField, Ext::BaseField),
}

impl<Ext: ExtensionField> From<CellId> for MixedCell<Ext> {
    fn from(cell_id: CellId) -> Self {
        MixedCell::Cell(cell_id)
    }
}

impl<Ext: ExtensionField> MixedCell<Ext> {
    pub fn add(&self, shift: Ext::BaseField) -> Self {
        match self {
            MixedCell::Constant(c) => MixedCell::Constant(*c + shift),
            MixedCell::Cell(c) => MixedCell::CellExpr(*c, Ext::BaseField::ONE, shift),
            MixedCell::CellExpr(c, s, sh) => MixedCell::CellExpr(*c, *s, *sh + shift),
        }
    }
    pub fn sub(&self, shift: Ext::BaseField) -> Self {
        match self {
            MixedCell::Constant(c) => MixedCell::Constant(*c - shift),
            MixedCell::Cell(c) => MixedCell::CellExpr(*c, Ext::BaseField::ONE, -shift),
            MixedCell::CellExpr(c, s, sh) => MixedCell::CellExpr(*c, *s, *sh - shift),
        }
    }
    pub fn mul(&self, scalar: Ext::BaseField) -> Self {
        match self {
            MixedCell::Constant(c) => MixedCell::Constant(*c * scalar),
            MixedCell::Cell(c) => MixedCell::CellExpr(*c, scalar, Ext::BaseField::ZERO),
            MixedCell::CellExpr(c, s, sh) => MixedCell::CellExpr(*c, *s * scalar, *sh * scalar),
        }
    }
    pub fn expr(&self, scalar: Ext::BaseField, shift: Ext::BaseField) -> Self {
        match self {
            MixedCell::Constant(c) => MixedCell::Constant(*c * scalar + shift),
            MixedCell::Cell(c) => MixedCell::Cell(*c),
            MixedCell::CellExpr(c, s, sh) => MixedCell::CellExpr(*c, *s * scalar, *sh * shift),
        }
    }
}

#[derive(Default)]
pub struct CircuitBuilder<Ext: ExtensionField> {
    pub cells: Vec<Cell<Ext>>,

    /// Number of layers in the circuit.
    pub n_layers: Option<u32>,

    pub(crate) n_witness_in: usize,
    pub(crate) n_witness_out: usize,
}
