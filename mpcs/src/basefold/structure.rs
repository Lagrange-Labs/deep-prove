use crate::{
    sum_check::classic::{Coefficients, SumcheckProof},
    util::{hash::MerkleHasher, merkle_tree::MerkleTree},
};
use core::fmt::Debug;
use ff_ext::ExtensionField;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use multilinear_extensions::mle::FieldType;

use std::{marker::PhantomData, slice};

pub use super::encoding::{EncodingProverParameters, EncodingScheme, RSCode, RSCodeDefaultSpec};
use super::{
    Basecode, BasecodeDefaultSpec,
    query_phase::{
        BatchedQueriesResultWithMerklePath, QueriesResultWithMerklePath,
        SimpleBatchQueriesResultWithMerklePath,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BasefoldParams<E: ExtensionField, Spec: BasefoldSpec<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(super) params: <Spec::EncodingScheme as EncodingScheme<E>>::PublicParameters,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BasefoldProverParams<E: ExtensionField, Spec: BasefoldSpec<E>> {
    pub encoding_params: <Spec::EncodingScheme as EncodingScheme<E>>::ProverParameters,
}

impl<E: ExtensionField, Spec: BasefoldSpec<E>> BasefoldProverParams<E, Spec> {
    pub fn get_max_message_size_log(&self) -> usize {
        self.encoding_params.get_max_message_size_log()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BasefoldVerifierParams<E: ExtensionField, Spec: BasefoldSpec<E>> {
    pub(super) encoding_params: <Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
}

/// A polynomial commitment together with all the data (e.g., the codeword, and Merkle tree)
/// used to generate this commitment and for assistant in opening
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct BasefoldCommitmentWithWitness<E: ExtensionField, H: MerkleHasher<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub(crate) codeword_tree: MerkleTree<E, H>,
    pub(crate) polynomials_bh_evals: Vec<FieldType<E>>,
    pub(crate) num_vars: usize,
    pub(crate) is_base: bool,
    pub(crate) num_polys: usize,
}

impl<E: ExtensionField, H: MerkleHasher<E>> BasefoldCommitmentWithWitness<E, H>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn to_commitment(&self) -> BasefoldCommitment<H::Digest> {
        BasefoldCommitment::new(
            self.codeword_tree.root(),
            self.num_vars,
            self.is_base,
            self.num_polys,
        )
    }

    pub fn get_evals_ref(&self) -> &[FieldType<E>] {
        &self.polynomials_bh_evals
    }

    pub fn get_root_ref(&self) -> &<H as MerkleHasher<E>>::Digest {
        self.codeword_tree.root_ref()
    }

    pub fn get_root_as(&self) -> <H as MerkleHasher<E>>::Digest {
        self.get_root_ref().clone()
    }

    pub fn get_codewords(&self) -> &Vec<FieldType<E>> {
        self.codeword_tree.leaves()
    }

    pub fn batch_codewords(&self, coeffs: &[E]) -> Vec<E> {
        self.codeword_tree.batch_leaves(coeffs)
    }

    pub fn codeword_size(&self) -> usize {
        self.codeword_tree.size().1
    }

    pub fn codeword_size_log(&self) -> usize {
        self.codeword_tree.height()
    }

    pub fn poly_size(&self) -> usize {
        1 << self.num_vars
    }

    pub fn get_codeword_entry_base(&self, index: usize) -> Vec<E::BaseField> {
        self.codeword_tree.get_leaf_as_base(index)
    }

    pub fn get_codeword_entry_ext(&self, index: usize) -> Vec<E> {
        self.codeword_tree.get_leaf_as_extension(index)
    }

    pub fn is_base(&self) -> bool {
        self.is_base
    }

    pub fn trivial_num_vars<Spec: BasefoldSpec<E>>(num_vars: usize) -> bool {
        num_vars <= Spec::get_basecode_msg_size_log()
    }

    pub fn is_trivial<Spec: BasefoldSpec<E>>(&self) -> bool {
        Self::trivial_num_vars::<Spec>(self.num_vars)
    }
}

// impl<E: ExtensionField, H: MerkleHasher<E>> From<BasefoldCommitmentWithWitness<E, H>>
//     for <H as MerkleHasher<E>>::Digest
// where
//     E::BaseField: Serialize + DeserializeOwned,
// {
//     fn from(val: BasefoldCommitmentWithWitness<E, H>) -> Self {
//         val.get_root_as()
//     }
// }

impl<D, E: ExtensionField, H: MerkleHasher<E, Digest = D>>
    From<&BasefoldCommitmentWithWitness<E, H>> for BasefoldCommitment<D>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn from(val: &BasefoldCommitmentWithWitness<E, H>) -> Self {
        val.to_commitment()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BasefoldCommitment<D> {
    pub(super) root: D,
    pub(super) num_vars: Option<usize>,
    pub(super) is_base: bool,
    pub(super) num_polys: Option<usize>,
}

impl<D: Clone> BasefoldCommitment<D> {
    pub fn new(root: D, num_vars: usize, is_base: bool, num_polys: usize) -> Self {
        Self {
            root,
            num_vars: Some(num_vars),
            is_base,
            num_polys: Some(num_polys),
        }
    }

    pub fn root(&self) -> D {
        self.root.clone()
    }

    pub fn num_vars(&self) -> Option<usize> {
        self.num_vars
    }

    pub fn is_base(&self) -> bool {
        self.is_base
    }
}

impl<E: ExtensionField, H: MerkleHasher<E>> PartialEq for BasefoldCommitmentWithWitness<E, H>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn eq(&self, other: &Self) -> bool {
        self.get_codewords().eq(other.get_codewords())
            && self.polynomials_bh_evals.eq(&other.polynomials_bh_evals)
    }
}

impl<E: ExtensionField, H: MerkleHasher<E>> Eq for BasefoldCommitmentWithWitness<E, H> where
    E::BaseField: Serialize + DeserializeOwned
{
}

pub trait BasefoldSpec<E: ExtensionField>: Debug + Clone + Default {
    type EncodingScheme: EncodingScheme<E>;
    type MerkleHasher: MerkleHasher<E>;

    fn get_number_queries() -> usize {
        Self::EncodingScheme::get_number_queries()
    }

    fn get_rate_log() -> usize {
        Self::EncodingScheme::get_rate_log()
    }

    fn get_basecode_msg_size_log() -> usize {
        Self::EncodingScheme::get_basecode_msg_size_log()
    }
}

#[derive(Debug, Default, Clone)]
pub struct BasefoldBasecodeParams<H> {
    _p: PhantomData<H>,
}

impl<E: ExtensionField, H: MerkleHasher<E>> BasefoldSpec<E> for BasefoldBasecodeParams<H>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type MerkleHasher = H;
    type EncodingScheme = Basecode<BasecodeDefaultSpec>;
}

#[derive(Debug, Default, Clone)]
pub struct BasefoldRSParams<H> {
    _p: PhantomData<H>,
}

impl<E: ExtensionField, H: MerkleHasher<E>> BasefoldSpec<E> for BasefoldRSParams<H>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type MerkleHasher = H;
    type EncodingScheme = RSCode<RSCodeDefaultSpec>;
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Basefold<E: ExtensionField, Spec: BasefoldSpec<E>>(PhantomData<(E, Spec)>);

// impl<E: ExtensionField, Spec: BasefoldSpec<E>> Serialize for Basefold<E, Spec> {
//    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//    where
//        S: Serializer,
//    {
//        serializer.serialize_str("base_fold")
//    }
//}

pub type BasefoldDefault<F, H> = Basefold<F, BasefoldRSParams<H>>;

impl<E: ExtensionField, Spec: BasefoldSpec<E>> Clone for Basefold<E, Spec> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<D> AsRef<[D]> for BasefoldCommitment<D> {
    fn as_ref(&self) -> &[D] {
        let root = &self.root;
        slice::from_ref(root)
    }
}

impl<E: ExtensionField, H: MerkleHasher<E>> AsRef<[<H as MerkleHasher<E>>::Digest]>
    for BasefoldCommitmentWithWitness<E, H>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn as_ref(&self) -> &[<H as MerkleHasher<E>>::Digest] {
        let root = self.get_root_ref();
        slice::from_ref(root)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub enum ProofQueriesResultWithMerklePath<E: ExtensionField, Digest>
where
    E::BaseField: Serialize + DeserializeOwned,
    Digest: Clone + Serialize + DeserializeOwned,
{
    Single(QueriesResultWithMerklePath<E, Digest>),
    Batched(BatchedQueriesResultWithMerklePath<E, Digest>),
    SimpleBatched(SimpleBatchQueriesResultWithMerklePath<E, Digest>),
}

impl<E: ExtensionField, Digest> ProofQueriesResultWithMerklePath<E, Digest>
where
    E::BaseField: Serialize + DeserializeOwned,
    Digest: Clone + Serialize + DeserializeOwned,
{
    pub fn as_single(&self) -> &QueriesResultWithMerklePath<E, Digest> {
        match self {
            Self::Single(x) => x,
            _ => panic!("Not a single query result"),
        }
    }

    pub fn as_batched(&self) -> &BatchedQueriesResultWithMerklePath<E, Digest> {
        match self {
            Self::Batched(x) => x,
            _ => panic!("Not a batched query result"),
        }
    }

    pub fn as_simple_batched(&self) -> &SimpleBatchQueriesResultWithMerklePath<E, Digest> {
        match self {
            Self::SimpleBatched(x) => x,
            _ => panic!("Not a simple batched query result"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BasefoldProof<E: ExtensionField, Digest>
where
    E::BaseField: Serialize + DeserializeOwned,
    Digest: Clone + Serialize + DeserializeOwned,
{
    pub(crate) sumcheck_messages: Vec<Vec<E>>,
    pub(crate) roots: Vec<Digest>,
    pub(crate) final_message: Vec<E>,
    pub(crate) query_result_with_merkle_path: ProofQueriesResultWithMerklePath<E, Digest>,
    pub(crate) sumcheck_proof: Option<SumcheckProof<E, Coefficients<E>>>,
    pub(crate) trivial_proof: Vec<FieldType<E>>,
}

impl<E: ExtensionField, Digest> BasefoldProof<E, Digest>
where
    E::BaseField: Serialize + DeserializeOwned,
    Digest: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    pub fn trivial(evals: Vec<FieldType<E>>) -> Self {
        Self {
            sumcheck_messages: vec![],
            roots: vec![],
            final_message: vec![],
            query_result_with_merkle_path: ProofQueriesResultWithMerklePath::Single(
                QueriesResultWithMerklePath::empty(),
            ),
            sumcheck_proof: None,
            trivial_proof: evals,
        }
    }

    pub fn is_trivial(&self) -> bool {
        !self.trivial_proof.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BasefoldCommitPhaseProof<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    pub(crate) sumcheck_messages: Vec<Vec<E>>,
    pub(crate) roots: Vec<D>,
    pub(crate) final_message: Vec<E>,
}
