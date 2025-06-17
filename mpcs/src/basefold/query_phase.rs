use crate::util::{
    arithmetic::{
        degree_2_eval, degree_2_zero_plus_one, inner_product, interpolate_over_boolean_hypercube,
        interpolate2_weights,
    },
    ext_to_usize, field_type_index_base, field_type_index_ext,
    hash::MerkleHasher,
    log2_strict,
    merkle_tree::{MerklePathWithoutLeafOrRoot, MerkleTree},
};
use ark_std::{end_timer, start_timer};
use core::fmt::Debug;
use ff_ext::ExtensionField;
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use multilinear_extensions::mle::FieldType;

use crate::util::plonky2_util::reverse_index_bits_in_place;
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelRefIterator, ParallelIterator},
};

use super::{
    encoding::EncodingScheme,
    structure::{BasefoldCommitment, BasefoldCommitmentWithWitness, BasefoldSpec},
};

pub fn prover_query_phase<E: ExtensionField, H: MerkleHasher<E>>(
    transcript: &mut impl Transcript<E>,
    comm: &BasefoldCommitmentWithWitness<E, H>,
    trees: &[MerkleTree<E, H>],
    num_verifier_queries: usize,
) -> QueriesResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let queries: Vec<_> = (0..num_verifier_queries)
        .map(|_| {
            transcript
                .get_and_append_challenge(b"query indices")
                .elements
        })
        .collect();

    // Transform the challenge queries from field elements into integers
    let queries_usize: Vec<usize> = queries
        .iter()
        .map(|x_index| ext_to_usize(x_index) % comm.codeword_size())
        .collect_vec();

    QueriesResult {
        inner: queries_usize
            .par_iter()
            .map(|x_index| {
                (
                    *x_index,
                    basefold_get_query::<E, H>(&comm.get_codewords()[0], trees, *x_index),
                )
            })
            .collect(),
    }
}

pub fn batch_prover_query_phase<E: ExtensionField, H: MerkleHasher<E>>(
    transcript: &mut impl Transcript<E>,
    codeword_size: usize,
    comms: &[BasefoldCommitmentWithWitness<E, H>],
    trees: &[MerkleTree<E, H>],
    num_verifier_queries: usize,
) -> BatchedQueriesResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let queries: Vec<_> = (0..num_verifier_queries)
        .map(|_| {
            transcript
                .get_and_append_challenge(b"query indices")
                .elements
        })
        .collect();

    // Transform the challenge queries from field elements into integers
    let queries_usize: Vec<usize> = queries
        .iter()
        .map(|x_index| ext_to_usize(x_index) % codeword_size)
        .collect_vec();

    BatchedQueriesResult {
        inner: queries_usize
            .par_iter()
            .map(|x_index| {
                (
                    *x_index,
                    batch_basefold_get_query::<E, H>(comms, trees, codeword_size, *x_index),
                )
            })
            .collect(),
    }
}

pub fn simple_batch_prover_query_phase<E: ExtensionField, H: MerkleHasher<E>>(
    transcript: &mut impl Transcript<E>,
    comm: &BasefoldCommitmentWithWitness<E, H>,
    trees: &[MerkleTree<E, H>],
    num_verifier_queries: usize,
) -> SimpleBatchQueriesResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let queries: Vec<_> = (0..num_verifier_queries)
        .map(|_| {
            transcript
                .get_and_append_challenge(b"query indices")
                .elements
        })
        .collect();

    // Transform the challenge queries from field elements into integers
    let queries_usize: Vec<usize> = queries
        .iter()
        .map(|x_index| ext_to_usize(x_index) % comm.codeword_size())
        .collect_vec();

    SimpleBatchQueriesResult {
        inner: queries_usize
            .par_iter()
            .map(|x_index| {
                (
                    *x_index,
                    simple_batch_basefold_get_query::<E, H>(comm.get_codewords(), trees, *x_index),
                )
            })
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verifier_query_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    indices: &[usize],
    vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
    queries: &QueriesResultWithMerklePath<E, <Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
    sum_check_messages: &[Vec<E>],
    fold_challenges: &[E],
    num_rounds: usize,
    num_vars: usize,
    final_message: &[E],
    roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
    comm: &BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
    partial_eq: &[E],
    eval: &E,
) where
    E::BaseField: Serialize + DeserializeOwned,
{
    let timer = start_timer!(|| "Verifier query phase");

    let encode_timer = start_timer!(|| "Encode final codeword");
    let mut message = final_message.to_vec();
    interpolate_over_boolean_hypercube(&mut message);
    if <Spec::EncodingScheme as EncodingScheme<E>>::message_is_even_and_odd_folding() {
        reverse_index_bits_in_place(&mut message);
    }
    let final_codeword =
        <Spec::EncodingScheme as EncodingScheme<E>>::encode_small(vp, &FieldType::Ext(message));
    let mut final_codeword = match final_codeword {
        FieldType::Ext(final_codeword) => final_codeword,
        _ => panic!("Final codeword must be extension field"),
    };
    reverse_index_bits_in_place(&mut final_codeword);
    end_timer!(encode_timer);

    let queries_timer = start_timer!(|| format!("Check {} queries", indices.len()));
    queries.check::<Spec>(
        indices,
        vp,
        fold_challenges,
        num_rounds,
        num_vars,
        &final_codeword,
        roots,
        comm,
    );
    end_timer!(queries_timer);

    let final_timer = start_timer!(|| "Final checks");
    assert_eq!(eval, &degree_2_zero_plus_one(&sum_check_messages[0]));

    // The sum-check part of the protocol
    for i in 0..fold_challenges.len() - 1 {
        assert_eq!(
            degree_2_eval(&sum_check_messages[i], fold_challenges[i]),
            degree_2_zero_plus_one(&sum_check_messages[i + 1])
        );
    }

    // Finally, the last sumcheck poly evaluation should be the same as the sum of the polynomial
    // sent from the prover
    assert_eq!(
        degree_2_eval(
            &sum_check_messages[fold_challenges.len() - 1],
            fold_challenges[fold_challenges.len() - 1]
        ),
        inner_product(final_message, partial_eq)
    );
    end_timer!(final_timer);

    end_timer!(timer);
}

#[allow(clippy::too_many_arguments)]
pub fn batch_verifier_query_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    indices: &[usize],
    vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
    queries: &BatchedQueriesResultWithMerklePath<
        E,
        <Spec::MerkleHasher as MerkleHasher<E>>::Digest,
    >,
    sum_check_messages: &[Vec<E>],
    fold_challenges: &[E],
    num_rounds: usize,
    num_vars: usize,
    final_message: &[E],
    roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
    comms: &[&BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>],
    coeffs: &[E],
    partial_eq: &[E],
    eval: &E,
) where
    E::BaseField: Serialize + DeserializeOwned,
{
    let timer = start_timer!(|| "Verifier batch query phase");
    let encode_timer = start_timer!(|| "Encode final codeword");
    let mut message = final_message.to_vec();
    if <Spec::EncodingScheme as EncodingScheme<E>>::message_is_even_and_odd_folding() {
        reverse_index_bits_in_place(&mut message);
    }
    interpolate_over_boolean_hypercube(&mut message);
    let final_codeword =
        <Spec::EncodingScheme as EncodingScheme<E>>::encode_small(vp, &FieldType::Ext(message));
    let mut final_codeword = match final_codeword {
        FieldType::Ext(final_codeword) => final_codeword,
        _ => panic!("Final codeword must be extension field"),
    };
    reverse_index_bits_in_place(&mut final_codeword);
    end_timer!(encode_timer);

    // For computing the weights on the fly, because the verifier is incapable of storing
    // the weights.

    let queries_timer = start_timer!(|| format!("Check {} queries", indices.len()));
    queries.check::<Spec>(
        indices,
        vp,
        fold_challenges,
        num_rounds,
        num_vars,
        &final_codeword,
        roots,
        comms,
        coeffs,
    );
    end_timer!(queries_timer);

    #[allow(unused)]
    let final_timer = start_timer!(|| "Final checks");
    assert_eq!(eval, &degree_2_zero_plus_one(&sum_check_messages[0]));

    // The sum-check part of the protocol
    for i in 0..fold_challenges.len() - 1 {
        assert_eq!(
            degree_2_eval(&sum_check_messages[i], fold_challenges[i]),
            degree_2_zero_plus_one(&sum_check_messages[i + 1])
        );
    }

    // Finally, the last sumcheck poly evaluation should be the same as the sum of the polynomial
    // sent from the prover
    assert_eq!(
        degree_2_eval(
            &sum_check_messages[fold_challenges.len() - 1],
            fold_challenges[fold_challenges.len() - 1]
        ),
        inner_product(final_message, partial_eq)
    );
    end_timer!(final_timer);
    end_timer!(timer);
}

#[allow(clippy::too_many_arguments)]
pub fn simple_batch_verifier_query_phase<E: ExtensionField, Spec: BasefoldSpec<E>>(
    indices: &[usize],
    vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
    queries: &SimpleBatchQueriesResultWithMerklePath<
        E,
        <Spec::MerkleHasher as MerkleHasher<E>>::Digest,
    >,
    sum_check_messages: &[Vec<E>],
    fold_challenges: &[E],
    batch_coeffs: &[E],
    num_rounds: usize,
    num_vars: usize,
    final_message: &[E],
    roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
    comm: &BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
    partial_eq: &[E],
    evals: &[E],
) where
    E::BaseField: Serialize + DeserializeOwned,
{
    let timer = start_timer!(|| "Verifier query phase");

    let encode_timer = start_timer!(|| "Encode final codeword");
    let mut message = final_message.to_vec();
    if <Spec::EncodingScheme as EncodingScheme<E>>::message_is_even_and_odd_folding() {
        reverse_index_bits_in_place(&mut message);
    }
    interpolate_over_boolean_hypercube(&mut message);
    let final_codeword =
        <Spec::EncodingScheme as EncodingScheme<E>>::encode_small(vp, &FieldType::Ext(message));
    let mut final_codeword = match final_codeword {
        FieldType::Ext(final_codeword) => final_codeword,
        _ => panic!("Final codeword must be extension field"),
    };
    reverse_index_bits_in_place(&mut final_codeword);
    end_timer!(encode_timer);

    // For computing the weights on the fly, because the verifier is incapable of storing
    // the weights.
    let queries_timer = start_timer!(|| format!("Check {} queries", indices.len()));
    queries.check::<Spec>(
        indices,
        vp,
        fold_challenges,
        batch_coeffs,
        num_rounds,
        num_vars,
        &final_codeword,
        roots,
        comm,
    );
    end_timer!(queries_timer);

    let final_timer = start_timer!(|| "Final checks");
    assert_eq!(
        &inner_product(batch_coeffs, evals),
        &degree_2_zero_plus_one(&sum_check_messages[0])
    );

    // The sum-check part of the protocol
    for i in 0..fold_challenges.len() - 1 {
        assert_eq!(
            degree_2_eval(&sum_check_messages[i], fold_challenges[i]),
            degree_2_zero_plus_one(&sum_check_messages[i + 1])
        );
    }

    // Finally, the last sumcheck poly evaluation should be the same as the sum of the polynomial
    // sent from the prover
    assert_eq!(
        degree_2_eval(
            &sum_check_messages[fold_challenges.len() - 1],
            fold_challenges[fold_challenges.len() - 1]
        ),
        inner_product(final_message, partial_eq)
    );
    end_timer!(final_timer);

    end_timer!(timer);
}

fn basefold_get_query<E: ExtensionField, H: MerkleHasher<E>>(
    poly_codeword: &FieldType<E>,
    trees: &[MerkleTree<E, H>],
    x_index: usize,
) -> SingleQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let mut index = x_index;
    let p1 = index | 1;
    let p0 = p1 - 1;

    let commitment_query = match poly_codeword {
        FieldType::Ext(poly_codeword) => {
            CodewordSingleQueryResult::new_ext(poly_codeword[p0], poly_codeword[p1], p0)
        }
        FieldType::Base(poly_codeword) => {
            CodewordSingleQueryResult::new_base(poly_codeword[p0], poly_codeword[p1], p0)
        }
        _ => unreachable!(),
    };
    index >>= 1;

    let mut oracle_queries = Vec::with_capacity(trees.len() + 1);
    for tree in trees {
        let p1 = index | 1;
        let p0 = p1 - 1;

        oracle_queries.push(CodewordSingleQueryResult::new_ext(
            tree.get_leaf_as_extension(p0)[0],
            tree.get_leaf_as_extension(p1)[0],
            p0,
        ));
        index >>= 1;
    }

    let oracle_query = OracleListQueryResult {
        inner: oracle_queries,
    };

    SingleQueryResult {
        oracle_query,
        commitment_query,
    }
}

fn batch_basefold_get_query<E: ExtensionField, H: MerkleHasher<E>>(
    comms: &[BasefoldCommitmentWithWitness<E, H>],
    trees: &[MerkleTree<E, H>],
    codeword_size: usize,
    x_index: usize,
) -> BatchedSingleQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let mut oracle_list_queries = Vec::with_capacity(trees.len());

    let mut index = x_index;
    index >>= 1;
    for tree in trees {
        let p1 = index | 1;
        let p0 = p1 - 1;
        oracle_list_queries.push(CodewordSingleQueryResult::<E>::new_ext(
            tree.get_leaf_as_extension(p0)[0],
            tree.get_leaf_as_extension(p1)[0],
            p0,
        ));
        index >>= 1;
    }
    let oracle_query = OracleListQueryResult {
        inner: oracle_list_queries,
    };

    let comm_queries = comms
        .iter()
        .map(|comm| {
            let x_index = x_index >> (log2_strict(codeword_size) - comm.codeword_size_log());
            let p1 = x_index | 1;
            let p0 = p1 - 1;
            match &comm.get_codewords()[0] {
                FieldType::Ext(poly_codeword) => {
                    CodewordSingleQueryResult::new_ext(poly_codeword[p0], poly_codeword[p1], p0)
                }
                FieldType::Base(poly_codeword) => {
                    CodewordSingleQueryResult::new_base(poly_codeword[p0], poly_codeword[p1], p0)
                }
                _ => unreachable!(),
            }
        })
        .collect_vec();

    let commitments_query = CommitmentsQueryResult {
        inner: comm_queries,
    };

    BatchedSingleQueryResult {
        oracle_query,
        commitments_query,
    }
}

fn simple_batch_basefold_get_query<E: ExtensionField, H: MerkleHasher<E>>(
    poly_codewords: &[FieldType<E>],
    trees: &[MerkleTree<E, H>],
    x_index: usize,
) -> SimpleBatchSingleQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    let mut index = x_index;
    let p1 = index | 1;
    let p0 = p1 - 1;

    let commitment_query = match poly_codewords[0] {
        FieldType::Ext(_) => SimpleBatchCommitmentSingleQueryResult::new_ext(
            poly_codewords
                .iter()
                .map(|c| field_type_index_ext(c, p0))
                .collect(),
            poly_codewords
                .iter()
                .map(|c| field_type_index_ext(c, p1))
                .collect(),
            p0,
        ),
        FieldType::Base(_) => SimpleBatchCommitmentSingleQueryResult::new_base(
            poly_codewords
                .iter()
                .map(|c| field_type_index_base(c, p0))
                .collect(),
            poly_codewords
                .iter()
                .map(|c| field_type_index_base(c, p1))
                .collect(),
            p0,
        ),
        _ => unreachable!(),
    };
    index >>= 1;

    let mut oracle_queries = Vec::with_capacity(trees.len() + 1);
    for tree in trees {
        let p1 = index | 1;
        let p0 = p1 - 1;

        oracle_queries.push(CodewordSingleQueryResult::new_ext(
            tree.get_leaf_as_extension(p0)[0],
            tree.get_leaf_as_extension(p1)[0],
            p0,
        ));
        index >>= 1;
    }

    let oracle_query = OracleListQueryResult {
        inner: oracle_queries,
    };

    SimpleBatchSingleQueryResult {
        oracle_query,
        commitment_query,
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
enum CodewordPointPair<E: ExtensionField> {
    Ext(E, E),
    Base(E::BaseField, E::BaseField),
}

impl<E: ExtensionField> CodewordPointPair<E> {
    pub fn as_ext(&self) -> (E, E) {
        match self {
            CodewordPointPair::Ext(x, y) => (*x, *y),
            CodewordPointPair::Base(x, y) => (E::from(*x), E::from(*y)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
enum SimpleBatchLeavesPair<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    Ext(Vec<(E, E)>),
    Base(Vec<(E::BaseField, E::BaseField)>),
}

impl<E: ExtensionField> SimpleBatchLeavesPair<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    #[allow(unused)]
    pub fn as_ext(&self) -> Vec<(E, E)> {
        match self {
            SimpleBatchLeavesPair::Ext(x) => x.clone(),
            SimpleBatchLeavesPair::Base(x) => {
                x.iter().map(|(x, y)| ((*x).into(), (*y).into())).collect()
            }
        }
    }

    pub fn batch(&self, coeffs: &[E]) -> (E, E) {
        match self {
            SimpleBatchLeavesPair::Ext(x) => {
                let mut result = (E::ZERO, E::ZERO);
                for (i, (x, y)) in x.iter().enumerate() {
                    result.0 += coeffs[i] * *x;
                    result.1 += coeffs[i] * *y;
                }
                result
            }
            SimpleBatchLeavesPair::Base(x) => {
                let mut result = (E::ZERO, E::ZERO);
                for (i, (x, y)) in x.iter().enumerate() {
                    result.0 += coeffs[i] * *x;
                    result.1 += coeffs[i] * *y;
                }
                result
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct CodewordSingleQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    codepoints: CodewordPointPair<E>,
    index: usize,
}

impl<E: ExtensionField> CodewordSingleQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn new_ext(left: E, right: E, index: usize) -> Self {
        Self {
            codepoints: CodewordPointPair::Ext(left, right),
            index,
        }
    }

    fn new_base(left: E::BaseField, right: E::BaseField, index: usize) -> Self {
        Self {
            codepoints: CodewordPointPair::Base(left, right),
            index,
        }
    }

    fn left_ext(&self) -> E {
        match &self.codepoints {
            CodewordPointPair::Ext(x, _) => *x,
            CodewordPointPair::Base(x, _) => E::from(*x),
        }
    }

    fn right_ext(&self) -> E {
        match &self.codepoints {
            CodewordPointPair::Ext(_, y) => *y,
            CodewordPointPair::Base(_, y) => E::from(*y),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct CodewordSingleQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    query: CodewordSingleQueryResult<E>,
    merkle_path: MerklePathWithoutLeafOrRoot<E, D>,
}

impl<E: ExtensionField, D> CodewordSingleQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    pub fn check_merkle_path<H: MerkleHasher<E, Digest = D>>(&self, root: &D) {
        // let timer = start_timer!(|| "CodewordSingleQuery::Check Merkle Path");
        match self.query.codepoints {
            CodewordPointPair::Ext(left, right) => {
                self.merkle_path.authenticate_leaves_root_ext::<H>(
                    left,
                    right,
                    self.query.index,
                    root,
                );
            }
            CodewordPointPair::Base(left, right) => {
                self.merkle_path.authenticate_leaves_root_base::<H>(
                    left,
                    right,
                    self.query.index,
                    root,
                );
            }
        }
        // end_timer!(timer);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct OracleListQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<CodewordSingleQueryResult<E>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct CommitmentsQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<CodewordSingleQueryResult<E>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct OracleListQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    inner: Vec<CodewordSingleQueryResultWithMerklePath<E, D>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct CommitmentsQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    inner: Vec<CodewordSingleQueryResultWithMerklePath<E, D>>,
}

impl<E: ExtensionField> ListQueryResult<E> for OracleListQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn get_inner(&self) -> &Vec<CodewordSingleQueryResult<E>> {
        &self.inner
    }

    fn get_inner_into(self) -> Vec<CodewordSingleQueryResult<E>> {
        self.inner
    }
}

impl<E: ExtensionField> ListQueryResult<E> for CommitmentsQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn get_inner(&self) -> &Vec<CodewordSingleQueryResult<E>> {
        &self.inner
    }

    fn get_inner_into(self) -> Vec<CodewordSingleQueryResult<E>> {
        self.inner
    }
}

impl<D, E: ExtensionField, H: MerkleHasher<E, Digest = D>> ListQueryResultWithMerklePath<E, H>
    for OracleListQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    fn get_inner(&self) -> &Vec<CodewordSingleQueryResultWithMerklePath<E, D>> {
        &self.inner
    }

    fn new(inner: Vec<CodewordSingleQueryResultWithMerklePath<E, D>>) -> Self {
        Self { inner }
    }
}

impl<D, E: ExtensionField, H: MerkleHasher<E, Digest = D>> ListQueryResultWithMerklePath<E, H>
    for CommitmentsQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    fn get_inner(&self) -> &Vec<CodewordSingleQueryResultWithMerklePath<E, D>> {
        &self.inner
    }

    fn new(inner: Vec<CodewordSingleQueryResultWithMerklePath<E, D>>) -> Self {
        Self { inner }
    }
}

trait ListQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn get_inner(&self) -> &Vec<CodewordSingleQueryResult<E>>;

    fn get_inner_into(self) -> Vec<CodewordSingleQueryResult<E>>;

    fn merkle_path<H: MerkleHasher<E>>(
        &self,
        path: impl Fn(usize, usize) -> MerklePathWithoutLeafOrRoot<E, H::Digest>,
    ) -> Vec<MerklePathWithoutLeafOrRoot<E, H::Digest>> {
        let ret = self
            .get_inner()
            .iter()
            .enumerate()
            .map(|(i, query_result)| path(i, query_result.index))
            .collect_vec();
        ret
    }
}

trait ListQueryResultWithMerklePath<E: ExtensionField, H: MerkleHasher<E>>: Sized
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn new(inner: Vec<CodewordSingleQueryResultWithMerklePath<E, H::Digest>>) -> Self;

    fn get_inner(&self) -> &Vec<CodewordSingleQueryResultWithMerklePath<E, H::Digest>>;

    fn from_query_and_trees<LQR: ListQueryResult<E>>(
        query_result: LQR,
        path: impl Fn(usize, usize) -> MerklePathWithoutLeafOrRoot<E, H::Digest>,
    ) -> Self {
        Self::new(
            query_result
                .merkle_path::<H>(path)
                .into_iter()
                .zip(query_result.get_inner_into())
                .map(
                    |(path, codeword_result)| CodewordSingleQueryResultWithMerklePath {
                        query: codeword_result,
                        merkle_path: path,
                    },
                )
                .collect_vec(),
        )
    }

    fn check_merkle_paths(&self, roots: &[<H as MerkleHasher<E>>::Digest]) {
        // let timer = start_timer!(|| "ListQuery::Check Merkle Path");
        self.get_inner()
            .iter()
            .zip(roots.iter())
            .for_each(|(q, root)| {
                q.check_merkle_path::<H>(root);
            });
        // end_timer!(timer);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct SingleQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    oracle_query: OracleListQueryResult<E>,
    commitment_query: CodewordSingleQueryResult<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct SingleQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    oracle_query: OracleListQueryResultWithMerklePath<E, D>,
    commitment_query: CodewordSingleQueryResultWithMerklePath<E, D>,
}

impl<E: ExtensionField, D> SingleQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    pub fn from_single_query_result<H: MerkleHasher<E, Digest = D>>(
        single_query_result: SingleQueryResult<E>,
        oracle_trees: &[MerkleTree<E, H>],
        commitment: &BasefoldCommitmentWithWitness<E, H>,
    ) -> Self {
        assert!(commitment.codeword_tree.height() > 0);
        Self {
            oracle_query: <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<E, H>>::from_query_and_trees(
                single_query_result.oracle_query,
                |i, j| oracle_trees[i].merkle_path_without_leaf_sibling_or_root(j),
            ),
            commitment_query: CodewordSingleQueryResultWithMerklePath {
                query: single_query_result.commitment_query,
                merkle_path: commitment
                    .codeword_tree
                    .merkle_path_without_leaf_sibling_or_root(
                        single_query_result.commitment_query.index,
                    ),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check<Spec: BasefoldSpec<E>>(
        &self,
        vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
        fold_challenges: &[E],
        num_rounds: usize,
        num_vars: usize,
        final_codeword: &[E],
        roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
        comm: &BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
        index: usize,
    ) where
        Spec::MerkleHasher: MerkleHasher<E, Digest = D>,
    {
        // let timer = start_timer!(|| "Checking codeword single query");
        <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
            E,
            Spec::MerkleHasher,
        >>::check_merkle_paths(&self.oracle_query, roots);
        self.commitment_query
            .check_merkle_path::<Spec::MerkleHasher>(&comm.root);

        let (mut curr_left, mut curr_right) = self.commitment_query.query.codepoints.as_ext();

        let mut right_index = index | 1;
        let mut left_index = right_index - 1;

        for (i, fold_challenge) in fold_challenges.iter().enumerate().take(num_rounds) {
            let (x0, x1, w) = <Spec::EncodingScheme as EncodingScheme<E>>::verifier_folding_coeffs(
                vp,
                num_vars + Spec::get_rate_log() - i - 1,
                left_index >> 1,
            );

            let res = interpolate2_weights([(x0, curr_left), (x1, curr_right)], w, *fold_challenge);

            let next_index = right_index >> 1;
            let next_oracle_value = if i < num_rounds - 1 {
                right_index = next_index | 1;
                left_index = right_index - 1;
                let next_oracle_query =
                    <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
                        E,
                        Spec::MerkleHasher,
                    >>::get_inner(&self.oracle_query)[i]
                        .clone();
                (curr_left, curr_right) = next_oracle_query.query.codepoints.as_ext();
                if next_index & 1 == 0 {
                    curr_left
                } else {
                    curr_right
                }
            } else {
                // Note that final_codeword has been bit-reversed, so no need to bit-reverse
                // next_index here.
                final_codeword[next_index]
            };
            assert_eq!(res, next_oracle_value, "Failed at round {i}");
            // end_timer!(round_timer);
        }
        // end_timer!(timer);
    }
}

pub struct QueriesResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<(usize, SingleQueryResult<E>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct QueriesResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    inner: Vec<(usize, SingleQueryResultWithMerklePath<E, D>)>,
}

impl<E: ExtensionField, D> QueriesResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    pub fn empty() -> Self {
        Self { inner: vec![] }
    }

    pub fn from_query_result<H: MerkleHasher<E, Digest = D>>(
        query_result: QueriesResult<E>,
        oracle_trees: &[MerkleTree<E, H>],
        commitment: &BasefoldCommitmentWithWitness<E, H>,
    ) -> Self {
        Self {
            inner: query_result
                .inner
                .into_iter()
                .map(|(i, q)| {
                    (
                        i,
                        SingleQueryResultWithMerklePath::from_single_query_result(
                            q,
                            oracle_trees,
                            commitment,
                        ),
                    )
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check<Spec: BasefoldSpec<E>>(
        &self,
        indices: &[usize],
        vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
        fold_challenges: &[E],
        num_rounds: usize,
        num_vars: usize,
        final_codeword: &[E],
        roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
        comm: &BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
    ) where
        Spec::MerkleHasher: MerkleHasher<E, Digest = D>,
    {
        self.inner.par_iter().zip(indices.par_iter()).for_each(
            |((index, query), index_in_proof)| {
                assert_eq!(index_in_proof, index);
                query.check::<Spec>(
                    vp,
                    fold_challenges,
                    num_rounds,
                    num_vars,
                    final_codeword,
                    roots,
                    comm,
                    *index,
                );
            },
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct BatchedSingleQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    oracle_query: OracleListQueryResult<E>,
    commitments_query: CommitmentsQueryResult<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct BatchedSingleQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    oracle_query: OracleListQueryResultWithMerklePath<E, D>,
    commitments_query: CommitmentsQueryResultWithMerklePath<E, D>,
}

impl<E: ExtensionField, D> BatchedSingleQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    pub fn from_batched_single_query_result<H: MerkleHasher<E, Digest = D>>(
        batched_single_query_result: BatchedSingleQueryResult<E>,
        oracle_trees: &[MerkleTree<E, H>],
        commitments: &[BasefoldCommitmentWithWitness<E, H>],
    ) -> Self {
        Self {
            oracle_query: <OracleListQueryResultWithMerklePath<E, H::Digest> as ListQueryResultWithMerklePath<E, H>>::from_query_and_trees(
                batched_single_query_result.oracle_query,
                |i, j| oracle_trees[i].merkle_path_without_leaf_sibling_or_root(j),
            ),
            commitments_query: <CommitmentsQueryResultWithMerklePath<E, H::Digest> as ListQueryResultWithMerklePath<E, H>>::from_query_and_trees(
                batched_single_query_result.commitments_query,
                |i, j| {
                    commitments[i]
                        .codeword_tree
                        .merkle_path_without_leaf_sibling_or_root(j)
                },
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check<Spec: BasefoldSpec<E>>(
        &self,
        vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
        fold_challenges: &[E],
        num_rounds: usize,
        num_vars: usize,
        final_codeword: &[E],
        roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
        comms: &[&BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>],
        coeffs: &[E],
        index: usize,
    ) where
        Spec::MerkleHasher: MerkleHasher<E, Digest = D>,
    {
        <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
            E,
            Spec::MerkleHasher,
        >>::check_merkle_paths(&self.oracle_query, roots);
        <CommitmentsQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
            E,
            Spec::MerkleHasher,
        >>::check_merkle_paths(
            &self.commitments_query,
            comms
                .iter()
                .map(|comm| comm.root())
                .collect_vec()
                .as_slice(),
        );
        // end_timer!(commit_timer);

        let mut curr_left = E::ZERO;
        let mut curr_right = E::ZERO;

        let mut right_index = index | 1;
        let mut left_index = right_index - 1;

        for (i, fold_challenge) in fold_challenges.iter().enumerate().take(num_rounds) {
            // let round_timer = start_timer!(|| format!("BatchedSingleQueryResult::round {}", i));
            let matching_comms = comms
                .iter()
                .enumerate()
                .filter(|(_, comm)| comm.num_vars().unwrap() == num_vars - i)
                .map(|(index, _)| index)
                .collect_vec();

            matching_comms.iter().for_each(|index| {
                let query =
                    <CommitmentsQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
                        E,
                        Spec::MerkleHasher,
                    >>::get_inner(&self.commitments_query)[*index]
                        .query;
                assert_eq!(query.index >> 1, left_index >> 1);
                curr_left += query.left_ext() * coeffs[*index];
                curr_right += query.right_ext() * coeffs[*index];
            });

            let (x0, x1, w) = Spec::EncodingScheme::verifier_folding_coeffs(
                vp,
                num_vars + Spec::get_rate_log() - i - 1,
                left_index >> 1,
            );

            let mut res =
                interpolate2_weights([(x0, curr_left), (x1, curr_right)], w, *fold_challenge);

            let next_index = right_index >> 1;

            let next_oracle_value = if i < num_rounds - 1 {
                right_index = next_index | 1;
                left_index = right_index - 1;
                let next_oracle_query =
                    <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
                        E,
                        Spec::MerkleHasher,
                    >>::get_inner(&self.oracle_query)[i]
                        .clone();
                curr_left = next_oracle_query.query.left_ext();
                curr_right = next_oracle_query.query.right_ext();
                if next_index & 1 == 0 {
                    curr_left
                } else {
                    curr_right
                }
            } else {
                // Note that in the last round, res is folded to an element in the final
                // codeword, but has not yet added the committed polynomial evaluations
                // at this position.
                // So we need to repeat the finding and adding procedure here.
                // The reason for the existence of one extra find-and-add is that the number
                // of different polynomial number of variables is one more than the number of
                // rounds.

                let matching_comms = comms
                    .iter()
                    .enumerate()
                    .filter(|(_, comm)| comm.num_vars().unwrap() == num_vars - i - 1)
                    .map(|(index, _)| index)
                    .collect_vec();

                matching_comms.iter().for_each(|index| {
                    let query: CodewordSingleQueryResult<E> =
                        <CommitmentsQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<E, Spec::MerkleHasher>>::get_inner(&self.commitments_query)[*index].query;
                    assert_eq!(query.index >> 1, next_index >> 1);
                    if next_index & 1 == 0 {
                        res += query.left_ext() * coeffs[*index];
                    } else {
                        res += query.right_ext() * coeffs[*index];
                    }
                });

                // Note that final_codeword has been bit-reversed, so no need to bit-reverse
                // next_index here.
                final_codeword[next_index]
            };
            assert_eq!(res, next_oracle_value, "Failed at round {i}");
            // end_timer!(round_timer);
        }
        // end_timer!(timer);
    }
}

pub struct BatchedQueriesResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<(usize, BatchedSingleQueryResult<E>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BatchedQueriesResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    inner: Vec<(usize, BatchedSingleQueryResultWithMerklePath<E, D>)>,
}

impl<E: ExtensionField, D> BatchedQueriesResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    pub fn from_batched_query_result<H: MerkleHasher<E, Digest = D>>(
        batched_query_result: BatchedQueriesResult<E>,
        oracle_trees: &[MerkleTree<E, H>],
        commitments: &[BasefoldCommitmentWithWitness<E, H>],
    ) -> Self {
        Self {
            inner: batched_query_result
                .inner
                .into_iter()
                .map(|(i, q)| {
                    (
                        i,
                        BatchedSingleQueryResultWithMerklePath::from_batched_single_query_result(
                            q,
                            oracle_trees,
                            commitments,
                        ),
                    )
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check<Spec: BasefoldSpec<E>>(
        &self,
        indices: &[usize],
        vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
        fold_challenges: &[E],
        num_rounds: usize,
        num_vars: usize,
        final_codeword: &[E],
        roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
        comms: &[&BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>],
        coeffs: &[E],
    ) where
        Spec::MerkleHasher: MerkleHasher<E, Digest = D>,
    {
        let timer = start_timer!(|| "BatchedQueriesResult::check");
        self.inner.par_iter().zip(indices.par_iter()).for_each(
            |((index, query), index_in_proof)| {
                assert_eq!(index, index_in_proof);
                query.check::<Spec>(
                    vp,
                    fold_challenges,
                    num_rounds,
                    num_vars,
                    final_codeword,
                    roots,
                    comms,
                    coeffs,
                    *index,
                );
            },
        );
        end_timer!(timer);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct SimpleBatchCommitmentSingleQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    leaves: SimpleBatchLeavesPair<E>,
    index: usize,
}

impl<E: ExtensionField> SimpleBatchCommitmentSingleQueryResult<E>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    fn new_ext(left: Vec<E>, right: Vec<E>, index: usize) -> Self {
        Self {
            leaves: SimpleBatchLeavesPair::Ext(left.into_iter().zip(right).collect()),
            index,
        }
    }

    fn new_base(left: Vec<E::BaseField>, right: Vec<E::BaseField>, index: usize) -> Self {
        Self {
            leaves: SimpleBatchLeavesPair::Base(left.into_iter().zip(right).collect()),
            index,
        }
    }

    #[allow(unused)]
    fn left_ext(&self) -> Vec<E> {
        match &self.leaves {
            SimpleBatchLeavesPair::Ext(x) => x.iter().map(|(x, _)| *x).collect(),
            SimpleBatchLeavesPair::Base(x) => x.iter().map(|(x, _)| E::from(*x)).collect(),
        }
    }

    #[allow(unused)]
    fn right_ext(&self) -> Vec<E> {
        match &self.leaves {
            SimpleBatchLeavesPair::Ext(x) => x.iter().map(|(_, x)| *x).collect(),
            SimpleBatchLeavesPair::Base(x) => x.iter().map(|(_, x)| E::from(*x)).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct SimpleBatchCommitmentSingleQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    query: SimpleBatchCommitmentSingleQueryResult<E>,
    merkle_path: MerklePathWithoutLeafOrRoot<E, D>,
}

impl<E: ExtensionField, D> SimpleBatchCommitmentSingleQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    pub fn check_merkle_path<H: MerkleHasher<E, Digest = D>>(&self, root: &D) {
        // let timer = start_timer!(|| "CodewordSingleQuery::Check Merkle Path");
        match &self.query.leaves {
            SimpleBatchLeavesPair::Ext(inner) => {
                self.merkle_path.authenticate_batch_leaves_root_ext::<H>(
                    inner.iter().map(|(x, _)| *x).collect(),
                    inner.iter().map(|(_, x)| *x).collect(),
                    self.query.index,
                    root,
                );
            }
            SimpleBatchLeavesPair::Base(inner) => {
                self.merkle_path.authenticate_batch_leaves_root_base::<H>(
                    inner.iter().map(|(x, _)| *x).collect(),
                    inner.iter().map(|(_, x)| *x).collect(),
                    self.query.index,
                    root,
                );
            }
        }
        // end_timer!(timer);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct SimpleBatchSingleQueryResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    oracle_query: OracleListQueryResult<E>,
    commitment_query: SimpleBatchCommitmentSingleQueryResult<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
struct SimpleBatchSingleQueryResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    oracle_query: OracleListQueryResultWithMerklePath<E, D>,
    commitment_query: SimpleBatchCommitmentSingleQueryResultWithMerklePath<E, D>,
}

impl<E: ExtensionField, D> SimpleBatchSingleQueryResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    pub fn from_single_query_result<H: MerkleHasher<E, Digest = D>>(
        single_query_result: SimpleBatchSingleQueryResult<E>,
        oracle_trees: &[MerkleTree<E, H>],
        commitment: &BasefoldCommitmentWithWitness<E, H>,
    ) -> Self {
        Self {
            oracle_query: <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<E, H>>::from_query_and_trees(
                single_query_result.oracle_query,
                |i, j| oracle_trees[i].merkle_path_without_leaf_sibling_or_root(j),
            ),
            commitment_query: SimpleBatchCommitmentSingleQueryResultWithMerklePath {
                query: single_query_result.commitment_query.clone(),
                merkle_path: commitment
                    .codeword_tree
                    .merkle_path_without_leaf_sibling_or_root(
                        single_query_result.commitment_query.index,
                    ),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check<Spec: BasefoldSpec<E>>(
        &self,
        vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
        fold_challenges: &[E],
        batch_coeffs: &[E],
        num_rounds: usize,
        num_vars: usize,
        final_codeword: &[E],
        roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
        comm: &BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
        index: usize,
    ) where
        Spec::MerkleHasher: MerkleHasher<E, Digest = D>,
    {
        <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
            E,
            Spec::MerkleHasher,
        >>::check_merkle_paths(&self.oracle_query, roots);
        self.commitment_query
            .check_merkle_path::<Spec::MerkleHasher>(&comm.root());

        let (mut curr_left, mut curr_right) =
            self.commitment_query.query.leaves.batch(batch_coeffs);

        let mut right_index = index | 1;
        let mut left_index = right_index - 1;

        for (i, fold_challenge) in fold_challenges.iter().enumerate().take(num_rounds) {
            // let round_timer = start_timer!(|| format!("SingleQueryResult::round {}", i));

            let (x0, x1, w) = <Spec::EncodingScheme as EncodingScheme<E>>::verifier_folding_coeffs(
                vp,
                num_vars + Spec::get_rate_log() - i - 1,
                left_index >> 1,
            );

            let res = interpolate2_weights([(x0, curr_left), (x1, curr_right)], w, *fold_challenge);

            let next_index = right_index >> 1;
            let next_oracle_value = if i < num_rounds - 1 {
                right_index = next_index | 1;
                left_index = right_index - 1;
                let next_oracle_query =
                    <OracleListQueryResultWithMerklePath<E, D> as ListQueryResultWithMerklePath<
                        E,
                        Spec::MerkleHasher,
                    >>::get_inner(&self.oracle_query)[i]
                        .clone();
                (curr_left, curr_right) = next_oracle_query.query.codepoints.as_ext();
                if next_index & 1 == 0 {
                    curr_left
                } else {
                    curr_right
                }
            } else {
                // Note that final_codeword has been bit-reversed, so no need to bit-reverse
                // next_index here.
                final_codeword[next_index]
            };
            assert_eq!(res, next_oracle_value, "Failed at round {i}");
            // end_timer!(round_timer);
        }
    }
}

pub struct SimpleBatchQueriesResult<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<(usize, SimpleBatchSingleQueryResult<E>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct SimpleBatchQueriesResultWithMerklePath<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Serialize + DeserializeOwned,
{
    inner: Vec<(usize, SimpleBatchSingleQueryResultWithMerklePath<E, D>)>,
}

impl<E: ExtensionField, D> SimpleBatchQueriesResultWithMerklePath<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
    D: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    pub fn from_query_result<H: MerkleHasher<E, Digest = D>>(
        query_result: SimpleBatchQueriesResult<E>,
        oracle_trees: &[MerkleTree<E, H>],
        commitment: &BasefoldCommitmentWithWitness<E, H>,
    ) -> Self {
        Self {
            inner: query_result
                .inner
                .into_iter()
                .map(|(i, q)| {
                    (
                        i,
                        SimpleBatchSingleQueryResultWithMerklePath::from_single_query_result(
                            q,
                            oracle_trees,
                            commitment,
                        ),
                    )
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check<Spec: BasefoldSpec<E>>(
        &self,
        indices: &[usize],
        vp: &<Spec::EncodingScheme as EncodingScheme<E>>::VerifierParameters,
        fold_challenges: &[E],
        batch_coeffs: &[E],
        num_rounds: usize,
        num_vars: usize,
        final_codeword: &[E],
        roots: &[<Spec::MerkleHasher as MerkleHasher<E>>::Digest],
        comm: &BasefoldCommitment<<Spec::MerkleHasher as MerkleHasher<E>>::Digest>,
    ) where
        Spec::MerkleHasher: MerkleHasher<E, Digest = D>,
    {
        self.inner.par_iter().zip(indices.par_iter()).for_each(
            |((index, query), index_in_proof)| {
                assert_eq!(index, index_in_proof);
                query.check::<Spec>(
                    vp,
                    fold_challenges,
                    batch_coeffs,
                    num_rounds,
                    num_vars,
                    final_codeword,
                    roots,
                    comm,
                    *index,
                );
            },
        );
    }
}
