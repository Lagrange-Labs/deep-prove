use std::marker::PhantomData;

use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::mle::FieldType;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};

use crate::util::{
    Deserialize, DeserializeOwned, Serialize, field_type_index_base, field_type_index_ext,
    hash::MerkleHasher, log2_strict,
};
use transcript::Transcript;

use ark_std::{end_timer, start_timer};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct MerkleTree<E: ExtensionField, H: MerkleHasher<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<Vec<H::Digest>>,
    leaves: Vec<FieldType<E>>,
    _phantom: PhantomData<H>,
}

impl<E: ExtensionField, H: MerkleHasher<E>> MerkleTree<E, H>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn compute_inner(leaves: &FieldType<E>) -> Vec<Vec<H::Digest>> {
        merkelize::<E, H>(&[leaves])
    }

    pub fn compute_inner_base(leaves: &[E::BaseField]) -> Vec<Vec<H::Digest>> {
        merkelize_base::<E, H>(&[leaves])
    }

    pub fn compute_inner_ext(leaves: &[E]) -> Vec<Vec<H::Digest>> {
        merkelize_ext::<E, H>(&[leaves])
    }

    pub fn root_from_inner(inner: &[Vec<H::Digest>]) -> H::Digest {
        inner.last().unwrap()[0].clone()
    }

    pub fn from_inner_leaves(inner: Vec<Vec<H::Digest>>, leaves: FieldType<E>) -> Self {
        Self {
            inner,
            leaves: vec![leaves],
            _phantom: PhantomData,
        }
    }

    pub fn from_leaves(leaves: FieldType<E>) -> Self {
        Self {
            inner: Self::compute_inner(&leaves),
            leaves: vec![leaves],
            _phantom: PhantomData,
        }
    }

    pub fn from_batch_leaves(leaves: Vec<FieldType<E>>) -> Self {
        Self {
            inner: merkelize::<E, H>(&leaves.iter().collect_vec()),
            leaves,
            _phantom: PhantomData,
        }
    }

    pub fn root(&self) -> H::Digest {
        Self::root_from_inner(&self.inner)
    }

    pub fn root_ref(&self) -> &H::Digest {
        &self.inner.last().unwrap()[0]
    }

    pub fn height(&self) -> usize {
        self.inner.len()
    }

    pub fn leaves(&self) -> &Vec<FieldType<E>> {
        &self.leaves
    }

    pub fn batch_leaves(&self, coeffs: &[E]) -> Vec<E> {
        (0..self.leaves[0].len())
            .into_par_iter()
            .map(|i| {
                self.leaves
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(leaf, coeff)| field_type_index_ext(leaf, i) * *coeff)
                    .sum()
            })
            .collect()
    }

    pub fn size(&self) -> (usize, usize) {
        (self.leaves.len(), self.leaves[0].len())
    }

    pub fn get_leaf_as_base(&self, index: usize) -> Vec<E::BaseField> {
        match &self.leaves[0] {
            FieldType::Base(_) => self
                .leaves
                .iter()
                .map(|leaves| field_type_index_base(leaves, index))
                .collect(),
            FieldType::Ext(_) => panic!(
                "Mismatching field type, calling get_leaf_as_base on a Merkle tree over extension fields"
            ),
            FieldType::Unreachable => unreachable!(),
        }
    }

    pub fn get_leaf_as_extension(&self, index: usize) -> Vec<E> {
        match &self.leaves[0] {
            FieldType::Base(_) => self
                .leaves
                .iter()
                .map(|leaves| field_type_index_ext(leaves, index))
                .collect(),
            FieldType::Ext(_) => self
                .leaves
                .iter()
                .map(|leaves| field_type_index_ext(leaves, index))
                .collect(),
            FieldType::Unreachable => unreachable!(),
        }
    }

    pub fn merkle_path_without_leaf_sibling_or_root(
        &self,
        leaf_index: usize,
    ) -> MerklePathWithoutLeafOrRoot<E, <H as MerkleHasher<E>>::Digest> {
        assert!(leaf_index < self.size().1);
        MerklePathWithoutLeafOrRoot::<E, <H as MerkleHasher<E>>::Digest>::new(
            self.inner
                .iter()
                .take(self.height() - 1)
                .enumerate()
                .map(|(index, layer)| layer[(leaf_index >> (index + 1)) ^ 1].clone())
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MerklePathWithoutLeafOrRoot<E: ExtensionField, D>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    inner: Vec<D>,
    _phantom: PhantomData<E>,
}

impl<E: ExtensionField, D> MerklePathWithoutLeafOrRoot<E, D>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn new(inner: Vec<D>) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &D> {
        self.inner.iter()
    }

    pub fn write_transcript<H: MerkleHasher<E, Digest = D>>(
        &self,
        transcript: &mut impl Transcript<E>,
    ) {
        self.inner
            .iter()
            .for_each(|hash| H::digest_to_transcript(hash, transcript));
    }

    pub fn authenticate_leaves_root_ext<H: MerkleHasher<E, Digest = D>>(
        &self,
        left: E,
        right: E,
        index: usize,
        root: &D,
    ) {
        authenticate_merkle_path_root::<E, H>(
            &self.inner,
            FieldType::Ext(vec![left, right]),
            index,
            root,
        )
    }

    pub fn authenticate_leaves_root_base<H: MerkleHasher<E, Digest = D>>(
        &self,
        left: E::BaseField,
        right: E::BaseField,
        index: usize,
        root: &<H as MerkleHasher<E>>::Digest,
    ) {
        authenticate_merkle_path_root::<E, H>(
            &self.inner,
            FieldType::Base(vec![left, right]),
            index,
            root,
        )
    }

    pub fn authenticate_batch_leaves_root_ext<H: MerkleHasher<E, Digest = D>>(
        &self,
        left: Vec<E>,
        right: Vec<E>,
        index: usize,
        root: &<H as MerkleHasher<E>>::Digest,
    ) {
        authenticate_merkle_path_root_batch::<E, H>(
            &self.inner,
            FieldType::Ext(left),
            FieldType::Ext(right),
            index,
            root,
        )
    }

    pub fn authenticate_batch_leaves_root_base<H: MerkleHasher<E, Digest = D>>(
        &self,
        left: Vec<E::BaseField>,
        right: Vec<E::BaseField>,
        index: usize,
        root: &<H as MerkleHasher<E>>::Digest,
    ) {
        authenticate_merkle_path_root_batch::<E, H>(
            &self.inner,
            FieldType::Base(left),
            FieldType::Base(right),
            index,
            root,
        )
    }
}

/// Merkle tree construction
/// TODO: Support merkelizing mixed-type values
fn merkelize<E: ExtensionField, H: MerkleHasher<E>>(
    values: &[&FieldType<E>],
) -> Vec<Vec<H::Digest>> {
    #[cfg(feature = "sanity-check")]
    for i in 0..(values.len() - 1) {
        assert_eq!(values[i].len(), values[i + 1].len());
    }
    let timer = start_timer!(|| format!("merkelize {} values", values[0].len() * values.len()));
    let log_v = log2_strict(values[0].len());
    let mut tree = Vec::with_capacity(log_v);
    // The first layer of hashes, half the number of leaves
    let mut hashes = vec![H::Digest::default(); values[0].len() >> 1];
    if values.len() == 1 {
        hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
            *hash = match &values[0] {
                FieldType::Base(values) => {
                    H::hash_two_leaves_base(&values[i << 1], &values[(i << 1) + 1])
                }
                FieldType::Ext(values) => {
                    H::hash_two_leaves(&values[i << 1], &values[(i << 1) + 1])
                }
                FieldType::Unreachable => unreachable!(),
            };
        });
    } else {
        hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
            *hash = match &values[0] {
                FieldType::Base(_) => H::hash_two_leaves_batch_base(
                    values
                        .iter()
                        .map(|values| field_type_index_base(values, i << 1))
                        .collect_vec()
                        .as_slice(),
                    values
                        .iter()
                        .map(|values| field_type_index_base(values, (i << 1) + 1))
                        .collect_vec()
                        .as_slice(),
                ),
                FieldType::Ext(_) => H::hash_two_leaves_batch_ext(
                    values
                        .iter()
                        .map(|values| field_type_index_ext(values, i << 1))
                        .collect_vec()
                        .as_slice(),
                    values
                        .iter()
                        .map(|values| field_type_index_ext(values, (i << 1) + 1))
                        .collect_vec()
                        .as_slice(),
                ),
                FieldType::Unreachable => unreachable!(),
            };
        });
    }

    tree.push(hashes);

    for i in 1..(log_v) {
        let oracle = tree[i - 1]
            .par_chunks_exact(2)
            .map(|ys| H::hash_two_digests(&ys[0], &ys[1]))
            .collect::<Vec<_>>();

        tree.push(oracle);
    }
    end_timer!(timer);
    tree
}

fn merkelize_base<E: ExtensionField, H: MerkleHasher<E>>(
    values: &[&[E::BaseField]],
) -> Vec<Vec<<H as MerkleHasher<E>>::Digest>> {
    #[cfg(feature = "sanity-check")]
    for i in 0..(values.len() - 1) {
        assert_eq!(values[i].len(), values[i + 1].len());
    }
    let timer = start_timer!(|| format!("merkelize {} values", values[0].len() * values.len()));
    let log_v = log2_strict(values[0].len());
    let mut tree = Vec::with_capacity(log_v);
    // The first layer of hashes, half the number of leaves
    let mut hashes = vec![H::Digest::default(); values[0].len() >> 1];
    if values.len() == 1 {
        hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
            *hash = H::hash_two_leaves_base(&values[0][i << 1], &values[0][(i << 1) + 1]);
        });
    } else {
        hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
            *hash = H::hash_two_leaves_batch_base(
                values
                    .iter()
                    .map(|values| values[i << 1])
                    .collect_vec()
                    .as_slice(),
                values
                    .iter()
                    .map(|values| values[(i << 1) + 1])
                    .collect_vec()
                    .as_slice(),
            );
        });
    }

    tree.push(hashes);

    for i in 1..(log_v) {
        let oracle = tree[i - 1]
            .par_chunks_exact(2)
            .map(|ys| H::hash_two_digests(&ys[0], &ys[1]))
            .collect::<Vec<_>>();
        tree.push(oracle);
    }
    end_timer!(timer);
    tree
}

fn merkelize_ext<E: ExtensionField, H: MerkleHasher<E>>(values: &[&[E]]) -> Vec<Vec<H::Digest>> {
    #[cfg(feature = "sanity-check")]
    for i in 0..(values.len() - 1) {
        assert_eq!(values[i].len(), values[i + 1].len());
    }
    let timer = start_timer!(|| format!("merkelize {} values", values[0].len() * values.len()));
    let log_v = log2_strict(values[0].len());
    let mut tree = Vec::with_capacity(log_v);
    // The first layer of hashes, half the number of leaves
    let mut hashes = vec![H::Digest::default(); values[0].len() >> 1];
    if values.len() == 1 {
        hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
            *hash = H::hash_two_leaves(&values[0][i << 1], &values[0][(i << 1) + 1]);
        });
    } else {
        hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
            *hash = H::hash_two_leaves_batch_ext(
                values
                    .iter()
                    .map(|values| values[i << 1])
                    .collect_vec()
                    .as_slice(),
                values
                    .iter()
                    .map(|values| values[(i << 1) + 1])
                    .collect_vec()
                    .as_slice(),
            );
        });
    }

    tree.push(hashes);

    for i in 1..(log_v) {
        let oracle = tree[i - 1]
            .par_chunks_exact(2)
            .map(|ys| H::hash_two_digests(&ys[0], &ys[1]))
            .collect::<Vec<_>>();

        tree.push(oracle);
    }
    end_timer!(timer);
    tree
}

fn authenticate_merkle_path_root<E: ExtensionField, H: MerkleHasher<E>>(
    path: &[<H as MerkleHasher<E>>::Digest],
    leaves: FieldType<E>,
    x_index: usize,
    root: &<H as MerkleHasher<E>>::Digest,
) {
    let mut x_index = x_index;
    assert_eq!(leaves.len(), 2);
    let mut hash = match leaves {
        FieldType::Base(leaves) => H::hash_two_leaves_base(&leaves[0], &leaves[1]),
        FieldType::Ext(leaves) => H::hash_two_leaves(&leaves[0], &leaves[1]),
        FieldType::Unreachable => unreachable!(),
    };

    // The lowest bit in the index is ignored. It can point to either leaves
    x_index >>= 1;
    for path_i in path.iter() {
        hash = if x_index & 1 == 0 {
            H::hash_two_digests(&hash, path_i)
        } else {
            H::hash_two_digests(path_i, &hash)
        };
        x_index >>= 1;
    }
    assert_eq!(&hash, root);
}

fn authenticate_merkle_path_root_batch<E: ExtensionField, H: MerkleHasher<E>>(
    path: &[H::Digest],
    left: FieldType<E>,
    right: FieldType<E>,
    x_index: usize,
    root: &H::Digest,
) {
    let mut x_index = x_index;
    let mut hash = if left.len() > 1 {
        match (left, right) {
            (FieldType::Base(left), FieldType::Base(right)) => {
                H::hash_two_leaves_batch_base(&left, &right)
            }
            (FieldType::Ext(left), FieldType::Ext(right)) => {
                H::hash_two_leaves_batch_ext(&left, &right)
            }
            _ => unreachable!(),
        }
    } else {
        match (left, right) {
            (FieldType::Base(left), FieldType::Base(right)) => {
                H::hash_two_leaves_base(&left[0], &right[0])
            }
            (FieldType::Ext(left), FieldType::Ext(right)) => {
                H::hash_two_leaves(&left[0], &right[0])
            }
            _ => unreachable!(),
        }
    };

    // The lowest bit in the index is ignored. It can point to either leaves
    x_index >>= 1;
    for path_i in path.iter() {
        hash = if x_index & 1 == 0 {
            H::hash_two_digests(&hash, path_i)
        } else {
            H::hash_two_digests(path_i, &hash)
        };
        x_index >>= 1;
    }
    assert_eq!(&hash, root);
}
