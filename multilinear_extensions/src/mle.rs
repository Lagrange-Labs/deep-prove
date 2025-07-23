use std::{any::TypeId, borrow::Cow, mem, sync::Arc};

use crate::{op_mle, util::ceil_log2};
use ark_std::{end_timer, rand::RngCore, start_timer};
use core::hash::Hash;
use ff_ext::{ExtensionField, FromUniformBytes};
use p3_field::{Field, FieldAlgebra};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt::Debug;

pub trait MultilinearExtension<E: ExtensionField>: Send + Sync {
    type Output;
    fn fix_variables(&self, partial_point: &[E]) -> Self::Output;
    fn fix_variables_in_place(&mut self, partial_point: &[E]);
    fn fix_high_variables(&self, partial_point: &[E]) -> Self::Output;
    fn fix_high_variables_in_place(&mut self, partial_point: &[E]);
    fn evaluate(&self, point: &[E]) -> E;
    fn num_vars(&self) -> usize;
    fn evaluations(&self) -> &FieldType<E>;
    fn evaluations_range(&self) -> Option<(usize, usize)>; // start offset
    fn evaluations_to_owned(self) -> FieldType<E>;
    fn merge(&mut self, rhs: Self::Output);
    fn get_ranged_mle(
        &self,
        num_range: usize,
        range_index: usize,
    ) -> RangedMultilinearExtension<'_, E>;
    #[deprecated = "TODO try to redesign this api for it's costly and create a new DenseMultilinearExtension "]
    fn resize_ranged(
        &self,
        num_instances: usize,
        new_size_per_instance: usize,
        num_range: usize,
        range_index: usize,
    ) -> DenseMultilinearExtension<E>;
    fn dup(&self, num_instances: usize, num_dups: usize) -> DenseMultilinearExtension<E>;

    fn fix_variables_parallel(&self, partial_point: &[E]) -> Self::Output;
    fn fix_variables_in_place_parallel(&mut self, partial_point: &[E]);

    fn name(&self) -> &'static str;

    fn get_ext_field_vec(&self) -> &[E] {
        match &self.evaluations() {
            FieldType::Ext(evaluations) => {
                let (start, offset) = self.evaluations_range().unwrap_or((0, evaluations.len()));
                &evaluations[start..][..offset]
            }
            _ => panic!("evaluation not in extension field"),
        }
    }

    fn get_base_field_vec(&self) -> &[E::BaseField] {
        match &self.evaluations() {
            FieldType::Base(evaluations) => {
                let (start, offset) = self.evaluations_range().unwrap_or((0, evaluations.len()));
                &evaluations[start..][..offset]
            }
            _ => panic!("evaluation not in base field"),
        }
    }
}

impl<E: ExtensionField> Debug for dyn MultilinearExtension<E, Output = DenseMultilinearExtension<E>> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.evaluations())
    }
}

impl<E: ExtensionField> From<Vec<Vec<E::BaseField>>> for DenseMultilinearExtension<E> {
    fn from(val: Vec<Vec<E::BaseField>>) -> Self {
        let per_instance_size = val[0].len();
        let next_pow2_per_instance_size = ceil_log2(per_instance_size);
        let evaluations = val
            .into_iter()
            .enumerate()
            .flat_map(|(i, mut instance)| {
                assert_eq!(
                    instance.len(),
                    per_instance_size,
                    "{}th instance with length {} != {} ",
                    i,
                    instance.len(),
                    per_instance_size
                );
                instance.resize(1 << next_pow2_per_instance_size, E::BaseField::ZERO);
                instance
            })
            .collect::<Vec<E::BaseField>>();
        assert!(evaluations.len().is_power_of_two());
        let num_vars = ceil_log2(evaluations.len());
        DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations)
    }
}

/// this is to avoid conflict implementation for Into of Vec<Vec<E::BaseField>>
pub trait IntoMLE<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into_mle(self) -> T;
}

impl<F: Field, E: ExtensionField> IntoMLE<DenseMultilinearExtension<E>> for Vec<F> {
    fn into_mle(self) -> DenseMultilinearExtension<E> {
        let next_pow2 = self.len().next_power_of_two();
        assert!(self.len().is_power_of_two(), "{}", self.len());
        DenseMultilinearExtension::from_evaluation_vec_smart::<F>(ceil_log2(next_pow2), self)
    }
}

impl<F: Field, E: ExtensionField> IntoMLE<DenseMultilinearExtension<E>> for &[F] {
    fn into_mle(self) -> DenseMultilinearExtension<E> {
        self.to_vec().into_mle()
    }
}

pub trait IntoMLEs<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into_mles(self) -> Vec<T>;
}

impl<F: Field, E: ExtensionField> IntoMLEs<DenseMultilinearExtension<E>> for Vec<Vec<F>> {
    fn into_mles(self) -> Vec<DenseMultilinearExtension<E>> {
        self.into_iter().map(|v| v.into_mle()).collect()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Default, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
/// Differentiate inner vector on base/extension field.
pub enum FieldType<E: ExtensionField> {
    Base(#[serde(skip)] Vec<E::BaseField>),
    Ext(Vec<E>),
    #[default]
    Unreachable,
}

impl<E: ExtensionField> FieldType<E> {
    pub fn len(&self) -> usize {
        match self {
            FieldType::Base(content) => content.len(),
            FieldType::Ext(content) => content.len(),
            FieldType::Unreachable => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            FieldType::Base(content) => content.is_empty(),
            FieldType::Ext(content) => content.is_empty(),
            FieldType::Unreachable => true,
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            FieldType::Base(_) => "Base",
            FieldType::Ext(_) => "Ext",
            FieldType::Unreachable => "Unreachable",
        }
    }
}

/// Stores a multilinear polynomial in dense evaluation form.
#[derive(Clone, PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct DenseMultilinearExtension<E: ExtensionField> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: FieldType<E>,
    /// Number of variables
    pub num_vars: usize,
}

impl<E: ExtensionField> From<DenseMultilinearExtension<E>>
    for Arc<dyn MultilinearExtension<E, Output = DenseMultilinearExtension<E>>>
{
    fn from(
        mle: DenseMultilinearExtension<E>,
    ) -> Arc<dyn MultilinearExtension<E, Output = DenseMultilinearExtension<E>>> {
        Arc::new(mle)
    }
}

pub type ArcDenseMultilinearExtension<E> = Arc<DenseMultilinearExtension<E>>;

fn cast_vec<A, B>(mut vec: Vec<A>) -> Vec<B> {
    let length = vec.len();
    let capacity = vec.capacity();
    let ptr = vec.as_mut_ptr();
    // Prevent `vec` from dropping its contents
    mem::forget(vec);

    // Convert the pointer to the new type
    let new_ptr = ptr as *mut B;

    // Create a new vector with the same length and capacity, but different type
    unsafe { Vec::from_raw_parts(new_ptr, length, capacity) }
}

impl<E: ExtensionField> DenseMultilinearExtension<E> {
    /// This function can tell T being Field or ExtensionField and invoke respective function
    pub fn from_evaluation_vec_smart<T: Clone + 'static>(
        num_vars: usize,
        evaluations: Vec<T>,
    ) -> Self {
        if TypeId::of::<T>() == TypeId::of::<E>() {
            return Self::from_evaluations_ext_vec(num_vars, cast_vec(evaluations));
        }

        if TypeId::of::<T>() == TypeId::of::<E::BaseField>() {
            return Self::from_evaluations_vec(num_vars, cast_vec(evaluations));
        }

        unimplemented!("type not support")
    }

    /// Construct a new polynomial from a list of evaluations where the index
    /// represents a point in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[E::BaseField]) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec())
    }

    /// Construct a new polynomial from a list of evaluations where the index
    /// represents a point in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<E::BaseField>) -> Self {
        // assert that the number of variables matches the size of evaluations
        // TODO: return error.
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations: FieldType::Base(evaluations),
        }
    }

    /// Identical to [`from_evaluations_slice`], with and exception that evaluation vector is in
    /// extension field
    pub fn from_evaluations_ext_slice(num_vars: usize, evaluations: &[E]) -> Self {
        Self::from_evaluations_ext_vec(num_vars, evaluations.to_vec())
    }

    /// Identical to [`from_evaluations_vec`], with and exception that evaluation vector is in
    /// extension field
    pub fn from_evaluations_ext_vec(num_vars: usize, evaluations: Vec<E>) -> Self {
        // assert that the number of variables matches the size of evaluations
        // TODO: return error.
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations: FieldType::Ext(evaluations),
        }
    }

    /// Generate a random evaluation of a multilinear poly
    pub fn random(nv: usize, mut rng: &mut impl RngCore) -> Self {
        let eval = (0..1 << nv)
            .map(|_| E::BaseField::random(&mut rng))
            .collect();
        DenseMultilinearExtension::from_evaluations_vec(nv, eval)
    }

    /// Sample a random list of multilinear polynomials.
    /// Returns
    /// - the list of polynomials,
    /// - its sum of polynomial evaluations over the boolean hypercube.
    pub fn random_mle_list(
        nv: usize,
        degree: usize,
        mut rng: &mut impl RngCore,
    ) -> (Vec<ArcDenseMultilinearExtension<E>>, E) {
        let start = start_timer!(|| "sample random mle list");
        let mut multiplicands = Vec::with_capacity(degree);
        for _ in 0..degree {
            multiplicands.push(Vec::with_capacity(1 << nv))
        }
        let mut sum = E::ZERO;

        for _ in 0..(1 << nv) {
            let mut product = E::ONE;

            for e in multiplicands.iter_mut() {
                let val = E::BaseField::random(&mut rng);
                e.push(val);
                product *= val
            }
            sum += product;
        }

        let list = multiplicands
            .into_iter()
            .map(|x| DenseMultilinearExtension::from_evaluations_vec(nv, x).into())
            .collect();

        end_timer!(start);
        (list, sum)
    }

    // Build a randomize list of mle-s whose sum is zero.
    pub fn random_zero_mle_list(
        nv: usize,
        degree: usize,
        mut rng: impl RngCore,
    ) -> Vec<ArcDenseMultilinearExtension<E>> {
        let start = start_timer!(|| "sample random zero mle list");

        let mut multiplicands = Vec::with_capacity(degree);
        for _ in 0..degree {
            multiplicands.push(Vec::with_capacity(1 << nv))
        }
        for _ in 0..(1 << nv) {
            multiplicands[0].push(E::BaseField::ZERO);
            for e in multiplicands.iter_mut().skip(1) {
                e.push(E::BaseField::random(&mut rng));
            }
        }

        let list = multiplicands
            .into_iter()
            .map(|x| DenseMultilinearExtension::from_evaluations_vec(nv, x).into())
            .collect();

        end_timer!(start);
        list
    }

    pub fn to_ext_field(&self) -> Self {
        match &self.evaluations {
            FieldType::Base(evaluations) => DenseMultilinearExtension::from_evaluations_ext_vec(
                self.num_vars(),
                evaluations.iter().map(|&x| E::from_base(x)).collect(),
            ),
            FieldType::Ext(evaluations) => DenseMultilinearExtension::from_evaluations_ext_vec(
                self.num_vars(),
                evaluations.clone(),
            ),
            FieldType::Unreachable => unreachable!(),
        }
    }
}

#[allow(clippy::wrong_self_convention)]
pub trait IntoInstanceIter<'a, T> {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
    fn into_instance_iter(&self, n_instances: usize) -> Self::IntoIter;
}

#[allow(clippy::wrong_self_convention)]
pub trait IntoInstanceIterMut<'a, T> {
    type ItemMut;
    type IntoIterMut: Iterator<Item = Self::ItemMut>;
    fn into_instance_iter_mut(&'a mut self, n_instances: usize) -> Self::IntoIterMut;
}

pub struct InstanceIntoIterator<'a, T> {
    pub evaluations: &'a [T],
    pub start: usize,
    pub offset: usize,
}

pub struct InstanceIntoIteratorMut<'a, T> {
    pub evaluations: &'a mut [T],
    pub start: usize,
    pub offset: usize,
    pub origin_len: usize,
}

impl<'a, T> Iterator for InstanceIntoIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.evaluations.len() {
            None
        } else {
            let next = &self.evaluations[self.start..][..self.offset];
            self.start += self.offset;
            Some(next)
        }
    }
}

impl<'a, T> Iterator for InstanceIntoIteratorMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.origin_len {
            None
        } else {
            let evaluation = mem::take(&mut self.evaluations);
            let (head, tail) = evaluation.split_at_mut(self.offset);
            self.evaluations = tail;
            self.start += self.offset;
            Some(head)
        }
    }
}

impl<'a, T> IntoInstanceIter<'a, T> for &'a [T] {
    type Item = &'a [T];
    type IntoIter = InstanceIntoIterator<'a, T>;

    fn into_instance_iter(&self, n_instances: usize) -> Self::IntoIter {
        assert!(self.len().is_multiple_of(n_instances));
        let offset = self.len() / n_instances;
        InstanceIntoIterator {
            evaluations: self,
            start: 0,
            offset,
        }
    }
}

impl<'a, T: 'a> IntoInstanceIterMut<'a, T> for Vec<T> {
    type ItemMut = &'a mut [T];
    type IntoIterMut = InstanceIntoIteratorMut<'a, T>;

    fn into_instance_iter_mut<'b>(&'a mut self, n_instances: usize) -> Self::IntoIterMut {
        assert!(self.len().is_multiple_of(n_instances));
        let offset = self.len() / n_instances;
        let origin_len = self.len();
        InstanceIntoIteratorMut {
            evaluations: self,
            start: 0,
            offset,
            origin_len,
        }
    }
}

impl<E: ExtensionField> MultilinearExtension<E> for DenseMultilinearExtension<E> {
    type Output = DenseMultilinearExtension<E>;
    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point`.
    fn fix_variables(&self, partial_point: &[E]) -> Self {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        let mut poly = Cow::Borrowed(self);

        // evaluate single variable of partial point from left to right
        // `Cow` type here to skip first evaluation vector copy
        for point in partial_point.iter() {
            match &mut poly {
                poly @ Cow::Borrowed(_) => {
                    *poly = op_mle!(self, |evaluations| {
                        Cow::Owned(DenseMultilinearExtension::from_evaluations_ext_vec(
                            self.num_vars() - 1,
                            evaluations
                                .chunks(2)
                                .map(|buf| *point * (buf[1] - buf[0]) + buf[0])
                                .collect(),
                        ))
                    });
                }
                Cow::Owned(poly) => poly.fix_variables_in_place(&[*point]),
            }
        }
        assert!(poly.num_vars == self.num_vars() - partial_point.len(),);
        poly.into_owned()
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point` in place
    fn fix_variables_in_place(&mut self, partial_point: &[E]) {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "partial point len {} >= num_vars {}",
            partial_point.len(),
            self.num_vars()
        );
        let nv = self.num_vars();
        // evaluate single variable of partial point from left to right
        for point in partial_point.iter() {
            // override buf[b1, b2,..bt, 0] = (1-point) * buf[b1, b2,..bt, 0] + point * buf[b1,
            // b2,..bt, 1] in parallel
            match &mut self.evaluations {
                FieldType::Base(evaluations) => {
                    let evaluations_ext = evaluations
                        .chunks(2)
                        .map(|buf| *point * (buf[1] - buf[0]) + buf[0])
                        .collect();
                    let _ = mem::replace(&mut self.evaluations, FieldType::Ext(evaluations_ext));
                }
                FieldType::Ext(evaluations) => {
                    (0..evaluations.len()).step_by(2).for_each(|b| {
                        evaluations[b >> 1] =
                            evaluations[b] + (evaluations[b + 1] - evaluations[b]) * *point
                    });
                }
                FieldType::Unreachable => unreachable!(),
            };
        }
        match &mut self.evaluations {
            FieldType::Base(_) => unreachable!(),
            FieldType::Ext(evaluations) => {
                evaluations.truncate(1 << (nv - partial_point.len()));
            }
            FieldType::Unreachable => unreachable!(),
        }

        self.num_vars = nv - partial_point.len();
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point` from high position
    fn fix_high_variables(&self, partial_point: &[E]) -> Self {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        let current_eval_size = self.evaluations.len();
        let mut poly = Cow::Borrowed(self);
        // `Cow` type here to skip first evaluation vector copy
        for point in partial_point.iter().rev() {
            match &mut poly {
                poly @ Cow::Borrowed(_) => {
                    let half_size = current_eval_size >> 1;
                    *poly = op_mle!(self, |evaluations| Cow::Owned(
                        DenseMultilinearExtension::from_evaluations_ext_vec(self.num_vars() - 1, {
                            let (lo, hi) = evaluations.split_at(half_size);
                            lo.par_iter()
                                .zip(hi)
                                .with_min_len(64)
                                .map(|(lo, hi)| *point * (*hi - *lo) + *lo)
                                .collect()
                        })
                    ));
                }
                Cow::Owned(poly) => poly.fix_high_variables_in_place(&[*point]),
            }
        }
        assert!(poly.num_vars == self.num_vars() - partial_point.len(),);
        poly.into_owned()
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point` from high position in place
    fn fix_high_variables_in_place(&mut self, partial_point: &[E]) {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        let nv = self.num_vars();
        let mut current_eval_size = self.evaluations.len();
        for point in partial_point.iter().rev() {
            let half_size = current_eval_size >> 1;
            match &mut self.evaluations {
                FieldType::Base(evaluations) => {
                    let (lo, hi) = evaluations.split_at(half_size);
                    let evaluations_ext = lo
                        .par_iter()
                        .zip(hi)
                        .with_min_len(64)
                        .map(|(lo, hi)| *point * (*hi - *lo) + *lo)
                        .collect();
                    let _ = mem::replace(&mut self.evaluations, FieldType::Ext(evaluations_ext));
                    current_eval_size = half_size;
                }
                FieldType::Ext(evaluations) => {
                    let (lo, hi) = evaluations.split_at_mut(half_size);
                    lo.par_iter_mut()
                        .zip(hi)
                        .with_min_len(64)
                        .for_each(|(lo, hi)| *lo += (*hi - *lo) * *point);
                    current_eval_size = half_size;
                }
                FieldType::Unreachable => unreachable!(),
            };
        }
        match &mut self.evaluations {
            FieldType::Base(_) => {}
            FieldType::Ext(evaluations) => {
                evaluations.truncate(current_eval_size);
            }
            FieldType::Unreachable => unreachable!(),
        }
        self.num_vars = nv - partial_point.len()
    }

    /// Evaluate the MLE at a give point.
    /// Returns an error if the MLE length does not match the point.
    fn evaluate(&self, point: &[E]) -> E {
        // TODO: return error.
        assert_eq!(
            self.num_vars(),
            point.len(),
            "MLE size does not match the point"
        );
        let mle = self.fix_variables_parallel(point);
        op_mle!(
            mle,
            |f| {
                assert_eq!(f.len(), 1);
                f[0]
            },
            |v| E::from(v)
        )
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point`.
    fn fix_variables_parallel(&self, partial_point: &[E]) -> Self {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        let mut poly = Cow::Borrowed(self);

        // evaluate single variable of partial point from left to right
        // `Cow` type here to skip first evaluation vector copy
        for point in partial_point.iter() {
            match &mut poly {
                poly @ Cow::Borrowed(_) => {
                    *poly = op_mle!(self, |evaluations| {
                        Cow::Owned(DenseMultilinearExtension::from_evaluations_ext_vec(
                            self.num_vars() - 1,
                            evaluations
                                .par_iter()
                                .chunks(2)
                                .with_min_len(64)
                                .map(|buf| *point * (*buf[1] - *buf[0]) + *buf[0])
                                .collect(),
                        ))
                    });
                }
                Cow::Owned(poly) => poly.fix_variables_in_place_parallel(&[*point]),
            }
        }
        assert!(poly.num_vars == self.num_vars() - partial_point.len(),);
        poly.into_owned()
    }

    /// Reduce the number of variables of `self` by fixing the
    /// `partial_point.len()` variables at `partial_point` in place
    fn fix_variables_in_place_parallel(&mut self, partial_point: &[E]) {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "partial point len {} >= num_vars {}",
            partial_point.len(),
            self.num_vars()
        );
        let nv = self.num_vars();
        // evaluate single variable of partial point from left to right
        for (i, point) in partial_point.iter().enumerate() {
            let max_log2_size = nv - i;
            // override buf[b1, b2,..bt, 0] = (1-point) * buf[b1, b2,..bt, 0] + point * buf[b1, b2,..bt, 1] in parallel
            match &mut self.evaluations {
                FieldType::Base(evaluations) => {
                    let evaluations_ext = evaluations
                        .par_iter()
                        .chunks(2)
                        .with_min_len(64)
                        .map(|buf| *point * (*buf[1] - *buf[0]) + *buf[0])
                        .collect();
                    let _ = mem::replace(&mut self.evaluations, FieldType::Ext(evaluations_ext));
                }
                FieldType::Ext(evaluations) => {
                    evaluations
                        .par_iter_mut()
                        .chunks(2)
                        .with_min_len(64)
                        .for_each(|mut buf| *buf[0] = *buf[0] + (*buf[1] - *buf[0]) * *point);

                    // sequentially update buf[b1, b2,..bt] = buf[b1, b2,..bt, 0]
                    for index in 0..1 << (max_log2_size - 1) {
                        evaluations[index] = evaluations[index << 1];
                    }
                }
                FieldType::Unreachable => unreachable!(),
            };
        }
        match &mut self.evaluations {
            FieldType::Base(_) => unreachable!(),
            FieldType::Ext(evaluations) => {
                evaluations.truncate(1 << (nv - partial_point.len()));
            }
            FieldType::Unreachable => unreachable!(),
        }

        self.num_vars = nv - partial_point.len();
    }

    fn evaluations(&self) -> &FieldType<E> {
        &self.evaluations
    }

    fn evaluations_to_owned(self) -> FieldType<E> {
        self.evaluations
    }

    fn evaluations_range(&self) -> Option<(usize, usize)> {
        None
    }

    fn name(&self) -> &'static str {
        "DenseMultilinearExtension"
    }

    /// assert and get base field vector
    /// panic if not the case
    fn get_base_field_vec(&self) -> &[E::BaseField] {
        match &self.evaluations {
            FieldType::Base(evaluations) => &evaluations[..],
            FieldType::Ext(_) => unreachable!(),
            FieldType::Unreachable => unreachable!(),
        }
    }

    fn merge(&mut self, rhs: DenseMultilinearExtension<E>) {
        assert_eq!(rhs.name(), "DenseMultilinearExtension");
        let rhs_num_vars = rhs.num_vars();
        match (&mut self.evaluations, rhs.evaluations_to_owned()) {
            (FieldType::Base(e1), FieldType::Base(e2)) => {
                e1.extend(e2);
                self.num_vars = ceil_log2(e1.len());
            }
            (FieldType::Ext(e1), FieldType::Ext(e2)) => {
                e1.extend(e2);
                self.num_vars = ceil_log2(e1.len());
            }
            (FieldType::Unreachable, b @ FieldType::Base(..)) => {
                self.num_vars = rhs_num_vars;
                self.evaluations = b;
            }
            (FieldType::Unreachable, b @ FieldType::Ext(..)) => {
                self.num_vars = rhs_num_vars;
                self.evaluations = b;
            }
            (a, b) => panic!("do not support merge different field type DME a: {a:?} b: {b:?}"),
        }
    }

    /// get ranged multiliear extension
    fn get_ranged_mle(
        &self,
        num_range: usize,
        range_index: usize,
    ) -> RangedMultilinearExtension<'_, E> {
        assert!(num_range > 0);
        let offset = self.evaluations.len() / num_range;
        let start = offset * range_index;
        RangedMultilinearExtension::new(self, start, offset)
    }

    /// resize to new size (num_instances * new_size_per_instance / num_range)
    /// and selected by range_index
    /// only support resize base fields, otherwise panic
    fn resize_ranged(
        &self,
        num_instances: usize,
        new_size_per_instance: usize,
        num_range: usize,
        range_index: usize,
    ) -> Self {
        println!("called deprecated api");
        assert!(num_range > 0 && num_instances > 0 && new_size_per_instance > 0);
        let new_len = (new_size_per_instance * num_instances) / num_range;
        match &self.evaluations {
            FieldType::Base(evaluations) => {
                let old_size_per_instance = evaluations.len() / num_instances;
                DenseMultilinearExtension::from_evaluations_vec(
                    ceil_log2(new_len),
                    evaluations
                        .chunks(old_size_per_instance)
                        .flat_map(|chunk| {
                            chunk
                                .iter()
                                .cloned()
                                .chain(std::iter::repeat(E::BaseField::ZERO))
                                .take(new_size_per_instance)
                        })
                        .skip(range_index * new_len)
                        .take(new_len)
                        .collect::<Vec<E::BaseField>>(),
                )
            }
            FieldType::Ext(_) => unreachable!(),
            FieldType::Unreachable => unreachable!(),
        }
    }

    /// dup to new size 1 << (self.num_vars + ceil_log2(num_dups))
    fn dup(&self, num_instances: usize, num_dups: usize) -> Self {
        assert!(num_dups.is_power_of_two());
        assert!(num_instances.is_power_of_two());
        match &self.evaluations {
            FieldType::Base(evaluations) => {
                let old_size_per_instance = evaluations.len() / num_instances;
                DenseMultilinearExtension::from_evaluations_vec(
                    self.num_vars + ceil_log2(num_dups),
                    evaluations
                        .chunks(old_size_per_instance)
                        .flat_map(|chunk| {
                            chunk
                                .iter()
                                .cycle()
                                .cloned()
                                .take(old_size_per_instance * num_dups)
                        })
                        .take(1 << (self.num_vars + ceil_log2(num_dups)))
                        .collect::<Vec<E::BaseField>>(),
                )
            }
            FieldType::Ext(_) => unreachable!(),
            FieldType::Unreachable => unreachable!(),
        }
    }
}

pub struct RangedMultilinearExtension<'a, E: ExtensionField> {
    pub inner: &'a DenseMultilinearExtension<E>,
    pub start: usize,
    pub offset: usize,
    pub(crate) num_vars: usize,
}

impl<'a, E: ExtensionField> RangedMultilinearExtension<'a, E> {
    pub fn new(
        inner: &'a DenseMultilinearExtension<E>,
        start: usize,
        offset: usize,
    ) -> RangedMultilinearExtension<'a, E> {
        assert!(inner.evaluations.len() >= offset);

        RangedMultilinearExtension {
            inner,
            start,
            offset,
            num_vars: ceil_log2(offset),
        }
    }
}

impl<'a, E: ExtensionField> MultilinearExtension<E> for RangedMultilinearExtension<'a, E> {
    type Output = DenseMultilinearExtension<E>;
    fn fix_variables(&self, partial_point: &[E]) -> Self::Output {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );

        if !partial_point.is_empty() {
            let first = partial_point[0];
            let inner = self.inner;
            let mut mle = op_mle!(inner, |evaluations| {
                DenseMultilinearExtension::from_evaluations_ext_vec(
                    self.num_vars() - 1,
                    // syntax: evaluations[start..(start+offset)]
                    evaluations[self.start..][..self.offset]
                        .chunks(2)
                        .map(|buf| first * (buf[1] - buf[0]) + buf[0])
                        .collect(),
                )
            });
            mle.fix_variables_in_place(&partial_point[1..]);
            mle
        } else {
            self.inner.clone()
        }
    }

    fn fix_variables_in_place(&mut self, _partial_point: &[E]) {
        unimplemented!()
    }

    fn fix_high_variables(&self, partial_point: &[E]) -> Self::Output {
        // TODO: return error.
        assert!(
            partial_point.len() <= self.num_vars(),
            "invalid size of partial point"
        );
        if !partial_point.is_empty() {
            let last = partial_point.last().unwrap();
            let inner = self.inner;
            let half_size = self.offset >> 1;
            let mut mle = op_mle!(inner, |evaluations| {
                DenseMultilinearExtension::from_evaluations_ext_vec(self.num_vars() - 1, {
                    let (lo, hi) = evaluations[self.start..][..self.offset].split_at(half_size);
                    lo.par_iter()
                        .zip(hi)
                        .with_min_len(64)
                        .map(|(lo, hi)| *last * (*hi - *lo) + *lo)
                        .collect()
                })
            });
            mle.fix_high_variables_in_place(&partial_point[..partial_point.len() - 1]);
            mle
        } else {
            self.inner.clone()
        }
    }

    fn fix_high_variables_in_place(&mut self, _partial_point: &[E]) {
        unimplemented!()
    }

    fn evaluate(&self, point: &[E]) -> E {
        self.inner.evaluate(point)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_variables_parallel(&self, partial_point: &[E]) -> Self::Output {
        self.inner.fix_variables_parallel(partial_point)
    }

    fn fix_variables_in_place_parallel(&mut self, _partial_point: &[E]) {
        unimplemented!()
    }

    fn evaluations(&self) -> &FieldType<E> {
        &self.inner.evaluations
    }

    fn evaluations_range(&self) -> Option<(usize, usize)> {
        Some((self.start, self.offset))
    }

    fn name(&self) -> &'static str {
        "RangedMultilinearExtension"
    }

    /// assert and get base field vector
    /// panic if not the case
    fn get_base_field_vec(&self) -> &[E::BaseField] {
        match &self.evaluations() {
            FieldType::Base(evaluations) => {
                let (start, offset) = self.evaluations_range().unwrap_or((0, evaluations.len()));
                &evaluations[start..][..offset]
            }
            FieldType::Ext(_) => unreachable!(),
            FieldType::Unreachable => unreachable!(),
        }
    }

    fn evaluations_to_owned(self) -> FieldType<E> {
        println!("FIXME: very expensive..");
        match &self.evaluations() {
            FieldType::Base(evaluations) => {
                let (start, offset) = self.evaluations_range().unwrap_or((0, evaluations.len()));
                FieldType::Base(evaluations[start..][..offset].to_vec())
            }
            FieldType::Ext(evaluations) => {
                let (start, offset) = self.evaluations_range().unwrap_or((0, evaluations.len()));
                FieldType::Ext(evaluations[start..][..offset].to_vec())
            }
            FieldType::Unreachable => unreachable!(),
        }
    }

    fn merge(&mut self, _rhs: DenseMultilinearExtension<E>) {
        unimplemented!()
    }

    fn get_ranged_mle(
        &self,
        _num_range: usize,
        _range_index: usize,
    ) -> RangedMultilinearExtension<'a, E> {
        unimplemented!()
    }

    fn resize_ranged(
        &self,
        _num_instances: usize,
        _new_size_per_instance: usize,
        _num_range: usize,
        _range_index: usize,
    ) -> DenseMultilinearExtension<E> {
        unimplemented!()
    }

    fn dup(&self, _num_instances: usize, _num_dups: usize) -> DenseMultilinearExtension<E> {
        unimplemented!()
    }
}

#[macro_export]
macro_rules! op_mle {
    ($a:ident, |$tmp_a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        match &$a.evaluations() {
            $crate::mle::FieldType::Base(a) => {
                let $tmp_a = if let Some((start, offset)) = $a.evaluations_range() {
                    &a[start..][..offset]
                } else {
                    &a[..]
                };
                let $b_out = $op;
                $op_b_out
            }
            $crate::mle::FieldType::Ext(a) => {
                let $tmp_a = if let Some((start, offset)) = $a.evaluations_range() {
                    &a[start..][..offset]
                } else {
                    &a[..]
                };
                $op
            }
            _ => unreachable!(),
        }
    };
    ($a:ident, |$tmp_a:ident| $op:expr) => {
        op_mle!($a, |$tmp_a| $op, |out| out)
    };
    (|$a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        op_mle!($a, |$a| $op, |$b_out| $op_b_out)
    };
    (|$a:ident| $op:expr) => {
        op_mle!(|$a| $op, |out| out)
    };
}

#[macro_export]
macro_rules! op_mle3_range {
    ($x:ident, $a:ident, $b:ident, $x_vec:ident, $a_vec:ident, $b_vec:ident, $op:expr, |$bb_out:ident| $op_bb_out:expr) => {{
        let $x = if let Some((start, offset)) = $x.evaluations_range() {
            &$x_vec[start..][..offset]
        } else {
            &$x_vec[..]
        };
        let $a = if let Some((start, offset)) = $a.evaluations_range() {
            &$a_vec[start..][..offset]
        } else {
            &$a_vec[..]
        };
        let $b = if let Some((start, offset)) = $b.evaluations_range() {
            &$b_vec[start..][..offset]
        } else {
            &$b_vec[..]
        };
        let $bb_out = $op;
        $op_bb_out
    }};
}

/// deal with x * a + b
#[macro_export]
macro_rules! op_mle_xa_b {
    (|$x:ident, $a:ident, $b:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$x.evaluations(), &$a.evaluations(), &$b.evaluations()) {
            (
                $crate::mle::FieldType::Base(x_vec),
                $crate::mle::FieldType::Base(a_vec),
                $crate::mle::FieldType::Base(b_vec),
            ) => {
                op_mle3_range!($x, $a, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Base(x_vec),
                $crate::mle::FieldType::Ext(a_vec),
                $crate::mle::FieldType::Base(b_vec),
            ) => {
                op_mle3_range!($x, $a, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Base(x_vec),
                $crate::mle::FieldType::Ext(a_vec),
                $crate::mle::FieldType::Ext(b_vec),
            ) => {
                op_mle3_range!($x, $a, $b, x_vec, a_vec, b_vec, $op, |$bb_out| $op_bb_out)
            }
            (x, a, b) => unreachable!(
                "unmatched pattern {:?} {:?} {:?}",
                x.variant_name(),
                a.variant_name(),
                b.variant_name()
            ),
        }
    };
    (|$x:ident, $a:ident, $b:ident| $op:expr) => {
        op_mle_xa_b!(|$x, $a, $b| $op, |out| out)
    };
}

/// deal with f1 * f2 * f3
/// applying cumulative rule for f1, f2, f3 to canonical form: Ext field comes first following by Base Field
#[macro_export]
macro_rules! op_mle_product_3 {
    (|$f1:ident, $f2:ident, $f3:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$f1.evaluations(), &$f2.evaluations(), &$f3.evaluations()) {
            // capture non-canonical form
            (
                $crate::mle::FieldType::Ext(_),
                $crate::mle::FieldType::Base(_),
                $crate::mle::FieldType::Ext(_),
            ) => {
                op_mle_product_3!(@internal |$f1, $f3, $f2| {
                    let ($f2, $f3) = ($f3, $f2);
                    $op
                }, |$bb_out| $op_bb_out)
            }
            // ...add more non-canonical form
            // default will go canonical form
            _ => op_mle_product_3!(@internal |$f1, $f2, $f3| $op, |$bb_out| $op_bb_out),
        }
    };
    (|$f1:ident, $f2:ident, $f3:ident| $op:expr) => {
        op_mle_product_3!(|$f1, $f2, $f3| $op, |out| out),
    };
    (@internal |$f1:ident, $f2:ident, $f3:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$f1.evaluations(), &$f2.evaluations(), &$f3.evaluations()) {
            (
                $crate::mle::FieldType::Base(f1_vec),
                $crate::mle::FieldType::Base(f2_vec),
                $crate::mle::FieldType::Base(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |$bb_out| $op_bb_out)
            }
            (
                $crate::mle::FieldType::Ext(f1_vec),
                $crate::mle::FieldType::Base(f2_vec),
                $crate::mle::FieldType::Base(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |out| out)
            }
            (
                $crate::mle::FieldType::Ext(f1_vec),
                $crate::mle::FieldType::Ext(f2_vec),
                $crate::mle::FieldType::Ext(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |out| out)
            }
            (
                $crate::mle::FieldType::Ext(f1_vec),
                $crate::mle::FieldType::Ext(f2_vec),
                $crate::mle::FieldType::Base(f3_vec),
            ) => {
                op_mle3_range!($f1, $f2, $f3, f1_vec, f2_vec, f3_vec, $op, |out| out)
            }
            // ... add more canonial case if missing
            (a, b, c) => unreachable!(
                "unmatched pattern {:?} {:?} {:?}",
                a.variant_name(),
                b.variant_name(),
                c.variant_name()
            ),
        }
    };
    (|$f1:ident, $f2:ident, $f3:ident| $op:expr) => {
        op_mle_product_3!(|$f1, $f2, $f3| $op, |out| out)
    };
}

/// macro support op(a, b) and tackles type matching internally.
/// Please noted that op must satisfy commutative rule w.r.t op(b, a) operand swap.
#[macro_export]
macro_rules! commutative_op_mle_pair {
    (|$first:ident, $second:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        match (&$first.evaluations(), &$second.evaluations()) {
            ($crate::mle::FieldType::Base(base1), $crate::mle::FieldType::Base(base2)) => {
                let $first = if let Some((start, offset)) = $first.evaluations_range() {
                    &base1[start..][..offset]
                } else {
                    &base1[..]
                };
                let $second = if let Some((start, offset)) = $second.evaluations_range() {
                    &base2[start..][..offset]
                } else {
                    &base2[..]
                };
                let $bb_out = $op;
                $op_bb_out
            }
            ($crate::mle::FieldType::Ext(ext), $crate::mle::FieldType::Base(base)) => {
                let $first = if let Some((start, offset)) = $first.evaluations_range() {
                    &ext[start..][..offset]
                } else {
                    &ext[..]
                };
                let $second = if let Some((start, offset)) = $second.evaluations_range() {
                    &base[start..][..offset]
                } else {
                    &base[..]
                };
                $op
            }
            ($crate::mle::FieldType::Base(base), $crate::mle::FieldType::Ext(ext)) => {
                let base = if let Some((start, offset)) = $first.evaluations_range() {
                    &base[start..][..offset]
                } else {
                    &base[..]
                };
                let ext = if let Some((start, offset)) = $second.evaluations_range() {
                    &ext[start..][..offset]
                } else {
                    &ext[..]
                };
                // swap first and second to make ext field come first before base field.
                // so the same coding template can apply.
                // that's why first and second operand must be commutative
                let $first = ext;
                let $second = base;
                $op
            }
            ($crate::mle::FieldType::Ext(ext), $crate::mle::FieldType::Ext(base)) => {
                let $first = if let Some((start, offset)) = $first.evaluations_range() {
                    &ext[start..][..offset]
                } else {
                    &ext[..]
                };
                let $second = if let Some((start, offset)) = $second.evaluations_range() {
                    &base[start..][..offset]
                } else {
                    &base[..]
                };
                $op
            }
            _ => unreachable!(),
        }
    };
    (|$a:ident, $b:ident| $op:expr) => {
        commutative_op_mle_pair!(|$a, $b| $op, |out| out)
    };
}
