use std::{
    array,
    cmp::max,
    iter::Sum,
    ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign},
    sync::Arc,
};

use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use multilinear_extensions::{
    mle::DenseMultilinearExtension, op_mle, virtual_poly::VirtualPolynomial,
};
use p3_field::Field;
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::structs::IOPProverState;

pub fn barycentric_weights<F: Field>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|&(i, _)| (i != j))
                .map(|(_, point_i)| *point_j - *point_i)
                .reduce(|acc, value| acc * value)
                .unwrap_or(F::ONE)
        })
        .collect::<Vec<_>>();
    batch_inversion(&mut weights);
    weights
}

// Computes the inverse of each field element in a vector {v_i} using a parallelized batch inversion.
pub fn batch_inversion<F: Field>(v: &mut [F]) {
    batch_inversion_and_mul(v, &F::ONE);
}

// Computes the inverse of each field element in a vector {v_i} sequentially (serial version).
pub fn serial_batch_inversion<F: Field>(v: &mut [F]) {
    serial_batch_inversion_and_mul(v, &F::ONE)
}

// Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}
pub fn batch_inversion_and_mul<F: Field>(v: &mut [F], coeff: &F) {
    // Divide the vector v evenly between all available cores
    let min_elements_per_thread = 1;
    let num_cpus_available = rayon::current_num_threads();
    let num_elems = v.len();
    let num_elem_per_thread = max(num_elems / num_cpus_available, min_elements_per_thread);

    // Batch invert in parallel, without copying the vector
    v.par_chunks_mut(num_elem_per_thread).for_each(|chunk| {
        serial_batch_inversion_and_mul(chunk, coeff);
    });
}

/// Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}.
/// This method is explicitly single-threaded.
fn serial_batch_inversion_and_mul<F: Field>(v: &mut [F], coeff: &F) {
    // Montgomery’s Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned vector by
    // coeff

    // First pass: compute [a, ab, abc, ...]
    let mut prod = Vec::with_capacity(v.len());
    let mut tmp = F::ONE;
    for f in v.iter().filter(|f| !f.is_zero()) {
        tmp.mul_assign(*f);
        prod.push(tmp);
    }

    // Invert `tmp`.
    tmp = tmp.try_inverse().unwrap(); // Guaranteed to be nonzero.

    // Multiply product by coeff, so all inverses will be scaled by coeff
    tmp *= *coeff;

    // Second pass: iterate backwards to compute inverses
    for (f, s) in v
        .iter_mut()
        // Backwards
        .rev()
        // Ignore normalized elements
        .filter(|f| !f.is_zero())
        // Backwards, skip last element, fill in one for last term.
        .zip(prod.into_iter().rev().skip(1).chain(Some(F::ONE)))
    {
        // tmp := tmp * f; f := tmp * s = 1/f
        let new_tmp = tmp * *f;
        *f = tmp * s;
        tmp = new_tmp;
    }
}

pub(crate) fn extrapolate<F: Field>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    inner_extrapolate::<F, true>(points, weights, evals, at)
}

pub(crate) fn serial_extrapolate<F: Field>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    inner_extrapolate::<F, false>(points, weights, evals, at)
}

fn inner_extrapolate<F: Field, const IS_PARALLEL: bool>(
    points: &[F],
    weights: &[F],
    evals: &[F],
    at: &F,
) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *at - *point).collect::<Vec<_>>();
        if IS_PARALLEL {
            batch_inversion(&mut coeffs);
        } else {
            serial_batch_inversion(&mut coeffs);
        }
        let mut sum = F::ZERO;
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= *weight;
            sum += *coeff
        });
        let sum_inv = sum.try_inverse().unwrap_or(F::ZERO);
        (coeffs, sum_inv)
    };
    coeffs
        .iter()
        .zip(evals)
        .map(|(coeff, eval)| *coeff * *eval)
        .sum::<F>()
        * sum_inv
}

/// Interpolate a uni-variate degree-`p_i.len()-1` polynomial and evaluate this
/// polynomial at `eval_at`:
///
///   \sum_{i=0}^len p_i * (\prod_{j!=i} (eval_at - j)/(i-j) )
///
/// This implementation is linear in number of inputs in terms of field
/// operations. It also has a quadratic term in primitive operations which is
/// negligible compared to field operations.
/// TODO: The quadratic term can be removed by precomputing the lagrange
/// coefficients.
pub(crate) fn interpolate_uni_poly<F: Field>(p_i: &[F], eval_at: F) -> F {
    let start = start_timer!(|| "sum check interpolate uni poly opt");

    let len = p_i.len();
    let mut evals = vec![];
    let mut prod = eval_at;
    evals.push(eval_at);

    // `prod = \prod_{j} (eval_at - j)`
    for e in 1..len {
        let tmp = eval_at - F::from_canonical_u64(e as u64);
        evals.push(tmp);
        prod *= tmp;
    }
    let mut res = F::ZERO;
    // we want to compute \prod (j!=i) (i-j) for a given i
    //
    // we start from the last step, which is
    //  denom[len-1] = (len-1) * (len-2) *... * 2 * 1
    // the step before that is
    //  denom[len-2] = (len-2) * (len-3) * ... * 2 * 1 * -1
    // and the step before that is
    //  denom[len-3] = (len-3) * (len-4) * ... * 2 * 1 * -1 * -2
    //
    // i.e., for any i, the one before this will be derived from
    //  denom[i-1] = denom[i] * (len-i) / i
    //
    // that is, we only need to store
    // - the last denom for i = len-1, and
    // - the ratio between current step and fhe last step, which is the product of (len-i) / i from
    //   all previous steps and we store this product as a fraction number to reduce field
    //   divisions.

    let mut denom_up = field_factorial::<F>(len - 1);
    let mut denom_down = F::ONE;

    for i in (0..len).rev() {
        res += p_i[i] * prod * denom_down * (denom_up * evals[i]).inverse();

        // compute denom for the next step is current_denom * (len-i)/i
        if i != 0 {
            denom_up *= -F::from_canonical_u64((len - i) as u64);
            denom_down *= F::from_canonical_u64(i as u64);
        }
    }
    end_timer!(start);
    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn field_factorial<F: Field>(a: usize) -> F {
    let mut res = F::ONE;
    for i in 2..=a {
        res *= F::from_canonical_u64(i as u64);
    }
    res
}

/// log2 ceil of x
pub fn ceil_log2(x: usize) -> usize {
    assert!(x > 0, "ceil_log2: x must be positive");
    // Calculate the number of bits in usize
    let usize_bits = std::mem::size_of::<usize>() * 8;
    usize_bits - (x - 1).leading_zeros() as usize
}

pub(crate) fn merge_sumcheck_polys<'a, E: ExtensionField>(
    prover_states: &[IOPProverState<'a, E>],
    max_thread_id: usize,
) -> VirtualPolynomial<'a, E> {
    let log2_max_thread_id = ceil_log2(max_thread_id);
    let mut poly = prover_states[0].poly.clone(); // giving only one evaluation left, this clone is low cost.
    poly.aux_info.max_num_variables = log2_max_thread_id; // size_log2 variates sumcheck
    for i in 0..poly.flattened_ml_extensions.len() {
        let ml_ext = DenseMultilinearExtension::from_evaluations_ext_vec(
            log2_max_thread_id,
            prover_states
                .iter()
                .map(|prover_state| {
                    let mle = &prover_state.poly.flattened_ml_extensions[i];
                    op_mle!(
                        mle,
                        |f| {
                            assert!(f.len() == 1);
                            f[0]
                        },
                        |_v| unreachable!()
                    )
                })
                .collect::<Vec<E>>(),
        );
        poly.flattened_ml_extensions[i] = Arc::new(ml_ext);
    }
    poly
}

#[derive(Clone, Copy, Debug)]
/// util collection to support fundamental operation
pub struct AdditiveArray<F, const N: usize>(pub [F; N]);

impl<F: Default, const N: usize> Default for AdditiveArray<F, N> {
    fn default() -> Self {
        Self(array::from_fn(|_| F::default()))
    }
}

impl<F: AddAssign, const N: usize> AddAssign for AdditiveArray<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(acc, item)| *acc += item);
    }
}

impl<F: AddAssign, const N: usize> Add for AdditiveArray<F, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: AddAssign + Default, const N: usize> Sum for AdditiveArray<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap_or_default()
    }
}

impl<F, const N: usize> Deref for AdditiveArray<F, N> {
    type Target = [F; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, const N: usize> DerefMut for AdditiveArray<F, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Debug)]
pub struct AdditiveVec<F>(pub Vec<F>);

impl<F> Deref for AdditiveVec<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for AdditiveVec<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Clone + Default> AdditiveVec<F> {
    pub fn new(len: usize) -> Self {
        Self(vec![F::default(); len])
    }
}

impl<F: AddAssign> AddAssign for AdditiveVec<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(acc, item)| *acc += item);
    }
}

impl<F: AddAssign> Add for AdditiveVec<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: MulAssign + Copy> MulAssign<F> for AdditiveVec<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

impl<F: MulAssign + Copy> Mul<F> for AdditiveVec<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self *= rhs;
        self
    }
}
