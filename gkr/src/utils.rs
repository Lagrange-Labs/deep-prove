use ff_ext::ExtensionField;
use p3_field::{Field, FieldAlgebra};
use sumcheck::util::ceil_log2;

use std::{iter, sync::Arc};

use ark_std::{end_timer, iterable::Iterable, start_timer};
use ff_ext::SmallField;
use itertools::Itertools;
use multilinear_extensions::mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension};

pub fn i64_to_field<F: SmallField>(x: i64) -> F {
    if x >= 0 {
        F::from_canonical_u64(x as u64)
    } else {
        -F::from_canonical_u64((-x) as u64)
    }
}

/// This is to compute a segment indicator. Specifically, it is an MLE of the
/// following vector:
///     segment_{\mathbf{x}}
///         = \sum_{\mathbf{b}=min_idx + 1}^{2^n - 1} \prod_{i=0}^{n-1} (x_i b_i + (1 - x_i)(1 - b_i))
pub(crate) fn segment_eval_greater_than<E: ExtensionField>(min_idx: usize, a: &[E]) -> E {
    let running_product2 = {
        let mut running_product = vec![E::ZERO; a.len() + 1];
        running_product[a.len()] = E::ONE;
        for i in (0..a.len()).rev() {
            let bit = E::from_canonical_u64(((min_idx >> i) & 1) as u64);
            running_product[i] =
                running_product[i + 1] * (a[i] * bit + (E::ONE - a[i]) * (E::ONE - bit));
        }
        running_product
    };
    // Here is an example of how this works:
    // Suppose min_idx = (110101)_2
    // Then ans = eq(11011, a[1..6])
    //          + eq(111, a[3..6], b[3..6])
    let mut ans = E::ZERO;
    for i in 0..a.len() {
        let bit = (min_idx >> i) & 1;
        if bit == 1 {
            continue;
        }
        ans += running_product2[i + 1] * a[i];
    }
    ans
}

/// This is to compute a variant of eq(\mathbf{x}, \mathbf{y}) for indices in
/// (min_idx, 2^n]. Specifically, it is an MLE of the following vector:
///     partial_eq_{\mathbf{x}}(\mathbf{y})
///         = \sum_{\mathbf{b}=min_idx + 1}^{2^n - 1} \prod_{i=0}^{n-1} (x_i y_i b_i + (1 - x_i)(1 - y_i)(1 - b_i))
#[allow(dead_code)]
pub(crate) fn eq_eval_greater_than<F: SmallField>(min_idx: usize, a: &[F], b: &[F]) -> F {
    assert!(a.len() >= b.len());
    // Compute running product of ( x_i y_i + (1 - x_i)(1 - y_i) )_{0 <= i <= n}
    let running_product = {
        let mut running_product = Vec::with_capacity(a.len() + 1);
        running_product.push(F::ONE);
        for i in 0..b.len() {
            let x = running_product[i] * (a[i] * b[i] + (F::ONE - a[i]) * (F::ONE - b[i]));
            running_product.push(x);
        }
        running_product
    };

    let running_product2 = {
        let mut running_product = vec![F::ZERO; b.len() + 1];
        running_product[b.len()] = F::ONE;
        for i in (0..b.len()).rev() {
            let bit = F::from_canonical_u64(((min_idx >> i) & 1) as u64);
            running_product[i] = running_product[i + 1]
                * (a[i] * b[i] * bit + (F::ONE - a[i]) * (F::ONE - b[i]) * (F::ONE - bit));
        }
        running_product
    };

    // Here is an example of how this works:
    // Suppose min_idx = (110101)_2
    // Then ans = eq(11011, a[1..6], b[1..6])eq(a[0..1], b[0..1])
    //          + eq(111, a[3..6], b[3..6])eq(a[0..3], b[0..3])
    let mut ans = F::ZERO;
    for i in 0..b.len() {
        let bit = (min_idx >> i) & 1;
        if bit == 1 {
            continue;
        }
        ans += running_product[i] * running_product2[i + 1] * a[i] * b[i];
    }
    for &ai in &a[b.len()..] {
        ans *= F::ONE - ai;
    }
    ans
}

/// This is to compute a variant of eq(\mathbf{x}, \mathbf{y}) for indices in
/// [0, max_idx]. Specifically, it is an MLE of the following vector:
///     partial_eq_{\mathbf{x}}(\mathbf{y})
///         = \sum_{\mathbf{b}=0}^{max_idx} \prod_{i=0}^{n-1} (x_i y_i b_i + (1 - x_i)(1 - y_i)(1 - b_i))
// TODO(Matthias): See whether we can remove this function.
#[allow(dead_code)]
pub(crate) fn eq_eval_less_or_equal_than<E: ExtensionField>(max_idx: usize, a: &[E], b: &[E]) -> E {
    assert!(a.len() >= b.len());
    // Compute running product of ( x_i y_i + (1 - x_i)(1 - y_i) )_{0 <= i <= n}
    let running_product = {
        let mut running_product = Vec::with_capacity(b.len() + 1);
        running_product.push(E::ONE);
        for i in 0..b.len() {
            let x = running_product[i] * (a[i] * b[i] + (E::ONE - a[i]) * (E::ONE - b[i]));
            running_product.push(x);
        }
        running_product
    };

    let running_product2 = {
        let mut running_product = vec![E::ZERO; b.len() + 1];
        running_product[b.len()] = E::ONE;
        for i in (0..b.len()).rev() {
            let bit = E::from_canonical_u64(((max_idx >> i) & 1) as u64);
            running_product[i] = running_product[i + 1]
                * (a[i] * b[i] * bit + (E::ONE - a[i]) * (E::ONE - b[i]) * (E::ONE - bit));
        }
        running_product
    };

    // Here is an example of how this works:
    // Suppose max_idx = (110101)_2
    // Then ans = eq(a, b)
    //          - eq(11011, a[1..6], b[1..6])eq(a[0..1], b[0..1])
    //          - eq(111, a[3..6], b[3..6])eq(a[0..3], b[0..3])
    let mut ans = running_product[b.len()];
    for i in 0..b.len() {
        let bit = (max_idx >> i) & 1;
        if bit == 1 {
            continue;
        }
        ans -= running_product[i] * running_product2[i + 1] * a[i] * b[i];
    }
    for &ai in &a[b.len()..] {
        ans *= E::ONE - ai;
    }
    ans
}

pub fn counter_eval<E: ExtensionField>(num_vars: usize, x: &[E]) -> E {
    assert_eq!(x.len(), num_vars, "invalid size of x");
    let mut ans = E::ZERO;
    for (i, &xi) in x.iter().enumerate() {
        ans += xi * E::from_canonical_u64(1 << i)
    }
    ans
}

/// Evaluate eq polynomial for 3 random points.
pub fn eq3_eval<E: ExtensionField>(x: &[E], y: &[E], z: &[E]) -> E {
    assert_eq!(x.len(), y.len(), "x and y have different length");

    let start = start_timer!(|| "eq3_eval");
    let mut res = E::ONE;
    for ((&xi, &yi), &zi) in x.iter().zip(y.iter()).zip(z.iter()) {
        res *= xi * yi * zi + (E::ONE - xi) * (E::ONE - yi) * (E::ONE - zi);
    }
    end_timer!(start);
    res
}

/// Evaluate eq polynomial for 4 random points
pub fn eq4_eval<E: ExtensionField>(x: &[E], y: &[E], z: &[E], w: &[E]) -> E {
    assert_eq!(x.len(), y.len(), "x and y have different length");

    let start = start_timer!(|| "eq3_eval");
    let mut res = E::ONE;
    for (((&xi, &yi), &zi), &wi) in x.iter().zip(y.iter()).zip(z.iter()).zip(w.iter()) {
        res *= xi * yi * zi * wi + (E::ONE - xi) * (E::ONE - yi) * (E::ONE - zi) * (E::ONE - wi);
    }
    end_timer!(start);
    res
}

pub fn tensor_product<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let mut res = vec![F::ZERO; a.len() * b.len()];
    for i in 0..a.len() {
        for j in 0..b.len() {
            res[i * b.len() + j] = a[i] * b[j];
        }
    }
    res
}

pub trait MultilinearExtensionFromVectors<E: ExtensionField> {
    fn mle(&self, lo_num_vars: usize, hi_num_vars: usize) -> ArcDenseMultilinearExtension<E>;
    fn mle_with_meta(
        &self,
        lo_num_vars: usize,
        hi_num_vars: usize,
        multi_threads_meta: (usize, usize),
    ) -> ArcDenseMultilinearExtension<E>;
    fn original_mle(&self) -> ArcDenseMultilinearExtension<E>;
}

impl<E: ExtensionField> MultilinearExtensionFromVectors<E> for &[Vec<E::BaseField>] {
    fn mle_with_meta(
        &self,
        lo_num_vars: usize,
        hi_num_vars: usize,
        multi_threads_meta: (usize, usize),
    ) -> ArcDenseMultilinearExtension<E> {
        let (thread_id, max_thread_id) = multi_threads_meta;
        let log2_max_thread_id = ceil_log2(max_thread_id);
        assert!(log2_max_thread_id <= hi_num_vars);
        assert!(
            self.len().is_power_of_two(),
            "not supporting non-power of 2 vector len {} ",
            self.len()
        );

        let num_instances_per_thread = 1 << (hi_num_vars - log2_max_thread_id);
        let vector_len_per_thread = 1 << (hi_num_vars - log2_max_thread_id + lo_num_vars);

        let mle = DenseMultilinearExtension::from_evaluations_vec(
            hi_num_vars - log2_max_thread_id + lo_num_vars,
            self.iter()
                .skip(thread_id * num_instances_per_thread)
                .flat_map(|instance| {
                    instance
                        .iter()
                        .cloned()
                        .chain(iter::repeat(E::BaseField::ZERO))
                        .take(1 << lo_num_vars)
                })
                .take(vector_len_per_thread)
                .collect_vec(),
        );
        assert!(mle.evaluations.len() == vector_len_per_thread);
        mle.into()
    }
    fn mle(&self, lo_num_vars: usize, hi_num_vars: usize) -> ArcDenseMultilinearExtension<E> {
        assert!(
            self.len().is_power_of_two(),
            "not supporting non-power of 2 vector len {} ",
            self.len()
        );
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            lo_num_vars + hi_num_vars,
            self.iter()
                .flat_map(|instance| {
                    instance
                        .iter()
                        .cloned()
                        .chain(iter::repeat(E::BaseField::ZERO))
                        .take(1 << lo_num_vars)
                })
                .take(1 << (lo_num_vars + hi_num_vars))
                .collect_vec(),
        ))
    }
    fn original_mle(&self) -> ArcDenseMultilinearExtension<E> {
        let lo_num_vars = ceil_log2(self[0].len());
        let hi_num_vars = ceil_log2(self.len());
        let n_zeros = (1 << lo_num_vars) - self[0].len();
        let n_zero_vecs = (1 << hi_num_vars) - self.len();
        let vecs = self.to_vec();

        DenseMultilinearExtension::from_evaluations_vec(
            lo_num_vars + hi_num_vars,
            vecs.into_iter()
                .flat_map(|instance| {
                    instance
                        .into_iter()
                        .chain(std::iter::repeat_n(E::BaseField::ZERO, n_zeros))
                })
                .chain(vec![E::BaseField::ZERO; n_zero_vecs])
                .collect_vec(),
        )
        .into()
    }
}

pub(crate) trait MatrixMLEColumnFirst<E: ExtensionField> {
    fn fix_row_col_first(&self, row_point_eq: &[E], col_num_vars: usize) -> Vec<E>;
    // TODO(Matthias): See whether we can remove this method.
    #[allow(dead_code)]
    fn fix_row_col_first_with_scalar(
        &self,
        row_point_eq: &[E],
        col_num_vars: usize,
        scalar: &E,
    ) -> Vec<E>;
    fn eval_col_first(&self, row_point_eq: &[E], col_point_eq: &[E]) -> E;
}

impl<E: ExtensionField> MatrixMLEColumnFirst<E> for &[usize] {
    fn fix_row_col_first(&self, row_point_eq: &[E], col_num_vars: usize) -> Vec<E> {
        let mut ans = vec![E::ZERO; 1 << col_num_vars];
        for (col, &non_zero_row) in self.iter().enumerate() {
            ans[col] = row_point_eq[non_zero_row];
        }
        ans
    }

    fn fix_row_col_first_with_scalar(
        &self,
        row_point_eq: &[E],
        col_num_vars: usize,
        scalar: &E,
    ) -> Vec<E> {
        let mut ans = vec![E::ZERO; 1 << col_num_vars];
        for (col, &non_zero_row) in self.iter().enumerate() {
            ans[col] = row_point_eq[non_zero_row] * *scalar;
        }
        ans
    }

    fn eval_col_first(&self, row_point_eq: &[E], col_point_eq: &[E]) -> E {
        self.iter()
            .enumerate()
            .fold(E::ZERO, |acc, (col, &non_zero_row)| {
                acc + row_point_eq[non_zero_row] * col_point_eq[col]
            })
    }
}

pub(crate) trait MatrixMLERowFirst<E: ExtensionField> {
    fn fix_row_row_first(&self, row_point_eq: &[E], col_num_vars: usize) -> Vec<E>;
    fn fix_row_row_first_with_scalar(
        &self,
        row_point_eq: &[E],
        col_num_vars: usize,
        scalar: &E,
    ) -> Vec<E>;
    fn eval_row_first(&self, row_point_eq: &[E], col_point_eq: &[E]) -> E;
}

impl<E: ExtensionField> MatrixMLERowFirst<E> for &[usize] {
    fn fix_row_row_first(&self, row_point_eq: &[E], col_num_vars: usize) -> Vec<E> {
        let mut ans = vec![E::ZERO; 1 << col_num_vars];
        for (row, &non_zero_col) in self.iter().enumerate() {
            ans[non_zero_col] = row_point_eq[row];
        }
        ans
    }

    fn fix_row_row_first_with_scalar(
        &self,
        row_point_eq: &[E],
        col_num_vars: usize,
        scalar: &E,
    ) -> Vec<E> {
        let mut ans = vec![E::ZERO; 1 << col_num_vars];
        for (row, &non_zero_col) in self.iter().enumerate() {
            ans[non_zero_col] = row_point_eq[row] * *scalar;
        }
        ans
    }

    fn eval_row_first(&self, row_point_eq: &[E], col_point_eq: &[E]) -> E {
        self.iter()
            .enumerate()
            .fold(E::ZERO, |acc, (row, &non_zero_col)| {
                acc + row_point_eq[row] * col_point_eq[non_zero_col]
            })
    }
}

// TODO(Matthias): See whether we can remove this trait.
#[allow(dead_code)]
pub(crate) trait SubsetIndices<F: SmallField> {
    fn subset_eq_with_scalar(&self, eq: &[F], scalar: &F) -> Vec<F>;
    fn subset_eq_eval(&self, eq_1: &[F]) -> F;
    fn subset_eq2_eval(&self, eq_1: &[F], eq_2: &[F]) -> F;
}

impl<F: SmallField> SubsetIndices<F> for &[usize] {
    fn subset_eq_with_scalar(&self, eq: &[F], scalar: &F) -> Vec<F> {
        let mut ans = vec![F::ZERO; eq.len()];
        for &non_zero_i in self.iter() {
            ans[non_zero_i] = eq[non_zero_i] * *scalar;
        }
        ans
    }
    fn subset_eq_eval(&self, eq_1: &[F]) -> F {
        self.iter()
            .fold(F::ZERO, |acc, &non_zero_i| acc + eq_1[non_zero_i])
    }
    fn subset_eq2_eval(&self, eq_1: &[F], eq_2: &[F]) -> F {
        self.iter().fold(F::ZERO, |acc, &non_zero_i| {
            acc + eq_1[non_zero_i] * eq_2[non_zero_i]
        })
    }
}

// test
#[cfg(test)]
mod test {
    use super::*;
    use ark_std::test_rng;
    use ff_ext::{FromUniformBytes, GoldilocksExt2};
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{DenseMultilinearExtension, MultilinearExtension},
        virtual_poly::build_eq_x_r_vec,
    };
    use p3_field::FieldAlgebra;

    #[test]
    fn test_ceil_log2() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(16), 4);
    }

    #[test]
    fn test_eq_eval_less_or_equal_than() {
        let mut rng = test_rng();
        let n = 5;
        let pow_n = 1 << n;
        let a = (0..n)
            .map(|_| GoldilocksExt2::random(&mut rng))
            .collect_vec();
        let b = (0..n)
            .map(|_| GoldilocksExt2::random(&mut rng))
            .collect_vec();

        let eq_vec = build_eq_x_r_vec(&a);

        {
            let max_idx = 0;
            let mut partial_eq_vec: Vec<_> = eq_vec[0..=max_idx].to_vec();
            partial_eq_vec.extend(vec![GoldilocksExt2::ZERO; pow_n - max_idx - 1]);
            let expected_ans =
                DenseMultilinearExtension::from_evaluations_ext_vec(n, partial_eq_vec).evaluate(&b);
            assert_eq!(expected_ans, eq_eval_less_or_equal_than(max_idx, &a, &b));
        }

        {
            let max_idx = 1;
            let mut partial_eq_vec: Vec<_> = eq_vec[0..=max_idx].to_vec();
            partial_eq_vec.extend(vec![GoldilocksExt2::ZERO; pow_n - max_idx - 1]);
            let expected_ans =
                DenseMultilinearExtension::from_evaluations_ext_vec(n, partial_eq_vec).evaluate(&b);
            assert_eq!(expected_ans, eq_eval_less_or_equal_than(max_idx, &a, &b));
        }

        {
            let max_idx = 12;
            let mut partial_eq_vec: Vec<_> = eq_vec[0..=max_idx].to_vec();
            partial_eq_vec.extend(vec![GoldilocksExt2::ZERO; pow_n - max_idx - 1]);
            let expected_ans =
                DenseMultilinearExtension::from_evaluations_ext_vec(n, partial_eq_vec).evaluate(&b);
            assert_eq!(expected_ans, eq_eval_less_or_equal_than(max_idx, &a, &b));
        }

        {
            let max_idx = 1 << ((n - 1) - 1);
            let mut partial_eq_vec: Vec<_> = eq_vec[0..=max_idx].to_vec();
            partial_eq_vec.extend(vec![GoldilocksExt2::ZERO; pow_n - max_idx - 1]);
            let expected_ans =
                DenseMultilinearExtension::from_evaluations_ext_vec(n, partial_eq_vec).evaluate(&b);
            assert_eq!(expected_ans, eq_eval_less_or_equal_than(max_idx, &a, &b));
        }

        {
            let max_idx = 1 << (n - 1);
            let mut partial_eq_vec: Vec<_> = eq_vec[0..=max_idx].to_vec();
            partial_eq_vec.extend(vec![GoldilocksExt2::ZERO; pow_n - max_idx - 1]);
            let expected_ans =
                DenseMultilinearExtension::from_evaluations_ext_vec(n, partial_eq_vec).evaluate(&b);
            assert_eq!(expected_ans, eq_eval_less_or_equal_than(max_idx, &a, &b));
        }
    }

    #[test]
    fn test_counter_eval() {
        let vec = (0..(1 << 4))
            .map(GoldilocksExt2::from_canonical_u64)
            .collect_vec();
        let point = vec![
            GoldilocksExt2::from_canonical_u64(97),
            GoldilocksExt2::from_canonical_u64(101),
            GoldilocksExt2::from_canonical_u64(23),
            GoldilocksExt2::from_canonical_u64(29),
        ];
        let got_value = counter_eval(4, &point);
        let poly = DenseMultilinearExtension::from_evaluations_ext_vec(4, vec);
        let expected_value = poly.evaluate(&point);
        assert_eq!(got_value, expected_value);
    }
}
