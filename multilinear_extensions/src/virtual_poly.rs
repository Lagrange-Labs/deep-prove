use std::{cmp::max, collections::HashMap, marker::PhantomData, mem::MaybeUninit, sync::Arc};

use crate::{
    mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension, MultilinearExtension},
    util::{bit_decompose, create_uninit_vec, max_usable_threads},
};
use ark_std::{end_timer, rand::Rng, start_timer};
use ff_ext::ExtensionField;
use itertools::Itertools;
use p3_field::{Field, FieldAlgebra};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use transcript::Transcript;

pub type ArcMultilinearExtension<'a, E> =
    Arc<dyn MultilinearExtension<E, Output = DenseMultilinearExtension<E>> + 'a>;
#[rustfmt::skip]
/// A virtual polynomial is a sum of products of multilinear polynomials;
/// where the multilinear polynomials are stored via their multilinear
/// extensions:  `(coefficient, DenseMultilinearExtension)`
///
/// * Number of products n = `polynomial.products.len()`,
/// * Number of multiplicands of ith product m_i =
///   `polynomial.products[i].1.len()`,
/// * Coefficient of ith product c_i = `polynomial.products[i].0`
///
/// The resulting polynomial is
///
/// $$ \sum_{i=0}^{n} c_i \cdot \prod_{j=0}^{m_i} P_{ij} $$
///
/// Example:
///  f = c0 * f0 * f1 * f2 + c1 * f3 * f4
/// where f0 ... f4 are multilinear polynomials
///
/// - flattened_ml_extensions stores the multilinear extension representation of
///   f0, f1, f2, f3 and f4
/// - products is
///   $$
///   \[
///   (c0, \[0, 1, 2\]),
///   (c1, \[3, 4\])
///   \]
///   $$
/// - raw_pointers_lookup_table maps fi to i
///
#[derive(Default, Clone)]
pub struct VirtualPolynomial<'a, E: ExtensionField> {
    /// Aux information about the multilinear polynomial
    pub aux_info: VPAuxInfo<E>,
    /// list of reference to products (as usize) of multilinear extension
    pub products: Vec<(E, Vec<usize>)>,
    /// Stores multilinear extensions in which product multiplicand can refer
    /// to.
    pub flattened_ml_extensions: Vec<ArcMultilinearExtension<'a, E>>,
    /// Pointers to the above poly extensions
    raw_pointers_lookup_table: HashMap<usize, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Auxiliary information about the multilinear polynomial
pub struct VPAuxInfo<E> {
    /// max number of multiplicands in each product
    pub max_degree: usize,
    /// max number of variables of the polynomial
    pub max_num_variables: usize,
    /// Associated field
    #[doc(hidden)]
    pub phantom: PhantomData<E>,
}
impl<E: ExtensionField> VPAuxInfo<E> {
    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_elements(&[
            E::BaseField::from_canonical_u64(self.max_degree as u64),
            E::BaseField::from_canonical_u64(self.max_num_variables as u64),
        ]);
    }
}
impl<E> VPAuxInfo<E> {
    /// List of list of MLEs num_vars (f1*f2 + f1*f3*f4 + ... )
    pub fn from_mle_list_dimensions(product_list: &[Vec<usize>]) -> Self {
        let mut max_num_vars = 0;
        let mut max_degree = 0;
        for product in product_list {
            max_degree = max(max_degree, product.len());
            max_num_vars = max(
                max_num_vars,
                *product
                    .iter()
                    .max()
                    .expect("At least one MLE in the product is required"),
            );
        }
        Self {
            max_degree,
            max_num_variables: max_num_vars,
            phantom: PhantomData,
        }
    }
}

impl<'a, E: ExtensionField> VirtualPolynomial<'a, E> {
    /// Creates an empty virtual polynomial with `max_num_variables`.
    pub fn new(max_num_variables: usize) -> Self {
        VirtualPolynomial {
            aux_info: VPAuxInfo {
                max_degree: 0,
                max_num_variables,
                phantom: PhantomData,
            },
            products: Vec::new(),
            flattened_ml_extensions: Vec::new(),
            raw_pointers_lookup_table: HashMap::new(),
        }
    }

    /// Creates an new virtual polynomial from a MLE and its coefficient.
    pub fn new_from_mle(mle: ArcMultilinearExtension<'a, E>, coefficient: E) -> Self {
        let mle_ptr: usize = Arc::as_ptr(&mle) as *const () as usize;
        let mut hm = HashMap::new();
        hm.insert(mle_ptr, 0);

        VirtualPolynomial {
            aux_info: VPAuxInfo {
                // The max degree is the max degree of any individual variable
                max_degree: 1,
                max_num_variables: mle.num_vars(),
                phantom: PhantomData,
            },
            // here `0` points to the first polynomial of `flattened_ml_extensions`
            products: vec![(coefficient, vec![0])],
            flattened_ml_extensions: vec![mle],
            raw_pointers_lookup_table: hm,
        }
    }

    /// Add a product of list of multilinear extensions to self
    /// Returns an error if the list is empty.
    ///
    /// mle in mle_list must be in same num_vars() in same product,
    /// while different product can have different num_vars()
    ///
    /// The MLEs will be multiplied together, and then multiplied by the scalar
    /// `coefficient`.
    pub fn add_mle_list(&mut self, mle_list: Vec<ArcMultilinearExtension<'a, E>>, coefficient: E) {
        let mle_list: Vec<ArcMultilinearExtension<E>> = mle_list.into_iter().collect();
        let mut indexed_product = Vec::with_capacity(mle_list.len());

        assert!(!mle_list.is_empty(), "input mle_list is empty");
        // sanity check: all mle in mle_list must have same num_vars()
        assert!(
            mle_list
                .iter()
                .map(|m| {
                    assert!(
                        m.num_vars() <= self.aux_info.max_num_variables,
                        "invalid max num vars"
                    );
                    m.num_vars()
                })
                .all_equal()
        );

        self.aux_info.max_degree = max(self.aux_info.max_degree, mle_list.len());

        for mle in mle_list {
            let mle_ptr: usize = Arc::as_ptr(&mle) as *const () as usize;
            if let Some(index) = self.raw_pointers_lookup_table.get(&mle_ptr) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.flattened_ml_extensions.len();
                self.flattened_ml_extensions.push(mle);
                self.raw_pointers_lookup_table.insert(mle_ptr, curr_index);
                indexed_product.push(curr_index);
            }
        }
        self.products.push((coefficient, indexed_product));
    }

    /// Multiple the current VirtualPolynomial by an MLE:
    /// - add the MLE to the MLE list;
    /// - multiple each product by MLE and its coefficient.
    ///
    /// Returns an error if the MLE has a different `num_vars()` from self.
    pub fn mul_by_mle(&mut self, mle: ArcMultilinearExtension<'a, E>, coefficient: E::BaseField) {
        let start = start_timer!(|| "mul by mle");

        assert_eq!(
            mle.num_vars(),
            self.aux_info.max_num_variables,
            "product has a multiplicand with wrong number of variables {} vs {}",
            mle.num_vars(),
            self.aux_info.max_num_variables
        );

        let mle_ptr = Arc::as_ptr(&mle) as *const () as usize;

        // check if this mle already exists in the virtual polynomial
        let mle_index = match self.raw_pointers_lookup_table.get(&mle_ptr) {
            Some(&p) => p,
            None => {
                self.raw_pointers_lookup_table
                    .insert(mle_ptr, self.flattened_ml_extensions.len());
                self.flattened_ml_extensions.push(mle);
                self.flattened_ml_extensions.len() - 1
            }
        };

        for (prod_coef, indices) in self.products.iter_mut() {
            // - add the MLE to the MLE list;
            // - multiple each product by MLE and its coefficient.
            indices.push(mle_index);
            *prod_coef *= coefficient;
        }

        // increase the max degree by one as the MLE has degree 1.
        self.aux_info.max_degree += 1;
        end_timer!(start);
    }

    /// in-place merge with another virtual polynomial
    pub fn merge(&mut self, other: &VirtualPolynomial<'a, E>) {
        let start = start_timer!(|| "virtual poly add");
        for (coefficient, products) in other.products.iter() {
            let cur: Vec<_> = products
                .iter()
                .map(|&x| other.flattened_ml_extensions[x].clone())
                .collect();

            self.add_mle_list(cur, *coefficient);
        }
        end_timer!(start);
    }

    /// Evaluate the virtual polynomial at point `point`.
    /// Returns an error is point.len() does not match `num_variables`.
    pub fn evaluate(&self, point: &[E]) -> E {
        let start = start_timer!(|| "evaluation");

        assert_eq!(
            self.aux_info.max_num_variables,
            point.len(),
            "wrong number of variables {} vs {}",
            self.aux_info.max_num_variables,
            point.len()
        );

        let evals: Vec<E> = self
            .flattened_ml_extensions
            .iter()
            .map(|x| x.evaluate(&point[0..x.num_vars()]))
            .collect();

        let res = self
            .products
            .iter()
            .map(|(c, p)| p.iter().map(|&i| evals[i]).product::<E>() * *c)
            .sum();

        end_timer!(start);
        res
    }

    /// Sample a random virtual polynomial, return the polynomial and its sum.
    pub fn random(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
        mut rng: &mut impl Rng,
    ) -> (Self, E) {
        let start = start_timer!(|| "sample random virtual polynomial");

        let mut sum = E::ZERO;
        let mut poly = VirtualPolynomial::new(nv);
        for _ in 0..num_products {
            let num_multiplicands =
                rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
            let (product, product_sum) =
                DenseMultilinearExtension::random_mle_list(nv, num_multiplicands, rng);
            let product: Vec<ArcMultilinearExtension<E>> =
                product.into_iter().map(|mle| mle as _).collect_vec();
            let coefficient = E::random(&mut rng);
            poly.add_mle_list(product, coefficient);
            sum += product_sum * coefficient;
        }

        end_timer!(start);
        (poly, sum)
    }

    /// Print out the evaluation map for testing. Panic if the num_vars() > 5.
    pub fn print_evals(&self) {
        if self.aux_info.max_num_variables > 5 {
            panic!("this function is used for testing only. cannot print more than 5 num_vars()")
        }
        for i in 0..1 << self.aux_info.max_num_variables {
            let point = bit_decompose(i, self.aux_info.max_num_variables);
            let point_fr: Vec<E> = point.iter().map(|&x| E::from_bool(x)).collect();
            println!("{} {:?}", i, self.evaluate(point_fr.as_ref()))
        }
        println!()
    }
}

/// Evaluate eq polynomial.
pub fn eq_eval<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len(), "x and y have different length");

    let start = start_timer!(|| "eq_eval");
    let mut res = F::ONE;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi * yi;
        res *= xi_yi + xi_yi - xi - yi + F::ONE;
    }
    end_timer!(start);
    res
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_sequential<E: ExtensionField>(r: &[E]) -> ArcDenseMultilinearExtension<E> {
    let evals = build_eq_x_r_vec_sequential(r);
    let mle = DenseMultilinearExtension::from_evaluations_ext_vec(r.len(), evals);

    mle.into()
}
/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))

#[tracing::instrument(
    level = "trace",
    skip_all,
    name = "multilinear_extensions::build_eq_x_r_vec_sequential"
)]
pub fn build_eq_x_r_vec_sequential<E: ExtensionField>(r: &[E]) -> Vec<E> {
    // avoid unnecessary allocation
    if r.is_empty() {
        return vec![E::ONE];
    }
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut evals = create_uninit_vec(1 << r.len());
    build_eq_x_r_helper_sequential(r, &mut evals, E::ONE);

    unsafe { std::mem::transmute(evals) }
}

/// A helper function to build eq(x, r)*init via dynamic programing tricks.
/// This function takes 2^num_var iterations, and per iteration with 1 multiplication.
fn build_eq_x_r_helper_sequential<E: ExtensionField>(r: &[E], buf: &mut [MaybeUninit<E>], init: E) {
    buf[0] = MaybeUninit::new(init);

    for (i, r) in r.iter().rev().enumerate() {
        let next_size = 1 << (i + 1);
        // suppose at the previous step we processed buf [0..size]
        // for the current step we are populating new buf[0..2*size]
        // for j traverse 0..size
        // buf[2*j + 1] = r * buf[j]
        // buf[2*j] = (1 - r) * buf[j]
        (0..next_size).step_by(2).rev().for_each(|index| {
            let prev_val = unsafe { buf[index >> 1].assume_init() };
            let tmp = *r * prev_val;
            buf[index + 1] = MaybeUninit::new(tmp);
            buf[index] = MaybeUninit::new(prev_val - tmp);
        });
    }
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<E: ExtensionField>(r: &[E]) -> ArcDenseMultilinearExtension<E> {
    let evals = build_eq_x_r_vec(r);
    let mle = DenseMultilinearExtension::from_evaluations_ext_vec(r.len(), evals);

    mle.into()
}
/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))

#[tracing::instrument(
    level = "trace",
    skip_all,
    name = "multilinear_extensions::build_eq_x_r_vec"
)]
pub fn build_eq_x_r_vec<E: ExtensionField>(r: &[E]) -> Vec<E> {
    // avoid unnecessary allocation
    if r.is_empty() {
        return vec![E::ONE];
    }
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations
    let nthreads = max_usable_threads();
    let nbits = nthreads.trailing_zeros() as usize;
    assert_eq!(1 << nbits, nthreads);

    if r.len() < nbits {
        build_eq_x_r_vec_sequential(r)
    } else {
        let eq_ts = build_eq_x_r_vec_sequential(&r[(r.len() - nbits)..]);
        let mut ret = create_uninit_vec(1 << r.len());

        // eq(x, r) = eq(x_lo, r_lo) * eq(x_hi, r_hi)
        // where rlen = r.len(), x_lo = x[0..rlen-nbits], x_hi = x[rlen-nbits..]
        //  r_lo = r[0..rlen-nbits] and r_hi = r[rlen-nbits..]
        // each thread is associated with x_hi, and it will computes the subset
        // { eq(x_lo, r_lo) * eq(x_hi, r_hi) } whose cardinality equals to 2^{rlen-nbits}
        ret.par_chunks_mut(1 << (r.len() - nbits))
            .zip((0..nthreads).into_par_iter())
            .for_each(|(chunks, tid)| {
                let eq_t = eq_ts[tid];

                build_eq_x_r_helper_sequential(&r[..(r.len() - nbits)], chunks, eq_t);
            });
        unsafe { std::mem::transmute::<Vec<MaybeUninit<E>>, Vec<E>>(ret) }
    }
}

#[cfg(test)]
mod tests {
    use crate::virtual_poly::{build_eq_x_r_vec, build_eq_x_r_vec_sequential};
    use ark_std::rand::thread_rng;
    use ff_ext::{FromUniformBytes, GoldilocksExt2};

    #[test]
    fn test_build_eq() {
        env_logger::init();
        let mut rng = thread_rng();

        for num_vars in 10..24 {
            let r = (0..num_vars)
                .map(|_| GoldilocksExt2::random(&mut rng))
                .collect::<Vec<GoldilocksExt2>>();

            let seq_start = std::time::Instant::now();
            let eq_r_seq = build_eq_x_r_vec_sequential(&r);
            let seq_time = seq_start.elapsed();

            let par_start = std::time::Instant::now();
            let eq_r_par = build_eq_x_r_vec(&r);
            let par_time = par_start.elapsed();

            assert_eq!(eq_r_par, eq_r_seq);
            log::info!(
                "nv = {}, par_time: {:?}, seq_time: {:?}, speedup: {}",
                num_vars,
                par_time,
                seq_time,
                (seq_time.as_micros() as f64) / (par_time.as_micros() as f64)
            );
        }
    }
}
