use std::marker::PhantomData;

use super::{EncodingProverParameters, EncodingScheme};
use crate::{
    Error,
    util::{field_type_index_mul_base, log2_strict, plonky2_util::reverse_bits},
    vec_mut,
};
use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use multilinear_extensions::mle::FieldType;
use p3_field::{Field, FieldAlgebra, PrimeField, TwoAdicField};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::util::plonky2_util::reverse_index_bits_in_place;

use crate::util::arithmetic::horner;

pub trait RSCodeSpec: std::fmt::Debug + Clone {
    fn get_number_queries() -> usize;

    fn get_rate_log() -> usize;

    fn get_basecode_msg_size_log() -> usize;
}

/// The FFT codes in this file are borrowed and adapted from Plonky2.
type FftRootTable<F> = Vec<Vec<F>>;

pub fn fft_root_table<F: PrimeField + TwoAdicField>(lg_n: usize) -> FftRootTable<F> {
    // bases[i] = g^2^i, for i = 0, ..., lg_n - 1
    // Note that the end of bases is g^{n/2} = -1
    let mut bases = Vec::with_capacity(lg_n);
    let mut base = F::two_adic_generator(lg_n);
    bases.push(base);
    for _ in 1..lg_n {
        base = base.square(); // base = g^2^_
        bases.push(base);
    }

    // The result table looks like this:
    // len=2: [1, g^{n/2}=-1]
    // len=2: [1, g^{n/4}]
    // len=4: [1, g^{n/8}, g^{n/4}, g^{3n/8}]
    // len=8: [1, g^{n/16}, ..., g^{7n/16}]
    // ...
    // len=n/2: [1, g, ..., g^{n/2-1}]
    // There is no need to compute the other halves of these powers, because
    // those would be simply the negations of the previous halves.
    let mut root_table = Vec::with_capacity(lg_n);
    for lg_m in 1..=lg_n {
        let half_m = 1 << (lg_m - 1);
        let base = bases[lg_n - lg_m];
        let mut root_row = Vec::with_capacity(half_m.max(2));
        root_row.push(F::ONE);
        for i in 1..half_m.max(2) {
            root_row.push(root_row[i - 1] * base);
        }
        root_table.push(root_row);
    }
    root_table
}

#[allow(unused)]
fn ifft<E: ExtensionField>(
    poly: &mut FieldType<E>,
    zero_factor: usize,
    root_table: &FftRootTable<E::BaseField>,
) {
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = (E::BaseField::ONE + E::BaseField::ONE)
        .inverse()
        .exp_u64(lg_n as u64);

    fft(poly, zero_factor, root_table);

    // We reverse all values except the first, and divide each by n.
    field_type_index_mul_base(poly, 0, &n_inv);
    field_type_index_mul_base(poly, n / 2, &n_inv);
    vec_mut!(|poly| for i in 1..(n / 2) {
        let j = n - i;
        let coeffs_i = poly[j] * n_inv;
        let coeffs_j = poly[i] * n_inv;
        poly[i] = coeffs_i;
        poly[j] = coeffs_j;
    })
}

/// Core FFT implementation.
fn fft_classic_inner<E: ExtensionField>(
    values: &mut FieldType<E>,
    r: usize,
    lg_n: usize,
    root_table: &[Vec<E::BaseField>],
) {
    // We've already done the first lg_packed_width (if they were required) iterations.

    for (lg_half_m, cur_root_table) in root_table.iter().enumerate().take(lg_n).skip(r) {
        let n = 1 << lg_n;
        let lg_m = lg_half_m + 1;
        let m = 1 << lg_m; // Subarray size (in field elements).
        let half_m = m / 2;
        debug_assert!(half_m != 0);

        // omega values for this iteration, as slice of vectors
        let omega_table = &cur_root_table[..];
        vec_mut!(|values| {
            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let omega = omega_table[j];
                    let t = values[k + half_m + j] * omega;
                    let u = values[k + j];
                    values[k + j] = u + t;
                    values[k + half_m + j] = u - t;
                }
            }
        })
    }
}

/// FFT implementation based on Section 32.3 of "Introduction to
/// Algorithms" by Cormen et al.
///
/// The parameter r signifies that the first 1/2^r of the entries of
/// input may be non-zero, but the last 1 - 1/2^r entries are
/// definitely zero.
pub fn fft<E: ExtensionField>(
    values: &mut FieldType<E>,
    r: usize,
    root_table: &[Vec<E::BaseField>],
) {
    vec_mut!(|values| reverse_index_bits_in_place(values));

    let n = values.len();
    let lg_n = log2_strict(n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    // After reverse_index_bits, the only non-zero elements of values
    // are at indices i*2^r for i = 0..n/2^r.  The loop below copies
    // the value at i*2^r to the positions [i*2^r + 1, i*2^r + 2, ...,
    // (i+1)*2^r - 1]; i.e. it replaces the 2^r - 1 zeros following
    // element i*2^r with the value at i*2^r.  This corresponds to the
    // first r rounds of the FFT when there are 2^r zeros at the end
    // of the original input.
    if r > 0 {
        // if r == 0 then this loop is a noop.
        let mask = !((1 << r) - 1);
        match values {
            FieldType::Base(values) => {
                for i in 0..n {
                    values[i] = values[i & mask];
                }
            }
            FieldType::Ext(values) => {
                for i in 0..n {
                    values[i] = values[i & mask];
                }
            }
            _ => panic!("Unsupported field type"),
        }
    }

    fft_classic_inner::<E>(values, r, lg_n, root_table);
}

pub fn coset_fft<E: ExtensionField>(
    coeffs: &mut FieldType<E>,
    shift: E::BaseField,
    zero_factor: usize,
    root_table: &[Vec<E::BaseField>],
) {
    let mut shift_power = E::BaseField::ONE;
    vec_mut!(|coeffs| {
        for coeff in coeffs.iter_mut() {
            *coeff *= shift_power;
            shift_power *= shift;
        }
    });
    fft(coeffs, zero_factor, root_table);
}

#[derive(Debug, Clone)]
pub struct RSCodeDefaultSpec {}

impl RSCodeSpec for RSCodeDefaultSpec {
    // According to Theorem 1 of paper <BaseFold in the List Decoding Regime>
    // (https://eprint.iacr.org/2024/1571), the soundness error is bounded by
    // $O(1/|F|) + (\sqrt{\rho}+\epsilon)^s$
    // where $s$ is the query complexity and $\epsilon$ is a small value
    // that can be ignored. So the number of queries can be estimated by
    // $$
    // \frac{2\lambda}{-\log\rho}
    // $$
    // If we take $\lambda=100$ and $\rho=1/2$, then the number of queries is $200$.
    fn get_number_queries() -> usize {
        200
    }

    fn get_rate_log() -> usize {
        1
    }

    fn get_basecode_msg_size_log() -> usize {
        7
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct RSCodeParameters<E: ExtensionField> {
    pub(crate) fft_root_table: FftRootTable<E::BaseField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct RSCodeProverParameters<E: ExtensionField> {
    pub(crate) fft_root_table: FftRootTable<E::BaseField>,
    pub(crate) gamma_powers: Vec<E::BaseField>,
    pub(crate) gamma_powers_inv_div_two: Vec<E::BaseField>,
    pub(crate) full_message_size_log: usize,
}

impl<E: ExtensionField> EncodingProverParameters for RSCodeProverParameters<E> {
    fn get_max_message_size_log(&self) -> usize {
        self.full_message_size_log
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSCodeVerifierParameters<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// The verifier also needs a FFT table (much smaller)
    /// for small-size encoding. It contains the same roots as the
    /// prover's version for the first few levels (i < basecode_msg_size_log)
    /// For the other levels (i >= basecode_msg_size_log),
    /// it contains only the g^(2^i).
    pub(crate) fft_root_table: FftRootTable<E::BaseField>,
    pub(crate) full_message_size_log: usize,
    pub(crate) gamma_powers: Vec<E::BaseField>,
    pub(crate) gamma_powers_inv_div_two: Vec<E::BaseField>,
}

#[derive(Debug, Clone)]
pub struct RSCode<Spec: RSCodeSpec> {
    _phantom_data: PhantomData<Spec>,
}

impl<E: ExtensionField, Spec: RSCodeSpec> EncodingScheme<E> for RSCode<Spec>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type PublicParameters = RSCodeParameters<E>;

    type ProverParameters = RSCodeProverParameters<E>;

    type VerifierParameters = RSCodeVerifierParameters<E>;

    fn setup(max_message_size_log: usize) -> Self::PublicParameters {
        RSCodeParameters {
            fft_root_table: fft_root_table::<E::BaseField>(
                max_message_size_log + Spec::get_rate_log(),
            ),
        }
    }

    fn trim(
        mut pp: Self::PublicParameters,
        max_message_size_log: usize,
    ) -> Result<(Self::ProverParameters, Self::VerifierParameters), Error> {
        if pp.fft_root_table.len() < max_message_size_log + Spec::get_rate_log() {
            return Err(Error::InvalidPcsParam(format!(
                "Public parameter is setup for a smaller message size (log={}) than the trimmed message size (log={})",
                pp.fft_root_table.len() - Spec::get_rate_log(),
                max_message_size_log,
            )));
        }
        if max_message_size_log < Spec::get_basecode_msg_size_log() {
            // Message smaller than this size will not be encoded in BaseFold.
            // So just give trivial parameters.
            return Ok((
                Self::ProverParameters {
                    fft_root_table: vec![],
                    gamma_powers: vec![],
                    gamma_powers_inv_div_two: vec![],
                    full_message_size_log: max_message_size_log,
                },
                Self::VerifierParameters {
                    fft_root_table: vec![],
                    gamma_powers: vec![],
                    gamma_powers_inv_div_two: vec![],
                    full_message_size_log: max_message_size_log,
                },
            ));
        }
        let mut gamma_powers = Vec::with_capacity(max_message_size_log);
        let mut gamma_powers_inv = Vec::with_capacity(max_message_size_log);
        gamma_powers.push(E::BaseField::GENERATOR);
        gamma_powers_inv.push(E::BaseField::GENERATOR.inverse());
        for i in 1..max_message_size_log + Spec::get_rate_log() {
            gamma_powers.push(gamma_powers[i - 1].square());
            gamma_powers_inv.push(gamma_powers_inv[i - 1].square());
        }
        let inv_of_two = E::BaseField::from_canonical_u64(2).inverse();
        gamma_powers_inv.iter_mut().for_each(|x| *x *= inv_of_two);
        pp.fft_root_table
            .truncate(max_message_size_log + Spec::get_rate_log());
        let verifier_fft_root_table = pp.fft_root_table
            [..Spec::get_basecode_msg_size_log() + Spec::get_rate_log()]
            .iter()
            .cloned()
            .chain(
                pp.fft_root_table[Spec::get_basecode_msg_size_log() + Spec::get_rate_log()..]
                    .iter()
                    .map(|v| vec![v[1]]),
            )
            .collect();
        Ok((
            Self::ProverParameters {
                fft_root_table: pp.fft_root_table,
                gamma_powers: gamma_powers.clone(),
                gamma_powers_inv_div_two: gamma_powers_inv.clone(),
                full_message_size_log: max_message_size_log,
            },
            Self::VerifierParameters {
                fft_root_table: verifier_fft_root_table,
                full_message_size_log: max_message_size_log,
                gamma_powers,
                gamma_powers_inv_div_two: gamma_powers_inv,
            },
        ))
    }

    fn encode(pp: &Self::ProverParameters, coeffs: &FieldType<E>) -> FieldType<E> {
        assert!(log2_strict(coeffs.len()) >= Spec::get_basecode_msg_size_log());
        // Use the full message size to determine the shift factor.
        Self::encode_internal(&pp.fft_root_table, coeffs, pp.full_message_size_log)
    }

    fn encode_small(vp: &Self::VerifierParameters, coeffs: &FieldType<E>) -> FieldType<E> {
        // Use the full message size to determine the shift factor.
        Self::encode_internal(&vp.fft_root_table, coeffs, vp.full_message_size_log)
    }

    fn get_number_queries() -> usize {
        Spec::get_number_queries()
    }

    fn get_rate_log() -> usize {
        Spec::get_rate_log()
    }

    fn get_basecode_msg_size_log() -> usize {
        Spec::get_basecode_msg_size_log()
    }

    fn message_is_left_and_right_folding() -> bool {
        false
    }

    fn prover_folding_coeffs(pp: &Self::ProverParameters, level: usize, index: usize) -> (E, E, E) {
        // The coefficients are for the bit-reversed codeword, so reverse the
        // bits before providing the coefficients.
        let index = reverse_bits(index, level);
        // level is the logarithmic of the codeword size after folded.
        // Therefore, the domain after folded is gamma^2^(full_log_n - level) H
        // where H is the multiplicative subgroup of size 2^level.
        // The element at index i in this domain is
        // gamma^2^(full_log_n - level) * ((2^level)-th root of unity)^i
        // The x0 and x1 are exactly the two square roots, i.e.,
        // x0 = gamma^2^(full_log_n - level - 1) * ((2^(level+1))-th root of unity)^i
        // Since root_table[i] stores the first half of the powers of
        // the 2^(i+1)-th roots of unity, we can avoid recomputing them.
        let x0 = if index < (1 << level) {
            pp.fft_root_table[level][index]
        } else {
            -pp.fft_root_table[level][index - (1 << level)]
        } * pp.gamma_powers[pp.full_message_size_log + Spec::get_rate_log() - level - 1];
        let x1 = -x0;
        // The weight is 1/(x1-x0) = -1/(2x0)
        // = -1/2 * (gamma^{-1})^2^(full_codeword_log_n - level - 1) * ((2^(level+1))-th root of unity)^{2^(level+1)-i}
        let w = -pp.gamma_powers_inv_div_two
            [pp.full_message_size_log + Spec::get_rate_log() - level - 1]
            * if index == 0 {
                E::BaseField::ONE
            } else if index < (1 << level) {
                -pp.fft_root_table[level][(1 << level) - index]
            } else if index == 1 << level {
                -E::BaseField::ONE
            } else {
                pp.fft_root_table[level][(1 << (level + 1)) - index]
            };
        (E::from(x0), E::from(x1), E::from(w))
    }

    fn verifier_folding_coeffs(
        vp: &Self::VerifierParameters,
        level: usize,
        index: usize,
    ) -> (E, E, E) {
        // The coefficients are for the bit-reversed codeword, so reverse the
        // bits before providing the coefficients.
        let index = reverse_bits(index, level);
        // The same as prover_folding_coeffs, except that the powers of
        // g is computed on the fly for levels exceeding the root table.
        let x0 = if level < Spec::get_basecode_msg_size_log() + Spec::get_rate_log() {
            if index < (1 << level) {
                vp.fft_root_table[level][index]
            } else {
                -vp.fft_root_table[level][index - (1 << level)]
            }
        } else {
            // In this case, the level-th row of fft root table of the verifier
            // only stores the first 2^(level+1)-th roots of unity.
            vp.fft_root_table[level][0].exp_u64(index as u64)
        } * vp.gamma_powers[vp.full_message_size_log + Spec::get_rate_log() - level - 1];
        let x1 = -x0;
        // The weight is 1/(x1-x0) = -1/(2x0)
        // = -1/2 * (gamma^{-1})^2^(full_log_n - level - 1) * ((2^(level+1))-th root of unity)^{2^(level+1)-i}
        let w = -vp.gamma_powers_inv_div_two
            [vp.full_message_size_log + Spec::get_rate_log() - level - 1]
            * if level < Spec::get_basecode_msg_size_log() + Spec::get_rate_log() {
                if index == 0 {
                    E::BaseField::ONE
                } else if index < (1 << level) {
                    -vp.fft_root_table[level][(1 << level) - index]
                } else if index == 1 << level {
                    -E::BaseField::ONE
                } else {
                    vp.fft_root_table[level][(1 << (level + 1)) - index]
                }
            } else {
                // In this case, this level of fft root table of the verifier
                // only stores the first 2^(level+1)-th root of unity.
                vp.fft_root_table[level][0].exp_u64((1 << (level + 1)) - index as u64)
            };
        (E::from(x0), E::from(x1), E::from(w))
    }
}

impl<Spec: RSCodeSpec> RSCode<Spec> {
    fn encode_internal<E: ExtensionField>(
        fft_root_table: &FftRootTable<E::BaseField>,
        coeffs: &FieldType<E>,
        full_message_size_log: usize,
    ) -> FieldType<E>
    where
        E::BaseField: Serialize + DeserializeOwned,
    {
        let lg_m = log2_strict(coeffs.len());
        let fft_root_table = &fft_root_table[..lg_m + Spec::get_rate_log()];
        assert!(
            lg_m <= full_message_size_log,
            "Encoded message exceeds the maximum supported message size of the table."
        );
        let rate = 1 << Spec::get_rate_log();
        let mut ret = match coeffs {
            FieldType::Base(coeffs) => {
                let mut coeffs = coeffs.clone();
                coeffs.extend(itertools::repeat_n(
                    E::BaseField::ZERO,
                    coeffs.len() * (rate - 1),
                ));
                FieldType::Base(coeffs)
            }
            FieldType::Ext(coeffs) => {
                let mut coeffs = coeffs.clone();
                coeffs.extend(itertools::repeat_n(E::ZERO, coeffs.len() * (rate - 1)));
                FieldType::Ext(coeffs)
            }
            _ => panic!("Unsupported field type"),
        };
        // Let gamma be the multiplicative generator of the base field.
        // The full domain is gamma H where H is the multiplicative subgroup
        // of size n * rate.
        // When the input message size is not n, but n/2^k, then the domain is
        // gamma^2^k H.
        coset_fft(
            &mut ret,
            E::BaseField::GENERATOR.exp_power_of_2(full_message_size_log - lg_m),
            Spec::get_rate_log(),
            fft_root_table,
        );
        ret
    }

    #[allow(unused)]
    fn folding_coeffs_naive<E: ExtensionField>(
        level: usize,
        index: usize,
        full_message_size_log: usize,
    ) -> (E, E, E) {
        // The coefficients are for the bit-reversed codeword, so reverse the
        // bits before providing the coefficients.
        let index = reverse_bits(index, level);
        // x0 is the index-th 2^(level+1)-th root of unity, multiplied by
        // the shift factor at level+1, which is gamma^2^(full_codeword_log_n - level - 1).
        let x0 = E::BaseField::two_adic_generator(level + 1).exp_u64(index as u64)
            * E::BaseField::GENERATOR
                .exp_power_of_2(full_message_size_log + Spec::get_rate_log() - level - 1);
        let x1 = -x0;
        let w = (x1 - x0).inverse();
        (E::from(x0), E::from(x1), E::from(w))
    }
}

#[allow(unused)]
fn naive_fft<E: ExtensionField>(poly: &[E], rate: usize, shift: E::BaseField) -> Vec<E> {
    let timer = start_timer!(|| "Encode RSCode");
    let message_size = poly.len();
    let domain_size_bit = log2_strict(message_size * rate);
    let root = E::BaseField::two_adic_generator(domain_size_bit);
    // The domain is shift * H where H is the multiplicative subgroup of size
    // message_size * rate.
    let mut domain = Vec::<E::BaseField>::with_capacity(message_size * rate);
    domain.push(shift);
    for i in 1..message_size * rate {
        domain.push(domain[i - 1] * root);
    }
    let mut res = vec![E::ZERO; message_size * rate];
    res.iter_mut()
        .enumerate()
        .for_each(|(i, target)| *target = horner(poly, &E::from(domain[i])));
    end_timer!(timer);

    res
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;
    use p3_goldilocks::Goldilocks;

    use crate::{
        basefold::encoding::test_util::test_codeword_folding,
        util::{field_type_index_ext, plonky2_util::reverse_index_bits_in_place_field_type},
    };
    use ff_ext::FromUniformBytes;

    use super::*;

    #[test]
    fn test_naive_fft() {
        let num_vars = 5;

        let poly: Vec<GoldilocksExt2> = (0..(1 << num_vars))
            .map(GoldilocksExt2::from_canonical_u64)
            .collect();
        let mut poly2 = FieldType::Ext(poly.clone());

        let naive = naive_fft::<GoldilocksExt2>(&poly, 1, Goldilocks::ONE);

        let root_table = fft_root_table(num_vars);
        fft::<GoldilocksExt2>(&mut poly2, 0, &root_table);

        let poly2 = match poly2 {
            FieldType::Ext(coeffs) => coeffs,
            _ => panic!("Wrong field type"),
        };
        assert_eq!(naive, poly2);
    }

    #[test]
    fn test_naive_fft_with_shift() {
        use rand::rngs::OsRng;
        let num_vars = 5;

        let poly: Vec<GoldilocksExt2> = (0..(1 << num_vars))
            .map(|_| GoldilocksExt2::random(&mut OsRng))
            .collect();
        let mut poly2 = FieldType::Ext(poly.clone());

        let naive = naive_fft::<GoldilocksExt2>(&poly, 1, Goldilocks::GENERATOR);

        let root_table = fft_root_table(num_vars);
        coset_fft::<GoldilocksExt2>(&mut poly2, Goldilocks::GENERATOR, 0, &root_table);

        let poly2 = match poly2 {
            FieldType::Ext(coeffs) => coeffs,
            _ => panic!("Wrong field type"),
        };
        assert_eq!(naive, poly2);
    }

    #[test]
    fn test_naive_fft_with_rate() {
        use rand::rngs::OsRng;
        let num_vars = 5;
        let rate_bits = 1;

        let poly: Vec<GoldilocksExt2> = (0..(1 << num_vars))
            .map(|_| GoldilocksExt2::random(&mut OsRng))
            .collect();
        let mut poly2 = vec![GoldilocksExt2::ZERO; poly.len() * (1 << rate_bits)];
        poly2.as_mut_slice()[..poly.len()].copy_from_slice(poly.as_slice());
        let mut poly2 = FieldType::Ext(poly2.clone());

        let naive = naive_fft::<GoldilocksExt2>(&poly, 1 << rate_bits, Goldilocks::GENERATOR);

        let root_table = fft_root_table(num_vars + rate_bits);
        coset_fft::<GoldilocksExt2>(&mut poly2, Goldilocks::GENERATOR, rate_bits, &root_table);

        let poly2 = match poly2 {
            FieldType::Ext(coeffs) => coeffs,
            _ => panic!("Wrong field type"),
        };
        assert_eq!(naive, poly2);
    }

    #[test]
    fn test_ifft() {
        let num_vars = 5;

        let poly: Vec<GoldilocksExt2> = (0..(1 << num_vars))
            .map(GoldilocksExt2::from_canonical_u64)
            .collect();
        let mut poly = FieldType::Ext(poly);
        let original = poly.clone();

        let root_table = fft_root_table(num_vars);
        fft::<GoldilocksExt2>(&mut poly, 0, &root_table);
        ifft::<GoldilocksExt2>(&mut poly, 0, &root_table);

        assert_eq!(original, poly);
    }

    #[test]
    fn prover_verifier_consistency() {
        type Code = RSCode<RSCodeDefaultSpec>;
        let pp: RSCodeParameters<GoldilocksExt2> = Code::setup(10);
        let (pp, vp) = Code::trim(pp, 10).unwrap();
        for level in 0..(10 + <Code as EncodingScheme<GoldilocksExt2>>::get_rate_log()) {
            for index in 0..(1 << level) {
                let (naive_x0, naive_x1, naive_w) =
                    Code::folding_coeffs_naive(level, index, pp.full_message_size_log);
                let (p_x0, p_x1, p_w) = Code::prover_folding_coeffs(&pp, level, index);
                let (v_x0, v_x1, v_w) = Code::verifier_folding_coeffs(&vp, level, index);
                // assert_eq!(v_w * (v_x1 - v_x0), GoldilocksExt2::ONE);
                // assert_eq!(p_w * (p_x1 - p_x0), GoldilocksExt2::ONE);
                assert_eq!(
                    (v_x0, v_x1, v_w, p_x0, p_x1, p_w),
                    (naive_x0, naive_x1, naive_w, naive_x0, naive_x1, naive_w),
                    "failed for level = {}, index = {}",
                    level,
                    index
                );
            }
        }
    }

    #[test]
    fn test_rs_codeword_folding() {
        test_codeword_folding::<GoldilocksExt2, RSCode<RSCodeDefaultSpec>>();
    }

    type E = GoldilocksExt2;
    type F = Goldilocks;
    type Code = RSCode<RSCodeDefaultSpec>;

    #[test]
    pub fn test_colinearity() {
        let num_vars = 10;

        let poly: Vec<E> = (0..(1 << num_vars)).map(E::from_canonical_u64).collect();
        let poly = FieldType::Ext(poly);

        let pp = <Code as EncodingScheme<E>>::setup(num_vars);
        let (pp, _) = Code::trim(pp, num_vars).unwrap();
        let mut codeword = Code::encode(&pp, &poly);
        reverse_index_bits_in_place_field_type(&mut codeword);
        let challenge = E::from_canonical_u64(2);
        let folded_codeword = Code::fold_bitreversed_codeword(&pp, &codeword, challenge);
        let codeword = match codeword {
            FieldType::Ext(coeffs) => coeffs,
            _ => panic!("Wrong field type"),
        };

        for (i, (a, b)) in folded_codeword.iter().zip(codeword.chunks(2)).enumerate() {
            let (x0, x1, _) = Code::prover_folding_coeffs(
                &pp,
                num_vars + <Code as EncodingScheme<E>>::get_rate_log() - 1,
                i,
            );
            // Check that (x0, b[0]), (x1, b[1]) and (challenge, a) are
            // on the same line, i.e.,
            // (b[0]-a)/(x0-challenge) = (b[1]-a)/(x1-challenge)
            // which is equivalent to
            // (x0-challenge)*(b[1]-a) = (x1-challenge)*(b[0]-a)
            assert_eq!(
                (x0 - challenge) * (b[1] - *a),
                (x1 - challenge) * (b[0] - *a),
                "failed for i = {}",
                i
            );
        }
    }

    #[test]
    pub fn test_low_degree() {
        let num_vars = 10;

        let poly: Vec<E> = (0..(1 << num_vars)).map(E::from_canonical_u64).collect();
        let poly = FieldType::Ext(poly);

        let pp = <Code as EncodingScheme<E>>::setup(num_vars);
        let (pp, _) = Code::trim(pp, num_vars).unwrap();
        let mut codeword = Code::encode(&pp, &poly);
        check_low_degree(&codeword, "low degree check for original codeword");
        let c0 = field_type_index_ext(&codeword, 0);
        let c_mid = field_type_index_ext(&codeword, codeword.len() >> 1);
        let c1 = field_type_index_ext(&codeword, 1);
        let c_mid1 = field_type_index_ext(&codeword, (codeword.len() >> 1) + 1);

        reverse_index_bits_in_place_field_type(&mut codeword);
        // After the bit inversion, the first element is still the first,
        // but the middle one is switched to the second.
        assert_eq!(c0, field_type_index_ext(&codeword, 0));
        assert_eq!(c_mid, field_type_index_ext(&codeword, 1));
        // The second element is placed at the middle, and the next to middle
        // element is still at the place.
        assert_eq!(c1, field_type_index_ext(&codeword, codeword.len() >> 1));
        assert_eq!(
            c_mid1,
            field_type_index_ext(&codeword, (codeword.len() >> 1) + 1)
        );

        // For RS codeword, the addition of the left and right halves is also
        // a valid codeword
        let codeword_vec = match &codeword {
            FieldType::Ext(coeffs) => coeffs.clone(),
            _ => panic!("Wrong field type"),
        };
        let mut left_right_sum: Vec<E> = codeword_vec
            .chunks(2)
            .map(|chunk| chunk[0] + chunk[1])
            .collect();
        assert_eq!(left_right_sum[0], c0 + c_mid);
        reverse_index_bits_in_place(&mut left_right_sum);
        assert_eq!(left_right_sum[1], c1 + c_mid1);
        check_low_degree(
            &FieldType::Ext(left_right_sum.clone()),
            "check low degree of left+right",
        );

        // The the difference of the left and right halves is also
        // a valid codeword after twisted by omega^(-i), regardless of the
        // shift of the coset.
        let mut left_right_diff: Vec<E> = codeword_vec
            .chunks(2)
            .map(|chunk| chunk[0] - chunk[1])
            .collect();
        assert_eq!(left_right_diff[0], c0 - c_mid);
        reverse_index_bits_in_place(&mut left_right_diff);
        assert_eq!(left_right_diff[1], c1 - c_mid1);
        let root_of_unity_inv = F::two_adic_generator(F::TWO_ADICITY)
            .inverse()
            .exp_power_of_2(F::TWO_ADICITY - log2_strict(left_right_diff.len()) - 1);
        for (i, coeff) in left_right_diff.iter_mut().enumerate() {
            *coeff *= root_of_unity_inv.exp_u64(i as u64);
        }
        assert_eq!(left_right_diff[0], c0 - c_mid);
        assert_eq!(left_right_diff[1], (c1 - c_mid1) * root_of_unity_inv);
        check_low_degree(
            &FieldType::Ext(left_right_diff.clone()),
            "check low degree of (left-right)*omega^(-i)",
        );

        let challenge = E::from_canonical_u64(2);
        let folded_codeword = Code::fold_bitreversed_codeword(&pp, &codeword, challenge);
        let c_fold = folded_codeword[0];
        let c_fold1 = folded_codeword[folded_codeword.len() >> 1];
        let mut folded_codeword = FieldType::Ext(folded_codeword);
        reverse_index_bits_in_place_field_type(&mut folded_codeword);
        assert_eq!(c_fold, field_type_index_ext(&folded_codeword, 0));
        assert_eq!(c_fold1, field_type_index_ext(&folded_codeword, 1));

        // The top level folding coefficient should have shift factor gamma
        let folding_coeffs = Code::prover_folding_coeffs(&pp, log2_strict(codeword.len()) - 1, 0);
        assert_eq!(folding_coeffs.0, E::from(F::GENERATOR));
        assert_eq!(folding_coeffs.0 + folding_coeffs.1, E::ZERO);
        assert_eq!(
            (folding_coeffs.1 - folding_coeffs.0) * folding_coeffs.2,
            E::ONE
        );
        // The three points (x0, c0), (x1, c_mid), (challenge, c_fold) should
        // be collinear
        assert_eq!(
            (c_mid - c_fold) * (folding_coeffs.0 - challenge),
            (c0 - c_fold) * (folding_coeffs.1 - challenge),
        );
        // So the folded value should be equal to
        // (gamma^{-1} * alpha * (c0 - c_mid) + (c0 + c_mid)) / 2
        assert_eq!(
            c_fold * F::GENERATOR * F::from_canonical_u64(2),
            challenge * (c0 - c_mid) + (c0 + c_mid) * F::GENERATOR
        );
        assert_eq!(
            c_fold * F::GENERATOR * F::from_canonical_u64(2),
            challenge * left_right_diff[0] + left_right_sum[0] * F::GENERATOR
        );
        assert_eq!(
            c_fold * F::from_canonical_u64(2),
            challenge * left_right_diff[0] * F::GENERATOR.inverse() + left_right_sum[0]
        );

        let folding_coeffs = Code::prover_folding_coeffs(&pp, log2_strict(codeword.len()) - 1, 1);
        let root_of_unity = F::two_adic_generator(log2_strict(codeword.len()));
        assert_eq!(root_of_unity.exp_u64(codeword.len() as u64), F::ONE);
        assert_eq!(root_of_unity.exp_u64((codeword.len() >> 1) as u64), -F::ONE);
        assert_eq!(
            folding_coeffs.0,
            E::from(F::GENERATOR) * E::from(root_of_unity).exp_u64((codeword.len() >> 2) as u64)
        );
        assert_eq!(folding_coeffs.0 + folding_coeffs.1, E::ZERO);
        assert_eq!(
            (folding_coeffs.1 - folding_coeffs.0) * folding_coeffs.2,
            E::ONE
        );

        // The folded codeword is the linear combination of the left+right and the
        // twisted left-right vectors.
        // The coefficients are respectively 1/2 and gamma^{-1}/2 * alpha.
        // In another word, the folded codeword multiplied by 2 is the linear
        // combination by coeffs: 1 and gamma^{-1} * alpha
        let gamma_inv = F::GENERATOR.inverse();
        let b = challenge * gamma_inv;
        let folded_codeword_vec = match &folded_codeword {
            FieldType::Ext(coeffs) => coeffs.clone(),
            _ => panic!("Wrong field type"),
        };
        assert_eq!(
            c_fold * F::from_canonical_u64(2),
            left_right_diff[0] * b + left_right_sum[0]
        );
        for (i, (c, (diff, sum))) in folded_codeword_vec
            .iter()
            .zip(left_right_diff.iter().zip(left_right_sum.iter()))
            .enumerate()
        {
            assert_eq!(*c + *c, *sum + b * *diff, "failed for i = {}", i);
        }

        check_low_degree(&folded_codeword, "low degree check for folded");
    }

    fn check_low_degree(codeword: &FieldType<E>, message: &str) {
        let mut codeword = codeword.clone();
        let codeword_bits = log2_strict(codeword.len());
        let root_table = fft_root_table(codeword_bits);
        let original = codeword.clone();
        ifft(&mut codeword, 0, &root_table);
        for i in (codeword.len() >> <Code as EncodingScheme<E>>::get_rate_log())..codeword.len() {
            assert_eq!(
                field_type_index_ext(&codeword, i),
                E::ZERO,
                "{}: zero check failed for i = {}",
                message,
                i
            )
        }
        fft(&mut codeword, 0, &root_table);
        let original = match original {
            FieldType::Ext(coeffs) => coeffs,
            _ => panic!("Wrong field type"),
        };
        let codeword = match codeword {
            FieldType::Ext(coeffs) => coeffs,
            _ => panic!("Wrong field type"),
        };
        original
            .iter()
            .zip(codeword.iter())
            .enumerate()
            .for_each(|(i, (a, b))| {
                assert_eq!(a, b, "{}: failed for i = {}", message, i);
            });
    }
}
