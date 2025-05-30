use std::marker::PhantomData;

use super::{EncodingProverParameters, EncodingScheme, concatenate_field_types};
use crate::{
    Error,
    util::{
        arithmetic::base_from_raw_bytes, log2_strict, num_of_bytes, plonky2_util::reverse_bits,
    },
    vec_mut,
};
use aes::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use multilinear_extensions::mle::FieldType;
use p3_field::{Field, FieldAlgebra, batch_multiplicative_inverse};
use rand::SeedableRng;
use rayon::prelude::{ParallelIterator, ParallelSlice, ParallelSliceMut};

use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::util::plonky2_util::reverse_index_bits_in_place;
use rand_chacha::{ChaCha8Rng, rand_core::RngCore};
use rayon::prelude::IntoParallelRefIterator;

use crate::util::arithmetic::{horner, steps};

pub trait BasecodeSpec: std::fmt::Debug + Clone {
    fn get_number_queries() -> usize;

    fn get_rate_log() -> usize;

    fn get_basecode_msg_size_log() -> usize;
}

#[derive(Debug, Clone)]
pub struct BasecodeDefaultSpec {}

impl BasecodeSpec for BasecodeDefaultSpec {
    fn get_number_queries() -> usize {
        766
    }

    fn get_rate_log() -> usize {
        3
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
pub struct BasecodeParameters<E: ExtensionField> {
    pub(crate) table: Vec<Vec<E::BaseField>>,
    pub(crate) table_w_weights: Vec<Vec<(E::BaseField, E::BaseField)>>,
    pub(crate) rng_seed: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::BaseField: Serialize",
    deserialize = "E::BaseField: DeserializeOwned"
))]
pub struct BasecodeProverParameters<E: ExtensionField, Spec: BasecodeSpec> {
    pub(crate) table: Vec<Vec<E::BaseField>>,
    pub(crate) table_w_weights: Vec<Vec<(E::BaseField, E::BaseField)>>,
    pub(crate) rng_seed: [u8; 32],
    #[serde(skip)]
    _phantom: PhantomData<fn() -> Spec>,
}

impl<E: ExtensionField, Spec: BasecodeSpec> EncodingProverParameters
    for BasecodeProverParameters<E, Spec>
{
    fn get_max_message_size_log(&self) -> usize {
        self.table.len() - Spec::get_rate_log()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasecodeVerifierParameters {
    pub(crate) rng_seed: [u8; 32],
    pub(crate) aes_key: [u8; 16],
    pub(crate) aes_iv: [u8; 16],
}

#[derive(Debug, Clone)]
pub struct Basecode<Spec: BasecodeSpec> {
    _phantom_data: PhantomData<Spec>,
}

impl<E: ExtensionField, Spec: BasecodeSpec> EncodingScheme<E> for Basecode<Spec>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type PublicParameters = BasecodeParameters<E>;

    type ProverParameters = BasecodeProverParameters<E, Spec>;

    type VerifierParameters = BasecodeVerifierParameters;

    fn setup(max_msg_size_log: usize) -> Self::PublicParameters {
        let rng = ChaCha8Rng::from_seed([0u8; 32]);
        let (table_w_weights, table) =
            get_table_aes::<E, _>(max_msg_size_log, Spec::get_rate_log(), &mut rng.clone());
        BasecodeParameters {
            table,
            table_w_weights,
            rng_seed: [0u8; 32],
        }
    }

    fn trim(
        mut pp: Self::PublicParameters,
        max_msg_size_log: usize,
    ) -> Result<(Self::ProverParameters, Self::VerifierParameters), Error> {
        if pp.table.len() < Spec::get_rate_log() + max_msg_size_log {
            return Err(Error::InvalidPcsParam(format!(
                "Public parameter is setup for a smaller message size (log={}) than the trimmed message size (log={})",
                pp.table.len() - Spec::get_rate_log(),
                max_msg_size_log,
            )));
        }
        pp.table_w_weights
            .truncate(Spec::get_rate_log() + max_msg_size_log);
        pp.table.truncate(Spec::get_rate_log() + max_msg_size_log);
        let mut key: [u8; 16] = [0u8; 16];
        let mut iv: [u8; 16] = [0u8; 16];
        let mut rng = ChaCha8Rng::from_seed(pp.rng_seed);
        rng.set_word_pos(0);
        rng.fill_bytes(&mut key);
        rng.fill_bytes(&mut iv);
        Ok((
            Self::ProverParameters {
                table_w_weights: pp.table_w_weights,
                table: pp.table,
                rng_seed: pp.rng_seed,
                _phantom: PhantomData,
            },
            Self::VerifierParameters {
                rng_seed: pp.rng_seed,
                aes_key: key,
                aes_iv: iv,
            },
        ))
    }

    fn encode(pp: &Self::ProverParameters, coeffs: &FieldType<E>) -> FieldType<E> {
        // Split the input into chunks of message size, encode each message, and return the codewords
        let basecode = encode_field_type_rs_basecode(
            coeffs,
            1 << Spec::get_rate_log(),
            1 << Spec::get_basecode_msg_size_log(),
        );

        // Apply the recursive definition of the BaseFold code to the list of base codewords,
        // and produce the final codeword
        evaluate_over_foldable_domain_generic_basecode::<E>(
            1 << Spec::get_basecode_msg_size_log(),
            coeffs.len(),
            Spec::get_rate_log(),
            basecode,
            &pp.table,
        )
    }

    fn encode_small(_vp: &Self::VerifierParameters, coeffs: &FieldType<E>) -> FieldType<E> {
        let mut basecodes =
            encode_field_type_rs_basecode(coeffs, 1 << Spec::get_rate_log(), coeffs.len());
        assert_eq!(basecodes.len(), 1);
        basecodes.remove(0)
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
        true
    }

    fn prover_folding_coeffs(pp: &Self::ProverParameters, level: usize, index: usize) -> (E, E, E) {
        let level = &pp.table_w_weights[level];
        (
            E::from(level[index].0),
            E::from(-level[index].0),
            E::from(level[index].1),
        )
    }

    fn verifier_folding_coeffs(
        vp: &Self::VerifierParameters,
        level: usize,
        index: usize,
    ) -> (E, E, E) {
        type Aes128Ctr64LE = ctr::Ctr32LE<aes::Aes128>;
        let mut cipher = Aes128Ctr64LE::new((vp.aes_key[..]).into(), (vp.aes_iv[..]).into());

        let x0: E::BaseField = query_root_table_from_rng_aes::<E>(level, index, &mut cipher);
        let x1 = -x0;

        let w = (x1 - x0).inverse();

        (E::from(x0), E::from(x1), E::from(w))
    }
}

fn encode_field_type_rs_basecode<E: ExtensionField>(
    poly: &FieldType<E>,
    rate: usize,
    message_size: usize,
) -> Vec<FieldType<E>> {
    match poly {
        FieldType::Ext(poly) => get_basecode(poly, rate, message_size)
            .iter()
            .map(|x| FieldType::Ext(x.clone()))
            .collect(),
        FieldType::Base(poly) => get_basecode(poly, rate, message_size)
            .iter()
            .map(|x| FieldType::Base(x.clone()))
            .collect(),
        _ => panic!("Unsupported field type"),
    }
}

// Split the input into chunks of message size, encode each message, and return the codewords
// FIXME: It is expensive for now because it is using naive FFT (although it is
// over a small domain)
fn get_basecode<F: Field>(poly: &[F], rate: usize, message_size: usize) -> Vec<Vec<F>> {
    let timer = start_timer!(|| "Encode basecode");
    // The domain is just counting 1, 2, 3, ... , domain_size
    let domain: Vec<F> = steps(F::ONE).take(message_size * rate).collect();
    let res = poly
        .par_chunks_exact(message_size)
        .map(|chunk| {
            let mut target = vec![F::ZERO; message_size * rate];
            // Just Reed-Solomon code, but with the naive domain
            target
                .iter_mut()
                .enumerate()
                .for_each(|(i, target)| *target = horner(chunk, &domain[i]));
            target
        })
        .collect::<Vec<Vec<F>>>();
    end_timer!(timer);

    res
}

// this function assumes all codewords in base_codeword has equivalent length
pub fn evaluate_over_foldable_domain_generic_basecode<E: ExtensionField>(
    base_message_length: usize,
    num_coeffs: usize,
    log_rate: usize,
    base_codewords: Vec<FieldType<E>>,
    table: &[Vec<E::BaseField>],
) -> FieldType<E> {
    let timer = start_timer!(|| "evaluate over foldable domain");
    let k = num_coeffs;
    let logk = log2_strict(k);
    let base_log_k = log2_strict(base_message_length);
    // concatenate together all base codewords
    //    let now = Instant::now();
    let mut coeffs_with_bc = concatenate_field_types(&base_codewords);
    //    println!("concatenate base codewords {:?}", now.elapsed());
    // iterate over array, replacing even indices with (evals[i] - evals[(i+1)])
    let mut chunk_size = base_codewords[0].len(); // block length of the base code
    for i in base_log_k..logk {
        // In beginning of each iteration, the current codeword size is 1<<i, after this iteration,
        // every two adjacent codewords are folded into one codeword of size 1<<(i+1).
        // Fetch the table that has the same size of the *current* codeword size.
        let level = &table[i + log_rate];
        // chunk_size is equal to 1 << (i+1), i.e., the codeword size after the current iteration
        // half_chunk is equal to 1 << i, i.e. the current codeword size
        chunk_size <<= 1;
        assert_eq!(level.len(), chunk_size >> 1);
        vec_mut!(coeffs_with_bc, |c| {
            c.par_chunks_mut(chunk_size).for_each(|chunk| {
                let half_chunk = chunk_size >> 1;
                for j in half_chunk..chunk_size {
                    // Suppose the current codewords are (a, b)
                    // The new codeword is computed by two halves:
                    // left  = a + t * b
                    // right = a - t * b
                    let rhs = chunk[j] * level[j - half_chunk];
                    chunk[j] = chunk[j - half_chunk] - rhs;
                    chunk[j - half_chunk] += rhs;
                }
            });
        });
    }
    end_timer!(timer);
    coeffs_with_bc
}

#[allow(clippy::type_complexity)]
pub fn get_table_aes<E: ExtensionField, Rng: RngCore + Clone>(
    poly_size_log: usize,
    rate: usize,
    rng: &mut Rng,
) -> (
    Vec<Vec<(E::BaseField, E::BaseField)>>,
    Vec<Vec<E::BaseField>>,
) {
    // The size (logarithmic) of the codeword for the polynomial
    let lg_n: usize = rate + poly_size_log;

    let mut key: [u8; 16] = [0u8; 16];
    let mut iv: [u8; 16] = [0u8; 16];
    rng.fill_bytes(&mut key);
    rng.fill_bytes(&mut iv);

    type Aes128Ctr64LE = ctr::Ctr32LE<aes::Aes128>;

    let mut cipher = Aes128Ctr64LE::new(key[..].into(), iv[..].into());

    // Allocate the buffer for storing n field elements (the entire codeword)
    let bytes = num_of_bytes::<E::BaseField>(1 << lg_n);
    let mut dest: Vec<u8> = vec![0u8; bytes];
    cipher.apply_keystream(&mut dest[..]);

    // Now, dest is a vector filled with random data for a field vector of size n

    // Collect the bytes into field elements
    let flat_table: Vec<E::BaseField> = dest
        .par_chunks_exact(num_of_bytes::<E::BaseField>(1))
        .map(|chunk| base_from_raw_bytes::<E>(chunk))
        .collect::<Vec<_>>();

    // Now, flat_table is a field vector of size n, filled with random field elements
    assert_eq!(flat_table.len(), 1 << lg_n);

    // Multiply -2 to every element to get the weights. Now weights = { -2x }
    let weights: Vec<E::BaseField> = flat_table
        .par_iter()
        .map(|el| E::BaseField::ZERO - *el - *el)
        .collect();

    // Then invert all the elements. Now weights = { -1/2x }
    let weights = batch_multiplicative_inverse(&weights);

    // Zip x and -1/2x together. The result is the list { (x, -1/2x) }
    // What is this -1/2x? It is used in linear interpolation over the domain (x, -x), which
    // involves computing 1/(b-a) where b=-x and a=x, and 1/(b-a) here is exactly -1/2x
    let flat_table_w_weights = flat_table
        .iter()
        .zip(weights)
        .map(|(el, w)| (*el, w))
        .collect_vec();

    // Split the positions from 0 to n-1 into slices of sizes:
    // 2, 2, 4, 8, ..., n/2, exactly lg_n number of them
    // The weights are (x, -1/2x), the table elements are just x

    let mut unflattened_table_w_weights = vec![Vec::new(); lg_n];
    let mut unflattened_table = vec![Vec::new(); lg_n];

    unflattened_table_w_weights[0] = flat_table_w_weights[1..2].to_vec();
    unflattened_table[0] = flat_table[1..2].to_vec();
    for i in 1..lg_n {
        unflattened_table[i] = flat_table[(1 << i)..(1 << (i + 1))].to_vec();
        let mut level = flat_table_w_weights[(1 << i)..(1 << (i + 1))].to_vec();
        reverse_index_bits_in_place(&mut level);
        unflattened_table_w_weights[i] = level;
    }

    (unflattened_table_w_weights, unflattened_table)
}

pub fn query_root_table_from_rng_aes<E: ExtensionField>(
    level: usize,
    index: usize,
    cipher: &mut ctr::Ctr32LE<aes::Aes128>,
) -> E::BaseField {
    let mut level_offset: u128 = 1;
    for lg_m in 1..=level {
        let half_m = 1 << (lg_m - 1);
        level_offset += half_m;
    }

    let pos = ((level_offset + (reverse_bits(index, level) as u128))
        * (E::BaseField::bits().next_power_of_two() as u128))
        .checked_div(8)
        .unwrap();

    cipher.seek(pos);

    let bytes = (E::BaseField::bits()).next_power_of_two() / 8;
    let mut dest: Vec<u8> = vec![0u8; bytes];
    cipher.apply_keystream(&mut dest);

    base_from_raw_bytes::<E>(&dest)
}

#[cfg(test)]
mod tests {
    use crate::basefold::encoding::test_util::test_codeword_folding;

    use super::*;
    use ff_ext::GoldilocksExt2;
    use multilinear_extensions::mle::DenseMultilinearExtension;

    #[test]
    fn time_rs_code() {
        use rand::rngs::OsRng;

        let poly = DenseMultilinearExtension::random(20, &mut OsRng);

        encode_field_type_rs_basecode::<GoldilocksExt2>(&poly.evaluations, 2, 64);
    }

    #[test]
    fn prover_verifier_consistency() {
        type Code = Basecode<BasecodeDefaultSpec>;
        let pp: BasecodeParameters<GoldilocksExt2> = Code::setup(10);
        let (pp, vp) = Code::trim(pp, 10).unwrap();
        for level in 0..(10 + <Code as EncodingScheme<GoldilocksExt2>>::get_rate_log()) {
            for index in 0..(1 << level) {
                assert_eq!(
                    Code::prover_folding_coeffs(&pp, level, index),
                    Code::verifier_folding_coeffs(&vp, level, index),
                    "failed for level = {}, index = {}",
                    level,
                    index
                );
            }
        }
    }

    #[test]
    fn test_basecode_codeword_folding() {
        test_codeword_folding::<GoldilocksExt2, Basecode<BasecodeDefaultSpec>>();
    }
}
