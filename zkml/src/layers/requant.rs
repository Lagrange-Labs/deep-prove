use crate::{
    Claim, commit::same_poly, iop::verifier::Verifier,
    lookup::logup_gkr::verifier::verify_logup_proof, quantization,
};
use ff::Field;
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::ops::{Add, Mul, Sub};
use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::ContextAux,
    lookup::{context::TableType, logup_gkr::structs::LogUpProof},
    quantization::Fieldizer,
};

use super::LayerCtx;

/// Information about a requantization step:
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub struct Requant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    // this is the range we expect the values to be in pre shift
    pub range: usize,
    /// The range we want the values to be in post requantizing
    pub after_range: usize,
}

/// Info related to the lookup protocol necessary to requantize
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequantCtx {
    pub requant: Requant,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RequantProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from activation + claim from lookup for the same poly
    /// e.g. the "link" between an activation and requant layer
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the requantization
    pub(crate) lookup: LogUpProof<E>,
}
impl Requant {
    pub fn op(&self, input: &crate::tensor::Tensor<Element>) -> crate::tensor::Tensor<Element> {
        crate::tensor::Tensor::<Element>::new(
            input.dims(),
            input.get_data().iter().map(|e| self.apply(e)).collect_vec(),
        )
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.tables.insert(TableType::Range);
        (
            LayerCtx::Requant(RequantCtx {
                requant: *self,
                poly_id: id,
                num_vars: aux
                    .last_output_shape
                    .iter()
                    .map(|dim| ceil_log2(*dim))
                    .sum::<usize>(),
            }),
            aux,
        )
    }
    /// Applies requantization to a single element.
    ///
    /// This function performs the following steps:
    /// 1. Adds a large offset (max_bit) to ensure all values are positive
    /// 2. Right-shifts by the specified amount to reduce the bit width
    /// 3. Subtracts the shifted offset to restore the correct value range
    ///
    /// The result is a value that has been scaled down to fit within the
    /// target bit width while preserving the relative magnitudes.
    #[inline(always)]
    pub fn apply(&self, e: &Element) -> Element {
        let max_bit = (self.range << 1) as i128;
        let tmp = e + max_bit;
        let tmp = tmp >> self.right_shift;
        tmp - (max_bit >> self.right_shift)
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![1, self.range]
    }

    pub fn write_to_transcript<E: ExtensionField, T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from(self.right_shift as u64));
        t.append_field_element(&E::BaseField::from(self.range as u64));
    }

    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values --> shifted to the right !
    /// TODO: have a "cache" of lookups for similar ranges
    pub fn to_mle<E: ExtensionField>(&self) -> Vec<E> {
        // TODO: make a +1 or -1 somewhere
        let min_range = -(self.after_range as Element) / 2;
        let max_range = (self.after_range as Element) / 2 - 1;
        (min_range..=max_range)
            .map(|i| i.to_field())
            .collect::<Vec<E>>()
    }
    /// Function that takes a list of field elements that need to be requantized (i.e. the output of a Dense layer)
    /// and splits each value into the correct decomposition for proving via lookups.
    pub fn prep_for_requantize<E: ExtensionField>(
        &self,
        input: &[Element],
    ) -> Vec<Vec<E::BaseField>> {
        // We calculate how many chunks we will split each entry of `input` into.
        // Since outputs of a layer are centered around zero (i.e. some are negative) in order for all the shifting
        // and the like to give the correct result we make sure that everything is positive.

        // The number of bits that get "sliced off" is equal to `self.right_shift`, we want to know how many limbs it takes to represent
        // this sliced off chunk in base `self.after_range`. To calculate this we perform ceiling division on `self.right_shift` by
        // `ceil_log2(self.after_range)` and then add one for the column that represents the output we will take to the next layer.
        let num_columns = (self.right_shift - 1) / ceil_log2(self.after_range) + 2;

        let num_vars = ceil_log2(input.len());

        let mut mle_evals = vec![vec![E::BaseField::ZERO; 1 << num_vars]; num_columns];

        // Bit mask for the bytes
        let bit_mask = self.after_range as i128 - 1;

        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        input.iter().enumerate().for_each(|(index, val)| {
            let pre_shift = val + max_bit as i128;
            let tmp = pre_shift >> self.right_shift;
            let input = tmp - subtract as i128;
            let input_field: E = input.to_field();

            mle_evals[0][index] = input_field.as_bases()[0];
            // the value of an input should always be basefield elements

            // This leaves us with only the part that is "discarded"
            let mut remainder_vals = pre_shift - (tmp << self.right_shift);
            mle_evals
                .iter_mut()
                .skip(1)
                .rev()
                .for_each(|discarded_chunk| {
                    let chunk = remainder_vals & bit_mask;
                    let value = chunk as i128 - (self.after_range as i128 >> 1);
                    let field_elem: E = value.to_field();
                    discarded_chunk[index] = field_elem.as_bases()[0];
                    remainder_vals >>= self.after_range.ilog2();
                });
            debug_assert_eq!(remainder_vals, 0);
        });

        debug_assert!({
            input.iter().enumerate().fold(true, |acc, (i, value)| {
                let calc_evals = mle_evals
                    .iter()
                    .map(|col| E::from(col[i]))
                    .collect::<Vec<E>>();

                let field_value: E = value.to_field();
                acc & (self.recombine_claims(&calc_evals) == field_value)
            })
        });
        mle_evals
    }

    pub fn gen_lookup_witness<E: ExtensionField>(
        &self,
        input: &[Element],
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        // We calculate how many chunks we will split each entry of `input` into.
        // Since outputs of a layer are centered around zero (i.e. some are negative) in order for all the shifting
        // and the like to give the correct result we make sure that everything is positive.

        // The number of bits that get "sliced off" is equal to `self.right_shift`, we want to know how many limbs it takes to represent
        // this sliced off chunk in base `self.after_range`. To calculate this we perform ceiling division on `self.right_shift` by
        // `ceil_log2(self.after_range)` and then add one for the column that represents the output we will take to the next layer.
        let num_columns = (self.right_shift - 1) / ceil_log2(self.after_range) + 2;

        let num_vars = ceil_log2(input.len());

        let mut lookups = vec![vec![0i128; 1 << num_vars]; num_columns];
        let mut lookups_field = vec![vec![E::BaseField::ZERO; 1 << num_vars]; num_columns];
        // Bit mask for the bytes
        let bit_mask = self.after_range as i128 - 1;

        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        input.iter().enumerate().for_each(|(index, val)| {
            let pre_shift = val + max_bit as i128;
            let tmp = pre_shift >> self.right_shift;
            let input = tmp - subtract as i128 + (self.after_range as i128 >> 1);
            let in_field: E = input.to_field();

            lookups[0][index] = input;
            lookups_field[0][index] = in_field.as_bases()[0];
            // the value of an input should always be basefield elements

            // This leaves us with only the part that is "discarded"
            let mut remainder_vals = pre_shift - (tmp << self.right_shift);
            lookups
                .iter_mut()
                .zip(lookups_field.iter_mut())
                .skip(1)
                .rev()
                .for_each(|(discarded_lookup_chunk, discarded_field_chunk)| {
                    let chunk = remainder_vals & bit_mask;
                    let value = chunk as i128;
                    let val_field: E = value.to_field();
                    discarded_lookup_chunk[index] = value;
                    discarded_field_chunk[index] = val_field.as_bases()[0];
                    remainder_vals >>= self.after_range.ilog2();
                });
            debug_assert_eq!(remainder_vals, 0);
        });

        debug_assert!({
            input.iter().enumerate().fold(true, |acc, (i, value)| {
                let calc_evals = lookups_field
                    .iter()
                    .map(|col| E::from(col[i]))
                    .collect::<Vec<E>>();

                let field_value: E = value.to_field();
                acc & (self.recombine_claims(&calc_evals) == field_value)
            })
        });
        (lookups.concat(), lookups_field)
    }

    /// Function to recombine claims of constituent MLEs into a single value to be used as the initial sumcheck evaluation
    /// of the subsequent proof.
    pub fn recombine_claims<
        E: From<u64> + Default + Add<Output = E> + Mul<Output = E> + Sub<Output = E> + Copy,
    >(
        &self,
        eval_claims: &[E],
    ) -> E {
        let max_bit = self.range << 1;
        let subtract = max_bit >> self.right_shift;

        // There may be padding claims so we only take the first `num_columns` claims

        let tmp_eval = E::from(1 << self.right_shift as u64)
            * (eval_claims[0] + E::from(subtract as u64) - E::from(self.after_range as u64 >> 1))
            + eval_claims
                .iter()
                .skip(1)
                .rev()
                .enumerate()
                .fold(E::default(), |acc, (i, &claim)| {
                    acc + E::from((self.after_range.pow(i as u32)) as u64) * (claim)
                });
        tmp_eval - E::from(max_bit as u64)
    }
}

impl RequantCtx {
    pub(crate) fn verify_requant<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &RequantProof<E>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proof
        let num_instances =
            (self.requant.right_shift - 1) / ceil_log2(self.requant.after_range) + 2;
        let verifier_claims = verify_logup_proof(
            &proof.lookup,
            num_instances,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;

        // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
        let sp_ctx = same_poly::Context::<E>::new(self.num_vars);
        let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
        sp_verifier.add_claim(last_claim)?;

        let first_claim = verifier_claims
            .claims()
            .first()
            .ok_or(anyhow::anyhow!("No claims found"))?;
        let point = first_claim.point.clone();
        // The first claim needs to be shifted down as we add a value to make sure that all its evals are in the range 0..1 << BIT_LEn
        let corrected_claim = Claim::<E>::new(
            point.clone(),
            first_claim.eval - E::from(1 << (*quantization::BIT_LEN - 1)),
        );
        sp_verifier.add_claim(corrected_claim)?;

        let new_output_claim = sp_verifier.verify(&proof.io_accumulation, verifier.transcript)?;
        // 3. Accumulate the new claim into the witness commitment protocol
        verifier
            .witness_verifier
            .add_claim(self.poly_id, new_output_claim)?;

        // Here we recombine all of the none dummy polynomials to get the actual claim that should be passed to the next layer
        let eval_claims = verifier_claims
            .claims()
            .iter()
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();
        let eval = self.requant.recombine_claims(&eval_claims);
        // 4. return the input claim for to be proven at subsequent step
        Ok(Claim { point, eval })
    }
}
