use crate::{
    Claim, Prover,
    commit::same_poly,
    iop::verifier::Verifier,
    layers::LayerProof,
    lookup::logup_gkr::{prover::batch_prove as logup_batch_prove, verifier::verify_logup_proof},
    quantization,
};
use anyhow::anyhow;
use ff::Field;
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use itertools::Itertools;
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use statrs::statistics::{Data, Distribution};
use std::ops::{Add, Mul, Sub};
use tracing::warn;
use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::PolyID,
    iop::context::ContextAux,
    lookup::{context::TableType, logup_gkr::structs::LogUpProof},
    quantization::Fieldizer,
};

use super::LayerCtx;

enum RequantResult {
    Ok(Element),
    OutOfRange(Element),
}

/// Information about a requantization step:
/// * what is the range of the input data
/// * what should be the shift to get back data in range within QuantInteger range
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
pub struct Requant {
    // what is the shift that needs to be applied to requantize input number to the correct range of QuantInteger.
    pub right_shift: usize,
    // this is the range we expect the values to be in pre shift
    // This is a magnitude: e.g. [-4;8] gives range = 12.
    // This is to make sure to offset the values to be positive integers before doing the shift
    // That info is used to construct a lookup table for the requantization so the size of the lookup table
    // is directly correlated to the range of the input data.
    pub range: usize,
    /// The range we want the values to be in post requantizing
    pub after_range: usize,
    /// TEST ONLY: this can be given to simulate a perfect requantization during inference. Note that it CAN NOT
    /// be proven currently.
    pub multiplier: Option<f32>,
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
    pub fn new(min_value: usize, right_shift: usize) -> Self {
        Self {
            right_shift,
            range: min_value,
            after_range: *quantization::RANGE as usize,
            multiplier: None,
        }
    }
    pub fn set_test_multiplier(&mut self, multiplier: f32) {
        self.multiplier = Some(multiplier);
    }
    pub fn op(&self, input: &crate::tensor::Tensor<Element>) -> crate::tensor::Tensor<Element> {
        let mut not_ok_count = 0;
        let res = input
            .get_data()
            .iter()
            .map(|e| match self.apply(e) {
                RequantResult::Ok(res) => res,
                RequantResult::OutOfRange(res) => {
                    not_ok_count += 1;
                    res
                }
            })
            .collect_vec();
        let d = Data::new(res.iter().map(|e| *e as f64).collect_vec());
        // Debug information to uncomment when debugging scaling factor. Sometimes the right shift is too high
        // and we can observe values being null'd, e.g. set to 0 very quickly. Which messes up the distribution and
        // thus the inference.
        let stats = (d.mean().unwrap(), d.variance().unwrap());
        println!(
            "AFTER REQUANT: shift {} : {:.2} % OUT OF RANGE (over total {})-> stats mean {:?} var {:?} \n\t->{:?}\n\t->{:?}",
            self.right_shift,
            not_ok_count as f32 / res.len() as f32 * 100.0,
            res.len(),
            stats.0,
            stats.1,
            &input.get_data()[..10.min(input.get_data().len())],
            &res[..10.min(res.len())],
        );
        crate::tensor::Tensor::<Element>::new(input.get_shape(), res)
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux,
    ) -> Option<(LayerCtx<E>, ContextAux)>
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.tables.insert(TableType::Range);
        Some((
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
        ))
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
    fn apply(&self, e: &Element) -> RequantResult {
        if let Some(multiplier) = self.multiplier {
            //panic!("this is only for test - disable manually");
            let res = (*e as f64 * multiplier as f64).round() as Element;
            if !(res >= *quantization::MIN && res <= *quantization::MAX) {
                return RequantResult::OutOfRange(
                    res.clamp(*quantization::MIN, *quantization::MAX),
                );
            } else {
                return RequantResult::Ok(res);
            }
        }
        let max_bit = (self.range << 1) as Element;
        let tmp = e + max_bit;
        assert!(
            tmp >= 0,
            "offset is too small: element {} + {} (self.range << 1) = {}",
            e,
            self.range << 1,
            tmp
        );
        let tmp = tmp >> self.right_shift;
        let res = tmp - (max_bit >> self.right_shift);
        if !(res >= *quantization::MIN && res <= *quantization::MAX) {
            warn!("{} is NOT quantized correctly: res {}", e, res);
            // RequantResult::OutOfRange(res.clamp(*quantization::MIN, *quantization::MAX))
            RequantResult::OutOfRange(res)
        } else {
            // warn!("{} is OK quantized correctl: res {}", e, res);
            RequantResult::Ok(res)
        }
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
        let bit_mask = self.after_range.next_power_of_two() as i128 - 1;

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
                    remainder_vals >>= ceil_log2(self.after_range);
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
            + eval_claims.iter().skip(1).rev().enumerate().fold(
                E::default(),
                |acc, (i, &claim)| {
                    acc + E::from((self.after_range.next_power_of_two().pow(i as u32)) as u64)
                        * (claim)
                },
            );
        tmp_eval - E::from(max_bit as u64)
    }
    pub(crate) fn prove_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: &Claim<E>,
        output: &[E],
        requant_info: &RequantCtx,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let prover_info = prover.next_lookup_witness()?;

        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;
        // For requant layers we have to extract the correct "chunk" from the list of claims
        let eval_claims = logup_proof
            .output_claims()
            .iter()
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();

        let combined_eval = requant_info.requant.recombine_claims(&eval_claims);

        // Pass the eval associated with the poly used in the activation step to the same poly prover
        let first_claim = logup_proof
            .output_claims()
            .first()
            .ok_or(anyhow!("No claims found"))?;
        let point = first_claim.point.clone();

        let corrected_claim = Claim::<E> {
            point: point.clone(),
            eval: first_claim.eval - E::from((*quantization::RANGE / 2) as u64),
        };
        println!("correct claim eval: {:?}", corrected_claim.eval);
        println!(
            "output eval: {:?}",
            output.to_vec().into_mle().evaluate(&corrected_claim.point)
        );
        // Add the claim used in the activation function
        same_poly_prover.add_claim(corrected_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;

        prover
            .witness_prover
            .add_claim(requant_info.poly_id, claim_acc_proof.extract_claim())?;

        prover.push_proof(LayerProof::Requant(RequantProof {
            io_accumulation: claim_acc_proof,
            lookup: logup_proof,
        }));

        Ok(Claim {
            point,
            eval: combined_eval,
        })
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
            first_claim.eval - E::from((*quantization::RANGE / 2) as u64),
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

//#[cfg(test)]
// mod tests {
//    use ark_std::rand::rngs::StdRng;
//
//    use super::*;
//    use crate::quantization::range_from_weight;
//    use crate::tensor::Tensor;
//    use crate::ScalingFactor;
//
//    #[test]
//    fn test_requant_shift_element() {
//        let n = 10;
//        let v1 = Tensor::random_seed(vec![n],Some(15420));
//        let v2= Tensor::random_seed(vec![n],Some(1567892312));
//        assert!(v1.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        assert!(v2.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        let res = v1.mul(&v2);
//        let s1 = ScalingFactor::from_tensor(&v1);
//        let s2 = ScalingFactor::from_tensor(&v2);
//        let s_res = ScalingFactor::from_tensor(&res);
//        println!("v1: {:?}", v1.get_data());
//        println!("v2: {:?}", v2.get_data());
//        println!("res: {:?}", res.get_data());
//        println!("s1: {:?}", s1);
//        println!("s2: {:?}", s2);
//        println!("s_res: {:?}", s_res);
//        let shift = s1.shift(s2, s_res);
//        println!("shift: {:?}", shift);
//        let res_max = res.get_data().iter().max().unwrap();
//        let res_min = res.get_data().iter().min().unwrap();
//        let requant_info = Requant {
//            right_shift:  shift,
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requant = requant_info.op(&res);
//        println!("res_requant: {:?}", res_requant.get_data());
//    }
//
//    use ark_std::rand::SeedableRng;
//    use ark_std::rand::Rng;
//    #[test]
//    fn test_requant_shift_model_like() {
//        let n = 10;
//        let mut rng = StdRng::seed_from_u64(15420);
//        let input_min = -1.0;
//        let input_max = 1.0;
//        println!("1");
//        let s_input = ScalingFactor::from_span(input_min, input_max);
//        let inputf :Vec<f32> = (0..n).map(|_| { rng.gen_range(input_min..=input_max) }).collect_vec();
//        let input: Vec<Element> = inputf.iter().map(|e| s_input.quantize(&e)).collect_vec();
//        let min_f32 = -0.2;
//        let max_f32 = 0.2;
//        println!("2");
//        let s_model = ScalingFactor::from_span(min_f32, max_f32);
//        println!("3");
//        let s_input = ScalingFactor::from_span(input_min, input_max);
//        println!("4");
//        let modelf :Vec<f32> = (0..n).map(|_| { rng.gen_range(min_f32..=max_f32) }).collect_vec();
//        let model :Vec<Element> = modelf.iter().map(|e| s_model.quantize(&e)).collect_vec();
//
//        let inputf = Tensor::new(vec![n], inputf);
//        let modelf  = Tensor::new(vec![n], modelf);
//        println!("5");
//        let resf = inputf.mul(&modelf);
//        println!("6");
//        let s_resf = ScalingFactor::from_tensor(&resf);
//        let s_resft = ScalingFactor::new(resf.get_data().iter().map(|e| e.abs()).fold(0.0f32,|a,b| a.max(b)));
//        println!("7");
//        let input = Tensor::new(vec![n], input);
//        let model= Tensor::new(vec![n], model);
//        assert!(input.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        assert!(model.get_data().iter().all(|e| *e >= *quantization::MIN && *e <= *quantization::MAX));
//        let (mins,maxs) : (Vec<_>,Vec<_>)= model.get_data().iter().map(|e| range_from_weight(e)).unzip();
//        let res_min = mins.iter().min().unwrap();
//        let res_max = maxs.iter().max().unwrap();
//        let s_res = ScalingFactor::from_span(*res_min as f32, *res_max as f32);
//        let res = input.mul(&model);
//        println!("input: {:?}", input.get_data());
//        println!("model: {:?}", model.get_data());
//        println!("res: {:?}", res.get_data());
//        println!("s1: {:?}", s_input);
//        println!("s2: {:?}", s_model);
//        println!("s_resf: {:?}", s_resf);
//        println!("s_res: {:?}", s_res);
//        let shift = s_input.shift(s_model, s_res);
//        let shiftf= s_input.shift(s_model, s_resf);
//        let shiftft = s_input.shift(s_model, s_resft);
//        println!("shift: {:?}", shift);
//        println!("shiftf: {:?}", shiftf);
//        println!("shiftft: {:?}", shiftft);
//        let requant = Requant {
//            right_shift:  shift,
//            // theoretical res_max and res_min at this point ! since we dont know the input when we create requant
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requant = requant.op(&res);
//        let requant = Requant {
//            right_shift:  shiftf,
//            // theoretical res_max and res_min at this point ! since we dont know the input when we create requant
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requantf = requant.op(&res);
//        let requant = Requant {
//            right_shift:  shiftft,
//            // theoretical res_max and res_min at this point ! since we dont know the input when we create requant
//            range: (res_max - res_min) as usize,
//            after_range: 1 << *quantization::BIT_LEN,
//        };
//        let res_requantft= requant.op(&res);
//        println!("res_requant: {:?}", res_requant.get_data());
//        println!("res_requantf: {:?}", res_requantf.get_data());
//        println!("res_requantft: {:?}", res_requantft.get_data());
//        //assert!(res_requant.get_data().iter().filter(|r| **r == 0 || **r == -1).collect::<Vec<_>>().len() < res_requant.get_data().len());
//    }
//
//}
