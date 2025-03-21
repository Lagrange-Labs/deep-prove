use super::{
    ChallengeStorage, Context, Proof, RequantProof, StepProof, TableProof,
    context::{ConvInfo, DenseInfo, PoolingInfo, StepInfo},
};
use crate::{
    Claim, Element, VectorTranscript,
    activation::Activation,
    commit::{compute_betas_eval, precommit, same_poly},
    convolution::Convolution,
    dense,
    iop::{ActivationProof, ConvProof, DenseProof, PoolingProof},
    lookup::{
        context::generate_lookup_witnesses,
        logup_gkr::{prover::batch_prove, structs::LogUpInput},
    },
    model::{InferenceStep, InferenceTrace, Layer},
    tensor::{ConvData, Tensor, get_root_of_unity},
};
use anyhow::{Context as CC, anyhow, bail};
use ff_ext::ExtensionField;

use gkr::util::ceil_log2;
use itertools::izip;
use multilinear_extensions::{
    mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
};
use serde::{Serialize, de::DeserializeOwned};

use sumcheck::structs::{IOPProverState, IOPVerifierState};
use timed::timed_instrument;
use tracing::{debug, instrument, trace, warn};
use transcript::Transcript;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx: &'a Context<E>,
    // proofs for each layer being filled
    proofs: Vec<StepProof<E>>,
    table_proofs: Vec<TableProof<E>>,
    transcript: &'a mut T,
    commit_prover: precommit::CommitProver<E>,
    /// the context of the witness part (IO of lookups, linked with matrix2vec for example)
    /// is generated during proving time. It is first generated and then the fiat shamir starts.
    /// The verifier doesn't know about the individual polys (otherwise it beats the purpose) so
    /// that's why it is generated at proof time.
    witness_ctx: Option<precommit::Context<E>>,
    /// The prover related to proving multiple claims about different witness polyy (io of lookups etc)
    witness_prover: precommit::CommitProver<E>,
    /// The lookup witnesses
    lookup_witness: Vec<LogUpInput<E>>,
    /// The Lookup table witness
    table_witness: Vec<LogUpInput<E>>,
    /// Stores all the challenges for the different lookup/table types
    challenge_storage: ChallengeStorage<E>,
}

impl<'a, E, T> Prover<'a, E, T>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new(ctx: &'a Context<E>, transcript: &'a mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            table_proofs: Vec::default(),
            commit_prover: precommit::CommitProver::new(),
            // at this step, we can't build the ctx since we don't know the individual polys
            witness_ctx: None,
            witness_prover: precommit::CommitProver::new(),
            lookup_witness: Vec::default(),
            table_witness: Vec::default(),
            challenge_storage: ChallengeStorage::default(),
        }
    }
    //#[instrument(name="prove step",skip_all,fields(step = step.layer.describe()),level = "debug")]
    fn prove_step<'b>(
        &mut self,
        last_claim: Claim<E>,
        input: &Tensor<E>,
        step: &InferenceStep<'b, E, E>,
        info: &StepInfo<E>,
    ) -> anyhow::Result<Claim<E>> {
        debug!("PROVER: proving layer {}", step.layer.to_string());
        let claim = match (step.layer, info) {
            (Layer::Dense(dense), StepInfo::Dense(info)) => {
                // NOTE: here we treat the ID of the step AS the ID of the polynomial. THat's okay because we only care
                // about these IDs being unique, so as long as the mapping between poly <-> id is correct, all good.
                // This is the case here since we treat each matrix as a different poly
                self.prove_dense_step(last_claim, input, &step.output, info, dense)
            }
            (Layer::Convolution(filter), StepInfo::Convolution(info)) => {
                self.prove_convolution_step(last_claim, &step.output, &step.conv_data, info, filter)
            }
            (Layer::Activation(Activation::Relu(..)), StepInfo::Activation(..))
            | (Layer::Requant(..), StepInfo::Requant(..)) => {
                self.prove_lookup(&last_claim, &step.output.get_data(), info)
            }
            (Layer::Pooling(..), StepInfo::Pooling(info)) => {
                self.prove_pooling(last_claim, input, &step.output, info)
            }
            _ => bail!(
                "inconsistent proof step {} and info step {} from ctx",
                step.layer.describe(),
                info.variant_name()
            ),
        };

        claim
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_lookup(
        &mut self,
        last_claim: &Claim<E>,
        output: &[E],
        step: &StepInfo<E>,
    ) -> anyhow::Result<Claim<E>> {
        // First we check that the step requires lookup
        if !step.requires_lookup() {
            return Err(anyhow!(
                "A step of type: {} does not require a lookup proof",
                step.variant_name()
            ));
        }
        let prover_info = self
            .lookup_witness
            .pop()
            .ok_or(anyhow!("No more lookup witness!"))?;

        // Run the lookup protocol and return the lookup proof
        let logup_proof = batch_prove(&prover_info, self.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;

        match step {
            StepInfo::Activation(info) => {
                // Activation proofs have two columns, input and output
                let input_claim = logup_proof.output_claims()[0].clone();
                let output_claim = logup_proof.output_claims()[1].clone();

                same_poly_prover.add_claim(output_claim)?;
                let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;
                // order is (output,mult)
                self.witness_prover
                    .add_claim(info.poly_id, claim_acc_proof.extract_claim())?;

                // Add the proof in
                self.proofs.push(StepProof::Activation(ActivationProof {
                    io_accumulation: claim_acc_proof,
                    lookup: logup_proof,
                }));
                Ok(input_claim)
            }
            StepInfo::Requant(requant_info) => {
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

                // Add the claim used in the activation function
                same_poly_prover.add_claim(first_claim.clone())?;
                let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;

                self.witness_prover
                    .add_claim(requant_info.poly_id, claim_acc_proof.extract_claim())?;

                self.proofs.push(StepProof::Requant(RequantProof {
                    io_accumulation: claim_acc_proof,
                    lookup: logup_proof,
                }));

                Ok(Claim {
                    point,
                    eval: combined_eval,
                })
            }
            _ => Err(anyhow!(
                "Should not be in prove_lookup function for step: {}",
                step.variant_name()
            )),
        }
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_tables(&mut self) -> anyhow::Result<()> {
        let mut poly_id = self.ctx.steps_info.len();

        self.table_witness
            .iter()
            .zip(self.ctx.lookup.iter())
            .try_for_each(|(table_witness, table_type)| {
                println!("PROVING table of type: {}", table_type.name());

                // Make the proof for the table
                let table_proof = batch_prove(&table_witness, self.transcript)?;

                // Add the multiplicity poly claim
                self.witness_prover.add_claim(
                    poly_id,
                    table_proof.output_claims().first().unwrap().clone(),
                )?;

                self.table_proofs.push(TableProof {
                    lookup: table_proof,
                });

                poly_id += 1;
                Ok(())
            })
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_pooling(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &Tensor<E>,
        // output of dense layer evaluation
        output: &Tensor<E>,
        info: &PoolingInfo,
    ) -> anyhow::Result<Claim<E>> {
        assert_eq!(input.dims().len(), 3, "Maxpool needs 3D inputs.");
        // Create the range check proof for the diff
        let prover_info = self
            .lookup_witness
            .pop()
            .ok_or(anyhow!("No more lookup witness!"))?;

        let logup_proof = batch_prove(&prover_info, self.transcript)?;

        // These are the polys that get passed to the zero check make sure their product is zero at every evaluation point
        let mut diff_polys = prover_info
            .column_evals()
            .iter()
            .map(|diff| {
                DenseMultilinearExtension::<E>::from_evaluations_slice(info.num_vars, diff).into()
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        // Run the Zerocheck that checks enforces that output does contain the maximum value for the kernel
        let mut vp = VirtualPolynomial::<E>::new(info.num_vars);

        // We reuse the logup point here for the zerocheck challenge
        let lookup_point = &logup_proof.output_claims()[0].point;

        // Comput the identity poly
        let batch_challenge = self
            .transcript
            .get_and_append_challenge(b"batch_pooling")
            .elements;

        let beta_eval = compute_betas_eval(&lookup_point);
        let beta_poly: ArcDenseMultilinearExtension<E> =
            DenseMultilinearExtension::<E>::from_evaluations_ext_vec(info.num_vars, beta_eval)
                .into();
        let mut challenge_combiner = batch_challenge;
        let lookup_parts = diff_polys
            .iter()
            .map(|diff| {
                let out = (vec![diff.clone(), beta_poly.clone()], challenge_combiner);
                challenge_combiner *= batch_challenge;
                out
            })
            .collect::<Vec<(Vec<_>, E)>>();

        diff_polys.push(beta_poly);
        vp.add_mle_list(diff_polys, E::ONE);
        lookup_parts
            .into_iter()
            .for_each(|(prod, coeff)| vp.add_mle_list(prod, coeff));

        #[allow(deprecated)]
        let (proof, sumcheck_state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        // We need to prove that the output of this step is the input to following activation function
        let output_mle = output.get_data().to_vec().into_mle();
        let mut same_poly_prover = same_poly::Prover::<E>::new(output_mle.clone());

        let zerocheck_point = &proof.point;
        let output_zerocheck_eval = output_mle.evaluate(zerocheck_point);

        // Accumulate claims about the output polynomial in each of the protocols we ran together with the final claim from the previous proof.
        let mut output_claims = Vec::<Claim<E>>::new();
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;

        let zerocheck_claim = Claim {
            point: zerocheck_point.clone(),
            eval: output_zerocheck_eval,
        };
        same_poly_prover.add_claim(zerocheck_claim.clone())?;
        output_claims.push(zerocheck_claim);

        // This is the proof for the output poly
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, self.transcript)?;

        let output_claim = claim_acc_proof.extract_claim();

        self.witness_prover
            .add_claim(info.poly_id, output_claim)
            .context("unable to add claim")?;
        // Now we must do the samething accumulating evals for the input poly as we fix variables on the input poly.
        // The point length is 2 longer because for now we only support MaxPool2D.

        let padded_input_shape = input.dims();
        let padded_input_row_length_log = ceil_log2(padded_input_shape[2]);
        // We can batch all of the claims for the input poly with 00, 10, 01, 11 fixed into one with random challenges
        let [r1, r2] = [self
            .transcript
            .get_and_append_challenge(b"input_batching")
            .elements; 2];

        let one_minus_r1 = E::ONE - r1;
        let one_minus_r2 = E::ONE - r2;
        // To the input claims we add evaluations at both the zerocheck point and lookup point
        // in the order 00, 01, 10, 11. These will be used in conjunction with r1 and r2 by the verifier to link the claims output by the sumcheck and lookup GKR
        // proofs with the claims fed to the same poly verifier.

        let multiplicands = [
            one_minus_r1 * one_minus_r2,
            one_minus_r1 * r2,
            r1 * one_minus_r2,
            r1 * r2,
        ];

        let zc_in_claim = izip!(
            multiplicands.iter(),
            sumcheck_state.get_mle_final_evaluations().iter(),
        )
        .fold(E::ZERO, |zc_acc, (m, zc)| {
            zc_acc + *m * (output_zerocheck_eval - *zc)
        });

        let point_1 = [
            &[r1],
            &zerocheck_point[..padded_input_row_length_log - 1],
            &[r2],
            &zerocheck_point[padded_input_row_length_log - 1..],
        ]
        .concat();

        let next_claim = Claim {
            point: point_1,
            eval: zc_in_claim,
        };

        // We don't need the last eval of the the sumcheck state as it is the beta poly

        let zerocheck_evals = sumcheck_state.get_mle_final_evaluations()[..4].to_vec();
        // Push the step proof to the list
        self.proofs.push(StepProof::Pooling(PoolingProof {
            sumcheck: proof,
            lookup: logup_proof,
            io_accumulation: claim_acc_proof,
            output_claims,
            zerocheck_evals,
            variable_gap: padded_input_row_length_log - 1,
        }));
        Ok(next_claim)
    }

    // Protocol for proving the correct computation of the FFT/iFFT matrix.
    // For more details look at the zkCNN paper.
    // F_middle : all intermidiate evaluations retrieved by the phiGinit algorithm
    // r1: the initial random point used to reduce the matrix into vector
    // r2: the random point produced by the sumcheck
    pub fn delegate_matrix_evaluation(
        &mut self,
        f_middle: &mut Vec<Vec<E>>,
        r1: Vec<E>,
        mut r2: Vec<E>,
        is_fft: bool,
    ) -> (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>) {
        let mut omegas = vec![E::ZERO; 1 << r1.len() as usize];
        self.phi_pow_init(&mut omegas, r1.len(), is_fft);

        let mut proofs: Vec<sumcheck::structs::IOPProof<E>> = Vec::new();
        let mut claims: Vec<Vec<E>> = Vec::new();

        for l in (0..(r1.len() - 1)).rev() {
            let mut phi = vec![E::ZERO; f_middle[l].len()];
            let beta = compute_betas_eval(&r2[0..(r2.len() - 1)]);

            for i in 0..(phi.len()) {
                if !is_fft && l == f_middle.len() - 1 {
                    phi[i] = (E::ONE - r2[r2.len() - 1])
                        * (E::ONE - r1[(f_middle.len() - 1) - l]
                            + r1[(f_middle.len() - 1) - l]
                                * omegas[i << ((f_middle.len() - 1) - l)]);
                } else {
                    phi[i] = E::ONE - r1[(f_middle.len() - 1) - l]
                        + (E::ONE - E::from(2) * r2[r2.len() - 1])
                            * r1[(f_middle.len() - 1) - l]
                            * omegas[i << ((f_middle.len() - 1) - l)];
                }
            }

            let f1 = beta.into_mle();
            let f2 = phi.into_mle();
            let f3 = f_middle[l].clone().into_mle();

            let mut vp = VirtualPolynomial::<E>::new(f1.num_vars);
            vp.add_mle_list(
                vec![f1.clone().into(), f2.clone().into(), f3.clone().into()],
                E::ONE,
            );
            #[allow(deprecated)]
            let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);
            let claim: Vec<E> = state.get_mle_final_evaluations();
            r2 = proof.point.clone();
            proofs.push(proof);
            claims.push(claim);
        }
        (proofs, claims)
    }

    // Compute powers of roots of unity
    pub fn phi_pow_init(&mut self, phi_mul: &mut Vec<E>, n: usize, is_fft: bool) {
        let length = 1 << n;
        let rou: E = get_root_of_unity(n);

        let mut phi = rou;
        if is_fft {
            phi = phi.invert().unwrap();
        }
        phi_mul[0] = E::ONE;
        for i in 1..length {
            phi_mul[i] = phi_mul[i - 1] * phi;
        }
    }

    // Efficiently compute the omegas of FFT/iFFT matrix reduced at rx
    // This is a copy-paste implementation from zkCNN paper
    pub fn phi_g_init(
        &mut self,
        phi_g: &mut Vec<E>,
        mid_phi_g: &mut Vec<Vec<E>>,
        rx: Vec<E>,
        scale: E,
        n: usize,
        is_fft: bool,
    ) {
        let mut phi_mul = vec![E::ZERO; 1 << n];
        self.phi_pow_init(&mut phi_mul, n, is_fft);
        if is_fft {
            phi_g[0] = scale;
            phi_g[1] = scale;
            for i in 1..(n + 1) {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;
                    let tmp1 = E::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];
                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] = phi_g[l] * (tmp1 + tmp2);
                }
                if i < n {
                    mid_phi_g[i - 1] = vec![E::ZERO; 1 << (i) as usize];
                    for b in 0..(1 << (i)) {
                        mid_phi_g[i - 1][b] = phi_g[b];
                    }
                }
            }
        } else {
            phi_g[0] = scale;
            for i in 1..n {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;

                    let tmp1 = E::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];
                    // printf("%d,%d\n",r,l );
                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] = phi_g[l] * (tmp1 + tmp2);
                }
                mid_phi_g[i - 1] = vec![E::ZERO; 1 << (i) as usize];
                for b in 0..(1 << (i)) {
                    mid_phi_g[i - 1][b] = phi_g[b];
                }
            }
            for b in 0..(1 << (n - 1)) {
                let l = b;
                let tmp1 = E::ONE - rx[0];
                let tmp2 = rx[0] * phi_mul[b];
                phi_g[l] = phi_g[l] * (tmp1 + tmp2);
            }
        }
    }
    // The prove_batch_fft and prove_batch_ifft are extensions of prove_fft and prove_ifft but in the batch setting.
    // Namely when we want to proof fft or ifft for MORE THAN ONE INSTANCES.
    // In particular, instead of proving y = Wx we want to prove Y = WX where Y,X are matrixes.
    // Following the matrix to matrix multiplication protocol, let y_eval = Y(r1,r2).
    // Then we want to prove a sumcheck instance of the form y_eval = sum_{i \in [n]}W(r1,i)X(i,r2).
    pub fn prove_batch_fft(
        &mut self,
        r: Vec<E>,
        x: &mut Vec<Vec<E>>,
    ) -> (
        sumcheck::structs::IOPProof<E>,
        Vec<E>,
        (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
    ) {
        let padded_rows = 2 * x[0].len();
        for i in 0..x.len() {
            x[i].resize(padded_rows, E::ZERO);
        }
        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; x[0].len().ilog2() as usize];
        let mut r2 = vec![E::ZERO; x.len().ilog2() as usize];
        for i in 0..r1.len() {
            r1[i] = r[i];
        }

        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; x[0].len()];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        self.phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            E::from(1),
            x[0].len().ilog2() as usize,
            false,
        );
        // compute X(i,r2)

        let mut f_m = x
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into_mle();

        f_m.fix_high_variables_in_place(&r2);

        // Construct the virtual polynomial and run the sumcheck prover

        let f_red = w_red.into_mle();

        let mut vp = VirtualPolynomial::<E>::new(f_m.num_vars);
        vp.add_mle_list(vec![f_m.clone().into(), f_red.clone().into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        let claims = state.get_mle_final_evaluations();
        let out_point = proof.point.clone();
        (
            proof,
            claims,
            self.delegate_matrix_evaluation(&mut f_middle, r1.clone(), out_point, false),
        )
    }

    pub fn prove_batch_ifft(
        &mut self,
        r: Vec<E>,
        prod: &Vec<Vec<E>>,
    ) -> (
        sumcheck::structs::IOPProof<E>,
        Vec<E>,
        (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
    ) {
        let scale: E = E::from(prod[0].len() as u64).invert().unwrap();

        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; prod[0].len().ilog2() as usize];
        let mut r2 = vec![E::ZERO; prod.len().ilog2() as usize];
        for i in 0..r1.len() {
            r1[i] = r[i];
        }
        assert_eq!(
            r1[r1.len() - 1],
            E::ZERO,
            "Error in randomness init batch ifft"
        );
        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; prod[0].len()];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        self.phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            scale,
            prod[0].len().ilog2() as usize,
            true,
        );
        let f_red = w_red.into_mle();
        // compute X(i,r2)
        let mut f_m = prod
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into_mle();
        f_m.fix_high_variables_in_place(&r2);

        // Construct the virtual polynomial and run the sumcheck prover
        let mut vp = VirtualPolynomial::<E>::new(f_m.num_vars);
        vp.add_mle_list(vec![f_m.clone().into(), f_red.clone().into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        let claims = state.get_mle_final_evaluations();

        let out_point = proof.point.clone();
        (
            proof,
            claims,
            self.delegate_matrix_evaluation(&mut f_middle, r1.clone(), out_point, true),
        )
        // return Proof;
    }

    // Prove convolution of a CNN network. This is a convolution between in a 3D matrix X of dimension k_x * n_x * n_x
    // and a 4D filter matrix W of dimension k_w * k_x * n_w * n_w. The output is a 3D matrix Y of dimension k_w * n_x * n_x
    // We want to batch prove the following: Y[i] = iFFT(sum_{j \in [n_x]}(FFT(X[j]) o FFT(W[i][j])).

    #[instrument(name = "Prover::prove_convolution_step", skip_all, level = "debug")]
    #[timed::timed_instrument(level = "debug")]
    fn prove_convolution_step(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // Struct containing all necessary information
        // to generate a convolution proof
        _output: &Tensor<E>,
        proving_data: &ConvData<E>,
        info: &ConvInfo<E>,
        filter: &Convolution,
    ) -> anyhow::Result<Claim<E>> {
        assert_eq!(
            filter.filter_size() * filter.kw() * 2,
            proving_data.output.len() * proving_data.output[0].len(),
            "Inconsistent output size"
        );
        assert_eq!(
            (filter.filter_size() * filter.kw()).ilog2() as usize,
            last_claim.point.len(),
            "Inconsistent random point size. Expected : {}, got: {}",
            ((filter.filter_size() * filter.kw()).ilog2()),
            last_claim.point.len()
        );
        let mut r = vec![E::ZERO; last_claim.point.len() + 1];
        let mut bias_point = vec![E::ZERO; filter.kw().ilog2() as usize];
        for i in 0..(filter.filter_size().ilog2() as usize) {
            r[i] = E::ONE - last_claim.point[i];
        }
        for i in 0..(filter.kw().ilog2() as usize) {
            r[i + (filter.filter_size().ilog2() as usize) + 1] =
                last_claim.point[i + (filter.filter_size().ilog2() as usize)];
            bias_point[i] = last_claim.point[i + (filter.filter_size().ilog2() as usize)];
        }
        let mut bias_eval = E::ZERO;
        if bias_point.len() != 0 {
            bias_eval = filter
                .bias
                .evals_flat::<E>()
                .into_mle()
                .evaluate(&bias_point);
        } else if filter.bias.data.len() == 1 {
            bias_eval = filter.bias.evals_flat::<E>()[0];
        }

        debug_assert!({
            let y = proving_data
                .output
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_mle()
                .evaluate(&r);
            debug_assert_eq!(last_claim.eval - bias_eval, y, "Error in Conv 1");
            last_claim.eval - bias_eval == y
        });

        let mut temp_t = self.transcript.clone();
        let (ifft_proof, ifft_claim, ifft_del_proof) =
            self.prove_batch_ifft(r.clone(), &proving_data.prod);

        assert_eq!(
            filter.filter_size().ilog2() as usize + 1,
            ifft_proof.point.len(),
            "Error in ifft sumceck"
        );
        debug_assert!({
            IOPVerifierState::<E>::verify(
                last_claim.eval - bias_eval,
                &ifft_proof.clone(),
                &info.ifft_aux.clone(),
                &mut temp_t,
            );
            println!("iFFT Sumcheck Correct");
            1 == 1
        });

        // After this point, the verifier holds an evaluation claim of proving_data.prod at P1.randomness[0][i]
        // Let r' = P1.randomness[0][i] and y is the evaluation claim of prod = proving_data.prod
        // What we want to do now is to prove that prod has been correctly computed from X_fft and w (= proving_data.w)
        // In other words we want to show that prod[i] = sum_{j \in [k_x]} x[j] o w[i][j] for each i in [k_w]
        // For this let r1 be the last log(k_w) elements of r and r2 the first log(n_x^2) elements
        // Compute the arrays beta1,beta2 such that beta1[i] = beta(i,r1) and beta2[i] = beta(i,r2)

        let mut r_ifft: Vec<E> = ifft_proof.point.clone();
        for i in (proving_data.output[0].len().ilog2() as usize)..r.len() {
            r_ifft.push(r[i]);
        }

        debug_assert!({
            let eval1 = proving_data
                .prod
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_mle()
                .evaluate(&r_ifft);
            let eval2 = ifft_claim[0];
            debug_assert_eq!(
                proving_data
                    .prod
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into_mle()
                    .evaluate(&r_ifft),
                ifft_claim[0],
                "Error in Conv 1"
            );
            eval1 == eval2
        });

        let r1 = &r_ifft[(proving_data.output[0].len().ilog2() as usize)..];
        let r2 = &r_ifft[..(proving_data.output[0].len().ilog2() as usize)];

        let beta2 = compute_betas_eval(r2);
        // Given beta1,beta2 observe that :
        // \sum_{i \in [k_w]} beta1[i]prod[i] = \sum_{i \in [k_w]}sum_{j \in [k_x]} x[j] o w[i][j] =
        // = sum_{j \in [k_x]}x[j]o(\sum_{i \in [k_w]}(beta[i]*w[i][j])). We let w_reduced[j] = \sum_{i \in [k_w]}(beta[i]*w[i][j])
        // We have  \sum_{i \in [k_w]} beta1[i]prod[i] = sum_{j \in [k_x]} x[j]o w_{reduced[j]}.
        // So here we compute w_reduced

        let beta_acc = vec![beta2.clone(); filter.kx()].concat();

        // After computing w_reduced, observe that y = \sum_{k \in [n_x^2]} sum_{j \in [k_x]} beta2[k]*x[j][k]*w_reduced[j][k]
        // This is a cubic sumcheck where v1 = [x[0][0],...,x[k_x][n_x^2]], v2 = [w_reduced[0][0],...,w_reduced[k_x][n_x^2]]
        // and v3 = [beta2,..(k_x times)..,beta2]. So, first initialzie v3 and then invoke the cubic sumceck.

        // We need to fix the high variables in place for the filter at r1.
        let f1 = filter
            .filter
            .evals_flat::<E>()
            .into_mle()
            .fix_high_variables(r1);

        let f2 = proving_data
            .input_fft
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .into_mle();
        let f3 = beta_acc.into_mle();

        let mut vp = VirtualPolynomial::<E>::new(f1.num_vars);
        vp.add_mle_list(
            vec![f1.clone().into(), f2.clone().into(), f3.clone().into()],
            E::ONE,
        );
        #[allow(deprecated)]
        let (hadamard_proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);
        let hadamard_claims = state.get_mle_final_evaluations();

        let point = [hadamard_proof.point.as_slice(), r1].concat();
        // let eval = hadamard_claims[0];
        self.commit_prover
            .add_claim(info.poly_id, Claim::new(point, hadamard_claims[0]))
            .context("unable to add convolution claim")?;
        self.commit_prover
            .add_claim(info.bias_poly_id, Claim::new(bias_point, bias_eval))
            .context("unable to add bias claim in convolution")?;

        // Finally prove the correct computation of the x_fft and get an evaluation claim of the input
        let (fft_proof, fft_claim, fft_del_proof) = self.prove_batch_fft(
            hadamard_proof.point.clone(),
            &mut proving_data.input.clone(),
        );

        self.proofs.push(StepProof::Convolution(ConvProof {
            fft_proof: fft_proof.clone(),
            fft_claims: fft_claim.clone(),
            ifft_proof,
            fft_delegation_proof: fft_del_proof.0,
            ifft_delegation_proof: ifft_del_proof.0,
            hadamard_proof: hadamard_proof.clone(),
            ifft_claims: ifft_claim,
            fft_delegation_claims: fft_del_proof.1,
            ifft_delegation_claims: ifft_del_proof.1,
            hadamard_clams: hadamard_claims,
            bias_claim: bias_eval,
        }));
        let mut input_point = fft_proof.point.clone();
        let mut v = input_point.pop().unwrap();
        v = (E::ONE - v).invert().unwrap();
        debug_assert!({
            let mut p = [
                input_point.clone(),
                hadamard_proof.point[((filter.filter_size() * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat();
            // println!("({},{}), {}",proving_data.input.len(),proving_data.input[0].len(),p.len());
            let y = proving_data
                .input
                .clone()
                .into_iter()
                .flat_map(|v| v.into_iter())
                .collect::<Vec<E>>()
                .into_mle()
                .evaluate(&p);
            assert_eq!(y, fft_claim[0] * v, "Error in input eval CONV PROVER");
            for i in 0..((filter.filter_size().ilog2()) as usize) {
                p[i] = E::ONE - p[i];
            }
            assert_eq!(
                proving_data.real_input.clone().into_mle().evaluate(&p),
                fft_claim[0] * v,
                "Error in real input eval CONV PROVER"
            );
            proving_data.real_input.clone().into_mle().evaluate(&p) == fft_claim[0] * v
        });
        for i in 0..input_point.len() {
            input_point[i] = E::ONE - input_point[i];
        }
        let final_claim = Claim {
            point: [
                input_point.clone(),
                hadamard_proof.point[((filter.filter_size() * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            eval: fft_claim[0] * v,
        };

        Ok(final_claim)
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_dense_step(
        &mut self,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &Tensor<E>,
        // output of dense layer evaluation
        output: &Tensor<E>,
        info: &DenseInfo<E>,
        dense: &dense::Dense,
    ) -> anyhow::Result<Claim<E>> {
        let matrix = &dense.matrix;
        let (nrows, ncols) = (matrix.nrows_2d(), matrix.ncols_2d());
        debug!("dense proving nrows: {} ncols: {}", nrows, ncols);
        assert_eq!(
            nrows,
            output.get_data().len(),
            "dense proving: nrows {} vs output {}",
            nrows,
            output.get_data().len()
        );
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(
            ncols,
            input.get_data().len(),
            "something's wrong with the input"
        );
        // Evaluates the bias at the random point so verifier can substract the evaluation
        // from the sumcheck claim that is only about the matrix2vec product.
        assert_eq!(
            dense.bias.get_data().len().ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        let bias_eval = dense
            .bias
            .evals_flat::<E>()
            .into_mle()
            .evaluate(&last_claim.point);
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle_2d();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = input.get_data().to_vec().into_mle();

        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let tmp_transcript = self.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        debug_assert!({
            let mut t = tmp_transcript;
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(num_vars);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = output.get_data().to_vec().into_mle();
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            let claimed_sum_no_bias = claimed_sum - bias_eval;
            debug_assert_eq!(claimed_sum, last_claim.eval, "sumcheck eval weird");
            debug_assert_eq!(
                claimed_sum_no_bias,
                proof.extract_sum(),
                "sumcheck output weird"
            );

            trace!("prover: claimed sum: {:?}", claimed_sum);
            let subclaim =
                IOPVerifierState::<E>::verify(claimed_sum_no_bias, &proof, &vp.aux_info, &mut t);
            // now assert that the polynomial evaluated at the random point of the sumcheck proof
            // is equal to last small poly sent by prover (`subclaim.expected_evaluation`). This
            // step can be done via PCS opening proofs for all steps but first (output of
            // inference) and last (input of inference)
            let computed_point = vp.evaluate(subclaim.point_flat().as_ref());

            let final_prover_point = state
                .get_mle_final_evaluations()
                .into_iter()
                .fold(E::ONE, |acc, eval| acc * eval);
            assert_eq!(computed_point, final_prover_point);

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // is done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let point = [proof.point.as_slice(), last_claim.point.as_slice()].concat();
        let eval = state.get_mle_final_evaluations()[0];
        self.commit_prover
            .add_claim(info.matrix_poly_id, Claim::new(point, eval))
            .context("unable to add matrix claim")?;
        // add the bias claim over the last claim input, since that is what is needed to "remove" the bias
        // to only verify the matrix2vec product via the sumcheck proof.
        self.commit_prover
            .add_claim(info.bias_poly_id, Claim::new(last_claim.point, bias_eval))
            .context("unable to add bias claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        self.proofs.push(StepProof::Dense(DenseProof {
            sumcheck: proof,
            bias_eval,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
    }

    pub fn prove<'b>(mut self, trace: InferenceTrace<'b, Element, E>) -> anyhow::Result<Proof<E>> {
        // write commitments and polynomials info to transcript
        self.ctx.write_to_transcript(self.transcript)?;
        // then create the context for the witness polys -
        self.instantiate_witness_ctx(&trace)?;
        let trace = trace.to_field();
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let r_i = self
            .transcript
            .read_challenges(trace.final_output().get_data().len().ilog2() as usize);
        let y_i = trace
            .last_step()
            .output
            .clone()
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&r_i);

        let mut last_claim = Claim {
            point: r_i,
            eval: y_i,
        };

        // let trace_size = trace.last_step().id;

        // we start by the output to prove up to the input, GKR style
        for ((input, step), info) in trace.iter().rev().zip(self.ctx.steps_info.iter()) {
            last_claim = self.prove_step(last_claim, input, step, &info)?;
        }

        // Now we have to make the table proofs
        self.prove_tables()?;

        // now provide opening proofs for all claims accumulated during the proving steps
        let commit_proof = self
            .commit_prover
            .prove(&self.ctx.weights, self.transcript)?;
        let mut output_proof = Proof {
            steps: self.proofs,
            table_proofs: self.table_proofs,
            commit: commit_proof,
            witness: None,
        };
        if let Some(witness_ctx) = self.witness_ctx {
            let witness_proof = self.witness_prover.prove(&witness_ctx, self.transcript)?;
            output_proof.witness = Some((witness_proof, witness_ctx));
        }
        Ok(output_proof)
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    #[timed_instrument(level = "debug")]
    fn instantiate_witness_ctx<'b>(
        &mut self,
        trace: &InferenceTrace<'b, Element, E>,
    ) -> anyhow::Result<()> {
        let (witness_ctx, challenge_storage, lookup_witnesses, table_witnesses) =
            generate_lookup_witnesses::<E, T>(trace, self.transcript)?;
        // let (lookup_witness, polys) =
        //     lookup::WitnessContext::<E>::initialise_witness_ctx(&self.ctx.lookup, trace)?;

        self.witness_ctx = Some(witness_ctx);
        self.challenge_storage = challenge_storage;
        self.lookup_witness = lookup_witnesses;
        self.table_witness = table_witnesses;
        // if !polys.is_empty() {
        //     let ctx = precommit::Context::generate(polys)
        //         .context("unable to generate ctx for witnesses")?;
        //     ctx.write_to_transcript(self.transcript)?;
        //     // Set the witness context
        //     self.witness_ctx = Some(ctx);
        //     // generate all the lookup related challenges
        //     self.challenge_storage = ChallengeStorage::<E>::initialise(self.ctx, self.transcript);
        // } else {
        //     warn!("no activation functions found - no witness commitment");
        // }
        // self.lookup_witness = lookup_witness;
        Ok(())
    }
}
