use std::collections::HashMap;

use super::{ChallengeStorage, Context, Proof, TableProof};
use crate::{
    Claim, Element, VectorTranscript,
    commit::{compute_betas_eval, precommit},
    layers::{
        LayerProof,
        provable::{NodeId, OpInfo, ProvableOp},
    },
    lookup::{
        context::{TABLE_POLY_ID_OFFSET, generate_lookup_witnesses},
        logup_gkr::{prover::batch_prove as logup_batch_prove, structs::LogUpInput},
    },
    model::{InferenceStep, InferenceTrace, ToIterator},
    tensor::get_root_of_unity,
};
use anyhow::anyhow;
use ff_ext::ExtensionField;

use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::VirtualPolynomial,
};
use serde::{Serialize, de::DeserializeOwned};

use sumcheck::structs::IOPProverState;
use timed::timed_instrument;
use tracing::debug;
use transcript::Transcript;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx: &'a Context<E>,
    // proofs for each layer being filled
    proofs: HashMap<NodeId, LayerProof<E>>,
    table_proofs: Vec<TableProof<E>>,
    pub(crate) transcript: &'a mut T,
    pub(crate) commit_prover: precommit::CommitProver<E>,
    /// the context of the witness part (IO of lookups, linked with matrix2vec for example)
    /// is generated during proving time. It is first generated and then the fiat shamir starts.
    /// The verifier doesn't know about the individual polys (otherwise it beats the purpose) so
    /// that's why it is generated at proof time.
    pub(crate) witness_ctx: Option<precommit::Context<E>>,
    /// The prover related to proving multiple claims about different witness polyy (io of lookups etc)
    pub(crate) witness_prover: precommit::CommitProver<E>,
    /// The lookup witnesses
    pub(crate) lookup_witness: HashMap<NodeId, LogUpInput<E>>,
    /// The Lookup table witness
    pub(crate) table_witness: Vec<LogUpInput<E>>,
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
            lookup_witness: HashMap::default(),
            table_witness: Vec::default(),
            challenge_storage: ChallengeStorage::default(),
        }
    }

    pub(crate) fn lookup_witness(&mut self, id: NodeId) -> anyhow::Result<LogUpInput<E>> {
        self.lookup_witness
            .remove(&id)
            .ok_or(anyhow!("No lookup witness found for node {id}!"))
    }

    pub(crate) fn push_proof(&mut self, node_id: NodeId, proof: LayerProof<E>) {
        self.proofs.insert(node_id, proof);
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_tables(&mut self) -> anyhow::Result<()> {
        let mut poly_id = TABLE_POLY_ID_OFFSET;

        self.table_witness
            .iter()
            .zip(self.ctx.lookup.iter())
            .try_for_each(|(table_witness, _table_type)| {
                // Make the proof for the table
                let table_proof = logup_batch_prove(table_witness, self.transcript)?;

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
        let mut omegas = vec![E::ZERO; 1 << r1.len()];
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
                        + (E::ONE - E::from_canonical_u64(2) * r2[r2.len() - 1])
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
            phi = phi.inverse();
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
                    phi_g[l] *= tmp1 + tmp2;
                }
                if i < n {
                    mid_phi_g[i - 1] = vec![E::ZERO; 1 << (i)];
                    mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
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
                    phi_g[l] *= tmp1 + tmp2;
                }
                mid_phi_g[i - 1] = vec![E::ZERO; 1 << i];
                mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
            }
            for b in 0..(1 << (n - 1)) {
                let l = b;
                let tmp1 = E::ONE - rx[0];
                let tmp2 = rx[0] * phi_mul[b];
                phi_g[l] *= tmp1 + tmp2;
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
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);

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
            E::ONE,
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
        let scale: E = E::from_canonical_u64(prod[0].len() as u64).inverse();

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

    pub fn prove<'b>(
        mut self,
        full_trace: InferenceTrace<'b, E, Element>,
    ) -> anyhow::Result<Proof<E>> {
        // write commitments and polynomials info to transcript
        self.ctx.write_to_transcript(self.transcript)?;
        // then create the context for the witness polys -
        debug!("Prover : instantiate witness ctx...");
        self.instantiate_witness_ctx(&full_trace)?;
        debug!("Prover : instantiate witness ctx done...");
        let trace = full_trace.to_field();
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let out_claims = trace
            .outputs()?
            .into_iter()
            .map(|out| {
                let r_i = self
                    .transcript
                    .read_challenges(out.get_data().len().ilog2() as usize);
                let y_i = out.get_data().to_vec().into_mle().evaluate(&r_i);
                Claim {
                    point: r_i,
                    eval: y_i,
                }
            })
            .collect_vec();

        let mut claims_by_layer: HashMap<NodeId, Vec<Claim<E>>> = HashMap::new();
        for (node_id, ctx) in self.ctx.steps_info.to_backward_iterator() {
            let InferenceStep {
                op: node_operation,
                step_data,
            } = trace
                .get_step(&node_id)
                .ok_or(anyhow!("Step in trace not found for node {}", node_id))?;
            println!(
                "Proving node with id {node_id}: {:?}",
                node_operation.describe()
            );
            let claims_for_prove = ctx.claims_for_node(&claims_by_layer, &out_claims)?;
            let claims = if node_operation.is_provable() {
                node_operation.prove(node_id, &ctx.ctx, claims_for_prove, step_data, &mut self)?
            } else {
                // we only propagate the claims, without changing them, as a non-provable layer
                // shouldn't change the input values
                claims_for_prove.into_iter().cloned().collect()
            };
            claims_by_layer.insert(node_id, claims);
        }

        // let trace_size = trace.last_step().id;

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
    #[timed_instrument]
    fn instantiate_witness_ctx<'b>(
        &mut self,
        trace: &InferenceTrace<'b, E, Element>,
    ) -> anyhow::Result<()> {
        let (witness_ctx, challenge_storage, lookup_witnesses, table_witnesses) =
            generate_lookup_witnesses::<E, T>(trace, &self.ctx.steps_info, self.transcript)?;
        // let (lookup_witness, polys) =
        //     lookup::WitnessContext::<E>::initialise_witness_ctx(&self.ctx.lookup, trace)?;

        self.witness_ctx = witness_ctx;
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
