use crate::{
    commit::{compute_betas_eval, identity_eval}, iop::context::ContextAux, layers::{requant::Requant, LayerProof, PolyID}, quantization, tensor::{get_root_of_unity, ConvData}, Claim, Prover
};
use anyhow::Context;
use ff_ext::ExtensionField;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use tracing::instrument;
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

use super::LayerCtx;

pub(crate) const BIAS_POLY_ID: PolyID = 200_000;
/// Convolution layer description (weights)
#[derive(Clone, Debug)]
pub struct Convolution {
    pub filter: Tensor<Element>,
    pub bias: Tensor<Element>,
}

/// Info about the convolution layer derived during the setup phase
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvCtx<E> {
    pub poly_id: PolyID,
    pub bias_poly_id: PolyID,
    pub fft_aux: VPAuxInfo<E>,
    pub ifft_aux: VPAuxInfo<E>,
    pub delegation_fft: Vec<VPAuxInfo<E>>,
    pub delegation_ifft: Vec<VPAuxInfo<E>>,
    pub hadamard: VPAuxInfo<E>,
    pub kw: usize,
    pub kx: usize,
    pub filter_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchoolBookConvCtx<E> {
    pub dummy: E,
}

/// Contains proof material related to one step of the inference for a convolution layer
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ConvProof<E: ExtensionField> {
    // Sumcheck proof for the FFT layer
    fft_proof: IOPProof<E>,
    // Proof for the evaluation delegation of the omegas matrix
    // It consists of multiple sumcheck proofs
    fft_delegation_proof: Vec<IOPProof<E>>,
    // Likewise for fft, we define ifft proofs
    ifft_proof: IOPProof<E>,
    ifft_delegation_proof: Vec<IOPProof<E>>,
    // Sumcheck proof for the hadamard product
    hadamard_proof: IOPProof<E>,
    // The evaluation claims produced by the corresponding sumchecks
    fft_claims: Vec<E>,
    ifft_claims: Vec<E>,
    fft_delegation_claims: Vec<Vec<E>>,
    ifft_delegation_claims: Vec<Vec<E>>,
    hadamard_clams: Vec<E>,
    bias_claim: E,
}

impl Convolution {
    pub fn new(filter: Tensor<Element>, bias: Tensor<Element>) -> Self {
        assert_eq!(filter.kw(), bias.dims()[0]);
        Self { filter, bias }
    }
    pub fn add_bias(&self, conv_out: &Tensor<Element>) -> Tensor<Element> {
        let mut arr = conv_out.data.clone();
        assert_eq!(conv_out.data.len(), conv_out.kw() * conv_out.filter_size());
        for i in 0..conv_out.kw() {
            for j in 0..conv_out.filter_size() {
                arr[i * conv_out.filter_size() + j] += self.bias.data[i];
            }
        }
        Tensor::new(conv_out.get_shape(), arr)
    }

    pub fn op<E: ExtensionField>(&self, input: &Tensor<Element>) -> (Tensor<Element>, ConvData<E>) {
        let (output, proving_data) = self.filter.fft_conv(input);
        (self.add_bias(&output), proving_data)
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.filter.get_shape()
    }

    pub fn kw(&self) -> usize {
        self.filter.kw()
    }

    pub fn kx(&self) -> usize {
        self.filter.kx()
    }

    pub fn nw(&self) -> usize {
        self.filter.nw()
    }

    pub fn ncols_2d(&self) -> usize {
        self.filter.ncols_2d()
    }

    pub fn nrows_2d(&self) -> usize {
        self.filter.nrows_2d()
    }
    pub fn filter_size(&self) -> usize {
        self.filter.filter_size()
    }
    pub fn requant_info<E: ExtensionField>(&self) -> Requant {
        let weights = self.filter.get_real_weights::<E>();
        let min_quant = *quantization::MIN as Element;
        let max_quant = *quantization::MAX as Element;

        let mut max_output: Element = 0;
        let mut min_output: Element = 0;
        let max_bias = self.bias.get_data().iter().max().unwrap();
        let min_bias = self.bias.get_data().iter().min().unwrap();

        // Keep the original iteration order: first over kernel height (i), then kernel width (j), then output channels (k)
        for i in 0..self.kw() {
            for j in 0..self.kx() {
                let mut min_temp: Element = *min_bias;
                let mut max_temp: Element = *max_bias;

                // Loop over output channels (k) and apply weights and bias
                for k in 0..(self.nw() * self.nw()) {
                    let weight = weights[i][j][k];
                    // PANICKING HERE
                    // let bias = self.bias.data[k]; // Bias for the current output channel
                    let bias = *max_bias;

                    if weight != 0 {
                        let (min_contrib, max_contrib) = if weight < 0 {
                            (max_quant * weight, min_quant * weight)
                        } else {
                            (min_quant * weight, max_quant * weight)
                        };

                        min_temp += min_contrib;
                        max_temp += max_contrib;
                    }

                    // Add the bias for this output channel `k`
                    min_temp += bias;
                    max_temp += bias;
                }

                // After processing all output channels for this (i, j) location, update the global min and max
                max_output = max_output.max(max_temp);
                min_output = min_output.min(min_temp);
            }
        }
        let max_range = 2 * (max_output - min_output).unsigned_abs().next_power_of_two();
        assert!(max_range.ilog2() as usize > *quantization::BIT_LEN);
        let shift = (2 * max_range).ilog2() as usize - *quantization::BIT_LEN;
        Requant {
            // range: (max_val - min_val) as usize,
            range: max_range as usize,
            right_shift: shift,
            after_range: 1 << *quantization::BIT_LEN,
        }
    }

    pub fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // TO SEE
        // last_output_size = filter.nrows_2d();
        // let filter_shape = filter.filter.dims();
        // let total_dims = last_output_shape.len();
        // last_output_shape = std::iter::once(filter_shape[0])
        // .chain(
        // last_output_shape
        // .iter()
        // .skip(total_dims - 2)
        // .map(|&dim| ceil_log2(dim)),
        // )
        // .collect::<Vec<usize>>();
        let mut filter_shape = self.filter.dims();
        filter_shape.remove(1);
        aux.last_output_shape = filter_shape;

        let mut delegation_fft: Vec<VPAuxInfo<E>> = Vec::new();
        let mut delegation_ifft: Vec<VPAuxInfo<E>> = Vec::new();
        // println!("{},{}",id,filter.filter_size());
        for i in (0..(self.filter_size().ilog2() as usize)).rev() {
            delegation_fft.push(VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                i + 1,
                i + 1,
                i + 1,
            ]]));
            delegation_ifft.push(VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                i + 1,
                i + 1,
                i + 1,
            ]]));
        }

        let conv_info = LayerCtx::Convolution(ConvCtx {
            poly_id: id,
            bias_poly_id: BIAS_POLY_ID + id,
            ifft_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                ((self.filter_size()).ilog2() as usize) + 1,
                ((self.filter_size()).ilog2() as usize) + 1,
            ]]),
            fft_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                ((self.filter_size()).ilog2() as usize) + 1,
                ((self.filter_size()).ilog2() as usize) + 1,
            ]]),
            hadamard: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                ((self.kx() * self.filter_size()).ilog2() as usize) + 1,
                ((self.kx() * self.filter_size()).ilog2() as usize) + 1,
                ((self.kx() * self.filter_size()).ilog2() as usize) + 1,
            ]]),
            delegation_fft,
            delegation_ifft,
            kw: self.kw(),
            kx: self.kx(),
            filter_size: self.filter_size(),
        });
        (conv_info, aux)
    }

    // Prove convolution of a CNN network. This is a convolution between in a 3D matrix X of dimension k_x * n_x * n_x
    // and a 4D filter matrix W of dimension k_w * k_x * n_w * n_w. The output is a 3D matrix Y of dimension k_w * n_x * n_x
    // We want to batch prove the following: Y[i] = iFFT(sum_{j \in [n_x]}(FFT(X[j]) o FFT(W[i][j])).
    #[instrument(name = "Prover::prove_convolution_step", skip_all, level = "debug")]
    #[timed::timed_instrument(level = "debug")]
    pub fn prove_convolution_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        // last random claim made
        last_claim: Claim<E>,
        // Struct containing all necessary information
        // to generate a convolution proof
        _output: &Tensor<E>,
        proving_data: &ConvData<E>,
        info: &ConvCtx<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        let filter = self;
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

        let mut temp_t = prover.transcript.clone();
        let (ifft_proof, ifft_claim, ifft_del_proof) =
            prover.prove_batch_ifft(r.clone(), &proving_data.prod);

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
        let (hadamard_proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);
        let hadamard_claims = state.get_mle_final_evaluations();

        let point = [hadamard_proof.point.as_slice(), r1].concat();
        // let eval = hadamard_claims[0];
        prover
            .commit_prover
            .add_claim(info.poly_id, Claim::new(point, hadamard_claims[0]))
            .context("unable to add convolution claim")?;
        prover
            .commit_prover
            .add_claim(info.bias_poly_id, Claim::new(bias_point, bias_eval))
            .context("unable to add bias claim in convolution")?;

        // Finally prove the correct computation of the x_fft and get an evaluation claim of the input
        let (fft_proof, fft_claim, fft_del_proof) = prover.prove_batch_fft(
            hadamard_proof.point.clone(),
            &mut proving_data.input.clone(),
        );

        prover.push_proof(LayerProof::Convolution(ConvProof {
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
}

impl<E> ConvCtx<E> {
    pub(crate) fn verify_convolution<E: ExtensionField, T: Transcript<E>>(
        &self,
        &mut verifier: Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &ConvProof<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        let conv_claim = last_claim.eval - proof.bias_claim;

        IOPVerifierState::<E>::verify(conv_claim, &proof.ifft_proof, &self.ifft_aux, t);
        assert_eq!(
            self.delegation_ifft.len(),
            proof.ifft_delegation_proof.len(),
            "Inconsistency in iFFT delegation proofs/aux size"
        );

        let mut iter = proof.ifft_delegation_proof.len();
        let mut claim = proof.ifft_claims[1];
        let mut exponents = pow_two_omegas(iter + 1, true);
        let mut prev_r = proof.ifft_proof.point.clone();
        for i in 0..iter {
            IOPVerifierState::<E>::verify(
                claim,
                &proof.ifft_delegation_proof[i],
                &self.delegation_ifft[i],
                verifier.transcript,
            );
            assert_eq!(
                identity_eval(
                    proof.ifft_delegation_proof[i].point.clone().as_slice(),
                    prev_r.clone().as_slice()
                ),
                proof.ifft_delegation_claims[i][0],
                "Error in identity evaluation ifft delegation iter : {}",
                i
            );
            assert_eq!(
                phi_eval(
                    proof.ifft_delegation_proof[i].point.clone(),
                    E::ONE - last_claim.point[i],
                    prev_r[prev_r.len() - 1],
                    exponents.clone(),
                    false
                ),
                proof.ifft_delegation_claims[i][1],
                "Error in phi computation ifft delegation iter : {}",
                i
            );

            prev_r = proof.ifft_delegation_proof[i].point.clone();
            claim = proof.ifft_delegation_claims[i][2];
        }
        let scale = E::from(1 << (iter + 1)).invert().unwrap();

        assert_eq!(
            claim,
            scale * (E::ONE) * prev_r[0] + scale * (E::ONE - prev_r[0]),
            "Error in final iFFT delegation step"
        );

        IOPVerifierState::<E>::verify(
            proof.ifft_claims[0],
            &proof.hadamard_proof,
            &self.hadamard,
            verifier.transcript,
        );
        assert_eq!(
            proof.hadamard_clams[2],
            identity_eval(&proof.ifft_proof.point, &proof.hadamard_proof.point),
            "Error in Beta evaluation"
        );

        verifier.commit_verifier.add_claim(
            self.poly_id,
            Claim::new(
                [
                    proof.hadamard_proof.point.clone(),
                    last_claim.point[((self.filter_size).ilog2() as usize)..].to_vec(),
                ]
                .concat(),
                proof.hadamard_clams[0],
            ),
        )?;

        verifier.commit_verifier.add_claim(
            self.bias_poly_id,
            Claim::new(
                last_claim.point[(proof.ifft_delegation_proof.len())..].to_vec(),
                proof.bias_claim,
            ),
        )?;

        // >>>>>> TODO : 1) Dont forget beta evaluation 2) verification of the last step of delegation <<<<<<<
        // Verify fft sumcheck
        IOPVerifierState::<E>::verify(proof.hadamard_clams[1], &proof.fft_proof, &self.fft_aux, t);
        claim = proof.fft_claims[1];

        assert_eq!(
            self.delegation_fft.len(),
            proof.fft_delegation_proof.len(),
            "Inconsistency in FFT delegation proofs/aux size"
        );
        iter = proof.fft_delegation_proof.len();
        // Verify delegation protocol of W iFFT matrix
        exponents = pow_two_omegas(iter + 1, false);
        prev_r = proof.fft_proof.point.clone();
        for i in 0..iter {
            IOPVerifierState::<E>::verify(
                claim,
                &proof.fft_delegation_proof[i],
                &self.delegation_fft[i],
                verifier.transcript,
            );

            assert_eq!(
                identity_eval(
                    proof.fft_delegation_proof[i].point.clone().as_slice(),
                    prev_r.clone().as_slice()
                ),
                proof.fft_delegation_claims[i][0],
                "Error in identity evaluation fft delegation iter : {}",
                i
            );

            assert_eq!(
                phi_eval(
                    proof.fft_delegation_proof[i].point.clone(),
                    proof.hadamard_proof.point[i],
                    prev_r[prev_r.len() - 1],
                    exponents.clone(),
                    i == 0
                ),
                proof.fft_delegation_claims[i][1],
                "Error in phi computation fft delegation iter : {}",
                i
            );

            claim = proof.fft_delegation_claims[i][2];
            prev_r = proof.fft_delegation_proof[i].point.clone();
        }
        assert_eq!(
            claim,
            (E::ONE - E::from(2) * proof.hadamard_proof.point[iter]) * prev_r[0] + E::ONE
                - prev_r[0],
            "Error in final FFT delegation step"
        );
        let mut input_point = proof.fft_proof.point.clone();
        let mut v = input_point.pop().unwrap();
        v = (E::ONE - v).invert().unwrap();
        for i in 0..input_point.len() {
            input_point[i] = E::ONE - input_point[i];
        }
        // the output claim for this step that is going to be verified at next step
        Ok(Claim {
            // the new randomness to fix at next layer is the randomness from the sumcheck !
            point: [
                input_point.clone(),
                proof.hadamard_proof.point[((self.filter_size * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
            // random point. 1 because vector is secondary.
            eval: proof.fft_claims[0] * v,
        })
    }
}

impl<E: ExtensionField> SchoolBookConvCtx<E>
where
    E: Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub fn step_info(&self, _id: PolyID, aux: ContextAux) -> (LayerCtx<E>, ContextAux) {
        let conv_info = LayerCtx::SchoolBookConvolution(SchoolBookConvCtx { dummy: E::ZERO });
        (conv_info, aux)
    }
}

pub fn pow_two_omegas<E: ExtensionField>(n: usize, is_fft: bool) -> Vec<E> {
    let mut pows = vec![E::ZERO; n - 1];
    let mut rou: E = get_root_of_unity(n);
    if is_fft {
        rou = rou.invert().unwrap();
    }
    pows[0] = rou;
    for i in 1..(n - 1) {
        pows[i] = pows[i - 1] * pows[i - 1];
    }
    return pows;
}

pub fn phi_eval<E: ExtensionField>(
    r: Vec<E>,
    rand1: E,
    rand2: E,
    exponents: Vec<E>,
    first_iter: bool,
) -> E {
    let mut eval = E::ONE;
    for i in 0..r.len() {
        eval *= E::ONE - r[i] + r[i] * exponents[exponents.len() - r.len() + i];
    }

    if first_iter {
        eval = (E::ONE - rand2) * (E::ONE - rand1 + rand1 * eval);
    } else {
        eval = E::ONE - rand1 + (E::ONE - E::from(2) * rand2) * rand1 * eval;
    }

    return eval;
}

#[cfg(test)]
mod test {
    use crate::testing::random_vector;

    use super::*;
    use goldilocks::GoldilocksExt2;

    fn random_vector_quant(n: usize) -> Vec<Element> {
        // vec![thread_rng().gen_range(-128..128); n]
        random_vector(n)
    }

    fn test_conv_offset_poly_id() {
        // just a large difference so we're guaranteed that the IDs won't overlap.
        // TODO: change that process by a deterministic ID depending on the position and additional info
        // not necessarily seuential
        assert!(BIAS_POLY_ID > dense::BIAS_POLY_ID + 100_000);
    }

    #[test]
    pub fn test_quantization() {
        let n_w = 1 << 2;
        let k_w = 1 << 0;
        let n_x = 1 << 3;
        let k_x = 1 << 0;

        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        let w1 = random_vector_quant(k_w * k_x * n_w * n_w);

        let conv = Convolution::new(
            Tensor::new_conv(
                vec![k_w, k_x, n_w, n_w],
                in_dimensions[0].clone(),
                w1.clone(),
            ),
            Tensor::new(vec![k_w], random_vector_quant(k_w)),
        );
        let info = conv.requant_info::<GoldilocksExt2>();
        println!("range : {}", info.range);
        for _ in 0..100 {
            let (out, _proving_data) = conv.op::<GoldilocksExt2>(&Tensor::new(
                vec![k_x, n_x, n_x],
                random_vector_quant(k_x * n_x * n_x),
            ));
            for j in 0..out.data.len() {
                if out.data[j] < 0 {
                    assert!((-out.data[j] as usize) < info.range);
                } else {
                    assert!((out.data[j] as usize) < info.range);
                }
            }
        }
    }
}
