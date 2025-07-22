use anyhow::ensure;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use crate::{Claim, Element, Tensor, quantization::TensorFielder, tensor::Number};

/// A matrix vector multiplication proving logic where the matrix is NOT committed to, i.e.
/// the verifier will have to evaluate himself the matrix at a random point.
/// NOTE: this can be changed down the line to provide flexibility but right now we don't need it.
pub struct MatVec<T> {
    matrix: Tensor<T>,
}
pub struct MatVecProof<E: ExtensionField> {
    sumcheck: IOPProof<E>,
    evaluations: Vec<E>,
}

impl<E: ExtensionField> MatVecProof<E> {
    /// Evaluation of the matrix at the last random point of the sumcheck
    pub fn matrix_eval(&self) -> E {
        self.evaluations[0]
    }
    /// Evaluation of the vector at the last random point of the sumcheck
    pub fn vec_eval(&self) -> E {
        self.evaluations[1]
    }
}
impl<T: Number> MatVec<T> {
    pub fn new(matrix: Tensor<T>) -> Self {
        Self { matrix }
    }

    /// Note that it flattens the input vector, and the output is "flat" as in it's a vector.
    /// When integrating this into a convolution layer for example, one needs to reshape the output to the expected shape.
    pub fn op(&self, input: &Tensor<T>) -> Tensor<T> {
        self.matrix.matvec(&input.flatten())
    }
    pub fn aux_info<E: ExtensionField>(&self) -> VPAuxInfo<E> {
        // we fix the rows variables during sumcheck so we only consider the columns
        let num_vars = self.matrix.ncols_2d().ilog2() as usize;
        VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![num_vars, num_vars]])
    }
}

impl MatVec<Element> {
    pub fn prove<E, T>(
        &self,
        transcript: &mut T,
        last_claim: &Claim<E>,
        input: &Tensor<E>,
    ) -> anyhow::Result<(MatVecProof<E>, Claim<E>)>
    where
        E: ExtensionField + Serialize + DeserializeOwned + Number,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        let (nrows, ncols) = (self.matrix.nrows_2d(), self.matrix.ncols_2d());
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "invalid last_claim dimension: mat {}x{} vs {}",
            nrows,
            ncols,
            last_claim.point.len()
        );
        assert_eq!(
            ncols,
            input.get_data().len(),
            "invalid input dimension: mat {}x{} vs {}",
            nrows,
            ncols,
            input.get_data().len()
        );
        let mut mat_mle = self.matrix.to_2d_mle();
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = input.get_data().to_vec().into_mle();
        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
        // let tmp_transcript = prover.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, transcript);
        debug_assert!(
            {
                let output = self.matrix.clone().to_fields().matvec(input);
                let claimed_sum = output
                    .get_data()
                    .to_vec()
                    .into_mle()
                    .evaluate(&last_claim.point);
                proof.extract_sum() == claimed_sum
            },
            "invalid sumcheck proof for matvec?"
        );
        let claim = Claim {
            point: proof.point.clone(),
            // [mat, vec] -> we want the vector evaluation for the next layer
            eval: state.get_mle_final_evaluations()[1],
        };
        let proof = MatVecProof {
            sumcheck: proof,
            evaluations: state.get_mle_final_evaluations(),
        };
        Ok((proof, claim))
    }

    pub fn evaluate_matrix_at<E: ExtensionField>(&self, point: &[E]) -> E {
        self.matrix.to_2d_mle().evaluate(point)
    }
}

pub fn verify<E, T>(
    transcript: &mut T,
    last_claim: &Claim<E>,
    proof: &MatVecProof<E>,
    aux_info: &VPAuxInfo<E>,
    // function that returns the evaluation of the matrix at a given point. the verifier is able to succinctly
    // evaluate the matrix at a given point or just naively.
    eval_matrix_at: impl FnOnce(&[E]) -> E,
) -> anyhow::Result<Claim<E>>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    T: Transcript<E>,
{
    let subclaim =
        IOPVerifierState::<E>::verify(last_claim.eval, &proof.sumcheck, aux_info, transcript);
    let matrix_point = subclaim
        .point_flat()
        .iter()
        .chain(last_claim.point.iter())
        .cloned()
        .collect_vec();
    let matrix_eval = eval_matrix_at(&matrix_point);
    ensure!(proof.matrix_eval() == matrix_eval);
    Ok(Claim {
        point: subclaim.point_flat(),
        eval: proof.vec_eval(),
    })
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::{default_transcript, testing::random_field_vector};

    use super::*;

    type F = GoldilocksExt2;
    #[test]
    fn test_matvec_prove_verify() {
        let matrix =
            Tensor::<Element>::random_seed(&vec![2, 3].into(), None).pad_next_power_of_two();
        let input = Tensor::<Element>::random_seed(&vec![3].into(), None).pad_next_power_of_two();
        let matvec = MatVec::new(matrix);
        let output = matvec.op(&input);
        let mut transcript = default_transcript::<F>();
        let output_point = random_field_vector(output.get_data().len().ilog2() as usize);
        let output_claim = Claim::new(
            output_point.clone(),
            output.to_mle_flat().evaluate(&output_point),
        );
        let (proof, claim) = matvec
            .prove(&mut transcript, &output_claim, &input.clone().to_fields())
            .expect("proving should work");
        let input_eval = input.to_mle_flat().evaluate(&claim.point);
        assert_eq!(input_eval, claim.eval);
        let input_claim = verify(
            &mut default_transcript(),
            &output_claim,
            &proof,
            &matvec.aux_info(),
            |point| matvec.evaluate_matrix_at(point),
        )
        .expect("verification failed");
        assert_eq!(claim, input_claim);
    }
}
