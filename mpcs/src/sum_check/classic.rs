use crate::{
    Error,
    sum_check::{SumCheck, VirtualPolynomial},
    util::{
        arithmetic::BooleanHypercube,
        expression::{Expression, Rotation},
        parallel::par_map_collect,
        poly_index_ext,
    },
};
use ark_std::{end_timer, start_timer};
use ff_ext::ExtensionField;
use itertools::Itertools;
use num_integer::Integer;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{borrow::Cow, collections::HashMap, fmt::Debug, marker::PhantomData};
use transcript::Transcript;
mod coeff;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, MultilinearExtension},
    virtual_poly::build_eq_x_r_vec,
};

pub(crate) use coeff::Coefficients;
pub use coeff::CoefficientsProver;
use p3_field::FieldAlgebra;

#[derive(Debug)]
pub struct ProverState<'a, E: ExtensionField> {
    num_vars: usize,
    expression: &'a Expression<E>,
    degree: usize,
    sum: E,
    lagranges: HashMap<i32, (usize, E)>,
    identity: E,
    eq_xys: Vec<DenseMultilinearExtension<E>>,
    polys: Vec<Vec<Cow<'a, DenseMultilinearExtension<E>>>>,
    challenges: &'a [E],
    round: usize,
    bh: BooleanHypercube,
}

impl<'a, E: ExtensionField> ProverState<'a, E> {
    fn new(num_vars: usize, sum: E, virtual_poly: VirtualPolynomial<'a, E>) -> Self {
        assert!(num_vars > 0 && virtual_poly.expression.max_used_rotation_distance() <= num_vars);
        let bh = BooleanHypercube::new(num_vars);
        let lagranges = {
            let bh = bh.iter().collect_vec();
            virtual_poly
                .expression
                .used_langrange()
                .into_iter()
                .map(|i| {
                    let b = bh[i.rem_euclid(1 << num_vars) as usize];
                    (i, (b, E::ONE))
                })
                .collect()
        };
        let eq_xys = virtual_poly
            .ys
            .iter()
            .map(|y| {
                DenseMultilinearExtension::from_evaluations_ext_vec(y.len(), build_eq_x_r_vec(y))
            })
            .collect_vec();
        let polys = virtual_poly
            .polys
            .iter()
            .map(|poly| {
                let mut polys = vec![
                    Cow::Owned(DenseMultilinearExtension::from_evaluations_vec(
                        0,
                        vec![E::BaseField::ZERO; 1]
                    ));
                    2 * num_vars
                ];
                polys[num_vars] = Cow::Borrowed(*poly);
                polys
            })
            .collect_vec();
        Self {
            num_vars,
            expression: virtual_poly.expression,
            degree: virtual_poly.expression.degree(),
            sum,
            lagranges,
            identity: E::ZERO,
            eq_xys,
            polys,
            challenges: virtual_poly.challenges,
            round: 0,
            bh,
        }
    }

    fn size(&self) -> usize {
        1 << (self.num_vars - self.round - 1)
    }

    fn next_round(&mut self, sum: E, challenge: &E) {
        self.sum = sum;
        self.identity += E::from_canonical_u64(1 << self.round) * *challenge;
        self.lagranges.values_mut().for_each(|(b, value)| {
            if b.is_even() {
                *value *= E::ONE - *challenge;
            } else {
                *value *= *challenge;
            }
            *b >>= 1;
        });
        self.eq_xys.iter_mut().for_each(|eq_xy| {
            if eq_xy.num_vars > 0 {
                eq_xy.fix_variables_in_place(&[*challenge])
            }
        });
        if self.round == 0 {
            let rotation_maps = self
                .expression
                .used_rotation()
                .into_iter()
                .filter(|&rotation| (rotation != Rotation::cur()))
                .map(|rotation| (rotation, self.bh.rotation_map(rotation)))
                .collect::<HashMap<_, _>>();
            for query in self.expression.used_query() {
                if query.rotation() != Rotation::cur() {
                    let poly = &self.polys[query.poly()][self.num_vars];
                    let rotated: Vec<_> = par_map_collect(&rotation_maps[&query.rotation()], |b| {
                        poly_index_ext(poly, *b)
                    });
                    let rotated = DenseMultilinearExtension::from_evaluations_ext_vec(
                        rotated.len().ilog2() as usize,
                        rotated,
                    )
                    .fix_variables(&[*challenge]);
                    self.polys[query.poly()]
                        [(query.rotation().0 + self.num_vars as i32) as usize] =
                        Cow::Owned(rotated);
                }
            }
            self.polys.iter_mut().for_each(|polys| {
                polys[self.num_vars] =
                    Cow::Owned(polys[self.num_vars].fix_variables(&[*challenge]));
            });
        } else {
            self.polys.iter_mut().for_each(|polys| {
                polys.iter_mut().for_each(|poly| {
                    // If it's constant, then fixing a variable is a no-op
                    if poly.num_vars > 0 {
                        poly.to_mut().fix_variables_in_place(&[*challenge]);
                    }
                });
            });
        }
        self.round += 1;
        self.bh = BooleanHypercube::new(self.num_vars - self.round);
    }

    fn into_evals(self) -> Vec<E> {
        assert_eq!(self.round, self.num_vars);
        self.polys
            .iter()
            .map(|polys| poly_index_ext(polys[self.num_vars].as_ref(), 0))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SumcheckProof<E: ExtensionField, RoundMessage: ClassicSumCheckRoundMessage<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    rounds: Vec<RoundMessage>,
    phantom: PhantomData<E>,
}

pub trait ClassicSumCheckProver<E: ExtensionField>: Clone + Debug {
    type RoundMessage: ClassicSumCheckRoundMessage<E> + Clone + Debug;

    fn new(state: &ProverState<E>) -> Self;

    fn prove_round(&self, state: &ProverState<E>) -> Self::RoundMessage;

    fn sum(&self, state: &ProverState<E>) -> E;
}

pub trait ClassicSumCheckRoundMessage<E: ExtensionField>: Sized + Debug {
    type Auxiliary: Default;

    fn write(&self, transcript: &mut impl Transcript<E>) -> Result<(), Error>;

    fn sum(&self) -> E;

    fn auxiliary(_degree: usize) -> Self::Auxiliary {
        Default::default()
    }

    fn evaluate(&self, aux: &Self::Auxiliary, challenge: &E) -> E;

    fn verify_consistency(
        degree: usize,
        mut sum: E,
        msgs: &[Self],
        challenges: &[E],
    ) -> Result<E, Error> {
        let aux = Self::auxiliary(degree);
        for (round, (msg, challenge)) in msgs.iter().zip(challenges.iter()).enumerate() {
            if sum != msg.sum() {
                let msg = if round == 0 {
                    format!("Expect sum {sum:?} but get {:?}", msg.sum())
                } else {
                    format!("Consistency failure at round {round}")
                };
                return Err(Error::InvalidSumcheck(msg));
            }
            sum = msg.evaluate(&aux, challenge);
        }
        Ok(sum)
    }
}

#[derive(Clone, Debug)]
pub struct ClassicSumCheck<P>(PhantomData<P>);

impl<E: ExtensionField, P: ClassicSumCheckProver<E>> SumCheck<E> for ClassicSumCheck<P>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    type ProverParam = ();
    type VerifierParam = ();
    type RoundMessage = P::RoundMessage;

    fn prove(
        _: &Self::ProverParam,
        num_vars: usize,
        virtual_poly: VirtualPolynomial<E>,
        sum: E,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(Vec<E>, Vec<E>, SumcheckProof<E, Self::RoundMessage>), Error> {
        let _timer = start_timer!(|| {
            let degree = virtual_poly.expression.degree();
            format!("sum_check_prove-{num_vars}-{degree}")
        });

        let mut state = ProverState::new(num_vars, sum, virtual_poly);
        let mut challenges = Vec::with_capacity(num_vars);
        let prover = P::new(&state);

        if cfg!(feature = "sanity-check") {
            assert_eq!(prover.sum(&state), state.sum);
        }

        let aux = P::RoundMessage::auxiliary(state.degree);

        let mut prover_messages = Vec::with_capacity(num_vars);

        for _round in 0..num_vars {
            let timer = start_timer!(|| format!("sum_check_prove_round-{_round}"));
            let msg = prover.prove_round(&state);
            end_timer!(timer);
            msg.write(transcript)?;

            if cfg!(feature = "sanity-check") {
                assert_eq!(
                    msg.evaluate(&aux, &E::ZERO) + msg.evaluate(&aux, &E::ONE),
                    state.sum
                );
            }

            let challenge = transcript
                .get_and_append_challenge(b"sumcheck round")
                .elements;
            challenges.push(challenge);

            let timer = start_timer!(|| format!("sum_check_next_round-{_round}"));
            state.next_round(msg.evaluate(&aux, &challenge), &challenge);
            end_timer!(timer);
            prover_messages.push(msg);
        }

        let proof = SumcheckProof {
            rounds: prover_messages,
            phantom: PhantomData,
        };
        Ok((challenges, state.into_evals(), proof))
    }

    fn verify(
        _: &Self::VerifierParam,
        num_vars: usize,
        degree: usize,
        sum: E,
        proof: &SumcheckProof<E, P::RoundMessage>,
        transcript: &mut impl Transcript<E>,
    ) -> Result<(E, Vec<E>), Error> {
        let (msgs, challenges) = {
            let mut msgs = Vec::with_capacity(num_vars);
            let mut challenges = Vec::with_capacity(num_vars);
            for i in 0..num_vars {
                proof.rounds[i].write(transcript)?;
                msgs.push(proof.rounds[i].clone());
                challenges.push(
                    transcript
                        .get_and_append_challenge(b"sumcheck round")
                        .elements,
                );
            }
            (msgs, challenges)
        };

        Ok((
            P::RoundMessage::verify_consistency(degree, sum, msgs.as_slice(), &challenges)?,
            challenges,
        ))
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        sum_check::eq_xy_eval,
        util::{arithmetic::inner_product, expression::Query, poly_iter_ext},
    };
    use transcript::BasicTranscript;

    use super::*;
    use ff_ext::GoldilocksExt2 as E;
    use p3_goldilocks::Goldilocks as Fr;

    #[test]
    fn test_sum_check_protocol() {
        let polys = [
            DenseMultilinearExtension::<E>::from_evaluations_vec(
                2,
                vec![
                    Fr::from_canonical_u64(1),
                    Fr::from_canonical_u64(2),
                    Fr::from_canonical_u64(3),
                    Fr::from_canonical_u64(4),
                ],
            ),
            DenseMultilinearExtension::from_evaluations_vec(
                2,
                vec![
                    Fr::from_canonical_u64(0),
                    Fr::from_canonical_u64(1),
                    Fr::from_canonical_u64(1),
                    Fr::from_canonical_u64(0),
                ],
            ),
            DenseMultilinearExtension::from_evaluations_vec(
                1,
                vec![Fr::from_canonical_u64(0), Fr::from_canonical_u64(1)],
            ),
        ];
        let points = vec![
            vec![E::from_canonical_u64(1), E::from_canonical_u64(2)],
            vec![E::from_canonical_u64(1)],
        ];
        let expression = Expression::<E>::eq_xy(0)
            * Expression::Polynomial(Query::new(0, Rotation::cur()))
            * E::from(Fr::from_canonical_u64(2))
            + Expression::<E>::eq_xy(0)
                * Expression::Polynomial(Query::new(1, Rotation::cur()))
                * E::from(Fr::from_canonical_u64(3))
            + Expression::<E>::eq_xy(1)
                * Expression::Polynomial(Query::new(2, Rotation::cur()))
                * E::from(Fr::from_canonical_u64(4));
        let virtual_poly =
            VirtualPolynomial::<E>::new(&expression, polys.iter(), &[], points.as_slice());
        let sum = inner_product(
            &poly_iter_ext(&polys[0]).collect_vec(),
            &build_eq_x_r_vec(&points[0]),
        ) * Fr::from_canonical_u64(2)
            + inner_product(
                &poly_iter_ext(&polys[1]).collect_vec(),
                &build_eq_x_r_vec(&points[0]),
            ) * Fr::from_canonical_u64(3)
            + inner_product(
                &poly_iter_ext(&polys[2]).collect_vec(),
                &build_eq_x_r_vec(&points[1]),
            ) * Fr::from_canonical_u64(4)
                * Fr::from_canonical_u64(2); // The third polynomial is summed twice because the hypercube is larger
        let mut transcript = BasicTranscript::new(b"sumcheck");
        let (challenges, evals, proof) =
            <ClassicSumCheck<CoefficientsProver<E>> as SumCheck<E>>::prove(
                &(),
                2,
                virtual_poly.clone(),
                sum,
                &mut transcript,
            )
            .unwrap();

        assert_eq!(polys[0].evaluate(&challenges), evals[0]);
        assert_eq!(polys[1].evaluate(&challenges), evals[1]);
        assert_eq!(polys[2].evaluate(&challenges[..1]), evals[2]);

        let mut transcript = BasicTranscript::new(b"sumcheck");

        let (new_sum, verifier_challenges) = <ClassicSumCheck<CoefficientsProver<E>> as SumCheck<
            E,
        >>::verify(
            &(), 2, 2, sum, &proof, &mut transcript
        )
        .unwrap();

        assert_eq!(verifier_challenges, challenges);
        assert_eq!(
            new_sum,
            evals[0] * eq_xy_eval(&points[0], &challenges[..2]) * Fr::from_canonical_u64(2)
                + evals[1] * eq_xy_eval(&points[0], &challenges[..2]) * Fr::from_canonical_u64(3)
                + evals[2] * eq_xy_eval(&points[1], &challenges[..1]) * Fr::from_canonical_u64(4)
        );

        let mut transcript = BasicTranscript::new(b"sumcheck");

        <ClassicSumCheck<CoefficientsProver<E>> as SumCheck<E>>::verify(
            &(),
            2,
            2,
            sum + Fr::ONE,
            &proof,
            &mut transcript,
        )
        .expect_err("Should panic");
    }
}
