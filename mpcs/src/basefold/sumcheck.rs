use ff_ext::ExtensionField;
use multilinear_extensions::mle::FieldType;
use p3_field::Field;
use rayon::prelude::*;

pub fn sum_check_first_round_field_type<E: ExtensionField>(
    eq: &mut [E],
    bh_values: &mut FieldType<E>,
) -> Vec<E> {
    // The input polynomials are in the form of evaluations. Instead of viewing
    // every one element as the evaluation of the polynomial at a single point,
    // we can view every two elements as partially evaluating the polynomial at
    // a single point, leaving the first variable free, and obtaining a univariate
    // polynomial. The one_level_interp_hc transforms the evaluation forms into
    // the coefficient forms, for every of these partial polynomials.
    one_level_interp_hc(eq);
    one_level_interp_hc_field_type(bh_values);
    parallel_pi_field_type(bh_values, eq)
    //    p_i(&bh_values, &eq)
}

pub fn sum_check_first_round<E: ExtensionField>(eq: &mut [E], bh_values: &mut [E]) -> Vec<E> {
    // The input polynomials are in the form of evaluations. Instead of viewing
    // every one element as the evaluation of the polynomial at a single point,
    // we can view every two elements as partially evaluating the polynomial at
    // a single point, leaving the first variable free, and obtaining a univariate
    // polynomial. The one_level_interp_hc transforms the evaluation forms into
    // the coefficient forms, for every of these partial polynomials.
    one_level_interp_hc(eq);
    one_level_interp_hc(bh_values);
    parallel_pi(bh_values, eq)
    //    p_i(&bh_values, &eq)
}

pub fn one_level_interp_hc_field_type<E: ExtensionField>(evals: &mut FieldType<E>) {
    match evals {
        FieldType::Ext(evals) => one_level_interp_hc(evals),
        FieldType::Base(evals) => one_level_interp_hc(evals),
        _ => unreachable!(),
    }
}

pub fn one_level_interp_hc<F: Field>(evals: &mut [F]) {
    if evals.len() == 1 {
        return;
    }
    evals.par_chunks_mut(2).for_each(|chunk| {
        chunk[1] -= chunk[0];
    });
}

pub fn one_level_eval_hc<F: Field>(evals: &mut Vec<F>, challenge: F) {
    *evals = evals
        .par_chunks(2)
        .map(|chunk| chunk[0] + challenge * chunk[1])
        .collect::<Vec<F>>()
}

fn parallel_pi_field_type<E: ExtensionField>(evals: &mut FieldType<E>, eq: &mut [E]) -> Vec<E> {
    match evals {
        FieldType::Ext(evals) => parallel_pi(evals, eq),
        FieldType::Base(evals) => parallel_pi_base(evals, eq),
        _ => unreachable!(),
    }
}

fn parallel_pi<F: Field>(evals: &[F], eq: &[F]) -> Vec<F> {
    if evals.len() == 1 {
        return vec![evals[0], evals[0], evals[0]];
    }

    // Manually write down the multiplication formular of two linear polynomials

    let (coeff_1, coeff_2, coeff_3) = evals
        .par_chunks(2)
        .zip(eq.par_chunks(2))
        .fold(
            || (F::ZERO, F::ZERO, F::ZERO),
            |(first, second, third), (eval_chunk, eq_chunk)| {
                (
                    first + eval_chunk[0] * eq_chunk[0],
                    second + (eval_chunk[1] * eq_chunk[0] + eval_chunk[0] * eq_chunk[1]),
                    third + eval_chunk[1] * eq_chunk[1],
                )
            },
        )
        .reduce(
            || (F::ZERO, F::ZERO, F::ZERO),
            |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3),
        );

    vec![coeff_1, coeff_2, coeff_3]
}

fn parallel_pi_base<E: ExtensionField>(evals: &[E::BaseField], eq: &[E]) -> Vec<E> {
    if evals.len() == 1 {
        return vec![E::from(evals[0]), E::from(evals[0]), E::from(evals[0])];
    }
    let mut coeffs = vec![E::ZERO, E::ZERO, E::ZERO];

    // Manually write down the multiplication formular of two linear polynomials
    let mut firsts = vec![E::ZERO; evals.len()];
    firsts.par_iter_mut().enumerate().for_each(|(i, f)| {
        if i % 2 == 0 {
            *f = E::from(evals[i]) * eq[i];
        }
    });

    let mut seconds = vec![E::ZERO; evals.len()];
    seconds.par_iter_mut().enumerate().for_each(|(i, f)| {
        if i % 2 == 0 {
            *f = E::from(evals[i + 1]) * eq[i] + E::from(evals[i]) * eq[i + 1];
        }
    });

    let mut thirds = vec![E::ZERO; evals.len()];
    thirds.par_iter_mut().enumerate().for_each(|(i, f)| {
        if i % 2 == 0 {
            *f = E::from(evals[i + 1]) * eq[i + 1];
        }
    });

    coeffs[0] = firsts.par_iter().copied().sum();
    coeffs[1] = seconds.par_iter().copied().sum();
    coeffs[2] = thirds.par_iter().copied().sum();

    coeffs
}

pub fn sum_check_challenge_round<F: Field>(
    eq: &mut Vec<F>,
    bh_values: &mut Vec<F>,
    challenge: F,
) -> Vec<F> {
    // Note that when the last round ends, every two elements are in
    // the coefficient form. Use the challenge to reduce the two elements
    // into a single value. This is equivalent to substituting the challenge
    // to the first variable of the poly.
    one_level_eval_hc(bh_values, challenge);
    one_level_eval_hc(eq, challenge);

    one_level_interp_hc(eq);
    one_level_interp_hc(bh_values);

    parallel_pi(bh_values, eq)
    // p_i(&bh_values,&eq)
}

pub fn sum_check_last_round<F: Field>(eq: &mut Vec<F>, bh_values: &mut Vec<F>, challenge: F) {
    one_level_eval_hc(bh_values, challenge);
    one_level_eval_hc(eq, challenge);
}

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::util::test::rand_vec;

    pub fn p_i<F: Field>(evals: &[F], eq: &[F]) -> Vec<F> {
        if evals.len() == 1 {
            return vec![evals[0], evals[0], evals[0]];
        }
        // evals coeffs
        let mut coeffs = vec![F::ZERO, F::ZERO, F::ZERO];
        let mut i = 0;
        while i < evals.len() {
            coeffs[0] += evals[i] * eq[i];
            coeffs[1] += evals[i + 1] * eq[i] + evals[i] * eq[i + 1];
            coeffs[2] += evals[i + 1] * eq[i + 1];
            i += 2;
        }

        coeffs
    }

    #[test]
    fn test_sumcheck() {
        let i = 10;
        let mut rng = ChaCha8Rng::from_entropy();
        let evals = rand_vec::<Goldilocks>(1 << i, &mut rng);
        let eq = rand_vec::<Goldilocks>(1 << i, &mut rng);
        let coeffs1 = p_i(&evals, &eq);
        let coeffs2 = parallel_pi(&evals, &eq);
        assert_eq!(coeffs1, coeffs2);
    }
}
