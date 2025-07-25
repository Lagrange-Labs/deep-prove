//! File containing code for generating the LogUp GKR circuit

use ark_std::Zero;
use ff_ext::ExtensionField;

use std::sync::Arc;

use itertools::izip;
use multilinear_extensions::{mle::DenseMultilinearExtension, util::ceil_log2};

use super::structs::Fraction;

#[derive(Debug, Clone)]
/// The different variants of layer that can be found in the LogUp GKR protocol.
pub enum LogUpLayer<E: ExtensionField> {
    /// This variant is used for every layer apart from the first in the LogUp protocol.
    Generic {
        numerator: Vec<E>,
        denominator: Vec<E>,
    },
    /// This is the first layer of the GKR protocol when proving a fractional sumcheck for a table.
    /// The numerator is the multiplicity polynomial and the denominator is the merged table polynomial.
    InitialTable {
        numerator: Vec<E>,
        denominator: Vec<E>,
    },
    /// This is the first layer of the GKR protocol when proving a fractional sumcheck for a set of lookups.
    /// The numerators will all be `-1` on this layer so we only store the denominator which is the merged lookups.
    InitialLookup { denominator: Vec<E> },
}

impl<E: ExtensionField> LogUpLayer<E> {
    /// Returns the number of variables that the [`DenseMultilinearExtension`]s at this layer have
    pub fn num_vars(&self) -> usize {
        // We right shift the denominator length by 1 because at each level of the GKR circuit the polynomials we run the sumcheck over have half the length
        match self {
            LogUpLayer::Generic { denominator, .. }
            | LogUpLayer::InitialTable { denominator, .. }
            | LogUpLayer::InitialLookup { denominator } => ceil_log2(denominator.len() >> 1),
        }
    }

    /// Function used to check if this is the final layer
    pub fn is_final_layer(&self) -> bool {
        self.num_vars().is_zero()
    }

    /// Produces the next layer of the GKR circuit (if there is meant to be one)
    pub fn next_layer(&self) -> Option<LogUpLayer<E>> {
        // First we check to see if we are at the final layer, if so return None
        if self.is_final_layer() {
            return None;
        }

        let half_layer_size = 1 << self.num_vars();

        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            }
            | LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => {
                // Split the numerator and denominator at the halfway point and sum the fractions
                let (num1, num2) = numerator.split_at(half_layer_size);
                let (denom1, denom2) = denominator.split_at(half_layer_size);

                let (next_numerator, next_denominator): (Vec<E>, Vec<E>) =
                    izip!(num1, denom1, num2, denom2)
                        .map(|(n1, d1, n2, d2)| {
                            (Fraction::<E>::new(*n1, *d1) + Fraction::<E>::new(*n2, *d2)).as_tuple()
                        })
                        .unzip();

                Some(LogUpLayer::Generic {
                    numerator: next_numerator,
                    denominator: next_denominator,
                })
            }
            LogUpLayer::InitialLookup { denominator } => {
                // In this case we only need to split the denominator polynomial in half as the numerators are all -1
                let (denom1, denom2) = denominator.split_at(half_layer_size);
                let (next_numerator, next_denominator): (Vec<E>, Vec<E>) = denom1
                    .iter()
                    .zip(denom2.iter())
                    .map(|(d1, d2)| {
                        (Fraction::<E>::new(-E::ONE, *d1) + Fraction::<E>::new(-E::ONE, *d2))
                            .as_tuple()
                    })
                    .unzip();

                Some(LogUpLayer::Generic {
                    numerator: next_numerator,
                    denominator: next_denominator,
                })
            }
        }
    }

    /// Gets the output values in order (`numerator`, `denominator`) if its the output layer, returns `None` otherwise.
    pub fn output_values(&self) -> Option<(E, E)> {
        match (self, self.is_final_layer()) {
            (
                LogUpLayer::Generic {
                    numerator,
                    denominator,
                },
                true,
            ) => Some((
                numerator[0] * denominator[1] + numerator[1] * denominator[0],
                denominator[0] * denominator[1],
            )),
            _ => None,
        }
    }

    /// Returns all evals at this layer in order numerators , denominators
    pub fn flat_evals(&self) -> Vec<E> {
        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            }
            | LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => [numerator.as_slice(), denominator.as_slice()].concat(),
            LogUpLayer::InitialLookup { denominator } => denominator.to_vec(),
        }
    }

    /// Gets the Densemultlinear extensions for this [`LogUpLayer`] in the order
    /// numerator low part, numerator high part, denominator low part, denominator high part.
    /// In the initial lookup case it is just the two denominator MLEs.
    pub fn get_mles(&self) -> Vec<Arc<DenseMultilinearExtension<E>>> {
        let num_vars = self.num_vars();
        let half_layer_size = 1 << num_vars;
        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            }
            | LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => {
                let (num_low, num_high) = numerator.split_at(half_layer_size);
                let (denom_low, denom_high) = denominator.split_at(half_layer_size);
                let num_low_mle = Arc::new(
                    DenseMultilinearExtension::<E>::from_evaluations_ext_slice(num_vars, num_low),
                );
                let num_high_mle = Arc::new(
                    DenseMultilinearExtension::<E>::from_evaluations_ext_slice(num_vars, num_high),
                );
                let denom_low_mle = Arc::new(
                    DenseMultilinearExtension::<E>::from_evaluations_ext_slice(num_vars, denom_low),
                );
                let denom_high_mle =
                    Arc::new(DenseMultilinearExtension::<E>::from_evaluations_ext_slice(
                        num_vars, denom_high,
                    ));

                vec![num_low_mle, num_high_mle, denom_low_mle, denom_high_mle]
            }
            LogUpLayer::InitialLookup { denominator } => {
                let (denom_low, denom_high) = denominator.split_at(half_layer_size);
                let denom_low_mle = Arc::new(
                    DenseMultilinearExtension::<E>::from_evaluations_ext_slice(num_vars, denom_low),
                );
                let denom_high_mle =
                    Arc::new(DenseMultilinearExtension::<E>::from_evaluations_ext_slice(
                        num_vars, denom_high,
                    ));

                vec![denom_low_mle, denom_high_mle]
            }
        }
    }
}

#[derive(Clone, Debug)]
/// A LogUp GKR circuit stored as its evaluations on each layer.
pub struct LogUpCircuit<E: ExtensionField> {
    /// The [`LogUpLayer`]s forming this circuit, from bottom to top
    layers: Vec<LogUpLayer<E>>,
}

impl<E: ExtensionField> LogUpCircuit<E> {
    /// Creates a new circuit from a [`LogUpLayer`]
    pub fn new(initial_layer: LogUpLayer<E>) -> LogUpCircuit<E> {
        let layers = std::iter::successors(Some(initial_layer), |layer| layer.next_layer())
            .collect::<Vec<LogUpLayer<E>>>();
        LogUpCircuit { layers }
    }

    /// Crates a new [`LogUpCircuit`] for a lookup variant, the inputs to this function are:
    /// - `lookup_evals`, a series of slices of equal length that represent the columns we are looking up
    /// - `constant_challenge`, the first term of the LogUp denominator
    /// - `column_separation_challenge`, the challenge used as domain separation for each of the columns
    pub fn new_lookup_circuit(
        lookup_columns: &[Vec<E::BaseField>],
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> LogUpCircuit<E> {
        let challenge_powers = std::iter::successors(Some(E::ONE), |prev| {
            Some(*prev * column_separation_challenge)
        })
        .take(lookup_columns.len())
        .collect::<Vec<E>>();

        let length = lookup_columns[0].len();

        let denominator = (0..length)
            .map(|i| {
                lookup_columns
                    .iter()
                    .zip(challenge_powers.iter())
                    .fold(constant_challenge, |acc, (col, challenge)| {
                        acc + *challenge * col[i]
                    })
            })
            .collect::<Vec<E>>();

        let initial_layer = LogUpLayer::InitialLookup { denominator };

        LogUpCircuit::<E>::new(initial_layer)
    }

    /// Crates a new [`LogUpCircuit`] for a table variant, the inputs to this function are:
    /// - `table_evals`, a series of slices of equal length that represent the columns of the table,
    /// - `multiplicities`, a slice of [`E::BaseField`] elements that are the evaluations of the multiplicity poly,
    /// - `constant_challenge`, the first term of the LogUp denominator,
    /// - `column_separation_challenge`, the challenge used as domain separation for each of the columns,
    pub fn new_table_circuit(
        table_columns: &[Vec<E::BaseField>],
        multiplicities: &[E::BaseField],
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> LogUpCircuit<E> {
        let challenge_powers = std::iter::successors(Some(E::ONE), |prev| {
            Some(*prev * column_separation_challenge)
        })
        .take(table_columns.len())
        .collect::<Vec<E>>();

        let length = table_columns[0].len();
        let numerator = multiplicities
            .iter()
            .map(|&val| E::from(val))
            .collect::<Vec<E>>();
        let denominator = (0..length)
            .map(|i| {
                table_columns
                    .iter()
                    .zip(challenge_powers.iter())
                    .fold(constant_challenge, |acc, (col, challenge)| {
                        acc + *challenge * col[i]
                    })
            })
            .collect::<Vec<E>>();

        let initial_layer = LogUpLayer::InitialTable {
            numerator,
            denominator,
        };

        LogUpCircuit::<E>::new(initial_layer)
    }

    /// Retrieves the output values of the circuit in the order `numerator`, `denominator`
    pub fn outputs(&self) -> Vec<E> {
        self.layers.last().map(|layer| layer.flat_evals()).unwrap()
    }

    /// Works out the total number of variables the [`LogUpCircuit`] has in its
    /// largest (aka input) layer.
    pub fn num_vars(&self) -> usize {
        self.layers[0].num_vars()
    }

    /// Getter for the layers.
    pub fn layers(&self) -> &[LogUpLayer<E>] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::random_field_vector;
    use ff_ext::GoldilocksExt2;
    use p3_field::{Field, FieldAlgebra};
    use p3_goldilocks::Goldilocks;

    use super::*;

    #[test]
    fn test_circuit_construction() {
        let column = random_field_vector::<Goldilocks>(1 << 10);

        let circuit = LogUpCircuit::<GoldilocksExt2>::new_lookup_circuit(
            &[column.clone()],
            GoldilocksExt2::ZERO,
            GoldilocksExt2::ONE,
        );

        let out = circuit.outputs();

        assert_eq!(out.len(), 4);

        let value = column.iter().fold(GoldilocksExt2::ZERO, |acc, val| {
            let inv = val.inverse();
            acc - GoldilocksExt2::from(inv)
        });

        let out_num = out[0] * out[3] + out[1] * out[2];
        let out_denom = out[2] * out[3];
        let calculated = out_num * out_denom.inverse();

        assert_eq!(value, calculated);
    }
}
