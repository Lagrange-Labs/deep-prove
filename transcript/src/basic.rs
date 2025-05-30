use ff_ext::{ExtensionField, PoseidonField};
use poseidon::challenger::{CanObserve, DefaultChallenger, FieldChallenger};

use crate::{Challenge, ForkableTranscript, Transcript};
use ff_ext::SmallField;

#[derive(Clone)]
pub struct BasicTranscript<E: ExtensionField> {
    challenger: DefaultChallenger<E::BaseField, <E::BaseField as PoseidonField>::T>,
}

impl<E: ExtensionField> BasicTranscript<E> {
    /// Create a new IOP transcript.
    pub fn new(label: &'static [u8]) -> Self {
        let mut challenger = DefaultChallenger::<E::BaseField, <E::BaseField as PoseidonField>::T>::new_poseidon_default();
        let label_f = E::BaseField::bytes_to_field_elements(label);
        challenger.observe_slice(label_f.as_slice());
        Self { challenger }
    }
}

impl<E: ExtensionField> Transcript<E> for BasicTranscript<E> {
    fn append_field_elements(&mut self, elements: &[E::BaseField]) {
        self.challenger.observe_slice(elements);
    }

    fn append_field_element_ext(&mut self, element: &E) {
        self.challenger.observe_ext_element(*element);
    }

    fn read_challenge(&mut self) -> Challenge<E> {
        Challenge {
            elements: self.challenger.sample_ext_element(),
        }
    }

    fn read_field_element_exts(&self) -> Vec<E> {
        unimplemented!()
    }

    fn read_field_element(&self) -> E::BaseField {
        unimplemented!()
    }

    fn send_challenge(&self, _challenge: E) {
        unimplemented!()
    }

    fn commit_rolling(&mut self) {
        // do nothing
    }
}

impl<E: ExtensionField> ForkableTranscript<E> for BasicTranscript<E> {}
