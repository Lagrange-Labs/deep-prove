#![deny(clippy::cargo)]
//! This repo is not properly implemented
//! Transcript APIs are placeholders; the actual logic is to be implemented later.

pub mod basic;
pub mod blake;
mod statistics;
pub mod synchronized;
pub use basic::BasicTranscript;
use ff_ext::SmallField;
use p3_field::FieldAlgebra;
pub use statistics::{BasicTranscriptWithStat, StatisticRecorder};
pub use synchronized::TranscriptSynchronized;

#[derive(Default, Copy, Clone, Eq, PartialEq, Debug)]
pub struct Challenge<F> {
    pub elements: F,
}

use ff_ext::ExtensionField;
/// The Transcript trait
pub trait Transcript<E: ExtensionField>: Clone {
    /// Append a slice of base field elements to the transcript.
    ///
    /// An implementation has to provide at least one of
    /// `append_field_elements` / `append_field_element`
    fn append_field_elements(&mut self, elements: &[E::BaseField]) {
        for e in elements {
            self.append_field_element(e);
        }
    }

    /// Append a single field element to the transcript.
    ///
    /// An implementation has to provide at least one of
    /// `append_field_elements` / `append_field_element`
    fn append_field_element(&mut self, element: &E::BaseField) {
        self.append_field_elements(&[*element])
    }

    /// Append a message to the transcript.
    fn append_message(&mut self, msg: &[u8]) {
        let msg_f = E::BaseField::bytes_to_field_elements(msg);
        self.append_field_elements(&msg_f);
    }

    /// Append an extension field element to the transcript.
    ///
    /// An implementation has to override at least one of
    /// `append_field_element_ext` / `append_field_element_exts`
    fn append_field_element_ext(&mut self, element: &E) {
        self.append_field_element_exts(&[*element])
    }

    /// Append a slice of extension field elements to the transcript.
    ///
    /// An implementation has to override at least one of
    /// `append_field_element_ext` / `append_field_element_exts`
    fn append_field_element_exts(&mut self, element: &[E]) {
        for e in element {
            self.append_field_element_ext(e);
        }
    }

    /// Append a challenge to the transcript.
    fn append_challenge(&mut self, challenge: Challenge<E>) {
        self.append_field_element_ext(&challenge.elements)
    }

    /// Generate a challenge from the current transcript
    /// and append it to the transcript.
    ///
    /// The output field element is statistical uniform as long
    /// as the field has a size less than 2^384.
    fn get_and_append_challenge(&mut self, label: &'static [u8]) -> Challenge<E> {
        self.append_message(label);
        self.read_challenge()
    }

    fn read_field_element_ext(&self) -> E {
        self.read_field_element_exts()[0]
    }

    fn read_field_element_exts(&self) -> Vec<E>;

    fn read_field_element(&self) -> E::BaseField;

    fn read_challenge(&mut self) -> Challenge<E>;

    fn send_challenge(&self, challenge: E);

    fn commit_rolling(&mut self);
}

/// Forkable Transcript trait, enable fork method
pub trait ForkableTranscript<E: ExtensionField>: Transcript<E> + Sized + Clone {
    /// Fork this transcript into n different threads.
    fn fork(self, n: usize) -> Vec<Self> {
        (0..n)
            .map(|i| {
                let mut fork = self.clone();
                fork.append_field_element(&E::BaseField::from_canonical_u64(i as u64));
                fork
            })
            .collect()
    }
}
