use crate::{Challenge, Transcript};
use blake3;
use ff_ext::ExtensionField;
use p3_field::PrimeField;

pub struct Digest(pub [u8; 32]);

/// A transcript implementation using BLAKE3
#[derive(Clone, Debug)]
pub struct BlakeTranscript {
    /// The BLAKE3 hasher
    hasher: blake3::Hasher,
}

impl BlakeTranscript {
    /// Create a new transcript
    pub fn new(label: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(label);
        Self { hasher }
    }
    pub fn new_empty() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }
}

fn prime_to_bytes<F: PrimeField>(element: &F) -> Vec<u8> {
    element.as_canonical_biguint().to_bytes_le()
}

impl BlakeTranscript {
    /// Append a message to the transcript
    pub fn append_message(&mut self, label: &[u8], message: &[u8]) {
        self.hasher.update(label);
        self.hasher.update(message);
    }

    pub fn append_field_element<F: PrimeField>(&mut self, label: &[u8], element: &F) {
        self.append_message(label, &prime_to_bytes(element));
    }

    /// Append a slice of field elements to the transcript
    pub fn append_field_elements<F: PrimeField>(&mut self, label: &[u8], elements: &[F]) {
        let mut bytes = Vec::with_capacity(elements.len() * 32); // Assuming 32 bytes per element
        for element in elements {
            bytes.extend_from_slice(&prime_to_bytes(element));
        }
        self.append_message(label, &bytes);
    }

    /// Generate a challenge by hashing the transcript
    pub fn challenge_bytes(&mut self, label: &[u8], dest: &mut [u8]) {
        self.hasher.update(label);
        let mut output = self.hasher.finalize_xof();
        output.fill(dest);
    }
}

impl<E: ExtensionField> Transcript<E> for BlakeTranscript {
    fn append_field_elements(&mut self, elements: &[E::BaseField]) {
        for element in elements {
            self.append_field_element(b"field_element", element);
        }
    }

    fn append_field_element_ext(&mut self, element: &E) {
        self.append_field_elements(b"field_element_ext", element.as_bases());
    }

    fn read_challenge(&mut self) -> Challenge<E> {
        let e = E::from_uniform_bytes(|dst| {
            self.challenge_bytes(b"challenge", dst);
        });
        Challenge { elements: e }
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

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;
    use p3_field::FieldAlgebra;
    use p3_goldilocks::Goldilocks;

    use super::*;

    fn test_challenge<E: ExtensionField, T: Transcript<E>>(transcript: &mut T) {
        let c1 = transcript.read_challenge();
        let c2 = transcript.read_challenge();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_transcript() {
        let mut transcript = BlakeTranscript::new(b"test");

        // Test appending messages
        transcript.append_message(b"msg1", b"hello");
        transcript.append_message(b"msg2", b"world");

        // Test appending field elements
        let element1 = GoldilocksExt2::new(Goldilocks::ONE, Goldilocks::ZERO);
        let element2 = GoldilocksExt2::new(Goldilocks::ONE, Goldilocks::ONE);
        transcript.append_field_element_ext(&element1);
        transcript.append_field_element_ext(&element2);
        test_challenge::<GoldilocksExt2, _>(&mut transcript);

        // Test generating challenges
        let mut challenge_bytes = [0u8; 32];
        transcript.challenge_bytes(b"challenge", &mut challenge_bytes);
        assert_ne!(challenge_bytes, [0u8; 32]);
    }
}
