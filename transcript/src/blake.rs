use crate::{Challenge, Transcript};
use blake3;
use ff::PrimeField;
use ff_ext::ExtensionField;

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

impl BlakeTranscript {
    /// Append a message to the transcript
    pub fn append_message(&mut self, label: &[u8], message: &[u8]) {
        self.hasher.update(label);
        self.hasher.update(message);
    }

    /// Append a field element to the transcript
    pub fn append_field_element<F: PrimeField>(&mut self, label: &[u8], element: &F) {
        self.append_message(label, element.to_repr().as_ref());
    }

    /// Append a slice of field elements to the transcript
    pub fn append_field_elements<F: PrimeField>(&mut self, label: &[u8], elements: &[F]) {
        let mut bytes = Vec::with_capacity(elements.len() * 32); // Assuming 32 bytes per element
        for element in elements {
            bytes.extend_from_slice(element.to_repr().as_ref());
        }
        self.append_message(label, &bytes);
    }

    /// Generate a challenge by hashing the transcript
    pub fn challenge_bytes(&mut self, label: &[u8], dest: &mut [u8]) {
        self.hasher.update(label);
        let mut output = self.hasher.finalize_xof();
        output.fill(dest);
    }

    /// Generate a field element challenge
    pub fn challenge_field_element<F: PrimeField>(&mut self, label: &[u8]) -> F {
        let mut repr = F::Repr::default();
        // let repr_bytes = &mut repr[..];
        self.challenge_bytes(label, repr.as_mut());
        F::from_repr(repr).unwrap()
    }

    /// Generate a vector of field element challenges
    pub fn challenge_field_elements<F: PrimeField>(&mut self, label: &[u8], n: usize) -> Vec<F> {
        let mut elements = Vec::with_capacity(n);
        for i in 0..n {
            let mut label_i = label.to_vec();
            label_i.extend_from_slice(&i.to_le_bytes());
            elements.push(self.challenge_field_element(&label_i));
        }
        elements
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
        let mut repr = <<E as ExtensionField>::BaseField as PrimeField>::Repr::default();
        self.challenge_bytes(b"challenge1", repr.as_mut());
        let base2 = E::BaseField::from_repr(repr).unwrap();
        self.challenge_bytes(b"challenge2", repr.as_mut());
        let base1 = E::BaseField::from_repr(repr).unwrap();
        Challenge {
            elements: E::from_bases(&[base1, base2]),
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

#[cfg(test)]
mod tests {
    use super::*;
    use goldilocks::GoldilocksExt2;

    #[test]
    fn test_transcript() {
        let mut transcript = BlakeTranscript::new(b"test");

        // Test appending messages
        transcript.append_message(b"msg1", b"hello");
        transcript.append_message(b"msg2", b"world");

        // Test appending field elements
        let element1 = GoldilocksExt2::from(123);
        let element2 = GoldilocksExt2::from(456);
        transcript.append_field_element(b"element1", &element1);
        transcript.append_field_elements(b"elements", &[element1, element2]);

        // Test generating challenges
        let mut challenge_bytes = [0u8; 32];
        transcript.challenge_bytes(b"challenge", &mut challenge_bytes);
        assert_ne!(challenge_bytes, [0u8; 32]);

        let challenge_element =
            transcript.challenge_field_element::<GoldilocksExt2>(b"challenge_element");
        assert_ne!(challenge_element, GoldilocksExt2::ZERO);

        let challenge_element2 =
            transcript.challenge_field_element::<GoldilocksExt2>(b"challenge_element");
        assert_ne!(challenge_element2, GoldilocksExt2::ZERO);
        assert_ne!(challenge_element, challenge_element2);

        let challenge_elements =
            transcript.challenge_field_elements::<GoldilocksExt2>(b"challenge_elements", 3);
        assert_eq!(challenge_elements.len(), 3);
        for element in &challenge_elements {
            assert_ne!(*element, GoldilocksExt2::ZERO);
        }
    }
}
