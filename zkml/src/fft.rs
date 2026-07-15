use anyhow::ensure;
use ark_ff::PrimeField;
use rayon::{
    iter::{IntoParallelRefMutIterator, ParallelIterator},
    prelude::ParallelSliceMut,
};

use crate::get_root_of_unity;

/// Returns a permutation to convert a vector from normal order to bit reverse
/// order.
///
/// This can be used to start a FFT using decimation in time or to finalise a
/// FFT with decimation in frequency.
fn bitreverse_permutation(length: usize) -> impl Iterator<Item = usize> {
    let shift = usize::BITS - length.ilog2();
    (0..length).map(move |i| i.reverse_bits() >> shift)
}

/// Applies a bitreverse order to the slice.
///
/// This can be used to start a FFT using decimation in time or to finalise a
/// FFT with decimation in frequency.
fn bitreverse<T>(d: &mut [T]) {
    for (orig, new) in bitreverse_permutation(d.len())
        .enumerate()
        // filter out duplicates
        .filter(|(orig, new)| orig < new)
    {
        d.swap(orig, new)
    }
}

/// Perform a radix-2 Cooley-Tukey FFT.
///
/// flag: false -> FFT
/// flag: true -> iFFT
pub fn fft<F: PrimeField + Send + Sync>(v: &mut Vec<F>, flag: bool) -> anyhow::Result<()> {
    ensure!(
        v.len().is_power_of_two(),
        "Input vector to fft must be a power of two",
    );

    // A length-1 FFT is mathematically the identity transform, so handling
    // this case with an early return preserves correctness while avoiding
    // the panic that the general code path below would otherwise hit for
    // n == 1: it unconditionally writes to `twiddle[1]`, even though
    // `twiddle` has length 1 in that case, and `bitreverse_permutation`
    // computes a right-shift of `usize::BITS - length.ilog2()` bits, which
    // equals a full-width shift and panics under overflow-checks.
    if v.len() == 1 {
        return Ok(());
    }

    let n = v.len();
    let logn = n.ilog2();

    // Perform bit reverse permutation. The code below performs decimation in
    // time (DIT), data is reordered prior to the butterflies.
    bitreverse(v);

    // Compute the twiddle factors
    let mut twiddle: Vec<F> = vec![F::ZERO; n];
    twiddle[0] = F::ONE;
    twiddle[1] = get_root_of_unity(logn as usize)?;

    if flag {
        twiddle[1] = twiddle[1]
            .inverse()
            .expect("Root of unity should not be zero");
    }
    for i in 2..n {
        twiddle[i] = twiddle[i - 1] * twiddle[1];
    }

    let mut i: usize = 2;
    while i <= n {
        v.par_chunks_mut(i).for_each(|chunk| {
            let half_i = i >> 1;
            for k in 0..half_i {
                let u = chunk[k];
                let l = chunk[k + half_i] * twiddle[n / i * k];
                chunk[k] = u + l;
                chunk[k + half_i] = u - l;
            }
        });
        i <<= 1;
    }

    if flag {
        let mut ilen = F::from(n as u64);
        ilen = ilen.inverse().expect("Tried to invert zero");
        debug_assert_eq!(ilen * F::from(n as u64), F::ONE, "Error in inv");
        v.par_iter_mut().for_each(|val| {
            *val *= ilen;
        });
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use itertools::Itertools;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{
        fft::{bitreverse, bitreverse_permutation, fft},
        testing::random_vector,
    };

    /// Minimal PoC: `1usize.is_power_of_two() == true`, so `fft()`'s stated
    /// precondition (`ensure!(v.len().is_power_of_two())`) admits a
    /// length-1 input. Before the fix, this panicked with an out-of-bounds
    /// write to `twiddle[1]` (`twiddle` has length 1 for n == 1). A
    /// length-1 FFT is mathematically the identity transform.
    #[test]
    fn fft_single_element_should_not_panic() {
        let mut v = vec![Fr::from(1u64)];

        fft(&mut v, false).unwrap();

        assert_eq!(v, vec![Fr::from(1u64)]);
    }

    /// Same PoC for the inverse transform (`flag == true`).
    #[test]
    fn ifft_single_element_should_not_panic() {
        let mut v = vec![Fr::from(1u64)];

        fft(&mut v, true).unwrap();

        assert_eq!(v, vec![Fr::from(1u64)]);
    }

    #[test]
    fn test_bitreverse_permutation() {
        for logn in 1..10 {
            let n = 1 << logn;

            let mut expected: Vec<usize> = vec![0; n];
            for i in 1..n {
                expected[i] = expected[i >> 1] >> 1 | (i & 1) << (logn - 1);
            }

            for (n, i) in expected.iter().zip(bitreverse_permutation(n)) {
                assert_eq!(*n, i);
            }
        }
    }

    #[test]
    fn test_bitreverse() {
        for logn in 1..10 {
            let size = 1 << logn;
            let mut data = random_vector(size);

            let mut expected = data.clone();
            let bit_reverse = bitreverse_permutation(size).collect_vec();
            let permutation: Vec<(usize, usize)> = (0..size)
                .into_par_iter()
                .filter_map(|i| {
                    if bit_reverse[i] < i {
                        Some((i, bit_reverse[i]))
                    } else {
                        None
                    }
                })
                .collect();

            for (i, j) in permutation {
                expected.swap(i, j);
            }

            bitreverse(&mut data);
            assert_eq!(expected, data);
        }
    }
}
