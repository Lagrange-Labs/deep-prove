# Cryptographic Subroutines

This page presents the different base cryptographic techniques used throughout the codebase of deep-prove. The specific implementation of different component (dense etc) can be found in the [layers](./layers.md) page.

- Our [Lookup arguments](./lookups.md) are an optimized version of [logup gkr](https://eprint.iacr.org/2023/1284) specific for our needs. 
- The [commitment schemes](./commitments.md) and specifically the accumulation procedure we use.
- The way deep prove handles [quantization and requantization](./quantization.md)