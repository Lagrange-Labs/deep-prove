#![feature(associated_const_equality)]

mod circuit;
pub mod error;
pub mod gadgets;
pub mod macros;
mod prover;
pub mod structs;
#[cfg(feature = "unsafe")]
pub mod unsafe_utils;
pub mod utils;
mod verifier;

pub use sumcheck::util;

#[cfg(test)]
mod test;
