#![deny(clippy::cargo)]
#![feature(decl_macro)]
pub mod macros;
mod prover;
pub mod structs;
pub mod util;
mod verifier;

#[cfg(test)]
mod test;
