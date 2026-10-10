//! `elem-simd-math` — vectorized elementwise math for `f32` and `f64`.
//!
//! This crate is a port of [SLEEF] as expressed by the [sleef-rs] crate, rewritten against the
//! stable-Rust SIMD abstraction of [`fearless_simd`] so that it builds on stable Rust with
//! runtime CPU feature detection. See `NOTICE` and `THIRD-PARTY-LICENSES` for the provenance and
//! licensing of the derived portions.
//!
//! [SLEEF]: https://sleef.org/
//! [sleef-rs]: https://github.com/burrbull/sleef-rs
//!
//! # Layers
//!
//! * [`f64x`] and [`f32x`] expose the ported kernels as functions generic over a
//!   [`fearless_simd::Simd`] token, operating on native-width vectors. Use these when you
//!   already own the SIMD loop.
//! * [`ElementwiseMath`] exposes slice entry points that pick the best available SIMD level at
//!   runtime, with a scalar fallback.
//!
//! # Accuracy
//!
//! Kernels come from SLEEF's four accuracy tiers and each function documents its tier's error
//! bound (`u05` = 0.5 ULP, `u10` = 1.0 ULP, `u15` = 1.5 ULP, `u35` = 3.5 ULP). Those bounds are
//! asserted by the ported upstream test suites against MPFR. They bound the vector body; the
//! value produced for inputs outside a kernel's tested domain is documented per function.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// The ported coefficients are written out to upstream's full decimal expansion. Shortening them
// to the shortest round-tripping form would change the constants, and several are deliberately
// not the nearest `f64` to a mathematical constant (they are the range-reduction splits).
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

pub mod common;
pub mod constants;
pub mod doubled;
pub mod f32x;
pub mod f64x;
pub(crate) mod tables;

mod dispatch;

pub use dispatch::{ElementwiseMath, LengthMismatch};

/// The `f64` slice entry points are also available as free functions here.
pub mod f64 {
    pub use crate::dispatch::f64::*;
}

/// The `f32` slice entry points are also available as free functions here.
pub mod f32 {
    pub use crate::dispatch::f32::*;
}
