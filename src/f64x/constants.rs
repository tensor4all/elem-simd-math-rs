//! Constants and broadcast helpers for the `f64` kernels.

pub use crate::constants::*;
pub use crate::tables::REMPITABDP;

use fearless_simd::prelude::*;
use fearless_simd::Simd;

use crate::common::{F64s, I64s, U64s};

/// Broadcasts an `f64` scalar.
#[inline(always)]
pub fn splat<S: Simd>(simd: S, x: f64) -> F64s<S> {
    F64s::<S>::splat(simd, x)
}

/// Broadcasts a `u64` scalar.
#[inline(always)]
pub fn splat_bits<S: Simd>(simd: S, x: u64) -> U64s<S> {
    U64s::<S>::splat(simd, x)
}

/// Broadcasts an `i64` scalar.
#[inline(always)]
pub fn splat_int<S: Simd>(simd: S, x: i64) -> I64s<S> {
    I64s::<S>::splat(simd, x)
}
