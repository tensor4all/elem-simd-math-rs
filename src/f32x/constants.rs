//! Constants and broadcast helpers for the `f32` kernels.

pub use crate::constants::*;
// `f64x::constants` re-exports `REMPITABDP` with `pub use`. `REMPITABSP` has no doc comment in
// `src/tables.rs`, so a `pub use` here would make an undocumented constant publicly reachable and
// trip `missing_docs`; a crate-visible re-export is enough for the `f32` kernels, which are the
// only consumer of the table.
pub(crate) use crate::tables::REMPITABSP;

use fearless_simd::prelude::*;
use fearless_simd::Simd;

use crate::common::{F32s, I32s, U32s};

/// Broadcasts an `f32` scalar.
#[inline(always)]
pub fn splat<S: Simd>(simd: S, x: f32) -> F32s<S> {
    F32s::<S>::splat(simd, x)
}

/// Broadcasts a `u32` scalar.
#[inline(always)]
pub fn splat_bits<S: Simd>(simd: S, x: u32) -> U32s<S> {
    U32s::<S>::splat(simd, x)
}

/// Broadcasts an `i32` scalar.
#[inline(always)]
pub fn splat_int<S: Simd>(simd: S, x: i32) -> I32s<S> {
    I32s::<S>::splat(simd, x)
}
