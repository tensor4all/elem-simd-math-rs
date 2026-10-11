//! Type aliases for the `f64` kernel family, and the translation table they follow.
//!
//! `sleef-rs` expresses these as extension traits on `Simd<f64, N>`/`Simd<f32, N>`. Extension
//! traits cannot be implemented for `S::f64s` here (the projection is not a local type, and two
//! such impls conflict), so they are free functions parameterised by the SIMD token. The
//! translation table used throughout this crate is:
//!
//! | `sleef-rs` | `elem-simd-math` |
//! |---|---|
//! | `F64x<N>` | `F64s<S>` |
//! | `x.mla(y, z)` | `x.mul_add(y, z)` |
//! | `F64x::splat(v)` | `F64s::<S>::splat(simd, v)` |
//! | `F64x::poly8(…)` | `poly8(simd, …)` |
//! | `m.select_splat(l, r)` | `select_splat(m, l, r)` |
//! | `m.select(l, r)` | `m.select(l, r)` |
//! | `x.to_bits()` / `F64x::from_bits(b)` | `to_bits(x)` / `from_bits(b)` |

use fearless_simd::Simd;

/// Native-width `f64` vector for SIMD level `S`.
pub type F64s<S> = <S as Simd>::f64s;
/// Native-width `u64` vector for SIMD level `S`.
pub type U64s<S> = <S as Simd>::u64s;
/// Native-width `i64` vector for SIMD level `S`.
pub type I64s<S> = <S as Simd>::i64s;
/// Native-width mask for `f64` lanes.
pub type M64s<S> = <S as Simd>::mask64s;
