//! Fused activation kernels: one vector pass per element.
//!
//! These compose the tier kernels of this family the way PyTorch's CPU
//! `Vectorized<float>` activations do, so a caller gets `sigmoid`, `silu`,
//! `softplus`, `gelu` and `gelu_tanh` without materializing the intermediate
//! `exp`/`erf`/`tanh` results. The formulas are the overflow-free ones:
//! `exp(-x)` overflows only to `+inf`, which the division turns into `0`, and
//! `softplus` evaluates its logarithm on `exp(-|x|) <= 1`.
//!
//! Infinite inputs follow the formulas: `sigmoid` and `softplus` reach their
//! limits, while `silu` and both GELU forms produce `NaN` at `-inf`
//! (`-inf * 0`), exactly as a composite evaluation does.
//!
//! Each function documents its own error bound. The bounds are not SLEEF
//! tiers: they follow from the tier kernel inside and the few roundings
//! around it, and they are asserted against MPFR in `tests/ulp.rs`.

use super::*;

/// `sqrt(2 / pi)`, the scale inside the GELU tanh approximation.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// Cubic coefficient of the GELU tanh approximation.
const GELU_TANH_CUBIC: f64 = 0.044_715;

/// Logistic sigmoid `1 / (1 + exp(-x))`, in the overflow-free form
/// `1 / (1 + e)` for `x > 0` and `e / (1 + e)` otherwise, with
/// `e = exp(-|x|) <= 1`: no intermediate overflows, and the far negative tail
/// keeps its subnormal values instead of flushing to zero.
///
/// The error bound of the returned value is `2.5 ULP`: the 1.0 ULP `exp`,
/// then one addition and one division.
#[inline(always)]
pub fn sigmoid<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let one = splat(simd, 1.);
    let e = u10::exp(simd, -x.abs());
    let numerator = x.simd_gt(splat(simd, 0.)).select(one, e);
    numerator / (one + e)
}

/// SiLU / swish `x * sigmoid(x)`.
///
/// The error bound of the returned value is `3.5 ULP` ([`sigmoid`] plus one
/// multiplication) wherever `exp(-|x|)` is a normal number; below that
/// (`x < -87` for `f32`, `x < -708` for `f64`) the subnormal exponential
/// carries fewer bits and so does the result.
#[inline(always)]
pub fn silu<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    x * sigmoid(simd, x)
}

/// Softplus `log(1 + exp(x))`, evaluated as `max(x, 0) + log1p(exp(-|x|))`.
///
/// The error bound of the returned value is `3.0 ULP`: the 1.0 ULP `exp` feeds
/// the 1.0 ULP `log1p`, whose argument is at most `1`, then one addition.
#[inline(always)]
pub fn softplus<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let zero = splat(simd, 0.);
    let tail = u10::log1p(simd, u10::exp(simd, -x.abs()));
    x.simd_gt(zero).select(x, zero) + tail
}

/// Exact-form GELU `x/2 * (1 + erf(x / sqrt(2)))` with the 1.0 ULP `erf`.
///
/// The returned value is within `3.0 ULP` of the formula evaluated exactly for
/// `x >= 0`; for `x < 0` the `1 + erf` cancellation bounds the relative error,
/// as in the composite form.
#[inline(always)]
pub fn gelu<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let half = splat(simd, 0.5);
    let z = x * splat(simd, core::f64::consts::FRAC_1_SQRT_2);
    half * x * (splat(simd, 1.) + u10::erf(simd, z))
}

/// GELU tanh approximation `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
///
/// The hyperbolic tangent is the 1.0 ULP kernel. The returned value is within
/// `3.0 ULP` of the formula evaluated exactly for `x >= 0`; for `x < 0` the
/// `1 + tanh` cancellation bounds the relative error, as in the composite form.
#[inline(always)]
pub fn gelu_tanh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let half = splat(simd, 0.5);
    let cubic = mla(simd, splat(simd, GELU_TANH_CUBIC) * x, x * x, x);
    let inner = u10::tanh(simd, splat(simd, SQRT_2_OVER_PI) * cubic);
    half * x * (splat(simd, 1.) + inner)
}
