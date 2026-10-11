//! The `fast` tier of the `f32` kernels.
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f32x/fast.rs`). These are upstream's deliberately
//! low-accuracy routines — about `350 ULP` — kept as upstream writes them, for callers that want
//! speed rather than accuracy. `sin` and `cos` fall back to the 3.5 ULP `u35` tier outside
//! `|x| < 30`.

use core::f32::consts::{FRAC_1_PI, FRAC_PI_2, PI};

use super::*;

/// Fast sine function.
///
/// The error bound of the returned value is `min(350 ULP, 2e-6)`.
#[inline(always)]
pub fn sin<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let t = d;

    let s = d * splat(simd, FRAC_1_PI);
    let u = rint(simd, s);
    let q = roundi(simd, s);
    d = mla(simd, u, splat(simd, -PI), d);

    let s = d * d;

    let mut u = splat(simd, -0.188_174_817_6_e-3);
    u = mla(simd, u, s, splat(simd, 0.832_350_272_7_e-2));
    u = mla(simd, u, s, splat(simd, -0.166_665_136_8));
    u = mla(simd, s * d, u, d);

    u = from_bits(
        simd,
        (q & splat_int(simd, 1))
            .simd_eq(splat_int(simd, 1))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, u),
    );

    let g = t.abs().simd_lt(splat(simd, 30.));
    if !g.all_true() {
        return g.select(u, super::u35::sin(simd, t));
    }

    u
}

/// Fast cosine function.
///
/// The error bound of the returned value is `min(350 ULP, 2e-6)`.
#[inline(always)]
pub fn cos<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let t = d;

    let s = mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5));
    let u = rint(simd, s);
    let q = roundi(simd, s);
    d = mla(simd, u, splat(simd, -PI), d - splat(simd, FRAC_PI_2));

    let s = d * d;

    let mut u = splat(simd, -0.188_174_817_6_e-3);
    u = mla(simd, u, s, splat(simd, 0.832_350_272_7_e-2));
    u = mla(simd, u, s, splat(simd, -0.166_665_136_8));
    u = mla(simd, s * d, u, d);

    u = from_bits(
        simd,
        (q & splat_int(simd, 1))
            .simd_eq(splat_int(simd, 0))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, u),
    );

    let g = t.abs().simd_lt(splat(simd, 30.));
    if !g.all_true() {
        return g.select(u, super::u35::cos(simd, t));
    }

    u
}

/// `ln(d)` at roughly 350 ULP, as used by [`pow`].
///
/// Upstream guards the range reduction with `cfg(feature = "enable_avx512f")`; those branches are
/// not ported.
#[inline(always)]
fn logk3f<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
    d = o.select(d * splat(simd, F1_32 * F1_32), d);
    let e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
    let m = ldexp3k(simd, d, -e);
    let e = o.select(e - splat_int(simd, 64), e);

    let x = (m - splat(simd, 1.)) / (splat(simd, 1.) + m);
    let x2 = x * x;

    let mut t = splat(simd, 0.239_282_846_450_805_664_062_5);
    t = mla(simd, t, x2, splat(simd, 0.285_182_118_415_832_519_531_25));
    t = mla(simd, t, x2, splat(simd, 0.400_005_877_017_974_853_515_625));
    t = mla(simd, t, x2, splat(simd, 0.666_666_686_534_881_591_796_875));
    t = mla(simd, t, x2, splat(simd, 2.));

    mla(
        simd,
        x,
        t,
        splat(simd, 0.693_147_180_559_945_286_226_764) * e.to_float::<F32s<S>>(),
    )
}

/// `exp(d)` at roughly 350 ULP, as used by [`pow`].
#[inline(always)]
fn expk3f<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let q = roundi(simd, d * splat(simd, R_LN2_F));

    let mut s = mla(simd, q.to_float::<F32s<S>>(), splat(simd, -L2U_F), d);
    s = mla(simd, q.to_float::<F32s<S>>(), splat(simd, -L2L_F), s);

    let mut u = splat(simd, 0.000_198_527_617_612_853_646_278_381);
    u = mla(
        simd,
        u,
        s,
        splat(simd, 0.001_393_043_552_525_341_510_772_71),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_333_360_776_305_198_669_433_59),
    );
    u = mla(simd, u, s, splat(simd, 0.041_666_485_369_205_474_853_515_6));
    u = mla(simd, u, s, splat(simd, 0.166_666_671_633_720_397_949_219));
    u = mla(simd, u, s, splat(simd, 0.5));

    u = mla(simd, s * s, u, s + splat(simd, 1.));
    u = ldexp2k(simd, u, q);

    from_bits(
        simd,
        (!d.simd_lt(splat(simd, -104.))).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Fast power function.
///
/// The error bound of the returned value is `350 ULP`.
#[inline(always)]
pub fn pow<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let mut result = expk3f(simd, logk3f(simd, x.abs()) * y);
    let yisint = is_integer(simd, y);
    let yisodd = is_odd(simd, y) & yisint;

    result = (x.is_sign_negative() & yisodd).select(-result, result);

    result = x.simd_eq(splat(simd, 0.)).select(splat(simd, 0.), result);
    y.simd_eq(splat(simd, 0.)).select(splat(simd, 1.), result)
}
