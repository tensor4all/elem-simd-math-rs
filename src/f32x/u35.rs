//! `f32` kernels with a 3.5 ULP error bound (SLEEF's `u35` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f32x/u35.rs`, and `expm1fk` from `src/f32x.rs`).

use core::f32::consts::{FRAC_1_PI, FRAC_2_PI, FRAC_PI_2, FRAC_PI_4, PI};

use super::*;
use crate::doubled::f32::add_checked_v;

/// `exp(x) - 1`, used internally where the result is scaled by a known factor.
#[inline(always)]
pub fn expm1fk<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let q = roundi(simd, d * splat(simd, R_LN2_F));
    let qf = q.to_float::<F32s<S>>();

    let s = mla(simd, qf, splat(simd, -L2U_F), d);
    let s = mla(simd, qf, splat(simd, -L2L_F), s);

    let s2 = s * s;
    let s4 = s2 * s2;

    let u = poly6(
        simd,
        s,
        s2,
        s4,
        0.000_198_527_617_612_853_646_278_381,
        0.001_393_043_552_525_341_510_772_71,
        0.008_333_360_776_305_198_669_433_59,
        0.041_666_485_369_205_474_853_515_6,
        0.166_666_671_633_720_397_949_219,
        0.5,
    );

    let u = mla(simd, s * s, u, s);

    q.simd_eq(splat_int(simd, 0))
        .select(u, ldexp2k(simd, u + splat(simd, 1.), q) - splat(simd, 1.))
}

/// Sine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn sin<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let mut q;
    let u;
    let r = d;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        q = roundi(simd, d * splat(simd, FRAC_1_PI));
        u = q.to_float::<F32s<S>>();
        d = mla(simd, u, splat(simd, -PI_A2_F), d);
        d = mla(simd, u, splat(simd, -PI_B2_F), d);
        d = mla(simd, u, splat(simd, -PI_C2_F), d);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX_F)).all_true() {
        q = roundi(simd, d * splat(simd, FRAC_1_PI));
        u = q.to_float::<F32s<S>>();
        d = mla(simd, u, splat(simd, -PI_A_F), d);
        d = mla(simd, u, splat(simd, -PI_B_F), d);
        d = mla(simd, u, splat(simd, -PI_C_F), d);
        d = mla(simd, u, splat(simd, -PI_D_F), d);
    } else {
        let (mut dfidf, dfii) = rempi(simd, d);
        q = dfii & splat_int(simd, 3);
        q = q
            + q
            + dfidf
                .0
                .simd_gt(splat(simd, 0.))
                .select(splat_int(simd, 2), splat_int(simd, 1));
        q >>= 2;
        let o = (dfii & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
        let mut x = DD::new(
            mul_sign(simd, splat(simd, -D_PI_F.0 * 0.5), dfidf.0),
            mul_sign(simd, splat(simd, -D_PI_F.1 * 0.5), dfidf.0),
        );
        x = dfidf + x;
        dfidf = select_doubled(simd, o, x, dfidf);
        d = dfidf.0 + dfidf.1;

        d = from_bits(
            simd,
            (r.is_infinite() | r.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, d),
        );
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((q & splat_int(simd, 1)).simd_eq(splat_int(simd, 1)))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, d),
    );

    let mut u = splat(simd, 2.608_315_980_978_659_354_150_3_e-6);
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );
    u = mla(simd, u, s, splat(simd, -0.166_666_597_127_914_428_710_938));

    u = s * (u * d) + d;

    is_neg_zero(simd, r).select(r, u)
}

/// Sine function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn sin_deterministic<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let r = d;

    let mut q = roundi(simd, d * splat(simd, FRAC_1_PI));
    let u = q.to_float::<F32s<S>>();
    d = mla(simd, u, splat(simd, -PI_A2_F), d);
    d = mla(simd, u, splat(simd, -PI_B2_F), d);
    d = mla(simd, u, splat(simd, -PI_C2_F), d);
    let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

    if !g.all_true() {
        let s = q.to_float::<F32s<S>>();
        let mut u = mla(simd, s, splat(simd, -PI_A_F), r);
        u = mla(simd, s, splat(simd, -PI_B_F), u);
        u = mla(simd, s, splat(simd, -PI_C_F), u);
        u = mla(simd, s, splat(simd, -PI_D_F), u);

        d = g.select(d, u);
        let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX_F));

        if !g.all_true() {
            let (mut dfidf, dfii) = rempi(simd, d);
            let mut q2 = dfii & splat_int(simd, 3);
            q2 = q2
                + q2
                + dfidf
                    .0
                    .simd_gt(splat(simd, 0.))
                    .select(splat_int(simd, 2), splat_int(simd, 1));
            q2 >>= 2;
            let o = (dfii & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
            let mut x = DD::new(
                mul_sign(simd, splat(simd, -D_PI_F.0 * 0.5), dfidf.0),
                mul_sign(simd, splat(simd, -D_PI_F.1 * 0.5), dfidf.0),
            );
            x = dfidf + x;
            dfidf = select_doubled(simd, o, x, dfidf);
            u = dfidf.0 + dfidf.1;

            u = from_bits(
                simd,
                (r.is_infinite() | r.is_nan())
                    .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                    | to_bits(simd, u),
            );

            q = g.select(q, q2);
            d = g.select(d, u);
        }
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((q & splat_int(simd, 1)).simd_eq(splat_int(simd, 1)))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, d),
    );

    let mut u = splat(simd, 2.608_315_980_978_659_354_150_3_e-6);
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );
    u = mla(simd, u, s, splat(simd, -0.166_666_597_127_914_428_710_938));

    u = s * (u * d) + d;

    is_neg_zero(simd, r).select(r, u)
}

/// Cosine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn cos<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let mut q;
    let r = d;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        q = roundi(simd, d * splat(simd, FRAC_1_PI) - splat(simd, 0.5));
        q = q + q + splat_int(simd, 1);

        let u = q.to_float::<F32s<S>>();
        d = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
        d = mla(simd, u, splat(simd, -PI_B2_F * 0.5), d);
        d = mla(simd, u, splat(simd, -PI_C2_F * 0.5), d);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX_F)).all_true() {
        q = roundi(simd, d * splat(simd, FRAC_1_PI) - splat(simd, 0.5));
        q = q + q + splat_int(simd, 1);

        let u = q.to_float::<F32s<S>>();
        d = mla(simd, u, splat(simd, -PI_A_F * 0.5), d);
        d = mla(simd, u, splat(simd, -PI_B_F * 0.5), d);
        d = mla(simd, u, splat(simd, -PI_C_F * 0.5), d);
        d = mla(simd, u, splat(simd, -PI_D_F * 0.5), d);
    } else {
        let (mut dfidf, dfii) = rempi(simd, d);
        q = dfii & splat_int(simd, 3);
        q = q
            + q
            + dfidf
                .0
                .simd_gt(splat(simd, 0.))
                .select(splat_int(simd, 8), splat_int(simd, 7));
        q >>= 1;
        let o = (dfii & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
        let y = dfidf
            .0
            .simd_gt(splat(simd, 0.))
            .select(splat(simd, 0.), splat(simd, -1.));
        let mut x = DD::new(
            mul_sign(simd, splat(simd, -D_PI_F.0 * 0.5), y),
            mul_sign(simd, splat(simd, -D_PI_F.1 * 0.5), y),
        );
        x = dfidf + x;
        dfidf = select_doubled(simd, o, x, dfidf);
        d = dfidf.0 + dfidf.1;

        d = from_bits(
            simd,
            (r.is_infinite() | r.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, d),
        );
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((q & splat_int(simd, 2)).simd_eq(splat_int(simd, 0)))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, d),
    );

    let u = splat(simd, 2.608_315_980_978_659_354_150_3_e-6);
    let u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    let u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );
    let u = mla(simd, u, s, splat(simd, -0.166_666_597_127_914_428_710_938));

    s * (u * d) + d
}

/// Cosine function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn cos_deterministic<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let r = d;

    let mut q = roundi(simd, d * splat(simd, FRAC_1_PI) - splat(simd, 0.5));
    q = q + q + splat_int(simd, 1);
    let u = q.to_float::<F32s<S>>();
    d = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
    d = mla(simd, u, splat(simd, -PI_B2_F * 0.5), d);
    d = mla(simd, u, splat(simd, -PI_C2_F * 0.5), d);
    let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

    if !g.all_true() {
        let s = q.to_float::<F32s<S>>();
        let mut u = mla(simd, s, splat(simd, -PI_A_F * 0.5), r);
        u = mla(simd, s, splat(simd, -PI_B_F * 0.5), u);
        u = mla(simd, s, splat(simd, -PI_C_F * 0.5), u);
        u = mla(simd, s, splat(simd, -PI_D_F * 0.5), u);

        d = g.select(d, u);
        let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX_F));

        if !g.all_true() {
            let (mut dfidf, dfii) = rempi(simd, d);
            let mut q2 = dfii & splat_int(simd, 3);
            q2 = q2
                + q2
                + dfidf
                    .0
                    .simd_gt(splat(simd, 0.))
                    .select(splat_int(simd, 8), splat_int(simd, 7));
            q2 >>= 1;
            let o = (dfii & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
            let y = dfidf
                .0
                .simd_gt(splat(simd, 0.))
                .select(splat(simd, 0.), splat(simd, -1.));
            let mut x = DD::new(
                mul_sign(simd, splat(simd, -D_PI_F.0 * 0.5), y),
                mul_sign(simd, splat(simd, -D_PI_F.1 * 0.5), y),
            );
            x = dfidf + x;
            dfidf = select_doubled(simd, o, x, dfidf);
            u = dfidf.0 + dfidf.1;

            u = from_bits(
                simd,
                (r.is_infinite() | r.is_nan())
                    .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                    | to_bits(simd, u),
            );

            q = g.select(q, q2);
            d = g.select(d, u);
        }
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((q & splat_int(simd, 2)).simd_eq(splat_int(simd, 0)))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, d),
    );

    let u = splat(simd, 2.608_315_980_978_659_354_150_3_e-6);
    let u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    let u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );
    let u = mla(simd, u, s, splat(simd, -0.166_666_597_127_914_428_710_938));

    s * (u * d) + d
}

/// Evaluates sine and cosine simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `3.5 ULP`. If the argument is a
/// `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sincos<S: Simd>(simd: S, d: F32s<S>) -> (F32s<S>, F32s<S>) {
    let q;
    let mut s = d;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        q = roundi(simd, d * splat(simd, FRAC_2_PI));
        let u = q.to_float::<F32s<S>>();
        s = mla(simd, u, splat(simd, -PI_A2_F * 0.5), s);
        s = mla(simd, u, splat(simd, -PI_B2_F * 0.5), s);
        s = mla(simd, u, splat(simd, -PI_C2_F * 0.5), s);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX_F)).all_true() {
        q = roundi(simd, d * splat(simd, FRAC_2_PI));
        let u = q.to_float::<F32s<S>>();
        s = mla(simd, u, splat(simd, -PI_A_F * 0.5), s);
        s = mla(simd, u, splat(simd, -PI_B_F * 0.5), s);
        s = mla(simd, u, splat(simd, -PI_C_F * 0.5), s);
        s = mla(simd, u, splat(simd, -PI_D_F * 0.5), s);
    } else {
        let (dfidf, dfii) = rempi(simd, d);
        q = dfii;
        s = dfidf.0 + dfidf.1;
        s = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, s),
        );
    }

    let t = s;

    s = s * s;

    let u = splat(simd, -0.000_195_169_282_960_705_459_117_889);
    let u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_332_157_507_538_795_471_191_41),
    );
    let u = mla(simd, u, s, splat(simd, -0.166_666_537_523_269_653_320_312));

    let rx = mla(simd, u * s, t, t);
    let rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let u = splat(simd, -2.718_118_423_672_422_068_193_55_e-7);
    let u = mla(
        simd,
        u,
        s,
        splat(simd, 2.479_904_469_510_074_704_885_48_e-5),
    );
    let u = mla(
        simd,
        u,
        s,
        splat(simd, -0.001_388_887_874_782_085_418_701_17),
    );
    let u = mla(simd, u, s, splat(simd, 0.041_666_664_183_139_801_025_390_6));
    let u = mla(simd, u, s, splat(simd, -0.5));

    let ry = mla(simd, s, u, splat(simd, 1.));

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, rsin),
    );

    let o = ((q + splat_int(simd, 1)) & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Evaluates sine and cosine simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `3.5 ULP`. If the argument is a
/// `NaN` or infinity, a `NaN` is returned.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn sincos_deterministic<S: Simd>(simd: S, d: F32s<S>) -> (F32s<S>, F32s<S>) {
    let mut q = roundi(simd, d * splat(simd, FRAC_2_PI));
    let u = q.to_float::<F32s<S>>();
    let mut s = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
    s = mla(simd, u, splat(simd, -PI_B2_F * 0.5), s);
    s = mla(simd, u, splat(simd, -PI_C2_F * 0.5), s);
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

    if !g.all_true() {
        let q2 = roundi(simd, d * splat(simd, FRAC_2_PI));
        let u = q2.to_float::<F32s<S>>();
        let mut t = mla(simd, u, splat(simd, -PI_A_F * 0.5), d);
        t = mla(simd, u, splat(simd, -PI_B_F * 0.5), t);
        t = mla(simd, u, splat(simd, -PI_C_F * 0.5), t);
        t = mla(simd, u, splat(simd, -PI_D_F * 0.5), t);

        q = g.select(q, q2);
        s = g.select(s, t);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX_F));

        if !g.all_true() {
            let (dfidf, dfii) = rempi(simd, d);
            let mut t = dfidf.0 + dfidf.1;
            t = from_bits(
                simd,
                (d.is_infinite() | d.is_nan())
                    .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                    | to_bits(simd, t),
            );

            q = g.select(q, dfii);
            s = g.select(s, t);
        }
    }

    let t = s;

    s = s * s;

    let u = splat(simd, -0.000_195_169_282_960_705_459_117_889);
    let u = mla(
        simd,
        u,
        s,
        splat(simd, 0.008_332_157_507_538_795_471_191_41),
    );
    let u = mla(simd, u, s, splat(simd, -0.166_666_537_523_269_653_320_312));

    let mut rx = mla(simd, u * s, t, t);
    rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let u = splat(simd, -2.718_118_423_672_422_068_193_55_e-7);
    let u = mla(
        simd,
        u,
        s,
        splat(simd, 2.479_904_469_510_074_704_885_48_e-5),
    );
    let u = mla(
        simd,
        u,
        s,
        splat(simd, -0.001_388_887_874_782_085_418_701_17),
    );
    let u = mla(simd, u, s, splat(simd, 0.041_666_664_183_139_801_025_390_6));
    let u = mla(simd, u, s, splat(simd, -0.5));

    let ry = mla(simd, s, u, splat(simd, 1.));

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, rsin),
    );

    let o = ((q + splat_int(simd, 1)) & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn tan<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let q;

    let mut x = d;

    if d.abs()
        .simd_lt(splat(simd, TRIGRANGEMAX2_F * 0.5))
        .all_true()
    {
        q = roundi(simd, d * splat(simd, FRAC_2_PI));
        let u = q.to_float::<F32s<S>>();
        x = mla(simd, u, splat(simd, -PI_A2_F * 0.5), x);
        x = mla(simd, u, splat(simd, -PI_B2_F * 0.5), x);
        x = mla(simd, u, splat(simd, -PI_C2_F * 0.5), x);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX_F)).all_true() {
        q = roundi(simd, d * (splat(simd, 2.) * splat(simd, FRAC_1_PI)));
        let u = q.to_float::<F32s<S>>();
        x = mla(simd, u, splat(simd, -PI_A_F * 0.5), x);
        x = mla(simd, u, splat(simd, -PI_B_F * 0.5), x);
        x = mla(simd, u, splat(simd, -PI_C_F * 0.5), x);
        x = mla(simd, u, splat(simd, -PI_D_F * 0.5), x);
    } else {
        let (dfidf, dfii) = rempi(simd, d);
        q = dfii;
        x = dfidf.0 + dfidf.1;
        x = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, x),
        );
        x = is_neg_zero(simd, d).select(d, x);
    }

    let s = x * x;

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
    x = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, x),
    );

    let s2 = s * s;
    let s4 = s2 * s2;

    let mut u = poly6(
        simd,
        s,
        s2,
        s4,
        0.009_272_458_031_773_567_199_707_03,
        0.003_319_849_958_643_317_222_595_21,
        0.024_299_807_846_546_173_095_703_1,
        0.053_449_530_154_466_629_028_320_3,
        0.133_383_005_857_467_651_367_188,
        0.333_331_853_151_321_411_132_812,
    );

    u = mla(simd, s, u * x, x);

    o.select(splat(simd, 1.) / u, u)
}

/// Tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn tan_deterministic<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut q = roundi(simd, d * splat(simd, FRAC_2_PI));
    let u = q.to_float::<F32s<S>>();
    let mut x = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
    x = mla(simd, u, splat(simd, -PI_B2_F * 0.5), x);
    x = mla(simd, u, splat(simd, -PI_C2_F * 0.5), x);
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F * 0.5));

    if !g.all_true() {
        let q2 = roundi(simd, d * splat(simd, FRAC_2_PI));
        let s = q.to_float::<F32s<S>>();
        let mut u = mla(simd, s, splat(simd, -PI_A_F * 0.5), d);
        u = mla(simd, s, splat(simd, -PI_B_F * 0.5), u);
        u = mla(simd, s, splat(simd, -PI_C_F * 0.5), u);
        u = mla(simd, s, splat(simd, -PI_D_F * 0.5), u);

        q = g.select(q, q2);
        x = g.select(x, u);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX_F));

        if !g.all_true() {
            let (dfidf, dfii) = rempi(simd, d);
            u = dfidf.0 + dfidf.1;
            u = from_bits(
                simd,
                (d.is_infinite() | d.is_nan())
                    .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                    | to_bits(simd, u),
            );
            u = is_neg_zero(simd, d).select(d, u);
            q = g.select(q, dfii);
            x = g.select(x, u);
        }
    }

    let s = x * x;

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
    x = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, x),
    );

    let s2 = s * s;
    let s4 = s2 * s2;

    let mut u = poly6(
        simd,
        s,
        s2,
        s4,
        0.009_272_458_031_773_567_199_707_03,
        0.003_319_849_958_643_317_222_595_21,
        0.024_299_807_846_546_173_095_703_1,
        0.053_449_530_154_466_629_028_320_3,
        0.133_383_005_857_467_651_367_188,
        0.333_331_853_151_321_411_132_812,
    );

    u = mla(simd, s, u * x, x);

    o.select(splat(simd, 1.) / u, u)
}

/// Evaluates `sin(pi * x)` and `cos(pi * x)` simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `3.5 ULP` for `x` in
/// `[-1e+7, 1e+7]`; outside that range an arbitrary value within `[-1, 1]` is returned. For a
/// `NaN` or infinite `x`, a `NaN` is returned.
#[inline(always)]
pub fn sincospi<S: Simd>(simd: S, d: F32s<S>) -> (F32s<S>, F32s<S>) {
    let u = d * splat(simd, 4.);
    let q = trunci(simd, u);
    let q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 31)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let s = u - q.to_float::<F32s<S>>();

    let t = s;
    let s = s * s;

    //

    let u = splat(simd, -0.360_092_526_5_e-4);
    let u = mla(simd, u, s, splat(simd, 0.249_008_811_1_e-2));
    let u = mla(simd, u, s, splat(simd, -0.807_455_107_6_e-1));
    let u = mla(simd, u, s, splat(simd, 0.785_398_185_3));

    let rx = u * t;

    //

    let u = splat(simd, 0.353_981_522_5_e-5);
    let u = mla(simd, u, s, splat(simd, -0.325_957_400_5_e-3));
    let u = mla(simd, u, s, splat(simd, 0.158_543_158_3_e-1));
    let u = mla(simd, u, s, splat(simd, -0.308_425_128_5));
    let u = mla(simd, u, s, splat(simd, 1.));

    let ry = u;

    //

    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (q & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, rsin),
    );

    let o = ((q + splat_int(simd, 2)) & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, rcos),
    );

    let o = d.abs().simd_gt(splat(simd, 1e+7));
    rsin = from_bits(
        simd,
        !o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) & to_bits(simd, rsin),
    );
    rcos = from_bits(
        simd,
        !o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) & to_bits(simd, rcos),
    );

    let o = d.is_infinite();
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) | to_bits(simd, rsin),
    );
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) | to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Arc tangent of `y / x`, with the quadrant taken from the signs of `y` and `x`.
#[inline(always)]
fn atan2kf<S: Simd>(simd: S, y: F32s<S>, x: F32s<S>) -> F32s<S> {
    let q = x
        .is_sign_negative()
        .select(splat_int(simd, -2), splat_int(simd, 0));
    let x = x.abs();

    let q = x.simd_lt(y).select(q + splat_int(simd, 1), q);
    let p = x.simd_lt(y);
    let s = p.select(-x, y);
    let mut t = x.max(y);

    let s = s / t;
    t = s * s;

    let t2 = t * t;
    let t4 = t2 * t2;

    let u = poly8(
        simd,
        t,
        t2,
        t4,
        0.002_823_638_962_581_753_730_773_93,
        -0.015_956_902_876_496_315_002_441_4,
        0.042_504_988_610_744_476_318_359_4,
        -0.074_890_092_015_266_418_457_031_2,
        0.106_347_933_411_598_205_566_406,
        -0.142_027_363_181_114_196_777_344,
        0.199_926_957_488_059_997_558_594,
        -0.333_331_018_686_294_555_664_062,
    );

    let t = mla(simd, s, t * u, s);
    mla(simd, q.to_float::<F32s<S>>(), splat(simd, FRAC_PI_2), t)
}

/// Arc tangent function of two variables.
///
/// Evaluates the arc tangent of `y / x`; the quadrant of the result is determined from the signs
/// of `x` and `y`. The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn atan2<S: Simd>(simd: S, y: F32s<S>, x: F32s<S>) -> F32s<S> {
    let mut r = atan2kf(simd, y.abs(), x);

    r = mul_sign(simd, r, x);
    r = (x.is_infinite() | x.simd_eq(splat(simd, 0.))).select(
        splat(simd, FRAC_PI_2) - visinf2(simd, x, mul_sign(simd, splat(simd, FRAC_PI_2), x)),
        r,
    );
    r = y.is_infinite().select(
        splat(simd, FRAC_PI_2) - visinf2(simd, x, mul_sign(simd, splat(simd, FRAC_PI_4), x)),
        r,
    );

    r = y.simd_eq(splat(simd, 0.)).select(
        from_bits(
            simd,
            x.is_sign_negative()
                .select(splat_bits(simd, PI.to_bits()), splat_bits(simd, 0)),
        ),
        r,
    );

    from_bits(
        simd,
        (x.is_nan() | y.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, mul_sign(simd, r, y)),
    )
}

/// Arc sine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn asin<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.) - d.abs()) * splat(simd, 0.5));
    let x = o.select(d.abs(), x2.sqrt());

    let u = splat(simd, 0.419_745_482_5_e-1);
    let u = mla(simd, u, x2, splat(simd, 0.242_404_602_5_e-1));
    let u = mla(simd, u, x2, splat(simd, 0.454_742_386_9_e-1));
    let u = mla(simd, u, x2, splat(simd, 0.749_502_927_1_e-1));
    let u = mla(simd, u, x2, splat(simd, 0.166_667_729_6));
    let u = mla(simd, u, x * x2, x);

    let r = o.select(u, mla(simd, u, splat(simd, -2.), splat(simd, FRAC_PI_2)));
    mul_sign(simd, r, d)
}

/// Arc cosine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn acos<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.) - d.abs()) * splat(simd, 0.5));
    let mut x = o.select(d.abs(), x2.sqrt());
    x = d.abs().simd_eq(splat(simd, 1.)).select(splat(simd, 0.), x);

    let u = splat(simd, 0.419_745_482_5_e-1);
    let u = mla(simd, u, x2, splat(simd, 0.242_404_602_5_e-1));
    let u = mla(simd, u, x2, splat(simd, 0.454_742_386_9_e-1));
    let u = mla(simd, u, x2, splat(simd, 0.749_502_927_1_e-1));
    let u = mla(simd, u, x2, splat(simd, 0.166_667_729_6));
    let u = u * (x2 * x);

    let y = splat(simd, FRAC_PI_2) - (mul_sign(simd, x, d) + mul_sign(simd, u, d));
    x += u;
    let r = o.select(y, x * splat(simd, 2.));
    (!o & d.simd_lt(splat(simd, 0.))).select(add_checked_v(DD::splat(simd, D_PI_F), -r).0, r)
}

/// Arc tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn atan<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let q = d
        .is_sign_negative()
        .select(splat_int(simd, 2), splat_int(simd, 0));
    let s = d.abs();

    let q = splat(simd, 1.).simd_lt(s).select(q + splat_int(simd, 1), q);
    let s = splat(simd, 1.).simd_lt(s).select(splat(simd, 1.) / s, s);

    let mut t = s * s;

    let t2 = t * t;
    let t4 = t2 * t2;

    let u = poly8(
        simd,
        t,
        t2,
        t4,
        0.002_823_638_962_581_753_730_773_93,
        -0.015_956_902_876_496_315_002_441_4,
        0.042_504_988_610_744_476_318_359_4,
        -0.074_890_092_015_266_418_457_031_2,
        0.106_347_933_411_598_205_566_406,
        -0.142_027_363_181_114_196_777_344,
        0.199_926_957_488_059_997_558_594,
        -0.333_331_018_686_294_555_664_062,
    );

    t = mla(simd, s, t * u, s);

    t = (q & splat_int(simd, 1))
        .simd_eq(splat_int(simd, 1))
        .select(splat(simd, FRAC_PI_2) - t, t);

    t = from_bits(
        simd,
        ((q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2)))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, t),
    );

    t
}

/// Hyperbolic sine function.
///
/// The error bound of the returned value is `3.5 ULP` for `x` in `[-88, 88]`. Outside that range
/// infinity with the correct sign, or a correct value within `3.5 ULP`, is returned.
#[inline(always)]
pub fn sinh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let e = expm1fk(simd, x.abs());
    let mut y = (e + splat(simd, 2.)) / (e + splat(simd, 1.));
    y *= splat(simd, 0.5) * e;

    y = (x.abs().simd_gt(splat(simd, 88.)) | y.is_nan()).select(splat(simd, f32::INFINITY), y);
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Hyperbolic cosine function.
///
/// The error bound of the returned value is `3.5 ULP` for `x` in `[-88, 88]`. Outside that range
/// infinity, or a correct value within `3.5 ULP`, is returned.
#[inline(always)]
pub fn cosh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let e = u10::exp(simd, x.abs());
    let mut y = mla(simd, splat(simd, 0.5), e, splat(simd, 0.5) / e);

    y = (x.abs().simd_gt(splat(simd, 88.)) | y.is_nan()).select(splat(simd, f32::INFINITY), y);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Hyperbolic tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn tanh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let d = expm1fk(simd, splat(simd, 2.) * x.abs());
    let mut y = d / (splat(simd, 2.) + d);

    y = (x.abs().simd_gt(splat(simd, 8.664_339_742)) | y.is_nan()).select(splat(simd, 1.), y);
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Natural logarithm.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn log<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
    d = o.select(d * splat(simd, F1_32 * F1_32), d);
    let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
    let m = ldexp3k(simd, d, -e);
    e = o.select(e - splat_int(simd, 64), e);
    let ef = e.to_float::<F32s<S>>();

    let mut x = (m - splat(simd, 1.)) / (splat(simd, 1.) + m);
    let x2 = x * x;

    let t = splat(simd, 0.239_282_846_450_805_664_062_5);
    let t = mla(simd, t, x2, splat(simd, 0.285_182_118_415_832_519_531_25));
    let t = mla(simd, t, x2, splat(simd, 0.400_005_877_017_974_853_515_625));
    let t = mla(simd, t, x2, splat(simd, 0.666_666_686_534_881_591_796_875));
    let t = mla(simd, t, x2, splat(simd, 2.));

    x = mla(
        simd,
        x,
        t,
        splat(simd, 0.693_147_180_559_945_286_226_764) * ef,
    );

    x = d
        .simd_eq(splat(simd, f32::INFINITY))
        .select(splat(simd, f32::INFINITY), x);
    x = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f32::NAN), x);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NEG_INFINITY), x)
}

/// Base-2 logarithmic function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn log2<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
    d = o.select(d * splat(simd, F1_32 * F1_32), d);
    let e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
    let m = ldexp3k(simd, d, -e);
    let e = o.select(e - splat_int(simd, 64), e);

    let x = (m - splat(simd, 1.)) / (m + splat(simd, 1.));
    let x2 = x * x;

    let t = splat(simd, 0.437_408_834_7);
    let t = mla(simd, t, x2, splat(simd, 0.576_484_382_2));
    let t = mla(simd, t, x2, splat(simd, 0.961_802_423));

    let mut r = mla(
        simd,
        x2 * x,
        t,
        mla(
            simd,
            x,
            splat(simd, 0.288_539_004_3_e+1),
            e.to_float::<F32s<S>>(),
        ),
    );

    r = d
        .simd_eq(splat(simd, f32::INFINITY))
        .select(splat(simd, f32::INFINITY), r);
    r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f32::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NEG_INFINITY), r)
}

/// Base-10 exponential function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn exp10<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut u = rint(simd, d * splat(simd, LOG10_2_F));
    let q = roundi(simd, u);

    let mut s = mla(simd, u, splat(simd, -L10U_F), d);
    s = mla(simd, u, splat(simd, -L10L_F), s);

    u = splat(simd, 0.206_400_498_7);
    u = mla(simd, u, s, splat(simd, 0.541_787_743_6));
    u = mla(simd, u, s, splat(simd, 0.117_128_682_1_e+1));
    u = mla(simd, u, s, splat(simd, 0.203_465_604_8_e+1));
    u = mla(simd, u, s, splat(simd, 0.265_094_876_3_e+1));
    u = mla(simd, u, s, splat(simd, 0.230_258_512_5_e+1));
    u = mla(simd, u, s, splat(simd, 0.1_e+1));

    u = ldexp2k(simd, u, q);

    u = d
        .simd_gt(splat(simd, 38.531_839_419_103_623_894_138_7))
        .select(splat(simd, f32::INFINITY), u);
    from_bits(
        simd,
        !d.simd_lt(splat(simd, -50.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Base-2 exponential function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn exp2<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut u = rint(simd, d);
    let q = roundi(simd, u);

    let s = d - u;

    u = splat(simd, 0.153_592_089_2_e-3);
    u = mla(simd, u, s, splat(simd, 0.133_926_270_1_e-2));
    u = mla(simd, u, s, splat(simd, 0.961_838_476_4_e-2));
    u = mla(simd, u, s, splat(simd, 0.555_034_726_9_e-1));
    u = mla(simd, u, s, splat(simd, 0.240_226_447_6));
    u = mla(simd, u, s, splat(simd, 0.693_147_182_5));
    u = mla(simd, u, s, splat(simd, 0.1_e+1));

    u = ldexp2k(simd, u, q);

    u = d
        .simd_ge(splat(simd, 128.))
        .select(splat(simd, f32::INFINITY), u);
    from_bits(
        simd,
        !d.simd_lt(splat(simd, -150.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Square root function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// Upstream's `u35::sqrt` is the hardware square root; as with the `f64` port this forwards to
/// the `u05` tier's `sqrt`.
#[inline(always)]
pub fn sqrt<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    super::u05::sqrt(simd, d)
}

/// Cube root function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn cbrt<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let mut q = splat(simd, 1.);

    let e = ilogbk(simd, d.abs()) + splat_int(simd, 1);
    d = ldexp2k(simd, d, -e);

    let t = e.to_float::<F32s<S>>() + splat(simd, 6144.);
    let qu = trunci(simd, t * splat(simd, 1. / 3.));
    let re = trunci(simd, t - qu.to_float::<F32s<S>>() * splat(simd, 3.));

    q = re
        .simd_eq(splat_int(simd, 1))
        .select(splat(simd, 1.259_921_049_894_873_164_767_210_6), q);
    q = re
        .simd_eq(splat_int(simd, 2))
        .select(splat(simd, 1.587_401_051_968_199_474_751_705_6), q);
    q = ldexp2k(simd, q, qu - splat_int(simd, 2048));

    q = mul_sign(simd, q, d);
    d = d.abs();

    let x = splat(simd, -0.601_564_466_953_277_587_890_625);
    let x = mla(simd, x, d, splat(simd, 2.820_889_234_542_846_679_687_5));
    let x = mla(simd, x, d, splat(simd, -5.532_182_216_644_287_109_375));
    let x = mla(simd, x, d, splat(simd, 5.898_262_500_762_939_453_125));
    let x = mla(simd, x, d, splat(simd, -3.809_541_702_270_507_812_5));
    let x = mla(simd, x, d, splat(simd, 2.224_125_623_703_002_929_687_5));

    let mut y = d * x * x;
    y = (y - splat(simd, 2. / 3.) * y * mla(simd, y, x, splat(simd, -1.))) * q;

    y
}

/// 2D Euclidean distance function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn hypot<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let x = x.abs();
    let y = y.abs();
    let min = x.min(y);
    let max = x.max(y);

    let t = min / max;
    let mut ret = max * mla(simd, t, t, splat(simd, 1.)).sqrt();
    ret = min.simd_eq(splat(simd, 0.)).select(max, ret);
    ret = (x.is_nan() | y.is_nan()).select(splat(simd, f32::NAN), ret);
    (x.simd_eq(splat(simd, f32::INFINITY)) | y.simd_eq(splat(simd, f32::INFINITY)))
        .select(splat(simd, f32::INFINITY), ret)
}
