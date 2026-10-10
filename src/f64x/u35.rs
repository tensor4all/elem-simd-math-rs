//! `f64` kernels with a 3.5 ULP error bound (SLEEF's `u35` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x/u35.rs`).

use core::f64::consts::{FRAC_1_PI, FRAC_2_PI, FRAC_PI_2, FRAC_PI_4, PI};

use super::*;
use crate::doubled::f64::add_checked_v;

/// `exp(x) - 1`, used internally where the result is scaled by a known factor.
#[inline(always)]
pub fn expm1k<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut u = rint(simd, d * splat(simd, R_LN2));
    let q = roundi(simd, u);

    let s = mla(simd, u, splat(simd, -L2_U), d);
    let s = mla(simd, u, splat(simd, -L2_L), s);

    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;

    u = poly10(
        simd,
        s,
        s2,
        s4,
        s8,
        2.088_606_211_072_836_875_363_41_e-9,
        2.511_129_308_928_765_186_106_61_e-8,
        2.755_739_112_349_004_718_933_38_e-7,
        2.755_723_629_119_288_276_294_23_e-6,
        2.480_158_715_923_547_299_879_1_e-5,
        0.000_198_412_698_960_509_205_564_975,
        0.001_388_888_888_897_744_922_079_62,
        0.008_333_333_333_316_527_216_649_84,
        0.041_666_666_666_666_504_759_142_2,
        0.166_666_666_666_666_851_703_837,
    );

    u = mla(simd, s2, splat(simd, 0.5), s2 * s * u) + s;

    q.simd_eq(splat_int(simd, 0))
        .select(u, ldexp2k(simd, u + splat(simd, 1.), q) - splat(simd, 1.))
}

/// Hyperbolic tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn tanh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let d = expm1k(simd, splat(simd, 2.) * x.abs());
    let mut y = d / (splat(simd, 2.) + d);

    y = (x.abs().simd_gt(splat(simd, 18.714_973_875)) | y.is_nan()).select(splat(simd, 1.), y);
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Natural logarithm.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn log<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let o = d.simd_lt(splat(simd, f64::MIN_POSITIVE));
    d = o.select(d * splat(simd, D1_32 * D1_32), d);
    let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
    let m = ldexp3k(simd, d, -e);
    e = o.select(e - splat_int(simd, 64), e);

    let mut x = (m - splat(simd, 1.)) / (splat(simd, 1.) + m);
    let x2 = x * x;

    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let x3 = x * x2;

    let t = poly7(
        simd,
        x2,
        x4,
        x8,
        0.153_487_338_491_425_068_243_146,
        0.152_519_917_006_351_951_593_857,
        0.181_863_266_251_982_985_677_316,
        0.222_221_366_518_767_365_905_163,
        0.285_714_294_746_548_025_383_248,
        0.399_999_999_950_799_600_689_777,
        0.666_666_666_666_777_874_006_3,
    );

    x = mla(
        simd,
        x,
        splat(simd, 2.),
        splat(simd, 0.693_147_180_559_945_286_226_764) * e.to_float::<F64s<S>>(),
    );
    x = mla(simd, x3, t, x);

    x = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), x);
    x = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f64::NAN), x);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f64::NEG_INFINITY), x)
}

/// Sine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn sin<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let r = d;
    let mut ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(simd, d * splat(simd, FRAC_1_PI));
        ql = roundi(simd, dql);
        d = mla(simd, dql, splat(simd, -PI_A2), d);
        d = mla(simd, dql, splat(simd, -PI_B2), d);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX)).all_true() {
        let dqh = trunc(simd, d * splat(simd, FRAC_1_PI / D1_24));
        let dqh = dqh * splat(simd, D1_24);
        let dql = rint(simd, mul_sub(simd, d, splat(simd, FRAC_1_PI), dqh));
        ql = roundi(simd, dql);

        d = mla(simd, dqh, splat(simd, -PI_A), d);
        d = mla(simd, dql, splat(simd, -PI_A), d);
        d = mla(simd, dqh, splat(simd, -PI_B), d);
        d = mla(simd, dql, splat(simd, -PI_B), d);
        d = mla(simd, dqh, splat(simd, -PI_C), d);
        d = mla(simd, dql, splat(simd, -PI_C), d);
        d = mla(simd, dqh + dql, splat(simd, -PI_D), d);
    } else {
        let (mut ddidd, ddii) = rempi(simd, d);
        ql = ddii & splat_int(simd, 3);
        ql = ql
            + ql
            + ddidd
                .0
                .simd_gt(splat(simd, 0.))
                .select(splat_int(simd, 2), splat_int(simd, 1));
        ql >>= 2;
        let o = (ddii & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
        let mut x = DD::new(
            mul_sign(simd, splat(simd, -D_PI.0 * 0.5), ddidd.0),
            mul_sign(simd, splat(simd, -D_PI.1 * 0.5), ddidd.0),
        );
        x = ddidd + x;
        ddidd = select_doubled(simd, o, x, ddidd);
        d = ddidd.0 + ddidd.1;
        d = from_bits(
            simd,
            (r.is_infinite() | r.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, d),
        );
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, d),
    );

    let s2 = s * s;
    let s4 = s2 * s2;

    let mut u = poly8(
        simd,
        s,
        s2,
        s4,
        -7.972_559_550_090_378_688_919_52_e-18,
        2.810_099_727_108_632_000_912_51_e-15,
        -7.647_122_191_181_588_332_884_84_e-13,
        1.605_904_306_056_645_016_290_54_e-10,
        -2.505_210_837_635_020_458_107_55_e-8,
        2.755_731_922_391_987_476_304_16_e-6,
        -0.000_198_412_698_412_696_162_806_809,
        0.008_333_333_333_333_329_748_238_15,
    );
    u = mla(simd, u, s, splat(simd, -0.166_666_666_666_666_657_414_808));

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
pub fn sin_deterministic<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    // This is the deterministic implementation of sin function. Returned
    // values from deterministic functions are bitwise consistent across
    // all platforms.
    let r = d;

    let dql = rint(simd, d * splat(simd, FRAC_1_PI));
    let mut ql = roundi(simd, dql);
    d = mla(simd, dql, splat(simd, -PI_A2), d);
    d = mla(simd, dql, splat(simd, -PI_B2), d);
    let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX2));

    if !g.all_true() {
        let mut dqh = trunc(simd, r * splat(simd, FRAC_1_PI / D1_24));
        dqh *= splat(simd, D1_24);
        let dql = rint(simd, mul_sub(simd, r, splat(simd, FRAC_1_PI), dqh));

        let mut u = mla(simd, dqh, splat(simd, -PI_A), r);
        u = mla(simd, dql, splat(simd, -PI_A), u);
        u = mla(simd, dqh, splat(simd, -PI_B), u);
        u = mla(simd, dql, splat(simd, -PI_B), u);
        u = mla(simd, dqh, splat(simd, -PI_C), u);
        u = mla(simd, dql, splat(simd, -PI_C), u);
        u = mla(simd, dqh + dql, splat(simd, -PI_D), u);

        ql = g.select(ql, roundi(simd, dql));
        d = g.select(d, u);
        let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (mut ddidd, ddii) = rempi(simd, r);
            let mut ql2 = ddii & splat_int(simd, 3);
            ql2 = ql2
                + ql2
                + ddidd
                    .0
                    .simd_gt(splat(simd, 0.))
                    .select(splat_int(simd, 2), splat_int(simd, 1));
            ql2 >>= 2;
            let o = (ddii & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
            let mut x = DD::new(
                mul_sign(simd, splat(simd, -D_PI.0 * 0.5), ddidd.0),
                mul_sign(simd, splat(simd, -D_PI.1 * 0.5), ddidd.0),
            );
            x = ddidd + x;
            ddidd = select_doubled(simd, o, x, ddidd);
            let u = ddidd.0 + ddidd.1;
            ql = g.select(ql, ql2);
            d = g.select(d, u);
            d = from_bits(
                simd,
                (r.is_infinite() | r.is_nan())
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    | to_bits(simd, d),
            );
        }
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, d),
    );

    let s2 = s * s;
    let s4 = s2 * s2;

    let mut u = poly8(
        simd,
        s,
        s2,
        s4,
        -7.972_559_550_090_378_688_919_52_e-18,
        2.810_099_727_108_632_000_912_51_e-15,
        -7.647_122_191_181_588_332_884_84_e-13,
        1.605_904_306_056_645_016_290_54_e-10,
        -2.505_210_837_635_020_458_107_55_e-8,
        2.755_731_922_391_987_476_304_16_e-6,
        -0.000_198_412_698_412_696_162_806_809,
        0.008_333_333_333_333_329_748_238_15,
    );
    u = mla(simd, u, s, splat(simd, -0.166_666_666_666_666_657_414_808));

    u = s * (u * d) + d;

    is_neg_zero(simd, r).select(r, u)
}

/// Cosine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn cos<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let r = d;
    let mut ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = mla(
            simd,
            splat(simd, 2.),
            rint(
                simd,
                mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5)),
            ),
            splat(simd, 1.),
        );
        ql = roundi(simd, dql);
        d = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
        d = mla(simd, dql, splat(simd, -PI_B2 * 0.5), d);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX)).all_true() {
        let dqh = trunc(
            simd,
            mla(
                simd,
                d,
                splat(simd, FRAC_1_PI / D1_23),
                splat(simd, -FRAC_1_PI / D1_24),
            ),
        );
        ql = roundi(
            simd,
            d * splat(simd, FRAC_1_PI) + mla(simd, dqh, splat(simd, -D1_23), splat(simd, -0.5)),
        );
        let dqh = dqh * splat(simd, D1_24);
        ql = ql + ql + splat_int(simd, 1);
        let dql = ql.to_float::<F64s<S>>();

        d = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        d = mla(simd, dql, splat(simd, -PI_A * 0.5), d);
        d = mla(simd, dqh, splat(simd, -PI_B * 0.5), d);
        d = mla(simd, dql, splat(simd, -PI_B * 0.5), d);
        d = mla(simd, dqh, splat(simd, -PI_C * 0.5), d);
        d = mla(simd, dql, splat(simd, -PI_C * 0.5), d);
        d = mla(simd, dqh + dql, splat(simd, -PI_D * 0.5), d);
    } else {
        let (mut ddidd, ddii) = rempi(simd, d);
        ql = ddii & splat_int(simd, 3);
        ql = ql
            + ql
            + ddidd
                .0
                .simd_gt(splat(simd, 0.))
                .select(splat_int(simd, 8), splat_int(simd, 7));
        ql >>= 1;
        let o = (ddii & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
        let y = ddidd
            .0
            .simd_gt(splat(simd, 0.))
            .select(splat(simd, 0.), splat(simd, -1.));
        let mut x = DD::new(
            mul_sign(simd, splat(simd, -D_PI.0 * 0.5), y),
            mul_sign(simd, splat(simd, -D_PI.1 * 0.5), y),
        );
        x = ddidd + x;
        ddidd = select_doubled(simd, o, x, ddidd);
        d = ddidd.0 + ddidd.1;
        d = from_bits(
            simd,
            (r.is_infinite() | r.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, d),
        );
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((ql & splat_int(simd, 2)).simd_eq(splat_int(simd, 0)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, d),
    );

    let s2 = s * s;
    let s4 = s2 * s2;

    let u = poly8(
        simd,
        s,
        s2,
        s4,
        -7.972_559_550_090_378_688_919_52_e-18,
        2.810_099_727_108_632_000_912_51_e-15,
        -7.647_122_191_181_588_332_884_84_e-13,
        1.605_904_306_056_645_016_290_54_e-10,
        -2.505_210_837_635_020_458_107_55_e-8,
        2.755_731_922_391_987_476_304_16_e-6,
        -0.000_198_412_698_412_696_162_806_809,
        0.008_333_333_333_333_329_748_238_15,
    );
    let u = mla(simd, u, s, splat(simd, -0.166_666_666_666_666_657_414_808));

    s * (u * d) + d
}

/// Cosine function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn cos_deterministic<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let r = d;

    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));
    let dql = mla(
        simd,
        splat(simd, 2.),
        rint(
            simd,
            mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5)),
        ),
        splat(simd, 1.),
    );
    let mut ql = roundi(simd, dql);
    d = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
    d = mla(simd, dql, splat(simd, -PI_B2 * 0.5), d);

    if !g.all_true() {
        let mut dqh = trunc(
            simd,
            mla(
                simd,
                r,
                splat(simd, FRAC_1_PI / D1_23),
                splat(simd, -FRAC_1_PI / D1_24),
            ),
        );
        let mut ql2 = roundi(
            simd,
            r * splat(simd, FRAC_1_PI) + mla(simd, dqh, splat(simd, -D1_23), splat(simd, -0.5)),
        );
        dqh *= splat(simd, D1_24);
        ql2 = ql2 + ql2 + splat_int(simd, 1);
        let dql = ql2.to_float::<F64s<S>>();

        let mut u = mla(simd, dqh, splat(simd, -PI_A * 0.5), r);
        u = mla(simd, dql, splat(simd, -PI_A * 0.5), u);
        u = mla(simd, dqh, splat(simd, -PI_B * 0.5), u);
        u = mla(simd, dql, splat(simd, -PI_B * 0.5), u);
        u = mla(simd, dqh, splat(simd, -PI_C * 0.5), u);
        u = mla(simd, dql, splat(simd, -PI_C * 0.5), u);
        u = mla(simd, dqh + dql, splat(simd, -PI_D * 0.5), u);

        ql = g.select(ql, ql2);
        d = g.select(d, u);
        let g = r.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (mut ddidd, ddii) = rempi(simd, r);
            let mut ql2 = ddii & splat_int(simd, 3);
            ql2 = ql2
                + ql2
                + ddidd
                    .0
                    .simd_gt(splat(simd, 0.))
                    .select(splat_int(simd, 8), splat_int(simd, 7));
            ql2 >>= 1;
            let o = (ddii & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
            let y = ddidd
                .0
                .simd_gt(splat(simd, 0.))
                .select(splat(simd, 0.), splat(simd, -1.));
            let mut x = DD::new(
                mul_sign(simd, splat(simd, -D_PI.0 * 0.5), y),
                mul_sign(simd, splat(simd, -D_PI.1 * 0.5), y),
            );
            x = ddidd + x;
            ddidd = select_doubled(simd, o, x, ddidd);
            let u = ddidd.0 + ddidd.1;
            ql = g.select(ql, ql2);
            d = g.select(d, u);
            d = from_bits(
                simd,
                (r.is_infinite() | r.is_nan())
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    | to_bits(simd, d),
            );
        }
    }

    let s = d * d;

    d = from_bits(
        simd,
        ((ql & splat_int(simd, 2)).simd_eq(splat_int(simd, 0)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, d),
    );

    let s2 = s * s;
    let s4 = s2 * s2;

    let u = poly8(
        simd,
        s,
        s2,
        s4,
        -7.972_559_550_090_378_688_919_52_e-18,
        2.810_099_727_108_632_000_912_51_e-15,
        -7.647_122_191_181_588_332_884_84_e-13,
        1.605_904_306_056_645_016_290_54_e-10,
        -2.505_210_837_635_020_458_107_55_e-8,
        2.755_731_922_391_987_476_304_16_e-6,
        -0.000_198_412_698_412_696_162_806_809,
        0.008_333_333_333_333_329_748_238_15,
    );
    let u = mla(simd, u, s, splat(simd, -0.166_666_666_666_666_657_414_808));

    s * (u * d) + d
}

/// Evaluates sine and cosine simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `3.5 ULP`.
#[inline(always)]
pub fn sincos<S: Simd>(simd: S, d: F64s<S>) -> (F64s<S>, F64s<S>) {
    let mut s;
    let ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(simd, d * splat(simd, FRAC_2_PI));
        ql = roundi(simd, dql);
        s = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
        s = mla(simd, dql, splat(simd, -PI_B2 * 0.5), s);
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX)).all_true() {
        let dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        let dqh = dqh * splat(simd, D1_24);
        let dql = rint(simd, d * splat(simd, FRAC_2_PI) - dqh);
        ql = roundi(simd, dql);

        s = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        s = mla(simd, dql, splat(simd, -PI_A * 0.5), s);
        s = mla(simd, dqh, splat(simd, -PI_B * 0.5), s);
        s = mla(simd, dql, splat(simd, -PI_B * 0.5), s);
        s = mla(simd, dqh, splat(simd, -PI_C * 0.5), s);
        s = mla(simd, dql, splat(simd, -PI_C * 0.5), s);
        s = mla(simd, dqh + dql, splat(simd, -PI_D * 0.5), s);
    } else {
        let (ddidd, ddii) = rempi(simd, d);
        ql = ddii;
        s = ddidd.0 + ddidd.1;
        s = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, s),
        );
    }

    let t = s;

    s = s * s;

    let mut u = splat(simd, 1.589_383_072_832_289_373_285_11_e-10);
    u = mla(
        simd,
        u,
        s,
        splat(simd, -2.505_069_435_025_397_733_493_18_e-8),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 2.755_731_317_768_463_605_125_47_e-6),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_198_412_698_278_911_770_864_914),
    );
    u = mla(simd, u, s, splat(simd, 0.008_333_333_333_319_184_596_174_6));
    u = mla(simd, u, s, splat(simd, -0.166_666_666_666_666_130_709_393));

    let rx = mla(simd, u * s, t, t);
    let rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let mut u = splat(simd, -1.136_153_502_390_974_295_315_23_e-11);
    u = mla(
        simd,
        u,
        s,
        splat(simd, 2.087_574_712_070_400_554_793_66_e-9),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -2.755_731_440_288_475_674_985_67_e-7),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 2.480_158_728_900_018_673_119_15_e-5),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.001_388_888_888_887_140_192_823_29),
    );
    u = mla(simd, u, s, splat(simd, 0.041_666_666_666_666_551_959_206_2));
    u = mla(simd, u, s, splat(simd, -0.5));

    let ry = mla(simd, s, u, splat(simd, 1.));

    let o = (ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (ql & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, rsin),
    );

    let o = ((ql + splat_int(simd, 1)) & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Evaluates sine and cosine simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `3.5 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn sincos_deterministic<S: Simd>(simd: S, d: F64s<S>) -> (F64s<S>, F64s<S>) {
    let mut s = d;

    let dql = rint(simd, s * splat(simd, FRAC_2_PI));
    let mut ql = roundi(simd, dql);
    s = mla(simd, dql, splat(simd, -PI_A2 * 0.5), s);
    s = mla(simd, dql, splat(simd, -PI_B2 * 0.5), s);
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));

    if !g.all_true() {
        let mut dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        dqh *= splat(simd, D1_24);
        let dql = rint(simd, d * splat(simd, FRAC_2_PI) - dqh);

        let mut u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        u = mla(simd, dql, splat(simd, -PI_A * 0.5), u);
        u = mla(simd, dqh, splat(simd, -PI_B * 0.5), u);
        u = mla(simd, dql, splat(simd, -PI_B * 0.5), u);
        u = mla(simd, dqh, splat(simd, -PI_C * 0.5), u);
        u = mla(simd, dql, splat(simd, -PI_C * 0.5), u);
        u = mla(simd, dqh + dql, splat(simd, -PI_D * 0.5), u);

        ql = g.select(ql, roundi(simd, dql));
        s = g.select(s, u);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (ddidd, ddii) = rempi(simd, d);
            let mut u = ddidd.0 + ddidd.1;
            u = from_bits(
                simd,
                (d.is_infinite() | d.is_nan())
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    | to_bits(simd, u),
            );

            ql = g.select(ql, ddii);
            s = g.select(s, u);
        }
    }

    let t = s;

    s = s * s;

    let mut u = splat(simd, 1.589_383_072_832_289_373_285_11_e-10);
    u = mla(
        simd,
        u,
        s,
        splat(simd, -2.505_069_435_025_397_733_493_18_e-8),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 2.755_731_317_768_463_605_125_47_e-6),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_198_412_698_278_911_770_864_914),
    );
    u = mla(simd, u, s, splat(simd, 0.008_333_333_333_319_184_596_174_6));
    u = mla(simd, u, s, splat(simd, -0.166_666_666_666_666_130_709_393));

    let mut rx = mla(simd, u * s, t, t);
    rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let mut u = splat(simd, -1.136_153_502_390_974_295_315_23_e-11);
    u = mla(
        simd,
        u,
        s,
        splat(simd, 2.087_574_712_070_400_554_793_66_e-9),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -2.755_731_440_288_475_674_985_67_e-7),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 2.480_158_728_900_018_673_119_15_e-5),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.001_388_888_888_887_140_192_823_29),
    );
    u = mla(simd, u, s, splat(simd, 0.041_666_666_666_666_551_959_206_2));
    u = mla(simd, u, s, splat(simd, -0.5));

    let ry = mla(simd, s, u, splat(simd, 1.));

    let o = (ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (ql & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, rsin),
    );

    let o = ((ql + splat_int(simd, 1)) & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn tan<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let ql;

    let mut x;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(simd, d * splat(simd, FRAC_2_PI));
        ql = roundi(simd, dql);
        x = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
        x = mla(simd, dql, splat(simd, -PI_B2 * 0.5), x);
    } else if d.abs().simd_lt(splat(simd, 1e+6)).all_true() {
        let dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        let dqh = dqh * splat(simd, D1_24);
        let dql = rint(simd, d * splat(simd, FRAC_2_PI) - dqh);
        ql = roundi(simd, dql);

        x = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        x = mla(simd, dql, splat(simd, -PI_A * 0.5), x);
        x = mla(simd, dqh, splat(simd, -PI_B * 0.5), x);
        x = mla(simd, dql, splat(simd, -PI_B * 0.5), x);
        x = mla(simd, dqh, splat(simd, -PI_C * 0.5), x);
        x = mla(simd, dql, splat(simd, -PI_C * 0.5), x);
        x = mla(simd, dqh + dql, splat(simd, -PI_D * 0.5), x);
    } else {
        let (ddidd, ddii) = rempi(simd, d);
        ql = ddii;
        x = ddidd.0 + ddidd.1;
        x = from_bits(
            simd,
            d.is_infinite()
                .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, x),
        );
        x = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, x),
        );
    }

    x *= splat(simd, 0.5);
    let s = x * x;

    let s2 = s * s;
    let s4 = s2 * s2;

    let mut u = poly8(
        simd,
        s,
        s2,
        s4,
        0.324_509_882_663_927_631_6_e-3,
        0.561_921_973_811_432_373_5_e-3,
        0.146_078_150_240_278_449_4_e-2,
        0.359_161_154_079_249_951_9_e-2,
        0.886_326_840_956_311_312_6_e-2,
        0.218_694_872_818_553_549_8_e-1,
        0.539_682_539_951_727_297_e-1,
        0.133_333_333_333_050_058_1,
    );
    u = mla(simd, u, s, splat(simd, 0.333_333_333_333_334_369_5));
    u = mla(simd, s, u * x, x);

    let y = mla(simd, u, u, splat(simd, -1.));
    x = u * splat(simd, -2.);

    let o = (ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));

    u = o.select(-y, x) / o.select(x, y);
    d.simd_eq(splat(simd, 0.)).select(d, u)
}

/// Tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn tan_deterministic<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let dql = rint(simd, d * splat(simd, FRAC_2_PI));
    let mut ql = roundi(simd, dql);
    let mut s = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
    s = mla(simd, dql, splat(simd, -PI_B2 * 0.5), s);
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));

    if !g.all_true() {
        let mut dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        dqh *= splat(simd, D1_24);
        let dql = rint(simd, d * splat(simd, FRAC_2_PI) - dqh);

        let mut u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        u = mla(simd, dql, splat(simd, -PI_A * 0.5), u);
        u = mla(simd, dqh, splat(simd, -PI_B * 0.5), u);
        u = mla(simd, dql, splat(simd, -PI_B * 0.5), u);
        u = mla(simd, dqh, splat(simd, -PI_C * 0.5), u);
        u = mla(simd, dql, splat(simd, -PI_C * 0.5), u);
        u = mla(simd, dqh + dql, splat(simd, -PI_D * 0.5), u);

        ql = g.select(ql, roundi(simd, dql));
        s = g.select(s, u);
        let g = d.abs().simd_lt(splat(simd, 1e+6));

        if !g.all_true() {
            let (ddidd, ddii) = rempi(simd, d);
            let ql2 = ddii;
            let mut u = ddidd.0 + ddidd.1;
            u = from_bits(
                simd,
                (d.is_infinite() | d.is_nan())
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    | to_bits(simd, u),
            );

            ql = g.select(ql, ql2);
            s = g.select(s, u);
        }
    }

    let x = s * splat(simd, 0.5);
    let s = x * x;

    let s2 = s * s;
    let s4 = s2 * s2;

    let mut u = poly8(
        simd,
        s,
        s2,
        s4,
        0.324_509_882_663_927_631_6_e-3,
        0.561_921_973_811_432_373_5_e-3,
        0.146_078_150_240_278_449_4_e-2,
        0.359_161_154_079_249_951_9_e-2,
        0.886_326_840_956_311_312_6_e-2,
        0.218_694_872_818_553_549_8_e-1,
        0.539_682_539_951_727_297_e-1,
        0.133_333_333_333_050_058_1,
    );
    u = mla(simd, u, s, splat(simd, 0.333_333_333_333_334_369_5));
    u = mla(simd, s, u * x, x);

    let y = mla(simd, u, u, splat(simd, -1.));
    let x = u * splat(simd, -2.);

    let o = (ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
    u = o.select(-y, x) / o.select(x, y);
    d.simd_eq(splat(simd, 0.)).select(d, u)
}

/// Evaluates `sin(pi * x)` and `cos(pi * x)` simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `3.5 ULP` for `x` in
/// `[-1e+7, 1e+7]`; outside that range an arbitrary value within `[-1, 1]` is returned. For a
/// `NaN` or infinite `x`, a `NaN` is returned.
#[inline(always)]
pub fn sincospi<S: Simd>(simd: S, d: F64s<S>) -> (F64s<S>, F64s<S>) {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q + bits_to_int(
        simd,
        (int_to_bits(simd, q) >> splat_bits(simd, 63)) ^ splat_bits(simd, 1),
    )) & splat_int(simd, !1);
    let s = u - q.to_float::<F64s<S>>();

    let t = s;
    let s = s * s;

    let mut u = splat(simd, 0.688_063_889_476_606_013_6_e-11);
    u = mla(simd, u, s, splat(simd, -0.175_715_956_454_231_019_9_e-8));
    u = mla(simd, u, s, splat(simd, 0.313_361_632_725_786_731_1_e-6));
    u = mla(simd, u, s, splat(simd, -0.365_762_041_638_848_645_2_e-4));
    u = mla(simd, u, s, splat(simd, 0.249_039_457_018_993_210_3_e-2));
    u = mla(simd, u, s, splat(simd, -0.807_455_121_882_805_632_e-1));
    u = mla(simd, u, s, splat(simd, 0.785_398_163_397_448_279));

    let rx = u * t;

    let mut u = splat(simd, -0.386_014_121_368_379_435_2_e-12);
    u = mla(simd, u, s, splat(simd, 0.115_005_788_802_968_141_5_e-9));
    u = mla(simd, u, s, splat(simd, -0.246_113_649_300_666_355_3_e-7));
    u = mla(simd, u, s, splat(simd, 0.359_086_044_662_351_671_3_e-5));
    u = mla(simd, u, s, splat(simd, -0.325_991_886_926_943_594_2_e-3));
    u = mla(simd, u, s, splat(simd, 0.158_543_442_438_154_116_9_e-1));
    u = mla(simd, u, s, splat(simd, -0.308_425_137_534_042_437_3));
    u = mla(simd, u, s, splat(simd, 1.));

    let ry = u;

    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (q & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, rsin),
    );

    let o = ((q + splat_int(simd, 2)) & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, rcos),
    );

    let o = d.abs().simd_gt(splat(simd, TRIGRANGEMAX3 / 4.));
    rsin = from_bits(
        simd,
        !o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) & to_bits(simd, rsin),
    );
    rcos = from_bits(
        simd,
        !o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) & to_bits(simd, rcos),
    );

    let o = d.is_infinite();
    rsin = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, rsin),
    );
    rcos = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Arc tangent of `y / x`, with the quadrant taken from the signs of `y` and `x`.
#[inline(always)]
fn atan2k<S: Simd>(simd: S, y: F64s<S>, x: F64s<S>) -> F64s<S> {
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
    let t8 = t4 * t4;
    let t16 = t8 * t8;

    let u = poly19(
        simd,
        t,
        t2,
        t4,
        t8,
        t16,
        -1.887_960_084_630_734_965_637_46_e-05,
        0.000_209_850_076_645_816_976_906_797,
        -0.001_106_118_314_866_724_825_634_71,
        0.003_700_267_441_887_131_192_324_03,
        -0.008_898_961_958_876_554_917_408_09,
        0.016_599_329_773_529_201_970_117,
        -0.025_451_762_493_231_264_161_686_1,
        0.033_785_258_000_135_306_999_389_7,
        -0.040_762_919_127_683_650_000_193_4,
        0.046_666_715_007_784_062_563_267_5,
        -0.052_367_485_230_348_245_761_611_3,
        0.058_766_639_292_667_358_085_431_3,
        -0.066_657_357_936_108_052_598_456_2,
        0.076_921_953_831_176_961_835_502_9,
        -0.090_908_995_008_245_008_229_153,
        0.111_111_105_648_261_418_443_745,
        -0.142_857_142_667_713_293_837_65,
        0.199_999_999_996_591_265_594_148,
        -0.333_333_333_333_311_110_369_124,
    );

    t = mla(simd, s, t * u, s);
    mla(simd, q.to_float::<F64s<S>>(), splat(simd, FRAC_PI_2), t)
}

/// Arc tangent function of two variables.
///
/// Evaluates the arc tangent of `y / x`; the quadrant of the result is determined from the signs
/// of `x` and `y`. The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn atan2<S: Simd>(simd: S, y: F64s<S>, x: F64s<S>) -> F64s<S> {
    let mut r = atan2k(simd, y.abs(), x);

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
        (x.is_nan() | y.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, mul_sign(simd, r, y)),
    )
}

/// Arc sine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn asin<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.) - d.abs()) * splat(simd, 0.5));
    let x = o.select(d.abs(), x2.sqrt());

    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let x16 = x8 * x8;

    let mut u = poly12(
        simd,
        x2,
        x4,
        x8,
        x16,
        0.316_158_765_065_393_462_8_e-1,
        -0.158_191_824_332_999_664_3_e-1,
        0.192_904_547_726_791_067_4_e-1,
        0.660_607_747_627_717_061_e-2,
        0.121_536_052_557_737_733_1_e-1,
        0.138_871_518_450_160_921_8_e-1,
        0.173_595_699_122_361_460_4_e-1,
        0.223_717_618_193_204_834_1_e-1,
        0.303_819_592_803_813_223_7_e-1,
        0.446_428_568_137_710_243_8_e-1,
        0.750_000_000_037_858_161_1_e-1,
        0.166_666_666_666_649_754_3,
    );

    u = mla(simd, u, x * x2, x);

    let r = o.select(u, mla(simd, u, splat(simd, -2.), splat(simd, FRAC_PI_2)));
    mul_sign(simd, r, d)
}

/// Arc cosine function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn acos<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.) - d.abs()) * splat(simd, 0.5));
    let mut x = o.select(d.abs(), x2.sqrt());
    x = d.abs().simd_eq(splat(simd, 1.)).select(splat(simd, 0.), x);

    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let x16 = x8 * x8;

    let mut u = poly12(
        simd,
        x2,
        x4,
        x8,
        x16,
        0.316_158_765_065_393_462_8_e-1,
        -0.158_191_824_332_999_664_3_e-1,
        0.192_904_547_726_791_067_4_e-1,
        0.660_607_747_627_717_061_e-2,
        0.121_536_052_557_737_733_1_e-1,
        0.138_871_518_450_160_921_8_e-1,
        0.173_595_699_122_361_460_4_e-1,
        0.223_717_618_193_204_834_1_e-1,
        0.303_819_592_803_813_223_7_e-1,
        0.446_428_568_137_710_243_8_e-1,
        0.750_000_000_037_858_161_1_e-1,
        0.166_666_666_666_649_754_3,
    );

    u *= x2 * x;

    let y = splat(simd, FRAC_PI_2) - (mul_sign(simd, x, d) + mul_sign(simd, u, d));
    x += u;
    let r = o.select(y, x * splat(simd, 2.));
    (!o & d.simd_lt(splat(simd, 0.))).select(add_checked_v(DD::splat(simd, D_PI), -r).0, r)
}

/// Arc tangent function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn atan<S: Simd>(simd: S, mut s: F64s<S>) -> F64s<S> {
    let q = s
        .is_sign_negative()
        .select(splat_int(simd, 2), splat_int(simd, 0));
    s = s.abs();

    let q = splat(simd, 1.).simd_lt(s).select(q + splat_int(simd, 1), q);
    s = splat(simd, 1.).simd_lt(s).select(splat(simd, 1.) / s, s);

    let mut t = s * s;

    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let t16 = t8 * t8;

    let u = poly19(
        simd,
        t,
        t2,
        t4,
        t8,
        t16,
        -1.887_960_084_630_734_965_637_46_e-5,
        0.000_209_850_076_645_816_976_906_797,
        -0.001_106_118_314_866_724_825_634_71,
        0.003_700_267_441_887_131_192_324_03,
        -0.008_898_961_958_876_554_917_408_09,
        0.016_599_329_773_529_201_970_117,
        -0.025_451_762_493_231_264_161_686_1,
        0.033_785_258_000_135_306_999_389_7,
        -0.040_762_919_127_683_650_000_193_4,
        0.046_666_715_007_784_062_563_267_5,
        -0.052_367_485_230_348_245_761_611_3,
        0.058_766_639_292_667_358_085_431_3,
        -0.066_657_357_936_108_052_598_456_2,
        0.076_921_953_831_176_961_835_502_9,
        -0.090_908_995_008_245_008_229_153,
        0.111_111_105_648_261_418_443_745,
        -0.142_857_142_667_713_293_837_65,
        0.199_999_999_996_591_265_594_148,
        -0.333_333_333_333_311_110_369_124,
    );

    t = mla(simd, s, t * u, s);

    t = (q & splat_int(simd, 1))
        .simd_eq(splat_int(simd, 1))
        .select(splat(simd, FRAC_PI_2) - t, t);
    t = from_bits(
        simd,
        ((q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, t),
    );

    t
}

/// Hyperbolic sine function.
///
/// The error bound of the returned value is `3.5 ULP` for `x` in `[-709, 709]`. Outside that range
/// infinity with the correct sign, or a correct value within `3.5 ULP`, is returned.
#[inline(always)]
pub fn sinh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let e = expm1k(simd, x.abs());

    let mut y = (e + splat(simd, 2.)) / (e + splat(simd, 1.));
    y *= splat(simd, 0.5) * e;

    y = (x.abs().simd_gt(splat(simd, 709.)) | y.is_nan()).select(splat(simd, f64::INFINITY), y);
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Hyperbolic cosine function.
///
/// The error bound of the returned value is `3.5 ULP` for `x` in `[-709, 709]`. Outside that range
/// infinity, or a correct value within `3.5 ULP`, is returned.
#[inline(always)]
pub fn cosh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let e = u10::exp(simd, x.abs());
    let mut y = mla(simd, splat(simd, 0.5), e, splat(simd, 0.5) / e);

    y = (x.abs().simd_gt(splat(simd, 709.)) | y.is_nan()).select(splat(simd, f64::INFINITY), y);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Base-2 logarithmic function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn log2<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let o = d.simd_lt(splat(simd, f64::MIN_POSITIVE));
    d = o.select(d * splat(simd, D1_32 * D1_32), d);
    let e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
    let m = ldexp3k(simd, d, -e);
    let e = o.select(e - splat_int(simd, 64), e);

    let x = (m - splat(simd, 1.)) / (m + splat(simd, 1.));
    let x2 = x * x;

    let mut t = splat(simd, 0.221_194_175_045_608_149);
    t = mla(simd, t, x2, splat(simd, 0.220_076_869_315_227_768_9));
    t = mla(simd, t, x2, splat(simd, 0.262_370_805_748_851_465_6));
    t = mla(simd, t, x2, splat(simd, 0.320_597_747_794_449_550_2));
    t = mla(simd, t, x2, splat(simd, 0.412_198_594_548_532_470_9));
    t = mla(simd, t, x2, splat(simd, 0.577_078_016_299_705_898_2));
    t = mla(simd, t, x2, splat(simd, 0.961_796_693_926_080_914_49));

    let s: DD<S> = v_add_checked(
        e.to_float::<F64s<S>>(),
        mul_as_doubled(x, splat(simd, 2.885_390_081_777_926_774)),
    );

    let mut r = mla(simd, t, x * x2, s.0 + s.1);

    r = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), r);
    r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f64::NAN), r);
    r = d
        .simd_eq(splat(simd, 0.))
        .select(splat(simd, f64::NEG_INFINITY), r);

    r
}

/// Base-10 exponential function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn exp10<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut u = rint(simd, d * splat(simd, LOG10_2));
    let q = roundi(simd, u);

    let s = mla(simd, u, splat(simd, -L10_U), d);
    let s = mla(simd, u, splat(simd, -L10_L), s);

    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    u = poly11(
        simd,
        s,
        s2,
        s4,
        s8,
        0.241_146_349_833_426_765_2_e-3,
        0.115_748_841_521_718_737_5_e-2,
        0.501_397_554_678_973_365_9_e-2,
        0.195_976_232_072_053_308_e-1,
        0.680_893_639_944_678_413_8_e-1,
        0.206_995_849_472_267_623_4,
        0.539_382_929_205_853_622_9,
        0.117_125_514_890_854_165_5_e+1,
        0.203_467_859_229_343_295_3_e+1,
        0.265_094_905_523_920_587_6_e+1,
        0.230_258_509_299_404_590_1_e+1,
    );

    u = mla(simd, u, s, splat(simd, 1.));

    u = ldexp2k(simd, u, q);

    u = d
        .simd_gt(splat(simd, 308.254_715_559_916_71))
        .select(splat(simd, f64::INFINITY), u);
    from_bits(
        simd,
        !d.simd_lt(splat(simd, -350.))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Base-2 exponential function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn exp2<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut u = rint(simd, d);
    let q = roundi(simd, u);

    let s = d - u;

    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    u = poly10(
        simd,
        s,
        s2,
        s4,
        s8,
        0.443_435_908_292_652_945_4_e-9,
        0.707_316_459_808_570_742_5_e-8,
        0.101_781_926_092_176_045_1_e-6,
        0.132_154_387_251_132_761_5_e-5,
        0.152_527_335_351_758_473_e-4,
        0.154_035_304_510_114_780_8_e-3,
        0.133_335_581_467_049_907_3_e-2,
        0.961_812_910_759_760_053_6_e-2,
        0.555_041_086_648_204_659_6_e-1,
        0.240_226_506_959_101_221_4,
    );
    u = mla(simd, u, s, splat(simd, 0.693_147_180_559_945_286_2));

    u = mla(simd, u, s, splat(simd, 1.));

    u = ldexp2k(simd, u, q);

    u = d
        .simd_ge(splat(simd, 1024.))
        .select(splat(simd, f64::INFINITY), u);
    from_bits(
        simd,
        !d.simd_lt(splat(simd, -2000.))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Square root function.
///
/// The error bound of the returned value is `3.5 ULP`.
///
/// Upstream's `u35::sqrt` forwards to `u05::sqrt`; this crate does not carry `f64x::u05` yet, so
/// the split (non-FMA) formulation of upstream's `u05::sqrt` — the one the port's target-feature
/// policy selects — is written out here. Replace the body with a call to `u05::sqrt` once that
/// module lands.
#[inline(always)]
pub fn sqrt<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let d = d.simd_lt(splat(simd, 0.)).select(splat(simd, f64::NAN), d);

    let o = d.simd_lt(splat(simd, 8.636_168_555_094_445_e-78));
    let d = o.select(d * splat(simd, 1.157_920_892_373_162_e77), d);
    let q = o.select(
        splat(simd, 2.938_735_877_055_718_8_e-39 * 0.5),
        splat(simd, 0.5),
    );

    let o = d.simd_gt(splat(simd, 1.340_780_792_994_259_7_e+154));
    let d = o.select(d * splat(simd, 7.458_340_731_200_207_e-155), d);
    let q = o.select(splat(simd, 1.157_920_892_373_162_e+77 * 0.5), q);

    let mut x = from_bits(
        simd,
        int_to_bits(
            simd,
            splat_int(simd, 0x5fe6_ec86_0000_0000_u64 as i64)
                - bits_to_int(simd, to_bits(simd, d + splat(simd, 1e-320)) >> 1),
        ),
    );

    x *= splat(simd, 1.5) - splat(simd, 0.5) * d * x * x;
    x *= splat(simd, 1.5) - splat(simd, 0.5) * d * x * x;
    x *= splat(simd, 1.5) - splat(simd, 0.5) * d * x * x;
    x *= d;

    let d2: DD<S> = (DD::new(d, splat(simd, 0.)) + mul_as_doubled(x, x)) * recip_as_doubled(x);

    x = (d2.0 + d2.1) * q;

    x = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), x);
    d.simd_eq(splat(simd, 0.)).select(d, x)
}

/// Cube root function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn cbrt<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let mut q = splat(simd, 1.);
    let e = ilogbk(simd, d.abs()) + splat_int(simd, 1);
    d = ldexp2k(simd, d, -e);

    let t = e.to_float::<F64s<S>>() + splat(simd, 6144.);
    let qu = trunci(simd, t * splat(simd, 1. / 3.));
    let re = trunci(simd, t - (qu.to_float::<F64s<S>>() * splat(simd, 3.)));

    q = re
        .simd_eq(splat_int(simd, 1))
        .select(splat(simd, 1.259_921_049_894_873_164_767_210_6), q);
    q = re
        .simd_eq(splat_int(simd, 2))
        .select(splat(simd, 1.587_401_051_968_199_474_751_705_6), q);
    q = ldexp2k(simd, q, qu - splat_int(simd, 2048));

    q = mul_sign(simd, q, d);

    d = d.abs();

    let mut x = splat(simd, -0.640_245_898_480_692_909_870_982);
    x = mla(simd, x, d, splat(simd, 2.961_551_030_200_395_118_185_95));
    x = mla(simd, x, d, splat(simd, -5.733_530_609_229_478_436_361_66));
    x = mla(simd, x, d, splat(simd, 6.039_903_689_894_587_479_614_07));
    x = mla(simd, x, d, splat(simd, -3.858_419_355_104_449_888_216_32));
    x = mla(simd, x, d, splat(simd, 2.230_727_530_249_660_972_572_2));

    let mut y = x * x;
    y = y * y;
    x -= mul_sub(simd, d, y, x) * splat(simd, 1. / 3.);
    y = d * x * x;
    y = (y - splat(simd, 2. / 3.) * y * mla(simd, y, x, splat(simd, -1.))) * q;

    y
}

/// 2D Euclidean distance function.
///
/// The error bound of the returned value is `3.5 ULP`.
#[inline(always)]
pub fn hypot<S: Simd>(simd: S, x: F64s<S>, y: F64s<S>) -> F64s<S> {
    let x = x.abs();
    let y = y.abs();
    let min = x.min(y);
    let max = x.max(y);

    let t = min / max;
    let mut ret = max * mla(simd, t, t, splat(simd, 1.)).sqrt();
    ret = min.simd_eq(splat(simd, 0.)).select(max, ret);
    ret = (x.is_nan() | y.is_nan()).select(splat(simd, f64::NAN), ret);
    (x.simd_eq(splat(simd, f64::INFINITY)) | y.simd_eq(splat(simd, f64::INFINITY)))
        .select(splat(simd, f64::INFINITY), ret)
}
