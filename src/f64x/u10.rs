//! `f64` kernels with a 1.0 ULP error bound (SLEEF's `u10` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x/u10.rs`).

use core::f64::consts::{FRAC_1_PI, FRAC_2_PI, FRAC_PI_2, FRAC_PI_4, PI};

use super::*;
use crate::doubled::f64::{
    add_as_doubled, add_checked, add_checked_as_doubled, add_checked_v, mul_as_doubled, mul_as_f,
    mul_v, normalize, recip, scale, sqrt, square, square_as_f, sub_checked, v_add_checked, DD,
};
// `add_v` arrives through `super::*`.

/// Natural exponential function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn exp<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
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
        0.208_127_637_823_716_445_7_e-8,
        0.251_121_070_304_228_802_2_e-7,
        0.275_576_262_816_949_119_2_e-6,
        0.275_572_340_202_538_823_9_e-5,
        0.248_015_868_747_968_626_4_e-4,
        0.198_412_698_985_586_585_e-3,
        0.138_888_888_891_449_779_7_e-2,
        0.833_333_333_331_493_821_e-2,
        0.416_666_666_666_660_259_8_e-1,
        0.166_666_666_666_666_907_2,
    )
    .mul_add(s, splat(simd, 0.5))
    .mul_add(s, splat(simd, 1.0))
    .mul_add(s, splat(simd, 1.0));

    u = ldexp2k(simd, u, q);

    u = d
        .simd_gt(splat(simd, 709.782_711_149_557_429_092_172_174_26))
        .select(splat(simd, f64::INFINITY), u);
    from_bits(
        simd,
        d.simd_lt(splat(simd, -1000.))
            .select(splat_bits(simd, 0), splat_bits(simd, u64::MAX))
            & to_bits(simd, u),
    )
}

/// `exp(x) - 1` without the final subtraction, as a double-double.
#[inline(always)]
pub fn expk2<S: Simd>(simd: S, d: DD<S>) -> DD<S> {
    let u = d.0 * splat(simd, R_LN2);
    let dq = rint(simd, u);
    let q = roundi(simd, dq);

    let dq = DD::new(dq, splat(simd, 0.));
    let s = (d + mul_v(dq, splat(simd, -L2_U))) + mul_v(dq, splat(simd, -L2_L));

    let s2 = square(s);
    let s4 = square(s2);
    let s8 = s4.0 * s4.0;

    let u = poly10(
        simd,
        s.0,
        s2.0,
        s4.0,
        s8,
        0.160_247_221_970_993_207_2_e-9,
        0.209_225_518_356_315_700_7_e-8,
        0.250_523_002_378_264_446_5_e-7,
        0.275_572_480_090_213_530_3_e-6,
        0.275_573_189_238_604_437_3_e-5,
        0.248_015_873_560_581_506_5_e-4,
        0.198_412_698_414_807_185_8_e-3,
        0.138_888_888_888_676_325_5_e-2,
        0.833_333_333_333_334_709_5_e-2,
        0.416_666_666_666_666_990_5_e-1,
    );

    let mut t = v_add_checked(
        splat(simd, 0.5),
        mul_v(s, splat(simd, 0.166_666_666_666_666_657_4)),
    );
    t = v_add_checked(splat(simd, 1.0), t * s);
    t = v_add_checked(splat(simd, 1.0), t * s);
    t = add_checked(t, mul_v(s4, u));

    t = DD::new(ldexp2k(simd, t.0, q), ldexp2k(simd, t.1, q));

    let keep =
        d.0.simd_lt(splat(simd, -1000.))
            .select(splat_bits(simd, 0), splat_bits(simd, u64::MAX));
    DD::new(
        from_bits(simd, keep & to_bits(simd, t.0)),
        from_bits(simd, keep & to_bits(simd, t.1)),
    )
}

/// `exp(x) - 1`.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn expm1<S: Simd>(simd: S, a: F64s<S>) -> F64s<S> {
    let d = add_v(expk2(simd, DD::new(a, splat(simd, 0.))), splat(simd, -1.));
    let mut x = d.0;
    x = a
        .simd_gt(splat(simd, 709.782_712_893_383_996_732_223))
        .select(splat(simd, f64::INFINITY), x);
    x = a
        .simd_lt(splat(simd, -36.736_800_569_677_101_399_113_302_437))
        .select(splat(simd, -1.), x);
    is_neg_zero(simd, a).select(splat(simd, -0.), x)
}

/// Sine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn sin<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut s;
    let mut ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(simd, d * splat(simd, FRAC_1_PI));
        ql = roundi(simd, dql);
        let u = mla(simd, dql, splat(simd, -PI_A2), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_B2));
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX)).all_true() {
        let dqh = trunc(simd, d * splat(simd, FRAC_1_PI / D1_24));
        let dqh = dqh * splat(simd, D1_24);
        let dql = rint(simd, mul_sub(simd, d, splat(simd, FRAC_1_PI), dqh));
        ql = roundi(simd, dql);

        let u = mla(simd, dqh, splat(simd, -PI_A), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_A));
        s = add_v(s, dqh * splat(simd, -PI_B));
        s = add_v(s, dql * splat(simd, -PI_B));
        s = add_v(s, dqh * splat(simd, -PI_C));
        s = add_v(s, dql * splat(simd, -PI_C));
        s = add_v(s, (dqh + dql) * splat(simd, -PI_D));
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
        s = normalize(ddidd);
        s.0 = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, s.0),
        );
    }

    let t = s;
    s = square(s);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;

    let mut u = mla(
        simd,
        poly6(
            simd,
            s.0,
            s2,
            s4,
            2.720_524_161_385_295_679_179_83_e-15,
            -7.642_925_941_139_544_719_002_3_e-13,
            1.605_893_701_172_778_962_116_23_e-10,
            -2.505_210_681_484_312_335_936_8_e-8,
            2.755_731_921_044_282_247_773_79_e-6,
            -0.000_198_412_698_412_046_454_654_947,
        ),
        s.0,
        splat(simd, 0.008_333_333_333_333_180_562_019_22),
    );

    let x = v_add_checked(
        splat(simd, 1.0),
        add_checked_as_doubled(splat(simd, -0.166_666_666_666_666_657_414_808), u * s.0) * s,
    );

    u = mul_as_f(t, x);

    u = from_bits(
        simd,
        ((ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, u),
    );
    d.simd_eq(splat(simd, 0.)).select(d, u)
}

/// Sine function.
///
/// The error bound of the returned value is `1.0 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn sin_deterministic<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut s;
    let mut ql;

    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));
    let dql = rint(simd, d * splat(simd, FRAC_1_PI));
    ql = roundi(simd, dql);
    let u = mla(simd, dql, splat(simd, -PI_A2), d);
    let mut x = add_checked_as_doubled(u, dql * splat(simd, -PI_B2));

    if !g.all_true() {
        let mut dqh = trunc(simd, d * splat(simd, FRAC_1_PI / D1_24));
        dqh *= splat(simd, D1_24);
        let dql = rint(simd, mul_sub(simd, d, splat(simd, FRAC_1_PI), dqh));

        let u = mla(simd, dqh, splat(simd, -PI_A), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_A));
        s = add_v(s, dqh * splat(simd, -PI_B));
        s = add_v(s, dql * splat(simd, -PI_B));
        s = add_v(s, dqh * splat(simd, -PI_C));
        s = add_v(s, dql * splat(simd, -PI_C));
        s = add_checked_v(s, (dqh + dql) * splat(simd, -PI_D));

        ql = g.select(ql, roundi(simd, dql));
        x = select_doubled(simd, g, x, s);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (mut ddidd, ddii) = rempi(simd, d);
            let mut ql2 = ddii & splat_int(simd, 3);
            ql2 = ql2
                + ql2
                + ddidd
                    .0
                    .simd_gt(splat(simd, 0.))
                    .select(splat_int(simd, 2), splat_int(simd, 1));
            ql2 >>= 2;
            let o = (ddii & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
            let mut t = DD::new(
                mul_sign(simd, splat(simd, -D_PI.0 * 0.5), ddidd.0),
                mul_sign(simd, splat(simd, -D_PI.1 * 0.5), ddidd.0),
            );
            t = ddidd + t;
            ddidd = select_doubled(simd, o, t, ddidd);
            s = normalize(ddidd);
            ql = g.select(ql, ql2);
            x = select_doubled(simd, g, x, s);
            x.0 = from_bits(
                simd,
                (d.is_infinite() | d.is_nan())
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    | to_bits(simd, x.0),
            );
        }
    }

    let t = x;
    s = square(x);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;

    let mut u = mla(
        simd,
        poly6(
            simd,
            s.0,
            s2,
            s4,
            2.720_524_161_385_295_679_179_83_e-15,
            -7.642_925_941_139_544_719_002_3_e-13,
            1.605_893_701_172_778_962_116_23_e-10,
            -2.505_210_681_484_312_335_936_8_e-8,
            2.755_731_921_044_282_247_773_79_e-6,
            -0.000_198_412_698_412_046_454_654_947,
        ),
        s.0,
        splat(simd, 0.008_333_333_333_333_180_562_019_22),
    );

    x = v_add_checked(
        splat(simd, 1.0),
        add_checked_as_doubled(splat(simd, -0.166_666_666_666_666_657_414_808), u * s.0) * s,
    );

    u = mul_as_f(t, x);

    u = from_bits(
        simd,
        ((ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, u),
    );

    d.simd_eq(splat(simd, 0.)).select(d, u)
}

/// Cosine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn cos<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut s;
    let mut ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(
            simd,
            mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5)),
        );
        let dql = mla(simd, splat(simd, 2.), dql, splat(simd, 1.0));
        ql = roundi(simd, dql);
        s = add_as_doubled(d, dql * splat(simd, -PI_A2) * splat(simd, 0.5));
        s = add_checked_v(s, dql * splat(simd, -PI_B2) * splat(simd, 0.5));
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

        let u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        s = add_as_doubled(u, dql * splat(simd, -PI_A) * splat(simd, 0.5));
        s = add_v(s, dqh * splat(simd, -PI_B) * splat(simd, 0.5));
        s = add_v(s, dql * splat(simd, -PI_B) * splat(simd, 0.5));
        s = add_v(s, dqh * splat(simd, -PI_C) * splat(simd, 0.5));
        s = add_v(s, dql * splat(simd, -PI_C) * splat(simd, 0.5));
        s = add_checked_v(s, (dqh + dql) * splat(simd, -PI_D) * splat(simd, 0.5));
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
        s = normalize(ddidd);
        s.0 = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, s.0),
        );
    }

    let t = s;
    s = square(s);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;

    let u = mla(
        simd,
        poly6(
            simd,
            s.0,
            s2,
            s4,
            2.720_524_161_385_295_679_179_83_e-15,
            -7.642_925_941_139_544_719_002_3_e-13,
            1.605_893_701_172_778_962_116_23_e-10,
            -2.505_210_681_484_312_335_936_8_e-8,
            2.755_731_921_044_282_247_773_79_e-6,
            -0.000_198_412_698_412_046_454_654_947,
        ),
        s.0,
        splat(simd, 0.008_333_333_333_333_180_562_019_22),
    );

    let x = v_add_checked(
        splat(simd, 1.0),
        add_checked_as_doubled(splat(simd, -0.166_666_666_666_666_657_414_808), u * s.0) * s,
    );

    let u = mul_as_f(t, x);

    from_bits(
        simd,
        ((ql & splat_int(simd, 2)).simd_eq(splat_int(simd, 0)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, u),
    )
}

/// Cosine function.
///
/// The error bound of the returned value is `1.0 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn cos_deterministic<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));
    let mut dql = rint(
        simd,
        mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5)),
    );
    dql = mla(simd, splat(simd, 2.), dql, splat(simd, 1.0));
    let mut ql = roundi(simd, dql);
    let mut x = add_as_doubled(d, dql * splat(simd, -PI_A2 * 0.5));
    x = add_checked_v(x, dql * splat(simd, -PI_B2 * 0.5));

    if !g.all_true() {
        let mut dqh = trunc(
            simd,
            mla(
                simd,
                d,
                splat(simd, FRAC_1_PI / D1_23),
                splat(simd, -FRAC_1_PI / D1_24),
            ),
        );
        let mut ql2 = roundi(
            simd,
            d * splat(simd, FRAC_1_PI) + mla(simd, dqh, splat(simd, -D1_23), splat(simd, -0.5)),
        );
        dqh *= splat(simd, D1_24);
        ql2 = ql2 + ql2 + splat_int(simd, 1);
        let dql = ql2.to_float::<F64s<S>>();

        let u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        let mut s = add_as_doubled(u, dql * splat(simd, -PI_A * 0.5));
        s = add_v(s, dqh * splat(simd, -PI_B * 0.5));
        s = add_v(s, dql * splat(simd, -PI_B * 0.5));
        s = add_v(s, dqh * splat(simd, -PI_C * 0.5));
        s = add_v(s, dql * splat(simd, -PI_C * 0.5));
        s = add_checked_v(s, (dqh + dql) * splat(simd, -PI_D * 0.5));

        ql = g.select(ql, ql2);
        x = select_doubled(simd, g, x, s);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (mut ddidd, ddii) = rempi(simd, d);
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
            let mut t = DD::new(
                mul_sign(simd, splat(simd, -D_PI.0 * 0.5), y),
                mul_sign(simd, splat(simd, -D_PI.1 * 0.5), y),
            );
            t = ddidd + t;
            ddidd = select_doubled(simd, o, t, ddidd);
            s = normalize(ddidd);
            ql = g.select(ql, ql2);
            x = select_doubled(simd, g, x, s);
            x.0 = from_bits(
                simd,
                (d.is_infinite() | d.is_nan())
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    | to_bits(simd, x.0),
            );
        }
    }

    let t = x;
    let s = square(x);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;

    let u = mla(
        simd,
        poly6(
            simd,
            s.0,
            s2,
            s4,
            2.720_524_161_385_295_679_179_83_e-15,
            -7.642_925_941_139_544_719_002_3_e-13,
            1.605_893_701_172_778_962_116_23_e-10,
            -2.505_210_681_484_312_335_936_8_e-8,
            2.755_731_921_044_282_247_773_79_e-6,
            -0.000_198_412_698_412_046_454_654_947,
        ),
        s.0,
        splat(simd, 0.008_333_333_333_333_180_562_019_22),
    );

    x = v_add_checked(
        splat(simd, 1.0),
        add_checked_as_doubled(splat(simd, -0.166_666_666_666_666_657_414_808), u * s.0) * s,
    );

    let u = mul_as_f(t, x);

    from_bits(
        simd,
        ((ql & splat_int(simd, 2)).simd_eq(splat_int(simd, 0)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, u),
    )
}

/// Evaluates sine and cosine simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `1.0 ULP`. If the argument is a
/// `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sincos<S: Simd>(simd: S, d: F64s<S>) -> (F64s<S>, F64s<S>) {
    let mut s;
    let ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(simd, d * splat(simd, FRAC_2_PI));
        ql = roundi(simd, dql);
        let u = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_B2) * splat(simd, 0.5));
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX)).all_true() {
        let dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        let dqh = dqh * splat(simd, D1_24);
        let dql = rint(simd, d * splat(simd, FRAC_2_PI) - dqh);
        ql = roundi(simd, dql);

        let u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_A) * splat(simd, 0.5));
        s = add_v(s, dqh * splat(simd, -PI_B) * splat(simd, 0.5));
        s = add_v(s, dql * splat(simd, -PI_B) * splat(simd, 0.5));
        s = add_v(s, dqh * splat(simd, -PI_C) * splat(simd, 0.5));
        s = add_v(s, dql * splat(simd, -PI_C) * splat(simd, 0.5));
        s = add_v(s, (dqh + dql) * splat(simd, -PI_D) * splat(simd, 0.5));
    } else {
        let (ddidd, ddii) = rempi(simd, d);
        ql = ddii;
        s = ddidd;
        let o = d.is_infinite() | d.is_nan();
        s = DD::new(
            from_bits(
                simd,
                o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, s.0),
            ),
            from_bits(
                simd,
                o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, s.1),
            ),
        );
    }

    let t = s;

    s.0 = square_as_f(s);

    let mut u = mla(
        simd,
        splat(simd, 1.589_383_072_832_289_373_285_11_e-10),
        s.0,
        splat(simd, -2.505_069_435_025_397_733_493_18_e-8),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 2.755_731_317_768_463_605_125_47_e-6),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -0.000_198_412_698_278_911_770_864_914),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.008_333_333_333_319_184_596_174_6),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -0.166_666_666_666_666_130_709_393),
    );
    let u = u * (s.0 * t.0);

    let x = add_checked_v(t, u);
    let rx = x.0 + x.1;

    let rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let mut u = mla(
        simd,
        splat(simd, -1.136_153_502_390_974_295_315_23_e-11),
        s.0,
        splat(simd, 2.087_574_712_070_400_554_793_66_e-9),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -2.755_731_440_288_475_674_985_67_e-7),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 2.480_158_728_900_018_673_119_15_e-5),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -0.001_388_888_888_887_140_192_823_29),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.041_666_666_666_666_551_959_206_2),
    );
    u = mla(simd, u, s.0, splat(simd, -0.5));

    let x: DD<S> = v_add_checked(splat(simd, 1.0), mul_as_doubled(s.0, u));
    let ry = x.0 + x.1;

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
/// Returns `(sin, cos)`. The error bound of the returned values is `1.0 ULP`. If the argument is a
/// `NaN` or infinity, a `NaN` is returned.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn sincos_deterministic<S: Simd>(simd: S, d: F64s<S>) -> (F64s<S>, F64s<S>) {
    let dql = rint(simd, d * splat(simd, FRAC_2_PI));
    let mut ql = roundi(simd, dql);
    let u = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
    let mut s = add_checked_as_doubled(u, dql * splat(simd, -PI_B2 * 0.5));
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));

    if !g.all_true() {
        let mut dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        dqh *= splat(simd, D1_24);
        let dql = rint(simd, d * splat(simd, FRAC_2_PI) - dqh);

        let u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        let mut x = add_checked_as_doubled(u, dql * splat(simd, -PI_A * 0.5));
        x = add_v(x, dqh * splat(simd, -PI_B * 0.5));
        x = add_v(x, dql * splat(simd, -PI_B * 0.5));
        x = add_v(x, dqh * splat(simd, -PI_C * 0.5));
        x = add_v(x, dql * splat(simd, -PI_C * 0.5));
        x = add_checked_v(x, (dqh + dql) * splat(simd, -PI_D * 0.5));

        ql = g.select(ql, roundi(simd, dql));
        s = select_doubled(simd, g, s, x);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (ddidd, ddii) = rempi(simd, d);
            let mut x = ddidd;
            let o = d.is_infinite() | d.is_nan();
            x = DD::new(
                from_bits(
                    simd,
                    o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, x.0),
                ),
                from_bits(
                    simd,
                    o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, x.1),
                ),
            );

            ql = g.select(ql, ddii);
            s = select_doubled(simd, g, s, x);
        }
    }

    let t = s;

    s.0 = square_as_f(s);

    let mut u = mla(
        simd,
        splat(simd, 1.589_383_072_832_289_373_285_11_e-10),
        s.0,
        splat(simd, -2.505_069_435_025_397_733_493_18_e-8),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 2.755_731_317_768_463_605_125_47_e-6),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -0.000_198_412_698_278_911_770_864_914),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.008_333_333_333_319_184_596_174_6),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -0.166_666_666_666_666_130_709_393),
    );
    let u = u * (s.0 * t.0);

    let x = add_checked_v(t, u);
    let mut rx = x.0 + x.1;

    rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let mut u = mla(
        simd,
        splat(simd, -1.136_153_502_390_974_295_315_23_e-11),
        s.0,
        splat(simd, 2.087_574_712_070_400_554_793_66_e-9),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -2.755_731_440_288_475_674_985_67_e-7),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 2.480_158_728_900_018_673_119_15_e-5),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, -0.001_388_888_888_887_140_192_823_29),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.041_666_666_666_666_551_959_206_2),
    );
    u = mla(simd, u, s.0, splat(simd, -0.5));

    let x: DD<S> = v_add_checked(splat(simd, 1.0), mul_as_doubled(s.0, u));
    let ry = x.0 + x.1;

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
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn tan<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut s;
    let ql;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2)).all_true() {
        let dql = rint(simd, d * splat(simd, FRAC_2_PI));
        ql = roundi(simd, dql);
        let u = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_B2) * splat(simd, 0.5));
    } else if d.abs().simd_lt(splat(simd, TRIGRANGEMAX)).all_true() {
        let dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        let dqh = dqh * splat(simd, D1_24);
        s = add_v(
            mul_v(
                DD::splat(
                    simd,
                    (
                        0.636_619_772_367_581_382_43,
                        -3.935_735_335_036_497_176_4_e-17,
                    ),
                ),
                d,
            ),
            d.simd_lt(splat(simd, 0.))
                .select(splat(simd, -0.5), splat(simd, 0.5))
                - dqh,
        );
        let dql = trunc(simd, s.0 + s.1);
        ql = roundi(simd, dql);

        let u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        s = add_checked_as_doubled(u, dql * splat(simd, -PI_A) * splat(simd, 0.5));
        s = add_v(s, dqh * splat(simd, -PI_B) * splat(simd, 0.5));
        s = add_v(s, dql * splat(simd, -PI_B) * splat(simd, 0.5));
        s = add_v(s, dqh * splat(simd, -PI_C) * splat(simd, 0.5));
        s = add_v(s, dql * splat(simd, -PI_C) * splat(simd, 0.5));
        s = add_v(s, (dqh + dql) * splat(simd, -PI_D) * splat(simd, 0.5));
    } else {
        let (ddidd, ddii) = rempi(simd, d);
        ql = ddii;
        s = ddidd;
        let o = d.is_infinite() | d.is_nan();
        s = DD::new(
            from_bits(
                simd,
                o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, s.0),
            ),
            from_bits(
                simd,
                o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, s.1),
            ),
        );
    }

    let t = scale(s, splat(simd, 0.5));
    s = square(t);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;
    let u = mla(
        simd,
        poly8(
            simd,
            s.0,
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
        ),
        s.0,
        splat(simd, 0.333_333_333_333_334_369_5),
    );
    let mut x = add_checked(t, mul_v(s * t, u));

    let y = v_add_checked(-splat(simd, 1.0), square(x));
    x = scale(x, splat(simd, -2.));

    let o = (ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));

    x = select_doubled(simd, o, -y, x) / select_doubled(simd, o, x, y);

    let u = x.0 + x.1;

    d.simd_eq(splat(simd, 0.)).select(d, u)
}

/// Tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn tan_deterministic<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let dql = rint(simd, d * splat(simd, FRAC_2_PI));
    let mut ql = roundi(simd, dql);
    let u = mla(simd, dql, splat(simd, -PI_A2 * 0.5), d);
    let mut s = add_checked_as_doubled(u, dql * splat(simd, -PI_B2 * 0.5));
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2));

    if !g.all_true() {
        let mut dqh = trunc(simd, d * splat(simd, FRAC_2_PI / D1_24));
        dqh *= splat(simd, D1_24);
        let mut x = add_v(
            mul_v(
                DD::splat(
                    simd,
                    (
                        0.636_619_772_367_581_382_43,
                        -3.935_735_335_036_497_176_4_e-17,
                    ),
                ),
                d,
            ),
            d.simd_lt(splat(simd, 0.))
                .select(splat(simd, -0.5), splat(simd, 0.5))
                - dqh,
        );
        let dql = trunc(simd, x.0 + x.1);

        let u = mla(simd, dqh, splat(simd, -PI_A * 0.5), d);
        x = add_checked_as_doubled(u, dql * splat(simd, -PI_A * 0.5));
        x = add_v(x, dqh * splat(simd, -PI_B * 0.5));
        x = add_v(x, dql * splat(simd, -PI_B * 0.5));
        x = add_v(x, dqh * splat(simd, -PI_C * 0.5));
        x = add_v(x, dql * splat(simd, -PI_C * 0.5));
        x = add_checked_v(x, (dqh + dql) * splat(simd, -PI_D * 0.5));

        ql = g.select(ql, roundi(simd, dql));
        s = select_doubled(simd, g, s, x);
        let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX));

        if !g.all_true() {
            let (ddidd, ddii) = rempi(simd, d);
            x = ddidd;
            let o = d.is_infinite() | d.is_nan();
            x = DD::new(
                from_bits(
                    simd,
                    o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, x.0),
                ),
                from_bits(
                    simd,
                    o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) | to_bits(simd, x.1),
                ),
            );

            ql = g.select(ql, ddii);
            s = select_doubled(simd, g, s, x);
        }
    }

    let t = scale(s, splat(simd, 0.5));
    s = square(t);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;
    let u = mla(
        simd,
        poly8(
            simd,
            s.0,
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
        ),
        s.0,
        splat(simd, 0.333_333_333_333_334_369_5),
    );
    let mut x = add_checked(t, mul_v(s * t, u));

    let y = v_add_checked(-splat(simd, 1.0), square(x));
    x = scale(x, splat(simd, -2.));

    let o = (ql & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));

    x = select_doubled(simd, o, -y, x) / select_doubled(simd, o, x, y);

    let u = x.0 + x.1;

    d.simd_eq(splat(simd, 0.)).select(d, u)
}

/// Arc tangent of `y / x` with 1.0 ULP accuracy, kept as a double-double.
#[inline(always)]
fn atan2k_u1<S: Simd>(simd: S, y: DD<S>, mut x: DD<S>) -> DD<S> {
    let q =
        x.0.is_sign_negative()
            .select(splat_int(simd, -2), splat_int(simd, 0));
    let p = x.0.simd_lt(splat(simd, 0.));
    let b = p.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
        & splat_bits(simd, (-0.0f64).to_bits());
    x = DD::new(
        from_bits(simd, b ^ to_bits(simd, x.0)),
        from_bits(simd, b ^ to_bits(simd, x.1)),
    );

    let q = x.0.simd_lt(y.0).select(q + splat_int(simd, 1), q);
    let p = x.0.simd_lt(y.0);
    let s = select_doubled(simd, p, -x, y);
    let mut t = select_doubled(simd, p, y, x);

    let s = s / t;
    t = square(s);
    t = normalize(t);

    let t2 = t.0 * t.0;
    let t4 = t2 * t2;
    let t8 = t4 * t4;

    let u = poly16(
        simd,
        t.0,
        t2,
        t4,
        t8,
        1.062_984_841_914_487_466_074_15_e-05,
        -0.000_125_620_649_967_286_867_384_336,
        0.000_705_576_642_963_934_123_897_74,
        -0.002_518_656_144_987_133_603_529_99,
        0.006_462_628_990_369_911_723_135_04,
        -0.012_828_133_366_339_903_101_427_4,
        0.020_802_479_992_414_579_790_249_7,
        -0.028_900_234_478_474_031_568_628_9,
        0.035_978_500_503_510_459_085_365_6,
        -0.041_848_579_703_592_507_506_027,
        0.047_084_301_165_328_398_819_376_3,
        -0.052_491_421_058_844_842_106_871_9,
        0.058_794_659_096_958_100_386_043_4,
        -0.066_662_088_477_879_549_719_418_2,
        0.076_922_533_029_620_376_865_409_5,
        -0.090_909_044_277_338_757_478_190_7,
    );
    let mut u = mla(simd, u, t.0, splat(simd, 0.111_111_108_376_896_236_538_123));
    u = mla(
        simd,
        u,
        t.0,
        splat(simd, -0.142_857_142_756_268_568_062_339),
    );
    u = mla(simd, u, t.0, splat(simd, 0.199_999_999_997_977_351_284_817));
    u = mla(
        simd,
        u,
        t.0,
        splat(simd, -0.333_333_333_333_317_605_173_818),
    );

    t = add_checked(s, mul_v(s * t, u));
    add_checked(
        mul_v(
            DD::new(
                splat(simd, 1.570_796_326_794_896_557_998_982),
                splat(simd, 6.123_233_995_736_766_035_868_82_e-17),
            ),
            q.to_float::<F64s<S>>(),
        ),
        t,
    )
}

/// Arc tangent function of two variables.
///
/// Evaluates the arc tangent of `y / x`; the quadrant of the result is determined from the signs
/// of `x` and `y`. The error bound of the returned value is `max(1.0 ULP, f64::MIN_POSITIVE)`.
#[inline(always)]
pub fn atan2<S: Simd>(simd: S, y: F64s<S>, x: F64s<S>) -> F64s<S> {
    let o = x
        .abs()
        .simd_lt(splat(simd, 5.562_684_646_268_008_398_4_e-309)); // nexttoward((1.0 / DBL_MAX), 1)
    let x = o.select(x * splat(simd, D1_53), x);
    let y = o.select(y * splat(simd, D1_23), y);

    let d = atan2k_u1(
        simd,
        DD::new(y.abs(), splat(simd, 0.)),
        DD::new(x, splat(simd, 0.)),
    );
    let mut r = d.0 + d.1;

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
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn asin<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.0) - d.abs()) * splat(simd, 0.5));
    let mut x = select_doubled(
        simd,
        o,
        DD::new(d.abs(), splat(simd, 0.)),
        sqrt_as_doubled(simd, x2),
    );
    x = select_doubled(
        simd,
        d.abs().simd_eq(splat(simd, 1.0)),
        DD::new(splat(simd, 0.), splat(simd, 0.)),
        x,
    );

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

    u *= x2 * x.0;

    let y = sub_checked(
        sub_checked(DD::splat(simd, (D_PI.0 / 4., D_PI.1 / 4.)), x),
        DD::new(u, splat(simd, 0.)),
    );

    let r = o.select(u + x.0, (y.0 + y.1) * splat(simd, 2.));
    mul_sign(simd, r, d)
}

/// Arc cosine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn acos<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.0) - d.abs()) * splat(simd, 0.5));
    let mut x = select_doubled(
        simd,
        o,
        DD::new(d.abs(), splat(simd, 0.)),
        sqrt_as_doubled(simd, x2),
    );
    x = select_doubled(
        simd,
        d.abs().simd_eq(splat(simd, 1.0)),
        DD::new(splat(simd, 0.), splat(simd, 0.)),
        x,
    );

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

    u *= x2 * x.0;

    let mut y = sub_checked(
        DD::splat(simd, (D_PI.0 / 2., D_PI.1 / 2.)),
        add_checked_as_doubled(mul_sign(simd, x.0, d), mul_sign(simd, u, d)),
    );
    x = add_checked_v(x, u);

    y = select_doubled(simd, o, y, scale(x, splat(simd, 2.)));

    y = select_doubled(
        simd,
        !o & d.simd_lt(splat(simd, 0.)),
        sub_checked(DD::splat(simd, D_PI), y),
        y,
    );

    y.0 + y.1
}

/// Arc tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn atan<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let d2 = atan2k_u1(
        simd,
        DD::new(d.abs(), splat(simd, 0.)),
        DD::new(splat(simd, 1.0), splat(simd, 0.)),
    );
    let mut r = d2.0 + d2.1;
    r = d
        .is_infinite()
        .select(splat(simd, 1.570_796_326_794_896_557_998_982), r);
    mul_sign(simd, r, d)
}

/// Hyperbolic sine function.
///
/// The error bound of the returned value is `1.0 ULP` for `x` in `[-709, 709]`. Outside that range
/// infinity with the correct sign, or a correct value within `1.0 ULP`, is returned.
#[inline(always)]
pub fn sinh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut y = x.abs();
    let mut d = expk2(simd, DD::new(y, splat(simd, 0.)));
    d = sub_checked(d, recip(d));
    y = (d.0 + d.1) * splat(simd, 0.5);

    y = (x.abs().simd_gt(splat(simd, 710.)) | y.is_nan()).select(splat(simd, f64::INFINITY), y);
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
/// The error bound of the returned value is `1.0 ULP` for `x` in `[-709, 709]`. Outside that range
/// infinity, or a correct value within `1.0 ULP`, is returned.
#[inline(always)]
pub fn cosh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut y = x.abs();
    let mut d = expk2(simd, DD::new(y, splat(simd, 0.)));
    d = add_checked(d, recip(d));
    y = (d.0 + d.1) * splat(simd, 0.5);

    y = (x.abs().simd_gt(splat(simd, 710.)) | y.is_nan()).select(splat(simd, f64::INFINITY), y);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Hyperbolic tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn tanh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut y = x.abs();
    let mut d = expk2(simd, DD::new(y, splat(simd, 0.)));
    let e = recip(d);
    d = (d + (-e)) / (d + e);
    y = d.0 + d.1;

    y = (x.abs().simd_gt(splat(simd, 18.714_973_875)) | y.is_nan()).select(splat(simd, 1.), y);
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Natural logarithm of a double-double, kept as a double-double.
#[inline(always)]
fn logk2<S: Simd>(simd: S, d: DD<S>) -> DD<S> {
    let e = ilogbk(simd, d.0 * splat(simd, 1. / 0.75));

    let m = DD::new(ldexp2k(simd, d.0, -e), ldexp2k(simd, d.1, -e));

    let x = add_v(m, splat(simd, -1.)) / add_v(m, splat(simd, 1.));
    let x2 = square(x);

    let x4 = x2.0 * x2.0;
    let x8 = x4 * x4;

    let t = mla(
        simd,
        poly7(
            simd,
            x2.0,
            x4,
            x8,
            0.138_604_363_904_671_679_108_56,
            0.131_699_838_841_615_374_240_845,
            0.153_914_168_346_271_945_653_214,
            0.181_816_523_941_564_611_721_589,
            0.222_222_246_326_620_354_039_96,
            0.285_714_285_511_134_091_777_308,
            0.400_000_000_000_914_013_309_483,
        ),
        x2.0,
        splat(simd, 0.666_666_666_666_664_853_302_393),
    );

    let mut s = mul_v(DD::splat(simd, D_LN2), e.to_float::<F64s<S>>());
    s = add_checked(s, scale(x, splat(simd, 2.)));
    add_checked(s, mul_v(x2 * x, t))
}

/// Inverse hyperbolic sine function.
///
/// The error bound of the returned value is `1.0 ULP` for `x` in `[-1.34e+154, 1.34e+154]`.
/// Outside that range infinity with the correct sign, or a correct value within `1.0 ULP`, is
/// returned.
#[inline(always)]
pub fn asinh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut y = x.abs();
    let o = y.simd_gt(splat(simd, 1.0));

    let mut d = select_doubled(simd, o, recip_as_doubled(x), DD::new(y, splat(simd, 0.)));
    d = sqrt(add_v(square(d), splat(simd, 1.0)));
    d = select_doubled(simd, o, mul_v(d, y), d);

    d = logk2(simd, normalize(add_v(d, x)));
    y = d.0 + d.1;

    y = (x.abs().simd_gt(splat(simd, SQRT_DBL_MAX)) | y.is_nan())
        .select(mul_sign(simd, splat(simd, f64::INFINITY), x), y);

    y = from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    );
    is_neg_zero(simd, x).select(splat(simd, -0.), y)
}

/// Inverse hyperbolic cosine function.
///
/// The error bound of the returned value is `1.0 ULP` for `x` in `[-1.34e+154, 1.34e+154]`.
/// Outside that range infinity with the correct sign, or a correct value within `1.0 ULP`, is
/// returned.
#[inline(always)]
pub fn acosh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let d = logk2(
        simd,
        add_v(
            sqrt(add_as_doubled(x, splat(simd, 1.0))) * sqrt(add_as_doubled(x, splat(simd, -1.))),
            x,
        ),
    );
    let mut y = d.0 + d.1;

    y = (x.abs().simd_gt(splat(simd, SQRT_DBL_MAX)) | y.is_nan())
        .select(splat(simd, f64::INFINITY), y);
    y = from_bits(
        simd,
        !x.simd_eq(splat(simd, 1.))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & to_bits(simd, y),
    );

    y = from_bits(
        simd,
        x.simd_lt(splat(simd, 1.))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    );
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Inverse hyperbolic tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn atanh<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut y = x.abs();
    let d = logk2(
        simd,
        add_as_doubled(splat(simd, 1.0), y) / add_as_doubled(splat(simd, 1.0), -y),
    );
    y = from_bits(
        simd,
        y.simd_gt(splat(simd, 1.))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(
                simd,
                y.simd_eq(splat(simd, 1.))
                    .select(splat(simd, f64::INFINITY), (d.0 + d.1) * splat(simd, 0.5)),
            ),
    );

    y = mul_sign(simd, y, x);
    y = from_bits(
        simd,
        (x.is_infinite() | y.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    );
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Natural logarithmic function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let m;

    let mut s = {
        let o = d.simd_lt(splat(simd, f64::MIN_POSITIVE));
        d = o.select(d * splat(simd, D1_32 * D1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        mul_v(DD::splat(simd, D_LN2), e.to_float::<F64s<S>>())
    };

    let x = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.0), m);
    let x2 = x.0 * x.0;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let t = poly7(
        simd,
        x2,
        x4,
        x8,
        0.153_207_698_850_270_135_3,
        0.152_562_905_100_342_871_6,
        0.181_860_593_293_778_599_6,
        0.222_221_451_983_938_000_9,
        0.285_714_293_279_429_931_7,
        0.399_999_999_963_525_199,
        0.666_666_666_666_733_354_1,
    );

    s = add_checked(s, scale(x, splat(simd, 2.)));
    s = add_checked_v(s, x2 * x.0 * t);

    let r = s.0 + s.1;

    let r = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), r);
    let r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f64::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f64::NEG_INFINITY), r)
}

/// Base-10 logarithmic function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log10<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let m;

    let mut s: DD<S> = {
        let o = d.simd_lt(splat(simd, f64::MIN_POSITIVE));
        d = o.select(d * splat(simd, D1_32 * D1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        mul_v(
            DD::new(
                splat(simd, 0.301_029_995_663_981_198_02),
                splat(simd, -2.803_728_127_785_170_339_e-18),
            ),
            e.to_float::<F64s<S>>(),
        )
    };

    let x = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.0), m);
    let x2 = x.0 * x.0;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let t = poly7(
        simd,
        x2,
        x4,
        x8,
        0.665_372_581_957_675_846_e-1,
        0.662_572_278_282_083_371_2_e-1,
        0.789_810_521_431_394_407_8_e-1,
        0.965_095_503_571_527_513_2_e-1,
        0.124_084_140_972_144_499_3,
        0.173_717_792_745_460_508_6,
        0.289_529_654_602_197_261_7,
    );

    s = add_checked(
        s,
        x * DD::new(
            splat(simd, 0.868_588_963_806_503_633_34),
            splat(simd, 1.143_005_969_409_638_931_1_e-17),
        ),
    );
    s = add_checked_v(s, x2 * x.0 * t);

    let r = s.0 + s.1;

    let r = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), r);
    let r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f64::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f64::NEG_INFINITY), r)
}

/// Base-2 logarithmic function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log2<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let m;

    let ef = {
        let o = d.simd_lt(splat(simd, f64::MIN_POSITIVE));
        d = o.select(d * splat(simd, D1_32 * D1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        e.to_float::<F64s<S>>()
    };

    let x: DD<S> = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.0), m);
    let x2 = x.0 * x.0;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let t = poly7(
        simd,
        x2,
        x4,
        x8,
        0.221_194_175_045_608_149,
        0.220_076_869_315_227_768_9,
        0.262_370_805_748_851_465_6,
        0.320_597_747_794_449_550_2,
        0.412_198_594_548_532_470_9,
        0.577_078_016_299_705_898_2,
        0.961_796_693_926_080_914_49,
    );

    let mut s = v_add_checked(
        ef,
        x * DD::new(
            splat(simd, 2.885_390_081_777_926_774),
            splat(simd, 6.056_160_499_551_673_643_4_e-18),
        ),
    );
    s = add_v(s, x2 * x.0 * t);

    let r = s.0 + s.1;

    let r = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), r);
    let r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f64::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f64::NEG_INFINITY), r)
}

/// Logarithm of one plus the argument.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log1p<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let m;

    let mut dp1 = d + splat(simd, 1.0);

    let mut s = {
        let o = dp1.simd_lt(splat(simd, f64::MIN_POSITIVE));
        dp1 = o.select(dp1 * splat(simd, D1_32 * D1_32), dp1);
        let mut e = ilogb2k(simd, dp1 * splat(simd, 1. / 0.75));
        let t = ldexp3k(simd, splat(simd, 1.0), -e);
        m = mla(simd, d, t, t - splat(simd, 1.));
        e = o.select(e - splat_int(simd, 64), e);
        mul_v(DD::splat(simd, D_LN2), e.to_float::<F64s<S>>())
    };

    let x = DD::new(m, splat(simd, 0.)) / add_checked_as_doubled(splat(simd, 2.), m);
    let x2 = x.0 * x.0;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let t = poly7(
        simd,
        x2,
        x4,
        x8,
        0.153_207_698_850_270_135_3,
        0.152_562_905_100_342_871_6,
        0.181_860_593_293_778_599_6,
        0.222_221_451_983_938_000_9,
        0.285_714_293_279_429_931_7,
        0.399_999_999_963_525_199,
        0.666_666_666_666_733_354_1,
    );

    s = add_checked(s, scale(x, splat(simd, 2.)));
    s = add_checked_v(s, x2 * x.0 * t);

    let mut r = s.0 + s.1;

    r = d
        .simd_gt(splat(simd, 1e+307))
        .select(splat(simd, f64::INFINITY), r);
    r = (d.simd_lt(splat(simd, -1.)) | d.is_nan()).select(splat(simd, f64::NAN), r);
    r = d
        .simd_eq(splat(simd, -1.))
        .select(splat(simd, f64::NEG_INFINITY), r);
    is_neg_zero(simd, d).select(splat(simd, -0.), r)
}

/// Base-10 exponential function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn exp10<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let u = rint(simd, d * splat(simd, LOG10_2));
    let q = roundi(simd, u);

    let s = mla(simd, u, splat(simd, -L10_U), d);
    let s = mla(simd, u, splat(simd, -L10_L), s);

    let mut u = mla(
        simd,
        splat(simd, 0.241_146_349_833_426_765_2_e-3),
        s,
        splat(simd, 0.115_748_841_521_718_737_5_e-2),
    );
    u = mla(simd, u, s, splat(simd, 0.501_397_554_678_973_365_9_e-2));
    u = mla(simd, u, s, splat(simd, 0.195_976_232_072_053_308_e-1));
    u = mla(simd, u, s, splat(simd, 0.680_893_639_944_678_413_8_e-1));
    u = mla(simd, u, s, splat(simd, 0.206_995_849_472_267_623_4));
    u = mla(simd, u, s, splat(simd, 0.539_382_929_205_853_622_9));
    u = mla(simd, u, s, splat(simd, 0.117_125_514_890_854_165_5_e+1));
    u = mla(simd, u, s, splat(simd, 0.203_467_859_229_343_295_3_e+1));
    u = mla(simd, u, s, splat(simd, 0.265_094_905_523_920_587_6_e+1));
    u = mla(simd, u, s, splat(simd, 0.230_258_509_299_404_590_1_e+1));

    u = normalize(v_add_checked(splat(simd, 1.0), mul_as_doubled::<S>(u, s))).0;

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
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn exp2<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let u = rint(simd, d);
    let q = roundi(simd, u);

    let s = d - u;

    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    let mut u = mla(
        simd,
        poly10(
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
        ),
        s,
        splat(simd, 0.693_147_180_559_945_286_2),
    );

    u = normalize(v_add_checked(splat(simd, 1.0), mul_as_doubled::<S>(u, s))).0;

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

/// Exponential function of a double-double, collapsed to the leading part.
#[inline(always)]
fn expk<S: Simd>(simd: S, d: DD<S>) -> F64s<S> {
    let mut u = (d.0 + d.1) * splat(simd, R_LN2);
    let dq = rint(simd, u);
    let q = roundi(simd, dq);

    let mut s = add_v(d, dq * splat(simd, -L2_U));
    s = add_v(s, dq * splat(simd, -L2_L));

    s = normalize(s);

    let s2 = s.0 * s.0;
    let s4 = s2 * s2;
    let s8 = s4 * s4;

    u = poly10(
        simd,
        s.0,
        s2,
        s4,
        s8,
        2.510_696_834_209_504_195_271_39_e-8,
        2.762_861_667_702_706_491_168_55_e-7,
        2.755_724_967_250_235_741_438_64_e-6,
        2.480_149_739_898_197_941_141_53_e-5,
        0.000_198_412_698_809_069_797_676_111,
        0.001_388_888_893_997_712_896_052_9,
        0.008_333_333_333_323_714_176_010_81,
        0.041_666_666_666_540_952_412_844_9,
        0.166_666_666_666_666_740_681_535,
        0.500_000_000_000_000_999_200_722,
    );

    let mut t = v_add_checked(splat(simd, 1.0), s);
    t = add_checked(t, mul_v(square(s), u));

    u = t.0 + t.1;
    u = ldexp2k(simd, u, q);

    from_bits(
        simd,
        !d.0.simd_lt(splat(simd, -1000.))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Natural logarithm, kept as a double-double.
#[inline(always)]
fn logk<S: Simd>(simd: S, mut d: F64s<S>) -> DD<S> {
    let m;

    let mut s = {
        let o = d.simd_lt(splat(simd, f64::MIN_POSITIVE));
        d = o.select(d * splat(simd, D1_32 * D1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        mul_v(DD::splat(simd, D_LN2), e.to_float::<F64s<S>>())
    };

    let x: DD<S> = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.0), m);
    let x2 = square(x);

    let x4 = x2.0 * x2.0;
    let x8 = x4 * x4;
    let x16 = x8 * x8;

    let t = poly9(
        simd,
        x2.0,
        x4,
        x8,
        x16,
        0.116_255_524_079_935_043_668_677,
        0.103_239_680_901_072_952_701_192,
        0.117_754_809_412_463_995_466_069,
        0.133_329_810_868_462_739_215_09,
        0.153_846_227_114_512_262_845_736,
        0.181_818_180_850_050_775_676_507,
        0.222_222_222_230_083_560_345_903,
        0.285_714_285_714_249_172_087_875,
        0.400_000_000_000_000_077_715_612,
    );

    let c = DD::new(
        splat(simd, 0.666_666_666_666_666_629_659_233),
        splat(simd, 3.805_549_625_424_120_563_366_16_e-17),
    );

    s = add_checked(s, scale(x, splat(simd, 2.)));
    add_checked(s, x2 * x * (mul_v(x2, t) + c))
}

/// Power function.
///
/// Returns `x` raised to the power of `y`. The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn pow<S: Simd>(simd: S, x: F64s<S>, y: F64s<S>) -> F64s<S> {
    if true {
        let yisint = is_integer(simd, y);
        let yisodd = is_odd(simd, y) & yisint;

        let d = mul_v(logk(simd, x.abs()), y);
        let mut result = expk(simd, d);
        result =
            d.0.simd_gt(splat(simd, 709.782_711_149_557_429_092_172_174_26))
                .select(splat(simd, f64::INFINITY), result);

        result *= x.simd_gt(splat(simd, 0.)).select(
            splat(simd, 1.),
            yisint.select(
                yisodd.select(splat(simd, -1.), splat(simd, 1.)),
                splat(simd, f64::NAN),
            ),
        );

        let efx = mul_sign(simd, x.abs() - splat(simd, 1.), y);

        result = y.is_infinite().select(
            from_bits(
                simd,
                !efx.simd_lt(splat(simd, 0.))
                    .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                    & to_bits(
                        simd,
                        efx.simd_eq(splat(simd, 0.))
                            .select(splat(simd, 1.), splat(simd, f64::INFINITY)),
                    ),
            ),
            result,
        );

        result = (x.is_infinite() | x.simd_eq(splat(simd, 0.))).select(
            mul_sign(
                simd,
                (y.is_sign_negative() ^ x.simd_eq(splat(simd, 0.)))
                    .select(splat(simd, 0.), splat(simd, f64::INFINITY)),
                yisodd.select(x, splat(simd, 1.)),
            ),
            result,
        );

        result = from_bits(
            simd,
            (x.is_nan() | y.is_nan()).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                | to_bits(simd, result),
        );

        (y.simd_eq(splat(simd, 0.)) | x.simd_eq(splat(simd, 1.))).select(splat(simd, 1.), result)
    } else {
        expk(simd, mul_v(logk(simd, x), y))
    }
}

/// Cube root function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn cbrt<S: Simd>(simd: S, mut d: F64s<S>) -> F64s<S> {
    let mut q2 = DD::new(splat(simd, 1.), splat(simd, 0.));

    let e = ilogbk(simd, d.abs()) + splat_int(simd, 1);
    d = ldexp2k(simd, d, -e);

    let t = e.to_float::<F64s<S>>() + splat(simd, 6144.);
    let qu = trunci(simd, t * splat(simd, 1. / 3.));
    let re = trunci(simd, t - qu.to_float::<F64s<S>>() * splat(simd, 3.));

    q2 = select_doubled(
        simd,
        re.simd_eq(splat_int(simd, 1)),
        DD::new(
            splat(simd, 1.259_921_049_894_873_190_7),
            splat(simd, -2.589_933_375_300_506_917_7_e-17),
        ),
        q2,
    );
    q2 = select_doubled(
        simd,
        re.simd_eq(splat_int(simd, 2)),
        DD::new(
            splat(simd, 1.587_401_051_968_199_583_4),
            splat(simd, -1.086_900_819_419_782_298_6_e-16),
        ),
        q2,
    );

    q2.0 = mul_sign(simd, q2.0, d);
    q2.1 = mul_sign(simd, q2.1, d);
    d = d.abs();

    let mut x = mla(
        simd,
        splat(simd, -0.640_245_898_480_692_909_870_982),
        d,
        splat(simd, 2.961_551_030_200_395_118_185_95),
    );
    x = mla(simd, x, d, splat(simd, -5.733_530_609_229_478_436_361_66));
    x = mla(simd, x, d, splat(simd, 6.039_903_689_894_587_479_614_07));
    x = mla(simd, x, d, splat(simd, -3.858_419_355_104_449_888_216_32));
    x = mla(simd, x, d, splat(simd, 2.230_727_530_249_660_972_572_2));

    let mut y = x * x;
    y = y * y;
    x -= mul_sub(simd, d, y, x) * splat(simd, 1. / 3.);

    let z = x;

    let mut u = mul_as_doubled::<S>(x, x);
    u = u * u;
    u = mul_v(u, d);
    u = add_v(u, -x);
    y = u.0 + u.1;

    y = splat(simd, -2. / 3.) * y * z;
    let mut v = add_v(mul_as_doubled::<S>(z, z), y);
    v = mul_v(v, d);
    v *= q2;
    let mut z = ldexp2k(simd, v.0 + v.1, qu - splat_int(simd, 2048));

    z = d
        .is_infinite()
        .select(mul_sign(simd, splat(simd, f64::INFINITY), q2.0), z);
    d.simd_eq(splat(simd, 0.))
        .select(from_bits(simd, sign_bit(simd, q2.0)), z)
}

/// `sin(pi * d)` for a reduced argument, kept as a double-double.
///
/// Upstream keeps this in `f64x.rs` rather than in the `u10` tier; `gammak` needs it, so it is
/// carried here.
#[inline(always)]
fn sinpik<S: Simd>(simd: S, d: F64s<S>) -> DD<S> {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 63)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));

    let s = u - q.to_float::<F64s<S>>();
    let t = s;
    let s = s * s;
    let s2 = mul_as_doubled(t, t);

    let mut u = mla(
        simd,
        select_splat(
            simd,
            o,
            9.944_803_876_268_437_740_902_08_e-16,
            -2.024_611_207_851_823_992_958_68_e-14,
        ),
        s,
        select_splat(
            simd,
            o,
            -3.897_962_260_629_327_991_640_47_e-13,
            6.948_218_305_801_794_613_277_84_e-12,
        ),
    );
    u = mla(
        simd,
        u,
        s,
        select_splat(
            simd,
            o,
            1.150_115_825_399_960_352_669_01_e-10,
            -1.757_247_499_528_531_799_526_64_e-9,
        ),
    );
    u = mla(
        simd,
        u,
        s,
        select_splat(
            simd,
            o,
            -2.461_136_950_104_469_749_535_9_e-8,
            3.133_616_889_668_683_928_784_22_e-7,
        ),
    );
    u = mla(
        simd,
        u,
        s,
        select_splat(
            simd,
            o,
            3.590_860_448_590_527_540_050_62_e-6,
            -3.657_620_418_216_155_192_036_1_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        s,
        select_splat(
            simd,
            o,
            -0.000_325_991_886_927_389_905_997_954,
            0.002_490_394_570_192_718_502_743_56,
        ),
    );

    let mut x = v_add_checked(
        u * s,
        select_doubled(
            simd,
            o,
            DD::new(
                splat(simd, 0.015_854_344_243_815_501_891_425_9),
                splat(simd, -1.046_932_722_806_315_219_088_45_e-18),
            ),
            DD::new(
                splat(simd, -0.080_745_512_188_280_785_248_473_1),
                splat(simd, 3.618_524_750_670_371_048_499_87_e-18),
            ),
        ),
    );
    x = s2 * x
        + select_doubled(
            simd,
            o,
            DD::new(
                splat(simd, -0.308_425_137_534_042_437_259_529),
                splat(simd, -1.956_984_921_336_335_503_383_45_e-17),
            ),
            DD::new(
                splat(simd, 0.785_398_163_397_448_278_999_491),
                splat(simd, 3.062_871_137_271_550_026_071_05_e-17),
            ),
        );

    x *= select_doubled(simd, o, s2, DD::new(t, splat(simd, 0.)));
    x = select_doubled(simd, o, add_v(x, splat(simd, 1.0)), x);

    let o = (q & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    x.0 = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, x.0),
    );
    x.1 = from_bits(
        simd,
        o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f64).to_bits())
            ^ to_bits(simd, x.1),
    );

    x
}

/// Gamma function core: returns the double-double pair used by [`tgamma`] and [`lgamma`].
fn gammak<S: Simd>(simd: S, a: F64s<S>) -> (DD<S>, DD<S>) {
    let mut clln = DD::new(splat(simd, 1.), splat(simd, 0.));
    let mut clld = DD::new(splat(simd, 1.), splat(simd, 0.));

    let otiny = a.abs().simd_lt(splat(simd, 1e-306));
    let oref = a.simd_lt(splat(simd, 0.5));

    let mut x = select_doubled(
        simd,
        otiny,
        DD::new(splat(simd, 0.), splat(simd, 0.)),
        select_doubled(
            simd,
            oref,
            add_as_doubled(splat(simd, 1.0), -a),
            DD::new(a, splat(simd, 0.)),
        ),
    );

    let o0 = splat(simd, 0.5).simd_le(x.0) & x.0.simd_le(splat(simd, 1.1));
    let o2 = splat(simd, 2.3).simd_le(x.0);

    let mut y = normalize(add_v(x, splat(simd, 1.0)) * x);
    y = normalize(add_v(x, splat(simd, 2.)) * y);
    y = normalize(add_v(x, splat(simd, 3.)) * y);
    y = normalize(add_v(x, splat(simd, 4.)) * y);

    let o = o2 & x.0.simd_le(splat(simd, 7.));
    clln = select_doubled(simd, o, y, clln);

    x = select_doubled(simd, o, add_v(x, splat(simd, 5.)), x);

    let t = o2.select(
        splat(simd, 1.) / x.0,
        normalize(add_v(x, select_splat(simd, o0, -1., -2.))).0,
    );

    let mut u = select3(
        simd,
        o2,
        o0,
        -156.801_412_704_022_726_379_848_862,
        0.294_791_677_282_761_419_6_e+2,
        0.707_481_600_086_460_927_9_e-7,
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            1.120_804_464_289_911_606_838_558_16,
            0.128_145_969_182_782_010_9_e+3,
            0.400_924_433_300_873_044_3_e-6,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            13.397_985_455_142_589_218_333_060_2,
            0.261_754_402_578_451_504_3_e+3,
            0.104_011_464_162_824_694_6_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.116_546_276_599_463_200_848_033_357,
            0.328_702_285_568_579_043_2_e+3,
            0.150_834_915_073_332_916_7_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -1.391_801_093_265_337_481_495_562_41,
            0.281_814_586_773_034_818_6_e+3,
            0.128_814_307_493_390_102_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.015_056_113_040_026_424_412_918_973_4,
            0.172_867_041_467_355_960_5_e+3,
            0.474_416_774_988_499_393_7_e-6,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.179_540_117_061_234_856_098_844_714,
            0.774_873_576_403_041_681_7_e+2,
            -0.655_481_630_654_248_990_2_e-7,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.002_481_743_600_264_997_730_942_489_28,
            0.251_285_664_308_093_075_2_e+2,
            -0.318_925_247_145_259_984_4_e-6,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.029_527_880_945_699_120_504_851_034_1,
            0.576_679_210_614_007_686_8_e+1,
            0.135_888_382_147_035_537_7_e-6,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.000_540_164_767_892_604_515_196_325_186,
            0.727_027_547_399_618_057_1,
            -0.434_393_127_715_733_604_e-6,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.006_403_362_833_808_069_794_787_256_2,
            0.839_670_912_457_914_780_9_e-1,
            0.972_478_589_740_677_955_5_e-6,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.000_162_516_262_783_915_816_896_611_252,
            -0.821_155_866_974_680_459_5_e-1,
            -0.203_688_605_722_596_601_1_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.001_914_438_498_565_477_526_465_972_39,
            0.682_883_182_834_188_445_8_e-1,
            0.437_336_314_181_972_581_5_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            7.204_895_416_020_010_558_983_115_17_e-5,
            -0.771_248_133_996_167_151_1_e-1,
            -0.943_995_126_830_400_867_7_e-5,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.000_839_498_720_672_087_279_971_000_786,
            0.833_749_202_301_731_495_7_e-1,
            0.205_072_703_037_638_980_4_e-4,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -5.171_790_908_260_592_193_293_944_22_e-5,
            -0.909_496_493_145_624_251_8_e-1,
            -0.449_262_018_343_118_401_8_e-4,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.000_592_166_437_353_693_882_857_342_347,
            0.100_099_631_357_592_935_8,
            0.994_575_123_607_187_593_1_e-4,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            6.972_813_758_365_857_774_037_435_39_e-5,
            -0.111_334_286_154_420_772_4,
            -0.223_154_759_903_498_319_6_e-3,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.000_784_039_221_720_066_627_493_314_301,
            0.125_509_667_321_302_087_5,
            0.509_669_524_710_196_762_2_e-3,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.000_229_472_093_621_399_176_949_318_732,
            -0.144_049_896_784_305_436_8,
            -0.119_275_391_166_788_697_1_e-2,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.002_681_327_160_493_827_160_473_958_490,
            0.169_557_177_004_194_981_1,
            0.289_051_033_074_221_031_e-2,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.003_472_222_222_222_222_222_175_164_840,
            -0.207_385_551_028_409_276_2,
            -0.738_555_102_867_446_185_8_e-2,
        ),
    );
    u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.083_333_333_333_333_333_335_592_087_900,
            0.270_580_808_427_781_593_9,
            0.205_808_084_277_845_533_5_e-1,
        ),
    );

    let mut y = add_v(x, splat(simd, -0.5)) * logk2(simd, x);
    y += -x;
    y += DD::new(
        splat(simd, 0.918_938_533_204_672_780_56), // 0.5*log(2*M_PI)
        splat(simd, -3.878_294_158_067_241_449_8_e-17),
    );

    let mut z = add_v(
        mul_as_doubled(u, t),
        select_splat(
            simd,
            o0,
            -0.400_685_634_386_531_486_2,
            -0.673_523_010_531_981_020_1_e-1,
        ),
    );
    z = add_v(
        mul_v(z, t),
        select_splat(
            simd,
            o0,
            0.822_467_033_424_113_203,
            0.322_467_033_424_113_203,
        ),
    );
    z = add_v(
        mul_v(z, t),
        select_splat(
            simd,
            o0,
            -0.577_215_664_901_532_865_5,
            0.422_784_335_098_467_134_5,
        ),
    );
    z = mul_v(z, t);

    let mut clc = select_doubled(simd, o2, y, z);

    clld = select_doubled(
        simd,
        o2,
        add_v(mul_as_doubled(u, t), splat(simd, 1.0)),
        clld,
    );

    y = clln;

    clc = select_doubled(
        simd,
        otiny,
        DD::new(
            splat(simd, 83.177_661_667_193_433_459_033_3), // log(2^120)
            splat(simd, 3.671_034_596_315_685_072_218_78_e-15),
        ),
        select_doubled(
            simd,
            oref,
            DD::new(
                splat(simd, 1.144_729_885_849_400_163_9), // log(M_PI)
                splat(simd, 1.026_595_116_270_782_638_e-17),
            ) + (-clc),
            clc,
        ),
    );
    clln = select_doubled(
        simd,
        otiny,
        DD::new(splat(simd, 1.), splat(simd, 0.)),
        select_doubled(simd, oref, clln, clld),
    );

    if !(!oref).all_true() {
        let t = a - splat(simd, D1_28)
            * trunci(simd, a * splat(simd, 1.0 / D1_28)).to_float::<F64s<S>>();
        x = clld * sinpik(simd, t);
    }

    clld = select_doubled(
        simd,
        otiny,
        DD::new(
            a * (splat(simd, D1_60) * splat(simd, D1_60)),
            splat(simd, 0.),
        ),
        select_doubled(simd, oref, x, y),
    );

    (clc, clln / clld)
}

/// Gamma function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn tgamma<S: Simd>(simd: S, a: F64s<S>) -> F64s<S> {
    let (da, db) = gammak(simd, a);
    let y = expk2(simd, da) * db;
    let r = y.0 + y.1;
    let o = a.simd_eq(splat(simd, f64::NEG_INFINITY))
        | (a.simd_lt(splat(simd, 0.)) & is_integer(simd, a))
        | (a.is_finite() & a.simd_lt(splat(simd, 0.)) & r.is_nan());
    let r = o.select(splat(simd, f64::NAN), r);

    let o = ((a.simd_eq(splat(simd, f64::INFINITY)) | a.is_finite())
        & a.simd_ge(splat(simd, -f64::MIN_POSITIVE)))
        & (a.simd_eq(splat(simd, 0.)) | a.simd_gt(splat(simd, 200.)) | r.is_nan());
    o.select(mul_sign(simd, splat(simd, f64::INFINITY), a), r)
}

/// Log gamma function.
///
/// The error bound of the returned value is `1.0 ULP` if the argument is positive. If the argument
/// is larger than `2e+305`, it may return infinity instead of the correct value. The error bound is
/// `max(1 ULP and 1e-15)`, if the argument is negative.
#[inline(always)]
pub fn lgamma<S: Simd>(simd: S, a: F64s<S>) -> F64s<S> {
    let (da, db) = gammak(simd, a);
    // `Doubled::abs`, which upstream keeps on the type rather than as a free function.
    let db = DD::new(
        fabs(simd, db.0),
        from_bits(
            simd,
            to_bits(simd, db.1) ^ (to_bits(simd, db.0) & splat_bits(simd, (-0.0f64).to_bits())),
        ),
    );
    let y = da + logk2(simd, db);
    let r = y.0 + y.1;

    let o = a.is_infinite()
        | (a.simd_le(splat(simd, 0.)) & is_integer(simd, a))
        | (a.is_finite() & r.is_nan());
    o.select(splat(simd, f64::INFINITY), r)
}

/// `z + y * x`, all in double-double.
#[inline(always)]
fn ddmla<S: Simd>(simd: S, x: F64s<S>, y: DD<S>, z: DD<S>) -> DD<S> {
    let _ = simd;
    z + mul_v(y, x)
}

/// Degree-2 double-double polynomial with double-double coefficients.
#[inline(always)]
fn poly2dd_b<S: Simd>(simd: S, x: F64s<S>, c1: DD<S>, c0: DD<S>) -> DD<S> {
    ddmla(simd, x, c1, c0)
}

/// Degree-2 double-double polynomial with a vector leading coefficient.
#[inline(always)]
fn poly2dd<S: Simd>(simd: S, x: F64s<S>, c1: F64s<S>, c0: DD<S>) -> DD<S> {
    ddmla(simd, x, DD::new(c1, splat(simd, 0.)), c0)
}

/// Degree-4 double-double polynomial.
#[inline(always)]
fn poly4dd<S: Simd>(simd: S, x: F64s<S>, c3: F64s<S>, c2: DD<S>, c1: DD<S>, c0: DD<S>) -> DD<S> {
    ddmla(
        simd,
        x * x,
        poly2dd(simd, x, c3, c2),
        poly2dd_b(simd, x, c1, c0),
    )
}

// Upstream's `Poly<Self>` instance, where every coefficient is a vector (`Poly::c2v(c) = c`),
// sitting alongside the shared scalar-coefficient `polyN` helpers in `super`. `erf` needs it for the
// branch whose coefficients differ per lane.

/// Vector-coefficient counterpart of [`poly2`].
#[inline(always)]
fn poly2v<S: Simd>(simd: S, x: F64s<S>, c1: F64s<S>, c0: F64s<S>) -> F64s<S> {
    mla(simd, x, c1, c0)
}

/// Vector-coefficient counterpart of [`poly4`].
#[inline(always)]
fn poly4v<S: Simd>(
    simd: S,
    x: F64s<S>,
    x2: F64s<S>,
    c3: F64s<S>,
    c2: F64s<S>,
    c1: F64s<S>,
    c0: F64s<S>,
) -> F64s<S> {
    mla(simd, x2, poly2v(simd, x, c3, c2), poly2v(simd, x, c1, c0))
}

/// Vector-coefficient counterpart of [`poly5`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn poly5v<S: Simd>(
    simd: S,
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    c4: F64s<S>,
    c3: F64s<S>,
    c2: F64s<S>,
    c1: F64s<S>,
    c0: F64s<S>,
) -> F64s<S> {
    mla(simd, x4, c4, poly4v(simd, x, x2, c3, c2, c1, c0))
}

/// Vector-coefficient counterpart of [`poly8`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn poly8v<S: Simd>(
    simd: S,
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    c7: F64s<S>,
    c6: F64s<S>,
    c5: F64s<S>,
    c4: F64s<S>,
    c3: F64s<S>,
    c2: F64s<S>,
    c1: F64s<S>,
    c0: F64s<S>,
) -> F64s<S> {
    mla(
        simd,
        x4,
        poly4v(simd, x, x2, c7, c6, c5, c4),
        poly4v(simd, x, x2, c3, c2, c1, c0),
    )
}

/// Vector-coefficient counterpart of [`poly16`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn poly16v<S: Simd>(
    simd: S,
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    cf: F64s<S>,
    ce: F64s<S>,
    cd: F64s<S>,
    cc: F64s<S>,
    cb: F64s<S>,
    ca: F64s<S>,
    c9: F64s<S>,
    c8: F64s<S>,
    c7: F64s<S>,
    c6: F64s<S>,
    c5: F64s<S>,
    c4: F64s<S>,
    c3: F64s<S>,
    c2: F64s<S>,
    c1: F64s<S>,
    c0: F64s<S>,
) -> F64s<S> {
    mla(
        simd,
        x8,
        poly8v(simd, x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
        poly8v(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Vector-coefficient counterpart of [`poly21`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn poly21v<S: Simd>(
    simd: S,
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    x16: F64s<S>,
    d4: F64s<S>,
    d3: F64s<S>,
    d2: F64s<S>,
    d1: F64s<S>,
    d0: F64s<S>,
    cf: F64s<S>,
    ce: F64s<S>,
    cd: F64s<S>,
    cc: F64s<S>,
    cb: F64s<S>,
    ca: F64s<S>,
    c9: F64s<S>,
    c8: F64s<S>,
    c7: F64s<S>,
    c6: F64s<S>,
    c5: F64s<S>,
    c4: F64s<S>,
    c3: F64s<S>,
    c2: F64s<S>,
    c1: F64s<S>,
    c0: F64s<S>,
) -> F64s<S> {
    mla(
        simd,
        x16,
        poly5v(simd, x, x2, x4, d4, d3, d2, d1, d0),
        poly16v(
            simd, x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}

/// Error function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn erf<S: Simd>(simd: S, a: F64s<S>) -> F64s<S> {
    let x = a.abs();
    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let x16 = x8 * x8;
    let o25 = x.simd_le(splat(simd, 2.5));

    let mut t2;
    if o25.all_true() {
        // Abramowitz and Stegun
        let t = poly21(
            simd,
            x,
            x2,
            x4,
            x8,
            x16,
            -0.208_327_100_252_522_209_7_e-14,
            0.715_190_997_079_089_700_9_e-13,
            -0.116_223_822_011_099_936_4_e-11,
            0.118_647_423_082_158_525_9_e-10,
            -0.849_997_317_835_461_344_0_e-10,
            0.450_764_746_259_884_162_9_e-9,
            -0.180_804_447_428_884_891_5_e-8,
            0.543_508_182_671_621_238_9_e-8,
            -0.114_393_989_575_862_848_4_e-7,
            0.121_544_236_268_088_924_3_e-7,
            0.166_987_875_618_125_035_5_e-7,
            -0.980_807_460_225_519_428_8_e-7,
            0.138_900_055_786_583_720_4_e-6,
            0.294_551_452_998_733_186_6_e-6,
            -0.184_291_827_300_399_828_3_e-5,
            0.341_798_783_611_536_213_6_e-5,
            0.386_023_635_649_312_910_1_e-5,
            -0.330_940_307_274_994_754_6_e-4,
            0.106_086_292_259_757_953_2_e-3,
            0.232_325_315_521_307_617_4_e-3,
            0.149_014_971_914_554_472_9_e-3,
        );
        t2 = poly4dd(
            simd,
            x,
            t,
            DD::new(
                splat(simd, 0.009_287_795_839_227_560_440_5),
                splat(simd, 7.928_755_946_396_110_749_3_e-19),
            ),
            DD::new(
                splat(simd, 0.042_275_531_758_784_692_937),
                splat(simd, 1.378_522_662_050_101_613_8_e-19),
            ),
            DD::new(
                splat(simd, 0.070_523_697_943_469_534_91),
                splat(simd, 9.584_662_807_079_209_284_2_e-19),
            ),
        );
        t2 = v_add_checked(splat(simd, 1.0), mul_v(t2, x));
        t2 = square(t2);
        t2 = square(t2);
        t2 = square(t2);
        t2 = square(t2);
        t2 = recip(t2);
    } else {
        let t = poly21v(
            simd,
            x,
            x2,
            x4,
            x8,
            x16,
            select_splat(
                simd,
                o25,
                -0.208_327_100_252_522_209_7_e-14,
                -0.402_401_513_075_262_193_2_e-18,
            ),
            select_splat(
                simd,
                o25,
                0.715_190_997_079_089_700_9_e-13,
                0.384_719_333_281_704_817_2_e-16,
            ),
            select_splat(
                simd,
                o25,
                -0.116_223_822_011_099_936_4_e-11,
                -0.174_931_624_145_564_408_8_e-14,
            ),
            select_splat(
                simd,
                o25,
                0.118_647_423_082_158_525_9_e-10,
                0.502_961_832_287_287_271_5_e-13,
            ),
            select_splat(
                simd,
                o25,
                -0.849_997_317_835_461_344_0_e-10,
                -0.102_522_146_685_146_316_4_e-11,
            ),
            select_splat(
                simd,
                o25,
                0.450_764_746_259_884_162_9_e-9,
                0.157_369_555_933_194_558_3_e-10,
            ),
            select_splat(
                simd,
                o25,
                -0.180_804_447_428_884_891_5_e-8,
                -0.188_465_855_804_020_370_9_e-9,
            ),
            select_splat(
                simd,
                o25,
                0.543_508_182_671_621_238_9_e-8,
                0.179_816_785_303_215_930_9_e-8,
            ),
            select_splat(
                simd,
                o25,
                -0.114_393_989_575_862_848_4_e-7,
                -0.138_074_534_235_503_314_2_e-7,
            ),
            select_splat(
                simd,
                o25,
                0.121_544_236_268_088_924_3_e-7,
                0.852_570_572_646_910_349_9_e-7,
            ),
            select_splat(
                simd,
                o25,
                0.166_987_875_618_125_035_5_e-7,
                -0.416_044_805_810_130_340_5_e-6,
            ),
            select_splat(
                simd,
                o25,
                -0.980_807_460_225_519_428_8_e-7,
                0.151_727_266_000_858_848_5_e-5,
            ),
            select_splat(
                simd,
                o25,
                0.138_900_055_786_583_720_4_e-6,
                -0.334_163_412_731_720_169_7_e-5,
            ),
            select_splat(
                simd,
                o25,
                0.294_551_452_998_733_186_6_e-6,
                -0.251_502_339_587_972_451_3_e-5,
            ),
            select_splat(
                simd,
                o25,
                -0.184_291_827_300_399_828_3_e-5,
                0.653_973_126_966_490_755_4_e-4,
            ),
            select_splat(
                simd,
                o25,
                0.341_798_783_611_536_213_6_e-5,
                -0.355_106_509_742_838_865_8_e-3,
            ),
            select_splat(
                simd,
                o25,
                0.386_023_635_649_312_910_1_e-5,
                0.121_073_609_795_836_886_4_e-2,
            ),
            select_splat(
                simd,
                o25,
                -0.330_940_307_274_994_754_6_e-4,
                -0.260_556_691_257_999_868_e-2,
            ),
            select_splat(
                simd,
                o25,
                0.106_086_292_259_757_953_2_e-3,
                0.125_282_320_243_609_319_3_e-2,
            ),
            select_splat(
                simd,
                o25,
                0.232_325_315_521_307_617_4_e-3,
                0.182_019_139_526_331_322_2_e-1,
            ),
            select_splat(
                simd,
                o25,
                0.149_014_971_914_554_472_9_e-3,
                -0.102_155_715_545_346_595_4,
            ),
        );
        t2 = poly4dd(
            simd,
            x,
            t,
            select_doubled(
                simd,
                o25,
                DD::new(
                    splat(simd, 0.009_287_795_839_227_560_440_5),
                    splat(simd, 7.928_755_946_396_110_749_3_e-19),
                ),
                DD::new(
                    splat(simd, -0.636_910_443_836_417_483_61),
                    splat(simd, -2.424_947_752_653_943_183_9_e-17),
                ),
            ),
            select_doubled(
                simd,
                o25,
                DD::new(
                    splat(simd, 0.042_275_531_758_784_692_937),
                    splat(simd, 1.378_522_662_050_101_613_8_e-19),
                ),
                DD::new(
                    splat(simd, -1.128_292_606_180_396_173_7),
                    splat(simd, -6.297_033_886_041_099_650_5_e-17),
                ),
            ),
            select_doubled(
                simd,
                o25,
                DD::new(
                    splat(simd, 0.070_523_697_943_469_534_91),
                    splat(simd, 9.584_662_807_079_209_284_2_e-19),
                ),
                DD::new(
                    splat(simd, -1.226_131_378_518_480_496_7_e-05),
                    splat(simd, -5.532_970_751_449_010_704_4_e-22),
                ),
            ),
        );
        let mut s2 = v_add_checked(splat(simd, 1.0), mul_v(t2, x));
        s2 = square(s2);
        s2 = square(s2);
        s2 = square(s2);
        s2 = square(s2);
        s2 = recip(s2);
        t2 = select_doubled(simd, o25, s2, DD::new(expk(simd, t2), splat(simd, 0.)));
    }

    t2 = add_v(t2, splat(simd, -1.));

    let mut z = -(t2.0 + t2.1);
    z = x
        .simd_lt(splat(simd, 1e-8))
        .select(x * splat(simd, 1.128_379_167_095_512_627_562_454_759_59), z);
    z = x.simd_ge(splat(simd, 6.)).select(splat(simd, 1.), z);
    z = a.is_infinite().select(splat(simd, 1.), z);
    z = a.simd_eq(splat(simd, 0.)).select(splat(simd, 0.), z);
    mul_sign(simd, z, a)
}
