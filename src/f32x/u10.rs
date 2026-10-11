//! `f32` kernels with a 1.0 ULP error bound (SLEEF's `u10` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f32x/u10.rs`, plus `expk2f` and `sinpifk` from
//! `src/f32x.rs`). Upstream's `f32` helpers keep the element suffix in their names
//! (`logkf`, `logk2f`, `expkf`, `expk2f`, `gammafk`, `atan2kf_u1`); those names are kept verbatim
//! here, as in the `u05`/`u35` tiers.
//!
//! Upstream's `cfg(target_feature = "fma")` branches are taken, and its `full_fp_rounding`
//! branches are not, per the crate-level deviations. Its AVX-512-only and NEON-only code paths are
//! not ported.

use core::f32::consts::{FRAC_1_PI, FRAC_2_PI, FRAC_PI_2, FRAC_PI_4, PI};

use super::*;
use crate::doubled::f32::{
    add_as_doubled, add_checked, add_checked_as_doubled, add_checked_v, add_v, mul_as_doubled,
    mul_as_f, mul_assign_v, mul_v, normalize, recip, scale, sqrt, square, square_as_f, sub_checked,
    v_add_checked, DD,
};

/// Sine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn sin<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut q;
    let mut s;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        let u = rint(simd, d * splat(simd, FRAC_1_PI));
        q = roundi(simd, u);
        let v = mla(simd, u, splat(simd, -PI_A2_F), d);
        s = add_as_doubled(v, u * splat(simd, -PI_B2_F));
        s = add_checked_v(s, u * splat(simd, -PI_C2_F));
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
        s = normalize(dfidf);

        s.0 = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, s.0),
        );
    }

    let t = s;
    let s = square(s);

    let mut u = mla(
        simd,
        splat(simd, 2.608_315_980_978_659_354_150_3_e-6),
        s.0,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );

    let x = v_add_checked(
        splat(simd, 1.),
        add_checked_as_doubled(splat(simd, -0.166_666_597_127_914_428_710_938), u * s.0) * s,
    );

    u = mul_as_f(t, x);

    u = from_bits(
        simd,
        (q & splat_int(simd, 1))
            .simd_eq(splat_int(simd, 1))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, u),
    );

    is_neg_zero(simd, d).select(d, u)
}

/// Sine function.
///
/// The error bound of the returned value is `1.0 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn sin_deterministic<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let u = rint(simd, d * splat(simd, FRAC_1_PI));
    let mut q = roundi(simd, u);
    let v = mla(simd, u, splat(simd, -PI_A2_F), d);
    let mut s = add_as_doubled(v, u * splat(simd, -PI_B2_F));
    s = add_checked_v(s, u * splat(simd, -PI_C2_F));
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

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
        let mut t = normalize(dfidf);

        t.0 = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, t.0),
        );

        q = g.select(q, q2);
        s = select_doubled(simd, g, s, t);
    }

    let t = s;
    let s = square(s);

    let mut u = mla(
        simd,
        splat(simd, 2.608_315_980_978_659_354_150_3_e-6),
        s.0,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );

    let x = v_add_checked(
        splat(simd, 1.),
        add_checked_as_doubled(splat(simd, -0.166_666_597_127_914_428_710_938), u * s.0) * s,
    );

    u = mul_as_f(t, x);

    u = from_bits(
        simd,
        (q & splat_int(simd, 1))
            .simd_eq(splat_int(simd, 1))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, u),
    );

    is_neg_zero(simd, d).select(d, u)
}

/// Cosine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn cos<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut q;
    let mut s;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        let dq = mla(
            simd,
            rint(
                simd,
                mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5)),
            ),
            splat(simd, 2.),
            splat(simd, 1.),
        );
        q = roundi(simd, dq);
        s = add_as_doubled(d, dq * splat(simd, -PI_A2_F) * splat(simd, 0.5));
        s = add_v(s, dq * splat(simd, -PI_B2_F) * splat(simd, 0.5));
        s = add_v(s, dq * splat(simd, -PI_C2_F) * splat(simd, 0.5));
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
        s = normalize(dfidf);

        s.0 = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, s.0),
        );
    }

    let t = s;
    let s = square(s);

    let u = mla(
        simd,
        splat(simd, 2.608_315_980_978_659_354_150_3_e-6),
        s.0,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    let u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );

    let x = v_add_checked(
        splat(simd, 1.),
        add_checked_as_doubled(splat(simd, -0.166_666_597_127_914_428_710_938), u * s.0) * s,
    );

    let u = mul_as_f(t, x);

    from_bits(
        simd,
        (q & splat_int(simd, 2))
            .simd_eq(splat_int(simd, 0))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
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
pub fn cos_deterministic<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let dq = mla(
        simd,
        rint(
            simd,
            mla(simd, d, splat(simd, FRAC_1_PI), splat(simd, -0.5)),
        ),
        splat(simd, 2.),
        splat(simd, 1.),
    );
    let mut q = roundi(simd, dq);
    let mut s = add_as_doubled(d, dq * splat(simd, -PI_A2_F * 0.5));
    s = add_v(s, dq * splat(simd, -PI_B2_F * 0.5));
    s = add_v(s, dq * splat(simd, -PI_C2_F * 0.5));
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

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
        let mut t = normalize(dfidf);

        t.0 = from_bits(
            simd,
            (d.is_infinite() | d.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, t.0),
        );

        q = g.select(q, q2);
        s = select_doubled(simd, g, s, t);
    }

    let t = s;
    let s = square(s);

    let u = mla(
        simd,
        splat(simd, 2.608_315_980_978_659_354_150_3_e-6),
        s.0,
        splat(simd, -0.000_198_106_907_191_686_332_225_8),
    );
    let u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.008_333_078_585_565_090_179_443_36),
    );

    let x = v_add_checked(
        splat(simd, 1.),
        add_checked_as_doubled(splat(simd, -0.166_666_597_127_914_428_710_938), u * s.0) * s,
    );

    let u = mul_as_f(t, x);

    from_bits(
        simd,
        (q & splat_int(simd, 2))
            .simd_eq(splat_int(simd, 0))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, u),
    )
}

/// Evaluates sine and cosine simultaneously.
///
/// Returns `(sin, cos)`. The error bound of the returned values is `1.0 ULP`. If the argument is a
/// `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sincos<S: Simd>(simd: S, d: F32s<S>) -> (F32s<S>, F32s<S>) {
    let q;
    let mut s;

    if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        let u = rint(simd, d * splat(simd, FRAC_2_PI));
        q = roundi(simd, u);
        let v = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
        s = add_as_doubled(v, u * splat(simd, -PI_B2_F * 0.5));
        s = add_checked_v(s, u * splat(simd, -PI_C2_F * 0.5));
    } else {
        let (dfidf, dfii) = rempi(simd, d);
        q = dfii;
        s = dfidf;
        let o = d.is_infinite() | d.is_nan();
        s.0 = from_bits(
            simd,
            o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) | to_bits(simd, s.0),
        );
    }

    let t = s;

    s.0 = square_as_f(s);

    let u = mla(
        simd,
        mla(
            simd,
            splat(simd, -0.000_195_169_282_960_705_459_117_889),
            s.0,
            splat(simd, 0.008_332_157_507_538_795_471_191_41),
        ),
        s.0,
        splat(simd, -0.166_666_537_523_269_653_320_312),
    ) * (s.0 * t.0);

    let x = add_checked_v(t, u);
    let rx = x.0 + x.1;

    let rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    splat(simd, -2.718_118_423_672_422_068_193_55_e-7),
                    s.0,
                    splat(simd, 2.479_904_469_510_074_704_885_48_e-5),
                ),
                s.0,
                splat(simd, -0.001_388_887_874_782_085_418_701_17),
            ),
            s.0,
            splat(simd, 0.041_666_664_183_139_801_025_390_6),
        ),
        s.0,
        splat(simd, -0.5),
    );

    let x = v_add_checked(splat(simd, 1.), mul_as_doubled::<S>(s.0, u));
    let ry = x.0 + x.1;

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rsin = from_bits(
        simd,
        (o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits()))
            ^ to_bits(simd, rsin),
    );

    let o = ((q + splat_int(simd, 1)) & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rcos = from_bits(
        simd,
        (o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits()))
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
pub fn sincos_deterministic<S: Simd>(simd: S, d: F32s<S>) -> (F32s<S>, F32s<S>) {
    let u = rint(simd, d * splat(simd, FRAC_2_PI));
    let mut q = roundi(simd, u);
    let v = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
    let mut s = add_as_doubled(v, u * splat(simd, -PI_B2_F * 0.5));
    s = add_checked_v(s, u * splat(simd, -PI_C2_F * 0.5));
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

    if !g.all_true() {
        let (dfidf, dfii) = rempi(simd, d);
        let mut t = dfidf;
        let o = d.is_infinite() | d.is_nan();
        t.0 = from_bits(
            simd,
            o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) | to_bits(simd, t.0),
        );
        q = g.select(q, dfii);
        s = select_doubled(simd, g, s, t);
    }

    let t = s;

    s.0 = square_as_f(s);

    let u = mla(
        simd,
        mla(
            simd,
            splat(simd, -0.000_195_169_282_960_705_459_117_889),
            s.0,
            splat(simd, 0.008_332_157_507_538_795_471_191_41),
        ),
        s.0,
        splat(simd, -0.166_666_537_523_269_653_320_312),
    ) * (s.0 * t.0);

    let x = add_checked_v(t, u);
    let mut rx = x.0 + x.1;

    rx = is_neg_zero(simd, d).select(splat(simd, -0.), rx);

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    splat(simd, -2.718_118_423_672_422_068_193_55_e-7),
                    s.0,
                    splat(simd, 2.479_904_469_510_074_704_885_48_e-5),
                ),
                s.0,
                splat(simd, -0.001_388_887_874_782_085_418_701_17),
            ),
            s.0,
            splat(simd, 0.041_666_664_183_139_801_025_390_6),
        ),
        s.0,
        splat(simd, -0.5),
    );

    let x = v_add_checked(splat(simd, 1.), mul_as_doubled::<S>(s.0, u));
    let ry = x.0 + x.1;

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 0));
    let mut rsin = o.select(rx, ry);
    let mut rcos = o.select(ry, rx);

    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rsin = from_bits(
        simd,
        (o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits()))
            ^ to_bits(simd, rsin),
    );

    let o = ((q + splat_int(simd, 1)) & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));
    rcos = from_bits(
        simd,
        (o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits()))
            ^ to_bits(simd, rcos),
    );

    (rsin, rcos)
}

/// Tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn tan<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let q;

    let mut s = if d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F)).all_true() {
        let u = rint(simd, d * splat(simd, FRAC_2_PI));
        q = roundi(simd, u);
        let v = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
        add_checked_v(
            add_as_doubled(v, u * splat(simd, -PI_B2_F * 0.5)),
            u * splat(simd, -PI_C2_F * 0.5),
        )
    } else {
        let (dfidf, dfii) = rempi(simd, d);
        q = dfii;
        let o = d.is_infinite() | d.is_nan();
        let m = o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0));
        DD::new(
            from_bits(simd, m | to_bits(simd, dfidf.0)),
            from_bits(simd, m | to_bits(simd, dfidf.1)),
        )
    };

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
    let n = o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
        & splat_bits(simd, (-0.0f32).to_bits());
    s = DD::new(
        from_bits(simd, to_bits(simd, s.0) ^ n),
        from_bits(simd, to_bits(simd, s.1) ^ n),
    );

    let t = s;
    s = square(s);
    s = normalize(s);

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    splat(simd, 0.004_466_364_625_841_379_165_649_41),
                    s.0,
                    splat(simd, -8.392_018_207_814_544_439_315_8_e-5),
                ),
                s.0,
                splat(simd, 0.010_963_924_229_145_050_048_828_1),
            ),
            s.0,
            splat(simd, 0.021_236_030_384_898_185_729_980_5),
        ),
        s.0,
        splat(simd, 0.054_068_714_380_264_282_226_562_5),
    );

    let mut x = add_checked_as_doubled(splat(simd, 0.133_325_666_189_193_725_585_938), u * s.0);
    x = v_add_checked(
        splat(simd, 1.),
        v_add_checked(splat(simd, 0.333_333_611_488_342_285_156_25), s * x) * s,
    );
    x = t * x;

    x = select_doubled(simd, o, recip(x), x);

    let u = x.0 + x.1;

    is_neg_zero(simd, d).select(d, u)
}

/// Tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
///
/// This is the deterministic variant: its results are bitwise consistent across all platforms, at
/// the cost of per-lane independence.
#[inline(always)]
pub fn tan_deterministic<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let u = rint(simd, d * splat(simd, FRAC_2_PI));
    let mut q = roundi(simd, u);
    let v = mla(simd, u, splat(simd, -PI_A2_F * 0.5), d);
    let mut s = add_as_doubled(v, u * splat(simd, -PI_B2_F * 0.5));
    s = add_checked_v(s, u * splat(simd, -PI_C2_F * 0.5));
    let g = d.abs().simd_lt(splat(simd, TRIGRANGEMAX2_F));

    if !g.all_true() {
        let (dfidf, dfii) = rempi(simd, d);
        let mut t = dfidf;
        let o = d.is_infinite() | d.is_nan();
        let m = o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0));
        t = DD::new(
            from_bits(simd, m | to_bits(simd, t.0)),
            from_bits(simd, m | to_bits(simd, t.1)),
        );
        q = g.select(q, dfii);
        s = select_doubled(simd, g, s, t);
    }

    let o = (q & splat_int(simd, 1)).simd_eq(splat_int(simd, 1));
    let n = o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
        & splat_bits(simd, (-0.0f32).to_bits());
    s = DD::new(
        from_bits(simd, to_bits(simd, s.0) ^ n),
        from_bits(simd, to_bits(simd, s.1) ^ n),
    );

    let t = s;
    s = square(s);
    s = normalize(s);

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    splat(simd, 0.004_466_364_625_841_379_165_649_41),
                    s.0,
                    splat(simd, -8.392_018_207_814_544_439_315_8_e-5),
                ),
                s.0,
                splat(simd, 0.010_963_924_229_145_050_048_828_1),
            ),
            s.0,
            splat(simd, 0.021_236_030_384_898_185_729_980_5),
        ),
        s.0,
        splat(simd, 0.054_068_714_380_264_282_226_562_5),
    );

    let mut x = add_checked_as_doubled(splat(simd, 0.133_325_666_189_193_725_585_938), u * s.0);
    x = v_add_checked(
        splat(simd, 1.),
        v_add_checked(splat(simd, 0.333_333_611_488_342_285_156_25), s * x) * s,
    );
    x = t * x;

    x = select_doubled(simd, o, recip(x), x);

    let u = x.0 + x.1;

    is_neg_zero(simd, d).select(d, u)
}

/// Arc tangent of `y / x` with 1.0 ULP accuracy, kept as a double-double.
#[inline(always)]
fn atan2kf_u1<S: Simd>(simd: S, y: DD<S>, mut x: DD<S>) -> DD<S> {
    let q =
        x.0.simd_lt(splat(simd, 0.))
            .select(splat_int(simd, -2), splat_int(simd, 0));
    let p = x.0.simd_lt(splat(simd, 0.));
    let r = p.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
        & splat_bits(simd, (-0.0f32).to_bits());
    x = DD::new(
        from_bits(simd, to_bits(simd, x.0) ^ r),
        from_bits(simd, to_bits(simd, x.1) ^ r),
    );

    let q = x.0.simd_lt(y.0).select(q + splat_int(simd, 1), q);
    let p = x.0.simd_lt(y.0);
    let s = select_doubled(simd, p, -x, y);
    let mut t = select_doubled(simd, p, y, x);

    let s = s / t;
    t = square(s);
    t = normalize(t);

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    mla(
                        simd,
                        mla(
                            simd,
                            mla(
                                simd,
                                splat(simd, -0.001_763_979_089_446_365_833_282_47),
                                t.0,
                                splat(simd, 0.010_790_090_076_625_347_137_451_2),
                            ),
                            t.0,
                            splat(simd, -0.030_956_460_162_997_245_788_574_2),
                        ),
                        t.0,
                        splat(simd, 0.057_736_508_548_259_735_107_421_9),
                    ),
                    t.0,
                    splat(simd, -0.083_895_072_340_965_270_996_093_8),
                ),
                t.0,
                splat(simd, 0.109_463_557_600_975_036_621_094),
            ),
            t.0,
            splat(simd, -0.142_626_821_994_781_494_140_625),
        ),
        t.0,
        splat(simd, 0.199_983_194_470_405_578_613_281),
    );

    t *= add_checked_as_doubled(splat(simd, -0.333_332_866_430_282_592_773_438), u * t.0);
    t = s * v_add_checked(splat(simd, 1.), t);
    add_checked(
        mul_v(
            DD::new(
                splat(simd, 1.570_796_370_506_286_621_1),
                splat(simd, -4.371_138_828_673_792_886_5_e-8),
            ),
            q.to_float::<F32s<S>>(),
        ),
        t,
    )
}

/// Arc tangent function of two variables.
///
/// Evaluates the arc tangent of `y / x`; the quadrant of the result is determined from the signs
/// of `x` and `y`. The error bound of the returned value is `max(1.0 ULP, f32::MIN_POSITIVE)`.
#[inline(always)]
pub fn atan2<S: Simd>(simd: S, mut y: F32s<S>, mut x: F32s<S>) -> F32s<S> {
    let o = x
        .abs()
        .simd_lt(splat(simd, 2.938_737_278_354_183_094_7_e-39)); // nexttowardf((1.0 / FLT_MAX), 1)
    x = o.select(x * splat(simd, F1_24), x);
    y = o.select(y * splat(simd, F1_24), y);

    let d = atan2kf_u1(
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
        (x.is_nan() | y.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, mul_sign(simd, r, y)),
    )
}

/// Arc sine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn asin<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.) - d.abs()) * splat(simd, 0.5));
    let mut x = select_doubled(
        simd,
        o,
        DD::new(d.abs(), splat(simd, 0.)),
        sqrt_as_doubled(simd, x2),
    );
    x = select_doubled(
        simd,
        d.abs().simd_eq(splat(simd, 1.)),
        DD::new(splat(simd, 0.), splat(simd, 0.)),
        x,
    );

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    splat(simd, 0.419_745_482_5_e-1),
                    x2,
                    splat(simd, 0.242_404_602_5_e-1),
                ),
                x2,
                splat(simd, 0.454_742_386_9_e-1),
            ),
            x2,
            splat(simd, 0.749_502_927_1_e-1),
        ),
        x2,
        splat(simd, 0.166_667_729_6),
    ) * (x2 * x.0);

    let y = sub_checked(
        sub_checked(
            DD::new(splat(simd, D_PI_F.0 / 4.), splat(simd, D_PI_F.1 / 4.)),
            x,
        ),
        DD::new(u, splat(simd, 0.)),
    );

    let r = o.select(u + x.0, (y.0 + y.1) * splat(simd, 2.));
    mul_sign(simd, r, d)
}

/// Arc cosine function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn acos<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let o = d.abs().simd_lt(splat(simd, 0.5));
    let x2 = o.select(d * d, (splat(simd, 1.) - d.abs()) * splat(simd, 0.5));

    let mut x = select_doubled(
        simd,
        o,
        DD::new(d.abs(), splat(simd, 0.)),
        sqrt_as_doubled(simd, x2),
    );
    x = select_doubled(
        simd,
        d.abs().simd_eq(splat(simd, 1.)),
        DD::new(splat(simd, 0.), splat(simd, 0.)),
        x,
    );

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                mla(
                    simd,
                    splat(simd, 0.419_745_482_5_e-1),
                    x2,
                    splat(simd, 0.242_404_602_5_e-1),
                ),
                x2,
                splat(simd, 0.454_742_386_9_e-1),
            ),
            x2,
            splat(simd, 0.749_502_927_1_e-1),
        ),
        x2,
        splat(simd, 0.166_667_729_6),
    ) * (x2 * x.0);

    let mut y = sub_checked(
        DD::new(splat(simd, D_PI_F.0 / 2.), splat(simd, D_PI_F.1 / 2.)),
        add_checked_as_doubled(mul_sign(simd, x.0, d), mul_sign(simd, u, d)),
    );
    x = add_checked_v(x, u);

    y = select_doubled(simd, o, y, scale(x, splat(simd, 2.)));

    y = select_doubled(
        simd,
        !o & d.simd_lt(splat(simd, 0.)),
        sub_checked(DD::splat(simd, D_PI_F), y),
        y,
    );

    y.0 + y.1
}

/// Arc tangent function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn atan<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let d2 = atan2kf_u1(
        simd,
        DD::new(d.abs(), splat(simd, 0.)),
        DD::new(splat(simd, 1.), splat(simd, 0.)),
    );
    let mut r = d2.0 + d2.1;
    r = d
        .is_infinite()
        .select(splat(simd, 1.570_796_326_794_896_557_998_982), r);
    mul_sign(simd, r, d)
}

/// Hyperbolic sine function.
///
/// The error bound of the returned value is `1.0 ULP` for `x` in `[-88.5, 88.5]`. If `x` is a
/// finite value out of that range, infinity with a correct sign, or a correct value within
/// `1.0 ULP`, is returned.
#[inline(always)]
pub fn sinh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let mut y = x.abs();
    let d = expk2f(simd, DD::new(y, splat(simd, 0.)));
    let d = sub_checked(d, recip(d));
    y = (d.0 + d.1) * splat(simd, 0.5);

    y = (x.abs().simd_gt(splat(simd, 89.)) | y.is_nan()).select(splat(simd, f32::INFINITY), y);
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
/// The error bound of the returned value is `1.0 ULP` for `x` in `[-88.5, 88.5]`. If `x` is a
/// finite value out of that range, infinity with a correct sign, or a correct value within
/// `1.0 ULP`, is returned.
#[inline(always)]
pub fn cosh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let mut y = x.abs();
    let d = expk2f(simd, DD::new(y, splat(simd, 0.)));
    let d = add_checked(d, recip(d));
    y = (d.0 + d.1) * splat(simd, 0.5);

    y = (x.abs().simd_gt(splat(simd, 89.)) | y.is_nan()).select(splat(simd, f32::INFINITY), y);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Hyperbolic tangent function.
///
/// The error bound of the returned value is `1.0001 ULP`.
#[inline(always)]
pub fn tanh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let mut y = x.abs();
    let d = expk2f(simd, DD::new(y, splat(simd, 0.)));
    let e = recip(d);
    let d = add_checked(d, -e) / add_checked(d, e);
    y = d.0 + d.1;

    y = (x.abs().simd_gt(splat(simd, 8.664_339_742)) | y.is_nan()).select(splat(simd, 1.), y);
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Natural logarithm, kept as a double-double.
#[inline(always)]
fn logk2f<S: Simd>(simd: S, d: DD<S>) -> DD<S> {
    let e = ilogbk(simd, d.0 * splat(simd, 1. / 0.75));
    let m = scale(d, pow2i(simd, -e));

    let x = add_v(m, splat(simd, -1.)) / add_v(m, splat(simd, 1.));
    let x2 = square(x);

    let t = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                splat(simd, 0.239_282_846_450_805_664_062_5),
                x2.0,
                splat(simd, 0.285_182_118_415_832_519_531_25),
            ),
            x2.0,
            splat(simd, 0.400_005_877_017_974_853_515_625),
        ),
        x2.0,
        splat(simd, 0.666_666_686_534_881_591_796_875),
    );

    let mut s = mul_v(DD::splat(simd, D_LN2_F), e.to_float::<F32s<S>>());
    s = add_checked(s, scale(x, splat(simd, 2.)));
    add_checked(s, mul_v(x2 * x, t))
}

/// Inverse hyperbolic sine function.
///
/// The error bound of the returned value is `1.001 ULP` for `x` in `[-1.84e+19, 1.84e+19]`.
/// Outside that range infinity with the correct sign, or a correct value within `1.0 ULP`, is
/// returned.
#[inline(always)]
pub fn asinh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let mut y = x.abs();
    let o = y.simd_gt(splat(simd, 1.));

    let mut d = select_doubled(simd, o, recip_as_doubled(x), DD::new(y, splat(simd, 0.)));
    d = sqrt(add_v(square(d), splat(simd, 1.)));
    d = select_doubled(simd, o, mul_v(d, y), d);

    d = logk2f(simd, normalize(add_v(d, x)));
    y = d.0 + d.1;

    y = (x.abs().simd_gt(splat(simd, SQRT_FLT_MAX)) | y.is_nan())
        .select(mul_sign(simd, splat(simd, f32::INFINITY), x), y);
    y = from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    );
    is_neg_zero(simd, x).select(splat(simd, -0.), y)
}

/// Inverse hyperbolic cosine function.
///
/// The error bound of the returned value is `1.001 ULP` for `x` in `[-1.84e+19, 1.84e+19]`.
/// Outside that range infinity with the correct sign, or a correct value within `1.0 ULP`, is
/// returned.
#[inline(always)]
pub fn acosh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let d = logk2f(
        simd,
        add_v(
            sqrt(add_as_doubled(x, splat(simd, 1.))) * sqrt(add_as_doubled(x, splat(simd, -1.))),
            x,
        ),
    );
    let mut y = d.0 + d.1;

    y = (x.abs().simd_gt(splat(simd, SQRT_FLT_MAX)) | y.is_nan())
        .select(splat(simd, f32::INFINITY), y);

    y = from_bits(
        simd,
        !x.simd_eq(splat(simd, 1.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, y),
    );

    y = from_bits(
        simd,
        x.simd_lt(splat(simd, 1.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    );
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Inverse hyperbolic tangent function.
///
/// The error bound of the returned value is `1.0001 ULP`.
#[inline(always)]
pub fn atanh<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let mut y = x.abs();
    let d = logk2f(
        simd,
        add_as_doubled(splat(simd, 1.), y) / add_as_doubled(splat(simd, 1.), -y),
    );
    y = from_bits(
        simd,
        y.simd_gt(splat(simd, 1.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(
                simd,
                y.simd_eq(splat(simd, 1.))
                    .select(splat(simd, f32::INFINITY), (d.0 + d.1) * splat(simd, 0.5)),
            ),
    );

    y = from_bits(
        simd,
        (x.is_infinite() | y.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    );
    y = mul_sign(simd, y, x);
    from_bits(
        simd,
        x.is_nan()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, y),
    )
}

/// Natural logarithmic function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let m;

    let mut s = {
        let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
        d = o.select(d * splat(simd, F1_32 * F1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        mul_v(DD::splat(simd, D_LN2_F), e.to_float::<F32s<S>>())
    };

    let x = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.), m);
    let x2 = x.0 * x.0;

    let t = mla(
        simd,
        mla(
            simd,
            splat(simd, 0.302_729_487_4),
            x2,
            splat(simd, 0.399_610_817_4),
        ),
        x2,
        splat(simd, 0.666_669_488),
    );

    s = add_checked(s, scale(x, splat(simd, 2.)));
    s = add_checked_v(s, x2 * x.0 * t);

    let r = s.0 + s.1;

    let r = d
        .simd_eq(splat(simd, f32::INFINITY))
        .select(splat(simd, f32::INFINITY), r);
    let r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f32::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NEG_INFINITY), r)
}

/// Base-10 logarithmic function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log10<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let m;

    let mut s = {
        let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
        d = o.select(d * splat(simd, F1_32 * F1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        mul_v::<S>(
            DD::new(splat(simd, 0.301_030_01), splat(simd, -1.432_098_889_e-8)),
            e.to_float::<F32s<S>>(),
        )
    };

    let x = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.), m);
    let x2 = x.0 * x.0;

    let t = mla(
        simd,
        mla(
            simd,
            splat(simd, 0.131_428_986_8),
            x2,
            splat(simd, 0.173_549_354_1),
        ),
        x2,
        splat(simd, 0.289_530_962_7),
    );

    s = add_checked(
        s,
        x * DD::new(splat(simd, 0.868_588_984), splat(simd, -2.170_757_285_e-8)),
    );
    s = add_checked_v(s, x2 * x.0 * t);

    let mut r = s.0 + s.1;

    r = d
        .simd_eq(splat(simd, f32::INFINITY))
        .select(splat(simd, f32::INFINITY), r);
    r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f32::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NEG_INFINITY), r)
}

/// Base-2 logarithmic function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log2<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let m;

    let ef = {
        let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
        d = o.select(d * splat(simd, F1_32 * F1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        e.to_float::<F32s<S>>()
    };

    let x = add_as_doubled::<S>(splat(simd, -1.), m) / add_as_doubled::<S>(splat(simd, 1.), m);
    let x2 = x.0 * x.0;

    let t = mla(
        simd,
        mla(
            simd,
            splat(simd, 0.437_455_028_3),
            x2,
            splat(simd, 0.576_479_017_7),
        ),
        x2,
        splat(simd, 0.961_801_290_512),
    );
    let mut s = v_add_checked(
        ef,
        x * DD::new(
            splat(simd, 2.885_390_043_258_666_992_2),
            splat(simd, 3.273_447_448_356_848_861_6_e-8),
        ),
    );
    s = add_v(s, x2 * x.0 * t);

    let mut r = s.0 + s.1;

    r = d
        .simd_eq(splat(simd, f32::INFINITY))
        .select(splat(simd, f32::INFINITY), r);
    r = (d.simd_lt(splat(simd, 0.)) | d.is_nan()).select(splat(simd, f32::NAN), r);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NEG_INFINITY), r)
}

/// Logarithm of one plus argument.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn log1p<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let m;

    let dp1 = d + splat(simd, 1.);

    let mut s = {
        let o = dp1.simd_lt(splat(simd, f32::MIN_POSITIVE));
        let dp1 = o.select(dp1 * splat(simd, F1_32 * F1_32), dp1);
        let e = ilogb2k(simd, dp1 * splat(simd, 1. / 0.75));
        let t = ldexp3k(simd, splat(simd, 1.), -e);
        m = mla(simd, d, t, t - splat(simd, 1.));
        let e = o.select(e - splat_int(simd, 64), e);
        mul_v(DD::splat(simd, D_LN2_F), e.to_float::<F32s<S>>())
    };

    let x = DD::new(m, splat(simd, 0.)) / add_checked_as_doubled(splat(simd, 2.), m);
    let x2 = x.0 * x.0;

    let t = mla(
        simd,
        mla(
            simd,
            splat(simd, 0.302_729_487_4),
            x2,
            splat(simd, 0.399_610_817_4),
        ),
        x2,
        splat(simd, 0.666_669_488),
    );

    s = add_checked(s, scale(x, splat(simd, 2.)));
    s = add_checked_v(s, x2 * x.0 * t);

    let mut r = s.0 + s.1;

    r = d
        .simd_gt(splat(simd, 1e+38))
        .select(splat(simd, f32::INFINITY), r);
    r = from_bits(
        simd,
        splat(simd, -1.)
            .simd_gt(d)
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, r),
    );
    r = d
        .simd_eq(splat(simd, -1.))
        .select(splat(simd, f32::NEG_INFINITY), r);
    is_neg_zero(simd, d).select(splat(simd, -0.), r)
}

/// Base-*e* exponential function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn exp<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let q = roundi(simd, d * splat(simd, R_LN2_F));

    let s = mla(simd, q.to_float::<F32s<S>>(), splat(simd, -L2U_F), d);
    let s = mla(simd, q.to_float::<F32s<S>>(), splat(simd, -L2L_F), s);

    let mut u = mla(
        simd,
        splat(simd, 0.000_198_527_617_612_853_646_278_381),
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

    u = splat(simd, 1.) + mla(simd, s * s, u, s);

    u = ldexp2k(simd, u, q);

    u = from_bits(
        simd,
        !d.simd_lt(splat(simd, -104.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    );
    splat(simd, 100.)
        .simd_lt(d)
        .select(splat(simd, f32::INFINITY), u)
}

/// `exp(x) - 1` without the final subtraction, as a double-double.
///
/// Upstream keeps this helper in `src/f32x.rs` rather than in the `u10` tier; the `u15` tier needs
/// it, so it is carried here.
#[inline(always)]
pub fn expk2f<S: Simd>(simd: S, d: DD<S>) -> DD<S> {
    let u = (d.0 + d.1) * splat(simd, R_LN2_F);
    let q = roundi(simd, u);
    let qf = q.to_float::<F32s<S>>();

    let mut s = add_v(d, qf * splat(simd, -L2U_F));
    s = add_v(s, qf * splat(simd, -L2L_F));

    let u = mla(
        simd,
        mla(
            simd,
            mla(
                simd,
                splat(simd, 0.198_096_022_4_e-3),
                s.0,
                splat(simd, 0.139_425_648_4_e-2),
            ),
            s.0,
            splat(simd, 0.833_345_670_3_e-2),
        ),
        s.0,
        splat(simd, 0.416_663_736_1_e-1),
    );

    let mut t = add_v(
        mul_v(s, u),
        splat(simd, 0.166_666_659_414_234_244_790_680_580_464),
    );
    t = add_v(s * t, splat(simd, 0.5));
    t = s + square(s) * t;

    t = v_add_checked(splat(simd, 1.), t);

    t = DD::new(ldexp2k(simd, t.0, q), ldexp2k(simd, t.1, q));

    let keep =
        !d.0.simd_lt(splat(simd, -104.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0));
    DD::new(
        from_bits(simd, keep & to_bits(simd, t.0)),
        from_bits(simd, keep & to_bits(simd, t.1)),
    )
}

/// Base-10 exponential function.
///
/// The error bound of the returned value is `1.09 ULP`.
#[inline(always)]
pub fn exp10<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut u = rint(simd, d * splat(simd, LOG10_2_F));
    let q = roundi(simd, u);

    let s = mla(simd, u, splat(simd, -L10U_F), d);
    let s = mla(simd, u, splat(simd, -L10L_F), s);

    u = mla(
        simd,
        splat(simd, 0.680_255_591_9_e-1),
        s,
        splat(simd, 0.207_808_032_6),
    );
    u = mla(simd, u, s, splat(simd, 0.539_390_385_2));
    u = mla(simd, u, s, splat(simd, 0.117_124_533_7_e+1));
    u = mla(simd, u, s, splat(simd, 0.203_467_869_8_e+1));
    u = mla(simd, u, s, splat(simd, 0.265_094_900_1_e+1));
    let x = add_checked_v::<S>(
        DD::new(
            splat(simd, 2.302_585_124_969_482_421_9),
            splat(simd, -3.170_517_251_649_359_315_7_e-08),
        ),
        u * s,
    );
    u = normalize(v_add_checked(splat(simd, 1.), mul_v(x, s))).0;

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

/// Base-*e* exponential function minus 1.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn expm1<S: Simd>(simd: S, a: F32s<S>) -> F32s<S> {
    let d = add_v(expk2f(simd, DD::new(a, splat(simd, 0.))), splat(simd, -1.));
    let mut x = d.0 + d.1;
    x = a
        .simd_gt(splat(simd, 88.722_831_726_074_218_75))
        .select(splat(simd, f32::INFINITY), x);
    x = a
        .simd_lt(splat(simd, -16.635_532_333_438_687_426_013_570))
        .select(splat(simd, -1.), x);
    is_neg_zero(simd, a).select(splat(simd, -0.), x)
}

/// Base-2 exponential function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn exp2<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut u = rint(simd, d);
    let q = roundi(simd, u);

    let s = d - u;

    u = mla(
        simd,
        splat(simd, 0.153_592_089_2_e-3),
        s,
        splat(simd, 0.133_926_270_1_e-2),
    );
    u = mla(simd, u, s, splat(simd, 0.961_838_476_4_e-2));
    u = mla(simd, u, s, splat(simd, 0.555_034_726_9_e-1));
    u = mla(simd, u, s, splat(simd, 0.240_226_447_6));
    u = mla(simd, u, s, splat(simd, 0.693_147_182_5));

    // Upstream selects this final step when the build enables FMA, which is the formulation this
    // port always takes.
    u = mla(simd, u, s, splat(simd, 1.));

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

/// Natural logarithm, collapsed to the leading part.
#[inline(always)]
fn logkf<S: Simd>(simd: S, mut d: F32s<S>) -> DD<S> {
    let m;

    let ef = {
        let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
        d = o.select(d * splat(simd, F1_32 * F1_32), d);
        let mut e = ilogb2k(simd, d * splat(simd, 1. / 0.75));
        m = ldexp3k(simd, d, -e);
        e = o.select(e - splat_int(simd, 64), e);
        e.to_float::<F32s<S>>()
    };

    let x = add_as_doubled(splat(simd, -1.), m) / add_as_doubled(splat(simd, 1.), m);
    let x2 = square(x);

    let t = mla(
        simd,
        mla(
            simd,
            splat(simd, 0.240_320_354_700_088_500_976_562),
            x2.0,
            splat(simd, 0.285_112_679_004_669_189_453_125),
        ),
        x2.0,
        splat(simd, 0.400_007_992_982_864_379_882_812),
    );
    let c = DD::new(
        splat(simd, 0.666_666_626_930_236_816_406_25),
        splat(simd, 3.691_838_612_596_143_320_843_11_e-9),
    );

    let mut s = mul_v(DD::splat(simd, D_LN2_F), ef);

    s = add_checked(s, scale(x, splat(simd, 2.)));
    add_checked(s, x2 * x * (mul_v(x2, t) + c))
}

/// Exponential function of a double-double, collapsed to the leading part.
#[inline(always)]
fn expkf<S: Simd>(simd: S, d: DD<S>) -> F32s<S> {
    let u = (d.0 + d.1) * splat(simd, R_LN2_F);
    let q = roundi(simd, u);
    let qf = q.to_float::<F32s<S>>();

    let mut s = add_v(d, qf * splat(simd, -L2U_F));
    s = add_v(s, qf * splat(simd, -L2L_F));

    s = normalize(s);

    let mut u = mla(
        simd,
        splat(simd, 0.001_363_246_468_827_128_410_339_36),
        s.0,
        splat(simd, 0.008_365_969_173_610_210_418_701_17),
    );
    u = mla(
        simd,
        u,
        s.0,
        splat(simd, 0.041_671_082_377_433_776_855_468_8),
    );
    u = mla(simd, u, s.0, splat(simd, 0.166_665_524_244_308_471_679_688));
    u = mla(simd, u, s.0, splat(simd, 0.499_999_850_988_388_061_523_438));

    let mut t = add_checked(s, mul_v(square(s), u));

    t = v_add_checked(splat(simd, 1.), t);
    u = t.0 + t.1;
    u = ldexpk(simd, u, q);

    from_bits(
        simd,
        !d.0.simd_lt(splat(simd, -104.))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, u),
    )
}

/// Power function.
///
/// Returns `x` raised to the power of `y`. The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn pow<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    if true {
        // Upstream writes `y.trunc()`, its saturating round-to-zero conversion; this port uses
        // the `trunc` helper, as [`is_integer`] documents.
        let yisint = trunc(simd, y).simd_eq(y) | y.abs().simd_gt(splat(simd, F1_24));
        let yisodd = (trunci(simd, y) & splat_int(simd, 1)).simd_eq(splat_int(simd, 1))
            & yisint
            & y.abs().simd_lt(splat(simd, F1_24));

        // Upstream clears `yisodd` when `y` is infinite on its NEON-only code paths; those are not
        // ported (see the crate-level deviations), so that correction does not appear here.
        let mut result = expkf(simd, mul_v(logkf(simd, x.abs()), y));

        result = result.is_nan().select(splat(simd, f32::INFINITY), result);

        result *= x.simd_gt(splat(simd, 0.)).select(
            splat(simd, 1.),
            yisint.select(
                yisodd.select(splat(simd, -1.), splat(simd, 1.)),
                splat(simd, f32::NAN),
            ),
        );

        let efx = mul_sign(simd, x.abs() - splat(simd, 1.), y);

        result = y.is_infinite().select(
            from_bits(
                simd,
                !efx.simd_lt(splat(simd, 0.))
                    .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                    & to_bits(
                        simd,
                        efx.simd_eq(splat(simd, 0.))
                            .select(splat(simd, 1.), splat(simd, f32::INFINITY)),
                    ),
            ),
            result,
        );

        result = (x.is_infinite() | x.simd_eq(splat(simd, 0.))).select(
            mul_sign(
                simd,
                (y.is_sign_negative() ^ x.simd_eq(splat(simd, 0.)))
                    .select(splat(simd, 0.), splat(simd, f32::INFINITY)),
                yisodd.select(x, splat(simd, 1.)),
            ),
            result,
        );

        result = from_bits(
            simd,
            (x.is_nan() | y.is_nan()).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                | to_bits(simd, result),
        );

        (y.simd_eq(splat(simd, 0.)) | x.simd_eq(splat(simd, 1.))).select(splat(simd, 1.), result)
    } else {
        expkf(simd, mul_v(logkf(simd, x), y))
    }
}

/// Cube root function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn cbrt<S: Simd>(simd: S, mut d: F32s<S>) -> F32s<S> {
    let mut q2 = DD::new(splat(simd, 1.), splat(simd, 0.));

    let e = ilogbk(simd, d.abs()) + splat_int(simd, 1);
    d = ldexp2k(simd, d, -e);

    let t = e.to_float::<F32s<S>>() + splat(simd, 6144.);
    let qu = trunci(simd, t * splat(simd, 1. / 3.));
    let re = trunci(simd, t - qu.to_float::<F32s<S>>() * splat(simd, 3.));

    q2 = select_doubled(
        simd,
        re.simd_eq(splat_int(simd, 1)),
        DD::new(
            splat(simd, 1.259_921_073_913_574_218_8),
            splat(simd, -2.401_870_169_421_727_041_5_e-8),
        ),
        q2,
    );
    q2 = select_doubled(
        simd,
        re.simd_eq(splat_int(simd, 2)),
        DD::new(
            splat(simd, 1.587_401_032_447_814_941_4),
            splat(simd, 1.952_038_530_816_935_235_6_e-8),
        ),
        q2,
    );

    q2.0 = mul_sign(simd, q2.0, d);
    q2.1 = mul_sign(simd, q2.1, d);
    d = d.abs();

    let mut x = mla(
        simd,
        splat(simd, -0.601_564_466_953_277_587_890_625),
        d,
        splat(simd, 2.820_889_234_542_846_679_687_5),
    );
    x = mla(simd, x, d, splat(simd, -5.532_182_216_644_287_109_375));
    x = mla(simd, x, d, splat(simd, 5.898_262_500_762_939_453_125));
    x = mla(simd, x, d, splat(simd, -3.809_541_702_270_507_812_5));
    x = mla(simd, x, d, splat(simd, 2.224_125_623_703_002_929_687_5));

    let mut y = x * x;
    y = y * y;
    x = x - neg_mul_add(simd, d, y, x) * splat(simd, -1. / 3.);

    let z = x;

    let mut u = mul_as_doubled::<S>(x, x);
    u = u * u;
    u = mul_v(u, d);
    u = add_v(u, -x);
    y = u.0 + u.1;

    y = splat(simd, -2. / 3.) * y * z;
    let mut v = add_v(mul_as_doubled(z, z), y);
    v = mul_v(v, d);
    v *= q2;
    let mut z = ldexp2k(simd, v.0 + v.1, qu - splat_int(simd, 2048));

    z = d
        .is_infinite()
        .select(mul_sign(simd, splat(simd, f32::INFINITY), q2.0), z);
    d.simd_eq(splat(simd, 0.))
        .select(from_bits(simd, sign_bit(simd, q2.0)), z)
}

/// `sin(pi * d)` for a reduced argument, kept as a double-double.
///
/// Upstream keeps this helper in `src/f32x.rs`; `gammafk` needs it, so it is carried here, as the
/// `u05` tier carries its own copy.
#[inline(always)]
fn sinpifk<S: Simd>(simd: S, d: F32s<S>) -> DD<S> {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 31)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 2));

    let s = u - q.to_float::<F32s<S>>();
    let t = s;
    let s = s * s;
    let s2 = mul_as_doubled(t, t);

    let mut u = mla(
        simd,
        select_splat(simd, o, -0.243_061_180_1_e-7, 0.309_384_205_4_e-6),
        s,
        select_splat(simd, o, 0.359_057_708_e-5, -0.365_730_738_8_e-4),
    );
    u = mla(
        simd,
        u,
        s,
        select_splat(simd, o, -0.325_991_772_1_e-3, 0.249_039_358_5_e-2),
    );

    let mut x = v_add_dd(
        simd,
        u * s,
        select_doubled(
            simd,
            o,
            DD::new(
                splat(simd, 0.015_854_343_771_934_509_277),
                splat(simd, 4.494_005_135_403_224_281_1_e-10),
            ),
            DD::new(
                splat(simd, -0.080_745_510_756_969_451_904),
                splat(simd, -1.337_366_533_907_693_625_8_e-9),
            ),
        ),
    );
    x = s2 * x
        + select_doubled(
            simd,
            o,
            DD::new(
                splat(simd, -0.308_425_128_459_930_419_92),
                splat(simd, -9.072_833_903_073_392_227_7_e-9),
            ),
            DD::new(
                splat(simd, 0.785_398_185_253_143_310_55),
                splat(simd, -2.185_733_861_756_648_485_5_e-8),
            ),
        );

    x *= select_doubled(simd, o, s2, DD::new(t, splat(simd, 0.)));
    x = select_doubled(simd, o, add_v(x, splat(simd, 1.)), x);

    let o = (q & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    x.0 = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, x.0),
    );
    x.1 = from_bits(
        simd,
        o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & splat_bits(simd, (-0.0f32).to_bits())
            ^ to_bits(simd, x.1),
    );

    x
}

/// Gamma function core: returns the double-double pair used by [`tgamma`] and [`lgamma`].
#[inline(always)]
fn gammafk<S: Simd>(simd: S, a: F32s<S>) -> (DD<S>, DD<S>) {
    let mut clln = DD::new(splat(simd, 1.), splat(simd, 0.));
    let mut clld = DD::new(splat(simd, 1.), splat(simd, 0.));

    let otiny = a.abs().simd_lt(splat(simd, 1e-30));
    let oref = a.simd_lt(splat(simd, 0.5));

    let x = select_doubled(
        simd,
        otiny,
        DD::new(splat(simd, 0.), splat(simd, 0.)),
        select_doubled(
            simd,
            oref,
            add_as_doubled(splat(simd, 1.), -a),
            DD::new(a, splat(simd, 0.)),
        ),
    );

    let o0 = splat(simd, 0.5).simd_le(x.0) & x.0.simd_le(splat(simd, 1.2));
    let o2 = splat(simd, 2.3).simd_le(x.0);

    let mut y = normalize(add_v(x, splat(simd, 1.)) * x);
    y = normalize(add_v(x, splat(simd, 2.)) * y);

    let o = o2 & x.0.simd_le(splat(simd, 7.));
    clln = select_doubled(simd, o, y, clln);

    let mut x = select_doubled(simd, o, add_v(x, splat(simd, 3.)), x);
    let t = o2.select(
        splat(simd, 1.) / x.0,
        normalize(add_v(x, select_splat(simd, o0, -1., -2.))).0,
    );

    let u = mla(
        simd,
        select3(
            simd,
            o2,
            o0,
            0.000_839_498_720_672_087_279_971_000_786,
            0.943_515_777_6,
            0.110_248_955_e-3,
        ),
        t,
        select3(
            simd,
            o2,
            o0,
            -5.171_790_908_260_592_193_293_944_22_e-5,
            0.867_006_361_5,
            0.816_001_993_4_e-4,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.000_592_166_437_353_693_882_857_342_347,
            0.482_670_247_6,
            0.152_846_885_6_e-3,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            6.972_813_758_365_857_774_037_435_39_e-5,
            -0.885_512_977_8_e-1,
            -0.235_506_871_8_e-3,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.000_784_039_221_720_066_627_493_314_301,
            0.101_382_523_8,
            0.496_224_209_2_e-3,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.000_229_472_093_621_399_176_949_318_732,
            -0.149_340_897_8,
            -0.119_348_801_7_e-2,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            -0.002_681_327_160_493_827_160_473_958_490,
            0.169_750_914,
            0.289_159_943_3_e-2,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.003_472_222_222_222_222_222_175_164_840,
            -0.207_245_454_2,
            -0.738_545_181_2_e-2,
        ),
    );
    let u = mla(
        simd,
        u,
        t,
        select3(
            simd,
            o2,
            o0,
            0.083_333_333_333_333_333_335_592_087_900,
            0.270_587_235_7,
            0.205_807_704_5_e-1,
        ),
    );

    y = add_v(x, splat(simd, -0.5)) * logk2f(simd, x);
    y += -x;
    y += DD::splat(simd, (0.918_938_517_570_495_6, 1.563_417_661_998_301_e-8)); // 0.5*log(2*M_PI)

    let mut z = add_v(
        mul_as_doubled(u, t),
        select_splat(
            simd,
            o0,
            -0.400_686_534_596_170_958_447_352_690_395,
            -0.673_523_028_297_382_446_749_257_758_235_e-1,
        ),
    );
    z = add_v(
        mul_v(z, t),
        select_splat(
            simd,
            o0,
            0.822_466_960_142_643_054_450_325_495_997,
            0.322_467_033_928_981_157_743_538_726_901,
        ),
    );
    z = add_v(
        mul_v(z, t),
        select_splat(
            simd,
            o0,
            -0.577_215_665_946_766_039_837_398_973_297,
            0.422_784_335_087_484_338_986_941_629_852,
        ),
    );
    mul_assign_v(&mut z, t);

    let mut clc = select_doubled(simd, o2, y, z);

    clld = select_doubled(simd, o2, add_v(mul_as_doubled(u, t), splat(simd, 1.)), clld);

    y = clln;

    clc = select_doubled(
        simd,
        otiny,
        DD::splat(simd, (41.588_829_040_527_344, 1.793_069_372_979_516_6_e-6)), // log(2^60)
        select_doubled(
            simd,
            oref,
            DD::splat(simd, (1.144_729_852_676_391_6, 3.317_300_922_844_879_e-8)) + (-clc), // log(M_PI)
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
        let t = a - splat(simd, F1_12)
            * trunci(simd, a * (splat(simd, 1.) / splat(simd, F1_12))).to_float::<F32s<S>>();
        x = clld * sinpifk(simd, t);
    }

    clld = select_doubled(
        simd,
        otiny,
        DD::new(
            a * (splat(simd, F1_30) * splat(simd, F1_30)),
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
pub fn tgamma<S: Simd>(simd: S, a: F32s<S>) -> F32s<S> {
    let (da, db) = gammafk(simd, a);
    let y = expk2f(simd, da) * db;
    let r = y.0 + y.1;

    let o = a.simd_eq(splat(simd, f32::NEG_INFINITY))
        | (a.simd_lt(splat(simd, 0.)) & is_integer(simd, a))
        | (a.is_finite() & a.simd_lt(splat(simd, 0.)) & r.is_nan());
    let r = o.select(splat(simd, f32::NAN), r);

    let o = (a.simd_eq(splat(simd, f32::INFINITY)) | a.is_finite())
        & a.simd_ge(splat(simd, -f32::MIN_POSITIVE))
        & (a.simd_eq(splat(simd, 0.)) | a.simd_gt(splat(simd, 36.)) | r.is_nan());
    o.select(mul_sign(simd, splat(simd, f32::INFINITY), a), r)
}

/// Log gamma function.
///
/// The error bound of the returned value is `1.0 ULP` if the argument is positive. If the argument
/// is larger than `4e+36`, it may return infinity instead of the correct value. The error bound is
/// `max(1 ULP and 1e-8)`, if the argument is negative.
#[inline(always)]
pub fn lgamma<S: Simd>(simd: S, a: F32s<S>) -> F32s<S> {
    let (da, db) = gammafk(simd, a);
    // `Doubled::abs`, which upstream keeps on the type rather than as a free function.
    let db = DD::new(
        fabs(simd, db.0),
        from_bits(
            simd,
            to_bits(simd, db.1) ^ (to_bits(simd, db.0) & splat_bits(simd, (-0.0f32).to_bits())),
        ),
    );
    let y = da + logk2f(simd, db);
    let r = y.0 + y.1;

    let o = a.is_infinite()
        | ((a.simd_le(splat(simd, 0.)) & is_integer(simd, a)) | (a.is_finite() & r.is_nan()));
    o.select(splat(simd, f32::INFINITY), r)
}

/// `z + y * x`, all in double-double.
#[inline(always)]
fn dfmla<S: Simd>(simd: S, x: F32s<S>, y: DD<S>, z: DD<S>) -> DD<S> {
    let _ = simd;
    z + mul_v(y, x)
}

/// Degree-2 double-double polynomial with double-double coefficients.
#[inline(always)]
fn poly2df_b<S: Simd>(simd: S, x: F32s<S>, c1: DD<S>, c0: DD<S>) -> DD<S> {
    dfmla(simd, x, c1, c0)
}

/// Degree-2 double-double polynomial with a vector leading coefficient.
#[inline(always)]
fn poly2df<S: Simd>(simd: S, x: F32s<S>, c1: F32s<S>, c0: DD<S>) -> DD<S> {
    dfmla(simd, x, DD::new(c1, splat(simd, 0.)), c0)
}

/// Degree-4 double-double polynomial.
#[inline(always)]
fn poly4df<S: Simd>(simd: S, x: F32s<S>, c3: F32s<S>, c2: DD<S>, c1: DD<S>, c0: DD<S>) -> DD<S> {
    dfmla(
        simd,
        x * x,
        poly2df(simd, x, c3, c2),
        poly2df_b(simd, x, c1, c0),
    )
}

// Upstream's `Poly<Self>` instance, where every coefficient is a vector (`Poly::c2v(c) = c`),
// sitting alongside the shared scalar-coefficient `polyN` helpers in `super`. `erf` needs it for
// the branch whose coefficients differ per lane.

/// Vector-coefficient counterpart of [`poly6`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn poly6v<S: Simd>(
    simd: S,
    x: F32s<S>,
    x2: F32s<S>,
    x4: F32s<S>,
    c5: F32s<S>,
    c4: F32s<S>,
    c3: F32s<S>,
    c2: F32s<S>,
    c1: F32s<S>,
    c0: F32s<S>,
) -> F32s<S> {
    mla(
        simd,
        x4,
        mla(simd, x, c5, c4),
        mla(simd, x2, mla(simd, x, c3, c2), mla(simd, x, c1, c0)),
    )
}

/// Error function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn erf<S: Simd>(simd: S, a: F32s<S>) -> F32s<S> {
    let x = a.abs();
    let x2 = x * x;
    let x4 = x2 * x2;
    let o25 = x.simd_le(splat(simd, 2.5));

    let mut t2;
    if o25.all_true() {
        // Abramowitz and Stegun
        let t = poly6(
            simd,
            x,
            x2,
            x4,
            -0.436_044_700_8_e-6,
            0.686_751_536_7_e-5,
            -0.304_515_67_e-4,
            0.980_853_656_1_e-4,
            0.239_552_391_6_e-3,
            0.145_990_154_1_e-3,
        );
        t2 = poly4df(
            simd,
            x,
            t,
            DD::new(
                splat(simd, 0.009_288_344_532_251_358_032_2),
                splat(simd, -2.786_374_589_702_533_075_5_e-11),
            ),
            DD::new(
                splat(simd, 0.042_275_499_552_488_327_026),
                splat(simd, 1.346_139_928_998_810_605_7_e-09),
            ),
            DD::new(
                splat(simd, 0.070_523_701_608_180_999_756),
                splat(simd, -3.661_630_931_870_736_516_3_e-09),
            ),
        );
        t2 = v_add_checked(splat(simd, 1.), mul_v(t2, x));
        t2 = square(t2);
        t2 = square(t2);
        t2 = square(t2);
        t2 = square(t2);
        t2 = recip(t2);
    } else {
        let t = poly6v(
            simd,
            x,
            x2,
            x4,
            select_splat(simd, o25, -0.436_044_700_8_e-6, -0.113_001_284_8_e-6),
            select_splat(simd, o25, 0.686_751_536_7_e-5, 0.411_527_298_6_e-5),
            select_splat(simd, o25, -0.304_515_67_e-4, -0.692_830_435_6_e-4),
            select_splat(simd, o25, 0.980_853_656_1_e-4, 0.717_269_256_7_e-3),
            select_splat(simd, o25, 0.239_552_391_6_e-3, -0.513_104_535_6_e-2),
            select_splat(simd, o25, 0.145_990_154_1_e-3, 0.270_863_715_6_e-1),
        );
        t2 = poly4df(
            simd,
            x,
            t,
            select_doubled(
                simd,
                o25,
                DD::new(
                    splat(simd, 0.009_288_344_532_251_358_032_2),
                    splat(simd, -2.786_374_589_702_533_075_5_e-11),
                ),
                DD::new(
                    splat(simd, -0.110_643_193_125_724_792_48),
                    splat(simd, 3.705_045_277_722_528_300_7_e-09),
                ),
            ),
            select_doubled(
                simd,
                o25,
                DD::new(
                    splat(simd, 0.042_275_499_552_488_327_026),
                    splat(simd, 1.346_139_928_998_810_605_7_e-09),
                ),
                DD::new(
                    splat(simd, -0.631_922_304_630_279_541_02),
                    splat(simd, -2.020_043_258_507_317_785_9_e-08),
                ),
            ),
            select_doubled(
                simd,
                o25,
                DD::new(
                    splat(simd, 0.070_523_701_608_180_999_756),
                    splat(simd, -3.661_630_931_870_736_516_3_e-09),
                ),
                DD::new(
                    splat(simd, -1.129_663_825_035_095_214_8),
                    splat(simd, 2.551_512_019_645_325_925_2_e-08),
                ),
            ),
        );
        mul_assign_v(&mut t2, x);
        let mut s2 = v_add_checked(splat(simd, 1.), t2);
        s2 = square(s2);
        s2 = square(s2);
        s2 = square(s2);
        s2 = square(s2);
        s2 = recip(s2);
        t2 = select_doubled(simd, o25, s2, DD::new(expkf(simd, t2), splat(simd, 0.)));
    }

    t2 = add_v(t2, splat(simd, -1.));
    t2 = select_doubled(
        simd,
        x.simd_lt(splat(simd, 1e-4)),
        mul_v(
            DD::new(
                splat(simd, -1.128_379_225_730_895_996_1),
                splat(simd, 5.863_538_342_219_759_109_7_e-08),
            ),
            x,
        ),
        t2,
    );

    let mut z = -(t2.0 + t2.1);
    z = x.simd_ge(splat(simd, 6.)).select(splat(simd, 1.), z);
    z = a.is_infinite().select(splat(simd, 1.), z);
    z = a.simd_eq(splat(simd, 0.)).select(splat(simd, 0.), z);
    mul_sign(simd, z, a)
}
