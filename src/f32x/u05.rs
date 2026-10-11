//! `f32` kernels with a 0.5 ULP error bound (SLEEF's `u05` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f32x/u05.rs`). The private `sinpifk` / `cospifk`
//! helpers live in upstream's `src/f32x.rs`; they are carried here because only this tier uses
//! them.

use super::*;
use crate::doubled::f32::{add_v, mul_as_doubled, mul_v, square, DD};

/// `sin(pi * d)` for a reduced argument, kept as a double-double.
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
    let s2 = mul_as_doubled::<S>(t, t);

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

/// Evaluates `sin(pi * d)` and `cos(pi * d)` for a given `d` simultaneously.
///
/// Evaluates the sine and cosine functions of `pi * d` at a time, and stores the two values in a
/// tuple. The error bound of the returned values is `max(0.506 ULP, f32::MIN_POSITIVE)` for `d`
/// in `[-1e+7, 1e+7]`. If `d` is a finite value out of this range, an arbitrary value within
/// `[-1, 1]` is returned. If `d` is a `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sincospi<S: Simd>(simd: S, d: F32s<S>) -> (F32s<S>, F32s<S>) {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 31)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let s = u - q.to_float::<F32s<S>>();

    let t = s;
    let s = s * s;
    let s2 = mul_as_doubled::<S>(t, t);

    //

    let mut u = splat(simd, 0.309_384_205_4_e-6);
    u = mla(simd, u, s, splat(simd, -0.365_730_738_8_e-4));
    u = mla(simd, u, s, splat(simd, 0.249_039_358_5_e-2));
    let mut x = v_add_dd(
        simd,
        u * s,
        DD::new(
            splat(simd, -0.080_745_510_756_969_451_904),
            splat(simd, -1.337_366_533_907_693_625_8_e-9),
        ),
    );
    x = s2 * x
        + DD::new(
            splat(simd, 0.785_398_185_253_143_310_55),
            splat(simd, -2.185_733_861_756_648_485_5_e-8),
        );

    x = mul_v(x, t);
    let rx = x.0 + x.1;

    let rx = is_neg_zero(simd, d).select(splat(simd, -0.0), rx);

    //

    let mut u = splat(simd, -0.243_061_180_1_e-7);
    u = mla(simd, u, s, splat(simd, 0.359_057_708_e-5));
    u = mla(simd, u, s, splat(simd, -0.325_991_772_1_e-3));
    let mut x = v_add_dd(
        simd,
        u * s,
        DD::new(
            splat(simd, 0.015_854_343_771_934_509_277),
            splat(simd, 4.494_005_135_403_224_281_1_e-10),
        ),
    );
    x = s2 * x
        + DD::new(
            splat(simd, -0.308_425_128_459_930_419_92),
            splat(simd, -9.072_833_903_073_392_227_7_e-9),
        );

    x = add_v(x * s2, splat(simd, 1.));
    let ry = x.0 + x.1;

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
        (!o).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) & to_bits(simd, rsin),
    );
    rcos = from_bits(
        simd,
        (!o).select(splat_bits(simd, u32::MAX), splat_bits(simd, 0)) & to_bits(simd, rcos),
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

/// Evaluates `sin(pi * d)` for a given `d`.
///
/// The error bound of the returned value is `max(0.506 ULP, f32::MIN_POSITIVE)` for `d` in
/// `[-1e+7, 1e+7]`. If `d` is a finite value out of this range, an arbitrary value within
/// `[-1, 1]` is returned. If `d` is a `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sinpi<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let x = sinpifk(simd, d);
    let mut r = x.0 + x.1;

    r = is_neg_zero(simd, d).select(splat(simd, -0.0), r);
    r = from_bits(
        simd,
        (!d.abs().simd_gt(splat(simd, TRIGRANGEMAX4_F)))
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & to_bits(simd, r),
    );
    from_bits(
        simd,
        d.is_infinite()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, r),
    )
}

/// `cos(pi * d)` for a reduced argument, kept as a double-double.
#[inline(always)]
fn cospifk<S: Simd>(simd: S, d: F32s<S>) -> DD<S> {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 31)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 0));

    let s = u - q.to_float::<F32s<S>>();
    let t = s;
    let s = s * s;
    let s2 = mul_as_doubled::<S>(t, t);

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

    let o = ((q + splat_int(simd, 2)) & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    x = DD::new(
        from_bits(
            simd,
            o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                & splat_bits(simd, (-0.0f32).to_bits())
                ^ to_bits(simd, x.0),
        ),
        from_bits(
            simd,
            o.select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
                & splat_bits(simd, (-0.0f32).to_bits())
                ^ to_bits(simd, x.1),
        ),
    );

    x
}

/// Evaluates `cos(pi * d)` for a given `d`.
///
/// The error bound of the returned value is `max(0.506 ULP, f32::MIN_POSITIVE)` for `d` in
/// `[-1e+7, 1e+7]`. If `d` is a finite value out of this range, an arbitrary value within
/// `[-1, 1]` is returned. If `d` is a `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn cospi<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let x = cospifk(simd, d);
    let r = x.0 + x.1;

    let r = d
        .abs()
        .simd_gt(splat(simd, TRIGRANGEMAX4_F))
        .select(splat(simd, 1.), r);
    from_bits(
        simd,
        d.is_infinite()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            | to_bits(simd, r),
    )
}

/// Square root function.
///
/// The error bound of the returned value is `0.5001 ULP`.
///
/// Upstream selects between two formulations at compile time. This port takes the
/// `cfg!(target_feature = "fma")` branch, whose final correction needs the fused [`mla`];
/// [`crate::f64x::u05::sqrt`] still carries upstream's other branch, and the two files therefore
/// differ in shape at this one function.
#[inline(always)]
pub fn sqrt<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let d = d.simd_lt(splat(simd, 0.)).select(splat(simd, f32::NAN), d);

    let o = d.simd_lt(splat(simd, 5.293_955_920_339_377_e-23));
    let d = o.select(d * splat(simd, 1.888_946_593_147_858_e+22), d);
    let q = o.select(splat(simd, 7.275_957_614_183_426_e-12), splat(simd, 1.));

    let mut y = from_bits(
        simd,
        int_to_bits(
            simd,
            splat_int(simd, 0x5f3759df)
                - (bits_to_int(simd, to_bits(simd, d)) >> splat_int(simd, 1)),
        ),
    );

    let mut x = d * y;
    let mut w = splat(simd, 0.5) * y;
    y = neg_mul_add(simd, x, w, splat(simd, 0.5));
    x = mla(simd, x, y, x);
    w = mla(simd, w, y, w);
    y = neg_mul_add(simd, x, w, splat(simd, 0.5));
    x = mla(simd, x, y, x);
    w = mla(simd, w, y, w);

    y = neg_mul_add(simd, x, w, splat(simd, 1.5));
    w += w;
    w *= y;
    x = w * d;
    y = mul_sub(simd, w, d, x);
    let mut z = neg_mul_add(simd, w, x, splat(simd, 1.));

    z = neg_mul_add(simd, w, y, z);
    w = splat(simd, 0.5) * x;
    w = mla(simd, w, z, y);
    w += x;

    w *= q;

    w = (d.simd_eq(splat(simd, 0.)) | d.simd_eq(splat(simd, f32::INFINITY))).select(d, w);

    d.simd_lt(splat(simd, 0.)).select(splat(simd, f32::NAN), w)
}

/// 2D Euclidean distance function.
///
/// The error bound of the returned value is `0.5001 ULP`.
#[inline(always)]
pub fn hypot<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let x = x.abs();
    let y = y.abs();
    let min = x.min(y);
    let n = min;
    let max = x.max(y);
    let d = max;

    let o = max.simd_lt(splat(simd, f32::MIN_POSITIVE));
    let n = o.select(n * splat(simd, F1_24), n);
    let d = o.select(d * splat(simd, F1_24), d);

    let t = DD::new(n, splat(simd, 0.)) / DD::new(d, splat(simd, 0.));
    let t: DD<S> = mul_v(
        crate::doubled::f32::sqrt(add_v(square(t), splat(simd, 1.))),
        max,
    );
    let mut ret = t.0 + t.1;
    ret = ret.is_nan().select(splat(simd, f32::INFINITY), ret);
    ret = min.simd_eq(splat(simd, 0.)).select(max, ret);
    ret = (x.is_nan() | y.is_nan()).select(splat(simd, f32::NAN), ret);
    (x.simd_eq(splat(simd, f32::INFINITY)) | y.simd_eq(splat(simd, f32::INFINITY)))
        .select(splat(simd, f32::INFINITY), ret)
}
