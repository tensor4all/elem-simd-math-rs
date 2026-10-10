//! `f64` kernels with a 0.5 ULP error bound (SLEEF's `u05` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x/u05.rs`). The private `sinpik` / `cospik`
//! helpers live in upstream's `src/f64x.rs`; they are carried here because only this tier uses
//! them.

use super::*;
use crate::doubled::f64::{add_v, mul_as_doubled, mul_v, recip_as_doubled, square, DD};

/// `sin(pi * d)` for a reduced argument, kept as a double-double.
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
    let s2 = mul_as_doubled::<S>(t, t);

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

    let mut x = v_add_dd(
        simd,
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
    x = select_doubled(simd, o, add_v(x, splat(simd, 1.)), x);

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

/// Evaluates `sin(pi * d)` and `cos(pi * d)` for a given `d` simultaneously.
///
/// Evaluates the sine and cosine functions of `pi * d` at a time, and stores the two values in a
/// tuple. The error bound of the returned values is `max(0.506 ULP, f64::MIN_POSITIVE)` for `d`
/// in `[-1e+9, 1e+9]`. If `d` is a finite value out of this range, an arbitrary value within
/// `[-1, 1]` is returned. If `d` is a `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sincospi<S: Simd>(simd: S, d: F64s<S>) -> (F64s<S>, F64s<S>) {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 63)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let s = u - q.to_float::<F64s<S>>();

    let t = s;
    let s = s * s;
    let s2 = mul_as_doubled::<S>(t, t);

    //

    let mut u = splat(simd, -2.024_611_207_851_823_992_958_68_e-14);
    u = mla(
        simd,
        u,
        s,
        splat(simd, 6.948_218_305_801_794_613_277_84_e-12),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -1.757_247_499_528_531_799_526_64_e-9),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 3.133_616_889_668_683_928_784_22_e-7),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -3.657_620_418_216_155_192_036_1_e-5),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 0.002_490_394_570_192_718_502_743_56),
    );
    let mut x = v_add_dd(
        simd,
        u * s,
        DD::new(
            splat(simd, -0.080_745_512_188_280_785_248_473_1),
            splat(simd, 3.618_524_750_670_371_048_499_87_e-18),
        ),
    );
    x = s2 * x
        + DD::new(
            splat(simd, 0.785_398_163_397_448_278_999_491),
            splat(simd, 3.062_871_137_271_550_026_071_05_e-17),
        );

    x = mul_v(x, t);
    let rx = x.0 + x.1;

    let rx = is_neg_zero(simd, d).select(splat(simd, -0.0), rx);

    //

    let mut u = splat(simd, 9.944_803_876_268_437_740_902_08_e-16);
    u = mla(
        simd,
        u,
        s,
        splat(simd, -3.897_962_260_629_327_991_640_47_e-13),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 1.150_115_825_399_960_352_669_01_e-10),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -2.461_136_950_104_469_749_535_9_e-8),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, 3.590_860_448_590_527_540_050_62_e-6),
    );
    u = mla(
        simd,
        u,
        s,
        splat(simd, -0.000_325_991_886_927_389_905_997_954),
    );
    let mut x = v_add_dd(
        simd,
        u * s,
        DD::new(
            splat(simd, 0.015_854_344_243_815_501_891_425_9),
            splat(simd, -1.046_932_722_806_315_219_088_45_e-18),
        ),
    );
    x = s2 * x
        + DD::new(
            splat(simd, -0.308_425_137_534_042_437_259_529),
            splat(simd, -1.956_984_921_336_335_503_383_45_e-17),
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
        (!o).select(splat_bits(simd, u64::MAX), splat_bits(simd, 0)) & to_bits(simd, rsin),
    );
    rcos = o.select(splat(simd, 1.), rcos);

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

/// Evaluates `sin(pi * d)` for a given `d`.
///
/// The error bound of the returned value is `max(0.506 ULP, f64::MIN_POSITIVE)` for `d` in
/// `[-1e+9, 1e+9]`. If `d` is a finite value out of this range, an arbitrary value within
/// `[-1, 1]` is returned. If `d` is a `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn sinpi<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let x = sinpik(simd, d);
    let mut r = x.0 + x.1;

    r = is_neg_zero(simd, d).select(splat(simd, -0.0), r);
    r = from_bits(
        simd,
        (!d.abs().simd_gt(splat(simd, TRIGRANGEMAX3 / 4.)))
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & to_bits(simd, r),
    );
    from_bits(
        simd,
        d.is_infinite()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, r),
    )
}

/// `cos(pi * d)` for a reduced argument, kept as a double-double.
#[inline(always)]
fn cospik<S: Simd>(simd: S, d: F64s<S>) -> DD<S> {
    let u = d * splat(simd, 4.);
    let mut q = trunci(simd, u);
    q = (q
        + (bits_to_int(simd, int_to_bits(simd, q) >> splat_bits(simd, 63)) ^ splat_int(simd, 1)))
        & splat_int(simd, !1);
    let o = (q & splat_int(simd, 2)).simd_eq(splat_int(simd, 0));

    let s = u - q.to_float::<F64s<S>>();
    let t = s;
    let s = s * s;
    let s2 = mul_as_doubled::<S>(t, t);

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
            0.002_490_394_570_192_718_502_743_560,
        ),
    );
    let mut x = v_add_dd(
        simd,
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
    x = select_doubled(simd, o, add_v(x, splat(simd, 1.)), x);

    let o = ((q + splat_int(simd, 2)) & splat_int(simd, 4)).simd_eq(splat_int(simd, 4));
    x = DD::new(
        from_bits(
            simd,
            o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                & splat_bits(simd, (-0.0f64).to_bits())
                ^ to_bits(simd, x.0),
        ),
        from_bits(
            simd,
            o.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
                & splat_bits(simd, (-0.0f64).to_bits())
                ^ to_bits(simd, x.1),
        ),
    );

    x
}

/// Evaluates `cos(pi * d)` for a given `d`.
///
/// The error bound of the returned value is `max(0.506 ULP, f64::MIN_POSITIVE)` for `d` in
/// `[-1e+9, 1e+9]`. If `d` is a finite value out of this range, an arbitrary value within
/// `[-1, 1]` is returned. If `d` is a `NaN` or infinity, a `NaN` is returned.
#[inline(always)]
pub fn cospi<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let x = cospik(simd, d);
    let r = x.0 + x.1;

    let r = d
        .abs()
        .simd_gt(splat(simd, TRIGRANGEMAX3 / 4.))
        .select(splat(simd, 1.), r);
    from_bits(
        simd,
        d.is_infinite()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            | to_bits(simd, r),
    )
}

/// Square root function.
///
/// The error bound of the returned value is `0.5001 ULP`.
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

    let d2: DD<S> = v_add_dd(simd, d, mul_as_doubled::<S>(x, x)) * recip_as_doubled::<S>(x);

    x = (d2.0 + d2.1) * q;

    x = d
        .simd_eq(splat(simd, f64::INFINITY))
        .select(splat(simd, f64::INFINITY), x);
    d.simd_eq(splat(simd, 0.)).select(d, x)
}

/// 2D Euclidean distance function.
///
/// The error bound of the returned value is `0.5001 ULP`.
#[inline(always)]
pub fn hypot<S: Simd>(simd: S, x: F64s<S>, y: F64s<S>) -> F64s<S> {
    let x = x.abs();
    let y = y.abs();
    let min = x.min(y);
    let n = min;
    let max = x.max(y);
    let d = max;

    let o = max.simd_lt(splat(simd, f64::MIN_POSITIVE));
    let n = o.select(n * splat(simd, D1_54), n);
    let d = o.select(d * splat(simd, D1_54), d);

    let t = DD::new(n, splat(simd, 0.)) / DD::new(d, splat(simd, 0.));
    let t: DD<S> = mul_v(
        crate::doubled::f64::sqrt(add_v(square(t), splat(simd, 1.))),
        max,
    );
    let mut ret = t.0 + t.1;
    ret = ret.is_nan().select(splat(simd, f64::INFINITY), ret);
    ret = min.simd_eq(splat(simd, 0.)).select(max, ret);
    ret = (x.is_nan() | y.is_nan()).select(splat(simd, f64::NAN), ret);
    (x.simd_eq(splat(simd, f64::INFINITY)) | y.simd_eq(splat(simd, f64::INFINITY)))
        .select(splat(simd, f64::INFINITY), ret)
}
