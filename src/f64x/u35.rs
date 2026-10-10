//! `f64` kernels with a 3.5 ULP error bound (SLEEF's `u35` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x/u35.rs`).

use super::*;

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
