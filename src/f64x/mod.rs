//! Vector helpers for the `f64` kernels.
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x.rs`, `src/f64x/constants.rs`) and the `doubled`
//! crate v0.3.3. See the crate-level [`crate::common`] docs for the translation table.

pub mod constants;
pub mod u10;
pub mod u35;

use fearless_simd::prelude::*;
use fearless_simd::{ExtractToken, Simd};

use crate::common::{F64s, I64s, M64s, U64s};
use crate::constants::*;
use crate::doubled::f64::{
    add_as_doubled, add_v, mul_as_doubled, mul_v, normalize, recip_as_doubled, scale,
    v_add_checked, DD,
};

pub use constants::{splat, splat_bits, splat_int};
use constants::REMPITABDP;

/// Reinterprets an `f64` vector as its bit pattern.
#[inline(always)]
pub fn to_bits<S: Simd>(x: F64s<S>) -> U64s<S> {
    x.bitcast()
}

/// Reinterprets a bit pattern as an `f64` vector.
#[inline(always)]
pub fn from_bits<S: Simd>(x: U64s<S>) -> F64s<S> {
    x.bitcast()
}

/// Reinterprets a bit pattern as a signed integer vector of the same lane count.
#[inline(always)]
pub fn bits_to_int<S: Simd>(x: U64s<S>) -> I64s<S> {
    x.bitcast()
}

/// Reinterprets a signed integer vector as a bit pattern.
#[inline(always)]
pub fn int_to_bits<S: Simd>(x: I64s<S>) -> U64s<S> {
    x.bitcast()
}

/// Sign bit of `x`, as a bit pattern.
#[inline(always)]
pub fn sign_bit<S: Simd>(x: F64s<S>) -> U64s<S> {
    to_bits(x) & splat_bits(x.token(), (-0.0f64).to_bits())
}

/// `1.0` carrying the sign of `x`.
#[inline(always)]
pub fn sign<S: Simd>(x: F64s<S>) -> F64s<S> {
    from_bits(sign_bit(x) ^ splat_bits(x.token(), 1.0f64.to_bits()))
}

/// `x` with the sign bit of `y` applied.
#[inline(always)]
pub fn mul_sign<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    from_bits(to_bits(x) ^ sign_bit(y))
}

/// `x` with the sign bit of `y` set.
#[inline(always)]
pub fn or_sign<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    from_bits(to_bits(x) | sign_bit(y))
}

/// `x` with the sign of `y`.
#[inline(always)]
pub fn copy_sign<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    from_bits(
        (splat_bits(x.token(), !(-0.0f64).to_bits()) & to_bits(x)) ^ sign_bit(y),
    )
}

/// `x * y + z`.
#[inline(always)]
pub fn mla<S: Simd>(x: F64s<S>, y: F64s<S>, z: F64s<S>) -> F64s<S> {
    x.mul_add(y, z)
}

/// `x * y - z`.
#[inline(always)]
pub fn mul_sub<S: Simd>(x: F64s<S>, y: F64s<S>, z: F64s<S>) -> F64s<S> {
    x.mul_sub(y, z)
}

/// `z - x * y`.
#[inline(always)]
pub fn neg_mul_add<S: Simd>(x: F64s<S>, y: F64s<S>, z: F64s<S>) -> F64s<S> {
    (-x).mul_add(y, z)
}

/// Whether `x` is exactly `-0.0`.
#[inline(always)]
pub fn is_neg_zero<S: Simd>(x: F64s<S>) -> M64s<S> {
    to_bits(x).simd_eq(splat_bits(x.token(), (-0.0f64).to_bits()))
}

/// Round to nearest integer, ties to even.
#[inline(always)]
pub fn rint<S: Simd>(d: F64s<S>) -> F64s<S> {
    let simd = d.token();
    let c = mul_sign(splat(simd, D1_52), d);
    d.abs()
        .simd_gt(splat(simd, D1_52))
        .select(d, or_sign((d + c) - c, d))
}

/// Truncate towards zero.
#[inline(always)]
pub fn trunc<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut fr = x - splat(simd, D1_31) * trunci(splat(simd, 1.0 / D1_31), x).to_float();
    fr -= trunci(simd, fr).to_float();
    (x.is_infinite() | x.abs().simd_ge(splat(simd, D1_52))).select(x, copy_sign(x - fr, x))
}

/// Round towards negative infinity.
#[inline(always)]
pub fn floor<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut fr = x - splat(simd, D1_31) * trunci(splat(simd, 1.0 / D1_31), x).to_float();
    fr -= trunci(simd, fr).to_float();
    fr = fr.simd_lt(splat(simd, 0.)).select(fr + splat(simd, 1.), fr);
    (x.is_infinite() | x.abs().simd_ge(splat(simd, D1_52))).select(x, copy_sign(x - fr, x))
}

/// Round towards positive infinity.
#[inline(always)]
pub fn ceil<S: Simd>(simd: S, x: F64s<S>) -> F64s<S> {
    let mut fr = x - splat(simd, D1_31) * trunci(splat(simd, 1.0 / D1_31), x).to_float();
    fr -= trunci(simd, fr).to_float();
    fr = fr.simd_le(splat(simd, 0.)).select(fr, fr - splat(simd, 1.));
    (x.is_infinite() | x.abs().simd_ge(splat(simd, D1_52))).select(x, copy_sign(x - fr, x))
}

/// Round away from zero.
#[inline(always)]
pub fn round<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut x = d + splat(simd, 0.5);
    let mut fr = x - splat(simd, D1_31) * trunci(splat(simd, 1.0 / D1_31), x).to_float();
    fr -= trunci(simd, fr).to_float();
    x = (x.simd_le(splat(simd, 0.)) & fr.simd_eq(splat(simd, 0.))).select(x - splat(simd, 1.), x);
    fr = fr.simd_lt(splat(simd, 0.)).select(fr + splat(simd, 1.), fr);
    x = d
        .simd_eq(splat(simd, 0.499_999_999_999_999_944_49))
        .select(splat(simd, 0.), x);
    (d.is_infinite() | d.abs().simd_ge(splat(simd, D1_52))).select(d, copy_sign(x - fr, d))
}

/// Truncate to an integer vector.
#[inline(always)]
pub fn trunci<S: Simd>(_simd: S, x: F64s<S>) -> I64s<S> {
    x.trunc().to_int::<I64s<S>>()
}

/// Round to an integer vector, ties to even.
#[inline(always)]
pub fn roundi<S: Simd>(_simd: S, x: F64s<S>) -> I64s<S> {
    x.round_ties_even().to_int::<I64s<S>>()
}

/// Whether `x` is integral.
#[inline(always)]
pub fn is_integer<S: Simd>(simd: S, x: F64s<S>) -> M64s<S> {
    let mut t = trunc(simd, x * splat(simd, 1.0 / D1_31));
    t = mla(splat(simd, -D1_31), t, x);
    trunc(simd, t).simd_eq(t) | x.abs().simd_gt(splat(simd, D1_53))
}

/// Whether `x` is an odd integer.
#[inline(always)]
pub fn is_odd<S: Simd>(simd: S, x: F64s<S>) -> M64s<S> {
    let mut t = trunc(simd, x * splat(simd, 1.0 / D1_31));
    t = mla(splat(simd, -D1_31), t, x);
    (trunci(simd, t) & splat_int(simd, 1)).simd_eq(splat_int(simd, 1))
        & x.abs().simd_lt(splat(simd, D1_53))
}

/// Selects between two scalars broadcast to vectors.
#[inline(always)]
pub fn select_splat<S: Simd>(mask: M64s<S>, if_true: f64, if_false: f64) -> F64s<S> {
    let simd = mask.token();
    mask.select(splat(simd, if_true), splat(simd, if_false))
}

/// Selects between three scalar values using two masks.
#[inline(always)]
pub fn select3<S: Simd>(o0: M64s<S>, o1: M64s<S>, d0: f64, d1: f64, d2: f64) -> F64s<S> {
    let simd = o0.token();
    o0.select(splat(simd, d0), select_splat(o1, d1, d2))
}

/// Selects between four scalar values using three masks.
#[inline(always)]
pub fn select4<S: Simd>(
    o0: M64s<S>,
    o1: M64s<S>,
    o2: M64s<S>,
    d0: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> F64s<S> {
    let simd = o0.token();
    o0.select(
        splat(simd, d0),
        o1.select(splat(simd, d1), select_splat(o2, d2, d3)),
    )
}

/// Selects between two double-double values using a mask.
#[inline(always)]
pub fn select_doubled<S: Simd>(mask: M64s<S>, l: DD<S>, r: DD<S>) -> DD<S> {
    DD::new(mask.select(l.0, r.0), mask.select(l.1, r.1))
}

/// `2^q` for an integer vector `q`.
#[inline(always)]
pub fn pow2i<S: Simd>(q: I64s<S>) -> F64s<S> {
    from_bits(int_to_bits((splat_int(q.token(), 0x3ff) + q) << 52))
}

/// `d * 2^e`, split into two half-range multiplications.
#[inline(always)]
pub fn ldexp2k<S: Simd>(d: F64s<S>, e: I64s<S>) -> F64s<S> {
    let e1 = e >> 1;
    d * pow2i(e1) * pow2i(e - e1)
}

/// `d * 2^q` by adding directly to the exponent field.
#[inline(always)]
pub fn ldexp3k<S: Simd>(d: F64s<S>, q: I64s<S>) -> F64s<S> {
    from_bits(to_bits(d) + int_to_bits(q << 52))
}

/// The exponent of `d`, without subnormal handling.
#[inline(always)]
pub fn ilogb2k<S: Simd>(_simd: S, d: F64s<S>) -> I64s<S> {
    bits_to_int((to_bits(d) >> 52) & splat_bits(d.token(), 0x7ff))
        - splat_int(d.token(), 0x3ff)
}

/// The exponent of `d`, scaling subnormals into range first.
#[inline(always)]
pub fn ilogbk<S: Simd>(simd: S, mut d: F64s<S>) -> I64s<S> {
    let o = d.simd_lt(splat(simd, 4.909_093_465_297_726_6_e-91));
    d = o.select(splat(simd, 2.037_035_976_334_486_e90) * d, d);
    let q = bits_to_int((to_bits(d) >> 52) & splat_bits(simd, 0x7ff));
    q - o.select(splat_int(simd, 300 + 0x3ff), splat_int(simd, 0x3ff))
}

/// `x * 2^q` over the full exponent range, including subnormals.
#[inline(always)]
pub fn ldexpk<S: Simd>(x: F64s<S>, q: I64s<S>) -> F64s<S> {
    let simd = q.token();
    let mut m = q >> 31;
    m = (((m + q) >> 9) - m) << 7;
    let q = q - (m << 2);
    m = splat_int(simd, 0x3ff) + m;
    m = !splat_int(simd, 0).simd_gt(m).to_int::<I64s<S>>() & m;
    m = m
        .simd_gt(splat_int(simd, 0x7ff))
        .select(splat_int(simd, 0x7ff), m);
    let y = from_bits(int_to_bits(m) << 52);
    x * y * y * y * y * pow2i(q)
}

/// Multiply by an integral power of two.
#[inline(always)]
pub fn ldexp<S: Simd>(x: F64s<S>, q: I64s<S>) -> F64s<S> {
    ldexpk(x, q)
}

/// The exponent of `d` as an integer vector.
#[inline(always)]
pub fn ilogb<S: Simd>(simd: S, d: F64s<S>) -> I64s<S> {
    let mut e = ilogbk(simd, d.abs()).to_float();
    e = d
        .simd_eq(splat(simd, 0.))
        .select(splat(simd, SLEEF_FP_ILOGB0 as f64), e);
    e = d
        .is_nan()
        .select(splat(simd, SLEEF_FP_ILOGBNAN as f64), e);
    e = d.is_infinite().select(splat(simd, f64::MAX), e);
    roundi(simd, e)
}

/// Picks `m` with the sign of `d` for infinite `d`, otherwise `d`.
#[inline(always)]
pub fn visinf2<S: Simd>(d: F64s<S>, m: F64s<S>) -> F64s<S> {
    let simd = d.token();
    from_bits(
        d.is_infinite()
            .select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))
            & ((to_bits(d) & sign_bit(d)) | to_bits(m)),
    )
}

/// Maximum, treating a NaN in `y` as `x`.
#[inline(always)]
pub fn fmax<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    y.is_nan().select(x, x.simd_max(y))
}

/// Minimum, treating a NaN in `y` as `x`.
#[inline(always)]
pub fn fmin<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    y.is_nan().select(x, x.simd_min(y))
}

/// Positive difference.
#[inline(always)]
pub fn fdim<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    let ret = x - y;
    (ret.simd_lt(splat(x.token(), 0.)) | x.simd_eq(y)).select(splat(x.token(), 0.), ret)
}

/// Absolute value.
#[inline(always)]
pub fn fabs<S: Simd>(x: F64s<S>) -> F64s<S> {
    x.abs()
}

/// The next representable value after `x` in the direction of `y`.
#[inline(always)]
pub fn nextafter<S: Simd>(x: F64s<S>, y: F64s<S>) -> F64s<S> {
    let simd = x.token();
    let x = x
        .simd_eq(splat(simd, 0.))
        .select(mul_sign(splat(simd, 0.), y), x);
    let xi2 = bits_to_int(to_bits(x));
    let c = x.is_sign_negative() ^ y.simd_ge(x);

    let xi2 = c.select(-(xi2 ^ splat_int(simd, i64::MIN)), xi2);
    let xi2 = x.simd_ne(y).select(xi2 - splat_int(simd, 1), xi2);
    let xi2 = c.select(-(xi2 ^ splat_int(simd, i64::MIN)), xi2);

    let mut ret = from_bits(int_to_bits(xi2));

    ret = (ret.simd_eq(splat(simd, 0.)) & x.simd_ne(splat(simd, 0.)))
        .select(mul_sign(splat(simd, 0.), x), ret);
    ret = (x.simd_eq(splat(simd, 0.)) & y.simd_eq(splat(simd, 0.))).select(y, ret);
    (x.is_nan() | y.is_nan()).select(splat(simd, f64::NAN), ret)
}

/// `sqrt(x)` as a double-double.
#[inline(always)]
pub fn sqrt_as_doubled<S: Simd>(x: F64s<S>) -> DD<S> {
    let simd = x.token();
    let t = x.sqrt();
    scale(
        (DD::new(x, splat(simd, 0.0)) + mul_as_doubled(t, t)) * recip_as_doubled(t),
        splat(simd, 0.5),
    )
}

/// Removes the integer part of `4 * x`, returning the fraction and the integer part.
#[inline(always)]
pub fn rempisub<S: Simd>(x: F64s<S>) -> (F64s<S>, I64s<S>) {
    let simd = x.token();
    let c = mul_sign(splat(simd, D1_52), x);
    let rint4x = (splat(simd, 4.) * x).abs().simd_gt(splat(simd, D1_52)).select(
        splat(simd, 4.) * x,
        or_sign(mla(splat(simd, 4.), x, c) - c, x),
    );
    let rintx = x
        .abs()
        .simd_gt(splat(simd, D1_52))
        .select(x, or_sign((x + c) - c, x));

    let fr = mla(splat(simd, -0.25), rint4x, x);
    let vi = mla(splat(simd, -4.), rintx, rint4x).to_int::<I64s<S>>();
    (fr, vi)
}

/// Reduces `a` modulo `pi/2` using the Payne–Hanek tables.
#[inline(always)]
pub fn rempi<S: Simd>(mut a: F64s<S>) -> (DD<S>, I64s<S>) {
    let simd = a.token();
    let mut ex = ilogb2k(simd, a);
    ex -= splat_int(simd, 55);
    let mut q = ex
        .simd_gt(splat_int(simd, 700 - 55))
        .to_int::<I64s<S>>()
        & splat_int(simd, -64);
    a = ldexp3k(a, q);
    ex = !(ex >> 31) & ex;
    ex <<= 2;
    let mut x = mul_as_doubled(a, from_slice_offset(REMPITABDP.as_slice(), ex));
    let (did, dii) = rempisub(x.0);
    q = dii;
    x.0 = did;
    x = normalize(x);
    let mut y = mul_as_doubled(a, from_slice_offset(&REMPITABDP[1..], ex));
    x += y;
    let (did, dii) = rempisub(x.0);
    q += dii;
    x.0 = did;
    x = normalize(x);
    y = DD::new(
        from_slice_offset(&REMPITABDP[2..], ex),
        from_slice_offset(&REMPITABDP[3..], ex),
    );
    y *= a;
    x += y;
    x = normalize(x);
    x *= DD::splat(simd, (D_PI.0 * 2., D_PI.1 * 2.));
    let o = a.abs().simd_lt(splat(simd, 0.7));
    x = DD::new(
        o.select(a, x.0),
        from_bits(o.select(splat_bits(simd, 0), splat_bits(simd, u64::MAX)) & to_bits(x.1)),
    );
    (x, q)
}

/// Gathers one lane per element of `vi` from `table`.
#[inline(always)]
pub fn from_slice_offset<S: Simd>(table: &[f64], vi: I64s<S>) -> F64s<S> {
    let simd = vi.token();
    let idx = vi.to_array();
    let mut values = F64s::<S>::splat(simd, 0.0).to_array();
    for (slot, &i) in values.as_mut().iter_mut().zip(idx.as_ref().iter()) {
        *slot = table[i as usize];
    }
    F64s::<S>::load_array(simd, values)
}

/// Estrin polynomial evaluation, scalar coefficients, degree 2.
#[inline(always)]
pub fn poly2<S: Simd>(x: F64s<S>, c1: f64, c0: f64) -> F64s<S> {
    let simd = x.token();
    mla(x, splat(simd, c1), splat(simd, c0))
}

/// Estrin polynomial evaluation, scalar coefficients, degree 3.
#[inline(always)]
pub fn poly3<S: Simd>(x: F64s<S>, x2: F64s<S>, c2: f64, c1: f64, c0: f64) -> F64s<S> {
    let simd = x.token();
    mla(x2, splat(simd, c2), poly2(x, c1, c0))
}

/// Estrin polynomial evaluation, scalar coefficients, degree 4.
#[inline(always)]
pub fn poly4<S: Simd>(x: F64s<S>, x2: F64s<S>, c3: f64, c2: f64, c1: f64, c0: f64) -> F64s<S> {
    let simd = x.token();
    mla(
        x2,
        mla(x, splat(simd, c3), splat(simd, c2)),
        mla(x, splat(simd, c1), splat(simd, c0)),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 5.
#[inline(always)]
pub fn poly5<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(x4, splat(x.token(), c4), poly4(x, x2, c3, c2, c1, c0))
}

/// Estrin polynomial evaluation, scalar coefficients, degree 6.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly6<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(x4, poly2(x, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

/// Estrin polynomial evaluation, scalar coefficients, degree 7.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly7<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(x4, poly3(x, x2, c6, c5, c4), poly4(x, x2, c3, c2, c1, c0))
}

/// Estrin polynomial evaluation, scalar coefficients, degree 8.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly8<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x4,
        poly4(x, x2, c7, c6, c5, c4),
        poly4(x, x2, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 9.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly9<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(x8, splat(x.token(), c8), poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}

/// Estrin polynomial evaluation, scalar coefficients, degree 10.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly10<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly2(x, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 11.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly11<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly3(x, x2, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 12.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly12<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly4(x, x2, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 13.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly13<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly5(x, x2, x4, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 14.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly14<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly6(x, x2, x4, cd, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 15.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly15<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly7(x, x2, x4, ce, cd, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 16.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly16<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x8,
        poly8(x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
        poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 17.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly17<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    x16: F64s<S>,
    d0: f64,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x16,
        splat(x.token(), d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 18.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly18<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    x16: F64s<S>,
    d1: f64,
    d0: f64,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x16,
        poly2(x, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 19.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly19<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    x16: F64s<S>,
    d2: f64,
    d1: f64,
    d0: f64,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x16,
        poly3(x, x2, d2, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 20.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly20<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    x16: F64s<S>,
    d3: f64,
    d2: f64,
    d1: f64,
    d0: f64,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x16,
        poly4(x, x2, d3, d2, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation, scalar coefficients, degree 21.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn poly21<S: Simd>(
    x: F64s<S>,
    x2: F64s<S>,
    x4: F64s<S>,
    x8: F64s<S>,
    x16: F64s<S>,
    d4: f64,
    d3: f64,
    d2: f64,
    d1: f64,
    d0: f64,
    cf: f64,
    ce: f64,
    cd: f64,
    cc: f64,
    cb: f64,
    ca: f64,
    c9: f64,
    c8: f64,
    c7: f64,
    c6: f64,
    c5: f64,
    c4: f64,
    c3: f64,
    c2: f64,
    c1: f64,
    c0: f64,
) -> F64s<S> {
    mla(
        x16,
        poly5(x, x2, x4, d4, d3, d2, d1, d0),
        poly16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}
