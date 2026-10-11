//! Vector helpers for the `f32` kernels.
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f32x.rs`, `src/f32x/constants.rs`) and the `doubled`
//! crate v0.3.3. See the crate-level [`crate::common`] docs for the translation table.
//!
//! Every helper takes the SIMD token as its first argument. `<S as Simd>::f32s` is not
//! invertible in argument position, so a signature whose only vector argument is that projection
//! cannot be inferred from a call; passing the token first makes every call site inferrable
//! without a turbofish.

pub mod constants;
pub mod fast;
pub mod u05;
pub mod u10;
pub mod u15;
pub mod u35;

use fearless_simd::prelude::*;
use fearless_simd::Simd;

pub use crate::common::{
    mla, mul_sub, neg_mul_add, poly10, poly11, poly12, poly13, poly14, poly15, poly16, poly17,
    poly18, poly19, poly2, poly20, poly21, poly3, poly4, poly5, poly6, poly7, poly8, poly9,
};
use crate::common::{F32s, I32s, M32s, U32s};
use crate::constants::*;
use crate::doubled::f32::{
    add_as_doubled, mul_as_doubled, mul_v, normalize, recip_as_doubled, scale, DD,
};

use constants::REMPITABSP;
pub use constants::{splat, splat_bits, splat_int};

/// Reinterprets an `f32` vector as its bit pattern.
#[inline(always)]
pub fn to_bits<S: Simd>(_simd: S, x: F32s<S>) -> U32s<S> {
    x.bitcast()
}

/// Reinterprets a bit pattern as an `f32` vector.
#[inline(always)]
pub fn from_bits<S: Simd>(_simd: S, x: U32s<S>) -> F32s<S> {
    x.bitcast()
}

/// Reinterprets a bit pattern as a signed integer vector of the same lane count.
#[inline(always)]
pub fn bits_to_int<S: Simd>(_simd: S, x: U32s<S>) -> I32s<S> {
    x.bitcast()
}

/// Reinterprets a signed integer vector as a bit pattern.
#[inline(always)]
pub fn int_to_bits<S: Simd>(_simd: S, x: I32s<S>) -> U32s<S> {
    x.bitcast()
}

/// Sign bit of `x`, as a bit pattern.
#[inline(always)]
pub fn sign_bit<S: Simd>(simd: S, x: F32s<S>) -> U32s<S> {
    to_bits(simd, x) & splat_bits(simd, (-0.0f32).to_bits())
}

/// `1.0` carrying the sign of `x`.
#[inline(always)]
pub fn sign<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    from_bits(simd, sign_bit(simd, x) ^ splat_bits(simd, 1.0f32.to_bits()))
}

/// `x` with the sign bit of `y` applied.
#[inline(always)]
pub fn mul_sign<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    from_bits(simd, to_bits(simd, x) ^ sign_bit(simd, y))
}

/// `x` with the sign bit of `y` set.
#[inline(always)]
pub fn or_sign<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    from_bits(simd, to_bits(simd, x) | sign_bit(simd, y))
}

/// `x` with the sign of `y`.
#[inline(always)]
pub fn copy_sign<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    from_bits(
        simd,
        (splat_bits(simd, !(-0.0f32).to_bits()) & to_bits(simd, x)) ^ sign_bit(simd, y),
    )
}

/// Whether `x` is exactly `-0.0`.
#[inline(always)]
pub fn is_neg_zero<S: Simd>(simd: S, x: F32s<S>) -> M32s<S> {
    to_bits(simd, x).simd_eq(splat_bits(simd, (-0.0f32).to_bits()))
}

/// Round to nearest integer, ties to even.
#[inline(always)]
pub fn rint<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let c = mul_sign(simd, splat(simd, F1_23), d);
    d.abs()
        .simd_gt(splat(simd, F1_23))
        .select(d, or_sign(simd, (d + c) - c, d))
}

/// Truncate towards zero.
#[inline(always)]
pub fn trunc<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let fr = x - trunci(simd, x).to_float::<F32s<S>>();
    (x.is_infinite() | x.abs().simd_ge(splat(simd, F1_23))).select(x, copy_sign(simd, x - fr, x))
}

/// Round towards negative infinity.
#[inline(always)]
pub fn floor<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let fr = x - trunci(simd, x).to_float::<F32s<S>>();
    let fr = fr.simd_lt(splat(simd, 0.)).select(fr + splat(simd, 1.), fr);
    (x.is_infinite() | x.abs().simd_ge(splat(simd, F1_23))).select(x, copy_sign(simd, x - fr, x))
}

/// Round towards positive infinity.
#[inline(always)]
pub fn ceil<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let fr = x - trunci(simd, x).to_float::<F32s<S>>();
    let fr = fr.simd_le(splat(simd, 0.)).select(fr, fr - splat(simd, 1.));
    (x.is_infinite() | x.abs().simd_ge(splat(simd, F1_23))).select(x, copy_sign(simd, x - fr, x))
}

/// Round away from zero.
#[inline(always)]
pub fn round<S: Simd>(simd: S, d: F32s<S>) -> F32s<S> {
    let mut x = d + splat(simd, 0.5);
    let fr = x - trunci(simd, x).to_float::<F32s<S>>();
    x = (x.simd_le(splat(simd, 0.)) & fr.simd_eq(splat(simd, 0.))).select(x - splat(simd, 1.), x);
    let fr = fr.simd_lt(splat(simd, 0.)).select(fr + splat(simd, 1.), fr);
    x = d
        .simd_eq(splat(simd, 0.499_999_970_197_677_612_3))
        .select(splat(simd, 0.), x);
    (d.is_infinite() | d.abs().simd_ge(splat(simd, F1_23))).select(d, copy_sign(simd, x - fr, d))
}

/// Truncate to an integer vector.
#[inline(always)]
pub fn trunci<S: Simd>(_simd: S, x: F32s<S>) -> I32s<S> {
    x.trunc().to_int::<I32s<S>>()
}

/// Round to an integer vector, ties to even.
#[inline(always)]
pub fn roundi<S: Simd>(_simd: S, x: F32s<S>) -> I32s<S> {
    x.round_ties_even().to_int::<I32s<S>>()
}

/// Whether `x` is integral.
///
/// Upstream writes this as `self.trunc().simd_eq(self)`, where `trunc` is its round-to-zero
/// *conversion*; this port uses the [`trunc`] helper instead. The two agree except for
/// `|x| >= 2^31`, where upstream's conversion of an out-of-range value says nothing about
/// integrality while [`trunc`] returns `x` itself, which is the mathematically correct answer
/// (every `f32` at or above `2^23` is an integer).
#[inline(always)]
pub fn is_integer<S: Simd>(simd: S, x: F32s<S>) -> M32s<S> {
    trunc(simd, x).simd_eq(x)
}

/// Whether `x` is an odd integer.
#[inline(always)]
pub fn is_odd<S: Simd>(simd: S, x: F32s<S>) -> M32s<S> {
    (trunci(simd, x) & splat_int(simd, 1)).simd_eq(splat_int(simd, 1))
        & x.abs().simd_lt(splat(simd, F1_24))
}

/// Selects between two scalars broadcast to vectors.
#[inline(always)]
pub fn select_splat<S: Simd>(simd: S, mask: M32s<S>, if_true: f32, if_false: f32) -> F32s<S> {
    mask.select(splat(simd, if_true), splat(simd, if_false))
}

/// Selects between three scalar values using two masks.
#[inline(always)]
pub fn select3<S: Simd>(simd: S, o0: M32s<S>, o1: M32s<S>, d0: f32, d1: f32, d2: f32) -> F32s<S> {
    o0.select(splat(simd, d0), select_splat(simd, o1, d1, d2))
}

/// Selects between four scalar values using three masks.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn select4<S: Simd>(
    simd: S,
    o0: M32s<S>,
    o1: M32s<S>,
    o2: M32s<S>,
    d0: f32,
    d1: f32,
    d2: f32,
    d3: f32,
) -> F32s<S> {
    o0.select(
        splat(simd, d0),
        o1.select(splat(simd, d1), select_splat(simd, o2, d2, d3)),
    )
}

/// Selects between two double-double values using a mask.
#[inline(always)]
pub fn select_doubled<S: Simd>(_simd: S, mask: M32s<S>, l: DD<S>, r: DD<S>) -> DD<S> {
    DD::new(mask.select(l.0, r.0), mask.select(l.1, r.1))
}

/// `2^q` for an integer vector `q`.
#[inline(always)]
pub fn pow2i<S: Simd>(simd: S, q: I32s<S>) -> F32s<S> {
    from_bits(simd, int_to_bits(simd, (splat_int(simd, 0x7f) + q) << 23))
}

/// `d * 2^e`, split into two half-range multiplications.
#[inline(always)]
pub fn ldexp2k<S: Simd>(simd: S, d: F32s<S>, e: I32s<S>) -> F32s<S> {
    let e1 = e >> 1;
    d * pow2i(simd, e1) * pow2i(simd, e - e1)
}

/// `d * 2^q` by adding directly to the exponent field.
#[inline(always)]
pub fn ldexp3k<S: Simd>(simd: S, d: F32s<S>, q: I32s<S>) -> F32s<S> {
    from_bits(simd, to_bits(simd, d) + int_to_bits(simd, q << 23))
}

/// The exponent of `d`, without subnormal handling.
#[inline(always)]
pub fn ilogb2k<S: Simd>(simd: S, d: F32s<S>) -> I32s<S> {
    bits_to_int(simd, (to_bits(simd, d) >> 23) & splat_bits(simd, 0xff)) - splat_int(simd, 0x7f)
}

/// The exponent of `d`, scaling subnormals into range first.
#[inline(always)]
pub fn ilogbk<S: Simd>(simd: S, mut d: F32s<S>) -> I32s<S> {
    let o = d.simd_lt(splat(simd, 5.421_010_862_427_522_e-20));
    d = o.select(splat(simd, 1.844_674_407_370_955_2_e19) * d, d);
    let q = bits_to_int(simd, (to_bits(simd, d) >> 23) & splat_bits(simd, 0xff));
    q - o.select(splat_int(simd, 64 + 0x7f), splat_int(simd, 0x7f))
}

/// `x * 2^q` over the full exponent range, including subnormals.
#[inline(always)]
pub fn ldexpk<S: Simd>(simd: S, x: F32s<S>, q: I32s<S>) -> F32s<S> {
    let mut m = q >> 31;
    m = (((m + q) >> 6) - m) << 4;
    let q = q - (m << 2);
    m = splat_int(simd, 0x7f) + m;
    m = (!splat_int(simd, 0).simd_gt(m)).select(splat_int(simd, -1), splat_int(simd, 0)) & m;
    m = m
        .simd_gt(splat_int(simd, 0xff))
        .select(splat_int(simd, 0xff), m);
    let y = from_bits(simd, int_to_bits(simd, m) << 23);
    x * y * y * y * y * pow2i(simd, q)
}

/// Multiply by an integral power of two.
#[inline(always)]
pub fn ldexp<S: Simd>(simd: S, x: F32s<S>, q: I32s<S>) -> F32s<S> {
    ldexpk(simd, x, q)
}

/// The exponent of `d` as an integer vector.
#[inline(always)]
pub fn ilogb<S: Simd>(simd: S, d: F32s<S>) -> I32s<S> {
    let mut e = ilogbk(simd, d.abs());
    e = d
        .simd_eq(splat(simd, 0.))
        .select(splat_int(simd, SLEEF_FP_ILOGB0), e);
    e = d.is_nan().select(splat_int(simd, SLEEF_FP_ILOGBNAN), e);
    d.is_infinite().select(splat_int(simd, i32::MAX), e)
}

/// The value of `m` carrying the sign of an infinite `d`, and zero where `d` is finite.
#[inline(always)]
pub fn visinf2<S: Simd>(simd: S, d: F32s<S>, m: F32s<S>) -> F32s<S> {
    from_bits(
        simd,
        d.is_infinite()
            .select(splat_bits(simd, u32::MAX), splat_bits(simd, 0))
            & ((to_bits(simd, d) & sign_bit(simd, d)) | to_bits(simd, m)),
    )
}

/// Maximum, treating a NaN in `y` as `x`.
#[inline(always)]
pub fn fmax<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let _ = simd;
    y.is_nan().select(x, x.max(y))
}

/// Minimum, treating a NaN in `y` as `x`.
#[inline(always)]
pub fn fmin<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let _ = simd;
    y.is_nan().select(x, x.min(y))
}

/// Positive difference.
#[inline(always)]
pub fn fdim<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let ret = x - y;
    (ret.simd_lt(splat(simd, 0.)) | x.simd_eq(y)).select(splat(simd, 0.), ret)
}

/// `v + d`, matching upstream's `F32x + Doubled<F32x>`.
#[inline(always)]
pub fn v_add_dd<S: Simd>(_simd: S, v: F32s<S>, d: DD<S>) -> DD<S> {
    let t = add_as_doubled::<S>(v, d.0);
    DD::new(t.0, t.1 + d.1)
}

/// Absolute value.
#[inline(always)]
pub fn fabs<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let _ = simd;
    x.abs()
}

/// The next representable value after `x` in the direction of `y`.
#[inline(always)]
pub fn nextafter<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let x = x
        .simd_eq(splat(simd, 0.))
        .select(mul_sign(simd, splat(simd, 0.), y), x);
    let xi2 = bits_to_int(simd, to_bits(simd, x));
    let c = x.is_sign_negative() ^ y.simd_ge(x);

    let xi2 = c.select(-(xi2 ^ splat_int(simd, i32::MIN)), xi2);
    let xi2 = x.simd_ne(y).select(xi2 - splat_int(simd, 1), xi2);
    let xi2 = c.select(-(xi2 ^ splat_int(simd, i32::MIN)), xi2);

    let mut ret = from_bits(simd, int_to_bits(simd, xi2));

    ret = (ret.simd_eq(splat(simd, 0.)) & x.simd_ne(splat(simd, 0.)))
        .select(mul_sign(simd, splat(simd, 0.), x), ret);
    ret = (x.simd_eq(splat(simd, 0.)) & y.simd_eq(splat(simd, 0.))).select(y, ret);
    (x.is_nan() | y.is_nan()).select(splat(simd, f32::NAN), ret)
}

/// `sqrt(x)` as a double-double.
#[inline(always)]
pub fn sqrt_as_doubled<S: Simd>(simd: S, x: F32s<S>) -> DD<S> {
    let t = x.sqrt();
    scale(
        (DD::new(x, splat(simd, 0.0)) + mul_as_doubled(t, t)) * recip_as_doubled(t),
        splat(simd, 0.5),
    )
}

/// Removes the integer part of `4 * x`, returning the fraction and the integer part.
#[inline(always)]
pub fn rempisub<S: Simd>(simd: S, x: F32s<S>) -> (F32s<S>, I32s<S>) {
    let c = mul_sign(simd, splat(simd, F1_23), x);
    let rint4x = (splat(simd, 4.) * x)
        .abs()
        .simd_gt(splat(simd, F1_23))
        .select(
            splat(simd, 4.) * x,
            or_sign(simd, mla(simd, splat(simd, 4.), x, c) - c, x),
        );
    let rintx = x
        .abs()
        .simd_gt(splat(simd, F1_23))
        .select(x, or_sign(simd, (x + c) - c, x));

    let fr = mla(simd, splat(simd, -0.25), rint4x, x);
    let vi = trunci(simd, mla(simd, splat(simd, -4.), rintx, rint4x));
    (fr, vi)
}

/// Reduces `a` modulo `pi/2` using the Payne–Hanek tables.
#[inline(always)]
pub fn rempi<S: Simd>(simd: S, mut a: F32s<S>) -> (DD<S>, I32s<S>) {
    let mut ex = ilogb2k(simd, a);
    ex -= splat_int(simd, 25);
    let mut q = ex
        .simd_gt(splat_int(simd, 90 - 25))
        .select(splat_int(simd, -1), splat_int(simd, 0))
        & splat_int(simd, -64);
    a = ldexp3k(simd, a, q);
    ex = !(ex >> 31) & ex;
    ex <<= 2;
    let mut x = mul_as_doubled(a, from_slice_offset(simd, REMPITABSP.as_slice(), ex));
    let (did, dii) = rempisub(simd, x.0);
    q = dii;
    x.0 = did;
    x = normalize(x);
    let mut y = mul_as_doubled(a, from_slice_offset(simd, &REMPITABSP[1..], ex));
    x += y;
    let (did, dii) = rempisub(simd, x.0);
    q += dii;
    x.0 = did;
    x = normalize(x);
    y = DD::new(
        from_slice_offset(simd, &REMPITABSP[2..], ex),
        from_slice_offset(simd, &REMPITABSP[3..], ex),
    );
    y = mul_v(y, a);
    x += y;
    x = normalize(x);
    x = mul_v(x, splat(simd, D_PI_F.0 * 2.)) + mul_v(x, splat(simd, D_PI_F.1 * 2.));
    let o = a.abs().simd_lt(splat(simd, 0.7));
    x = DD::new(
        o.select(a, x.0),
        from_bits(
            simd,
            o.select(splat_bits(simd, 0), splat_bits(simd, u32::MAX)) & to_bits(simd, x.1),
        ),
    );
    (x, q)
}

/// Gathers one lane per element of `vi` from `table`.
#[inline(always)]
pub fn from_slice_offset<S: Simd>(simd: S, table: &[f32], vi: I32s<S>) -> F32s<S> {
    let idx = vi.to_array();
    let mut values = F32s::<S>::splat(simd, 0.0).to_array();
    for (slot, &i) in values.as_mut().iter_mut().zip(idx.as_ref().iter()) {
        // Upstream reads through an unchecked gather, so an infinite argument, whose exponent
        // overflows the Payne-Hanek index, reads past the end of the table there. Only the
        // in-range part is meaningful and the caller masks the result for non-finite arguments,
        // so clamping keeps the in-domain values untouched.
        *slot = table[(i as usize).min(table.len() - 1)];
    }
    F32s::<S>::load_array(simd, values)
}

/// `x` rounded to an integer towards zero, for a non-negative `x` that is already integral.
#[inline(always)]
fn trunc_positive<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let fr = x - trunci(simd, x).to_float::<F32s<S>>();
    x.abs().simd_ge(splat(simd, F1_23)).select(x, x - fr)
}

/// `x` with its last mantissa bit cleared, used by `toward0`.
#[inline(always)]
fn toward0<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let t = from_bits(
        simd,
        to_bits(simd, x) + int_to_bits(simd, splat_int(simd, -1)),
    );
    x.simd_eq(splat(simd, 0.)).select(splat(simd, 0.), t)
}

/// Floating-point remainder, truncated towards zero.
///
/// The error bound of the returned value is `0.5 ULP`.
#[inline(always)]
pub fn fmod<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let n = x.abs();
    let d = y.abs();
    let s = splat(simd, 1.);
    let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE));
    let n = o.select(n * splat(simd, F1_25), n);
    let d = o.select(d * splat(simd, F1_25), d);
    let s = o.select(s * splat(simd, 1. / F1_25), s);
    let rde = toward0(simd, splat(simd, 1.) / d);
    let mut r: DD<S> = DD::new(n, splat(simd, 0.));

    for _ in 0..8 {
        // ceil(log2(FLT_MAX) / 22) + 1 iterations are sufficient.
        let r0: F32s<S> = r.0;
        let mut q = trunc_positive(simd, toward0(simd, r0) * rde);
        let big = (splat(simd, 3.) * d).simd_gt(r0) & r0.simd_ge(d);
        q = big.select(splat(simd, 2.), q);
        let mid = (d + d).simd_gt(r0) & r0.simd_ge(d);
        q = mid.select(splat(simd, 1.), q);
        r = normalize(r + mul_v(-mul_as_doubled(splat(simd, 1.), q), d));
        if r.0.simd_lt(d).all_true() {
            break;
        }
    }

    let mut ret = r.0 * s;
    ret = r.0.simd_eq(d).select(splat(simd, 0.), ret);

    ret = mul_sign(simd, ret, x);

    ret = n.simd_lt(d).select(x, ret);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NAN), ret)
}

/// IEEE remainder, rounded to nearest even quotient.
///
/// The error bound of the returned value is `0.5 ULP`.
#[inline(always)]
pub fn remainder<S: Simd>(simd: S, x: F32s<S>, y: F32s<S>) -> F32s<S> {
    let mut n = x.abs();
    let mut d = y.abs();
    let mut s = splat(simd, 1.);
    let o = d.simd_lt(splat(simd, f32::MIN_POSITIVE * 2.));
    n = o.select(n * splat(simd, F1_25), n);
    d = o.select(d * splat(simd, F1_25), d);
    s = o.select(s * splat(simd, 1. / F1_25), s);
    let rd = splat(simd, 1.) / d;
    let mut r: DD<S> = DD::new(n, splat(simd, 0.));
    let mut qisodd = S::mask32s::splat(simd, false);

    for _ in 0..8 {
        // ceil(log2(FLT_MAX) / 22) + 1 iterations are sufficient.
        let mut q = rint(simd, r.0 * rd);
        q =
            r.0.abs()
                .simd_lt(d * splat(simd, 1.5))
                .select(mul_sign(simd, splat(simd, 1.), r.0), q);
        q = (r.0.abs().simd_lt(d * splat(simd, 0.5))
            | (!qisodd & r.0.abs().simd_eq(d * splat(simd, 0.5))))
        .select(splat(simd, 0.), q);
        if q.simd_eq(splat(simd, 0.)).all_true() {
            break;
        }
        q = (q * (-d))
            .is_infinite()
            .select(q - mul_sign(simd, splat(simd, 1.), r.0), q);
        qisodd ^= is_odd(simd, q);
        r = normalize(r + mul_v(mul_as_doubled(q, splat(simd, -1.)), d));
    }

    let mut ret = r.0 * s;
    ret = mul_sign(simd, ret, x);
    ret = y
        .is_infinite()
        .select(x.is_infinite().select(splat(simd, f32::NAN), x), ret);
    d.simd_eq(splat(simd, 0.))
        .select(splat(simd, f32::NAN), ret)
}

/// Splits `x` into its fractional and integral parts.
///
/// The returned pair is `(fractional, integral)`.
#[inline(always)]
pub fn modf<S: Simd>(simd: S, x: F32s<S>) -> (F32s<S>, F32s<S>) {
    let fr = x - trunci(simd, x).to_float::<F32s<S>>();
    let fr = x
        .abs()
        .simd_gt(splat(simd, F1_23))
        .select(splat(simd, 0.), fr);

    (copy_sign(simd, fr, x), copy_sign(simd, x - fr, x))
}

/// Fractional component of a floating-point number, as returned by `frexp`.
#[inline(always)]
pub fn frfrexp<S: Simd>(simd: S, x: F32s<S>) -> F32s<S> {
    let x = x
        .abs()
        .simd_lt(splat(simd, f32::MIN_POSITIVE))
        .select(x * splat(simd, F1_32), x);

    let mut xm = to_bits(simd, x);
    xm &= splat_bits(simd, 0x_807f_ffff);
    xm |= splat_bits(simd, 0x_3f00_0000);

    let ret = from_bits(simd, xm);

    let ret = x
        .is_infinite()
        .select(mul_sign(simd, splat(simd, f32::INFINITY), x), ret);
    x.simd_eq(splat(simd, 0.)).select(x, ret)
}

/// Exponent of a floating-point number, as returned by `frexp`.
///
/// Upstream's body is commented out and the function returns zero for every input; this port keeps
/// that stub rather than replacing it with an unported implementation. No ported kernel calls it.
#[inline(always)]
pub fn expfrexp<S: Simd>(simd: S, _x: F32s<S>) -> I32s<S> {
    splat_int(simd, 0)
}

/// Fused multiply-add: `x * y + z` with a single rounding.
///
/// The error bound of the returned value is `max(0.500_01 ULP, f32::MIN_POSITIVE)`. Upstream's
/// fallback returns infinity with the correct sign once the magnitude of the exact result exceeds
/// `1e+33`; this port evaluates the exact product-plus-add with a single rounding instead, which
/// stays finite wherever the exact result is representable.
#[inline(always)]
pub fn fma<S: Simd>(_simd: S, x: F32s<S>, y: F32s<S>, z: F32s<S>) -> F32s<S> {
    x.mul_add_precise(y, z)
}
