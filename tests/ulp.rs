//! ULP-accuracy tests for the `f64` kernels, against MPFR.
//!
//! The bounds asserted here are the ones `sleef-rs` rev `3671d46` asserts for the same functions.
//! Every test drives the public slice entry points, i.e. the path production uses (runtime
//! dispatch, lane loop, padded tail), over lengths below, at and above the SIMD lane width.

#![cfg(feature = "simd")]

mod common;

use core::mem::MaybeUninit;

use common::{assert_close, check_binary, check_pair, check_unary, read, PREC};
use elem_simd_math::{f64 as ops, ElementwiseMath, LengthMismatch};
use rug::ops::Pow;
use rug::Float;

/// `sqrt(f64::MAX)`, the upper bound upstream uses for the inverse hyperbolic ranges.
const SQRT_DBL_MAX: f64 = 1.340_780_792_994_259_7e+154;

/// Working precision for the `*pi` oracles.
///
/// `sin(x * pi)` is only as accurate as the fractional part of the product, and near a zero the
/// product loses one bit per bit of `|x|`: at the default 128 bits a half-integer `x` around `2^21`
/// leaves the answer about `x * 2^-128` away from the true zero, which is far outside the asserted
/// sub-ULP bound. The inputs themselves are exact in every precision, so raising the working
/// precision is enough.
const PI_PREC: u32 = 512;

fn sinpi_oracle(x: Float) -> Float {
    (Float::with_val(PI_PREC, &x) * Float::with_val(PI_PREC, rug::float::Constant::Pi)).sin()
}

fn cospi_oracle(x: Float) -> Float {
    (Float::with_val(PI_PREC, &x) * Float::with_val(PI_PREC, rug::float::Constant::Pi)).cos()
}

// ---------------------------------------------------------------- unary ----------

#[test]
fn exp_ulp() {
    check_unary("exp", ops::exp, Float::exp, -1000.0, 710.0, 1.0);
}

#[test]
fn exp2_ulp() {
    check_unary("exp2", ops::exp2, Float::exp2, -2000.0, 1024.0, 1.0);
}

#[test]
fn exp10_ulp() {
    check_unary("exp10", ops::exp10, Float::exp10, -350.0, 308.26, 1.09);
}

#[test]
fn expm1_ulp() {
    check_unary("expm1", ops::expm1, Float::exp_m1, -37.0, 710.0, 1.0);
}

#[test]
fn log_ulp() {
    check_unary("log", ops::log, Float::ln, 0.0, f64::MAX, 1.0);
}

#[test]
fn log2_ulp() {
    check_unary("log2", ops::log2, Float::log2, 0.0, f64::MAX, 1.0);
}

#[test]
fn log10_ulp() {
    check_unary("log10", ops::log10, Float::log10, 0.0, f64::MAX, 1.0);
}

#[test]
fn log1p_ulp() {
    check_unary("log1p", ops::log1p, Float::ln_1p, -1.0, 1e307, 1.0);
}

#[test]
fn sin_ulp() {
    check_unary("sin", ops::sin, Float::sin, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn cos_ulp() {
    check_unary("cos", ops::cos, Float::cos, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn tan_ulp() {
    check_unary("tan", ops::tan, Float::tan, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn sin_u35_ulp() {
    check_unary("sin_u35", ops::sin_u35, Float::sin, f64::MIN, f64::MAX, 3.5);
}

#[test]
fn cos_u35_ulp() {
    check_unary("cos_u35", ops::cos_u35, Float::cos, f64::MIN, f64::MAX, 3.5);
}

#[test]
fn tan_u35_ulp() {
    check_unary("tan_u35", ops::tan_u35, Float::tan, f64::MIN, f64::MAX, 3.5);
}

#[test]
fn log_u35_ulp() {
    check_unary("log_u35", ops::log_u35, Float::ln, 0.0, f64::MAX, 3.5);
}

#[test]
fn tanh_u35_ulp() {
    check_unary("tanh_u35", ops::tanh_u35, Float::tanh, -19.0, 19.0, 3.5);
}

#[test]
fn asin_ulp() {
    check_unary("asin", ops::asin, Float::asin, -1.0, 1.0, 1.0);
}

#[test]
fn acos_ulp() {
    check_unary("acos", ops::acos, Float::acos, -1.0, 1.0, 1.0);
}

#[test]
fn atan_ulp() {
    check_unary("atan", ops::atan, Float::atan, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn sinh_ulp() {
    check_unary("sinh", ops::sinh, Float::sinh, -709.0, 709.0, 1.0);
}

#[test]
fn cosh_ulp() {
    check_unary("cosh", ops::cosh, Float::cosh, -709.0, 709.0, 1.0);
}

#[test]
fn tanh_ulp() {
    check_unary("tanh", ops::tanh, Float::tanh, -19.0, 19.0, 1.0);
}

#[test]
fn asinh_ulp() {
    check_unary(
        "asinh",
        ops::asinh,
        Float::asinh,
        -SQRT_DBL_MAX,
        SQRT_DBL_MAX,
        1.0,
    );
}

#[test]
fn acosh_ulp() {
    check_unary("acosh", ops::acosh, Float::acosh, 1.0, SQRT_DBL_MAX, 1.0);
}

#[test]
fn atanh_ulp() {
    check_unary("atanh", ops::atanh, Float::atanh, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn cbrt_ulp() {
    check_unary("cbrt", ops::cbrt, Float::cbrt, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn erf_ulp() {
    check_unary("erf", ops::erf, Float::erf, f64::MIN, f64::MAX, 0.75);
}

#[test]
fn erfc_ulp() {
    check_unary("erfc", ops::erfc, Float::erfc, f64::MIN, 26.2, 1.5);
    check_unary("erfc", ops::erfc, Float::erfc, 26.2, f64::MAX, 2.5);
}

#[test]
fn tgamma_ulp() {
    check_unary("tgamma", ops::tgamma, Float::gamma, f64::MIN, f64::MAX, 1.0);
}

#[test]
fn lgamma_ulp() {
    check_unary("lgamma", ops::lgamma, Float::ln_gamma, 0.0, 2e305, 1.0);
}

#[test]
fn sinpi_ulp() {
    let max = 1e9 / 4.;
    check_unary("sinpi", ops::sinpi, sinpi_oracle, -max, max, 0.506);
}

#[test]
fn cospi_ulp() {
    let max = 1e9 / 4.;
    check_unary("cospi", ops::cospi, cospi_oracle, -max, max, 0.506);
}

#[test]
fn sqrt_ulp() {
    check_unary("sqrt", ops::sqrt, Float::sqrt, 0.0, f64::MAX, 0.50001);
}

// --------------------------------------------------------------- binary ----------

fn atan2_oracle(y: Float, x: &Float) -> Float {
    y.atan2(x)
}

fn pow_oracle(x: Float, y: &Float) -> Float {
    x.pow(y)
}

#[test]
fn atan2_ulp() {
    check_binary(
        "atan2",
        ops::atan2,
        atan2_oracle,
        f64::MIN,
        f64::MAX,
        f64::MIN,
        f64::MAX,
        1.0,
    );
}

#[test]
fn pow_ulp() {
    check_binary(
        "pow",
        ops::pow,
        pow_oracle,
        f64::MIN,
        f64::MAX,
        f64::MIN,
        f64::MAX,
        1.0,
    );
}

#[test]
fn hypot_ulp() {
    check_binary(
        "hypot",
        ops::hypot,
        |x: Float, y: &Float| x.hypot(y),
        f64::MIN,
        f64::MAX,
        f64::MIN,
        f64::MAX,
        0.5,
    );
}

// ----------------------------------------------------------------- pairs ----------

#[test]
fn sincos_ulp() {
    check_pair(
        "sincos",
        ops::sincos,
        |x: Float| {
            let p = x.prec();
            x.sin_cos(Float::new(p))
        },
        f64::MIN,
        f64::MAX,
        1.0,
    );
}

#[test]
fn sincospi_ulp() {
    let max = 1e9 / 4.;
    check_pair(
        "sincospi",
        ops::sincospi,
        |x: Float| {
            (Float::with_val(PI_PREC, &x) * Float::with_val(PI_PREC, rug::float::Constant::Pi))
                .sin_cos(Float::new(PI_PREC))
        },
        -max,
        max,
        0.506,
    );
}

// --------------------------------------------------------- interface checks -------

#[test]
fn trait_methods_dispatch() {
    let src = [0.25f64, 1.5, -2.0, 3.0];
    let mut out = vec![MaybeUninit::uninit(); src.len()];
    <f64 as ElementwiseMath>::exp(&src, &mut out).unwrap();
    assert!((read(out)[1] - 1.5f64.exp()).abs() < 1e-12);

    let mut out = vec![MaybeUninit::uninit(); src.len()];
    <f64 as ElementwiseMath>::pow(&src, &src, &mut out).unwrap();
    assert!((read(out)[3] - 27.0).abs() < 1e-12);

    let mut s = vec![MaybeUninit::uninit(); src.len()];
    let mut c = vec![MaybeUninit::uninit(); src.len()];
    <f64 as ElementwiseMath>::sincos(&src, &mut s, &mut c).unwrap();
    let (s, c) = (read(s), read(c));
    assert!((s[1] - 1.5f64.sin()).abs() < 1e-12);
    assert!((c[1] - 1.5f64.cos()).abs() < 1e-12);
}

#[test]
fn length_mismatch_is_reported() {
    let src = [1.0f64, 2.0];
    let mut out = vec![MaybeUninit::uninit(); 3];
    assert_eq!(
        ops::exp(&src, &mut out),
        Err(LengthMismatch { src: 2, dst: 3 })
    );
    assert_eq!(
        ops::pow(&src, &src[..1], &mut out[..2]),
        Err(LengthMismatch { src: 1, dst: 2 })
    );
}

/// Every length from 0 to `4 * lanes + 3` must be handled, so the vector body, the tail and the
/// empty case are all reached.
#[test]
fn all_lengths_are_covered() {
    for len in 0..=(4 * 8 + 3) {
        let src: Vec<f64> = (0..len).map(|i| i as f64 * 0.125 - 1.0).collect();
        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        ops::sin(&src, &mut out).unwrap();
        let got = read(out);
        for (i, &x) in src.iter().enumerate() {
            assert!((got[i] - x.sin()).abs() < 1e-12, "len {len}, lane {i}");
        }
    }
}

/// Special values must match the oracle exactly: NaN stays NaN, signed zero keeps its sign, and
/// overflow saturates the way the platform does.
#[test]
fn special_values() {
    let cases: [f64; 12] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        1.0e308,
        -1.0e308,
    ];

    let mut out = vec![MaybeUninit::uninit(); cases.len()];
    ops::exp(&cases, &mut out).unwrap();
    let got = read(out);
    for (i, &x) in cases.iter().enumerate() {
        let want = Float::with_val(PREC, x).exp();
        assert_close("exp special", 0, i, x, got[i], &want, 1.0);
    }

    let mut out = vec![MaybeUninit::uninit(); cases.len()];
    ops::log(&cases, &mut out).unwrap();
    let got = read(out);
    for (i, &x) in cases.iter().enumerate() {
        let want = Float::with_val(PREC, x).ln();
        if want.is_nan() {
            assert!(got[i].is_nan(), "log({x:e}) = {}", got[i]);
        } else {
            assert_close("log special", 0, i, x, got[i], &want, 3.5);
        }
    }

    // Signed zero must survive where the C library preserves it.
    let zeros = [-0.0f64, 0.0];
    let mut out = vec![MaybeUninit::uninit(); zeros.len()];
    ops::sin(&zeros, &mut out).unwrap();
    let got = read(out);
    assert!(got[0].is_sign_negative(), "sin(-0.0) lost its sign");
    assert!(got[1].is_sign_positive(), "sin(+0.0) lost its sign");

    let mut out = vec![MaybeUninit::uninit(); zeros.len()];
    ops::expm1(&zeros, &mut out).unwrap();
    let got = read(out);
    assert!(got[0].is_sign_negative(), "expm1(-0.0) lost its sign");
    assert!(got[1].is_sign_positive(), "expm1(+0.0) lost its sign");
}
