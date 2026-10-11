//! ULP-accuracy tests for the `f32` kernels, against MPFR.
//!
//! The bounds asserted here are the ones `sleef-rs` rev `3671d46` asserts for the same functions.
//! They differ from the `f64` suite's: most are still 1.0 ULP, but `tanh`/`atanh` are 1.0001,
//! `erfc` is 1.5, and the `u35` tier is 3.5.

#![cfg(feature = "simd")]

mod common;

use core::mem::MaybeUninit;

use common::{assert_close, check_binary, check_pair, check_unary, read, PREC};
use elem_simd_math::{f32 as ops, ElementwiseMath, LengthMismatch};
use rug::ops::Pow;
use rug::Float;

/// `sqrt(f32::MAX)`, the upper bound upstream uses for the inverse hyperbolic ranges.
const SQRT_FLT_MAX: f32 = 1.844_674_407_370_955_2e19;

/// `Float::with_val(p, pi) * x`, the oracle the `*pi` functions are checked against.
fn pi_scaled(x: Float, f: fn(Float) -> Float) -> Float {
    let p = x.prec();
    f(x * Float::with_val(p, rug::float::Constant::Pi))
}

fn sinpi_oracle(x: Float) -> Float {
    pi_scaled(x, Float::sin)
}

fn cospi_oracle(x: Float) -> Float {
    pi_scaled(x, Float::cos)
}

fn sincospi_oracle(x: Float) -> (Float, Float) {
    let p = x.prec();
    (x * Float::with_val(p, rug::float::Constant::Pi)).sin_cos(Float::new(p))
}

fn atan2_oracle(y: Float, x: &Float) -> Float {
    y.atan2(x)
}

fn pow_oracle(x: Float, y: &Float) -> Float {
    x.pow(y)
}

const PI_RANGE: f64 = 1e7 / 4.;

// ------------------------------------------------------------------ u10 ----------

#[test]
fn exp_ulp() {
    check_unary("exp", ops::exp, Float::exp, -104.0, 100.0, 1.0);
}

#[test]
fn exp2_ulp() {
    check_unary("exp2", ops::exp2, Float::exp2, -150.0, 128.0, 1.0);
}

#[test]
fn exp10_ulp() {
    check_unary("exp10", ops::exp10, Float::exp10, -50.0, 38.54, 1.0);
}

#[test]
fn expm1_ulp() {
    check_unary("expm1", ops::expm1, Float::exp_m1, -16.64, 88.73, 1.0);
}

#[test]
fn log_ulp() {
    check_unary("log", ops::log, Float::ln, 0.0, f32::MAX as f64, 1.0);
}

#[test]
fn log2_ulp() {
    check_unary("log2", ops::log2, Float::log2, 0.0, f32::MAX as f64, 1.0);
}

#[test]
fn log10_ulp() {
    check_unary("log10", ops::log10, Float::log10, 0.0, f32::MAX as f64, 1.0);
}

#[test]
fn log1p_ulp() {
    check_unary("log1p", ops::log1p, Float::ln_1p, -1.0, 1e38, 1.0);
}

#[test]
fn sin_ulp() {
    check_unary(
        "sin",
        ops::sin,
        Float::sin,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn cos_ulp() {
    check_unary(
        "cos",
        ops::cos,
        Float::cos,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn tan_ulp() {
    check_unary(
        "tan",
        ops::tan,
        Float::tan,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
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
    check_unary(
        "atan",
        ops::atan,
        Float::atan,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn sinh_ulp() {
    check_unary("sinh", ops::sinh, Float::sinh, -88.5, 88.5, 1.0);
}

#[test]
fn cosh_ulp() {
    check_unary("cosh", ops::cosh, Float::cosh, -88.5, 88.5, 1.0);
}

#[test]
fn tanh_ulp() {
    check_unary("tanh", ops::tanh, Float::tanh, -8.7, 8.7, 1.0001);
}

#[test]
fn asinh_ulp() {
    check_unary(
        "asinh",
        ops::asinh,
        Float::asinh,
        -SQRT_FLT_MAX as f64,
        SQRT_FLT_MAX as f64,
        1.0,
    );
}

#[test]
fn acosh_ulp() {
    check_unary(
        "acosh",
        ops::acosh,
        Float::acosh,
        1.0,
        SQRT_FLT_MAX as f64,
        1.0,
    );
}

#[test]
fn atanh_ulp() {
    check_unary(
        "atanh",
        ops::atanh,
        Float::atanh,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0001,
    );
}

#[test]
fn cbrt_ulp() {
    check_unary(
        "cbrt",
        ops::cbrt,
        Float::cbrt,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn tgamma_ulp() {
    check_unary(
        "tgamma",
        ops::tgamma,
        Float::gamma,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn lgamma_ulp() {
    check_unary("lgamma", ops::lgamma, Float::ln_gamma, 0.0, 4e36, 1.0);
}

#[test]
fn erf_ulp() {
    check_unary(
        "erf",
        ops::erf,
        Float::erf,
        f32::MIN as f64,
        f32::MAX as f64,
        0.75,
    );
}

#[test]
fn u35_sin_ulp() {
    check_unary(
        "sin_u35",
        ops::sin_u35,
        Float::sin,
        f32::MIN as f64,
        f32::MAX as f64,
        3.5,
    );
}

#[test]
fn u35_cos_ulp() {
    check_unary(
        "cos_u35",
        ops::cos_u35,
        Float::cos,
        f32::MIN as f64,
        f32::MAX as f64,
        3.5,
    );
}

#[test]
fn u35_tan_ulp() {
    check_unary(
        "tan_u35",
        ops::tan_u35,
        Float::tan,
        f32::MIN as f64,
        f32::MAX as f64,
        3.5,
    );
}

#[test]
fn u35_log_ulp() {
    check_unary(
        "log_u35",
        ops::log_u35,
        Float::ln,
        0.0,
        f32::MAX as f64,
        3.5,
    );
}

#[test]
fn u35_log2_ulp() {
    check_unary(
        "log2_u35",
        ops::log2_u35,
        Float::log2,
        0.0,
        f32::MAX as f64,
        3.5,
    );
}

#[test]
fn u35_exp2_ulp() {
    check_unary("exp2_u35", ops::exp2_u35, Float::exp2, -150.0, 128.0, 3.5);
}

#[test]
fn u35_exp10_ulp() {
    check_unary(
        "exp10_u35",
        ops::exp10_u35,
        Float::exp10,
        -38.54,
        38.54,
        3.5,
    );
}

#[test]
fn u35_tanh_ulp() {
    check_unary("tanh_u35", ops::tanh_u35, Float::tanh, -8.7, 8.7, 3.5);
}

#[test]
fn u35_cbrt_ulp() {
    check_unary(
        "cbrt_u35",
        ops::cbrt_u35,
        Float::cbrt,
        f32::MIN as f64,
        f32::MAX as f64,
        3.5,
    );
}

// ---------------------------------------------------------------- binary ----------

#[test]
fn pow_ulp() {
    check_binary(
        "pow",
        ops::pow,
        pow_oracle,
        f32::MIN as f64,
        f32::MAX as f64,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn atan2_ulp() {
    check_binary(
        "atan2",
        ops::atan2,
        atan2_oracle,
        f32::MIN as f64,
        f32::MAX as f64,
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn hypot_ulp() {
    check_binary(
        "hypot",
        ops::hypot,
        |x: Float, y: &Float| x.hypot(y),
        f32::MIN as f64,
        f32::MAX as f64,
        f32::MIN as f64,
        f32::MAX as f64,
        0.5,
    );
}

#[test]
fn sincos_ulp() {
    check_pair(
        "sincos",
        ops::sincos,
        |x: Float| {
            let p = x.prec();
            x.sin_cos(Float::new(p))
        },
        f32::MIN as f64,
        f32::MAX as f64,
        1.0,
    );
}

#[test]
fn u35_sincos_ulp() {
    check_pair(
        "sincos_u35",
        ops::sincos_u35,
        |x: Float| {
            let p = x.prec();
            x.sin_cos(Float::new(p))
        },
        f32::MIN as f64,
        f32::MAX as f64,
        3.5,
    );
}

#[test]
fn sincospi_ulp() {
    check_pair(
        "sincospi",
        ops::sincospi,
        sincospi_oracle,
        -PI_RANGE,
        PI_RANGE,
        0.506,
    );
}

// ------------------------------------------------------------- single-range -------

#[test]
fn sinpi_ulp() {
    check_unary(
        "sinpi",
        ops::sinpi,
        sinpi_oracle,
        -PI_RANGE,
        PI_RANGE,
        0.506,
    );
}

#[test]
fn cospi_ulp() {
    check_unary(
        "cospi",
        ops::cospi,
        cospi_oracle,
        -PI_RANGE,
        PI_RANGE,
        0.506,
    );
}

#[test]
fn sqrt_ulp() {
    check_unary("sqrt", ops::sqrt, Float::sqrt, 0.0, f32::MAX as f64, 0.5);
}

#[test]
fn erfc_ulp() {
    check_unary(
        "erfc",
        ops::erfc,
        Float::erfc,
        f32::MIN as f64,
        f32::MAX as f64,
        1.5,
    );
}

// --------------------------------------------------------- interface checks -------

#[test]
fn trait_methods_dispatch() {
    let src = [0.25f32, 1.5, -2.0, 3.0];
    let mut out = vec![MaybeUninit::uninit(); src.len()];
    <f32 as ElementwiseMath>::exp(&src, &mut out).unwrap();
    assert!((read(out)[1] - 1.5f32.exp()).abs() < 1e-5);

    let mut out = vec![MaybeUninit::uninit(); src.len()];
    <f32 as ElementwiseMath>::pow(&src, &src, &mut out).unwrap();
    assert!((read(out)[3] - 27.0).abs() < 1e-4);

    let mut s = vec![MaybeUninit::uninit(); src.len()];
    let mut c = vec![MaybeUninit::uninit(); src.len()];
    <f32 as ElementwiseMath>::sincos(&src, &mut s, &mut c).unwrap();
    let (s, c) = (read(s), read(c));
    assert!((s[1] - 1.5f32.sin()).abs() < 1e-6);
    assert!((c[1] - 1.5f32.cos()).abs() < 1e-6);
}

#[test]
fn length_mismatch_is_reported() {
    let src = [1.0f32, 2.0];
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

/// Every length from 0 to `16 * lanes + 3` must be handled, so the vector body, the tail and the
/// empty case are all reached.
#[test]
fn all_lengths_are_covered() {
    for len in 0..=(16 * 16 + 3) {
        let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.125 - 1.0).collect();
        let mut out: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); len];
        ops::sin(&src, &mut out).unwrap();
        let got = read(out);
        for (i, &x) in src.iter().enumerate() {
            assert!((got[i] - x.sin()).abs() < 1e-5, "len {len}, lane {i}");
        }
    }
}

/// Special values must match the oracle: NaN stays NaN, signed zero keeps its sign, and overflow
/// saturates the way the platform does.
#[test]
fn special_values() {
    let cases: [f32; 12] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        1.0e38,
        -1.0e38,
    ];

    let mut out = vec![MaybeUninit::uninit(); cases.len()];
    ops::exp(&cases, &mut out).unwrap();
    let got = read(out);
    for (i, &x) in cases.iter().enumerate() {
        let want = Float::with_val(PREC, x).exp();
        if want.is_nan() {
            assert!(got[i].is_nan(), "exp({x:e}) = {}", got[i]);
        } else {
            let ulp = <f32 as common::UlpScalar>::ulp(got[i], &want);
            assert!(
                ulp <= 1.0,
                "exp({x:e}) = {} (want {want}, {ulp} ULP)",
                got[i]
            );
        }
    }

    let zeros = [-0.0f32, 0.0];
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
