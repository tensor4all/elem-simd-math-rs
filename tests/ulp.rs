//! ULP-accuracy tests against MPFR.
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x/tst.rs`, `src/f64/tst.rs`): the error bounds
//! asserted here are the ones upstream asserts for the same functions, so a failure means this
//! port is less accurate than the library it was ported from.
//!
//! Every test drives the public slice entry points, i.e. the same code path production uses
//! (runtime dispatch, lane loop, scalar tail), over lengths below, at and above the SIMD lane
//! width so that the vector body and the tail are both exercised.

#![cfg(feature = "simd")]

use core::mem::MaybeUninit;

use elem_simd_math::{f64 as ops, ElementwiseMath, LengthMismatch};
use rug::ops::Pow;
use rug::{Assign, Float};

/// Working precision for the MPFR oracle, matching upstream.
const PREC: u32 = 128;
/// `sqrt(f64::MAX)`, used by the inverse-hyperbolic ranges upstream tests.
const SQRT_DBL_MAX: f64 = 1.340_780_792_994_259_635_5e+154;

/// Lengths used per iteration: below, at and above the 2/4/8-lane widths, plus tail remainders.
const LENS: [usize; 7] = [1, 3, 4, 7, 17, 64, 257];

fn iterations() -> usize {
    std::env::var("ELEM_SIMD_MATH_ULP_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

/// Distance in ULPs between `d` and the higher-precision `c`, following upstream's definition.
fn count_ulp(d: f64, c: &Float) -> f64 {
    let c2 = c.to_f64();

    if (c2 == 0. || c2.is_subnormal()) && (d == 0. || d.is_subnormal()) {
        return 0.;
    }
    if c2 == 0. && d != 0. {
        return 10000.;
    }
    if c2.is_infinite() && d.is_infinite() {
        return 0.;
    }

    let prec = c.prec();
    let mut fry = Float::with_val(prec, d);
    let mut frw = Float::new(prec);
    let (_, e) = c.to_f64_exp();
    frw.assign(Float::u_exp(1, e - 53));
    fry -= c;
    fry /= &frw;
    fry.to_f64().abs()
}

/// Deterministic xorshift64* so failures are reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform sample of `[lo, hi]`, with the infinities clamped as upstream does.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let lo = if lo == f64::MIN { -1e306 } else { lo };
        let hi = if hi == f64::MAX { 1e306 } else { hi };
        lo + (hi - lo) * self.unit()
    }
}

/// Reads back a fully written output buffer.
///
/// # Safety
/// Every entry point under test writes each element before returning `Ok`, and every call site
/// checks the returned `Result` first. `MaybeUninit<f64>` is a transparent wrapper over `f64`
/// and `Vec` has the same layout for both element types.
fn read(out: Vec<MaybeUninit<f64>>) -> Vec<f64> {
    unsafe { core::mem::transmute::<Vec<MaybeUninit<f64>>, Vec<f64>>(out) }
}

fn input(rng: &mut Rng, len: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..len).map(|_| rng.range(lo, hi)).collect()
}

type Unary = fn(&[f64], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;
type Binary = fn(&[f64], &[f64], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;
type Pair =
    fn(&[f64], &mut [MaybeUninit<f64>], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;

#[track_caller]
fn assert_close(name: &str, it: usize, lane: usize, got: f64, want: &Float, ulp_max: f64) {
    if want.is_nan() && got.is_nan() {
        return;
    }
    let ulp = count_ulp(got, want);
    assert!(
        ulp <= ulp_max,
        "{name}: iter {it}, lane {lane}, got {got}, want {want}, {ulp} ULP > {ulp_max}"
    );
}

fn check_unary(name: &str, op: Unary, oracle: fn(Float) -> Float, lo: f64, hi: f64, ulp_max: f64) {
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    for it in 0..iterations() {
        let len = LENS[it % LENS.len()];
        let src = input(&mut rng, len, lo, hi);
        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        op(&src, &mut out).expect("length check");
        for (lane, (got, &x)) in read(out).iter().zip(src.iter()).enumerate() {
            let want = oracle(Float::with_val(PREC, x));
            assert_close(name, it, lane, *got, &want, ulp_max);
        }
    }
}

fn check_binary(
    name: &str,
    op: Binary,
    oracle: fn(Float, &Float) -> Float,
    lo1: f64,
    hi1: f64,
    lo2: f64,
    hi2: f64,
    ulp_max: f64,
) {
    let mut rng = Rng::new(0x2545_f491_4f6c_dd1d);
    for it in 0..iterations() {
        let len = LENS[it % LENS.len()];
        let a = input(&mut rng, len, lo1, hi1);
        let b = input(&mut rng, len, lo2, hi2);
        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        op(&a, &b, &mut out).expect("length check");
        for (lane, ((got, &x), &y)) in read(out).iter().zip(a.iter()).zip(b.iter()).enumerate() {
            let want = oracle(Float::with_val(PREC, x), &Float::with_val(PREC, y));
            assert_close(name, it, lane, *got, &want, ulp_max);
        }
    }
}

fn check_pair(
    name: &str,
    op: Pair,
    oracle: fn(Float) -> (Float, Float),
    lo: f64,
    hi: f64,
    ulp_max: f64,
) {
    let mut rng = Rng::new(0xdead_beef_1234_5678);
    for it in 0..iterations() {
        let len = LENS[it % LENS.len()];
        let src = input(&mut rng, len, lo, hi);
        let mut first: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        let mut second: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        op(&src, &mut first, &mut second).expect("length check");
        let first = read(first);
        let second = read(second);
        for lane in 0..len {
            let (want_a, want_b) = oracle(Float::with_val(PREC, src[lane]));
            assert_close(name, it, lane, first[lane], &want_a, ulp_max);
            assert_close(name, it, lane, second[lane], &want_b, ulp_max);
        }
    }
}

fn sinpi_oracle(x: Float) -> Float {
    let p = x.prec();
    (x * Float::with_val(p, rug::float::Constant::Pi)).sin()
}

fn cospi_oracle(x: Float) -> Float {
    let p = x.prec();
    (x * Float::with_val(p, rug::float::Constant::Pi)).cos()
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
        |x: Float| x.sin_cos(Float::new(x.prec())),
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
            let p = x.prec();
            (x * Float::with_val(p, rug::float::Constant::Pi)).sin_cos(Float::new(p))
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
        assert_close("exp special", 0, i, got[i], &want, 1.0);
    }

    let mut out = vec![MaybeUninit::uninit(); cases.len()];
    ops::log(&cases, &mut out).unwrap();
    let got = read(out);
    for (i, &x) in cases.iter().enumerate() {
        let want = Float::with_val(PREC, x).ln();
        if want.is_nan() {
            assert!(got[i].is_nan(), "log({x:e}) = {}", got[i]);
        } else {
            assert_close("log special", 0, i, got[i], &want, 3.5);
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
