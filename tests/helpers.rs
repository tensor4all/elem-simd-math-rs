//! Tests for the shared `f64` helper kernels.
//!
//! These drive the vector helpers in `elem_simd_math::f64x` directly, one `fearless_simd` level
//! at a time, so every lane is exercised. The bounds marked "upstream" are the ones
//! `sleef-rs` asserts for the same function.

#![cfg(feature = "simd")]

use elem_simd_math::common::F64s;
use elem_simd_math::f64x;
use fearless_simd::prelude::*;
use fearless_simd::{dispatch, Level, Simd};
use rug::{Assign, Float};

const PREC: u32 = 128;

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
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn lanes<S: Simd>(&mut self, lo: f64, hi: f64) -> Vec<f64> {
        (0..F64s::<S>::LEN).map(|_| self.range(lo, hi)).collect()
    }
}

fn trials() -> usize {
    std::env::var("ELEM_SIMD_MATH_ULP_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000)
}

#[track_caller]
fn close(name: &str, trial: usize, lane: usize, got: f64, want: &Float, ulp_max: f64) {
    if want.is_nan() && got.is_nan() {
        return;
    }
    let ulp = count_ulp(got, want);
    assert!(
        ulp <= ulp_max,
        "{name}: trial {trial}, lane {lane}, got {got}, want {want}, {ulp} ULP > {ulp_max}"
    );
}

fn binary<S: Simd>(
    simd: S,
    name: &str,
    kernel: fn(S, F64s<S>, F64s<S>) -> F64s<S>,
    oracle: fn(Float, &Float) -> Float,
    lo: f64,
    hi: f64,
    ulp_max: f64,
) {
    let mut rng = Rng::new(0x0123_4567_89ab_cdef);
    for trial in 0..trials() {
        let a = rng.lanes::<S>(lo, hi);
        let b = rng.lanes::<S>(lo, hi);
        let va = F64s::<S>::from_slice(simd, &a);
        let vb = F64s::<S>::from_slice(simd, &b);
        let arr = kernel(simd, va, vb).to_array();
        for lane in 0..a.len() {
            let want = oracle(
                Float::with_val(PREC, a[lane]),
                &Float::with_val(PREC, b[lane]),
            );
            close(name, trial, lane, arr.as_ref()[lane], &want, ulp_max);
        }
    }
}

/// `remainder`; upstream asserts 0.5 ULP.
#[test]
fn remainder_ulp() {
    let level = Level::new();
    dispatch!(level, simd => binary(
        simd,
        "remainder",
        f64x::remainder,
        |x: Float, y: &Float| x.remainder(y),
        f64::MIN,
        f64::MAX,
        0.5,
    ));
}

/// `fmod`; upstream has no test for it (only a TODO), so 0.5 ULP is checked here as the bound
/// the algorithm's design implies, not as a ported upstream assertion.
#[test]
fn fmod_ulp() {
    let level = Level::new();
    dispatch!(level, simd => binary(
        simd,
        "fmod",
        f64x::fmod,
        |x: Float, y: &Float| {
            let q = (x.clone() / y).trunc();
            x - q * y
        },
        f64::MIN,
        f64::MAX,
        0.5,
    ));
}

/// `nextafter`; upstream asserts 0.1 ULP.
#[test]
fn nextafter_ulp() {
    let level = Level::new();
    dispatch!(level, simd => binary(
        simd,
        "nextafter",
        f64x::nextafter,
        |mut x: Float, y: &Float| {
            let prec = x.prec();
            x.set_prec(53);
            x.next_toward(y);
            x.set_prec(prec);
            x
        },
        f64::MIN,
        f64::MAX,
        0.1,
    ));
}

/// `fma`; upstream documents `max(0.500_01 ULP, f64::MIN_POSITIVE)`.
#[test]
fn fma_ulp() {
    let level = Level::new();
    dispatch!(level, simd => fma_inner(simd));
}

fn fma_inner<S: Simd>(simd: S) {
    {
        let mut rng = Rng::new(0xfeed_face_dead_beef);
        for trial in 0..trials() {
            let a = rng.lanes::<S>(f64::MIN, f64::MAX);
            let b = rng.lanes::<S>(f64::MIN, f64::MAX);
            let c = rng.lanes::<S>(f64::MIN, f64::MAX);
            let va = F64s::<S>::from_slice(simd, &a);
            let vb = F64s::<S>::from_slice(simd, &b);
            let vc = F64s::<S>::from_slice(simd, &c);
            let arr = f64x::fma(simd, va, vb, vc).to_array();
            for lane in 0..a.len() {
                if arr.as_ref()[lane].abs() > 1e300 {
                    // Upstream returns infinity beyond this magnitude; skip those.
                    continue;
                }
                let want = Float::with_val(PREC, a[lane]) * Float::with_val(PREC, b[lane])
                    + Float::with_val(PREC, c[lane]);
                close("fma", trial, lane, arr.as_ref()[lane], &want, 0.50001);
            }
        }
    }
}

/// `frfrexp`/`expfrexp`/`ldexp`: the decomposition must round-trip exactly, and the fractional
/// part must lie in `[0.5, 1)`.
#[test]
fn frexp_roundtrip() {
    let level = Level::new();
    dispatch!(level, simd => frexp_inner(simd));
}

fn frexp_inner<S: Simd>(simd: S) {
    {
        let mut rng = Rng::new(0xabcd_ef01_2345_6789);
        for trial in 0..trials() {
            let mut a = rng.lanes::<S>(f64::MIN, f64::MAX);
            a.extend_from_slice(&[1.0, 0.5, 2.0, 1e-300, 1e300, -1.0]);
            let a = &a[..F64s::<S>::LEN];
            let v = F64s::<S>::from_slice(simd, a);
            let m = f64x::frfrexp(simd, v);
            let e = f64x::expfrexp(simd, v);
            let back = f64x::ldexp(simd, m, e);
            let marr = m.to_array();
            let barr = back.to_array();
            for (lane, &x) in a.iter().enumerate() {
                if x == 0.0 || x.is_nan() || x.is_infinite() {
                    continue;
                }
                let got = barr.as_ref()[lane];
                assert!(
                    got.to_bits() == x.to_bits(),
                    "trial {trial}, lane {lane}: ldexp(frfrexp({x:e}), expfrexp({x:e})) = {got:e}"
                );
                assert!(
                    marr.as_ref()[lane].abs() >= 0.5 && marr.as_ref()[lane].abs() < 1.0,
                    "trial {trial}, lane {lane}: fractional part {} out of [0.5, 1)",
                    marr.as_ref()[lane]
                );
            }
        }
    }
}

/// `ilogb` matches the platform's `ilogb` on the same values.
#[test]
fn ilogb_matches_platform() {
    let level = Level::new();
    dispatch!(level, simd => ilogb_inner(simd));
}

fn ilogb_inner<S: Simd>(simd: S) {
    {
        let mut rng = Rng::new(0x5555_aaaa_5555_aaaa);
        for trial in 0..trials() {
            let a = rng.lanes::<S>(f64::MIN_POSITIVE, f64::MAX);
            let v = F64s::<S>::from_slice(simd, &a);
            let arr = f64x::ilogb(simd, v).to_array();
            for (lane, &x) in a.iter().enumerate() {
                if x == 0.0 || x.is_nan() || x.is_infinite() {
                    continue;
                }
                assert_eq!(
                    arr.as_ref()[lane],
                    libm_ilogb(x),
                    "trial {trial}, lane {lane}, ilogb({x:e})"
                );
            }
        }
    }
}

unsafe extern "C" {
    fn ilogb(x: f64) -> std::ffi::c_int;
}

fn libm_ilogb(x: f64) -> i64 {
    // SAFETY: the C library's `ilogb` is a pure function of its argument.
    unsafe { ilogb(x) as i64 }
}

/// `modf` splits the value exactly, and `fdim`/`fmax`/`fmin` follow their C definitions.
#[test]
fn modf_and_ordering_helpers() {
    let level = Level::new();
    dispatch!(level, simd => modf_inner(simd));
}

fn modf_inner<S: Simd>(simd: S) {
    {
        let mut rng = Rng::new(0x1111_2222_3333_4444);
        for trial in 0..trials() {
            let a = rng.lanes::<S>(-1e6, 1e6);
            let b = rng.lanes::<S>(-1e6, 1e6);
            let va = F64s::<S>::from_slice(simd, &a);
            let vb = F64s::<S>::from_slice(simd, &b);

            let (frac, int) = f64x::modf(simd, va);
            let far = frac.to_array();
            let iar = int.to_array();
            let fdim = f64x::fdim(simd, va, vb).to_array();
            let fmax = f64x::fmax(simd, va, vb).to_array();
            let fmin = f64x::fmin(simd, va, vb).to_array();

            for lane in 0..a.len() {
                let (x, y) = (a[lane], b[lane]);
                let got_frac = far.as_ref()[lane];
                let got_int = iar.as_ref()[lane];
                assert!(
                    got_frac.to_bits() == x.fract().to_bits(),
                    "trial {trial}, lane {lane}: modf({x}) fraction is {got_frac}"
                );
                assert!(
                    got_int.to_bits() == x.trunc().to_bits(),
                    "trial {trial}, lane {lane}: modf({x}) integral part is {got_int}"
                );
                assert_eq!(
                    got_frac + got_int,
                    x,
                    "trial {trial}, lane {lane}: modf does not recombine"
                );

                let want_fdim = if x > y { x - y } else { 0.0 };
                assert_eq!(
                    fdim.as_ref()[lane],
                    want_fdim,
                    "fdim({x}, {y}), lane {lane}"
                );
                assert_eq!(fmax.as_ref()[lane], x.max(y), "fmax({x}, {y}), lane {lane}");
                assert_eq!(fmin.as_ref()[lane], x.min(y), "fmin({x}, {y}), lane {lane}");
            }
        }
    }
}

/// `fmod` on exact large integrals.
///
/// The quotient estimate has to stay conservative: with a fused multiply-add the product
/// `toward0(r) * rd` can round up, and a quotient that is one too large makes the loop overshoot
/// and return a small negative remainder instead of zero. `7 * 2^51` is the smallest case where
/// the difference shows, and `f64`'s own `%` is the reference.
fn check_fmod<S: Simd>(simd: S, x: f64, y: f64) {
    let got = f64x::fmod(simd, F64s::<S>::splat(simd, x), F64s::<S>::splat(simd, y)).to_array();
    assert_eq!(got.as_ref()[0], x % y, "fmod({x:e}, {y:e})");
}

#[test]
fn fmod_of_large_integrals() {
    let level = Level::new();
    let cases = [
        7.0 * 2.0f64.powi(51),
        3.0 * 2.0f64.powi(52) + 1.0,
        (1u64 << 62) as f64,
        1e300,
        f64::MAX,
        6.0 * 2.0f64.powi(53),
    ];
    for &x in cases.iter() {
        for &y in [1.0f64, 2.0, 3.0, 8.0, 1.0e150].iter() {
            dispatch!(level, simd => check_fmod(simd, x, y));
        }
    }
}
