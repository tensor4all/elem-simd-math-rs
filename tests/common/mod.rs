//! Shared harness for the MPFR ULP suites.
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64/tst.rs`, `src/f32/tst.rs`): the bounds the suites
//! assert are the ones upstream asserts for the same functions, so a failure means this crate is
//! less accurate than the library it was ported from.
//!
//! The harness is generic over the scalar type so that the `f32` and `f64` suites share it; only
//! the ULP metric and the case tables differ, and those live in `tests/ulp.rs` and
//! `tests/ulp32.rs`.

#![allow(dead_code)]

use core::mem::MaybeUninit;

use elem_simd_math::LengthMismatch;
use rug::{Assign, Float};

/// Working precision for the MPFR oracle, matching upstream.
pub const PREC: u32 = 128;

/// Lengths used per iteration. These cover every tail remainder for 2-, 4-, 8- and 16-lane
/// vectors, the exact lane counts, multiples of them, and a long case.
pub const LENS: [usize; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 64, 257];

/// Number of samples per function, overridable for a quick smoke run.
pub fn iterations() -> usize {
    std::env::var("ELEM_SIMD_MATH_ULP_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

/// A unary slice entry point.
pub type UnaryOp<T> = fn(&[T], &mut [MaybeUninit<T>]) -> Result<(), LengthMismatch>;

/// A binary slice entry point.
pub type BinaryOp<T> = fn(&[T], &[T], &mut [MaybeUninit<T>]) -> Result<(), LengthMismatch>;

/// An entry point that writes two outputs.
pub type PairOp<T> =
    fn(&[T], &mut [MaybeUninit<T>], &mut [MaybeUninit<T>]) -> Result<(), LengthMismatch>;

/// A scalar type with an ULP metric against an MPFR value.
pub trait UlpScalar: Copy + 'static {
    /// Distance in ULPs between `d` and the higher-precision `c`, following upstream's definition.
    fn ulp(d: Self, c: &Float) -> f64;

    /// The same value widened to `f64`, for the deterministic input generator and diagnostics.
    fn from_f64(x: f64) -> Self;

    /// The largest magnitude a full-range sample uses, so the sample stays finite for this type.
    const SAMPLE_LIMIT: f64;

    /// The MPFR image of this value, at the oracle's working precision.
    fn to_float(self) -> Float;

    /// Whether this value is NaN.
    fn is_nan(self) -> bool;

    /// This value widened to `f64`, for diagnostics.
    fn widen(self) -> f64;
}

impl UlpScalar for f64 {
    const SAMPLE_LIMIT: f64 = 1e306;

    fn ulp(d: f64, c: &Float) -> f64 {
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

    fn from_f64(x: f64) -> Self {
        x
    }

    fn to_float(self) -> Float {
        Float::with_val(PREC, self)
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    fn widen(self) -> f64 {
        self
    }
}

impl UlpScalar for f32 {
    const SAMPLE_LIMIT: f64 = 1e38;

    fn ulp(d: f32, c: &Float) -> f64 {
        let c2 = c.to_f32();
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
        let (_, e) = c.to_f32_exp();
        frw.assign(Float::u_exp(1, e - 24));
        fry -= c;
        fry /= &frw;
        fry.to_f64().abs()
    }

    fn from_f64(x: f64) -> Self {
        x as f32
    }

    fn to_float(self) -> Float {
        Float::with_val(PREC, self)
    }

    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    fn widen(self) -> f64 {
        self as f64
    }
}

/// Deterministic xorshift64* so failures are reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
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

    /// A uniform sample of `[lo, hi]`, clamped to `limit` so that a full-range sample stays finite
    /// for the element type instead of rounding to infinity.
    pub fn range(&mut self, lo: f64, hi: f64, limit: f64) -> f64 {
        let lo = if lo <= -limit { -limit } else { lo };
        let hi = if hi >= limit { limit } else { hi };
        lo + (hi - lo) * self.unit()
    }
}

/// Reads back a fully written output buffer.
///
/// # Safety
/// Every entry point under test writes each element before returning `Ok`, and every call site
/// checks the returned `Result` first.
pub fn read<T: Copy>(out: Vec<MaybeUninit<T>>) -> Vec<T> {
    out.into_iter()
        .map(|v| unsafe { v.assume_init() })
        .collect()
}

fn input<T: UlpScalar>(rng: &mut Rng, len: usize, lo: f64, hi: f64) -> Vec<T> {
    (0..len)
        .map(|_| T::from_f64(rng.range(lo, hi, T::SAMPLE_LIMIT)))
        .collect()
}

/// Asserts that `got` is within `ulp_max` ULP of the oracle value `want`.
#[track_caller]
pub fn assert_close<T: UlpScalar>(
    name: &str,
    it: usize,
    lane: usize,
    x: T,
    got: T,
    want: &Float,
    ulp_max: f64,
) {
    if want.is_nan() && got.is_nan() {
        return;
    }
    let ulp = T::ulp(got, want);
    assert!(
        ulp <= ulp_max,
        "{name}: iter {it}, lane {lane}, x={}, got {}, want {want}, {ulp} ULP > {ulp_max}",
        x.widen(),
        got.widen()
    );
}

#[track_caller]
pub fn check_unary<T: UlpScalar>(
    name: &str,
    op: UnaryOp<T>,
    oracle: fn(Float) -> Float,
    lo: f64,
    hi: f64,
    ulp_max: f64,
) {
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    for it in 0..iterations() {
        let len = LENS[it % LENS.len()];
        let src = input::<T>(&mut rng, len, lo, hi);
        let mut out: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); len];
        op(&src, &mut out).expect("length check");
        for (lane, (got, &x)) in read(out).iter().zip(src.iter()).enumerate() {
            assert_close(name, it, lane, x, *got, &oracle(x.to_float()), ulp_max);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn check_binary<T: UlpScalar>(
    name: &str,
    op: BinaryOp<T>,
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
        let a = input::<T>(&mut rng, len, lo1, hi1);
        let b = input::<T>(&mut rng, len, lo2, hi2);
        let mut out: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); len];
        op(&a, &b, &mut out).expect("length check");
        for (lane, ((got, &x), &y)) in read(out).iter().zip(a.iter()).zip(b.iter()).enumerate() {
            let want = oracle(x.to_float(), &y.to_float());
            assert_close(name, it, lane, x, *got, &want, ulp_max);
        }
    }
}

pub fn check_pair<T: UlpScalar>(
    name: &str,
    op: PairOp<T>,
    oracle: fn(Float) -> (Float, Float),
    lo: f64,
    hi: f64,
    ulp_max: f64,
) {
    let mut rng = Rng::new(0xdead_beef_1234_5678);
    for it in 0..iterations() {
        let len = LENS[it % LENS.len()];
        let src = input::<T>(&mut rng, len, lo, hi);
        let mut first: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); len];
        let mut second: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); len];
        op(&src, &mut first, &mut second).expect("length check");
        let first = read(first);
        let second = read(second);
        for lane in 0..len {
            let (want_a, want_b) = oracle(src[lane].to_float());
            assert_close(name, it, lane, src[lane], first[lane], &want_a, ulp_max);
            assert_close(name, it, lane, src[lane], second[lane], &want_b, ulp_max);
        }
    }
}
