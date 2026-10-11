//! Standalone `f64` throughput benchmark.
//!
//! Compares each implemented kernel against the platform's scalar implementation in the same
//! binary, over the tensor sizes that motivate the crate, and cross-checks the two results so a
//! fast-but-wrong kernel cannot look good.
//!
//! Run with `cargo bench`. The measurement protocol is recorded in `docs/benchmark-protocol.md`
//! and must be fixed before the numbers are collected.

use core::hint::black_box;
use core::mem::MaybeUninit;
use std::time::Instant;

use elem_simd_math::{f64 as ops, LengthMismatch};

/// Primary size: 2^23 elements, matching the `real.*` rows of tenferro-rs#2033.
const PRIMARY: usize = 1 << 23;
/// Secondary sizes, to show scaling.
const SIZES: [usize; 3] = [1 << 16, 1 << 20, PRIMARY];
/// Warmups and timed rounds per case.
const WARMUP: usize = 2;
const ROUNDS: usize = 7;

type Unary = fn(&[f64], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;
type Binary = fn(&[f64], &[f64], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;
type Pair =
    fn(&[f64], &mut [MaybeUninit<f64>], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;

struct UnaryCase {
    name: &'static str,
    kernel: Unary,
    scalar: fn(f64) -> f64,
}

struct BinaryCase {
    name: &'static str,
    kernel: Binary,
    scalar: fn(f64, f64) -> f64,
}

const UNARY: &[UnaryCase] = &[
    UnaryCase {
        name: "exp",
        kernel: ops::exp,
        scalar: f64::exp,
    },
    UnaryCase {
        name: "exp2",
        kernel: ops::exp2,
        scalar: f64::exp2,
    },
    UnaryCase {
        name: "exp10",
        kernel: ops::exp10,
        scalar: |v| 10.0f64.powf(v),
    },
    UnaryCase {
        name: "expm1",
        kernel: ops::expm1,
        scalar: f64::exp_m1,
    },
    UnaryCase {
        name: "log",
        kernel: ops::log,
        scalar: f64::ln,
    },
    UnaryCase {
        name: "log2",
        kernel: ops::log2,
        scalar: f64::log2,
    },
    UnaryCase {
        name: "log10",
        kernel: ops::log10,
        scalar: f64::log10,
    },
    UnaryCase {
        name: "log1p",
        kernel: ops::log1p,
        scalar: f64::ln_1p,
    },
    UnaryCase {
        name: "sin",
        kernel: ops::sin,
        scalar: f64::sin,
    },
    UnaryCase {
        name: "cos",
        kernel: ops::cos,
        scalar: f64::cos,
    },
    UnaryCase {
        name: "tan",
        kernel: ops::tan,
        scalar: f64::tan,
    },
    UnaryCase {
        name: "asin",
        kernel: ops::asin,
        scalar: f64::asin,
    },
    UnaryCase {
        name: "acos",
        kernel: ops::acos,
        scalar: f64::acos,
    },
    UnaryCase {
        name: "atan",
        kernel: ops::atan,
        scalar: f64::atan,
    },
    UnaryCase {
        name: "sinh",
        kernel: ops::sinh,
        scalar: f64::sinh,
    },
    UnaryCase {
        name: "cosh",
        kernel: ops::cosh,
        scalar: f64::cosh,
    },
    UnaryCase {
        name: "tanh",
        kernel: ops::tanh,
        scalar: f64::tanh,
    },
    UnaryCase {
        name: "asinh",
        kernel: ops::asinh,
        scalar: f64::asinh,
    },
    UnaryCase {
        name: "acosh",
        kernel: ops::acosh,
        scalar: f64::acosh,
    },
    UnaryCase {
        name: "atanh",
        kernel: ops::atanh,
        scalar: f64::atanh,
    },
    UnaryCase {
        name: "cbrt",
        kernel: ops::cbrt,
        scalar: f64::cbrt,
    },
    UnaryCase {
        name: "sqrt",
        kernel: ops::sqrt,
        scalar: f64::sqrt,
    },
    UnaryCase {
        name: "sin (u35)",
        kernel: ops::sin_u35,
        scalar: f64::sin,
    },
    UnaryCase {
        name: "cos (u35)",
        kernel: ops::cos_u35,
        scalar: f64::cos,
    },
    UnaryCase {
        name: "log (u35)",
        kernel: ops::log_u35,
        scalar: f64::ln,
    },
    UnaryCase {
        name: "tanh (u35)",
        kernel: ops::tanh_u35,
        scalar: f64::tanh,
    },
];

// `erf`, `erfc`, `tgamma`, `lgamma`, `sinpi` and `cospi` have no scalar equivalent in the
// standard library, so they only exist with the `simd` feature.
#[cfg(feature = "simd")]
const UNARY_SIMD_ONLY: &[UnaryCase] = &[
    UnaryCase {
        name: "erf",
        kernel: ops::erf,
        scalar: libm_erf,
    },
    UnaryCase {
        name: "erfc",
        kernel: ops::erfc,
        scalar: libm_erfc,
    },
    UnaryCase {
        name: "tgamma",
        kernel: ops::tgamma,
        scalar: libm_tgamma,
    },
    UnaryCase {
        name: "lgamma",
        kernel: ops::lgamma,
        scalar: libm_lgamma,
    },
    UnaryCase {
        name: "sinpi",
        kernel: ops::sinpi,
        scalar: |v| (v * std::f64::consts::PI).sin(),
    },
    UnaryCase {
        name: "cospi",
        kernel: ops::cospi,
        scalar: |v| (v * std::f64::consts::PI).cos(),
    },
];

#[cfg(not(feature = "simd"))]
const UNARY_SIMD_ONLY: &[UnaryCase] = &[];

const BINARY: &[BinaryCase] = &[
    BinaryCase {
        name: "pow",
        kernel: ops::pow,
        scalar: f64::powf,
    },
    BinaryCase {
        name: "atan2",
        kernel: ops::atan2,
        scalar: f64::atan2,
    },
    BinaryCase {
        name: "hypot",
        kernel: ops::hypot,
        scalar: f64::hypot,
    },
];

#[cfg(feature = "simd")]
extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
    fn tgamma(x: f64) -> f64;
    fn lgamma(x: f64) -> f64;
}

#[cfg(feature = "simd")]
fn libm_erf(x: f64) -> f64 {
    // SAFETY: the C library's `erf` is a pure function of its argument.
    unsafe { erf(x) }
}
#[cfg(feature = "simd")]
fn libm_erfc(x: f64) -> f64 {
    // SAFETY: as above.
    unsafe { erfc(x) }
}
#[cfg(feature = "simd")]
fn libm_tgamma(x: f64) -> f64 {
    // SAFETY: as above.
    unsafe { tgamma(x) }
}
#[cfg(feature = "simd")]
fn libm_lgamma(x: f64) -> f64 {
    // SAFETY: as above.
    unsafe { lgamma(x) }
}

fn uninit(len: usize) -> Vec<MaybeUninit<f64>> {
    vec![MaybeUninit::uninit(); len]
}

/// Reads back a fully written buffer.
///
/// # Safety contract: the call under test wrote every element.
fn read(out: Vec<MaybeUninit<f64>>) -> Vec<f64> {
    unsafe { core::mem::transmute::<Vec<MaybeUninit<f64>>, Vec<f64>>(out) }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn time_scalar_unary(src: &[f64], out: &mut [MaybeUninit<f64>], f: fn(f64) -> f64) -> f64 {
    let start = Instant::now();
    for (slot, &x) in out.iter_mut().zip(src.iter()) {
        slot.write(f(black_box(x)));
    }
    black_box(&out[0]);
    start.elapsed().as_secs_f64()
}

fn time_kernel_unary(src: &[f64], out: &mut [MaybeUninit<f64>], kernel: Unary) -> f64 {
    let start = Instant::now();
    kernel(black_box(src), out).expect("lengths match");
    black_box(&out[0]);
    start.elapsed().as_secs_f64()
}

fn time_scalar_binary(
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [MaybeUninit<f64>],
    f: fn(f64, f64) -> f64,
) -> f64 {
    let start = Instant::now();
    for ((slot, &a), &b) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        slot.write(f(black_box(a), black_box(b)));
    }
    black_box(&out[0]);
    start.elapsed().as_secs_f64()
}

fn time_kernel_binary(
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [MaybeUninit<f64>],
    kernel: Binary,
) -> f64 {
    let start = Instant::now();
    kernel(black_box(lhs), black_box(rhs), out).expect("lengths match");
    black_box(&out[0]);
    start.elapsed().as_secs_f64()
}

fn time_kernel_pair(
    src: &[f64],
    a: &mut [MaybeUninit<f64>],
    b: &mut [MaybeUninit<f64>],
    kernel: Pair,
) -> f64 {
    let start = Instant::now();
    kernel(black_box(src), a, b).expect("lengths match");
    black_box(&a[0]);
    black_box(&b[0]);
    start.elapsed().as_secs_f64()
}

/// Inputs chosen so that every kernel stays inside its documented range and produces a finite,
/// non-degenerate result.
fn unary_input(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            let x = t * 2.0 - 1.0;
            match i % 4 {
                0 => x * 0.5,
                1 => (x + 1.5).abs() + 0.125,
                2 => x * 8.0,
                _ => x * 0.125 + 0.75,
            }
        })
        .collect()
}

fn main() {
    println!("elem-simd-math f64 throughput");
    println!("level: {:?}", fearless_simd::Level::new());
    println!("warmup {WARMUP}, rounds {ROUNDS}, statistic = median wall time, ns/element");
    println!();

    for size in SIZES {
        let src = unary_input(size);
        let mut out = uninit(size);
        let mut pair_a = uninit(size);
        let mut pair_b = uninit(size);
        let rhs: Vec<f64> = (0..size).map(|i| 1.0 + (i % 64) as f64 * 0.03125).collect();

        println!("== n = {size} ==");
        println!(
            "{:<12} {:>10} {:>10} {:>9} {:>10} {:>9}",
            "op", "scalar", "kernel", "speedup", "max |diff|", "rel"
        );

        for case in UNARY.iter().chain(UNARY_SIMD_ONLY) {
            for _ in 0..WARMUP {
                let _ = time_kernel_unary(&src, &mut out, case.kernel);
            }
            let kernel_times: Vec<f64> = (0..ROUNDS)
                .map(|_| time_kernel_unary(&src, &mut out, case.kernel))
                .collect();
            let scalar_times: Vec<f64> = (0..ROUNDS)
                .map(|_| time_scalar_unary(&src, &mut out, case.scalar))
                .collect();

            let mut out = uninit(size);
            (case.kernel)(&src, &mut out).expect("lengths match");
            let got = read(out);
            let mut max_diff = 0.0f64;
            let mut max_rel = 0.0f64;
            for (i, &x) in src.iter().enumerate() {
                let want = (case.scalar)(x);
                let diff = (got[i] - want).abs();
                if got[i].is_finite() && want.is_finite() {
                    max_diff = max_diff.max(diff);
                    let scale = want.abs().max(1e-300);
                    max_rel = max_rel.max(diff / scale);
                }
            }

            let kernel_s = median(kernel_times) / size as f64 * 1e9;
            let scalar_s = median(scalar_times) / size as f64 * 1e9;
            println!(
                "{:<12} {:>10.3} {:>10.3} {:>8.2}x {:>10.2e} {:>9.1e}",
                case.name,
                scalar_s,
                kernel_s,
                scalar_s / kernel_s,
                max_diff,
                max_rel
            );
        }

        for case in BINARY {
            for _ in 0..WARMUP {
                let _ = time_kernel_binary(&src, &rhs, &mut out, case.kernel);
            }
            let kernel_times: Vec<f64> = (0..ROUNDS)
                .map(|_| time_kernel_binary(&src, &rhs, &mut out, case.kernel))
                .collect();
            let scalar_times: Vec<f64> = (0..ROUNDS)
                .map(|_| time_scalar_binary(&src, &rhs, &mut out, case.scalar))
                .collect();

            let mut out = uninit(size);
            (case.kernel)(&src, &rhs, &mut out).expect("lengths match");
            let got = read(out);
            let mut max_diff = 0.0f64;
            let mut max_rel = 0.0f64;
            for (i, (&x, &y)) in src.iter().zip(rhs.iter()).enumerate() {
                let want = (case.scalar)(x, y);
                let diff = (got[i] - want).abs();
                if got[i].is_finite() && want.is_finite() {
                    max_diff = max_diff.max(diff);
                    max_rel = max_rel.max(diff / want.abs().max(1e-300));
                }
            }

            let kernel_s = median(kernel_times) / size as f64 * 1e9;
            let scalar_s = median(scalar_times) / size as f64 * 1e9;
            println!(
                "{:<12} {:>10.3} {:>10.3} {:>8.2}x {:>10.2e} {:>9.1e}",
                case.name,
                scalar_s,
                kernel_s,
                scalar_s / kernel_s,
                max_diff,
                max_rel
            );
        }

        // sincos is measured separately: it produces two outputs.
        for _ in 0..WARMUP {
            let _ = time_kernel_pair(&src, &mut pair_a, &mut pair_b, ops::sincos);
        }
        let pair_times: Vec<f64> = (0..ROUNDS)
            .map(|_| time_kernel_pair(&src, &mut pair_a, &mut pair_b, ops::sincos))
            .collect();
        let scalar_pair = |s: &[f64], a: &mut [MaybeUninit<f64>], b: &mut [MaybeUninit<f64>]| {
            let start = Instant::now();
            for ((sa, sb), &x) in a.iter_mut().zip(b.iter_mut()).zip(s.iter()) {
                sa.write(black_box(x.sin()));
                sb.write(black_box(x.cos()));
            }
            black_box(&a[0]);
            black_box(&b[0]);
            start.elapsed().as_secs_f64()
        };
        for _ in 0..WARMUP {
            let _ = scalar_pair(&src, &mut pair_a, &mut pair_b);
        }
        let scalar_pair_times: Vec<f64> = (0..ROUNDS)
            .map(|_| scalar_pair(&src, &mut pair_a, &mut pair_b))
            .collect();
        let pair_s = median(pair_times) / size as f64 * 1e9;
        let scalar_pair_s = median(scalar_pair_times) / size as f64 * 1e9;
        println!(
            "{:<12} {:>10.3} {:>10.3} {:>8.2}x",
            "sincos",
            scalar_pair_s,
            pair_s,
            scalar_pair_s / pair_s
        );
        println!();
    }
}
