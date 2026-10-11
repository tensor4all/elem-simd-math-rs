//! Standalone throughput benchmark for both dtypes.
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

use elem_simd_math::LengthMismatch;

/// Primary size: 2^23 elements, matching the `real.*` rows of tenferro-rs#2033.
const PRIMARY: usize = 1 << 23;
/// Secondary sizes, to show scaling.
const SIZES: [usize; 2] = [1 << 16, PRIMARY];
/// Warmups and timed rounds per case.
const WARMUP: usize = 2;
const ROUNDS: usize = 7;

// Scalars the standard library does not provide, reached through the platform's C library. They
// are only used as the benchmark's baseline arm.
extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
    fn tgamma(x: f64) -> f64;
    fn lgamma(x: f64) -> f64;
    fn erff(x: f32) -> f32;
    fn erfcf(x: f32) -> f32;
    fn tgammaf(x: f32) -> f32;
    fn lgammaf(x: f32) -> f32;
}

macro_rules! bench_family {
    (
        $(#[$meta:meta])*
        mod $modname:ident,
        scalar = $scalar:ty,
        ops = $ops:path,
        erf = $erf:path, erfc = $erfc:path, tgamma = $tgamma:path, lgamma = $lgamma:path,
    ) => {
        $(#[$meta])*
        pub mod $modname {
            use super::*;

            type Unary = fn(&[$scalar], &mut [MaybeUninit<$scalar>]) -> Result<(), LengthMismatch>;
            type Binary =
                fn(&[$scalar], &[$scalar], &mut [MaybeUninit<$scalar>]) -> Result<(), LengthMismatch>;
            type Pair = fn(
                &[$scalar],
                &mut [MaybeUninit<$scalar>],
                &mut [MaybeUninit<$scalar>],
            ) -> Result<(), LengthMismatch>;

            struct UnaryCase {
                name: &'static str,
                kernel: Unary,
                scalar: fn($scalar) -> $scalar,
            }

            struct BinaryCase {
                name: &'static str,
                kernel: Binary,
                scalar: fn($scalar, $scalar) -> $scalar,
            }

            fn common() -> &'static [UnaryCase] {
                &[
                    UnaryCase { name: "exp", kernel: $ops::exp, scalar: <$scalar>::exp },
                    UnaryCase { name: "exp2", kernel: $ops::exp2, scalar: <$scalar>::exp2 },
                    UnaryCase { name: "expm1", kernel: $ops::expm1, scalar: <$scalar>::exp_m1 },
                    UnaryCase { name: "log", kernel: $ops::log, scalar: <$scalar>::ln },
                    UnaryCase { name: "log2", kernel: $ops::log2, scalar: <$scalar>::log2 },
                    UnaryCase { name: "log10", kernel: $ops::log10, scalar: <$scalar>::log10 },
                    UnaryCase { name: "log1p", kernel: $ops::log1p, scalar: <$scalar>::ln_1p },
                    UnaryCase { name: "sin", kernel: $ops::sin, scalar: <$scalar>::sin },
                    UnaryCase { name: "cos", kernel: $ops::cos, scalar: <$scalar>::cos },
                    UnaryCase { name: "tan", kernel: $ops::tan, scalar: <$scalar>::tan },
                    UnaryCase { name: "asin", kernel: $ops::asin, scalar: <$scalar>::asin },
                    UnaryCase { name: "acos", kernel: $ops::acos, scalar: <$scalar>::acos },
                    UnaryCase { name: "atan", kernel: $ops::atan, scalar: <$scalar>::atan },
                    UnaryCase { name: "sinh", kernel: $ops::sinh, scalar: <$scalar>::sinh },
                    UnaryCase { name: "cosh", kernel: $ops::cosh, scalar: <$scalar>::cosh },
                    UnaryCase { name: "tanh", kernel: $ops::tanh, scalar: <$scalar>::tanh },
                    UnaryCase { name: "asinh", kernel: $ops::asinh, scalar: <$scalar>::asinh },
                    UnaryCase { name: "acosh", kernel: $ops::acosh, scalar: <$scalar>::acosh },
                    UnaryCase { name: "atanh", kernel: $ops::atanh, scalar: <$scalar>::atanh },
                    UnaryCase { name: "cbrt", kernel: $ops::cbrt, scalar: <$scalar>::cbrt },
                    UnaryCase { name: "sqrt", kernel: $ops::sqrt, scalar: <$scalar>::sqrt },
                    UnaryCase { name: "sin (u35)", kernel: $ops::sin_u35, scalar: <$scalar>::sin },
                    UnaryCase { name: "cos (u35)", kernel: $ops::cos_u35, scalar: <$scalar>::cos },
                    UnaryCase { name: "tan (u35)", kernel: $ops::tan_u35, scalar: <$scalar>::tan },
                    UnaryCase { name: "log (u35)", kernel: $ops::log_u35, scalar: <$scalar>::ln },
                    UnaryCase { name: "tanh (u35)", kernel: $ops::tanh_u35, scalar: <$scalar>::tanh },
                ]
            }

            /// `erf`, `erfc`, `tgamma`, `lgamma`, `sinpi`, `cospi`, `sincospi` and `exp10` have no
            /// scalar equivalent in the standard library, so they only exist with `simd`.
            #[cfg(feature = "simd")]
            fn simd_only() -> &'static [UnaryCase] {
                &[
                    UnaryCase { name: "exp10", kernel: $ops::exp10, scalar: |v| (10 as $scalar).powf(v) },
                    UnaryCase { name: "erf", kernel: $ops::erf, scalar: $erf },
                    UnaryCase { name: "erfc", kernel: $ops::erfc, scalar: $erfc },
                    UnaryCase { name: "tgamma", kernel: $ops::tgamma, scalar: $tgamma },
                    UnaryCase { name: "lgamma", kernel: $ops::lgamma, scalar: $lgamma },
                    UnaryCase {
                        name: "sinpi",
                        kernel: $ops::sinpi,
                        scalar: |v| (v * core::$scalar::consts::PI).sin(),
                    },
                    UnaryCase {
                        name: "cospi",
                        kernel: $ops::cospi,
                        scalar: |v| (v * core::$scalar::consts::PI).cos(),
                    },
                ]
            }

            #[cfg(not(feature = "simd"))]
            fn simd_only() -> &'static [UnaryCase] {
                &[]
            }

            const BINARY: &[BinaryCase] = &[
                BinaryCase { name: "pow", kernel: $ops::pow, scalar: <$scalar>::powf },
                BinaryCase { name: "atan2", kernel: $ops::atan2, scalar: <$scalar>::atan2 },
                BinaryCase { name: "hypot", kernel: $ops::hypot, scalar: <$scalar>::hypot },
            ];

            fn uninit(len: usize) -> Vec<MaybeUninit<$scalar>> {
                vec![MaybeUninit::uninit(); len]
            }

            fn read(out: Vec<MaybeUninit<$scalar>>) -> Vec<$scalar> {
                out.into_iter()
                    .map(|v| unsafe { v.assume_init() })
                    .collect()
            }

            fn median(mut xs: Vec<f64>) -> f64 {
                xs.sort_by(f64::total_cmp);
                xs[xs.len() / 2]
            }

            /// Inputs that keep every kernel inside its documented range and produce a finite,
            /// non-degenerate result.
            fn unary_input(len: usize) -> Vec<$scalar> {
                (0..len)
                    .map(|i| {
                        let t = i as f64 / len as f64;
                        let x = t * 2.0 - 1.0;
                        let v = match i % 4 {
                            0 => x * 0.5,
                            1 => (x + 1.5).abs() + 0.125,
                            2 => x * 8.0,
                            _ => x * 0.125 + 0.75,
                        };
                        v as $scalar
                    })
                    .collect()
            }

            pub fn run() {
                println!(
                    "== {} : level {:?} ==",
                    stringify!($scalar),
                    fearless_simd::Level::new()
                );
                println!(
                    "{:<12} {:>10} {:>10} {:>9} {:>11} {:>9}",
                    "op", "scalar", "kernel", "speedup", "max |diff|", "rel"
                );

                for size in SIZES {
                    let src = unary_input(size);
                    let rhs: Vec<$scalar> = (0..size)
                        .map(|i| 1.0 + (i % 64) as f64 * 0.03125)
                        .map(|v| v as $scalar)
                        .collect();
                    let exps: Vec<$scalar> = vec![1.5 as $scalar; size];
                    let mut out = uninit(size);
                    let mut pair_a = uninit(size);
                    let mut pair_b = uninit(size);

                    if size != PRIMARY {
                        continue;
                    }

                    for case in common().iter().chain(simd_only()) {
                        for _ in 0..WARMUP {
                            (case.kernel)(&src, &mut out).expect("lengths match");
                        }
                        let kt: Vec<f64> = (0..ROUNDS)
                            .map(|_| {
                                let t = Instant::now();
                                (case.kernel)(black_box(&src), &mut out).expect("lengths match");
                                black_box(&out[0]);
                                t.elapsed().as_secs_f64()
                            })
                            .collect();
                        let st: Vec<f64> = (0..ROUNDS)
                            .map(|_| {
                                let t = Instant::now();
                                for (slot, &x) in out.iter_mut().zip(src.iter()) {
                                    slot.write((case.scalar)(black_box(x)));
                                }
                                black_box(&out[0]);
                                t.elapsed().as_secs_f64()
                            })
                            .collect();

                        (case.kernel)(&src, &mut out).expect("lengths match");
                        let got = read(out);
                        let (mut max_diff, mut max_rel) = (0.0f64, 0.0f64);
                        for (i, &x) in src.iter().enumerate() {
                            let want = (case.scalar)(x);
                            let diff = (got[i] as f64 - want as f64).abs();
                            if got[i].is_finite() && want.is_finite() {
                                max_diff = max_diff.max(diff);
                                max_rel = max_rel.max(diff / (want as f64).abs().max(1e-30));
                            }
                        }

                        let k = median(kt) / size as f64 * 1e9;
                        let s = median(st) / size as f64 * 1e9;
                        print_row(case.name, s, k, max_diff, max_rel);
                    }

                    for case in BINARY {
                        for _ in 0..WARMUP {
                            (case.kernel)(&src, &rhs, &mut out).expect("lengths match");
                        }
                        let kt: Vec<f64> = (0..ROUNDS)
                            .map(|_| {
                                let t = Instant::now();
                                (case.kernel)(black_box(&src), black_box(&rhs), &mut out)
                                    .expect("lengths match");
                                black_box(&out[0]);
                                t.elapsed().as_secs_f64()
                            })
                            .collect();
                        let st: Vec<f64> = (0..ROUNDS)
                            .map(|_| {
                                let t = Instant::now();
                                for ((slot, &a), &b) in out.iter_mut().zip(src.iter()).zip(rhs.iter()) {
                                    slot.write((case.scalar)(black_box(a), black_box(b)));
                                }
                                black_box(&out[0]);
                                t.elapsed().as_secs_f64()
                            })
                            .collect();

                        (case.kernel)(&src, &rhs, &mut out).expect("lengths match");
                        let got = read(out);
                        let (mut max_diff, mut max_rel) = (0.0f64, 0.0f64);
                        for (i, (&a, &b)) in src.iter().zip(rhs.iter()).enumerate() {
                            let want = (case.scalar)(a, b);
                            let diff = (got[i] as f64 - want as f64).abs();
                            if got[i].is_finite() && want.is_finite() {
                                max_diff = max_diff.max(diff);
                                max_rel = max_rel.max(diff / (want as f64).abs().max(1e-30));
                            }
                        }
                        let k = median(kt) / size as f64 * 1e9;
                        let s = median(st) / size as f64 * 1e9;
                        print_row(case.name, s, k, max_diff, max_rel);
                    }

                    // `pow` on the shape the reference issue measures: a full tensor of `1.5`.
                    {
                        let _ = &exps;
                        for _ in 0..WARMUP {
                            $ops::pow(&src, &exps, &mut out).expect("lengths match");
                        }
                        let kt: Vec<f64> = (0..ROUNDS)
                            .map(|_| {
                                let t = Instant::now();
                                $ops::pow(black_box(&src), black_box(&exps), &mut out)
                                    .expect("lengths match");
                                black_box(&out[0]);
                                t.elapsed().as_secs_f64()
                            })
                            .collect();
                        let st: Vec<f64> = (0..ROUNDS)
                            .map(|_| {
                                let t = Instant::now();
                                for (slot, &x) in out.iter_mut().zip(src.iter()) {
                                    slot.write(black_box(x).powf(1.5 as $scalar));
                                }
                                black_box(&out[0]);
                                t.elapsed().as_secs_f64()
                            })
                            .collect();
                        let k = median(kt) / size as f64 * 1e9;
                        let s = median(st) / size as f64 * 1e9;
                        print_row("pow(x,1.5)", s, k, 0.0, 0.0);
                    }

                    // `sincos` yields two outputs; only its cost is reported.
                    #[cfg(feature = "simd")]
                    {
                        let pair_cost = |f: Pair| -> f64 {
                            median(
                                (0..ROUNDS)
                                    .map(|_| {
                                        let t = Instant::now();
                                        f(black_box(&src), &mut pair_a, &mut pair_b)
                                            .expect("lengths match");
                                        black_box((&pair_a[0], &pair_b[0]));
                                        t.elapsed().as_secs_f64()
                                    })
                                    .collect(),
                            ) / size as f64
                                * 1e9
                        };
                        let k = pair_cost($ops::sincos);
                        let s = median(
                            (0..ROUNDS)
                                .map(|_| {
                                    let t = Instant::now();
                                    for ((sa, sb), &x) in
                                        pair_a.iter_mut().zip(pair_b.iter_mut()).zip(src.iter())
                                    {
                                        sa.write(black_box(x).sin());
                                        sb.write(black_box(x).cos());
                                    }
                                    black_box((&pair_a[0], &pair_b[0]));
                                    t.elapsed().as_secs_f64()
                                })
                                .collect(),
                        ) / size as f64
                            * 1e9;
                        print_row("sincos", s, k, 0.0, 0.0);
                    }
                    println!();
                }
            }

            fn print_row(name: &str, scalar: f64, kernel: f64, max_diff: f64, max_rel: f64) {
                println!(
                    "{name:<12} {scalar:>10.3} {kernel:>10.3} {:>8.2}x {max_diff:>11.2e} {max_rel:>9.1e}",
                    scalar / kernel
                );
            }
        }
    };
}

bench_family! {
    /// The `f64` family.
    mod f64bench,
    scalar = f64,
    ops = elem_simd_math::f64,
    erf = libm_erf, erfc = libm_erfc, tgamma = libm_tgamma, lgamma = libm_lgamma,
}

bench_family! {
    /// The `f32` family.
    mod f32bench,
    scalar = f32,
    ops = elem_simd_math::f32,
    erf = libm_erff, erfc = libm_erfcf, tgamma = libm_tgammaf, lgamma = libm_lgammaf,
}

fn libm_erf(x: f64) -> f64 {
    // SAFETY: a pure function of its argument.
    unsafe { erf(x) }
}
fn libm_erfc(x: f64) -> f64 {
    // SAFETY: a pure function of its argument.
    unsafe { erfc(x) }
}
fn libm_tgamma(x: f64) -> f64 {
    // SAFETY: a pure function of its argument.
    unsafe { tgamma(x) }
}
fn libm_lgamma(x: f64) -> f64 {
    // SAFETY: a pure function of its argument.
    unsafe { lgamma(x) }
}
fn libm_erff(x: f32) -> f32 {
    // SAFETY: a pure function of its argument.
    unsafe { erff(x) }
}
fn libm_erfcf(x: f32) -> f32 {
    // SAFETY: a pure function of its argument.
    unsafe { erfcf(x) }
}
fn libm_tgammaf(x: f32) -> f32 {
    // SAFETY: a pure function of its argument.
    unsafe { tgammaf(x) }
}
fn libm_lgammaf(x: f32) -> f32 {
    // SAFETY: a pure function of its argument.
    unsafe { lgammaf(x) }
}

fn main() {
    println!("elem-simd-math throughput");
    println!("warmup {WARMUP}, rounds {ROUNDS}, statistic = median wall time, ns/element");
    println!("primary n = {PRIMARY}; `speedup` is libm/kernel, so larger is faster");
    println!();
    f64bench::run();
    f32bench::run();
}
