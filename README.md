# elem-simd-math

Vectorized elementwise math for `f64`, on stable Rust, with runtime CPU feature detection.

The kernels are a port of [SLEEF](https://sleef.org/) as expressed by
[`sleef-rs`](https://github.com/burrbull/sleef-rs), rewritten against the stable-Rust SIMD
abstraction of [`fearless_simd`](https://crates.io/crates/fearless_simd). See `NOTICE` and
`THIRD-PARTY-LICENSES` for provenance and licensing.

## Usage

```rust
use core::mem::MaybeUninit;
use elem_simd_math::ElementwiseMath;

let src: Vec<f64> = (0..8).map(|i| i as f64 * 0.25 - 1.0).collect();
let mut dst: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); src.len()];

f64::exp(&src, &mut dst).unwrap();
// SAFETY: `exp` wrote every element before returning `Ok`.
let dst: Vec<f64> = unsafe { core::mem::transmute(dst) };
assert!((dst[4] - 0.0f64.exp()).abs() < 1e-12);
```

The destination is `&mut [MaybeUninit<T>]` so that callers can write straight into uninitialized
storage; every element is written exactly once before `Ok` is returned. A length mismatch is
reported as [`LengthMismatch`] and leaves the destination untouched.

If you already own a SIMD loop, the kernels are also available directly, generic over a
`fearless_simd::Simd` token, in `f64x`.

## Accuracy

Every kernel documents the error bound of its SLEEF accuracy tier — `u05` = 0.5 ULP, `u10` = 1.0
ULP, `u15` = 1.5 ULP, `u35` = 3.5 ULP. Those bounds are not estimates: the test suite ports
upstream's MPFR oracle and asserts the same bound upstream asserts for the same function.

Upstream chooses between an FMA and a non-FMA formulation at compile time. This port always uses
the non-FMA formulation, which is the one upstream ships by default and which stays within the
documented bound on every SIMD level, including levels without fused multiply-add. The
double-double layer is built only from split-based formulas and uses no fused operations at all.

## Scope

`f64` only for now. The kernels, the shared helper layer and the double-double layer are written so
that an `f32` family can be added alongside them without relayout: see the "Remaining work" section
of `HANDOFF.md`.

## Accuracy and speed

Every kernel carries the error bound of the SLEEF accuracy tier it came from (`u05` = 0.5 ULP,
`u10` = 1.0 ULP, `u15` = 1.5 ULP, `u35` = 3.5 ULP), and the ULP suite in `tests/ulp.rs` asserts
that bound against MPFR. The table below is the recorded run of `cargo bench`
(`docs/benchmark-results.md`): `n = 2^23` elements, one thread, `Avx2` (four `f64` lanes), baseline
= the platform's `libm` called one element at a time from the same binary. ★ marks the rows of
`tensor4all/tenferro-rs#2033`.

| function | ns/elem (libm) | ns/elem (kernel) | speedup | bound | #2033 |
|---|---:|---:|---:|---:|:--:|
| `sin (u35)` | 8.60 | 1.70 | **5.06x** | 3.5 ULP |  |
| `exp10` | 12.35 | 2.45 | **5.04x** | 1.09 ULP |  |
| `tanh (u35)` | 13.04 | 2.91 | **4.48x** | 3.5 ULP |  |
| `cos (u35)` | 8.72 | 2.09 | **4.17x** | 3.5 ULP |  |
| `sin` | 8.55 | 2.61 | **3.27x** | 1.0 ULP | ★ |
| `sincos` | 12.19 | 3.74 | **3.26x** | 1.0 ULP |  |
| `cos` | 8.69 | 2.97 | **2.93x** | 1.0 ULP | ★ |
| `tan` | 10.52 | 3.69 | **2.85x** | 1.0 ULP |  |
| `sinh` | 13.52 | 4.76 | **2.84x** | 1.0 ULP |  |
| `cbrt` | 17.99 | 6.36 | **2.83x** | 1.0 ULP |  |
| `log (u35)` | 5.12 | 1.81 | **2.83x** | 3.5 ULP |  |
| `log1p` | 7.84 | 2.82 | **2.78x** | 1.0 ULP | ★ |
| `atan2` | 15.05 | 5.66 | **2.66x** | 1.0 ULP |  |
| `log10` | 7.93 | 3.03 | **2.62x** | 1.0 ULP |  |
| `tanh` | 13.07 | 5.40 | **2.42x** | 1.0 ULP | ★ |
| `exp` | 5.25 | 2.20 | **2.39x** | 1.0 ULP | ★ |
| `atan` | 8.43 | 3.66 | **2.30x** | 1.0 ULP |  |
| `asin` | 6.59 | 2.95 | **2.23x** | 1.0 ULP |  |
| `cospi` | 9.38 | 4.30 | **2.18x** | 0.506 ULP |  |
| `sqrt` | 2.46 | 1.19 | **2.07x** | 0.5001 ULP |  |
| `sinpi` | 9.16 | 4.47 | **2.05x** | 0.506 ULP |  |
| `acos` | 6.60 | 3.27 | **2.02x** | 1.0 ULP |  |
| `exp2` | 3.79 | 1.96 | **1.94x** | 1.0 ULP |  |
| `log` | 5.18 | 2.67 | **1.94x** | 1.0 ULP | ★ |
| `hypot` | 6.54 | 3.51 | **1.86x** | 0.5001 ULP |  |
| `expm1` | 7.68 | 4.29 | **1.79x** | 1.0 ULP | ★ |
| `cosh` | 7.94 | 4.67 | **1.70x** | 1.0 ULP |  |
| `atanh` | 8.68 | 5.23 | **1.66x** | 1.0 ULP |  |
| `log2` | 4.41 | 2.83 | **1.56x** | 1.0 ULP |  |
| `asinh` | 12.37 | 9.97 | **1.24x** | 1.0 ULP |  |
| `tgamma` | 40.09 | 32.59 | **1.23x** | 1.0 ULP |  |
| `erf` | 8.91 | 8.91 | **1.00x** | 1.0 ULP |  |
| `acosh` | 7.34 | 8.34 | **0.88x** | 1.0 ULP |  |
| `pow` | 12.67 | 16.24 | **0.78x** | 1.0 ULP | ★ |
| `erfc` | 9.71 | 14.49 | **0.67x** | 2.5 ULP |  |
| `lgamma` | 20.79 | 33.54 | **0.62x** | 1.0 ULP |  |

`ns/elem` is a time, so smaller is faster; `speedup` is `libm / kernel`, so **larger is faster**.
`1.00x` means the kernel ties with the platform's `libm`.

### Which `libm`

`libm` is not one implementation, and the `speedup` column is only meaningful against the one the
host links:

| platform | math functions live in | license |
|---|---|---|
| Linux + glibc | `libm.so.6` (must link `-lm`) | LGPL-2.1-or-later |
| Linux + musl | `libc.so` (the separate `libm` is an empty compatibility stub) | MIT |
| macOS | `libSystem`; `/usr/lib/libm.dylib` is a stub, `-lm` is a no-op | Apple |
| Windows | UCRT (`ucrtbase.dll`) | Microsoft |
| Julia | a bundled **OpenLibm** | BSD + ISC |
| Rust `std` | whatever the C library above provides | — |
| Rust `libm` crate | pure Rust, identical everywhere | MIT |

So the numbers above are against **glibc 2.39**; another platform's `libm` will score differently,
and a portable, reproducible baseline means depending on the `libm` crate instead. On this host
glibc's `pow` measures 0.504 ULP — effectively correctly rounded — while its `erf`, `erfc`,
`lgamma` and `tgamma` measure 3.15, 3.15, 2.24 and 4.60 ULP, i.e. *worse* than the kernels here.

### The rows that are slower than `libm`

`pow`, `erfc`, `lgamma`, `acosh`, `erf` and `tgamma` are at or below `1.00x`. They are not
translation defects: measured in one binary against upstream `sleef-rs`, this port is faster than
its source on every one of them (`pow` 1.40x, `erf` 1.92x, `erfc` 2.04x, `tgamma` 1.56x,
`lgamma` 1.50x). Two different reasons are behind the gap:

* **`pow`** — glibc's implementation is both faster *and* more accurate (0.504 ULP against SLEEF's
  1.0), and SLEEF offers no cheaper tier for it. A vectorised port of the FDLIBM `pow` was also
  written and measured here and lost as well (0.59x of glibc); see `docs/findings-pow.md`. Keep the
  platform `libm` for `pow`. The reference benchmark's `pow` workload — a constant exponent of
  1.5 — is far better served by `x * sqrt(x)`, which measures 0.70 ns/elem.
* **`erfc`, `lgamma`, `acosh`, `erf`** — SLEEF deliberately spends time to beat glibc's accuracy
  (`erf` 0.75 ULP against glibc's 3.15). Matching glibc's speed here would mean giving up that
  accuracy, which this port does not do.

## Features

- `simd` (default): enable the runtime-dispatched SIMD kernels. Without it the entry points fall
  back to the platform's scalar transcendental functions. Functions with no scalar equivalent in
  the standard library (`erf`, `erfc`, `tgamma`, `lgamma`, `sinpi`, `cospi`, `sincospi`) are only
  available with `simd`.

## Benchmarks in this repository

| what | code | recorded result |
|---|---|---|
| kernels against the host's `libm`, with a per-case correctness gate | `benches/throughput.rs` | `docs/benchmark-results.md` |
| the host's `libm` against the pure-Rust `libm` crate, speed and ULP | `benches/portable_libm.rs` | `docs/portable-libm-results.md` |
| the protocol both runs follow (baseline, host, statistics, acceptance) | — | `docs/benchmark-protocol.md` |

```bash
cargo bench -j 16 --bench throughput      # the table above
cargo bench -j 16 --bench portable_libm   # which libm, and at what cost
```

Both harnesses are single-threaded, deterministic, and read their inputs from a fixed seed, so
the tables are reproducible. `ELEM_SIMD_MATH_ULP_ITERS` scales the ULP sampling in the test suite.

## Testing

```bash
cargo test
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The ULP suite samples `ELEM_SIMD_MATH_ULP_ITERS` inputs per function (default 1500) at lengths
below, at and above the SIMD lane width.
