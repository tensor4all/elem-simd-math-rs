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

Upstream chooses between an FMA and a split formulation at compile time. This port always uses the
FMA formulation, the one upstream also states its bounds for, and evaluates every fused operation
through `SimdFloat::mul_add_precise`. That rounds the exact product-plus-add once on *every* level —
the hardware FMA instruction where the level has one, a correctly rounded emulation where it does
not — so a level without FMA computes the same values as one with it, only more slowly. The
`f64` kernels were then compared against upstream's `+fma` build bit for bit; `HANDOFF.md` records
which functions and over how many inputs.

## Scope

`f64` and `f32`, over the whole elementwise analytic surface that SLEEF's `u05`/`u10`/`u15`/`u35`
tiers cover. `f32` needed no separate integration: the kernels, the shared helper layer and the
double-double layer are generic over the vector type, so `src/f32x` mirrors `src/f64x` and a single
macro generates both slice APIs.

## Accuracy and speed

Every kernel carries the error bound of the SLEEF accuracy tier it came from (`u05` = 0.5 ULP,
`u10` = 1.0 ULP, `u15` = 1.5 ULP, `u35` = 3.5 ULP), and the ULP suites assert that bound against
MPFR — `tests/ulp.rs` for `f64` and `tests/ulp32.rs` for `f32` — using the same bounds upstream
asserts for the same functions. The two tables below are the recorded run of
`cargo bench` (`docs/benchmark-results.md`): `n = 2^23` elements, one thread, `Avx2`, so four `f64`
lanes and eight `f32` lanes. The baseline is the platform's `libm` called one element at a time
from the same binary, and ★ marks the rows of `tensor4all/tenferro-rs#2033`.

### `f64`, 4 lanes

| function | ns/elem (libm) | ns/elem (kernel) | speedup | bound | #2033 |
|---|---:|---:|---:|---:|:--:|
| `sin (u35)` | 8.52 | 1.62 | **5.27x** | 3.5 ULP |  |
| `exp10` | 11.95 | 2.27 | **5.26x** | 1.09 ULP |  |
| `sincos` | 18.03 | 3.61 | **5.00x** | 1.0 ULP |  |
| `cos (u35)` | 8.66 | 1.88 | **4.61x** | 3.5 ULP |  |
| `tanh (u35)` | 12.64 | 2.86 | **4.43x** | 3.5 ULP |  |
| `tan (u35)` | 10.21 | 2.37 | **4.31x** | 3.5 ULP |  |
| `sin` | 8.52 | 2.56 | **3.32x** | 1.0 ULP | ★ |
| `cos` | 8.66 | 2.94 | **2.94x** | 1.0 ULP | ★ |
| `log10` | 7.72 | 2.65 | **2.91x** | 1.0 ULP |  |
| `atan2` | 14.78 | 5.13 | **2.88x** | 1.0 ULP |  |
| `sinh` | 13.17 | 4.62 | **2.85x** | 1.0 ULP |  |
| `tan` | 10.18 | 3.60 | **2.82x** | 1.0 ULP |  |
| `log (u35)` | 4.75 | 1.70 | **2.79x** | 3.5 ULP |  |
| `cbrt` | 16.98 | 6.17 | **2.75x** | 1.0 ULP |  |
| `log1p` | 7.48 | 2.81 | **2.66x** | 1.0 ULP | ★ |
| `asin` | 6.55 | 2.63 | **2.49x** | 1.0 ULP |  |
| `tanh` | 12.65 | 5.30 | **2.39x** | 1.0 ULP | ★ |
| `atan` | 8.37 | 3.53 | **2.37x** | 1.0 ULP |  |
| `exp` | 4.93 | 2.11 | **2.33x** | 1.0 ULP | ★ |
| `cospi` | 9.12 | 4.29 | **2.13x** | 0.506 ULP |  |
| `acos` | 6.56 | 3.17 | **2.07x** | 1.0 ULP |  |
| `sqrt` | 2.46 | 1.19 | **2.07x** | 0.5001 ULP |  |
| `sinpi` | 9.01 | 4.46 | **2.02x** | 0.506 ULP |  |
| `exp2` | 3.31 | 1.78 | **1.86x** | 1.0 ULP |  |
| `hypot` | 6.42 | 3.46 | **1.86x** | 0.5001 ULP |  |
| `log` | 4.74 | 2.61 | **1.81x** | 1.0 ULP | ★ |
| `expm1` | 7.20 | 4.25 | **1.69x** | 1.0 ULP | ★ |
| `cosh` | 7.40 | 4.55 | **1.63x** | 1.0 ULP |  |
| `atanh` | 7.91 | 5.05 | **1.57x** | 1.0 ULP |  |
| `log2` | 4.17 | 2.76 | **1.51x** | 1.0 ULP |  |
| `tgamma` | 40.13 | 32.41 | **1.24x** | 1.0 ULP |  |
| `asinh` | 12.05 | 9.92 | **1.21x** | 1.0 ULP |  |
| `erf` | 8.73 | 8.82 | **0.99x** | 1.0 ULP |  |
| `acosh` | 7.36 | 8.24 | **0.89x** | 1.0 ULP |  |
| `pow` | 12.08 | 16.36 | **0.74x** | 1.0 ULP | ★ |
| `pow(x,1.5)` | 11.46 | 16.22 | **0.71x** | 1.0 ULP |  |
| `erfc` | 9.46 | 14.07 | **0.67x** | 2.5 ULP |  |
| `lgamma` | 20.70 | 33.90 | **0.61x** | 1.0 ULP |  |

### `f32`, 8 lanes

| function | ns/elem (libm) | ns/elem (kernel) | speedup | bound |
|---|---:|---:|---:|---:|
| `tan (u35)` | 11.61 | 0.63 | **18.36x** | 3.5 ULP |
| `tanh (u35)` | 13.84 | 0.91 | **15.28x** | 3.5 ULP |
| `sinh` | 14.64 | 2.09 | **7.00x** | 1.0 ULP |
| `sin (u35)` | 3.19 | 0.47 | **6.84x** | 3.5 ULP |
| `asin` | 6.99 | 1.02 | **6.81x** | 1.0 ULP |
| `tan` | 11.59 | 1.78 | **6.50x** | 1.0 ULP |
| `cos (u35)` | 3.40 | 0.55 | **6.20x** | 3.5 ULP |
| `log1p` | 7.40 | 1.20 | **6.15x** | 1.0 ULP |
| `log10` | 7.14 | 1.18 | **6.04x** | 1.0 ULP |
| `tanh` | 13.84 | 2.40 | **5.78x** | 1.0001 ULP |
| `exp2` | 3.00 | 0.57 | **5.28x** | 1.0 ULP |
| `exp10` | 5.08 | 0.97 | **5.26x** | 1.0 ULP |
| `exp` | 3.01 | 0.58 | **5.20x** | 1.0 ULP |
| `acos` | 6.68 | 1.30 | **5.14x** | 1.0 ULP |
| `log (u35)` | 3.22 | 0.68 | **4.78x** | 3.5 ULP |
| `atan2` | 11.67 | 2.48 | **4.70x** | 1.0 ULP |
| `expm1` | 8.38 | 1.92 | **4.36x** | 1.0 ULP |
| `sincos` | 6.16 | 1.44 | **4.28x** | 1.0 ULP |
| `cosh` | 8.05 | 1.93 | **4.17x** | 1.0 ULP |
| `atan` | 5.99 | 1.51 | **3.96x** | 1.0 ULP |
| `sqrt` | 1.41 | 0.36 | **3.96x** | 0.5 ULP |
| `atanh` | 7.82 | 2.13 | **3.67x** | 1.0001 ULP |
| `tgamma` | 32.58 | 9.33 | **3.49x** | 1.0 ULP |
| `asinh` | 12.00 | 3.95 | **3.04x** | 1.0001 ULP |
| `log` | 3.22 | 1.14 | **2.82x** | 1.0 ULP |
| `sin` | 3.19 | 1.21 | **2.63x** | 1.0 ULP |
| `log2` | 3.22 | 1.24 | **2.61x** | 1.0 ULP |
| `erf` | 8.88 | 3.58 | **2.48x** | 0.75 ULP |
| `cospi` | 3.63 | 1.46 | **2.48x** | 0.506 ULP |
| `cos` | 3.39 | 1.39 | **2.44x** | 1.0 ULP |
| `acosh` | 7.87 | 3.39 | **2.32x** | 1.0001 ULP |
| `sinpi` | 3.42 | 1.49 | **2.30x** | 0.506 ULP |
| `hypot` | 3.36 | 1.61 | **2.08x** | 0.5 ULP |
| `erfc` | 9.51 | 4.99 | **1.91x** | 1.5 ULP |
| `cbrt` | 4.96 | 2.75 | **1.81x** | 1.0 ULP |
| `lgamma` | 15.83 | 9.75 | **1.62x** | 1.0 ULP |
| `pow` | 5.68 | 4.93 | **1.15x** | 1.0 ULP |
| `pow(x,1.5)` | 5.30 | 4.93 | **1.08x** | 1.0 ULP |
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
| kernels against the host's `libm`, with per-case accuracy columns | `benches/throughput.rs` | `docs/benchmark-results.md` |
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
