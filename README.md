# elem-simd-math

Vectorized elementwise math for `f32` and `f64`, on stable Rust, with runtime CPU feature
detection.

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
`fearless_simd::Simd` token, in `f64x` and `f32x`.

## Accuracy

Every kernel documents the error bound of its SLEEF accuracy tier — `u05` = 0.5 ULP, `u10` = 1.0
ULP, `u15` = 1.5 ULP, `u35` = 3.5 ULP. Those bounds are not estimates: the test suite ports
upstream's MPFR oracle and asserts the same bound upstream asserts for the same function.

Upstream chooses between an FMA and a non-FMA formulation at compile time. This port always uses
the non-FMA formulation, which is the one upstream ships by default and which stays within the
documented bound on every SIMD level, including levels without fused multiply-add. The
double-double layer is built only from split-based formulas and uses no fused operations at all.

## Features

- `simd` (default): enable the runtime-dispatched SIMD kernels. Without it the entry points fall
  back to the platform's scalar transcendental functions. Functions with no scalar equivalent in
  the standard library (`erf`, `erfc`, `tgamma`, `lgamma`, `sinpi`, `cospi`, `sincospi`) are only
  available with `simd`.

## Testing

```bash
cargo test
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The ULP suite samples `ELEM_SIMD_MATH_ULP_ITERS` inputs per function (default 1500) at lengths
below, at and above the SIMD lane width.
