# Phase C benchmark protocol

Recorded **before** collecting the numbers, per the Performance-Gated Experiment Protocol of
`tenferro-rs/PERFORMANCE_TIPS.md`. The `real.*` rows of `tensor4all/tenferro-rs#2033` motivate the
primary case size.

## Question

Does the ported SIMD kernel set reduce the per-element cost of the `f64` and `f32` elementwise
analytic operations relative to the platform's scalar transcendental functions, by enough to
matter, at the sizes the issue reports?

## Configuration

| item | value |
|---|---|
| baseline | the platform libm, called one element at a time from the same binary (`f64::exp`/`f32::exp`, …; for `erf`/`erfc`/`tgamma`/`lgamma`, the libm symbols reached through `extern "C"`) |
| candidate | the crate's slice entry points, which select the best `fearless_simd` level at runtime |
| harness | `benches/throughput.rs`, `cargo bench`, `harness = false`, `bench` profile (release plus debug line tables) |
| build | `cargo bench -j 16`, one benchmark family per scalar type; default features (`simd`), `bench` profile (release plus debug line tables, so backtraces stay readable while the numbers match the release build) |
| thread count | 1. The kernels are serial and the harness spawns no threads; `RAYON_NUM_THREADS=1` is exported anyway so that a future parallel path is visibly pinned |
| host | AMD EPYC 7713P (Zen 3), 64 cores, Linux, no CUDA. `Level::new()` selects `Avx2`, so the `f64` vector width is 4 lanes (256-bit) and the `f32` width is 8; the harness prints the level at startup |
| affinity | `taskset -c 2`, so the two families are measured on the same core |
| case list | every entry point in `src/dispatch.rs` for both `f64` and `f32` that the harness can time against a scalar arm, at `n = 2^23` |
| statistic | median of 7 timed rounds per arm, after 2 warmups, reported as ns/element |
| correctness columns | the harness reports `max abs(kernel - scalar)` and the maximum relative difference per case for the reader; the binding accuracy gate is the MPFR suite in `tests/ulp.rs` / `tests/ulp32.rs`, since a relative difference is meaningless next to a zero or a pole (see `docs/benchmark-results.md`) |
| acceptance | at `n = 2^23` the candidate improves by at least 1.5x over the scalar arm on `exp`, `log`, `sin`, `cos`, `tanh`, `expm1`, `log1p` and `pow`, and no case is slower than the scalar arm |

## Result

Both families are now measured at `n = 2^23`, single thread, `Avx2`, against the platform libm.
`docs/benchmark-results.md` holds the full output and the per-row tables.

**`f32` fails it on `pow`, by a much smaller margin than `f64`.** `exp` 5.27x, `log` 2.80x, `sin`
2.76x, `cos` 3.04x, `tanh` 5.94x, `expm1` 4.41x and `log1p` 5.99x all clear 1.5x, but `pow` is
**1.14x**, under the threshold — although unlike `f64` it is *faster* than the scalar arm rather
than slower. No case at all is slower on `f32`; the least favourable rows are `lgamma` at 1.63x
and `cbrt` at 1.82x. The `u35` tier is the fastest part of the surface, from 7.7x to 18.7x.

**`f64` fails it on `pow` alone.** `exp` 2.44x, `log` 1.91x, `sin` 3.29x, `cos` 2.93x, `tanh`
2.43x, `expm1` 1.78x and `log1p` 2.75x all clear 1.5x, but `pow` is **0.75x**, and the no-slower
clause is broken by `pow` (0.75x) and `pow` with a constant exponent (0.71x), `erfc` (0.67x) and
`lgamma` (0.61x).

The `pow` result is understood rather than tolerated. glibc's `pow` measures 0.504 ULP, better than
the kernel's 1.0, and on `f64` it is faster too, so the vector path loses on both axes there and is
not worth taking. On `f32` the kernel is the faster arm but the less accurate one, a trade the
consumer can only make with its own accuracy budget in hand. `docs/findings-pow.md` records the measurements; `docs/findings-tgamma.md`
records the same trade-off in the other direction for `tgamma` (1.24x here, 4.6 ULP for glibc).
The other three slow rows (`erfc`, `lgamma`, and `pow` with a constant exponent) are double-double
heavy or already correctly rounded, and each beats the scalar arm on accuracy.

The gate is reported as failed for `f64`, not relaxed. The reference benchmark's `pow` workload
uses a constant exponent of 1.5, where `x * sqrt(x)` measures 0.70 ns/element — about 20x faster
than `powf` and within 1 ULP — so the consumer fixes that row by not calling `pow` at all.

### Earlier run, before the FMA switch

The first complete run, in this file's history, failed the gate on most of the surface (`exp`
2.12x, `log` 1.42x, `sin` 2.06x, `cos` 1.78x, `tanh` 1.60x, `expm1` 1.31x, `log1p` 1.94x, `pow`
0.61x). The diagnosis was that the port evaluated every `mla` with two roundings and every
double-double formula in the split (non-FMA) form. Switching to upstream's FMA formulation, with a
guaranteed single-rounding `mul_add`, is what produced the numbers above; upstream asserts the same
error bounds for that build, and `f64` was then verified bit-for-bit identical to it.

## Limitations stated up front

- The reference arms of the issue (PyTorch, JAX, Julia) are **not** reproduced here: their
  devcontainer, MKL and Python stack are out of scope for this crate. The numbers are a paired
  baseline/candidate comparison inside one binary, not a ratio against PyTorch.
- Only the single-thread configuration is measured. Multi-thread scaling is separate work.
- `sinpi`/`cospi`'s scalar arm computes `sin(pi*x)` directly, which is *less* accurate than the
  kernel; it exists only to show the cost of the vector path, and its relative-difference column
  is expected to be large for huge arguments.
