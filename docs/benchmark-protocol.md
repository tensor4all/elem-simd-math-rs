# Phase C benchmark protocol

Recorded **before** collecting the numbers, per the Performance-Gated Experiment Protocol of
`tenferro-rs/PERFORMANCE_TIPS.md`. The `real.*` rows of `tensor4all/tenferro-rs#2033` motivate the
primary case size.

## Question

Does the ported SIMD kernel set reduce the per-element cost of the `f64` elementwise analytic
operations relative to the platform's scalar transcendental functions, by enough to matter, at
the sizes the issue reports?

## Configuration

| item | value |
|---|---|
| baseline | the platform libm, called one element at a time from the same binary (`f64::exp`, …; for `erf`/`erfc`/`tgamma`/`lgamma`, the libm symbols reached through `extern "C"`) |
| candidate | the crate's slice entry points, which select the best `fearless_simd` level at runtime |
| harness | `benches/throughput.rs`, `cargo bench`, `harness = false`, release profile with `debug-assertions = true` |
| build | `cargo bench -j 16`; default features (`simd`) |
| thread count | 1. The kernels are serial and the harness spawns no threads; `RAYON_NUM_THREADS=1` is exported anyway so that a future parallel path is visibly pinned |
| host | AMD EPYC 7713P (Zen 3), 64 cores, Linux, no CUDA. `Level::new()` selects `Avx2`, so the `f64` vector width is 4 lanes (256-bit) and the `f32` width is 8; the harness prints the level at startup |
| affinity | none |
| case list | every entry point in `src/dispatch.rs`, at `n = 2^16`, `2^20` and `2^23`; `2^23` is the primary case |
| statistic | median of 7 timed rounds per arm, after 2 warmups, reported as ns/element |
| correctness gate | the harness reports `max abs(kernel - scalar)` and the maximum relative difference per case; a case whose relative difference exceeds the 3.5 ULP-equivalent relative error is a failure, not a speedup |
| acceptance | at `n = 2^23` the candidate improves by at least 1.5x over the scalar arm on `exp`, `log`, `sin`, `cos`, `tanh`, `expm1`, `log1p` and `pow`, and no case is slower than the scalar arm |

## Result (recorded 2026-10-11, first complete run)

`docs/benchmark-results.md` holds the full output. At `n = 2^23`, single thread, Avx2 (4 `f64`
lanes), the candidate wins on most of the surface but **the predeclared gate is not met yet**:
`exp` 2.12x, `log` 1.42x, `sin` 2.06x, `cos` 1.78x, `tanh` 1.60x, `expm1` 1.31x, `log1p` 1.94x,
`pow` 0.61x, and `asinh`/`acosh`/`erf`/`erfc`/`tgamma`/`lgamma`/`hypot` are at or below 1x.

Diagnosis: the port currently evaluates every `mla` with two roundings and every double-double
formula in the split (non-FMA) form, because `sleef-rs` documents its error bounds against its
default, non-FMA build. That is where the accuracy is, but it is also where the time goes: the
DD-heavy kernels (`pow`, `tgamma`, `lgamma`, `erfc`, `asinh`, `acosh`, `hypot`) pay several times
more floating-point operations than upstream's FMA build does. Closing the gate needs upstream's
FMA formulation (`cfg(target_feature = "fma")` branch) evaluated with a guaranteed single-rounding
`mul_add`, which upstream asserts the same bounds for.

## Limitations stated up front

- The reference arms of the issue (PyTorch, JAX, Julia) are **not** reproduced here: their
  devcontainer, MKL and Python stack are out of scope for this crate. The numbers are a paired
  baseline/candidate comparison inside one binary, not a ratio against PyTorch.
- Only the single-thread configuration is measured. Multi-thread scaling is separate work.
- `sinpi`/`cospi`'s scalar arm computes `sin(pi*x)` directly, which is *less* accurate than the
  kernel; it exists only to show the cost of the vector path, and its relative-difference column
  is expected to be large for huge arguments.
