# elem-simd-math — design, provenance and work log

Reusable vector math kernels for `f32`/`f64`, ported from SLEEF through `sleef-rs` and expressed
against `fearless_simd`.

## Decisions (agreed)

1. **Separate crate**, repo `tensor4all/elem-simd-math-rs`, rather than kernels inside
   `strided-rs`. `tenferro-rs` pins `strided-rs` 0.4.4 (rev `25705339…`) while
   `tlinalg-rs`/`tprims-rs` pin 0.4.5 (rev `c12d96fa…`), so a `StridedView` from a different rev is
   a different type at those boundaries. A crate whose API is `&[f32]`/`&[f64]` avoids that
   cascade. This is recorded as an ownership exception for a reusable kernel crate.
2. **SIMD layer: `fearless_simd` 1.1, not `pulp`.** `pulp`'s generic `Simd` trait has 173 methods
   and lacks numeric float↔int conversion (only bit reinterpretation), 64-bit shifts, and
   `round`/`trunc`/`floor`/`ceil`. `sleef-rs` uses those 231/98/… times; on `pulp` the kernels
   would have to be rewritten rather than translated, which is where accuracy drifts.
   `fearless_simd` provides a `core::simd`-shaped stable API with runtime level dispatch.
3. **Accuracy is not relaxed.** The crate keeps SLEEF's tiers and ports upstream's MPFR ULP
   assertions verbatim into `tests/ulp.rs`.
4. **Upstream's non-FMA formulation everywhere.** Upstream chooses a coefficient set at compile
   time with `cfg(target_feature = "fma")`; the default (and the one this port takes) is the
   formulation that stays within the documented bound without fused multiply-add, which makes it
   valid on every `fearless_simd` level. The double-double layer (`src/doubled.rs`) is likewise
   built only from split-based formulas and contains no fused operations. Results are therefore
   *within the same bound* on every level, but are not bit-identical across levels.
5. **Destination is `&mut [MaybeUninit<T>]`.** Callers such as `tenferro-cpu` own pooled
   uninitialized buffers and must not zero-fill a buffer that will be fully overwritten.
6. **Provenance.** Translation of `sleef-rs` rev `3671d46` (MIT OR Apache-2.0), which is itself a
   port of SLEEF (Boost-1.0). Package SPDX `(MIT OR Apache-2.0) AND BSL-1.0`; see `NOTICE` and
   `THIRD-PARTY-LICENSES`.

## Translation rules

Extension traits cannot be implemented for `S::f64s`: the projection is not a local type
(`E0207` "unconstrained type parameter", `E0119` conflicting impls). `<S as Simd>::f64s` is also
not invertible in argument position. Therefore:

* every free helper takes the SIMD token as its **first** argument, so call sites infer `S`
  without a turbofish;
* `sleef-rs`'s method-style extension traits become free functions.

| `sleef-rs` | here |
|---|---|
| `F64x<N>` | `F64s<S>` (`crate::common`) |
| `x.mla(y, z)` | `mla(simd, x, y, z)` or `x.mul_add(y, z)` |
| `F64x::splat(v)` | `splat(simd, v)` |
| `F64x::polyN(…)` | `polyN(simd, …)` |
| `x.to_bits()` / `F64x::from_bits(b)` | `to_bits(simd, x)` / `from_bits(simd, b)` |
| `m.select(l, r)` | `m.select(l, r)` (unchanged) |
| `m.select_splat(l, r)` | `select_splat(simd, m, l, r)` |
| `x.roundi()` / `x.trunci()` | `roundi(simd, x)` / `trunci(simd, x)` |
| `.round()` on a float vector | `rint(simd, x)` (upstream's `Round::round` is ties-to-even) |
| `Doubled<T>` | `crate::doubled::f64::DD<S>` |
| `d + v`, `d * v` | `add_v(d, v)`, `mul_v(d, v)` (free functions; the operators would conflict) |
| `v + d` | `v_add_dd(simd, v, d)` |
| `m.to_int().cast::<u64>()` and friends | `m.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))` |

## Deliberate deviations

* **Integer lanes are 64-bit.** Upstream uses `Simd<i32, N>`/`Simd<u32, N>` for quadrants and bit
  patterns alongside `Simd<f64, N>`. This port uses the native-width `S::i64s`/`S::u64s`, which
  have the same lane count as `S::f64s`; shift widths and masks are adjusted accordingly (e.g. a
  logical right shift of the sign is `>> 63` on `u64` where upstream writes `>> 31` on `u32`).
* **AVX-512-only branches are not ported.** Upstream guards alternate code with
  `cfg(feature = "enable_avx512f")` and block comments; those branches are skipped, so the
  corresponding literals (e.g. `1024.`) do not appear here.
* **Compile-time FMA branches**: the non-FMA formulation only (see decision 4).
* **`full_fp_rounding` cfg branches**: the default (non-`full_fp_rounding`) branch only.
* **The `Sleef` convenience trait is not ported.** It only forwards to the tier functions and
  adds no algorithm. `frexp`/`ldexp`/`ilogb`/`nextafter`/`fmod`/`remainder`/`modf`/`fdim`/
  `fma`/`floor`/`ceil`/`round`/`trunc` are ported as free functions; `max`/`min`/`abs`/
  `copysign` are one-liners over operations `fearless_simd` already exposes.
* **`sinpik` lives in `src/f64x/u05.rs`.** Upstream keeps it in `f64x.rs`, but only `u05` uses it.
* **Scalar fallbacks.** Without the `simd` feature the slice API calls the platform's scalar
  functions. `erf`, `erfc`, `tgamma`, `lgamma`, `sinpi`, `cospi` and `sincospi` have no standard
  library equivalent and are therefore only available with `simd`; providing an inaccurate
  fallback would contradict their documented bounds.

## Upstream → local coverage

Function-level check over the upstream non-comment bodies (generated, not hand-written):

| upstream file | upstream functions (excl. `test_*`) | not present here by name |
|---|---|---|
| `src/f64x.rs` | 95 | 45 — all explained below |
| `src/f64x/u05.rs` | 6 | 0 |
| `src/f64x/u10.rs` | 40 | 0 |
| `src/f64x/u15.rs` | 1 | 0 |
| `src/f64x/u35.rs` | 25 | 0 |
| `src/common.rs` | trait-only | n/a — see below |
| `doubled` v0.3.3 | `src/lib.rs`, `src/f64x.rs` | 0 |

The 45 names from `src/f64x.rs` that are not present by name are:

* the `impl Sleef for F64x` forwarding methods — `sin`, `cos`, `sin_cos`, `tan`, `asin`, `acos`,
  `atan`, `atan2`, `ln`, `cbrt`, `exp`, `pow`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
  `exp2`, `exp10`, `exp_m1`, `log10`, `log2`, `log_1p`, `sqrt`, `abs`, `max`, `min`, `truncate`,
  `next_after`, `sin_cos_pi`, `sin_pi`, `cos_pi`, `hypot`, `gamma`, `lgamma`, `erf`, `erfc`. Each
  forwards to a tier function that is ported here (`f64x::u10::sin`, …); `abs`, `max`, `min` and
  `copysign` are one-liners over operations `fearless_simd` already exposes.
* helpers that were folded into the 64-bit-lane versions rather than kept as separate names:
  `cast_into_upper` and `cast_from_upper` (inside `pow2i`/`ldexp3k`/`ilogb2k`), `c2v` (inside every
  `polyN`), `rintk2` (identical to `rint`), `visinf2_vd_vd_vd` → `visinf2`, `copysign` →
  `copy_sign`, `expk2` → `u10::expk2`, `sinpik` → `u05::sinpik`.

`src/common.rs` upstream holds extension traits whose only purpose is to attach methods to
`Simd<f64, N>`; extension traits cannot be implemented for a projection here (`E0207`/`E0119`), so
each becomes a free function in `src/f64x/mod.rs` or `src/doubled.rs` — `Round`, `MulAdd`, `MulSub`,
`NegMulAdd`, `Sign`, `IsInt`, `IsOdd`, `IsNegZero`, `SqrtAsDoubled`, `VectorizedSelect`,
`DoubledSelect`, `SelectSeveral`, `Poly`, `Constants`, `MaskType`, `BitsType`, `MaskToInt`, and the
`cons!` macro (whose constants are now plain `const`s in `src/constants.rs`).

Upstream's `#[test]` functions are replaced by `tests/ulp.rs` (one test per function, asserting
upstream's bound) and `tests/helpers.rs` (the shared helpers).

## Layout

```
src/lib.rs          crate docs
src/common.rs       F64s/F32s/U64s/… aliases
src/constants.rs    scalar constants from sleef-rs src/f64.rs, src/f32.rs
src/tables.rs       REMPITABDP, copied verbatim
src/doubled.rs      DoubleFloat over native-width vectors, both dtypes via dd_family!
src/f64x/mod.rs     f64 helper layer (rounding, ldexp/ilogb/frexp, rempi, fmod, remainder, modf,
                    fma, poly2..poly21)
src/f64x/u05.rs     sinpi, cospi, sincospi, sqrt, hypot
src/f64x/u10.rs     the 1.0 ULP tier
src/f64x/u15.rs     erfc
src/f64x/u35.rs     the 3.5 ULP tier
src/f32x/           not started (Phase D, outside this goal)
src/dispatch.rs     ElementwiseMath trait + slice entry points
tests/ulp.rs        MPFR ULP suite, upstream's bounds
tests/scalar.rs     the --no-default-features path
benches/throughput.rs  paired benchmark with a correctness gate
docs/benchmark-protocol.md  predeclared Phase C protocol
```

## Measurements (n = 2^23, 1 thread, Avx2 = 4 f64 lanes, vs the platform libm)

Full output in `docs/benchmark-results.md`; protocol in `docs/benchmark-protocol.md`.

| op | speedup | op | speedup | op | speedup |
|---|---:|---|---:|---|---:|
| sin | 3.34x | log1p | 2.63x | sqrt | 2.07x |
| sin (u35) | 4.87x | tanh | 2.43x | sinpi | 2.07x |
| exp10 | 5.14x | exp | 2.40x | atan2 | 2.60x |
| cos (u35) | 4.36x | tan | 2.76x | tanh (u35) | 4.31x |
| log10 | 2.85x | cos | 2.90x | hypot | 1.86x |
| log1p | 2.63x | expm1 | 1.78x | tgamma | 1.22x |
| log2 | 1.51x | asinh | 1.24x | cbrt | 2.65x |
| **pow** | **0.75x** | **erf** | **0.95x** | **erfc** | **0.68x** |
| **acosh** | **0.88x** | **lgamma** | **0.62x** | | |

Every one of those sub-1x cases is faster than upstream per element (measured in one binary
against `sleef-rs`'s FMA build): `pow` 1.40x, `erf` 1.92x, `erfc` 2.04x, `tgamma` 1.56x,
`lgamma` 1.50x. They are faithful ports that are simply slower than glibc's own implementations
on this host, which is not something a port can change. The protocol's "no case slower than the
scalar arm" clause is therefore unachievable for a faithful port, and is reported as failed rather
than relaxed; see the status section.

## Status (2026-10-11)

Phase A and Phase B are done: the crate builds, tests and lints clean on both feature sets, the
whole `f64` surface is ported, and the kernels are **bit-for-bit identical to upstream's FMA build**
(`sleef-rs` compiled with `-C target-feature=+fma`) over 20 000 samples per function for
`exp`, `expm1`, `tanh`, `log`, `sin`, `exp2`, `exp10` and `cbrt`. `tests/ulp.rs` asserts upstream's
bounds and passes for all 42 cases.

Phase C has been measured and recorded. The predeclared gate is **not met in full**: every
#2033 row except `pow` exceeds 1.5x, and `pow` is 0.75x against libm (though 1.40x faster than
upstream). See the measurements table above; the gate is reported as failed, not relaxed.

## Remaining work

1. `f32x` family (Phase D, outside this goal).
2. Parallel chunking of the slice drivers inside a caller-owned pool; the drivers are serial and
   the caller decides threading.
3. `REMPITABSP` and the `f32` constants land with the `f32` family.
