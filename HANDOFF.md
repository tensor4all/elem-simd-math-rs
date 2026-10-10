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

Function-level check (upstream non-comment function bodies vs. this crate):

| upstream file | upstream fns | not present locally |
|---|---|---|
| `src/f64x.rs` | 114 (incl. the `Sleef` trait forwarding methods) | only the forwarding trait methods and the AVX-512-only/commented-out helpers; see "Deliberate deviations" |
| `src/f64x/u05.rs` | see below | — |
| `src/f64x/u10.rs` | 40 | 0 |
| `src/f64x/u15.rs` | see below | — |
| `src/f64x/u35.rs` | 25 | 0 |

A literal-level check over the non-comment bodies finds only two systematic differences:

* `u10::exp` — upstream carries an FMA coefficient set and a non-FMA set; this port keeps the
  non-FMA set (decision 4), so the ten FMA-only coefficients are absent by design.
* `u35::sincospi` — the literal `31.0` is the `u32` logical shift width, written as `63` here
  because the integer lanes are 64-bit.

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

## Remaining work

1. `f32x` family (Phase D, outside this goal).
2. Parallel chunking of the slice drivers inside a caller-owned pool; the drivers are serial and
   the caller decides threading.
3. `REMPITABSP` and the `f32` constants land with the `f32` family.
