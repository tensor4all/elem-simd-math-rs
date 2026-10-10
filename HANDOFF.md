# elem-simd-math — design and work log

Status: **work in progress, does not compile yet.** This file is the handoff for the next session.

## Goal

Close the `real.exp / log / sin / cos / tanh / expm1 / log1p / pow` rows of
[`tensor4all/tenferro-rs#2033`](https://github.com/tensor4all/tenferro-rs/issues/2033) by
porting SLEEF's vector math to a reusable crate that tenferro can call from its
compact-contiguous analytic fast path.

## Decisions (agreed)

1. **Separate crate**, repo `tensor4all/elem-simd-math-rs`, rather than kernels inside
   `strided-rs`. `tenferro-rs` pins `strided-rs` 0.4.4 (rev `25705339…`) while
   `tlinalg-rs`/`tprims-rs` pin 0.4.5 (rev `c12d96fa…`); a `StridedView` from a different rev is a
   different type at those boundaries, so keeping the kernels behind a `&[f32]`/`&[f64]` API
   avoids a three-repository pin cascade. This is an ownership exception for a reusable kernel
   crate, not a proven unavoidable cascade.
2. **SIMD layer: `fearless_simd` 1.1, not `pulp`.** `pulp`'s generic `Simd` trait has 173
   methods and lacks the operations this port needs: numeric float↔int conversion (only bit
   reinterpretation), 64-bit variable/fixed shifts, and `round`/`trunc`/`floor`/`ceil`.
   `sleef-rs` uses those 140/91/42/25 times respectively; on `pulp` the kernels would have to be
   *rewritten* rather than translated, which is exactly where accuracy drifts. `fearless_simd`
   provides a `core::simd`-shaped stable API (verified: `bitcast`, `to_int`/`to_float`,
   `round_ties_even`/`trunc`, `shl`/`shr`/`shlv`/`shrv` for `i64`/`u64`, `select`, masks with
   `&`/`|`/`!`/`all_true`/`any_true`, `swizzle_dyn`, `Level` runtime dispatch).
3. **Scope: full C99 surface, all four accuracy tiers, `f32` and `f64`, before tenferro sees
   it.** A standalone `f64` benchmark is the intermediate checkpoint.
4. **Accuracy is not relaxed.** The port keeps SLEEF's tiers and *ports upstream's MPFR ULP
   assertions verbatim* (`rug` builds against the system GMP/MPFR in ~4 s here), so the bound is
   a test obligation rather than a claim.
5. **Provenance.** Translation of `sleef-rs` rev `3671d46` (MIT OR Apache-2.0) which is itself a
   port of SLEEF (Boost-1.0, `BSL-1.0`). Crate SPDX `(MIT OR Apache-2.0) AND BSL-1.0`, with
   `NOTICE` and `THIRD-PARTY-LICENSES`.
6. **Target-feature independence.** Upstream picks an FMA and a split formulation at compile
   time. This port always uses the split (non-FMA) formulation, so results do not depend on
   whether the build enabled FMA.

## Translation rules

Extension traits cannot be implemented for `S::f64s`: the projection is not a local type
(`E0207` "unconstrained type parameter", `E0119` conflicting impls). So the `sleef-rs`
method-style helpers become free functions.

| `sleef-rs` | here |
|---|---|
| `F64x<N>` | `F64s<S>` (`crate::common`) |
| `x.mla(y, z)` | `x.mul_add(y, z)` |
| `F64x::splat(v)` | `crate::f64x::splat(simd, v)` |
| `F64x::poly8(…)` | `poly8(simd, …)` |
| `x.to_bits()` / `F64x::from_bits(b)` | `to_bits(x)` / `from_bits(b)` |
| `m.select(l, r)` | `m.select(l, r)` (unchanged) |
| `m.select_splat(l, r)` | `select_splat(m, l, r)` |
| `x.roundi()` / `x.trunci()` | `roundi(simd, x)` / `trunci(simd, x)` |
| `Doubled<T>` | `crate::doubled::f64::DD<S>` |
| `d + v`, `d * v` | `add_v(d, v)`, `mul_v(d, v)` (operators on `DD` would conflict) |
| `m.to_int().cast::<u64>()` and friends | `m.select(splat_bits(simd, u64::MAX), splat_bits(simd, 0))` |

**Important**: `<S as Simd>::f64s` in argument position is not invertible, so every free helper
must either take the SIMD token as its first argument or be called with a turbofish
(`rint::<S>(…)`). The current code uses a mixture and does not compile; the next session should
adopt the **token-first signature everywhere** for consistency.

## Layout

```
src/lib.rs          crate docs
src/common.rs       F64s/F32s/U64s/… aliases
src/constants.rs    scalar constants from sleef-rs src/f64.rs, src/f32.rs
src/tables.rs       REMPITABDP / REMPITABSP, copied verbatim
src/doubled.rs      DoubleFloat over native-width vectors, dtypes via dd_family! macro
src/f64x/mod.rs     f64 helper layer (rint/trunc/floor/ceil/round, ldexp/ilogb, rempi, poly2..poly21)
src/f64x/u10.rs     exp, expm1, expk2  (log1p, pow, logk still missing)
src/f64x/u35.rs     log, tanh, expm1k  (sin, cos, sincos, tan, atan…, cbrt, hypot still missing)
src/f32x/           not started
src/dispatch.rs     ElementwiseMath trait + slice entry points (f64: exp/log/tanh/expm1)
benches/throughput.rs  criterion bench against the scalar libm
```

## Remaining work

1. Make the helper layer compile (token-first signatures), then `u35`, then `u10`.
2. Port `f64x/u05.rs`, `f64x/u15.rs`, and the rest of `u10`/`u35` (`pow`, `log1p`, `logk`, `sin`,
   `cos`, `sincos`, `tan`, `atan`, `atan2`, `asin`, `acos`, `sinh`, `cosh`, `cbrt`, `hypot`,
   `gamma`, `lgamma`, `erf`, `erfc`, `sinpi`, `cospi`, …).
3. Port the `f32x` family with the same shape.
4. Port `sleef-rs/src/f64x/tst.rs` + `src/tst_common` as the MPFR ULP harness and add one test
   per function with the upstream bound.
5. Run the standalone `f64` benchmark and record ns/element against libm.
6. `tenferro-rs` PR: dependency, `VectorKernel` threaded through `map_unary`, measurement, work
   log.

## Effort

Upstream function bodies to translate: `f64x` 4 526 lines, `f32x` 3 400 lines, plus `tables.rs`
1 082 (data, already copied) and `doubled` ≈ 450. Realistically **~10 000 lines** of new Rust,
of which ~90 % is mechanical. Two to three focused sessions, plus review and CI.
