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
4. **Upstream's FMA formulation everywhere, through `mul_add_precise`.** Upstream chooses a
   coefficient set at compile time with `cfg(target_feature = "fma")`. This port always takes the
   FMA set and evaluates every fused operation with `SimdFloat::mul_add_precise`, which rounds the
   exact product-plus-add once on every level — the hardware instruction where the level has one,
   a correctly rounded emulation where it does not. `SimdFloat::mul_add` is *not* usable here: it
   is documented as possibly two roundings, and on SSE2 and WebAssembly without `relaxed-simd` it
   is literally `a * b + c`, which zeroes the residual of every `mul_as_doubled` and collapses the
   double-double layer. With the precise variant a level without FMA computes the same values as
   one with it, so the kernels are bit-identical across levels, not merely within the same bound.
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
* **Compile-time FMA branches**: the FMA formulation, evaluated through
  `SimdFloat::mul_add_precise` (see decision 4).
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
| `src/f32x.rs` | 35 | 6 — all explained below |
| `src/f32x/constants.rs` | 0 | 0 |
| `src/f32x/fast.rs` | 3 | 0 |
| `src/f32x/u05.rs` | 5 | 0 |
| `src/f32x/u10.rs` | 31 | 0 |
| `src/f32x/u15.rs` | 1 | 0 |
| `src/f32x/u35.rs` | 24 | 1 — `xsqrtf_u35`, which upstream keeps inside a block comment |
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

The six names from `src/f32x.rs` that are not present by name are the same kind of folding as
above: `copysign` → `copy_sign`, `rintfk2` → `rint` (identical), `visinf2_vf_vf_v` → `visinf2`,
`sinpifk`/`cospifk` → the private helpers of `src/f32x/u05.rs` (upstream keeps them in `f32x.rs`;
the `f64` port made the same move), and `expk2` → `u10::expk2f`, which keeps upstream's trailing
`f` because `u15` reaches it by name.

The `f32` kernels drop the trailing `f` that upstream's names carry (`expf` → `exp`, `erff` →
`erf`, `sincosf_deterministic` → `sincos_deterministic`), so that one macro can generate the two
slice APIs from one list of names. The private helpers keep upstream's spelling.

Upstream's `#[test]` functions are replaced by `tests/ulp.rs` and `tests/ulp32.rs` (one test per
function, asserting upstream's bound) and `tests/helpers.rs` (the shared helpers).

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
src/f32x/mod.rs     f32 helper layer, mirroring src/f64x/mod.rs one-for-one
src/f32x/u05.rs     sinpi, cospi, sincospi, sqrt, hypot
src/f32x/u10.rs     the 1.0 ULP tier
src/f32x/u15.rs     erfc
src/f32x/u35.rs     the 3.5 ULP tier
src/f32x/fast.rs    the ~350 ULP tier, reachable as f32x::fast but not part of the slice API
src/dispatch.rs     ElementwiseMath trait + slice entry points; one macro generates the f32 and
                    f64 modules, so the two families cannot drift apart
tests/ulp.rs        MPFR ULP suite for f64, upstream's bounds
tests/ulp32.rs      MPFR ULP suite for f32, upstream's f32 bounds
tests/common/mod.rs the ULP harness both suites share
tests/scalar.rs     the --no-default-features path
benches/throughput.rs  paired benchmark for both families, with a correctness gate
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

## Status (2026-10-12)

Phases A, B, C and D are done: the crate builds, tests and lints clean on both feature sets, the
whole `f64` surface is ported, and the kernels are **bit-for-bit identical to upstream's FMA build**
(`sleef-rs` compiled with `-C target-feature=+fma`) over 20 000 samples per function for
`exp`, `expm1`, `tanh`, `log`, `sin`, `exp2`, `exp10` and `cbrt`. `tests/ulp.rs` asserts upstream's
bounds and passes for all 42 cases.

Phase C has been measured and recorded. The predeclared gate is **not met in full**: every
#2033 row except `pow` exceeds 1.5x, and `pow` is 0.75x against libm (though 1.40x faster than
upstream). See the measurements table above; the gate is reported as failed, not relaxed.

## Phases D–G (in progress)

| phase | state |
|---|---|
| D1 element-generic helpers shared in `common.rs` | done; f64 still bit-identical to upstream |
| D2 `f32` constants, `REMPITABSP`, aliases, `dd_family!` for f32 | done |
| D3 `src/f32x/mod.rs` + `constants.rs` helper layer | done; all 50 f64 helpers have an f32 counterpart |
| D4 `f32x/u05`, `u10`, `u15`, `u35`, `fast` | done; 4 858 lines, 65 public kernels |
| D5 `f32` slice API, `f32` ULP tests, benchmark for both dtypes | done; 47 f32 ULP tests pass, both dtypes are bit-identical to upstream's FMA build, and both benchmark families are recorded in `docs/benchmark-results.md` |
| E push + PR + merge `elem-simd-math-rs` (no crates.io publish) | pending |
| F integrate into `tenferro-cpu` for f32 and f64 | worktree `../.worktrees/tenferro-simd-math` on branch `simd-math-analytic-fast-path` prepared |
| G report on #2033 | pending |

The `f32` surface mirrors the `f64` one exactly; upstream's `src/f32x/` exports `u05`
(`sincospi`, `sqrt`, `hypot`, `sinpi`, `cospi`), `u10` (the 1.0 ULP tier), `u15` (`erfc`), `u35`
(the 3.5 ULP tier) and `fast` (`sin`, `cos`, `pow` at roughly 350 ULP).

`tests/common/mod.rs` holds one ULP harness generic over the scalar type, so the two suites share
it; `tests/ulp32.rs` carries the f32 case tables, whose bounds differ from f64's: `tanh` and
`atanh` are 1.0001, `erfc` is 1.5, and the `u35` tier is 3.5.

### What the f32 work and its review turned up

1. **`from_slice_offset` indexed past the Payne–Hanek table for non-finite arguments.** Upstream
   reaches the table through an unchecked gather, so `sin(inf)` reads past `REMPITABSP` there; our
   version is a safe Rust index, which panicked instead. The index is now clamped to the table,
   which leaves every in-domain value untouched and lets the caller's non-finite mask produce the
   NaN `sin(inf)` is supposed to be. `f64` had the same latent overrun for `inf` and is clamped
   too.
2. **The `*pi` oracles needed more than 128 bits.** `sin(pi * x)` loses a bit per bit of `x`, so at
   the default precision a half-integer argument around `2^21` came back ~`x * 2^-128` away from
   the true zero — larger than the asserted sub-ULP bound, and *our* result was the correct one.
   The oracles now work at 512 bits. The `f64` suite never hit it because an exact integer is
   vanishingly unlikely in a 53-bit sample; in `f32` it happens to one sample in sixteen.
3. **The f32 sample generator asked for `f32::MAX` but got `1e306`.** The clamp that keeps
   full-range samples finite was written for `f64`; for `f32` it produced `inf`, which is what
   tripped (1). `UlpScalar` now carries the per-type limit.
4. **`f64x::fmod` and `f64x::remainder` returned a *negative* remainder for some exact large
   integrals.** Upstream clears the low bit of the quotient on its FMA branch, because the fused
   product `toward0(r) * rd` can round up and the loop then overshoots; the port had dropped that
   correction and also computed `rd` as `1 / toward0(d)` where upstream writes `toward0(1 / d)`.
   `fmod(7 * 2^51, 1)` returned `-2` instead of `0`. Both are fixed and `tests/helpers.rs` pins the
   case; the `f32` counterparts never had the correction upstream, so nothing changed there.
5. **`f32x::ldexpk` grouped the four `2^m` factors differently from upstream.** Upstream multiplies
   by `u*u*u*u` and only then by `2^q`; the port multiplied left to right, so an intermediate that
   overflows upstream did not overflow here. Mathematically the port was the better one, but it
   broke bit-identity, so the grouping now matches.
6. **The `tests/ulp32.rs` bounds for `f32` `asinh` and `acosh` were 1.0 where upstream asserts
   1.0001.** The port passed, so this was only a matter of asserting the same thing upstream does.
7. **`f64x::u05::sqrt` was still upstream's non-FMA branch** — the only kernel left on it, because
   `u05` was ported before the FMA decision was taken. It has the `1.34e154` scaling and the
   split-form Newton iterations that the FMA branch drops, so it was neither bit-identical to
   upstream's `+fma` build nor as fast. It is now the FMA branch, like the `f32` one next to it.

The `f64` kernels were compared against upstream's `+fma` build over 20 016 inputs each for `sin`,
`cos`, `tan`, `sincos`, `sinpi`, `cospi`, `expm1`, `log`, `exp2`, `erfc` and `atan` (`f32`) and the
eight named in the status section (`f64`); the comparison harness itself is a scratch crate outside
this repository, since pulling `sleef-rs` in as a dev-dependency would drag in a nightly
`portable_simd` build.

## Remaining work

## Fused activations (2026-10-21, tenferro-rs#2233)

`sigmoid`, `silu`, `softplus`, `gelu` and `gelu_tanh` as one-pass vector kernels
(`src/f32x/activation.rs`, `src/f64x/activation.rs`), composed from the tier
kernels the way PyTorch's CPU `Vectorized<float>` activations are. Motivation:
tenferro's composite formulation (seven materialized passes) ran sigmoid at
172 us and gelu at 333 us for 1024x64 `f32` where PyTorch takes 49 / 61 us; the
fused kernels bring tenferro to 44 / 65 us.

- Formulas are the overflow-free ones (`exp(-|x|)` only); the far negative tail
  stays subnormal instead of flushing (asserted by `*_keep_their_subnormal_tail`).
- Bounds are stated per function and are not SLEEF tiers: `sigmoid` 2.5 ULP,
  `silu` 3.5 ULP (normal results), `softplus` 3.0 ULP, `gelu_tanh` and the
  `f64` `gelu` 3.0 ULP for `x >= 0`. The `f32` `gelu` uses the
  Abramowitz–Stegun 7.1.26 `erf` that PyTorch's CPU GELU uses (about `5e-7`
  absolute on `erf` after `f32` rounding), so its contract is
  `3.0e-7 * |x|` plus 3 ULP. For `x < 0` the `1 + erf` / `1 + tanh`
  cancellation sets the relative error of both GELU forms.
- Measured on an mpmath oracle over 2^18 samples per kernel and dtype before
  the bounds were set: `f32` sigmoid 2.25, silu 3.28, softplus 1.84 ULP;
  `f64` sigmoid 1.91, silu 2.80, softplus 1.32 ULP; `f32` gelu at 0.60 of its
  bound. The MPFR suites in `tests/` assert the bounds in CI.
- Throughput at 2^16 elements, one core: `f32` sigmoid 0.63 ns/elem, silu
  0.67, softplus 2.4, gelu 0.97, gelu_tanh 2.85; `f64` sigmoid 1.84, gelu 5.9.

## `sqrt` gets a vector kernel (2026-10-27, issue #9)

`sqrt` was the one slice entry that did not go through `dispatch!`: an unconditional
`scalar_apply(src, dst, <$ty>::sqrt)`. It now uses `common::hw_sqrt`, one `sqrt` method on the
vector type — the hardware instruction, one correctly rounded result per lane, so the slice path is
bit-identical to the scalar `sqrt` and not merely inside a bound. `tests/ulp.rs` and
`tests/ulp32.rs` pin `-0.0`, negatives, subnormals, `MIN_POSITIVE`, `MAX`, `±inf` and NaN beside the
MPFR suite.

SLEEF's `u05::sqrt` was not used for this: it is a Newton refinement bounded at 0.5001 ULP, and the
issue's acceptance criterion is bit-identity with the scalar path. `f64x::u05::sqrt` and
`f32x::u05::sqrt` stay where they are, used by `u35::sqrt` and the double-double kernels.

Measured at the issue's size, `n = 33_554_432`, one thread pinned to CPU 2, destination buffer
reused, best of 5 after 2 warmups, observer on CPUs 16-23: `f64` 40.15 ms / 1.1966 ns per element
before, 21.35 ms / 0.6363 after (1.88x, against 22.9 ms for a bare `_mm256_sqrt_pd`); `f32` 12.35
to 10.85 ms (1.14x, already memory bound). At the crate's `n = 2^23` protocol the paired speedup
against libm went from 2.07x to 3.89x (`f64`) and from 3.96x to 10.57x (`f32`); the details and one
discarded baseline outlier are in `docs/benchmark-results.md`.
