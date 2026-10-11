# `pow` is closed: neither `libm` nor a vectorised port of it beats glibc

Recorded 2026-10-11. Status: **closed, no change to the crate.**

`pow` is the one row of `tensor4all/tenferro-rs#2033` that this crate does not improve. Both
plausible remedies were tried and measured; both lost.

## 1. Scalar cost of the three candidate implementations

Same inputs (x ∈ [0.25, 100]), single thread, ns/element, ULP against MPFR.

| implementation | y = 1.5 (constant) | y ∈ [-3, 3] | measured ULP |
|---|---:|---:|---:|
| glibc (the host's `libm`) | 12.86 | 12.13 | 0.503 |
| FDLIBM / musl (`libm-rs` crate, i.e. the `pow` this port would copy) | 51.42 | 44.13 | 0.770 |
| SLEEF (`sleef-rs`, the source of this crate's `pow`) | 75.95 | 75.25 | 0.609 |

glibc's `pow` is the only one that is both fast and correctly rounded (0.503 ULP). It is a
scalar, table-driven implementation; the table lookup is exactly what a portable SIMD port
cannot reproduce cheaply.

## 2. A vectorised port of the FDLIBM `pow` was written and measured

`libm-rs`'s `pow` has no table lookups (`grep` confirms zero), so it is vectorisable in principle,
and its scalar cost is 1.7x below SLEEF's. The port was written into this crate's `f64x` layer
with every early return turned into a `select`, and two scalar variable-shift tricks replaced by
`trunc`/`round_ties_even` (documented in the file header while it existed). Measured in the same
harness as the two implementations above:

| implementation | ns/element |
|---|---:|
| glibc | 14.23 |
| this crate's SLEEF kernel | 20.71 |
| vectorised FDLIBM port | **23.99** |

The port is 0.59x of glibc and 0.86x of the kernel it would replace. The reason is structural: the
scalar original returns early after one path, whereas a SIMD version must evaluate *both* `log2`
intervals plus the `x ∈ {0, ±1, ±inf}`, `y ∈ {±inf, 1, 2, 0.5}` and subnormal-result fixes for every
lane. Guarding the rare paths with a uniform `any_true` branch would recover perhaps 30%, still
leaving it behind glibc. The port was therefore deleted rather than shipped.

## 3. What to do instead

* Keep `f64::powf` (the platform `libm`) for `pow`; the tenferro fast path should exclude it.
* For the workload `#2033` actually measures — a **constant exponent of 1.5** — `x * sqrt(x)`
  measures **0.70 ns/element**, 20x faster than glibc's `pow` and within 1 ULP. That
  specialisation belongs to the caller, which knows whether the exponent is constant.

## Reproducing

The three-way scalar comparison is `benches/portable_libm.rs` (recorded in
`docs/portable-libm-results.md`) plus the SLEEF scalar arm; the SIMD attempt lived in
`src/f64x/msun.rs` in commit `dc042d7^..` and was removed in the commit that added this file.
