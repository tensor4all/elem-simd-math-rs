# Open finding: `tgamma`/`lgamma` diverge from upstream and cost 4x

Recorded 2026-10-11. Status: **open**. Not required by the rows of
`tensor4all/tenferro-rs#2033`; these two functions were ported for completeness.

## Reproduction

`/tmp/cmp/src/bin/slowcmp.rs` builds `elem-simd-math` and `sleef-rs` rev `3671d46` (with
`RUSTFLAGS="-C target-feature=+fma"`, i.e. upstream's FMA build) into one binary and runs both over
the same 2^20-element input. Numbers are per element; upstream runs 2 lanes, this port 4.

| function | this port | upstream | ratio | bit mismatches |
|---|---:|---:|---:|---:|
| `tgamma` | 221.0 ns | 51.4 ns | 0.23x | 1623 / 1048576 |
| `lgamma` | 219.8 ns | 47.6 ns | 0.22x | 3848 / 1048576 |
| `erfc`   | 14.1 ns  | 30.4 ns  | 2.16x | 0 |
| `erf`    | 10.9 ns  | 21.2 ns  | 1.94x | 0 |
| `pow`    | 16.0 ns  | 22.7 ns  | 1.42x | — |

`erf`/`erfc`/`pow` are within a factor of two of upstream in the direction the 2x lane-width
advantage predicts, i.e. those are faithful. `tgamma`/`lgamma` are 4.3x and 4.6x **slower per
element** and bit-different, so something in `gammak` is not a faithful transcription.

## What has been ruled out

- Every numeric literal and every control-flow construct (`if`/`else`/`for`/`while`/`break`
  counts) in `gammak` matches upstream; `mla` 22/22, `select_doubled` 12/12, `normalize` 5/5,
  46 exponent-notation constants on both sides.
- The mismatches are 1 ULP only (`rel <= 2.1e-16`) and occur only for **negative** arguments
  (627/200000 over `[-15, 15)`, none for positive `x`), which points at the reflection branch.
- `tgamma` and `lgamma` bodies match upstream line for line, including the final `select`s.

## Next step

Bisect `gammak`'s reflection branch by dumping its `(da, db)` pair for a negative input from both
implementations. The most likely candidates are the `o2.select(x.0.recip(), ...)` reciprocal and
the `select3`/`select_doubled` nesting in the negative-argument path.
