# Throughput results — `f64` and `f32`

`cargo bench --bench throughput`, pinned to one core (`taskset -c 2`), median of 7 timed rounds
after 2 warmups, `n = 2^23` (8 388 608 elements) per case. `speedup` is `scalar / kernel`, so
larger is faster. The `scalar` arm is the platform libm called one element at a time from the same
binary; `kernel` is this crate's slice entry point, which selects the best `fearless_simd` level at
runtime. The `bench` profile inherits `release` and adds debug line tables, so the numbers match
the release build while backtraces stay readable.

Host: AMD EPYC 7713P (Zen 3), 64 cores, Linux, no CUDA, rustc 1.97.0. `Level::new()` selects
`Avx2`, so the `f64` vector width is 4 lanes and the `f32` width is 8. The harness is serial and
spawns no threads. `max rel` is the largest relative difference between the two arms over the
whole random buffer; it is a diagnostic, not the gate — next to a zero or a pole it is meaningless,
and the binding accuracy gate is the MPFR suite in `tests/ulp.rs` / `tests/ulp32.rs`.

## Summary

### `f64`, 4 lanes

| op | scalar | kernel | speedup | max rel |
|---|---:|---:|---:|---:|
| `sin (u35)` | 8.519 | 1.617 | 5.27x | 2.6e-16 |
| `exp10` | 11.954 | 2.271 | 5.26x | 2.2e-16 |
| `sincos` | 18.033 | 3.610 | 5.00x | 0.0e0 |
| `cos (u35)` | 8.662 | 1.879 | 4.61x | 2.4e-16 |
| `tanh (u35)` | 12.642 | 2.856 | 4.43x | 6.5e-16 |
| `tan (u35)` | 10.211 | 2.367 | 4.31x | 4.4e-16 |
| `sin` | 8.515 | 2.565 | 3.32x | 2.2e-16 |
| `cos` | 8.664 | 2.945 | 2.94x | 2.2e-16 |
| `log10` | 7.719 | 2.649 | 2.91x | 2.6e-16 |
| `atan2` | 14.780 | 5.131 | 2.88x | 2.2e-16 |
| `sinh` | 13.170 | 4.621 | 2.85x | 3.0e-16 |
| `tan` | 10.175 | 3.604 | 2.82x | 2.2e-16 |
| `log (u35)` | 4.747 | 1.700 | 2.79x | 3.8e-16 |
| `cbrt` | 16.981 | 6.175 | 2.75x | 2.2e-16 |
| `log1p` | 7.476 | 2.809 | 2.66x | 2.2e-16 |
| `asin` | 6.547 | 2.632 | 2.49x | 2.2e-16 |
| `tanh` | 12.649 | 5.302 | 2.39x | 3.9e-16 |
| `atan` | 8.370 | 3.531 | 2.37x | 2.2e-16 |
| `exp` | 4.926 | 2.112 | 2.33x | 2.2e-16 |
| `cospi` | 9.117 | 4.285 | 2.13x | 1.0e0 |
| `acos` | 6.558 | 3.174 | 2.07x | 2.2e-16 |
| `sqrt` | 2.457 | 1.188 | 2.07x | 0.0e0 |
| `sinpi` | 9.006 | 4.456 | 2.02x | 2.7e-10 |
| `exp2` | 3.313 | 1.784 | 1.86x | 2.2e-16 |
| `hypot` | 6.420 | 3.456 | 1.86x | 2.2e-16 |
| `log` | 4.741 | 2.613 | 1.81x | 2.2e-16 |
| `expm1` | 7.198 | 4.255 | 1.69x | 2.2e-16 |
| `cosh` | 7.401 | 4.546 | 1.63x | 2.2e-16 |
| `atanh` | 7.913 | 5.055 | 1.57x | 5.0e-13 |
| `log2` | 4.173 | 2.762 | 1.51x | 2.2e-16 |
| `tgamma` | 40.128 | 32.412 | 1.24x | 7.4e-16 |
| `asinh` | 12.053 | 9.922 | 1.21x | 2.5e-16 |
| `erf` | 8.728 | 8.819 | 0.99x | 2.2e-16 |
| `acosh` | 7.355 | 8.245 | 0.89x | 2.7e-16 |
| `pow` | 12.084 | 16.358 | 0.74x | 2.2e-16 |
| `pow(x,1.5)` | 11.457 | 16.223 | 0.71x | 0.0e0 |
| `erfc` | 9.463 | 14.074 | 0.67x | 7.2e-16 |
| `lgamma` | 20.703 | 33.902 | 0.61x | 6.4e-13 |

### `f32`, 8 lanes

| op | scalar | kernel | speedup | max rel |
|---|---:|---:|---:|---:|
| `tan (u35)` | 11.606 | 0.632 | 18.36x | 2.3e-7 |
| `tanh (u35)` | 13.837 | 0.906 | 15.28x | 3.5e-7 |
| `sinh` | 14.640 | 2.092 | 7.00x | 1.6e-7 |
| `sin (u35)` | 3.193 | 0.467 | 6.84x | 1.4e-7 |
| `asin` | 6.986 | 1.025 | 6.81x | 1.2e-7 |
| `tan` | 11.585 | 1.783 | 6.50x | 1.2e-7 |
| `cos (u35)` | 3.401 | 0.548 | 6.20x | 1.3e-7 |
| `log1p` | 7.397 | 1.203 | 6.15x | 1.2e-7 |
| `log10` | 7.145 | 1.183 | 6.04x | 1.4e-7 |
| `tanh` | 13.835 | 2.395 | 5.78x | 2.1e-7 |
| `exp2` | 3.005 | 0.570 | 5.28x | 1.2e-7 |
| `exp10` | 5.084 | 0.966 | 5.26x | 1.2e-7 |
| `exp` | 3.014 | 0.580 | 5.20x | 1.2e-7 |
| `acos` | 6.681 | 1.301 | 5.14x | 1.2e-7 |
| `log (u35)` | 3.224 | 0.675 | 4.78x | 2.4e-7 |
| `atan2` | 11.668 | 2.485 | 4.70x | 1.2e-7 |
| `expm1` | 8.382 | 1.922 | 4.36x | 1.2e-7 |
| `sincos` | 6.164 | 1.440 | 4.28x | 0.0e0 |
| `cosh` | 8.055 | 1.930 | 4.17x | 1.2e-7 |
| `atan` | 5.985 | 1.512 | 3.96x | 1.2e-7 |
| `sqrt` | 1.408 | 0.356 | 3.96x | 0.0e0 |
| `atanh` | 7.821 | 2.128 | 3.67x | 2.0e-5 |
| `tgamma` | 32.580 | 9.332 | 3.49x | 4.7e-7 |
| `asinh` | 11.999 | 3.951 | 3.04x | 1.4e-7 |
| `log` | 3.217 | 1.142 | 2.82x | 1.2e-7 |
| `sin` | 3.188 | 1.211 | 2.63x | 1.2e-7 |
| `log2` | 3.221 | 1.235 | 2.61x | 1.2e-7 |
| `erf` | 8.878 | 3.579 | 2.48x | 1.2e-7 |
| `cospi` | 3.632 | 1.464 | 2.48x | 1.0e0 |
| `cos` | 3.390 | 1.388 | 2.44x | 1.2e-7 |
| `acosh` | 7.871 | 3.389 | 2.32x | 1.5e-7 |
| `sinpi` | 3.424 | 1.492 | 2.30x | 3.4e-1 |
| `hypot` | 3.356 | 1.612 | 2.08x | 0.0e0 |
| `erfc` | 9.508 | 4.990 | 1.91x | 2.4e-7 |
| `cbrt` | 4.958 | 2.746 | 1.81x | 9.4e-8 |
| `lgamma` | 15.832 | 9.753 | 1.62x | 2.1e-4 |
| `pow` | 5.685 | 4.935 | 1.15x | 1.2e-7 |
| `pow(x,1.5)` | 5.301 | 4.930 | 1.08x | 0.0e0 |

## Reading the numbers

- The `u35` rows are the deliberately looser 3.5 ULP tier. They are the fastest kernels in the set
  and the right choice when the caller already tolerates a few ULP.
- `sinpi`/`cospi`'s scalar arm computes `sin(pi*x)` in double precision directly, which is *less*
  accurate than the kernel for large arguments; that column exists only to price the vector path.
  The large `rel` entries there are the scalar arm's error, not the kernel's.
- `sqrt` has no vector kernel: the hardware square root is already vectorized by LLVM, so the row
  shows what the scalar libm call costs on top of it.
- `pow` is the one operation the scalar library wins on `f64`; see `docs/findings-pow.md`. On `f32`
  SLEEF's `powf` is faster but its bound is 1.0 ULP against glibc's 0.504, so the trade-off runs the
  other way.
- `tgamma`, `lgamma`, `erfc`, `asinh` and `acosh` are the double-double-heavy kernels: they win on
  accuracy by a wide margin (see `docs/findings-tgamma.md`) but their cost is set by the polynomial
  degree, not by the lane count, so the speedup there is the smallest.

## Raw output

```
   Compiling elem-simd-math v0.1.0 (/home/shinaoka/tensor4all/elem-simd-math-rs)
    Finished `bench` profile [optimized + debuginfo] target(s) in 5m 37s
     Running benches/throughput.rs (target/release/deps/throughput-38ec02648781601c)
elem-simd-math throughput
warmup 2, rounds 7, statistic = median wall time, ns/element
primary n = 8388608; `speedup` is libm/kernel, so larger is faster

== f64 : level Avx2(Avx2 { _private: () }) ==
op               scalar     kernel   speedup  max |diff|       rel
exp               4.926      2.112     2.33x    4.55e-13   2.2e-16
exp2              3.313      1.784     1.86x    2.84e-14   2.2e-16
expm1             7.198      4.255     1.69x    4.55e-13   2.2e-16
log               4.741      2.613     1.81x    8.88e-16   2.2e-16
log2              4.173      2.762     1.51x    1.78e-15   2.2e-16
log10             7.719      2.649     2.91x    8.88e-16   2.6e-16
log1p             7.476      2.809     2.66x    8.88e-16   2.2e-16
sin               8.515      2.565     3.32x    1.11e-16   2.2e-16
cos               8.664      2.945     2.94x    1.11e-16   2.2e-16
tan              10.175      3.604     2.82x    1.42e-14   2.2e-16
asin              6.547      2.632     2.49x    2.22e-16   2.2e-16
acos              6.558      3.174     2.07x    4.44e-16   2.2e-16
atan              8.370      3.531     2.37x    2.22e-16   2.2e-16
sinh             13.170      4.621     2.85x    2.27e-13   3.0e-16
cosh              7.401      4.546     1.63x    2.27e-13   2.2e-16
tanh             12.649      5.302     2.39x    2.22e-16   3.9e-16
asinh            12.053      9.922     1.21x    4.44e-16   2.5e-16
acosh             7.355      8.245     0.89x    4.44e-16   2.7e-16
atanh             7.913      5.055     1.57x    3.03e-12   5.0e-13
cbrt             16.981      6.175     2.75x    2.22e-16   2.2e-16
sqrt              2.457      1.188     2.07x      0.00e0     0.0e0
sin (u35)         8.519      1.617     5.27x    2.22e-16   2.6e-16
cos (u35)         8.662      1.879     4.61x    2.22e-16   2.4e-16
tan (u35)        10.211      2.367     4.31x    1.16e-10   4.4e-16
log (u35)         4.747      1.700     2.79x    1.78e-15   3.8e-16
tanh (u35)       12.642      2.856     4.43x    3.33e-16   6.5e-16
exp10            11.954      2.271     5.26x     1.49e-8   2.2e-16
erf               8.728      8.819     0.99x    1.11e-16   2.2e-16
erfc              9.463     14.074     0.67x    2.22e-16   7.2e-16
tgamma           40.128     32.412     1.24x    5.82e-11   7.4e-16
lgamma           20.703     33.902     0.61x    3.55e-15   6.4e-13
sinpi             9.006      4.456     2.02x    2.75e-15   2.7e-10
cospi             9.117      4.285     2.13x    2.70e-15     1.0e0
pow              12.084     16.358     0.74x    5.68e-14   2.2e-16
atan2            14.780      5.131     2.88x    2.22e-16   2.2e-16
hypot             6.420      3.456     1.86x    1.78e-15   2.2e-16
pow(x,1.5)       11.457     16.223     0.71x      0.00e0     0.0e0
sincos           18.033      3.610     5.00x      0.00e0     0.0e0

== f32 : level Avx2(Avx2 { _private: () }) ==
op               scalar     kernel   speedup  max |diff|       rel
exp               3.014      0.580     5.20x     2.44e-4    1.2e-7
exp2              3.005      0.570     5.28x     1.53e-5    1.2e-7
expm1             8.382      1.922     4.36x     2.44e-4    1.2e-7
log               3.217      1.142     2.82x     4.77e-7    1.2e-7
log2              3.221      1.235     2.61x     9.54e-7    1.2e-7
log10             7.145      1.183     6.04x     4.77e-7    1.4e-7
log1p             7.397      1.203     6.15x     9.54e-7    1.2e-7
sin               3.188      1.211     2.63x     5.96e-8    1.2e-7
cos               3.390      1.388     2.44x     5.96e-8    1.2e-7
tan              11.585      1.783     6.50x      1.00e0    1.2e-7
asin              6.986      1.025     6.81x     1.19e-7    1.2e-7
acos              6.681      1.301     5.14x     2.38e-7    1.2e-7
atan              5.985      1.512     3.96x     1.19e-7    1.2e-7
sinh             14.640      2.092     7.00x     1.22e-4    1.6e-7
cosh              8.055      1.930     4.17x     1.22e-4    1.2e-7
tanh             13.835      2.395     5.78x     1.19e-7    2.1e-7
asinh            11.999      3.951     3.04x     2.38e-7    1.4e-7
acosh             7.871      3.389     2.32x     2.38e-7    1.5e-7
atanh             7.821      2.128     3.67x     8.49e-5    2.0e-5
cbrt              4.958      2.746     1.81x     1.19e-7    9.4e-8
sqrt              1.408      0.356     3.96x      0.00e0     0.0e0
sin (u35)         3.193      0.467     6.84x     1.19e-7    1.4e-7
cos (u35)         3.401      0.548     6.20x     1.19e-7    1.3e-7
tan (u35)        11.606      0.632    18.36x     4.88e-4    2.3e-7
log (u35)         3.224      0.675     4.78x     9.54e-7    2.4e-7
tanh (u35)       13.837      0.906    15.28x     1.79e-7    3.5e-7
exp10             5.084      0.966     5.26x      8.00e0    1.2e-7
erf               8.878      3.579     2.48x     5.96e-8    1.2e-7
erfc              9.508      4.990     1.91x     1.19e-7    2.4e-7
tgamma           32.580      9.332     3.49x     2.50e-1    4.7e-7
lgamma           15.832      9.753     1.62x     1.91e-6    2.1e-4
sinpi             3.424      1.492     2.30x     1.64e-6    3.4e-1
cospi             3.632      1.464     2.48x     1.61e-6     1.0e0
pow               5.685      4.935     1.15x     3.05e-5    1.2e-7
atan2            11.668      2.485     4.70x     1.19e-7    1.2e-7
hypot             3.356      1.612     2.08x      0.00e0     0.0e0
pow(x,1.5)        5.301      4.930     1.08x      0.00e0     0.0e0
sincos            6.164      1.440     4.28x      0.00e0     0.0e0
```
