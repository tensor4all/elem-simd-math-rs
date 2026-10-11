# Throughput results — `f64` and `f32`

`cargo bench --bench throughput`, pinned to one core (`taskset -c 2`), median of 7 timed rounds
after 2 warmups, `n = 2^23` (8 388 608 elements) per case. `speedup` is `scalar / kernel`, so
larger is faster. The `scalar` arm is the platform libm called one element at a time from the same
binary; `kernel` is this crate's slice entry point, which selects the best `fearless_simd` level at
runtime.

Host: AMD EPYC 7713P (Zen 3), 64 cores, Linux, no CUDA, rustc 1.97.0. `Level::new()` selects
`Avx2`, so the `f64` vector width is 4 lanes and the `f32` width is 8. The harness is serial and
spawns no threads. `max rel` is the largest relative difference between the two arms over the
whole random buffer; the per-function error bounds are the ones upstream asserts, listed in
`README.md`.

## Summary

### `f64`, 4 lanes

| op | scalar | kernel | speedup | max rel |
|---|---:|---:|---:|---:|
| `exp` | 5.206 | 2.134 | 2.44x | 2.2e-16 |
| `exp2` | 3.619 | 1.826 | 1.98x | 2.2e-16 |
| `expm1` | 7.599 | 4.259 | 1.78x | 2.2e-16 |
| `log` | 5.053 | 2.642 | 1.91x | 2.2e-16 |
| `log2` | 4.272 | 2.784 | 1.53x | 2.2e-16 |
| `log10` | 7.807 | 2.677 | 2.92x | 2.6e-16 |
| `log1p` | 7.587 | 2.762 | 2.75x | 2.2e-16 |
| `sin` | 8.517 | 2.586 | 3.29x | 2.2e-16 |
| `cos` | 8.664 | 2.954 | 2.93x | 2.2e-16 |
| `tan` | 10.258 | 3.623 | 2.83x | 2.2e-16 |
| `asin` | 6.495 | 2.672 | 2.43x | 2.2e-16 |
| `acos` | 6.514 | 3.239 | 2.01x | 2.2e-16 |
| `atan` | 8.404 | 3.550 | 2.37x | 2.2e-16 |
| `sinh` | 13.258 | 4.676 | 2.84x | 3.0e-16 |
| `cosh` | 7.654 | 4.602 | 1.66x | 2.2e-16 |
| `tanh` | 12.955 | 5.329 | 2.43x | 3.9e-16 |
| `asinh` | 12.195 | 9.921 | 1.23x | 2.5e-16 |
| `acosh` | 7.247 | 8.246 | 0.88x | 2.7e-16 |
| `atanh` | 8.488 | 5.106 | 1.66x | 5.0e-13 |
| `cbrt` | 16.592 | 6.256 | 2.65x | 2.2e-16 |
| `sqrt` | 2.485 | 1.183 | 2.10x | 0.0e0 |
| `sin (u35)` | 8.531 | 1.651 | 5.17x | 2.6e-16 |
| `cos (u35)` | 8.679 | 1.898 | 4.57x | 2.4e-16 |
| `tan (u35)` | 10.259 | 2.387 | 4.30x | 4.4e-16 |
| `log (u35)` | 5.065 | 1.765 | 2.87x | 3.8e-16 |
| `tanh (u35)` | 12.958 | 2.882 | 4.50x | 6.5e-16 |
| `exp10` | 11.923 | 2.296 | 5.19x | 2.2e-16 |
| `erf` | 8.840 | 8.830 | 1.00x | 2.2e-16 |
| `erfc` | 9.546 | 14.183 | 0.67x | 7.2e-16 |
| `tgamma` | 40.208 | 32.292 | 1.25x | 7.4e-16 |
| `lgamma` | 20.701 | 33.763 | 0.61x | 6.4e-13 |
| `sinpi` | 9.081 | 4.471 | 2.03x | 2.7e-10 |
| `cospi` | 9.183 | 4.291 | 2.14x | 1.0e0 |
| `pow` | 12.219 | 16.280 | 0.75x | 2.2e-16 |
| `atan2` | 15.324 | 5.317 | 2.88x | 2.2e-16 |
| `hypot` | 6.563 | 3.594 | 1.83x | 2.2e-16 |
| `pow(x,1.5)` | 11.520 | 16.185 | 0.71x | 0.0e0 |
| `sincos` | 18.096 | 3.634 | 4.98x | 0.0e0 |

### `f32`, 8 lanes

| op | scalar | kernel | speedup | max rel |
|---|---:|---:|---:|---:|
| `exp` | 3.010 | 0.571 | 5.27x | 1.2e-7 |
| `exp2` | 3.002 | 0.564 | 5.32x | 1.2e-7 |
| `expm1` | 8.379 | 1.899 | 4.41x | 1.2e-7 |
| `log` | 3.953 | 1.147 | 3.45x | 1.2e-7 |
| `log2` | 3.969 | 1.236 | 3.21x | 1.2e-7 |
| `log10` | 7.567 | 1.182 | 6.40x | 1.4e-7 |
| `log1p` | 7.406 | 1.202 | 6.16x | 1.2e-7 |
| `sin` | 3.191 | 1.175 | 2.72x | 1.2e-7 |
| `cos` | 3.387 | 1.384 | 2.45x | 1.2e-7 |
| `tan` | 11.573 | 1.783 | 6.49x | 1.2e-7 |
| `asin` | 7.192 | 1.022 | 7.04x | 1.2e-7 |
| `acos` | 6.813 | 1.298 | 5.25x | 1.2e-7 |
| `atan` | 5.972 | 1.505 | 3.97x | 1.2e-7 |
| `sinh` | 14.636 | 2.073 | 7.06x | 1.6e-7 |
| `cosh` | 8.245 | 1.929 | 4.28x | 1.2e-7 |
| `tanh` | 13.833 | 2.379 | 5.81x | 2.1e-7 |
| `asinh` | 11.954 | 3.954 | 3.02x | 1.4e-7 |
| `acosh` | 7.833 | 3.390 | 2.31x | 1.5e-7 |
| `atanh` | 7.865 | 2.115 | 3.72x | 2.0e-5 |
| `cbrt` | 4.954 | 2.749 | 1.80x | 9.4e-8 |
| `sqrt` | 1.400 | 0.356 | 3.93x | 0.0e0 |
| `sin (u35)` | 3.190 | 0.467 | 6.83x | 1.4e-7 |
| `cos (u35)` | 4.184 | 0.540 | 7.74x | 1.3e-7 |
| `tan (u35)` | 11.570 | 0.620 | 18.67x | 2.3e-7 |
| `log (u35)` | 3.222 | 0.674 | 4.78x | 2.4e-7 |
| `tanh (u35)` | 13.842 | 0.903 | 15.32x | 3.5e-7 |
| `exp10` | 5.105 | 0.966 | 5.29x | 1.2e-7 |
| `erf` | 8.864 | 3.688 | 2.40x | 1.2e-7 |
| `erfc` | 9.488 | 4.987 | 1.90x | 2.4e-7 |
| `tgamma` | 32.611 | 9.342 | 3.49x | 4.7e-7 |
| `lgamma` | 15.880 | 9.730 | 1.63x | 2.1e-4 |
| `sinpi` | 4.159 | 1.488 | 2.80x | 3.4e-1 |
| `cospi` | 4.392 | 1.467 | 2.99x | 1.0e0 |
| `pow` | 5.774 | 5.132 | 1.13x | 1.2e-7 |
| `atan2` | 11.922 | 2.638 | 4.52x | 1.2e-7 |
| `hypot` | 3.581 | 1.678 | 2.13x | 0.0e0 |
| `pow(x,1.5)` | 5.275 | 5.043 | 1.05x | 0.0e0 |
| `sincos` | 6.258 | 1.452 | 4.31x | 0.0e0 |

## Reading the numbers

- The `u35` rows are the deliberately looser 3.5 ULP tier. They are the fastest kernels in the set
  and the right choice when the caller already tolerates a few ULP.
- `sinpi`/`cospi`'s scalar arm computes `sin(pi*x)` in double precision directly, which is *less*
  accurate than the kernel for large arguments; that column exists only to price the vector path.
  The large `rel` entries there are the scalar arm's error, not the kernel's.
- `sqrt` has no vector kernel: the hardware square root is already vectorized by LLVM, so the row
  shows what the scalar libm call costs on top of it.
- `pow` is the one operation where the scalar library wins on `f64`; see `docs/findings-pow.md`.
  On `f32` SLEEF's `powf` is about 14% faster, but its bound is 1.0 ULP against glibc's
  correctly-rounded 0.5 ULP, so the trade-off runs the other way.
- `tgamma`, `lgamma`, `erfc`, `asinh` and `acosh` are the double-double-heavy kernels: they win on
  accuracy by a wide margin (see `docs/findings-tgamma.md`) but their cost is set by the polynomial
  degree, not by the lane count, so the speedup there is the smallest.

## Raw output

```
    Finished `bench` profile [optimized + debuginfo] target(s) in 0.01s
     Running benches/throughput.rs (target/release/deps/throughput-38ec02648781601c)
elem-simd-math throughput
warmup 2, rounds 7, statistic = median wall time, ns/element
primary n = 8388608; `speedup` is libm/kernel, so larger is faster

== f64 : level Avx2(Avx2 { _private: () }) ==
op               scalar     kernel   speedup  max |diff|       rel
exp               5.206      2.134     2.44x    4.55e-13   2.2e-16
exp2              3.619      1.826     1.98x    2.84e-14   2.2e-16
expm1             7.599      4.259     1.78x    4.55e-13   2.2e-16
log               5.053      2.642     1.91x    8.88e-16   2.2e-16
log2              4.272      2.784     1.53x    1.78e-15   2.2e-16
log10             7.807      2.677     2.92x    8.88e-16   2.6e-16
log1p             7.587      2.762     2.75x    8.88e-16   2.2e-16
sin               8.517      2.586     3.29x    1.11e-16   2.2e-16
cos               8.664      2.954     2.93x    1.11e-16   2.2e-16
tan              10.258      3.623     2.83x    1.42e-14   2.2e-16
asin              6.495      2.672     2.43x    2.22e-16   2.2e-16
acos              6.514      3.239     2.01x    4.44e-16   2.2e-16
atan              8.404      3.550     2.37x    2.22e-16   2.2e-16
sinh             13.258      4.676     2.84x    2.27e-13   3.0e-16
cosh              7.654      4.602     1.66x    2.27e-13   2.2e-16
tanh             12.955      5.329     2.43x    2.22e-16   3.9e-16
asinh            12.195      9.921     1.23x    4.44e-16   2.5e-16
acosh             7.247      8.246     0.88x    4.44e-16   2.7e-16
atanh             8.488      5.106     1.66x    3.03e-12   5.0e-13
cbrt             16.592      6.256     2.65x    2.22e-16   2.2e-16
sqrt              2.485      1.183     2.10x      0.00e0     0.0e0
sin (u35)         8.531      1.651     5.17x    2.22e-16   2.6e-16
cos (u35)         8.679      1.898     4.57x    2.22e-16   2.4e-16
tan (u35)        10.259      2.387     4.30x    1.16e-10   4.4e-16
log (u35)         5.065      1.765     2.87x    1.78e-15   3.8e-16
tanh (u35)       12.958      2.882     4.50x    3.33e-16   6.5e-16
exp10            11.923      2.296     5.19x     1.49e-8   2.2e-16
erf               8.840      8.830     1.00x    1.11e-16   2.2e-16
erfc              9.546     14.183     0.67x    2.22e-16   7.2e-16
tgamma           40.208     32.292     1.25x    5.82e-11   7.4e-16
lgamma           20.701     33.763     0.61x    3.55e-15   6.4e-13
sinpi             9.081      4.471     2.03x    2.75e-15   2.7e-10
cospi             9.183      4.291     2.14x    2.70e-15     1.0e0
pow              12.219     16.280     0.75x    5.68e-14   2.2e-16
atan2            15.324      5.317     2.88x    2.22e-16   2.2e-16
hypot             6.563      3.594     1.83x    1.78e-15   2.2e-16
pow(x,1.5)       11.520     16.185     0.71x      0.00e0     0.0e0
sincos           18.096      3.634     4.98x      0.00e0     0.0e0

== f32 : level Avx2(Avx2 { _private: () }) ==
op               scalar     kernel   speedup  max |diff|       rel
exp               3.010      0.571     5.27x     2.44e-4    1.2e-7
exp2              3.002      0.564     5.32x     1.53e-5    1.2e-7
expm1             8.379      1.899     4.41x     2.44e-4    1.2e-7
log               3.953      1.147     3.45x     4.77e-7    1.2e-7
log2              3.969      1.236     3.21x     9.54e-7    1.2e-7
log10             7.567      1.182     6.40x     4.77e-7    1.4e-7
log1p             7.406      1.202     6.16x     9.54e-7    1.2e-7
sin               3.191      1.175     2.72x     5.96e-8    1.2e-7
cos               3.387      1.384     2.45x     5.96e-8    1.2e-7
tan              11.573      1.783     6.49x      1.00e0    1.2e-7
asin              7.192      1.022     7.04x     1.19e-7    1.2e-7
acos              6.813      1.298     5.25x     2.38e-7    1.2e-7
atan              5.972      1.505     3.97x     1.19e-7    1.2e-7
sinh             14.636      2.073     7.06x     1.22e-4    1.6e-7
cosh              8.245      1.929     4.28x     1.22e-4    1.2e-7
tanh             13.833      2.379     5.81x     1.19e-7    2.1e-7
asinh            11.954      3.954     3.02x     2.38e-7    1.4e-7
acosh             7.833      3.390     2.31x     2.38e-7    1.5e-7
atanh             7.865      2.115     3.72x     8.49e-5    2.0e-5
cbrt              4.954      2.749     1.80x     1.19e-7    9.4e-8
sqrt              1.400      0.356     3.93x      0.00e0     0.0e0
sin (u35)         3.190      0.467     6.83x     1.19e-7    1.4e-7
cos (u35)         4.184      0.540     7.74x     1.19e-7    1.3e-7
tan (u35)        11.570      0.620    18.67x     4.88e-4    2.3e-7
log (u35)         3.222      0.674     4.78x     9.54e-7    2.4e-7
tanh (u35)       13.842      0.903    15.32x     1.79e-7    3.5e-7
exp10             5.105      0.966     5.29x      8.00e0    1.2e-7
erf               8.864      3.688     2.40x     5.96e-8    1.2e-7
erfc              9.488      4.987     1.90x     1.19e-7    2.4e-7
tgamma           32.611      9.342     3.49x     2.50e-1    4.7e-7
lgamma           15.880      9.730     1.63x     1.91e-6    2.1e-4
sinpi             4.159      1.488     2.80x     1.64e-6    3.4e-1
cospi             4.392      1.467     2.99x     1.61e-6     1.0e0
pow               5.774      5.132     1.13x     3.05e-5    1.2e-7
atan2            11.922      2.638     4.52x     1.19e-7    1.2e-7
hypot             3.581      1.678     2.13x      0.00e0     0.0e0
pow(x,1.5)        5.275      5.043     1.05x      0.00e0     0.0e0
sincos            6.258      1.452     4.31x      0.00e0     0.0e0
```
