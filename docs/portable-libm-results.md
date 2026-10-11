    Finished `bench` profile [optimized + debuginfo] target(s) in 0.01s
     Running benches/portable_libm.rs (target/release/deps/portable_libm-f24536db7eb4a1ce)
platform libm vs the pure-Rust `libm` crate
1048576 elements, single thread, ns/element; ULP measured against MPFR at 128 bits

fn         libm crate     platform      ratio     ULP libm ULP platform
pow(x,1.5)        47.97        14.49      0.30x        0.794        0.504
erf             28.41        20.04      0.71x        0.729        0.847
erfc            31.32        20.25      0.65x        2.498        2.498
lgamma          19.68        19.10      0.97x        2.292        2.136
tgamma          86.39        78.72      0.91x        7.079        4.162
acosh           11.79         8.92      0.76x        1.108        1.108

ratio > 1 means the pure-Rust `libm` is faster than the platform's.
