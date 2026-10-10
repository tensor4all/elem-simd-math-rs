//! Throughput of the ported kernels against the platform scalar implementation.
//!
//! Run with `cargo bench`. The sizes match the tensor sizes used in
//! `tensor4all/tenferro-rs#2033`.

use core::hint::black_box;
use core::mem::MaybeUninit;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const SIZES: [usize; 3] = [1 << 16, 1 << 20, 1 << 23];

struct Ops {
    name: &'static str,
    slice: fn(&[f64], &mut [MaybeUninit<f64>]) -> Result<(), elem_simd_math::LengthMismatch>,
    scalar: fn(f64) -> f64,
}

const OPS: &[Ops] = &[
    Ops {
        name: "exp",
        slice: elem_simd_math::f64::exp,
        scalar: f64::exp,
    },
    Ops {
        name: "log",
        slice: elem_simd_math::f64::log,
        scalar: f64::ln,
    },
    Ops {
        name: "tanh",
        slice: elem_simd_math::f64::tanh,
        scalar: f64::tanh,
    },
    Ops {
        name: "expm1",
        slice: elem_simd_math::f64::expm1,
        scalar: f64::exp_m1,
    },
];

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64 elementwise");
    for size in SIZES {
        let src: Vec<f64> = (0..size).map(|i| (i % 1024) as f64 * 0.003 - 1.5).collect();
        let mut dst: Vec<MaybeUninit<f64>> = Vec::with_capacity(size);
        dst.resize_with(size, MaybeUninit::uninit);
        group.throughput(Throughput::Elements(size as u64));
        for op in OPS {
            group.bench_with_input(BenchmarkId::new(op.name, size), &size, |b, _| {
                b.iter(|| {
                    let mut out = core::mem::take(&mut dst);
                    (op.slice)(black_box(&src), &mut out).unwrap();
                    dst = out;
                    black_box(&dst[0]);
                });
            });
            group.bench_with_input(
                BenchmarkId::new(format!("{} (scalar)", op.name), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let mut out = core::mem::take(&mut dst);
                        for (slot, &value) in out.iter_mut().zip(src.iter()) {
                            slot.write((op.scalar)(black_box(value)));
                        }
                        dst = out;
                        black_box(&dst[0]);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
