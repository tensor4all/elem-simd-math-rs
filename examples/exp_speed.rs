//! Ad-hoc timing check used while tuning the benchmark harness.
use core::hint::black_box;
use core::mem::MaybeUninit;
use elem_simd_math::f64 as ops;
use std::time::Instant;

const N: usize = 1 << 23;

fn main() {
    let src: Vec<f64> = (0..N)
        .map(|i| {
            let t = i as f64 / N as f64;
            let x = t * 2.0 - 1.0;
            match i % 4 {
                0 => x * 0.5,
                1 => (x + 1.5).abs() + 0.125,
                2 => x * 8.0,
                _ => x * 0.125 + 0.75,
            }
        })
        .collect();
    let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); N];

    for (label, plain) in [("plain", true), ("black_box", false)] {
        ops::exp(&src, &mut out).unwrap();
        let t = Instant::now();
        if plain {
            ops::exp(black_box(&src), &mut out).unwrap();
        } else {
            let f: fn(
                &[f64],
                &mut [MaybeUninit<f64>],
            ) -> Result<(), elem_simd_math::LengthMismatch> = ops::exp;
            f(black_box(&src), &mut out).unwrap();
        }
        println!(
            "{label}: {:.3} ns/elem",
            t.elapsed().as_secs_f64() / N as f64 * 1e9
        );
    }

    let t = Instant::now();
    for (slot, &x) in out.iter_mut().zip(src.iter()) {
        slot.write(black_box(x).exp());
    }
    println!(
        "scalar libm: {:.3} ns/elem",
        t.elapsed().as_secs_f64() / N as f64 * 1e9
    );
}
