//! Platform `libm` against the pure-Rust `libm` crate.
//!
//! The crate's main benchmark (`throughput.rs`) measures the kernels against whatever `libm` the
//! host links, which differs per platform. This one answers the other question the README needs:
//! how the host's `libm` compares with the portable Rust one, in both speed and accuracy, for the
//! functions where the two differ.
//!
//! Run with `cargo bench --bench portable_libm`.

use core::hint::black_box;
use std::time::Instant;

use rug::ops::Pow;
use rug::{Assign, Float};

const N: usize = 1 << 20;
const PREC: u32 = 128;

extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
    fn lgamma(x: f64) -> f64;
    fn tgamma(x: f64) -> f64;
    fn acosh(x: f64) -> f64;
}

/// Distance in ULPs between `d` and the higher-precision `c`.
fn count_ulp(d: f64, c: &Float) -> f64 {
    let c2 = c.to_f64();
    if c2 == 0. && d == 0. {
        return 0.;
    }
    let prec = c.prec();
    let mut fry = Float::with_val(prec, d);
    let mut frw = Float::new(prec);
    let (_, e) = c.to_f64_exp();
    frw.assign(Float::u_exp(1, e - 53));
    fry -= c;
    fry /= &frw;
    fry.to_f64().abs()
}

/// Deterministic xorshift64* so the tables are reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

type Unary = fn(f64) -> f64;

struct Case {
    name: &'static str,
    portable: Unary,
    platform: Unary,
    oracle: fn(f64) -> Float,
    lo: f64,
    hi: f64,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "erf",
            portable: libm::erf,
            platform: |x| unsafe { erf(x) },
            oracle: |x: f64| Float::with_val(PREC, x).erf(),
            lo: -4.0,
            hi: 4.0,
        },
        Case {
            name: "erfc",
            portable: libm::erfc,
            platform: |x| unsafe { erfc(x) },
            oracle: |x: f64| Float::with_val(PREC, x).erfc(),
            lo: -4.0,
            hi: 20.0,
        },
        Case {
            name: "lgamma",
            portable: libm::lgamma,
            platform: |x| unsafe { lgamma(x) },
            oracle: |x: f64| Float::with_val(PREC, x).ln_gamma(),
            lo: 0.1,
            hi: 20.0,
        },
        Case {
            name: "tgamma",
            portable: libm::tgamma,
            platform: |x| unsafe { tgamma(x) },
            oracle: |x: f64| Float::with_val(PREC, x).gamma(),
            lo: 0.1,
            hi: 20.0,
        },
        Case {
            name: "acosh",
            portable: libm::acosh,
            platform: |x| unsafe { acosh(x) },
            oracle: |x: f64| Float::with_val(PREC, x).acosh(),
            lo: 1.0,
            hi: 100.0,
        },
    ]
}

fn pow_case() -> (f64, f64, f64, f64) {
    let mut rng = Rng::new(9);
    let xs: Vec<f64> = (0..N).map(|_| 0.25 + rng.unit() * 100.0).collect();

    let t = Instant::now();
    let mut acc = 0.0f64;
    for &x in &xs {
        acc += libm::pow(x, 1.5);
    }
    let portable = t.elapsed().as_secs_f64() / N as f64 * 1e9;
    let t = Instant::now();
    for &x in &xs {
        acc += black_box(x.powf(1.5));
    }
    let platform = t.elapsed().as_secs_f64() / N as f64 * 1e9;
    black_box(acc);

    let (mut mp, mut mg) = (0.0f64, 0.0f64);
    for &x in xs.iter().step_by(37) {
        let want = Float::with_val(PREC, x).pow(&Float::with_val(PREC, 1.5));
        mp = mp.max(count_ulp(libm::pow(x, 1.5), &want));
        mg = mg.max(count_ulp(x.powf(1.5), &want));
    }
    (portable, platform, mp, mg)
}

fn main() {
    println!("platform libm vs the pure-Rust `libm` crate");
    println!("{N} elements, single thread, ns/element; ULP measured against MPFR at {PREC} bits");
    println!();
    println!(
        "{:<8} {:>12} {:>12} {:>10} {:>12} {:>12}",
        "fn", "libm crate", "platform", "ratio", "ULP libm", "ULP platform"
    );

    let (pp, pl, mu, mg) = pow_case();
    println!(
        "{:<8} {:>12.2} {:>12.2} {:>9.2}x {:>12.3} {:>12.3}",
        "pow(x,1.5)",
        pp,
        pl,
        pl / pp,
        mu,
        mg
    );

    for c in cases() {
        let mut rng = Rng::new(11);
        let xs: Vec<f64> = (0..N).map(|_| c.lo + (c.hi - c.lo) * rng.unit()).collect();

        let t = Instant::now();
        let mut acc = 0.0f64;
        for &x in &xs {
            acc += (c.portable)(x);
        }
        let a = t.elapsed().as_secs_f64() / N as f64 * 1e9;

        let t = Instant::now();
        for &x in &xs {
            acc += black_box((c.platform)(x));
        }
        let b = t.elapsed().as_secs_f64() / N as f64 * 1e9;
        black_box(acc);

        let mut m1 = 0.0f64;
        let mut m2 = 0.0f64;
        for &x in xs.iter().step_by(37) {
            let want = (c.oracle)(x);
            m1 = m1.max(count_ulp((c.portable)(x), &want));
            m2 = m2.max(count_ulp((c.platform)(x), &want));
        }

        println!(
            "{:<8} {:>12.2} {:>12.2} {:>9.2}x {:>12.3} {:>12.3}",
            c.name,
            a,
            b,
            b / a,
            m1,
            m2
        );
    }

    println!();
    println!("ratio > 1 means the pure-Rust `libm` is faster than the platform's.");
}
