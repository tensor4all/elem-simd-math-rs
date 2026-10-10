//! The `--no-default-features` path: every entry point must equal the platform's scalar
//! implementation, at lengths that exercise any chunking the caller might do.

#![cfg(not(feature = "simd"))]

use core::mem::MaybeUninit;

use elem_simd_math::{f64 as ops, ElementwiseMath, LengthMismatch};

fn read(out: Vec<MaybeUninit<f64>>) -> Vec<f64> {
    // SAFETY: every entry point under test writes each element before returning `Ok`.
    unsafe { core::mem::transmute::<Vec<MaybeUninit<f64>>, Vec<f64>>(out) }
}

type Unary = fn(&[f64], &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch>;

fn unary_matches_std(name: &str, op: Unary, expected: fn(f64) -> f64) {
    for len in 0..=37 {
        let src: Vec<f64> = (0..len).map(|i| i as f64 * 0.25 - 4.0).collect();
        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        op(&src, &mut out).expect("length check");
        for (lane, (got, &x)) in read(out).iter().zip(src.iter()).enumerate() {
            let want = expected(x);
            let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
            assert!(same, "{name}: len {len}, lane {lane}, {got} != {want}");
        }
    }
}

#[test]
fn scalar_path_matches_std() {
    unary_matches_std("exp", ops::exp, f64::exp);
    unary_matches_std("exp2", ops::exp2, f64::exp2);
    unary_matches_std("exp10", ops::exp10, |v| 10.0f64.powf(v));
    unary_matches_std("expm1", ops::expm1, f64::exp_m1);
    unary_matches_std("log", ops::log, f64::ln);
    unary_matches_std("log2", ops::log2, f64::log2);
    unary_matches_std("log10", ops::log10, f64::log10);
    unary_matches_std("log1p", ops::log1p, f64::ln_1p);
    unary_matches_std("sin", ops::sin, f64::sin);
    unary_matches_std("cos", ops::cos, f64::cos);
    unary_matches_std("tan", ops::tan, f64::tan);
    unary_matches_std("asin", ops::asin, f64::asin);
    unary_matches_std("acos", ops::acos, f64::acos);
    unary_matches_std("atan", ops::atan, f64::atan);
    unary_matches_std("sinh", ops::sinh, f64::sinh);
    unary_matches_std("cosh", ops::cosh, f64::cosh);
    unary_matches_std("tanh", ops::tanh, f64::tanh);
    unary_matches_std("asinh", ops::asinh, f64::asinh);
    unary_matches_std("acosh", ops::acosh, f64::acosh);
    unary_matches_std("atanh", ops::atanh, f64::atanh);
    unary_matches_std("cbrt", ops::cbrt, f64::cbrt);
    unary_matches_std("sqrt", ops::sqrt, f64::sqrt);
}

#[test]
fn scalar_binary_matches_std() {
    for len in 0..=19 {
        let lhs: Vec<f64> = (0..len).map(|i| i as f64 * 0.5 + 0.5).collect();
        let rhs: Vec<f64> = (0..len).map(|i| 1.0 + i as f64 * 0.125).collect();
        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        ops::pow(&lhs, &rhs, &mut out).expect("length check");
        for (lane, ((got, &a), &b)) in read(out).iter().zip(lhs.iter()).zip(rhs.iter()).enumerate()
        {
            let want = a.powf(b);
            assert!(
                got.to_bits() == want.to_bits(),
                "pow lane {lane}: {got} != {want}"
            );
        }

        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        ops::atan2(&lhs, &rhs, &mut out).expect("length check");
        for (lane, ((got, &a), &b)) in read(out).iter().zip(lhs.iter()).zip(rhs.iter()).enumerate()
        {
            let want = a.atan2(b);
            assert!(
                got.to_bits() == want.to_bits(),
                "atan2 lane {lane}: {got} != {want}"
            );
        }

        let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        ops::hypot(&lhs, &rhs, &mut out).expect("length check");
        for (lane, ((got, &a), &b)) in read(out).iter().zip(lhs.iter()).zip(rhs.iter()).enumerate()
        {
            let want = a.hypot(b);
            assert!(
                got.to_bits() == want.to_bits(),
                "hypot lane {lane}: {got} != {want}"
            );
        }
    }
}

#[test]
fn scalar_sincos_matches_std() {
    for len in 0..=19 {
        let src: Vec<f64> = (0..len).map(|i| i as f64 * 0.25 - 2.0).collect();
        let mut s: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        let mut c: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
        ops::sincos(&src, &mut s, &mut c).expect("length check");
        let (s, c) = (read(s), read(c));
        for (lane, &x) in src.iter().enumerate() {
            assert_eq!(s[lane].to_bits(), x.sin().to_bits());
            assert_eq!(c[lane].to_bits(), x.cos().to_bits());
        }
    }
}

#[test]
fn trait_methods_match_std() {
    let src = [0.25f64, 1.5, -2.0, 3.0];
    let mut out = vec![MaybeUninit::uninit(); src.len()];
    <f64 as ElementwiseMath>::log(&src, &mut out).unwrap();
    for (got, &x) in read(out).iter().zip(src.iter()) {
        assert!(got.is_nan() && x.is_nan() || got.to_bits() == x.ln().to_bits());
    }
}

#[test]
fn length_mismatch_is_reported() {
    let src = [1.0f64, 2.0];
    let mut out = vec![MaybeUninit::uninit(); 3];
    assert_eq!(
        ops::exp(&src, &mut out),
        Err(LengthMismatch { src: 2, dst: 3 })
    );
}
