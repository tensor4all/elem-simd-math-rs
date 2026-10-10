//! `f64` kernels with a 1.0 ULP error bound (SLEEF's `u10` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f64x/u10.rs`).

use super::*;
use crate::doubled::f64::{add_checked, mul_v, square, v_add_checked, DD};

/// Natural exponential function.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn exp<S: Simd>(simd: S, d: F64s<S>) -> F64s<S> {
    let mut u = rint::<S>(d * splat::<S>(simd, R_LN2));
    let q = roundi(simd, u);

    let s = mla::<S>(u, splat::<S>(simd, -L2_U), d);
    let s = mla::<S>(u, splat::<S>(simd, -L2_L), s);

    let s2 = s * s;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    u = poly10::<S>(
        s,
        s2,
        s4,
        s8,
        0.208_127_637_823_716_445_7_e-8,
        0.251_121_070_304_228_802_2_e-7,
        0.275_576_262_816_949_119_2_e-6,
        0.275_572_340_202_538_823_9_e-5,
        0.248_015_868_747_968_626_4_e-4,
        0.198_412_698_985_586_585_e-3,
        0.138_888_888_891_449_779_7_e-2,
        0.833_333_333_331_493_821_e-2,
        0.416_666_666_666_660_259_8_e-1,
        0.166_666_666_666_666_907_2,
    )
    .mul_add(s, splat::<S>(simd, 0.5))
    .mul_add(s, splat::<S>(simd, 1.0))
    .mul_add(s, splat::<S>(simd, 1.0));

    u = ldexp2k::<S>(u, q);

    u = d
        .simd_gt(splat::<S>(simd, 709.782_711_149_557_429_092_172_174_26))
        .select(splat::<S>(simd, f64::INFINITY), u);
    from_bits::<S>(
        d.simd_lt(splat::<S>(simd, -1000.))
            .select(splat_bits(simd, 0), splat_bits(simd, u64::MAX))
            & to_bits::<S>(u),
    )
}

/// `exp(x) - 1` as a double-double, without the final subtraction.
#[inline(always)]
pub fn expk2<S: Simd>(simd: S, d: DD<S>) -> DD<S> {
    let u = d.0 * splat::<S>(simd, R_LN2);
    let dq = rint::<S>(u);
    let q = roundi(simd, dq);

    let s = add_v::<S>(add_v::<S>(d, mul_v::<S>(dq, splat::<S>(simd, -L2_U))), mul_v::<S>(dq, splat::<S>(simd, -L2_L)));

    let s2 = square(s);
    let s4 = square(s2);
    let s8 = s4.0 * s4.0;

    let u = poly10::<S>(
        s.0,
        s2.0,
        s4.0,
        s8,
        0.160_247_221_970_993_207_2_e-9,
        0.209_225_518_356_315_700_7_e-8,
        0.250_523_002_378_264_446_5_e-7,
        0.275_572_480_090_213_530_3_e-6,
        0.275_573_189_238_604_437_3_e-5,
        0.248_015_873_560_581_506_5_e-4,
        0.198_412_698_414_807_185_8_e-3,
        0.138_888_888_888_676_325_5_e-2,
        0.833_333_333_333_334_709_5_e-2,
        0.416_666_666_666_666_990_5_e-1,
    );

    let mut t = add_checked_v(splat::<S>(simd, 0.5), mul_v::<S>(s, splat::<S>(simd, 0.166_666_666_666_666_657_4)));
    t = add_checked_v(splat::<S>(simd, 1.0), mul_v::<S>(t, s));
    t = add_checked_v(splat::<S>(simd, 1.0), mul_v::<S>(t, s));
    t = add_checked(t, mul_v::<S>(s4, u));

    t = DD::<S>::new(ldexp2k::<S>(t.0, q), ldexp2k::<S>(t.1, q));

    let keep = d
        .0
        .simd_lt(splat::<S>(simd, -1000.))
        .select(splat_bits(simd, 0), splat_bits(simd, u64::MAX));
    DD::<S>::new(from_bits::<S>(keep & to_bits::<S>(t.0)), from_bits::<S>(keep & to_bits::<S>(t.1)))
}

/// `exp(x) - 1`.
///
/// The error bound of the returned value is `1.0 ULP`.
#[inline(always)]
pub fn expm1<S: Simd>(simd: S, a: F64s<S>) -> F64s<S> {
    let d = add_v::<S>(expk2(simd, DD::<S>::new(a, splat::<S>(simd, 0.))), splat::<S>(simd, -1.));
    let mut x = d.0;
    x = a
        .simd_gt(splat::<S>(simd, 709.782_712_893_383_996_732_223))
        .select(splat::<S>(simd, f64::INFINITY), x);
    x = a
        .simd_lt(splat::<S>(simd, -36.736_800_569_677_101_399_113_302_437))
        .select(splat::<S>(simd, -1.), x);
    is_neg_zero::<S>(a).select(splat::<S>(simd, -0.), x)
}
