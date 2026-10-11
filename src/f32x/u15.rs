//! `f32` kernels with a 1.5 ULP error bound (SLEEF's `u15` tier).
//!
//! Ported from `sleef-rs` rev `3671d46` (`src/f32x/u15.rs`). The double-double `exp(x) - 1`
//! helper lives in upstream's `src/f32x.rs`; as in the `f64` port it comes from the `u10` tier
//! here.

use super::u10::expk2f;
use super::*;
use crate::doubled::f32::{mul_v, DD};

/// Complementary error function.
///
/// The error bound of the returned value is `max(1.5 ULP, f32::MIN_POSITIVE)`.
///
/// Upstream writes the three polynomial groups after `let mut d = u * t;` as
/// `Doubled::select4(o0, o1, o2, c0, ..)` over `f64` coefficients, which the `doubled` crate
/// converts with `Doubled::<f32>::from_f64`; this port has no such constructor, so each
/// coefficient is split into its `f32` pair here and the two halves are selected separately.
#[inline(always)]
pub fn erfc<S: Simd>(simd: S, a: F32s<S>) -> F32s<S> {
    let s = a;
    let a = a.abs();
    let o0 = a.simd_lt(splat(simd, 1.));
    let o1 = a.simd_lt(splat(simd, 2.2));
    let o2 = a.simd_lt(splat(simd, 4.3));
    let o3 = a.simd_lt(splat(simd, 10.1));

    let u = select_doubled(
        simd,
        o1,
        DD::new(a, splat(simd, 0.)),
        DD::new(splat(simd, 1.), splat(simd, 0.)) / DD::new(a, splat(simd, 0.)),
    );

    let mut t = select4(
        simd,
        o0,
        o1,
        o2,
        -0.863_804_161_8_e-4,
        -0.623_697_724_2_e-5,
        -0.386_950_403_5,
        0.111_534_416_7_e+1,
    );
    t = mla(
        simd,
        t,
        u.0,
        select4(
            simd,
            o0,
            o1,
            o2,
            0.600_016_617_7_e-3,
            0.574_982_150_3_e-4,
            0.128_807_723_5_e+1,
            -0.945_490_419_9,
        ),
    );
    t = mla(
        simd,
        t,
        u.0,
        select4(
            simd,
            o0,
            o1,
            o2,
            -0.166_570_360_3_e-2,
            0.600_285_147_8_e-5,
            -0.181_680_321_7_e+1,
            -0.366_725_951_4,
        ),
    );
    t = mla(
        simd,
        t,
        u.0,
        select4(
            simd,
            o0,
            o1,
            o2,
            0.179_515_627_7_e-3,
            -0.285_103_637_7_e-2,
            0.124_915_087_2_e+1,
            0.715_566_337_1,
        ),
    );
    t = mla(
        simd,
        t,
        u.0,
        select4(
            simd,
            o0,
            o1,
            o2,
            0.191_410_612_3_e-1,
            0.226_051_807_4_e-1,
            -0.132_885_798_8,
            -0.126_294_726_5_e-1,
        ),
    );

    let mut d = mul_v(u, t);
    d += DD::new(
        select4(
            simd,
            o0,
            o1,
            o2,
            -0.102_775_359_343_930_288_081_655_368_891,
            -0.105_247_583_459_338_632_253_369_014_063,
            -0.482_365_310_333_045_318_680_618_892_669,
            -0.498_961_546_254_537_647_970_305_302_739,
        ),
        select4(
            simd,
            o0,
            o1,
            o2,
            -1.680_298_8_e-9,
            3.506_222_2_e-9,
            -1.409_371_5_e-10,
            -8.178_137_e-9,
        ),
    );
    d *= u;
    d += DD::new(
        select4(
            simd,
            o0,
            o1,
            o2,
            -0.636_619_483_208_481_931_303_752_546_439,
            -0.635_609_463_574_589_034_216_723_775_292,
            -0.134_450_203_224_533_979_217_859_332_703_e-2,
            -0.471_199_543_422_848_492_080_722_832_666_e-4,
        ),
        select4(
            simd,
            o0,
            o1,
            o2,
            2.505_796_6_e-8,
            -1.561_850_4_e-8,
            5.636_847_e-11,
            -1.545_792_1_e-12,
        ),
    );
    d *= u;
    d += DD::new(
        select4(
            simd,
            o0,
            o1,
            o2,
            -0.112_837_917_790_537_404_939_545_770_596_e+1,
            -0.112_855_987_376_668_622_084_547_028_949_e+1,
            -0.572_319_781_150_472_949_561_786_101_080,
            -0.572_364_030_327_966_044_425_932_623_525,
        ),
        select4(
            simd,
            o0,
            o1,
            o2,
            4.782_552_3_e-8,
            -4.596_212_e-8,
            2.447_162_8_e-8,
            1.940_558_2_e-9,
        ),
    );

    let mut x = mul_v(select_doubled(simd, o1, d, DD::new(-a, splat(simd, 0.))), a);
    x = select_doubled(simd, o1, x, x + d);

    x = expk2f(simd, x);
    x = select_doubled(simd, o1, x, x * u);

    let mut r = o3.select(x.0 + x.1, splat(simd, 0.));
    r = s.is_sign_negative().select(splat(simd, 2.) - r, r);
    s.is_nan().select(splat(simd, f32::NAN), r)
}
