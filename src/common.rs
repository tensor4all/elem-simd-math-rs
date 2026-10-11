//! Type aliases for the `f64` kernel family, and the translation table they follow.
//!
//! `sleef-rs` expresses these as extension traits on `Simd<f64, N>`/`Simd<f32, N>`. Extension
//! traits cannot be implemented for `S::f64s` here (the projection is not a local type, and two
//! such impls conflict), so they are free functions parameterised by the SIMD token. The
//! translation table used throughout this crate is:
//!
//! | `sleef-rs` | `elem-simd-math` |
//! |---|---|
//! | `F64x<N>` | `F64s<S>` |
//! | `x.mla(y, z)` | `x.mul_add(y, z)` |
//! | `F64x::splat(v)` | `F64s::<S>::splat(simd, v)` |
//! | `F64x::poly8(…)` | `poly8(simd, …)` |
//! | `m.select_splat(l, r)` | `select_splat(m, l, r)` |
//! | `m.select(l, r)` | `m.select(l, r)` |
//! | `x.to_bits()` / `F64x::from_bits(b)` | `to_bits(x)` / `from_bits(b)` |

use fearless_simd::Simd;

/// Native-width `f64` vector for SIMD level `S`.
pub type F64s<S> = <S as Simd>::f64s;
/// Native-width `u64` vector for SIMD level `S`.
pub type U64s<S> = <S as Simd>::u64s;
/// Native-width `i64` vector for SIMD level `S`.
pub type I64s<S> = <S as Simd>::i64s;
/// Native-width mask for `f64` lanes.
pub type M64s<S> = <S as Simd>::mask64s;

/// Native-width `f32` vector for SIMD level `S`.
pub type F32s<S> = <S as Simd>::f32s;
/// Native-width `u32` vector for SIMD level `S`.
pub type U32s<S> = <S as Simd>::u32s;
/// Native-width `i32` vector for SIMD level `S`.
pub type I32s<S> = <S as Simd>::i32s;
/// Native-width mask for `f32` lanes.
pub type M32s<S> = <S as Simd>::mask32s;

use fearless_simd::prelude::*;

/// `x * y + z`, fused wherever the CPU supports it.
///
/// `sleef-rs` selects a formulation at compile time with `cfg(target_feature = "fma")`. This crate
/// always takes the FMA formulation, which upstream also asserts the documented bounds for, and
/// evaluates it with a fused operation on every level that has one. Shared by the `f64` and `f32`
/// kernel families, hence generic over the vector type.
#[inline(always)]
pub fn mla<S: Simd, V: SimdFloat<S>>(_simd: S, x: V, y: V, z: V) -> V {
    x.mul_add(y, z)
}

/// `x * y - z`, fused wherever the CPU supports it. See [`mla`].
#[inline(always)]
pub fn mul_sub<S: Simd, V: SimdFloat<S>>(_simd: S, x: V, y: V, z: V) -> V {
    x.mul_sub(y, z)
}

/// `z - x * y`, fused wherever the CPU supports it. See [`mla`].
#[inline(always)]
pub fn neg_mul_add<S: Simd, V: SimdFloat<S>>(_simd: S, x: V, y: V, z: V) -> V {
    (-x).mul_add(y, z)
}

/// Estrin polynomial evaluation with scalar coefficients, degree 2.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly2<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c1: V::Element,
    c0: V::Element,
) -> V {
    let _ = (x2, x4);
    mla(simd, x, V::splat(simd, c1), V::splat(simd, c0))
}

/// Estrin polynomial evaluation with scalar coefficients, degree 3.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly3<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(simd, x2, V::splat(simd, c2), poly2(simd, x, x2, x4, c1, c0))
}

/// Estrin polynomial evaluation with scalar coefficients, degree 4.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly4<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    let _ = x4;
    mla(
        simd,
        x2,
        mla(simd, x, V::splat(simd, c3), V::splat(simd, c2)),
        mla(simd, x, V::splat(simd, c1), V::splat(simd, c0)),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 5.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly5<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x4,
        V::splat(simd, c4),
        poly4(simd, x, x2, x4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 6.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly6<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x4,
        poly2(simd, x, x2, x4, c5, c4),
        poly4(simd, x, x2, x4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 7.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly7<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x4,
        poly3(simd, x, x2, x4, c6, c5, c4),
        poly4(simd, x, x2, x4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 8.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly8<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x4,
        poly4(simd, x, x2, x4, c7, c6, c5, c4),
        poly4(simd, x, x2, x4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 9.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly9<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        V::splat(simd, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 10.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly10<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly2(simd, x, x2, x4, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 11.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly11<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly3(simd, x, x2, x4, ca, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 12.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly12<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly4(simd, x, x2, x4, cb, ca, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 13.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly13<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly5(simd, x, x2, x4, cc, cb, ca, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 14.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly14<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly6(simd, x, x2, x4, cd, cc, cb, ca, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 15.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly15<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly7(simd, x, x2, x4, ce, cd, cc, cb, ca, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 16.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly16<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    cf: V::Element,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x8,
        poly8(simd, x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
        poly8(simd, x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 17.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly17<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    x16: V,
    d0: V::Element,
    cf: V::Element,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x16,
        V::splat(simd, d0),
        poly16(
            simd, x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 18.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly18<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    x16: V,
    d1: V::Element,
    d0: V::Element,
    cf: V::Element,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x16,
        poly2(simd, x, x2, x4, d1, d0),
        poly16(
            simd, x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 19.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly19<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    x16: V,
    d2: V::Element,
    d1: V::Element,
    d0: V::Element,
    cf: V::Element,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x16,
        poly3(simd, x, x2, x4, d2, d1, d0),
        poly16(
            simd, x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 20.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly20<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    x16: V,
    d3: V::Element,
    d2: V::Element,
    d1: V::Element,
    d0: V::Element,
    cf: V::Element,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x16,
        poly4(simd, x, x2, x4, d3, d2, d1, d0),
        poly16(
            simd, x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}

/// Estrin polynomial evaluation with scalar coefficients, degree 21.
///
/// Every `polyN` takes the same `(simd, x, x2, x4 [, x8] [, x16], coefficients...)`
/// shape so the nesting below stays uniform; the higher powers are unused below degree 5.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn poly21<S: Simd, V: SimdFloat<S> + Copy>(
    simd: S,
    x: V,
    x2: V,
    x4: V,
    x8: V,
    x16: V,
    d4: V::Element,
    d3: V::Element,
    d2: V::Element,
    d1: V::Element,
    d0: V::Element,
    cf: V::Element,
    ce: V::Element,
    cd: V::Element,
    cc: V::Element,
    cb: V::Element,
    ca: V::Element,
    c9: V::Element,
    c8: V::Element,
    c7: V::Element,
    c6: V::Element,
    c5: V::Element,
    c4: V::Element,
    c3: V::Element,
    c2: V::Element,
    c1: V::Element,
    c0: V::Element,
) -> V {
    mla(
        simd,
        x16,
        poly5(simd, x, x2, x4, d4, d3, d2, d1, d0),
        poly16(
            simd, x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
        ),
    )
}
