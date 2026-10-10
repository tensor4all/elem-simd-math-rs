//! Double-double ("double-float") arithmetic over SIMD vectors.
//!
//! Ported from the `doubled` crate v0.3.3 (`src/lib.rs`, `src/f64x.rs`, `src/f32x.rs`) as used by
//! `sleef-rs` rev `3671d46`. Only the operations the ported kernels call are kept.
//!
//! # Target-feature independence
//!
//! Upstream selects between an FMA-based and a split-based formulation at compile time
//! (`cfg(target_feature = "fma")`). This port always uses the split-based formulation: it is the
//! one upstream ships and tests for targets without FMA, it uses no fused operations, and it
//! makes results independent of whether the build happened to enable FMA.
//!
//! # Shape
//!
//! Extension traits cannot be implemented for `S::f64s` (the projection is not a local type), so
//! upstream's method-style extension traits (`MulAsDoubled`, …) become free functions here, and
//! the operators are implemented on the local [`f64::DD`] / [`f32::DD`] types.
//!
//! Every function is defined once by the `dd_family!` macro and instantiated for `f64` and `f32`,
//! so the two families cannot drift apart.

macro_rules! dd_family {
    (
        $(#[$meta:meta])*
        mod $modname:ident, vector = $vec:ident, bits = $bits:ident, mask = $mask:expr, elem = $elem:ty
    ) => {
        $(#[$meta])*
        pub mod $modname {
            use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

            use fearless_simd::prelude::*;
            use fearless_simd::{ExtractToken, Simd};

            /// A value carried as an unevaluated sum of two vectors `(hi, lo)`.
            ///
            /// `hi` holds the leading part and `lo` the rounding error, so the pair represents
            /// `hi + lo` to roughly twice the significand width of the element type.
            #[derive(Clone, Copy, Debug, PartialEq)]
            pub struct DD<S: Simd>(pub S::$vec, pub S::$vec);

            impl<S: Simd> DD<S> {
                /// Pairs `hi` and `lo` without normalizing or ordering them.
                #[inline(always)]
                pub const fn new(hi: S::$vec, lo: S::$vec) -> Self {
                    Self(hi, lo)
                }

                /// Broadcasts a scalar double-double.
                #[inline(always)]
                pub fn splat(simd: S, value: ($elem, $elem)) -> Self {
                    Self(
                        S::$vec::splat(simd, value.0),
                        S::$vec::splat(simd, value.1),
                    )
                }
            }

            /// Sign/exponent-preserving high part of a Veltkamp split.
            #[inline(always)]
            pub fn split_upper<S: Simd>(x: S::$vec) -> S::$vec {
                let bits = x.bitcast::<S::$bits>() & S::$bits::splat(x.token(), $mask);
                bits.bitcast()
            }

            /// Re-establishes `|hi| >= |lo|` after an approximate addition.
            #[inline(always)]
            pub fn normalize<S: Simd>(d: DD<S>) -> DD<S> {
                let s0 = d.0 + d.1;
                DD::new(s0, d.0 - s0 + d.1)
            }

            /// Multiplies both components by `other`.
            #[inline(always)]
            pub fn scale<S: Simd>(d: DD<S>, other: S::$vec) -> DD<S> {
                DD::new(d.0 * other, d.1 * other)
            }

            /// Adds `other`, assuming `|self| >= |other|`.
            #[inline(always)]
            pub fn add_checked<S: Simd>(d: DD<S>, other: DD<S>) -> DD<S> {
                let r0 = d.0 + other.0;
                DD::new(r0, d.0 - r0 + other.0 + d.1 + other.1)
            }

            /// Adds `other`, assuming `|self| >= |other|`.
            #[inline(always)]
            pub fn add_checked_v<S: Simd>(d: DD<S>, other: S::$vec) -> DD<S> {
                let r0 = d.0 + other;
                DD::new(r0, d.0 - r0 + other + d.1)
            }

            /// Subtracts `other`, assuming `|self| >= |other|`.
            #[inline(always)]
            pub fn sub_checked<S: Simd>(d: DD<S>, other: DD<S>) -> DD<S> {
                let r0 = d.0 - other.0;
                DD::new(r0, d.0 - r0 - other.0 + d.1 - other.1)
            }

            /// Squares the represented value.
            #[inline(always)]
            pub fn square<S: Simd>(d: DD<S>) -> DD<S> {
                let xh = split_upper::<S>(d.0);
                let xl = d.0 - xh;
                let r0 = d.0 * d.0;
                DD::new(
                    r0,
                    xh * xh + (-r0) + (xh + xh) * xl + xl * xl + d.0 * (d.1 + d.1),
                )
            }

            /// Squares and collapses to a single vector.
            #[inline(always)]
            pub fn square_as_f<S: Simd>(d: DD<S>) -> S::$vec {
                let xh = split_upper::<S>(d.0);
                let xl = d.0 - xh;
                xh * d.1 + xh * d.1 + xl * xl + (xh * xl + xh * xl) + xh * xh
            }

            /// Multiplies and collapses to a single vector.
            #[inline(always)]
            pub fn mul_as_f<S: Simd>(d: DD<S>, other: DD<S>) -> S::$vec {
                let xh = split_upper::<S>(d.0);
                let xl = d.0 - xh;
                let yh = split_upper::<S>(other.0);
                let yl = other.0 - yh;
                d.1 * yh + xh * other.1 + xl * yl + xh * yl + xl * yh + xh * yh
            }

            /// Reciprocal of the represented value.
            #[inline(always)]
            pub fn recip<S: Simd>(d: DD<S>) -> DD<S> {
                let simd = d.0.token();
                let t = S::$vec::splat(simd, 1.0 as $elem) / d.0;
                let dh = split_upper::<S>(d.0);
                let dl = d.0 - dh;
                let th = split_upper::<S>(t);
                let tl = t - th;
                let q0 = t;
                DD::new(
                    q0,
                    t * (S::$vec::splat(simd, 1.0 as $elem)
                        - dh * th
                        - dh * tl
                        - dl * th
                        - dl * tl
                        - d.1 * t),
                )
            }

            /// Square root of the represented value.
            #[inline(always)]
            pub fn sqrt<S: Simd>(d: DD<S>) -> DD<S> {
                let simd = d.0.token();
                let t = (d.0 + d.1).sqrt();
                scale(
                    (d + mul_as_doubled(t, t)) * recip_as_doubled(t),
                    S::$vec::splat(simd, 0.5 as $elem),
                )
            }

            /// The exact product `x * y` split into two components.
            #[inline(always)]
            pub fn mul_as_doubled<S: Simd>(x: S::$vec, y: S::$vec) -> DD<S> {
                let xh = split_upper::<S>(x);
                let xl = x - xh;
                let yh = split_upper::<S>(y);
                let yl = y - yh;
                let r0 = x * y;
                DD::new(r0, xh * yh + (-r0) + xl * yh + xh * yl + xl * yl)
            }

            /// Reciprocal of a vector as an unevaluated sum.
            #[inline(always)]
            pub fn recip_as_doubled<S: Simd>(x: S::$vec) -> DD<S> {
                let simd = x.token();
                let t = S::$vec::splat(simd, 1.0 as $elem) / x;
                let dh = split_upper::<S>(x);
                let dl = x - dh;
                let th = split_upper::<S>(t);
                let tl = t - th;
                DD::new(
                    t,
                    t * (S::$vec::splat(simd, 1.0 as $elem) - dh * th - dh * tl - dl * th - dl * tl),
                )
            }

            /// The exact sum `x + y` split into two components.
            #[inline(always)]
            pub fn add_as_doubled<S: Simd>(x: S::$vec, y: S::$vec) -> DD<S> {
                let r0 = x + y;
                let v = r0 - x;
                DD::new(r0, x - (r0 - v) + (y - v))
            }

            /// The exact sum `x + y` split into two components, assuming `|x| >= |y|`.
            #[inline(always)]
            pub fn add_checked_as_doubled<S: Simd>(x: S::$vec, y: S::$vec) -> DD<S> {
                let r0 = x + y;
                DD::new(r0, x - r0 + y)
            }

            /// `d + v`, without normalizing.
            #[inline(always)]
            pub fn add_v<S: Simd>(d: DD<S>, v: S::$vec) -> DD<S> {
                let r0 = d.0 + v;
                let x = r0 - d.0;
                DD::new(r0, d.0 - (r0 - x) + (v - x) + d.1)
            }

            /// `d * v`, without normalizing.
            #[inline(always)]
            pub fn mul_v<S: Simd>(d: DD<S>, v: S::$vec) -> DD<S> {
                let xh = split_upper::<S>(d.0);
                let xl = d.0 - xh;
                let yh = split_upper::<S>(v);
                let yl = v - yh;
                let r0 = d.0 * v;
                DD::new(r0, xh * yh + (-r0) + xl * yh + xh * yl + xl * yl + d.1 * v)
            }

            /// `v + d`, matching upstream's `T + Doubled<T>`.
            #[inline(always)]
            pub fn v_add_checked<S: Simd>(v: S::$vec, other: DD<S>) -> DD<S> {
                let r0 = v + other.0;
                DD::new(r0, v - r0 + other.0 + other.1)
            }

            /// Adds a vector in place, without normalizing.
            #[inline(always)]
            pub fn add_assign_v<S: Simd>(d: &mut DD<S>, other: S::$vec) {
                *d = *d + DD::new(other, S::$vec::splat(other.token(), 0.0 as $elem));
            }

            /// Multiplies by a vector in place, without normalizing.
            #[inline(always)]
            pub fn mul_assign_v<S: Simd>(d: &mut DD<S>, other: S::$vec) {
                *d = *d * DD::new(other, S::$vec::splat(other.token(), 0.0 as $elem));
            }

            impl<S: Simd> Add for DD<S> {
                type Output = Self;
                #[inline(always)]
                fn add(self, other: Self) -> Self {
                    let r0 = self.0 + other.0;
                    let v = r0 - self.0;
                    DD::new(r0, self.0 - (r0 - v) + (other.0 - v) + (self.1 + other.1))
                }
            }

            
            impl<S: Simd> Neg for DD<S> {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    DD::new(-self.0, -self.1)
                }
            }

            impl<S: Simd> Mul for DD<S> {
                type Output = Self;
                #[inline(always)]
                fn mul(self, other: Self) -> Self {
                    let xh = split_upper::<S>(self.0);
                    let xl = self.0 - xh;
                    let yh = split_upper::<S>(other.0);
                    let yl = other.0 - yh;
                    let r0 = self.0 * other.0;
                    DD::new(
                        r0,
                        xh * yh + (-r0)
                            + xl * yh
                            + xh * yl
                            + xl * yl
                            + self.0 * other.1
                            + self.1 * other.0,
                    )
                }
            }

            
            impl<S: Simd> AddAssign for DD<S> {
                #[inline(always)]
                fn add_assign(&mut self, other: Self) {
                    *self = *self + other;
                }
            }

            
            impl<S: Simd> MulAssign for DD<S> {
                #[inline(always)]
                fn mul_assign(&mut self, other: Self) {
                    *self = *self * other;
                }
            }

                    }
    };
}

dd_family! {
    /// Double-double arithmetic over native-width `f64` vectors.
    mod f64,
    vector = f64s,
    bits = u64s,
    mask = 0x_ffff_ffff_f800_0000_u64,
    elem = f64
}

dd_family! {
    /// Double-double arithmetic over native-width `f32` vectors.
    mod f32,
    vector = f32s,
    bits = u32s,
    mask = 0x_ffff_f000_u32,
    elem = f32
}
