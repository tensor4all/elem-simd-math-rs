//! Double-double ("double-float") arithmetic over SIMD vectors.
//!
//! Ported from the `doubled` crate v0.3.3 (`src/lib.rs`, `src/f64x.rs`, `src/f32x.rs`) as used by
//! `sleef-rs` rev `3671d46`. Only the operations the ported kernels call are kept.
//!
//! # Fused formulation
//!
//! Upstream selects between an FMA-based and a split-based formulation at compile time
//! (`cfg(target_feature = "fma")`). This port always uses the FMA-based one, evaluated with
//! [`fearless_simd::SimdFloat::mul_add`], which is a single fused operation wherever the CPU has
//! FMA and degrades to a separate multiply and add elsewhere. On an FMA-capable level the results
//! are therefore identical to upstream's FMA build; on a level without FMA they are what
//! upstream's FMA branch computes when it is compiled without `+fma`. The split-based formulation
//! was measurably slower on the double-double-heavy kernels without being more accurate.
//!
//! # Shape
//!
//! Extension traits cannot be implemented for `S::f64s` (the projection is not a local type), so
//! upstream's method-style extension traits (`MulAsDoubled`, …) become free functions here, and
//! the operators are implemented on the local [`f64::DD`] type.
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
            use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

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
                let r0 = d.0 * d.0;
                DD::new(r0, (d.0 + d.0).mul_add(d.1, d.0.mul_add(d.0, -r0)))
            }

            /// Squares and collapses to a single vector.
            #[inline(always)]
            pub fn square_as_f<S: Simd>(d: DD<S>) -> S::$vec {
                d.0.mul_add(d.0, d.0 * d.1 + d.0 * d.1)
            }

            /// Multiplies and collapses to a single vector.
            #[inline(always)]
            pub fn mul_as_f<S: Simd>(d: DD<S>, other: DD<S>) -> S::$vec {
                d.0.mul_add(other.0, d.1.mul_add(other.0, d.0 * other.1))
            }

            /// Reciprocal of the represented value.
            #[inline(always)]
            pub fn recip<S: Simd>(d: DD<S>) -> DD<S> {
                let one = S::$vec::splat(d.0.token(), 1.0 as $elem);
                let q0 = one / d.0;
                DD::new(
                    q0,
                    q0 * (-d.1).mul_add(q0, (-d.0).mul_add(q0, one)),
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
                let r0 = x * y;
                DD::new(r0, x.mul_add(y, -r0))
            }

            /// Reciprocal of a vector as an unevaluated sum.
            #[inline(always)]
            pub fn recip_as_doubled<S: Simd>(x: S::$vec) -> DD<S> {
                let one = S::$vec::splat(x.token(), 1.0 as $elem);
                let q0 = one / x;
                DD::new(q0, q0 * (-x).mul_add(q0, one))
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
                let r0 = d.0 * v;
                DD::new(r0, d.1.mul_add(v, d.0.mul_add(v, -r0)))
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
                *d += DD::new(other, S::$vec::splat(other.token(), 0.0 as $elem));
            }

            /// Multiplies by a vector in place, without normalizing.
            #[inline(always)]
            pub fn mul_assign_v<S: Simd>(d: &mut DD<S>, other: S::$vec) {
                *d *= DD::new(other, S::$vec::splat(other.token(), 0.0 as $elem));
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
                    let r0 = self.0 * other.0;
                    DD::new(
                        r0,
                        self.0.mul_add(
                            other.1,
                            self.1.mul_add(other.0, self.0.mul_add(other.0, -r0)),
                        ),
                    )
                }
            }


            impl<S: Simd> Sub for DD<S> {
                type Output = Self;
                #[inline(always)]
                fn sub(self, other: Self) -> Self {
                    self + (-other)
                }
            }

            impl<S: Simd> Div for DD<S> {
                type Output = Self;
                #[inline(always)]
                fn div(self, other: Self) -> Self {
                    let one = S::$vec::splat(self.0.token(), 1.0 as $elem);
                    let t = one / other.0;

                    let q0 = self.0 * t;
                    let u = t.mul_add(self.0, -q0);
                    let mut q1 = (-other.1).mul_add(t, (-other.0).mul_add(t, one));
                    q1 = q0.mul_add(q1, self.1.mul_add(t, u));

                    DD::new(q0, q1)
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
