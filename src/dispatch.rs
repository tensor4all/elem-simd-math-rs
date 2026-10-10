//! Slice entry points: elementwise application over contiguous buffers.
//!
//! Every function writes `dst[i] = f(src[i])` for all `i`, using the best SIMD level the CPU
//! supports at runtime. `dst` may be uninitialized storage: each element is written exactly once
//! on success, so callers do not have to zero-fill a buffer that is fully overwritten.
//!
//! Without the (default) `simd` feature the implementations fall back to the platform's scalar
//! transcendental functions. Functions with no scalar equivalent in the standard library
//! (`erf`, `erfc`, `tgamma`, `lgamma`, `sinpi`, `cospi`, `sincospi`) are only available with
//! `simd`; they would otherwise need a second scalar implementation of SLEEF to stay honest
//! about their error bounds.

use core::mem::MaybeUninit;

/// The source and destination lengths differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LengthMismatch {
    /// Number of elements in the first input slice.
    pub src: usize,
    /// Number of elements in the destination slice.
    pub dst: usize,
}

impl core::fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "input has {} elements but the destination has {}",
            self.src, self.dst
        )
    }
}

impl std::error::Error for LengthMismatch {}

/// Elementwise math over contiguous slices.
///
/// Implemented for `f32` and `f64`. The accuracy tier and error bound of each function are
/// documented on the kernels in [`crate::f64x`] and [`crate::f32x`]; each method selects the most
/// accurate tier available for that function. The tier-specific entry points are the `*_u35`
/// free functions in [`crate::f64`].
pub trait ElementwiseMath: Copy + 'static {
    /// Natural exponential function; 1.0 ULP.
    fn exp(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Base-2 exponential function; 1.0 ULP.
    fn exp2(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Base-10 exponential function; 1.09 ULP.
    fn exp10(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Natural exponential function minus one; 1.0 ULP.
    fn expm1(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Natural logarithm; 1.0 ULP.
    fn log(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Base-2 logarithm; 1.0 ULP.
    fn log2(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Base-10 logarithm; 1.0 ULP.
    fn log10(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Logarithm of one plus the argument; 1.0 ULP.
    fn log1p(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Sine function; 1.0 ULP.
    fn sin(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Cosine function; 1.0 ULP.
    fn cos(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Tangent function; 1.0 ULP.
    fn tan(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Arc sine function; 1.0 ULP.
    fn asin(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Arc cosine function; 1.0 ULP.
    fn acos(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Arc tangent function; 1.0 ULP.
    fn atan(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Hyperbolic sine function; 1.0 ULP.
    fn sinh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Hyperbolic cosine function; 1.0 ULP.
    fn cosh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Hyperbolic tangent function; 1.0 ULP.
    fn tanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Inverse hyperbolic sine function; 1.0 ULP.
    fn asinh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Inverse hyperbolic cosine function; 1.0 ULP.
    fn acosh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Inverse hyperbolic tangent function; 1.0 ULP.
    fn atanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Cube root function; 1.0 ULP.
    fn cbrt(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Square root function; `0.5001 ULP`.
    fn sqrt(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Argument raised to an argument power; 1.0 ULP.
    fn pow(lhs: &[Self], rhs: &[Self], dst: &mut [MaybeUninit<Self>])
        -> Result<(), LengthMismatch>;
    /// Arc tangent of two arguments; 1.0 ULP.
    fn atan2(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch>;
    /// Euclidean distance in two dimensions; `0.5001 ULP`.
    fn hypot(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch>;
    /// Sine and cosine together; 1.0 ULP.
    fn sincos(
        src: &[Self],
        sin: &mut [MaybeUninit<Self>],
        cos: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch>;
}

/// Writes `f(src[i])` into every slot of `dst`.
#[cfg_attr(feature = "simd", allow(dead_code))]
pub(crate) fn scalar_apply<T: Copy>(src: &[T], dst: &mut [MaybeUninit<T>], f: impl Fn(T) -> T) {
    for (slot, &value) in dst.iter_mut().zip(src.iter()) {
        slot.write(f(value));
    }
}

/// Writes `f(src[i])` into every slot of the two destinations.
#[cfg_attr(feature = "simd", allow(dead_code))]
pub(crate) fn scalar_apply_ff<T: Copy>(
    src: &[T],
    first: &mut [MaybeUninit<T>],
    second: &mut [MaybeUninit<T>],
    f: impl Fn(T) -> (T, T),
) {
    for ((slot_a, slot_b), &value) in first.iter_mut().zip(second.iter_mut()).zip(src.iter()) {
        let (a, b) = f(value);
        slot_a.write(a);
        slot_b.write(b);
    }
}

/// Writes `f(lhs[i], rhs[i])` into every slot of `dst`.
#[cfg_attr(feature = "simd", allow(dead_code))]
pub(crate) fn scalar_apply2<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    dst: &mut [MaybeUninit<T>],
    f: impl Fn(T, T) -> T,
) {
    for ((slot, &a), &b) in dst.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        slot.write(f(a, b));
    }
}

pub mod f64 {
    //! `f64` slice entry points.

    #[cfg(feature = "simd")]
    use super::scalar_apply;
    use super::LengthMismatch;
    #[cfg(not(feature = "simd"))]
    use super::{scalar_apply, scalar_apply2, scalar_apply_ff};
    use core::mem::MaybeUninit;

    #[cfg(feature = "simd")]
    use fearless_simd::prelude::*;
    #[cfg(feature = "simd")]
    use fearless_simd::Simd;

    fn lengths(src: usize, dst: usize) -> Result<(), LengthMismatch> {
        if src == dst {
            Ok(())
        } else {
            Err(LengthMismatch { src, dst })
        }
    }

    fn lengths2(lhs: usize, rhs: usize, dst: usize) -> Result<(), LengthMismatch> {
        if lhs == rhs && lhs == dst {
            Ok(())
        } else {
            Err(LengthMismatch {
                src: lhs.min(rhs),
                dst,
            })
        }
    }

    /// Applies a unary `kernel` lane-wise over `src`, writing `dst`.
    #[cfg(feature = "simd")]
    fn run<S: Simd>(
        simd: S,
        kernel: fn(S, crate::common::F64s<S>) -> crate::common::F64s<S>,
        src: &[f64],
        dst: &mut [MaybeUninit<f64>],
    ) {
        let n = crate::common::F64s::<S>::LEN;
        let mut chunks = src.chunks_exact(n);
        let mut out_chunks = dst.chunks_exact_mut(n);
        for (chunk, out) in (&mut chunks).zip(&mut out_chunks) {
            let v = crate::common::F64s::<S>::from_slice(simd, chunk);
            let r = kernel(simd, v);
            let arr = r.to_array();
            for (slot, &value) in out.iter_mut().zip(arr.as_ref().iter()) {
                slot.write(value);
            }
        }
        for (slot, &value) in out_chunks
            .into_remainder()
            .iter_mut()
            .zip(chunks.remainder().iter())
        {
            slot.write(value);
        }
    }

    /// Applies a binary `kernel` lane-wise over `lhs` and `rhs`, writing `dst`.
    #[cfg(feature = "simd")]
    fn run2<S: Simd>(
        simd: S,
        kernel: fn(S, crate::common::F64s<S>, crate::common::F64s<S>) -> crate::common::F64s<S>,
        lhs: &[f64],
        rhs: &[f64],
        dst: &mut [MaybeUninit<f64>],
        scalar: fn(f64, f64) -> f64,
    ) {
        let n = crate::common::F64s::<S>::LEN;
        let mut lhs_chunks = lhs.chunks_exact(n);
        let mut rhs_chunks = rhs.chunks_exact(n);
        let mut out_chunks = dst.chunks_exact_mut(n);
        for ((a, b), out) in (&mut lhs_chunks).zip(&mut rhs_chunks).zip(&mut out_chunks) {
            let va = crate::common::F64s::<S>::from_slice(simd, a);
            let vb = crate::common::F64s::<S>::from_slice(simd, b);
            let r = kernel(simd, va, vb);
            let arr = r.to_array();
            for (slot, &value) in out.iter_mut().zip(arr.as_ref().iter()) {
                slot.write(value);
            }
        }
        for ((slot, &a), &b) in out_chunks
            .into_remainder()
            .iter_mut()
            .zip(lhs_chunks.remainder().iter())
            .zip(rhs_chunks.remainder().iter())
        {
            slot.write(scalar(a, b));
        }
    }

    /// Applies a `kernel` returning two vectors, writing `sin` and `cos`.
    #[cfg(feature = "simd")]
    fn run_ff<S: Simd>(
        simd: S,
        kernel: fn(S, crate::common::F64s<S>) -> (crate::common::F64s<S>, crate::common::F64s<S>),
        src: &[f64],
        first: &mut [MaybeUninit<f64>],
        second: &mut [MaybeUninit<f64>],
        scalar: fn(f64) -> (f64, f64),
    ) {
        let n = crate::common::F64s::<S>::LEN;
        let mut chunks = src.chunks_exact(n);
        let mut first_chunks = first.chunks_exact_mut(n);
        let mut second_chunks = second.chunks_exact_mut(n);
        for ((chunk, out_a), out_b) in (&mut chunks).zip(&mut first_chunks).zip(&mut second_chunks)
        {
            let v = crate::common::F64s::<S>::from_slice(simd, chunk);
            let (a, b) = kernel(simd, v);
            let arr_a = a.to_array();
            let arr_b = b.to_array();
            for (slot, &value) in out_a.iter_mut().zip(arr_a.as_ref().iter()) {
                slot.write(value);
            }
            for (slot, &value) in out_b.iter_mut().zip(arr_b.as_ref().iter()) {
                slot.write(value);
            }
        }
        for ((slot_a, slot_b), &value) in first_chunks
            .into_remainder()
            .iter_mut()
            .zip(second_chunks.into_remainder().iter_mut())
            .zip(chunks.remainder().iter())
        {
            let (a, b) = scalar(value);
            slot_a.write(a);
            slot_b.write(b);
        }
    }

    macro_rules! unary {
        ($($name:ident, $kernel:path, $scalar:expr, $doc:literal;)*) => {
            $(
                #[doc = $doc]
                #[inline]
                pub fn $name(
                    src: &[f64],
                    dst: &mut [MaybeUninit<f64>],
                ) -> Result<(), LengthMismatch> {
                    lengths(src.len(), dst.len())?;
                    #[cfg(feature = "simd")]
                    {
                        let level = fearless_simd::Level::new();
                        fearless_simd::dispatch!(level, simd => run(simd, $kernel, src, dst));
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        scalar_apply(src, dst, $scalar);
                    }
                    Ok(())
                }
            )*
        };
    }

    unary! {
        exp,        crate::f64x::u10::exp,        f64::exp,      "Natural exponential function; 1.0 ULP.";
        exp2,       crate::f64x::u10::exp2,       f64::exp2,     "Base-2 exponential function; 1.0 ULP.";
        exp10,      crate::f64x::u10::exp10,      |v: f64| f64::powf(10.0, v),    "Base-10 exponential function; 1.09 ULP.";
        expm1,      crate::f64x::u10::expm1,      f64::exp_m1,   "Natural exponential function minus one; 1.0 ULP.";
        log,        crate::f64x::u10::log,        f64::ln,       "Natural logarithm; 1.0 ULP.";
        log2,       crate::f64x::u10::log2,       f64::log2,     "Base-2 logarithm; 1.0 ULP.";
        log10,      crate::f64x::u10::log10,      f64::log10,    "Base-10 logarithm; 1.0 ULP.";
        log1p,      crate::f64x::u10::log1p,      f64::ln_1p,    "Logarithm of one plus the argument; 1.0 ULP.";
        sin,        crate::f64x::u10::sin,        f64::sin,      "Sine function; 1.0 ULP.";
        cos,        crate::f64x::u10::cos,        f64::cos,      "Cosine function; 1.0 ULP.";
        tan,        crate::f64x::u10::tan,        f64::tan,      "Tangent function; 1.0 ULP.";
        asin,       crate::f64x::u10::asin,       f64::asin,     "Arc sine function; 1.0 ULP.";
        acos,       crate::f64x::u10::acos,       f64::acos,     "Arc cosine function; 1.0 ULP.";
        atan,       crate::f64x::u10::atan,       f64::atan,     "Arc tangent function; 1.0 ULP.";
        sinh,       crate::f64x::u10::sinh,       f64::sinh,     "Hyperbolic sine function; 1.0 ULP.";
        cosh,       crate::f64x::u10::cosh,       f64::cosh,     "Hyperbolic cosine function; 1.0 ULP.";
        tanh,       crate::f64x::u10::tanh,       f64::tanh,     "Hyperbolic tangent function; 1.0 ULP.";
        asinh,      crate::f64x::u10::asinh,      f64::asinh,    "Inverse hyperbolic sine function; 1.0 ULP.";
        acosh,      crate::f64x::u10::acosh,      f64::acosh,    "Inverse hyperbolic cosine function; 1.0 ULP.";
        atanh,      crate::f64x::u10::atanh,      f64::atanh,    "Inverse hyperbolic tangent function; 1.0 ULP.";
        cbrt,       crate::f64x::u10::cbrt,       f64::cbrt,     "Cube root function; 1.0 ULP.";
    }

    macro_rules! unary_u35 {
        ($($name:ident, $kernel:path, $scalar:expr, $doc:literal;)*) => {
            $(
                #[doc = $doc]
                #[inline]
                pub fn $name(
                    src: &[f64],
                    dst: &mut [MaybeUninit<f64>],
                ) -> Result<(), LengthMismatch> {
                    lengths(src.len(), dst.len())?;
                    #[cfg(feature = "simd")]
                    {
                        let level = fearless_simd::Level::new();
                        fearless_simd::dispatch!(level, simd => run(simd, $kernel, src, dst));
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        scalar_apply(src, dst, $scalar);
                    }
                    Ok(())
                }
            )*
        };
    }

    unary_u35! {
        sin_u35,   crate::f64x::u35::sin,   f64::sin,    "Sine function, 3.5 ULP tier.";
        cos_u35,   crate::f64x::u35::cos,   f64::cos,    "Cosine function, 3.5 ULP tier.";
        tan_u35,   crate::f64x::u35::tan,   f64::tan,    "Tangent function, 3.5 ULP tier.";
        asin_u35,  crate::f64x::u35::asin,  f64::asin,   "Arc sine function, 3.5 ULP tier.";
        acos_u35,  crate::f64x::u35::acos,  f64::acos,   "Arc cosine function, 3.5 ULP tier.";
        atan_u35,  crate::f64x::u35::atan,  f64::atan,   "Arc tangent function, 3.5 ULP tier.";
        sinh_u35,  crate::f64x::u35::sinh,  f64::sinh,   "Hyperbolic sine, 3.5 ULP tier.";
        cosh_u35,  crate::f64x::u35::cosh,  f64::cosh,   "Hyperbolic cosine, 3.5 ULP tier.";
        tanh_u35,  crate::f64x::u35::tanh,  f64::tanh,   "Hyperbolic tangent, 3.5 ULP tier.";
        log_u35,   crate::f64x::u35::log,   f64::ln,     "Natural logarithm, 3.5 ULP tier.";
        log2_u35,  crate::f64x::u35::log2,  f64::log2,   "Base-2 logarithm, 3.5 ULP tier.";
        exp2_u35,  crate::f64x::u35::exp2,  f64::exp2,   "Base-2 exponential, 3.5 ULP tier.";
        exp10_u35, crate::f64x::u35::exp10, |v: f64| f64::powf(10.0, v), "Base-10 exponential, 3.5 ULP tier.";
        cbrt_u35,  crate::f64x::u35::cbrt,  f64::cbrt,   "Cube root, 3.5 ULP tier.";
    }

    /// Square root; `0.5001 ULP`. Equivalent to the hardware square root.
    #[inline]
    pub fn sqrt(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        scalar_apply(src, dst, f64::sqrt);
        Ok(())
    }

    /// Sine and cosine evaluated together; 1.0 ULP.
    #[inline]
    pub fn sincos(
        src: &[f64],
        sin_dst: &mut [MaybeUninit<f64>],
        cos_dst: &mut [MaybeUninit<f64>],
    ) -> Result<(), LengthMismatch> {
        lengths(src.len(), sin_dst.len())?;
        lengths(src.len(), cos_dst.len())?;
        #[cfg(feature = "simd")]
        {
            let level = fearless_simd::Level::new();
            fearless_simd::dispatch!(level, simd => run_ff(simd, crate::f64x::u10::sincos, src, sin_dst, cos_dst, |v: f64| (v.sin(), v.cos())));
        }
        #[cfg(not(feature = "simd"))]
        {
            scalar_apply_ff(src, sin_dst, cos_dst, |v: f64| (v.sin(), v.cos()));
        }
        Ok(())
    }

    /// Sine and cosine of `pi * x` evaluated together; `0.506 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn sincospi(
        src: &[f64],
        sin_dst: &mut [MaybeUninit<f64>],
        cos_dst: &mut [MaybeUninit<f64>],
    ) -> Result<(), LengthMismatch> {
        lengths(src.len(), sin_dst.len())?;
        lengths(src.len(), cos_dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run_ff(simd, crate::f64x::u05::sincospi, src, sin_dst, cos_dst, |v: f64| ((v * core::f64::consts::PI).sin(), (v * core::f64::consts::PI).cos())));
        Ok(())
    }

    macro_rules! binary {
        ($($name:ident, $kernel:path, $scalar:expr, $doc:literal;)*) => {
            $(
                #[doc = $doc]
                #[inline]
                pub fn $name(
                    lhs: &[f64],
                    rhs: &[f64],
                    dst: &mut [MaybeUninit<f64>],
                ) -> Result<(), LengthMismatch> {
                    lengths2(lhs.len(), rhs.len(), dst.len())?;
                    #[cfg(feature = "simd")]
                    {
                        let level = fearless_simd::Level::new();
                        fearless_simd::dispatch!(level, simd => run2(simd, $kernel, lhs, rhs, dst, $scalar));
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        scalar_apply2(lhs, rhs, dst, $scalar);
                    }
                    Ok(())
                }
            )*
        };
    }

    binary! {
        pow,    crate::f64x::u10::pow,    |a: f64, b: f64| f64::powf(a, b),  "Argument raised to an argument power; 1.0 ULP.";
        atan2,  crate::f64x::u10::atan2,  |a: f64, b: f64| f64::atan2(a, b), "Arc tangent of two arguments; 1.0 ULP.";
    }

    /// Euclidean distance in two dimensions; `0.5001 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn hypot(
        lhs: &[f64],
        rhs: &[f64],
        dst: &mut [MaybeUninit<f64>],
    ) -> Result<(), LengthMismatch> {
        lengths2(lhs.len(), rhs.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run2(simd, crate::f64x::u05::hypot, lhs, rhs, dst, f64::hypot));
        Ok(())
    }

    /// Euclidean distance in two dimensions; scalar implementation.
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn hypot(
        lhs: &[f64],
        rhs: &[f64],
        dst: &mut [MaybeUninit<f64>],
    ) -> Result<(), LengthMismatch> {
        lengths2(lhs.len(), rhs.len(), dst.len())?;
        scalar_apply2(lhs, rhs, dst, f64::hypot);
        Ok(())
    }

    /// Error function; 1.0 ULP.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn erf(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, crate::f64x::u10::erf, src, dst));
        Ok(())
    }

    /// Complementary error function; `max(1.5, 2.5) ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn erfc(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, crate::f64x::u15::erfc, src, dst));
        Ok(())
    }

    /// Gamma function; 1.0 ULP.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn tgamma(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, crate::f64x::u10::tgamma, src, dst));
        Ok(())
    }

    /// Logarithm of the absolute gamma function; 1.0 ULP.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn lgamma(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, crate::f64x::u10::lgamma, src, dst));
        Ok(())
    }

    /// Sine of `pi * x`; `0.506 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn sinpi(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, crate::f64x::u05::sinpi, src, dst));
        Ok(())
    }

    /// Cosine of `pi * x`; `0.506 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn cospi(src: &[f64], dst: &mut [MaybeUninit<f64>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, crate::f64x::u05::cospi, src, dst));
        Ok(())
    }
}

pub mod f32 {
    //! `f32` slice entry points (port in progress).

    use super::LengthMismatch;
    use core::mem::MaybeUninit;

    /// Reserved for the `f32` kernels once they land.
    #[inline]
    pub fn _placeholder(_src: &[f32], _dst: &mut [MaybeUninit<f32>]) -> Result<(), LengthMismatch> {
        Ok(())
    }
}

impl ElementwiseMath for f64 {
    fn exp(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::exp(src, dst)
    }
    fn exp2(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::exp2(src, dst)
    }
    fn exp10(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::exp10(src, dst)
    }
    fn expm1(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::expm1(src, dst)
    }
    fn log(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::log(src, dst)
    }
    fn log2(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::log2(src, dst)
    }
    fn log10(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::log10(src, dst)
    }
    fn log1p(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::log1p(src, dst)
    }
    fn sin(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::sin(src, dst)
    }
    fn cos(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::cos(src, dst)
    }
    fn tan(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::tan(src, dst)
    }
    fn asin(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::asin(src, dst)
    }
    fn acos(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::acos(src, dst)
    }
    fn atan(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::atan(src, dst)
    }
    fn sinh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::sinh(src, dst)
    }
    fn cosh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::cosh(src, dst)
    }
    fn tanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::tanh(src, dst)
    }
    fn asinh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::asinh(src, dst)
    }
    fn acosh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::acosh(src, dst)
    }
    fn atanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::atanh(src, dst)
    }
    fn cbrt(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::cbrt(src, dst)
    }
    fn sqrt(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::sqrt(src, dst)
    }
    fn pow(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::f64::pow(lhs, rhs, dst)
    }
    fn atan2(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::f64::atan2(lhs, rhs, dst)
    }
    fn hypot(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::f64::hypot(lhs, rhs, dst)
    }
    fn sincos(
        src: &[Self],
        sin_dst: &mut [MaybeUninit<Self>],
        cos_dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::f64::sincos(src, sin_dst, cos_dst)
    }
}
