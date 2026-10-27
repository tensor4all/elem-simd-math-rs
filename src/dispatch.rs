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
/// Implemented for `f64`. The accuracy tier and error bound of each function are
/// documented on the kernels in [`crate::f64x`]; each method selects the most
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

macro_rules! binary {
        ($($name:ident, $kernel:path, $scalar:expr, $doc:literal;)*) => {
            $(
                #[doc = $doc]
                #[inline]
                pub fn $name(
                    lhs: &[Scalar],
                    rhs: &[Scalar],
                    dst: &mut [MaybeUninit<Scalar>],
                ) -> Result<(), LengthMismatch> {
                    lengths2(lhs.len(), rhs.len(), dst.len())?;
                    #[cfg(feature = "simd")]
                    {
                        let level = fearless_simd::Level::new();
                        fearless_simd::dispatch!(level, simd => run2(simd, $kernel, lhs, rhs, dst));
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

macro_rules! unary {
        ($($name:ident, $kernel:path, $scalar:expr, $doc:literal;)*) => {
            $(
                #[doc = $doc]
                #[inline]
                pub fn $name(
                    src: &[Scalar],
                    dst: &mut [MaybeUninit<Scalar>],
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

macro_rules! pair {
        ($($name:ident, $kernel:path, $scalar:expr, $doc:literal;)*) => {
            $(
                #[doc = $doc]
                #[inline]
                pub fn $name(
                    src: &[Scalar],
                    first: &mut [MaybeUninit<Scalar>],
                    second: &mut [MaybeUninit<Scalar>],
                ) -> Result<(), LengthMismatch> {
                    lengths(src.len(), first.len())?;
                    lengths(src.len(), second.len())?;
                    #[cfg(feature = "simd")]
                    {
                        let level = fearless_simd::Level::new();
                        fearless_simd::dispatch!(level, simd => run_ff(simd, $kernel, src, first, second));
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        scalar_apply_ff(src, first, second, $scalar);
                    }
                    Ok(())
                }
            )*
        };
    }

/// Generates the flat-slice entry points for one scalar type.
///
/// `$modname` is the module to create, `$ty` the scalar, `$vec` the matching
/// [`fearless_simd`] vector type and `$kernels` the module holding the vector kernels. Both
/// instantiations must expose the same function names, so the tier modules drop the `f` suffix
/// that upstream `sleef-rs` uses for `f32`.
macro_rules! slice_api {
    ($modname:ident, $ty:ty, $vec:ident, $kernels:ident) => {
pub mod $modname {
    //! Slice entry points for a scalar type.

    use super::LengthMismatch;
    #[cfg(not(feature = "simd"))]
    use super::{scalar_apply, scalar_apply2, scalar_apply_ff};
    #[cfg(feature = "simd")]
    use crate::common::$vec;
    use core::mem::MaybeUninit;

    /// The scalar this instantiation works on; the tier macros below name it by alias so that a
    /// single definition can serve both dtypes.
    type Scalar = $ty;

    /// The module holding the vector kernels for `Scalar`.
    #[cfg(feature = "simd")]
    use crate::$kernels as xkernels;

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

    /// A unary kernel over native-width vectors.
    #[cfg(feature = "simd")]
    type UnaryKernel<S> = fn(S, crate::common::$vec<S>) -> crate::common::$vec<S>;

    /// Applies a unary `kernel` lane-wise over `src`, writing `dst`.
    ///
    /// The elements past the last whole vector are handled by running the same kernel over a
    /// padded copy, so every element goes through the kernel and the documented error bound
    /// applies to the whole result — including the short inputs where the platform's scalar
    /// functions are measurably worse than the kernel.
    // `#[inline(always)]` is required by `fearless_simd`: the code called from a `dispatch!`
    // closure is only compiled with the selected level's target features if it is inlined into
    // that closure.
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn run<S: Simd>(simd: S, kernel: UnaryKernel<S>, src: &[$ty], dst: &mut [MaybeUninit<$ty>]) {
        let n = crate::common::$vec::<S>::LEN;
        let mut chunks = src.chunks_exact(n);
        let mut out_chunks = dst.chunks_exact_mut(n);
        for (chunk, out) in (&mut chunks).zip(&mut out_chunks) {
            let v = crate::common::$vec::<S>::from_slice(simd, chunk);
            let r = kernel(simd, v);
            let arr = r.to_array();
            for (slot, &value) in out.iter_mut().zip(arr.as_ref().iter()) {
                slot.write(value);
            }
        }
        let tail = chunks.remainder();
        let out_tail = out_chunks.into_remainder();
        if tail.is_empty() {
            return;
        }
        let mut padded = $vec::<S>::splat(simd, 0.0).to_array();
        for (slot, &value) in padded.as_mut().iter_mut().zip(tail.iter()) {
            *slot = value;
        }
        let v = $vec::<S>::load_array(simd, padded);
        let arr = kernel(simd, v).to_array();
        for (slot, &value) in out_tail.iter_mut().zip(arr.as_ref().iter()) {
            slot.write(value);
        }
    }

    /// Applies a binary `kernel` lane-wise over `lhs` and `rhs`, writing `dst`.
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn run2<S: Simd>(
        simd: S,
        kernel: fn(S, crate::common::$vec<S>, crate::common::$vec<S>) -> crate::common::$vec<S>,
        lhs: &[$ty],
        rhs: &[$ty],
        dst: &mut [MaybeUninit<$ty>],
    ) {
        let n = crate::common::$vec::<S>::LEN;
        let mut lhs_chunks = lhs.chunks_exact(n);
        let mut rhs_chunks = rhs.chunks_exact(n);
        let mut out_chunks = dst.chunks_exact_mut(n);
        for ((a, b), out) in (&mut lhs_chunks).zip(&mut rhs_chunks).zip(&mut out_chunks) {
            let va = crate::common::$vec::<S>::from_slice(simd, a);
            let vb = crate::common::$vec::<S>::from_slice(simd, b);
            let r = kernel(simd, va, vb);
            let arr = r.to_array();
            for (slot, &value) in out.iter_mut().zip(arr.as_ref().iter()) {
                slot.write(value);
            }
        }
        let (ta, tb) = (lhs_chunks.remainder(), rhs_chunks.remainder());
        let out_tail = out_chunks.into_remainder();
        if ta.is_empty() {
            return;
        }
        let mut pa = $vec::<S>::splat(simd, 0.0).to_array();
        let mut pb = $vec::<S>::splat(simd, 0.0).to_array();
        for (slot, &value) in pa.as_mut().iter_mut().zip(ta.iter()) {
            *slot = value;
        }
        for (slot, &value) in pb.as_mut().iter_mut().zip(tb.iter()) {
            *slot = value;
        }
        let arr = kernel(
            simd,
            $vec::<S>::load_array(simd, pa),
            $vec::<S>::load_array(simd, pb),
        )
        .to_array();
        for (slot, &value) in out_tail.iter_mut().zip(arr.as_ref().iter()) {
            slot.write(value);
        }
    }

    /// A kernel that produces two vectors from one.
    #[cfg(feature = "simd")]
    type FfKernel<S> =
        fn(S, crate::common::$vec<S>) -> (crate::common::$vec<S>, crate::common::$vec<S>);

    /// Applies a `kernel` returning two vectors, writing `sin` and `cos`.
    ///
    /// The tail is handled the same way as in [`run`]: by `scalar` when one is given, otherwise by
    /// running the same vector kernel over a padded copy.
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn run_ff<S: Simd>(
        simd: S,
        kernel: FfKernel<S>,
        src: &[$ty],
        first: &mut [MaybeUninit<$ty>],
        second: &mut [MaybeUninit<$ty>],
    ) {
        let n = crate::common::$vec::<S>::LEN;
        let mut chunks = src.chunks_exact(n);
        let mut first_chunks = first.chunks_exact_mut(n);
        let mut second_chunks = second.chunks_exact_mut(n);
        for ((chunk, out_a), out_b) in (&mut chunks).zip(&mut first_chunks).zip(&mut second_chunks)
        {
            let v = crate::common::$vec::<S>::from_slice(simd, chunk);
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
        let tail = chunks.remainder();
        let first_tail = first_chunks.into_remainder();
        let second_tail = second_chunks.into_remainder();
        if tail.is_empty() {
            return;
        }
        let mut padded = $vec::<S>::splat(simd, 0.0).to_array();
        for (slot, &value) in padded.as_mut().iter_mut().zip(tail.iter()) {
            *slot = value;
        }
        let (a, b) = kernel(simd, $vec::<S>::load_array(simd, padded));
        let (arr_a, arr_b) = (a.to_array(), b.to_array());
        for ((slot_a, slot_b), (&va, &vb)) in first_tail
            .iter_mut()
            .zip(second_tail.iter_mut())
            .zip(arr_a.as_ref().iter().zip(arr_b.as_ref().iter()))
        {
            slot_a.write(va);
            slot_b.write(vb);
        }
    }



    unary! {
        exp,        xkernels::u10::exp,        <$ty>::exp,      "Natural exponential function; 1.0 ULP.";
        exp2,       xkernels::u10::exp2,       <$ty>::exp2,     "Base-2 exponential function; 1.0 ULP.";
        exp10,      xkernels::u10::exp10,      |v: $ty| <$ty>::powf(10.0, v),    "Base-10 exponential function; 1.09 ULP.";
        expm1,      xkernels::u10::expm1,      <$ty>::exp_m1,   "Natural exponential function minus one; 1.0 ULP.";
        log,        xkernels::u10::log,        <$ty>::ln,       "Natural logarithm; 1.0 ULP.";
        log2,       xkernels::u10::log2,       <$ty>::log2,     "Base-2 logarithm; 1.0 ULP.";
        log10,      xkernels::u10::log10,      <$ty>::log10,    "Base-10 logarithm; 1.0 ULP.";
        log1p,      xkernels::u10::log1p,      <$ty>::ln_1p,    "Logarithm of one plus the argument; 1.0 ULP.";
        sin,        xkernels::u10::sin,        <$ty>::sin,      "Sine function; 1.0 ULP.";
        cos,        xkernels::u10::cos,        <$ty>::cos,      "Cosine function; 1.0 ULP.";
        tan,        xkernels::u10::tan,        <$ty>::tan,      "Tangent function; 1.0 ULP.";
        asin,       xkernels::u10::asin,       <$ty>::asin,     "Arc sine function; 1.0 ULP.";
        acos,       xkernels::u10::acos,       <$ty>::acos,     "Arc cosine function; 1.0 ULP.";
        atan,       xkernels::u10::atan,       <$ty>::atan,     "Arc tangent function; 1.0 ULP.";
        sinh,       xkernels::u10::sinh,       <$ty>::sinh,     "Hyperbolic sine function; 1.0 ULP.";
        cosh,       xkernels::u10::cosh,       <$ty>::cosh,     "Hyperbolic cosine function; 1.0 ULP.";
        tanh,       xkernels::u10::tanh,       <$ty>::tanh,     "Hyperbolic tangent function; 1.0 ULP.";
        asinh,      xkernels::u10::asinh,      <$ty>::asinh,    "Inverse hyperbolic sine function; 1.0 ULP.";
        acosh,      xkernels::u10::acosh,      <$ty>::acosh,    "Inverse hyperbolic cosine function; 1.0 ULP.";
        atanh,      xkernels::u10::atanh,      <$ty>::atanh,    "Inverse hyperbolic tangent function; 1.0 ULP.";
        cbrt,       xkernels::u10::cbrt,       <$ty>::cbrt,     "Cube root function; 1.0 ULP.";
        sin_deterministic, xkernels::u10::sin_deterministic, <$ty>::sin,
            "Sine, bitwise identical across lanes and platforms; 1.0 ULP.";
        cos_deterministic, xkernels::u10::cos_deterministic, <$ty>::cos,
            "Cosine, bitwise identical across lanes and platforms; 1.0 ULP.";
        tan_deterministic, xkernels::u10::tan_deterministic, <$ty>::tan,
            "Tangent, bitwise identical across lanes and platforms; 1.0 ULP.";
        sqrt,       crate::common::hw_sqrt,    <$ty>::sqrt,    "Square root; 0.5 ULP.";
    }



    unary! {
        sin_u35,   xkernels::u35::sin,   <$ty>::sin,    "Sine function, 3.5 ULP tier.";
        cos_u35,   xkernels::u35::cos,   <$ty>::cos,    "Cosine function, 3.5 ULP tier.";
        tan_u35,   xkernels::u35::tan,   <$ty>::tan,    "Tangent function, 3.5 ULP tier.";
        asin_u35,  xkernels::u35::asin,  <$ty>::asin,   "Arc sine function, 3.5 ULP tier.";
        acos_u35,  xkernels::u35::acos,  <$ty>::acos,   "Arc cosine function, 3.5 ULP tier.";
        atan_u35,  xkernels::u35::atan,  <$ty>::atan,   "Arc tangent function, 3.5 ULP tier.";
        sinh_u35,  xkernels::u35::sinh,  <$ty>::sinh,   "Hyperbolic sine, 3.5 ULP tier.";
        cosh_u35,  xkernels::u35::cosh,  <$ty>::cosh,   "Hyperbolic cosine, 3.5 ULP tier.";
        tanh_u35,  xkernels::u35::tanh,  <$ty>::tanh,   "Hyperbolic tangent, 3.5 ULP tier.";
        log_u35,   xkernels::u35::log,   <$ty>::ln,     "Natural logarithm, 3.5 ULP tier.";
        log2_u35,  xkernels::u35::log2,  <$ty>::log2,   "Base-2 logarithm, 3.5 ULP tier.";
        exp2_u35,  xkernels::u35::exp2,  <$ty>::exp2,   "Base-2 exponential, 3.5 ULP tier.";
        exp10_u35, xkernels::u35::exp10, |v: $ty| <$ty>::powf(10.0, v), "Base-10 exponential, 3.5 ULP tier.";
        cbrt_u35,  xkernels::u35::cbrt,  <$ty>::cbrt,   "Cube root, 3.5 ULP tier.";
        sin_deterministic_u35, xkernels::u35::sin_deterministic, <$ty>::sin,
            "Sine, lane-order independent, 3.5 ULP tier.";
        cos_deterministic_u35, xkernels::u35::cos_deterministic, <$ty>::cos,
            "Cosine, lane-order independent, 3.5 ULP tier.";
        tan_deterministic_u35, xkernels::u35::tan_deterministic, <$ty>::tan,
            "Tangent, lane-order independent, 3.5 ULP tier.";
    }

    pair! {
        sincos, xkernels::u10::sincos, |v: $ty| (v.sin(), v.cos()),
            "Sine and cosine evaluated together; 1.0 ULP.";
        sincos_deterministic, xkernels::u10::sincos_deterministic, |v: $ty| (v.sin(), v.cos()),
            "Sine and cosine together, lane-order independent; 1.0 ULP.";
        sincos_u35, xkernels::u35::sincos, |v: $ty| (v.sin(), v.cos()),
            "Sine and cosine evaluated together, 3.5 ULP tier.";
        sincos_deterministic_u35, xkernels::u35::sincos_deterministic, |v: $ty| (v.sin(), v.cos()),
            "Sine and cosine together, lane-order independent, 3.5 ULP tier.";
    }

    /// Sine and cosine of `pi * x` together; `0.506 ULP`.
    ///
    /// There is no scalar implementation in the standard library that is accurate enough for
    /// large arguments, so the tail uses the vector kernel over a padded copy.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn sincospi(
        src: &[$ty],
        first: &mut [MaybeUninit<$ty>],
        second: &mut [MaybeUninit<$ty>],
    ) -> Result<(), LengthMismatch> {
        lengths(src.len(), first.len())?;
        lengths(src.len(), second.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run_ff(simd, xkernels::u05::sincospi, src, first, second));
        Ok(())
    }

    /// Sine and cosine of `pi * x` together, 3.5 ULP tier.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn sincospi_u35(
        src: &[$ty],
        first: &mut [MaybeUninit<$ty>],
        second: &mut [MaybeUninit<$ty>],
    ) -> Result<(), LengthMismatch> {
        lengths(src.len(), first.len())?;
        lengths(src.len(), second.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run_ff(simd, xkernels::u35::sincospi, src, first, second));
        Ok(())
    }



    binary! {
        pow,    xkernels::u10::pow,    |a: $ty, b: $ty| <$ty>::powf(a, b),  "Argument raised to an argument power; 1.0 ULP.";
        atan2,  xkernels::u10::atan2,  |a: $ty, b: $ty| <$ty>::atan2(a, b), "Arc tangent of two arguments; 1.0 ULP.";
    }

    /// Euclidean distance in two dimensions; `0.5001 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn hypot(
        lhs: &[$ty],
        rhs: &[$ty],
        dst: &mut [MaybeUninit<$ty>],
    ) -> Result<(), LengthMismatch> {
        lengths2(lhs.len(), rhs.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run2(simd, xkernels::u05::hypot, lhs, rhs, dst));
        Ok(())
    }

    /// Euclidean distance in two dimensions; scalar implementation.
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn hypot(
        lhs: &[$ty],
        rhs: &[$ty],
        dst: &mut [MaybeUninit<$ty>],
    ) -> Result<(), LengthMismatch> {
        lengths2(lhs.len(), rhs.len(), dst.len())?;
        scalar_apply2(lhs, rhs, dst, <$ty>::hypot);
        Ok(())
    }

    unary! {
        sigmoid, xkernels::activation::sigmoid,
            |v: $ty| { let e = (-v.abs()).exp(); if v > 0.0 { 1.0 / (1.0 + e) } else { e / (1.0 + e) } },
            "Logistic sigmoid `1 / (1 + exp(-x))` in one pass, overflow-free; 2.5 ULP.";
        silu, xkernels::activation::silu,
            |v: $ty| { let e = (-v.abs()).exp(); v * if v > 0.0 { 1.0 / (1.0 + e) } else { e / (1.0 + e) } },
            "SiLU / swish `x * sigmoid(x)` in one pass; 3.5 ULP wherever `exp(-|x|)` is normal (see the kernel for the subnormal tail).";
        softplus, xkernels::activation::softplus, |v: $ty| v.max(0.0) + (-v.abs()).exp().ln_1p(),
            "Softplus `log(1 + exp(x))` in one pass, overflow-free; 3.0 ULP.";
        gelu_tanh, xkernels::activation::gelu_tanh,
            |v: $ty| 0.5 * v * (1.0 + ((0.797_884_560_802_865_4 as $ty) * (v + (0.044_715 as $ty) * v * v * v)).tanh()),
            "GELU tanh approximation `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))` in one pass; 3.0 ULP for `x >= 0`, see the kernel for `x < 0`.";
    }

    /// Exact-form GELU `x/2 * (1 + erf(x / sqrt(2)))` in one pass. See the kernel
    /// ([`crate::f32x::activation::gelu`] / [`crate::f64x::activation::gelu`]) for
    /// the error bound: the `f32` kernel evaluates `erf` as PyTorch's CPU GELU does
    /// (absolute error about `5e-7` on `erf` after rounding, so `3.0e-7 * |x|` plus
    /// 3 ULP on the result), the `f64` kernel uses the 1.0 ULP `erf`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn gelu(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::activation::gelu, src, dst));
        Ok(())
    }

    /// Error function; 1.0 ULP.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn erf(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::u10::erf, src, dst));
        Ok(())
    }

    /// Complementary error function; `max(1.5, 2.5) ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn erfc(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::u15::erfc, src, dst));
        Ok(())
    }

    /// Gamma function; 1.0 ULP.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn tgamma(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::u10::tgamma, src, dst));
        Ok(())
    }

    /// Logarithm of the absolute gamma function; 1.0 ULP.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn lgamma(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::u10::lgamma, src, dst));
        Ok(())
    }

    /// Sine of `pi * x`; `0.506 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn sinpi(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::u05::sinpi, src, dst));
        Ok(())
    }

    /// Cosine of `pi * x`; `0.506 ULP`.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn cospi(src: &[$ty], dst: &mut [MaybeUninit<$ty>]) -> Result<(), LengthMismatch> {
        lengths(src.len(), dst.len())?;
        let level = fearless_simd::Level::new();
        fearless_simd::dispatch!(level, simd => run(simd, xkernels::u05::cospi, src, dst));
        Ok(())
    }
}

impl ElementwiseMath for $ty {
    fn exp(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::exp(src, dst)
    }
    fn exp2(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::exp2(src, dst)
    }
    fn exp10(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::exp10(src, dst)
    }
    fn expm1(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::expm1(src, dst)
    }
    fn log(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::log(src, dst)
    }
    fn log2(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::log2(src, dst)
    }
    fn log10(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::log10(src, dst)
    }
    fn log1p(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::log1p(src, dst)
    }
    fn sin(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::sin(src, dst)
    }
    fn cos(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::cos(src, dst)
    }
    fn tan(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::tan(src, dst)
    }
    fn asin(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::asin(src, dst)
    }
    fn acos(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::acos(src, dst)
    }
    fn atan(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::atan(src, dst)
    }
    fn sinh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::sinh(src, dst)
    }
    fn cosh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::cosh(src, dst)
    }
    fn tanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::tanh(src, dst)
    }
    fn asinh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::asinh(src, dst)
    }
    fn acosh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::acosh(src, dst)
    }
    fn atanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::atanh(src, dst)
    }
    fn cbrt(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::cbrt(src, dst)
    }
    fn sqrt(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::$modname::sqrt(src, dst)
    }
    fn pow(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::$modname::pow(lhs, rhs, dst)
    }
    fn atan2(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::$modname::atan2(lhs, rhs, dst)
    }
    fn hypot(
        lhs: &[Self],
        rhs: &[Self],
        dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::$modname::hypot(lhs, rhs, dst)
    }
    fn sincos(
        src: &[Self],
        sin_dst: &mut [MaybeUninit<Self>],
        cos_dst: &mut [MaybeUninit<Self>],
    ) -> Result<(), LengthMismatch> {
        self::$modname::sincos(src, sin_dst, cos_dst)
    }
}
    };
}

slice_api!(f64, f64, F64s, f64x);
slice_api!(f32, f32, F32s, f32x);
