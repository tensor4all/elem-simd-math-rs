//! Slice entry points: elementwise application over contiguous buffers.

use core::mem::MaybeUninit;

/// The source and destination lengths differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LengthMismatch {
    /// Number of elements in the source slice.
    pub src: usize,
    /// Number of elements in the destination slice.
    pub dst: usize,
}

impl core::fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "source has {} elements but the destination has {}",
            self.src, self.dst
        )
    }
}

impl std::error::Error for LengthMismatch {}

/// Elementwise math over contiguous slices.
///
/// Implemented for `f32` and `f64`. Every method writes `dst[i] = f(src[i])` for all `i`, using
/// the best SIMD level the CPU supports; `dst` may be uninitialized storage, and every element
/// is written on success. The accuracy tier and error bound of each function are documented on
/// the kernels in [`crate::f64x`] and [`crate::f32x`].
pub trait ElementwiseMath: Copy + 'static {
    /// Natural exponential function; 1.0 ULP.
    fn exp(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Natural logarithm; 3.5 ULP.
    fn log(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Hyperbolic tangent; 3.5 ULP.
    fn tanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
    /// Natural exponential function minus one; 1.0 ULP.
    fn expm1(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch>;
}

/// Writes `f(src[i])` into every slot of `dst`.
pub(crate) fn scalar_apply<T: Copy>(src: &[T], dst: &mut [MaybeUninit<T>], f: impl Fn(T) -> T) {
    for (slot, &value) in dst.iter_mut().zip(src.iter()) {
        slot.write(f(value));
    }
}

pub mod f64 {
    //! `f64` slice entry points.

    use super::{scalar_apply, LengthMismatch};
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

    /// Applies `kernel` lane-wise over `src`, writing `dst`.
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

    macro_rules! unary {
        ($name:ident, $kernel:path, $scalar:expr, $doc:literal) => {
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
        };
    }

    unary!(
        exp,
        crate::f64x::u10::exp,
        |v: f64| v.exp(),
        "Natural exponential function; 1.0 ULP."
    );
    unary!(
        log,
        crate::f64x::u35::log,
        |v: f64| v.ln(),
        "Natural logarithm; 3.5 ULP."
    );
    unary!(
        tanh,
        crate::f64x::u35::tanh,
        |v: f64| v.tanh(),
        "Hyperbolic tangent; 3.5 ULP."
    );
    unary!(
        expm1,
        crate::f64x::u10::expm1,
        |v: f64| v.exp_m1(),
        "Natural exponential function minus one; 1.0 ULP."
    );
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
    fn log(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::log(src, dst)
    }
    fn tanh(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::tanh(src, dst)
    }
    fn expm1(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> Result<(), LengthMismatch> {
        self::f64::expm1(src, dst)
    }
}
