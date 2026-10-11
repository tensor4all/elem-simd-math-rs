//! Scalar constants shared by the ported kernels.
//!
//! Values and names follow `sleef-rs` rev `3671d46` (`src/f64.rs`, `src/f32.rs`), which in turn
//! carries SLEEF's `misc.h` / `sleefdp.c` constants. Double-double constants are kept as
//! `(hi, lo)` tuples; see [`crate::doubled`].

// ---------------------------------------------------------------- f64 ----------

/// `2^63`.
pub const D1_63: f64 = (1u64 << 63) as f64;
/// `2^60`.
pub const D1_60: f64 = (1u64 << 60) as f64;
/// `2^54`.
pub const D1_54: f64 = (1u64 << 54) as f64;
/// `2^53`.
pub const D1_53: f64 = (1u64 << 53) as f64;
/// `2^52`.
pub const D1_52: f64 = (1u64 << 52) as f64;
/// `2^32`.
pub const D1_32: f64 = (1u64 << 32) as f64;
/// `2^31`.
pub const D1_31: f64 = (1u64 << 31) as f64;
/// `2^28`.
pub const D1_28: f64 = (1u64 << 28) as f64;
/// `2^24`.
pub const D1_24: f64 = (1u64 << 24) as f64;
/// `2^23`.
pub const D1_23: f64 = (1u64 << 23) as f64;

/// `ilogb(0)` as returned by SLEEF.
pub const SLEEF_FP_ILOGB0: i32 = -2_147_483_648;
/// `ilogb(NaN)` as returned by SLEEF.
pub const SLEEF_FP_ILOGBNAN: i32 = 2_147_483_647;
/// `sqrt(f64::MAX)`.
pub const SQRT_DBL_MAX: f64 = 1.340_780_792_994_259_635_5e+154;
/// Argument bound for the 1e9-range `*pi` functions.
pub const TRIGRANGEMAX3: f64 = 1e+9;
/// High part of `ln 2`.
pub const L2_U: f64 = 0.693_147_180_559_662_956_511_601_805_686_950_683_593_75;
/// Low part of `ln 2`.
pub const L2_L: f64 = 0.282_352_905_630_315_771_225_884_481_750_134_360_255_254_120_68_e-12;
/// `1 / ln 2`.
pub const R_LN2: f64 =
    1.442_695_040_888_963_407_359_924_681_001_892_137_426_645_954_152_985_934_135_449_406_931;
/// High part of `log10 2`.
pub const L10_U: f64 = 0.301_029_995_663_839_144_98;
/// Low part of `log10 2`.
pub const L10_L: f64 = 1.420_502_322_726_609_941_8_e-13;
/// `log2 10`.
pub const LOG10_2: f64 = 3.321_928_094_887_362_347_870_319_429_489_390_175_864_831_393;

/// `pi` reduced into two parts with a trailing zero run of 28 bits, together with the matching
/// low parts; `PI_A + PI_B + PI_C + PI_D` approximates `pi`.
pub const PI_A: f64 = 3.141_592_621_803_283_691_4;
/// See [`PI_A`].
pub const PI_B: f64 = 3.178_650_942_459_171_346_9_e-8;
/// See [`PI_A`].
pub const PI_C: f64 = 1.224_646_786_410_718_850_2_e-16;
/// See [`PI_A`].
pub const PI_D: f64 = 1.273_663_432_702_189_981_6_e-24;
/// Argument bound for the four-term reduction.
pub const TRIGRANGEMAX: f64 = 1e+14;
/// Two-term `pi` split used for arguments up to [`TRIGRANGEMAX2`].
pub const PI_A2: f64 = 3.141_592_653_589_793_116;
/// See [`PI_A2`].
pub const PI_B2: f64 = 1.224_646_799_147_353_207_2_e-16;
/// Argument bound for the two-term reduction.
pub const TRIGRANGEMAX2: f64 = 15.;

/// `2 / pi` as a double-double.
pub const M_2_PI: (f64, f64) = (
    0.636_619_772_367_581_382_43,
    -3.935_735_335_036_497_176_4_e-17,
);

/// `pi` as a double-double.
pub const D_PI: (f64, f64) = (3.141_592_653_589_793_116, 1.224_646_799_147_353_207_2_e-16);
/// `ln 2` as a double-double.
pub const D_LN2: (f64, f64) = (
    0.693_147_180_559_945_286_226_764,
    2.319_046_813_846_299_558_417_771_e-17,
);
