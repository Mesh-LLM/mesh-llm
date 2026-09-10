//! Pure-Rust CPU CacheGen reference (#1652): the deterministic golden
//! encoder/decoder later GPU implementations must match bit-for-bit.
//!
//! Layout follows the crate's semantic-ownership rules:
//! [`reference`] owns quantization/calibration/delta math,
//! [`rans`] owns the entropy coder, and [`cachegen`] owns the segment
//! container and the identity surface the L3 manifest consumes.

pub mod container;
pub mod rans;
pub mod reference;
