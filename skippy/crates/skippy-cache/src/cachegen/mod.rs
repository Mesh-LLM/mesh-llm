//! CacheGen archive codecs and scalar references.
//!
//! [`anchor_delta`] encodes F32 KV with ten-token anchors and high-precision
//! deltas. [`lmcache`] remains the F16 and block-quantized path. [`container`],
//! [`reference`], and [`rans`] retain the earlier simplified prototype and its
//! historical fixtures; production must not select that path.

pub mod anchor_delta;
pub mod archive;
pub mod container;
pub mod lmcache;
pub mod rans;
pub mod reference;
