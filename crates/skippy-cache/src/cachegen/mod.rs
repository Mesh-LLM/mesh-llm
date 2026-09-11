//! CacheGen experiments and the pinned LMCache-compatible CPU reference
//! used by the acceptance gate in #1652.
//!
//! [`lmcache`] owns the active reference algorithm. [`container`], [`reference`],
//! and [`rans`] retain the earlier simplified prototype and its historical
//! fixtures; production and acceptance-gate code must not select that path.

pub mod container;
pub mod lmcache;
pub mod rans;
pub mod reference;
