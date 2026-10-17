//! The capsule producer this plugin seals with. The neutral pieces (JCS,
//! keys, COSE, the ledger, checkpoints, padding, sequencing, timestamps,
//! the anchor client) come from the `capsule-emit`
//! crate. The mesh-specific record kinds and the split-stage and runtime
//! attestation modules live here, written against capsule-emit's public
//! extension points; nothing mesh-specific enters capsule-emit.

pub use capsule_emit_lib::{
    anchor, checkpoint, cose, jcs, keys, ledger, padding, sequence, timestamp,
};

/// Offline verification: the plugin's tests check sealed records with it.
#[cfg(test)]
pub use capsule_emit_lib::verify;

pub mod capsule;
pub mod index;
pub mod runtime_attest;
pub mod stage;
pub mod stage_verify;
pub mod tee_attest;
