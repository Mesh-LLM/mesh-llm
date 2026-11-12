//! `models resolve` and `models restore-inputs`: hand consumers exactly one
//! cadence-authorized, integrity-pinned test-model artifact. xtask's other
//! `models` commands reuse the manifest, field, JSON byte, and serving-entry
//! owners here.

pub mod argv;
pub mod fields;
pub mod json_bytes;
pub mod manifest;
pub mod resolve;
pub mod restore_inputs;
pub mod serving_entry;
