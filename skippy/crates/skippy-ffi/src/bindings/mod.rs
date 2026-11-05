//! Bindgen-generated FFI declarations for the native surfaces Skippy consumes.
//!
//! These files are **generated** from the pinned llama.cpp headers by
//! `bindgen` (see `bindgen/generator.rs` and `build.rs`):
//!
//! - [`skippy`]: Skippy's own public native-runtime C ABI
//!   (`include/skippy.h` from the patched llama.cpp checkout).
//! - [`llama`]: the upstream llama.cpp surface consumed directly
//!   (`llama.h`, narrow allowlist).
//! - [`mtmd`]: the llama.cpp multimodal surface (`mtmd.h` / `mtmd-helper.h`).
//!
//! Do not edit these files by hand. Regenerate with
//! `SKIPPY_FFI_BINDGEN=regen cargo check -p skippy-ffi` after
//! `scripts/prepare-llama.sh pinned`; CI fails when they drift from the
//! pinned headers (`SKIPPY_FFI_BINDGEN=check`).
//!
//! The generated declarations are the single source of truth for the native
//! ABI layout. Hand-written code in this crate keeps only safe wrappers,
//! opaque-handle ownership, dynamic-library symbol loading, and ergonomic
//! re-exports (see `lib.rs` and the sibling modules).

pub mod llama;
pub mod mtmd;
pub mod skippy;
