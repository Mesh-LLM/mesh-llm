//! Conversion and stateful Core ML execution for the paired Qwen3.6 DFlash drafter.

#[cfg(feature = "conversion")]
pub mod conversion;

#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
pub use runtime::AneDraft;
