//! Conversion and stateful Core ML execution for paired DFlash drafts.

#[cfg(feature = "conversion")]
pub mod conversion;
pub mod profile;

#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
pub use runtime::AneDraft;
