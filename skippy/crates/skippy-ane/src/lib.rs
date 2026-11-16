//! Apple Neural Engine conversion and draft execution for paired DFlash models.

#[cfg(feature = "conversion")]
pub mod conversion;

#[cfg(target_os = "macos")]
pub mod draft;
