//! Base properties attached to every event.
//!
//! These describe the build and the coarse shape of the machine. They are
//! chosen to answer "which platforms and versions are in use" without
//! accumulating enough detail to fingerprint one machine.

use crate::event::Properties;
use mesh_llm_build_info::{BUILD_VERSION, is_sha_build};

/// Library name reported to PostHog, so events from the node are separable
/// from anything the website or console might send later.
pub const LIB_NAME: &str = "mesh-llm-rust";

/// How this binary was built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuildChannel {
    /// A tagged release, e.g. `0.76.0`.
    Release,
    /// A release candidate or other pre-release, e.g. `0.76.0-rc8`.
    Prerelease,
    /// A build stamped with a commit sha.
    Development,
}

impl BuildChannel {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Prerelease => "prerelease",
            Self::Development => "development",
        }
    }

    /// Classify a build version string.
    #[must_use]
    pub fn classify(version: &str) -> Self {
        if is_sha_build(version) {
            Self::Development
        } else if version.contains('-') {
            Self::Prerelease
        } else {
            Self::Release
        }
    }
}

/// The operating system family, as a fixed string.
#[must_use]
pub const fn os_family() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "other"
    }
}

/// The CPU architecture, as a fixed string.
#[must_use]
pub const fn architecture() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        "other"
    }
}

/// Build the base property set for this process.
///
/// The version string is passed through [`crate::Label::sanitize_or_redact`]
/// because `MESH_LLM_BUILD_VERSION` is set by the build environment rather
/// than by this crate.
#[must_use]
pub fn base_properties() -> Properties {
    Properties::new()
        .with(
            "mesh_llm_version",
            crate::Label::sanitize_or_redact(BUILD_VERSION),
        )
        .with(
            "build_channel",
            BuildChannel::classify(BUILD_VERSION).as_str(),
        )
        .with("os", os_family())
        .with("arch", architecture())
        .with("$lib", LIB_NAME)
        .with(
            "$lib_version",
            crate::Label::sanitize_or_redact(BUILD_VERSION),
        )
}

#[cfg(test)]
#[path = "properties/tests.rs"]
mod tests;
