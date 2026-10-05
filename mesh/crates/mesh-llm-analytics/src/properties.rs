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

/// How this process was started, as a fixed string.
///
/// This exists because an install identifier is one *state directory*, not
/// one machine. A container gets a fresh `$HOME` on every run, so it mints a
/// new identifier and reports `install_first_run` every single time — which
/// reads as organic adoption and is not. Labelling the environment keeps
/// those runs in the data (they are real executions) while making them
/// separable from someone's laptop.
///
/// Deliberately a label and not a new opt-out: suppressing these runs would
/// silently discard data, and the honest fix for "is this a person?" is to
/// let the question be asked rather than answered in advance.
///
/// Checked most-specific first. A CI job usually runs in a container too, and
/// "ci" is the more useful of the two answers.
#[must_use]
pub fn exec_env() -> &'static str {
    if crate::consent::detect_ci() {
        "ci"
    } else if in_container() {
        "container"
    } else if in_service_manager() {
        "service"
    } else {
        "plain"
    }
}

/// Files and variables that mark a container runtime.
///
/// `container` is set by podman, systemd-nspawn, and LXC;
/// `KUBERNETES_SERVICE_HOST` is injected into every pod.
fn in_container() -> bool {
    std::path::Path::new("/.dockerenv").exists()
        || std::path::Path::new("/run/.containerenv").exists()
        || env_is_set("container")
        || env_is_set("KUBERNETES_SERVICE_HOST")
}

/// Whether a service manager started this process rather than a person.
///
/// `INVOCATION_ID` is set by systemd for every unit it starts. launchd has no
/// equivalent marker, so macOS agents fall through to `plain`; that is a known
/// blind spot rather than a claim that none exist.
fn in_service_manager() -> bool {
    env_is_set("INVOCATION_ID")
}

fn env_is_set(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|value| !value.is_empty())
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
        .with("exec_env", exec_env())
        .with("$lib", LIB_NAME)
        .with(
            "$lib_version",
            crate::Label::sanitize_or_redact(BUILD_VERSION),
        )
}

#[cfg(test)]
#[path = "properties/tests.rs"]
mod tests;
