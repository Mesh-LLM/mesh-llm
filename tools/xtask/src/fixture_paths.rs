//! Locates fixture outputs using Cargo's build-time profile directory, including
//! explicit targets and Cargo configuration that is unavailable at test runtime.

use std::path::PathBuf;

/// The profile directory containing this compilation's binaries and examples.
pub(crate) fn target_profile_dir() -> PathBuf {
    PathBuf::from(env!("XTASK_CARGO_PROFILE_DIR"))
}

/// The directory the fixture example binaries are built into.
pub(crate) fn examples_dir() -> PathBuf {
    target_profile_dir().join("examples")
}
