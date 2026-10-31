//! Locates the build-output directories test fixtures live in, derived
//! from the repository root and `CARGO_TARGET_DIR` rather than the test
//! binary's location, which cargo does not guarantee.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// The repository root this crate was compiled from: `CARGO_MANIFEST_DIR`
/// is the `tools/xtask` directory, so two parents up is the checkout root.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CARGO_MANIFEST_DIR always has two parents")
        .to_path_buf()
}

/// The profile segment examples are built into: the same debug/release
/// decision cargo made for the test binary that is asking.
fn profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// The profile directory (`<target>/<profile>`) that holds `examples/`.
pub(crate) fn target_profile_dir() -> PathBuf {
    resolve(
        &repo_root(),
        std::env::var_os("CARGO_TARGET_DIR").as_deref(),
        profile_name(),
    )
}

/// The directory the fixture example binaries are built into.
pub(crate) fn examples_dir() -> PathBuf {
    target_profile_dir().join("examples")
}

/// Pure core of [`target_profile_dir`]. A relative `CARGO_TARGET_DIR` is
/// resolved against the repository root, which only matches cargo's own
/// resolution (against its invocation directory) when cargo ran from
/// there; the automation recipe exports an absolute path. A
/// `[build] target-dir` config entry is not observable from a test process.
fn resolve(repo_root: &Path, cargo_target_dir: Option<&OsStr>, profile: &str) -> PathBuf {
    let base = match cargo_target_dir {
        Some(dir) if !dir.is_empty() => {
            let dir = Path::new(dir);
            if dir.is_absolute() {
                dir.to_path_buf()
            } else {
                repo_root.join(dir)
            }
        }
        _ => repo_root.join("target"),
    };
    base.join(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_or_empty_target_dir_defaults_to_repo_target() {
        assert_eq!(
            resolve(Path::new("/repo"), None, "debug"),
            Path::new("/repo").join("target/debug")
        );
        assert_eq!(
            resolve(Path::new("/repo"), Some(OsStr::new("")), "debug"),
            Path::new("/repo").join("target/debug")
        );
    }

    #[test]
    fn absolute_target_dir_override_is_honored() {
        assert_eq!(
            resolve(Path::new("/repo"), Some(OsStr::new("/tmp/custom")), "debug"),
            Path::new("/tmp/custom").join("debug")
        );
    }

    #[test]
    fn relative_target_dir_override_resolves_against_the_repo_root() {
        assert_eq!(
            resolve(Path::new("/repo"), Some(OsStr::new("out")), "release"),
            Path::new("/repo").join("out/release")
        );
    }
}
