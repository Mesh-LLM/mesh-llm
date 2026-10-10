//! Where a release's bundled default plugins are.
//!
//! A release archive carries each default plugin's release archive, unchanged,
//! in `plugins/` beside the `mesh-llm` binary (the release pipeline checks
//! each against the plugin release's SHA256SUMS and its build-provenance
//! attestation). A package or a formula may keep that directory under its
//! own prefix instead. The node installs a default only from here, after
//! checking the archive against this build's pin; it never downloads one.

use std::path::{Path, PathBuf};

/// Names a bundled-plugins directory to use instead of the one beside the
/// binary (a development build, or a test).
pub const BUNDLED_PLUGINS_DIR_ENV: &str = "MESH_LLM_BUNDLED_PLUGINS_DIR";

/// The release these plugins were bundled with; versioned package installs
/// keep them under `lib/mesh-llm/<release>/plugins`.
pub const BUNDLED_RELEASE: &str = env!("CARGO_PKG_VERSION");

/// The file name a plugin's release archive has in the bundle, the name its
/// own release published it under.
pub fn bundled_archive_name(name: &str, version: &str, target: &str) -> String {
    format!("{name}-{version}-{target}.tar.gz")
}

/// The directories a bundled plugin may be in, in order, for the binary at
/// `executable` of release `release`: beside the binary (the release archive
/// as installed by `install.sh`), then the layouts packages and formulas use
/// for the native runtime (`lib/mesh-llm/<release>`, `lib/mesh-llm`,
/// `libexec`).
pub fn bundled_plugin_dir_candidates(executable: &Path, release: &str) -> Vec<PathBuf> {
    let executable = executable
        .canonicalize()
        .unwrap_or_else(|_| executable.to_path_buf());
    let Some(executable_dir) = executable.parent() else {
        return Vec::new();
    };
    let mut candidates = vec![executable_dir.join("plugins")];
    if let Some(prefix) = executable_dir.parent() {
        candidates.push(
            prefix
                .join("lib")
                .join("mesh-llm")
                .join(release)
                .join("plugins"),
        );
        candidates.push(prefix.join("lib").join("mesh-llm").join("plugins"));
        candidates.push(prefix.join("libexec").join("plugins"));
    }
    candidates
}

/// This install's bundled-plugins directory: `MESH_LLM_BUNDLED_PLUGINS_DIR`
/// when set, else the first candidate that exists. `None` when there is none
/// (a development build, or a platform the release bundles nothing for).
pub fn bundled_plugins_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(BUNDLED_PLUGINS_DIR_ENV).filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let executable = std::env::current_exe().ok()?;
    bundled_plugin_dir_candidates(&executable, BUNDLED_RELEASE)
        .into_iter()
        .find(|dir| dir.is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_follow_the_native_runtime_layouts() {
        let temp = tempfile::tempdir().unwrap();
        let prefix = temp.path().canonicalize().unwrap();
        let binary = prefix.join("bin").join("mesh-llm");
        assert_eq!(
            bundled_plugin_dir_candidates(&binary, "0.79.0"),
            vec![
                prefix.join("bin").join("plugins"),
                prefix.join("lib/mesh-llm/0.79.0/plugins"),
                prefix.join("lib/mesh-llm/plugins"),
                prefix.join("libexec/plugins"),
            ]
        );
    }

    #[test]
    fn the_archive_name_is_the_plugin_releases_own() {
        assert_eq!(
            bundled_archive_name("capsules", "0.1.3", "x86_64-unknown-linux-gnu"),
            "capsules-0.1.3-x86_64-unknown-linux-gnu.tar.gz"
        );
    }
}
