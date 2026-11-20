//! Mesh release policy over the runtime installer. Generic installation accepts explicit inputs.

mod product;
pub use product::{product_runtime_release, product_runtime_tree_sha256};
pub use skippy_runtime_install::discover_native_runtime_bundle_dirs as discover_native_runtime_bundle_dirs_for_release;
pub use skippy_runtime_install::*;
use std::path::PathBuf;

pub const CURRENT_MESH_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;

/// Runtime release required by this build, independent of the Mesh product version.
pub fn current_runtime_release() -> &'static str {
    skippy_native_runtime::runtime_release_version()
}

/// Skippy release this build is pinned to. Skippy publishes the native
/// runtimes; Mesh only downloads them. Must match the Skippy revision in
/// `Cargo.lock` (checked by a test below).
pub const SKIPPY_RELEASE: &str = "0.80.0";

/// Where Skippy publishes its releases.
pub const SKIPPY_RELEASES_URL: &str = "https://github.com/Mesh-LLM/skippy/releases";

/// Mesh names for the runtime locations, kept for existing installs. Skippy
/// reads its own `SKIPPY_NATIVE_RUNTIME_*` names.
const MESH_RUNTIME_ENV_ALIASES: [(&str, &str); 3] = [
    (
        "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
        NATIVE_RUNTIME_BUNDLE_DIR_ENV,
    ),
    (
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
        NATIVE_RUNTIME_CACHE_DIR_ENV,
    ),
    (
        "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL",
        NATIVE_RUNTIME_MANIFEST_URL_ENV,
    ),
];

/// Pass the `MESH_LLM_NATIVE_RUNTIME_*` settings on to Skippy under its own
/// names. A `SKIPPY_*` value that is already set wins.
///
/// # Safety
/// Mutates the process environment: call before any other thread exists.
pub unsafe fn adopt_mesh_native_runtime_env() {
    for (mesh, skippy) in MESH_RUNTIME_ENV_ALIASES {
        if std::env::var_os(skippy).is_none()
            && let Some(value) = std::env::var_os(mesh)
        {
            // SAFETY: single-threaded per this function's contract.
            unsafe { std::env::set_var(skippy, value) };
        }
    }
}

/// Runtime download location supplied to the runtime installer: the pinned
/// Skippy release, whatever the Mesh version or build.
pub fn mesh_native_runtime_catalog() -> NativeRuntimeCatalog {
    NativeRuntimeCatalog {
        release_tags: [(
            current_runtime_release().to_string(),
            format!("v{SKIPPY_RELEASE}"),
        )]
        .into(),
        releases_url: SKIPPY_RELEASES_URL.to_string(),
        rolling_release: None,
    }
}

pub fn default_release_manifest_url(skippy_release: &str) -> String {
    format!("{SKIPPY_RELEASES_URL}/download/v{skippy_release}/native-runtimes.json")
}

/// The runtime manifest this build downloads from by default.
pub fn default_manifest_url() -> String {
    mesh_native_runtime_catalog().manifest_url(current_runtime_release())
}

/// Options for the required Skippy release and Mesh publication channel.
pub fn mesh_native_runtime_install_options() -> NativeRuntimeInstallOptions {
    NativeRuntimeInstallOptions::new(current_runtime_release(), mesh_native_runtime_catalog())
}

/// Manifest options for the required Skippy release and Mesh publication channel.
pub fn mesh_native_runtime_manifest_options() -> NativeRuntimeManifestOptions {
    NativeRuntimeManifestOptions::new(current_runtime_release(), mesh_native_runtime_catalog())
}

pub fn discover_native_runtime_bundle_dirs(
    explicit_dirs: &[PathBuf],
) -> anyhow::Result<Vec<PathBuf>> {
    skippy_runtime_install::discover_native_runtime_bundle_dirs(
        explicit_dirs,
        current_runtime_release(),
    )
}

pub fn discover_local_native_runtimes(
    explicit_dirs: &[PathBuf],
    cache: &NativeRuntimeCache,
) -> anyhow::Result<Vec<InstalledNativeRuntime>> {
    skippy_runtime_install::discover_local_native_runtimes(
        explicit_dirs,
        cache,
        current_runtime_release(),
    )
}

pub fn discover_local_native_runtimes_with_filter(
    explicit_dirs: &[PathBuf],
    cache: &NativeRuntimeCache,
    include: impl Fn(&InstalledNativeRuntime) -> bool,
) -> anyhow::Result<Vec<InstalledNativeRuntime>> {
    skippy_runtime_install::discover_local_native_runtimes_with_filter(
        explicit_dirs,
        cache,
        current_runtime_release(),
        include,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtimes_download_from_the_pinned_skippy_release() {
        assert_eq!(
            default_manifest_url(),
            format!(
                "https://github.com/Mesh-LLM/skippy/releases/download/v{SKIPPY_RELEASE}/native-runtimes.json"
            )
        );
        assert_eq!(
            default_manifest_url(),
            default_release_manifest_url(SKIPPY_RELEASE)
        );
    }

    #[test]
    fn another_runtime_release_uses_its_own_tag() {
        assert_eq!(
            mesh_native_runtime_catalog().manifest_url("7.8.9"),
            default_release_manifest_url("7.8.9")
        );
    }

    #[test]
    fn pinned_skippy_release_matches_the_lockfile() {
        let lock = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../Cargo.lock"));
        let entry = format!("name = \"skippy-native-runtime\"\nversion = \"{SKIPPY_RELEASE}\"\n");
        assert!(
            lock.replace("\r\n", "\n").contains(&entry),
            "SKIPPY_RELEASE must equal the Skippy version in Cargo.lock"
        );
    }

    #[test]
    fn current_mesh_version_uses_release_version() {
        assert_eq!(CURRENT_MESH_VERSION, mesh_llm_build_info::RELEASE_VERSION);
    }

    #[test]
    fn mesh_options_supply_release_and_channel_to_both_operations() {
        let install = mesh_native_runtime_install_options();
        let manifest = mesh_native_runtime_manifest_options();
        assert_eq!(install.release_version, current_runtime_release());
        assert_eq!(manifest.release_version, current_runtime_release());
        assert_eq!(install.catalog, mesh_native_runtime_catalog());
        assert_eq!(manifest.catalog, install.catalog);
    }
}
