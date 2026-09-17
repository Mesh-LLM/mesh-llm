//! Mesh release policy over the runtime installer. Generic installation accepts explicit inputs.

pub use skippy_runtime_install::*;
use std::path::PathBuf;

pub const CURRENT_MESH_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;

/// Mesh release-channel policy supplied to the runtime installer.
pub fn mesh_native_runtime_catalog() -> NativeRuntimeCatalog {
    NativeRuntimeCatalog {
        releases_url: "https://github.com/Mesh-LLM/mesh-llm/releases".to_string(),
        rolling_release: mesh_llm_build_info::is_sha_build(mesh_llm_build_info::BUILD_VERSION)
            .then(|| CURRENT_MESH_VERSION.to_string()),
    }
}

pub fn default_release_manifest_url(mesh_version: &str) -> String {
    format!(
        "https://github.com/Mesh-LLM/mesh-llm/releases/download/v{mesh_version}/native-runtimes.json"
    )
}

pub fn default_manifest_url(build_version: &str, release_version: &str) -> String {
    if mesh_llm_build_info::is_sha_build(build_version) {
        "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
            .to_string()
    } else {
        default_release_manifest_url(release_version)
    }
}

/// Options for the current Mesh release and build channel.
pub fn mesh_native_runtime_install_options() -> NativeRuntimeInstallOptions {
    NativeRuntimeInstallOptions::new(CURRENT_MESH_VERSION, mesh_native_runtime_catalog())
}

/// Manifest options for the current Mesh release and build channel.
pub fn mesh_native_runtime_manifest_options() -> NativeRuntimeManifestOptions {
    NativeRuntimeManifestOptions::new(CURRENT_MESH_VERSION, mesh_native_runtime_catalog())
}

pub fn discover_native_runtime_bundle_dirs(
    explicit_dirs: &[PathBuf],
) -> anyhow::Result<Vec<PathBuf>> {
    skippy_runtime_install::discover_native_runtime_bundle_dirs(explicit_dirs, CURRENT_MESH_VERSION)
}

pub fn discover_local_native_runtimes(
    explicit_dirs: &[PathBuf],
    cache: &NativeRuntimeCache,
) -> anyhow::Result<Vec<InstalledNativeRuntime>> {
    skippy_runtime_install::discover_local_native_runtimes(
        explicit_dirs,
        cache,
        CURRENT_MESH_VERSION,
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
        CURRENT_MESH_VERSION,
        include,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_manifest_url_uses_release_download_for_release_builds() {
        assert_eq!(
            default_manifest_url("0.68.0", "0.68.0"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.68.0/native-runtimes.json"
        );
    }

    #[test]
    fn default_manifest_url_uses_latest_download_for_sha_builds() {
        assert_eq!(
            default_manifest_url("0.68.0+gAB131C", "0.68.0"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
        );
        assert_eq!(
            default_manifest_url("0.68.0+gAB131C.dirty", "0.68.0"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
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
        assert_eq!(install.release_version, CURRENT_MESH_VERSION);
        assert_eq!(manifest.release_version, CURRENT_MESH_VERSION);
        assert_eq!(install.catalog, mesh_native_runtime_catalog());
        assert_eq!(manifest.catalog, install.catalog);
    }
}
