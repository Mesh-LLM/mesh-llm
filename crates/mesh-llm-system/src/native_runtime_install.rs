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

/// Mesh release-channel policy supplied to the runtime installer.
pub fn mesh_native_runtime_catalog() -> NativeRuntimeCatalog {
    mesh_catalog_for(
        mesh_llm_build_info::BUILD_VERSION,
        CURRENT_MESH_VERSION,
        current_runtime_release(),
    )
}

fn mesh_catalog_for(build: &str, product: &str, runtime: &str) -> NativeRuntimeCatalog {
    NativeRuntimeCatalog {
        release_tags: [(runtime.to_string(), format!("v{product}"))].into(),
        releases_url: "https://github.com/Mesh-LLM/mesh-llm/releases".to_string(),
        rolling_release: mesh_llm_build_info::is_sha_build(build).then(|| runtime.to_string()),
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
    fn runtime_release_maps_to_product_publication_without_identity_coupling() {
        for (product, runtime) in [("0.76.1", "0.76.1"), ("99.0.0", "1.2.3")] {
            let catalog = mesh_catalog_for(product, product, runtime);
            assert_eq!(
                catalog.manifest_url(runtime),
                default_release_manifest_url(product)
            );
            assert_eq!(
                catalog.manifest_url("7.8.9"),
                default_release_manifest_url("7.8.9")
            );
            for build in [
                format!("{product}+gabcdef"),
                format!("{product}+gabcdef.dirty"),
            ] {
                let rolling = mesh_catalog_for(&build, product, runtime);
                assert_eq!(
                    rolling.manifest_url(runtime),
                    default_manifest_url(&build, product)
                );
                assert_eq!(
                    rolling.manifest_url("7.8.9"),
                    default_release_manifest_url("7.8.9")
                );
            }
        }
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
