//! Mesh release policy over the runtime installer. Generic installation accepts explicit inputs.

mod product;
pub use product::{product_runtime_release, product_runtime_tree_sha256};
pub use skippy_runtime_install::*;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

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

/// Mesh names for the runtime locations. Skippy reads its own
/// `SKIPPY_NATIVE_RUNTIME_*` names; every function in this module that takes
/// a location also honours these, by passing them to Skippy as explicit
/// inputs. Empty values count as unset.
///
/// Cache directory and manifest URL are single values: an argument beats the
/// Mesh variable, which beats the Skippy one.
///
/// Bundle directories are a search list: arguments, then the Mesh variable,
/// then the Skippy variable, then the directories beside the executable.
/// Earlier entries win when two hold the same runtime. Every listed directory
/// must exist, whichever name supplied it, because Skippy's discovery always
/// reads its own variable.
pub const MESH_NATIVE_RUNTIME_BUNDLE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR";
pub const MESH_NATIVE_RUNTIME_CACHE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR";
pub const MESH_NATIVE_RUNTIME_MANIFEST_URL_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL";

fn bundle_dirs_from(value: Option<OsString>) -> Vec<PathBuf> {
    value
        .map(|value| {
            std::env::split_paths(&value)
                .filter(|dir| !dir.as_os_str().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn cache_dir_from(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

fn manifest_url_from(value: Option<OsString>) -> Option<String> {
    value
        .and_then(|value| value.into_string().ok())
        .filter(|value| !value.trim().is_empty())
}

/// `explicit` followed by the Mesh bundle variable's directories.
fn with_mesh_bundle_dirs(explicit: &[PathBuf], mesh: Option<OsString>) -> Vec<PathBuf> {
    let mut dirs = explicit.to_vec();
    for dir in bundle_dirs_from(mesh) {
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    dirs
}

fn mesh_bundle_dirs(explicit: &[PathBuf]) -> Vec<PathBuf> {
    with_mesh_bundle_dirs(
        explicit,
        std::env::var_os(MESH_NATIVE_RUNTIME_BUNDLE_DIR_ENV),
    )
}

fn mesh_cache_dir() -> Option<PathBuf> {
    cache_dir_from(std::env::var_os(MESH_NATIVE_RUNTIME_CACHE_DIR_ENV))
}

fn mesh_manifest_url() -> Option<String> {
    manifest_url_from(std::env::var_os(MESH_NATIVE_RUNTIME_MANIFEST_URL_ENV))
}

pub fn native_runtime_cache(cache_dir: Option<&Path>) -> anyhow::Result<NativeRuntimeCache> {
    let mesh = mesh_cache_dir();
    skippy_runtime_install::native_runtime_cache(cache_dir.or(mesh.as_deref()))
}

pub fn default_native_runtime_cache() -> anyhow::Result<NativeRuntimeCache> {
    native_runtime_cache(None)
}

fn with_mesh_install_env(mut options: NativeRuntimeInstallOptions) -> NativeRuntimeInstallOptions {
    options.bundle_dirs = mesh_bundle_dirs(&options.bundle_dirs);
    options.cache_dir = options.cache_dir.or_else(mesh_cache_dir);
    if options.manifest_path.is_none() {
        options.manifest_url = options.manifest_url.or_else(mesh_manifest_url);
    }
    options
}

fn with_mesh_manifest_env(
    mut options: NativeRuntimeManifestOptions,
) -> NativeRuntimeManifestOptions {
    options.bundle_dirs = mesh_bundle_dirs(&options.bundle_dirs);
    if options.manifest_path.is_none() {
        options.manifest_url = options.manifest_url.or_else(mesh_manifest_url);
    }
    options
}

pub async fn install_native_runtime(
    options: NativeRuntimeInstallOptions,
) -> anyhow::Result<NativeRuntimeInstallOutcome> {
    skippy_runtime_install::install_native_runtime(with_mesh_install_env(options)).await
}

pub async fn load_release_manifest(
    options: NativeRuntimeManifestOptions,
) -> anyhow::Result<NativeRuntimeReleaseManifest> {
    skippy_runtime_install::load_release_manifest(with_mesh_manifest_env(options)).await
}

pub async fn load_release_manifest_with_sources(
    options: NativeRuntimeManifestOptions,
) -> anyhow::Result<(NativeRuntimeReleaseManifest, NativeRuntimeCatalogSources)> {
    skippy_runtime_install::load_release_manifest_with_sources(with_mesh_manifest_env(options))
        .await
}

pub fn discover_native_runtime_bundle_dirs_for_release(
    explicit_dirs: &[PathBuf],
    release: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    skippy_runtime_install::discover_native_runtime_bundle_dirs(
        &mesh_bundle_dirs(explicit_dirs),
        release,
    )
}

/// Discover local runtimes for an explicit release rather than this build's.
pub fn discover_local_native_runtimes_for_release(
    explicit_dirs: &[PathBuf],
    cache: &NativeRuntimeCache,
    release: &str,
) -> anyhow::Result<Vec<InstalledNativeRuntime>> {
    skippy_runtime_install::discover_local_native_runtimes(
        &mesh_bundle_dirs(explicit_dirs),
        cache,
        release,
    )
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
        &mesh_bundle_dirs(explicit_dirs),
        current_runtime_release(),
    )
}

pub fn discover_local_native_runtimes(
    explicit_dirs: &[PathBuf],
    cache: &NativeRuntimeCache,
) -> anyhow::Result<Vec<InstalledNativeRuntime>> {
    skippy_runtime_install::discover_local_native_runtimes(
        &mesh_bundle_dirs(explicit_dirs),
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
        &mesh_bundle_dirs(explicit_dirs),
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
    fn mesh_bundle_variable_follows_explicit_dirs_and_drops_empty_entries() {
        let explicit = [PathBuf::from("/explicit")];
        assert_eq!(with_mesh_bundle_dirs(&explicit, None), explicit);
        assert_eq!(with_mesh_bundle_dirs(&explicit, Some("".into())), explicit);
        let joined = std::env::join_paths(["/mesh", "", "/explicit"]).unwrap();
        assert_eq!(
            with_mesh_bundle_dirs(&explicit, Some(joined)),
            [PathBuf::from("/explicit"), PathBuf::from("/mesh")]
        );
    }

    #[test]
    fn empty_mesh_cache_and_manifest_values_count_as_unset() {
        assert_eq!(cache_dir_from(None), None);
        assert_eq!(cache_dir_from(Some("".into())), None);
        assert_eq!(cache_dir_from(Some("/c".into())), Some(PathBuf::from("/c")));
        assert_eq!(manifest_url_from(None), None);
        assert_eq!(manifest_url_from(Some("  ".into())), None);
        assert_eq!(
            manifest_url_from(Some("https://example.invalid/m.json".into())).as_deref(),
            Some("https://example.invalid/m.json")
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
