//! Native runtime install option and outcome types.

use serde::{Deserialize, Serialize};
use skippy_native_runtime::{InstalledNativeRuntime, RuntimeSelection};
use std::path::PathBuf;
use std::sync::Arc;
pub const NATIVE_RUNTIME_CACHE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR";
pub const NATIVE_RUNTIME_MANIFEST_URL_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL";

pub type NativeRuntimeDownloadProgressCallback =
    Arc<dyn Fn(NativeRuntimeDownloadProgress) + Send + Sync + 'static>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeVerificationPolicy {
    #[default]
    RequireChecksum,
    RequireChecksumAndSignature,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeBundleInstallPolicy {
    #[default]
    UseInPlace,
    InstallExplicitBundlesIntoCache,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeDownloadProgress {
    pub native_runtime_id: String,
    pub url: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub finished: bool,
}

/// Catalog location and release-channel selection supplied by the embedding product.
/// Loading a catalog never consults compiled product build metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRuntimeCatalog {
    /// Root containing `download/v<release>/native-runtimes.json`.
    pub releases_url: String,
    /// If set, requests for this release use `latest/download/native-runtimes.json`.
    /// Other releases remain pinned. Standalone callers can leave this unset.
    pub rolling_release: Option<String>,
}

impl NativeRuntimeCatalog {
    pub fn manifest_url(&self, release: &str) -> String {
        let root = self.releases_url.trim_end_matches('/');
        if self.rolling_release.as_deref() == Some(release) {
            format!("{root}/latest/download/native-runtimes.json")
        } else {
            format!("{root}/download/v{release}/native-runtimes.json")
        }
    }
}

#[derive(Clone)]
pub struct NativeRuntimeManifestOptions {
    pub catalog: NativeRuntimeCatalog,
    pub mesh_version: String,
    pub manifest_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub allow_default_manifest_url: bool,
}

#[derive(Clone)]
pub struct NativeRuntimeInstallOptions {
    pub catalog: NativeRuntimeCatalog,
    pub mesh_version: String,
    pub skippy_abi_version: Option<String>,
    pub selection: RuntimeSelection,
    pub manifest_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub verification_policy: NativeRuntimeVerificationPolicy,
    pub bundle_install_policy: NativeRuntimeBundleInstallPolicy,
    pub progress: Option<NativeRuntimeDownloadProgressCallback>,
    pub allow_download: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeInstallStatus {
    AlreadyInstalled,
    Installed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeInstallOutcome {
    pub status: NativeRuntimeInstallStatus,
    pub runtime: InstalledNativeRuntime,
    pub resolution: skippy_native_runtime::NativeRuntimeResolution,
    /// Which catalogs were consulted to reach this resolution.
    #[serde(default)]
    pub sources: crate::manifest::NativeRuntimeCatalogSources,
}

impl NativeRuntimeManifestOptions {
    /// Construct options from explicit runtime release metadata and catalog policy.
    pub fn new(release_version: impl Into<String>, catalog: NativeRuntimeCatalog) -> Self {
        Self {
            catalog,
            mesh_version: release_version.into(),
            manifest_path: None,
            manifest_url: None,
            bundle_dirs: Vec::new(),
            allow_default_manifest_url: true,
        }
    }
}

impl NativeRuntimeInstallOptions {
    /// Construct options without consulting an embedding product's build metadata.
    pub fn new(release_version: impl Into<String>, catalog: NativeRuntimeCatalog) -> Self {
        Self {
            catalog,
            mesh_version: release_version.into(),
            skippy_abi_version: None,
            selection: RuntimeSelection::Recommended,
            manifest_path: None,
            manifest_url: None,
            bundle_dirs: Vec::new(),
            cache_dir: None,
            verification_policy: NativeRuntimeVerificationPolicy::RequireChecksum,
            bundle_install_policy: NativeRuntimeBundleInstallPolicy::UseInPlace,
            progress: None,
            allow_download: true,
        }
    }
}
