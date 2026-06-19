use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
pub use mesh_llm_native_runtime::{
    CachePrunePlan, CandidateEvaluation, CandidateRejection, HostGpuProfile, HostRuntimeProfile,
    InstalledNativeRuntime, NATIVE_RUNTIME_MANIFEST_FILE, NativeRuntimeArtifact,
    NativeRuntimeCache, NativeRuntimeCacheRoot, NativeRuntimeFlavor, NativeRuntimeFlavorParseError,
    NativeRuntimeLoadPlan, NativeRuntimeManifest, NativeRuntimePruneMode,
    NativeRuntimeReleaseManifest, NativeRuntimeResolution, NativeRuntimeResolver,
    NativeRuntimeSource, RuntimeSelection, native_runtime_cache_root, select_native_runtime,
};

use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

pub const CURRENT_MESH_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeDownloadProgress {
    pub native_runtime_id: String,
    pub url: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub finished: bool,
}

#[derive(Clone)]
pub struct NativeRuntimeManifestOptions {
    pub mesh_version: String,
    pub manifest_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub allow_default_manifest_url: bool,
}

#[derive(Clone)]
pub struct NativeRuntimeInstallOptions {
    pub mesh_version: String,
    pub skippy_abi_version: Option<String>,
    pub selection: RuntimeSelection,
    pub manifest_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub verification_policy: NativeRuntimeVerificationPolicy,
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
    pub resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
}

impl Default for NativeRuntimeManifestOptions {
    fn default() -> Self {
        Self {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            manifest_path: None,
            manifest_url: None,
            bundle_dirs: Vec::new(),
            allow_default_manifest_url: true,
        }
    }
}

impl Default for NativeRuntimeInstallOptions {
    fn default() -> Self {
        Self {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            skippy_abi_version: None,
            selection: RuntimeSelection::Recommended,
            manifest_path: None,
            manifest_url: None,
            bundle_dirs: Vec::new(),
            cache_dir: None,
            verification_policy: NativeRuntimeVerificationPolicy::RequireChecksum,
            progress: None,
            allow_download: true,
        }
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

fn request_default_manifest_url(mesh_version: &str) -> String {
    if mesh_version == mesh_llm_build_info::RELEASE_VERSION {
        default_manifest_url(
            mesh_llm_build_info::BUILD_VERSION,
            mesh_llm_build_info::RELEASE_VERSION,
        )
    } else {
        default_release_manifest_url(mesh_version)
    }
}

pub fn current_skippy_abi_version() -> String {
    format!(
        "{}.{}.{}",
        skippy_ffi::ABI_VERSION_MAJOR,
        skippy_ffi::ABI_VERSION_MINOR,
        skippy_ffi::ABI_VERSION_PATCH
    )
}

pub fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
    native_runtime_cache(None)
}

pub fn native_runtime_cache(cache_dir: Option<&Path>) -> Result<NativeRuntimeCache> {
    let root = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => match std::env::var_os(NATIVE_RUNTIME_CACHE_DIR_ENV) {
            Some(path) => PathBuf::from(path),
            None => dirs::cache_dir()
                .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
                .context("cannot determine native runtime cache directory")?
                .join("mesh-llm")
                .join("native-runtimes"),
        },
    };
    Ok(NativeRuntimeCache::new(root))
}

pub fn host_runtime_profile() -> HostRuntimeProfile {
    mesh_llm_hardware_profile::host_runtime_profile()
}

pub async fn load_release_manifest(
    options: NativeRuntimeManifestOptions,
) -> Result<NativeRuntimeReleaseManifest> {
    let mut artifacts = Vec::new();
    let mut mesh_version = options.mesh_version.clone();
    let mut skippy_abi = current_skippy_abi_version();
    if let Some(path) = options.manifest_path {
        let manifest = NativeRuntimeReleaseManifest::read_from_path(&path)?;
        mesh_version = manifest.mesh_version.clone();
        skippy_abi = manifest.skippy_abi.clone();
        artifacts.extend(manifest.artifacts);
    } else if let Some(url) = manifest_url(&options) {
        let manifest = download_release_manifest(&url).await?;
        mesh_version = manifest.mesh_version.clone();
        skippy_abi = manifest.skippy_abi.clone();
        artifacts.extend(manifest.artifacts);
    }
    append_bundle_artifacts(
        &mut artifacts,
        &mut mesh_version,
        &mut skippy_abi,
        &options.bundle_dirs,
    )?;
    Ok(NativeRuntimeReleaseManifest {
        mesh_version,
        skippy_abi,
        artifacts,
    })
}

pub async fn install_native_runtime(
    options: NativeRuntimeInstallOptions,
) -> Result<NativeRuntimeInstallOutcome> {
    let manifest = load_release_manifest(NativeRuntimeManifestOptions {
        mesh_version: options.mesh_version.clone(),
        manifest_path: options.manifest_path.clone(),
        manifest_url: options.manifest_url.clone(),
        bundle_dirs: options.bundle_dirs.clone(),
        allow_default_manifest_url: true,
    })
    .await?;
    if manifest.artifacts.is_empty() {
        bail!("no native runtime manifest entries found");
    }
    let skippy_abi_version = options
        .skippy_abi_version
        .clone()
        .unwrap_or_else(|| manifest.skippy_abi.clone());
    let cache = native_runtime_cache(options.cache_dir.as_deref())?;
    let resolution = NativeRuntimeResolver::new(
        &options.mesh_version,
        host_runtime_profile(),
        manifest,
        cache.clone(),
    )
    .with_skippy_abi_version(skippy_abi_version)
    .with_bundle_dirs(options.bundle_dirs.clone())
    .resolve(&options.selection)?;
    install_resolved_runtime(&cache, resolution, &options).await
}

async fn install_resolved_runtime(
    cache: &NativeRuntimeCache,
    resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
    options: &NativeRuntimeInstallOptions,
) -> Result<NativeRuntimeInstallOutcome> {
    match &resolution.source {
        NativeRuntimeSource::Installed { path: _ } => installed_outcome(cache, resolution),
        NativeRuntimeSource::Bundle { path } => {
            let runtime = cache.install_from_dir(path)?;
            Ok(NativeRuntimeInstallOutcome {
                status: NativeRuntimeInstallStatus::Installed,
                runtime,
                resolution,
            })
        }
        NativeRuntimeSource::Download { url } if options.allow_download => {
            let runtime =
                download_and_install_runtime(cache, &resolution.selected, url, options).await?;
            Ok(NativeRuntimeInstallOutcome {
                status: NativeRuntimeInstallStatus::Installed,
                runtime,
                resolution,
            })
        }
        NativeRuntimeSource::Download { url: _ } => {
            bail!("selected native runtime is downloadable, but downloads are disabled")
        }
        NativeRuntimeSource::Missing => {
            bail!(
                "selected native runtime {} is not installed and no bundle or download URL was available",
                resolution.selected.id
            )
        }
    }
}

fn installed_outcome(
    cache: &NativeRuntimeCache,
    resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
) -> Result<NativeRuntimeInstallOutcome> {
    let runtime = cache
        .find_installed(
            resolution.selected.mesh_version_or(CURRENT_MESH_VERSION),
            resolution.selected.native_runtime_id(),
        )?
        .context("selected native runtime was not found in cache")?;
    Ok(NativeRuntimeInstallOutcome {
        status: NativeRuntimeInstallStatus::AlreadyInstalled,
        runtime,
        resolution,
    })
}

async fn download_release_manifest(url: &str) -> Result<NativeRuntimeReleaseManifest> {
    let text = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("build native runtime manifest HTTP client")?
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .with_context(|| format!("download native runtime release manifest {url}"))?
        .error_for_status()
        .with_context(|| format!("native runtime release manifest request failed for {url}"))?
        .text()
        .await
        .with_context(|| format!("read native runtime release manifest {url}"))?;
    NativeRuntimeReleaseManifest::from_json_str(&text)
        .with_context(|| format!("parse native runtime release manifest {url}"))
}

async fn download_and_install_runtime(
    cache: &NativeRuntimeCache,
    artifact: &NativeRuntimeArtifact,
    url: &str,
    options: &NativeRuntimeInstallOptions,
) -> Result<InstalledNativeRuntime> {
    let temp = tempfile::Builder::new()
        .prefix("mesh-native-runtime-")
        .tempdir()
        .context("create native runtime download workspace")?;
    let archive = temp
        .path()
        .join(format!("{}.tar.gz", artifact.native_runtime_id()));
    download_runtime_archive(url, &archive, artifact, options).await?;
    let extracted = temp.path().join("extracted");
    fs::create_dir_all(&extracted).with_context(|| {
        format!(
            "create native runtime extraction dir {}",
            extracted.display()
        )
    })?;
    extract_runtime_archive(&archive, &extracted)?;
    let bundle_dir = find_extracted_runtime_dir(&extracted)?;
    cache.install_from_dir(&bundle_dir)
}

async fn download_runtime_archive(
    url: &str,
    path: &Path,
    artifact: &NativeRuntimeArtifact,
    options: &NativeRuntimeInstallOptions,
) -> Result<()> {
    verify_download_policy_before_fetch(artifact, options.verification_policy)?;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("build native runtime download HTTP client")?
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .with_context(|| format!("download native runtime {url}"))?
        .error_for_status()
        .with_context(|| format!("native runtime request failed for {url}"))?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("create native runtime archive {}", path.display()))?;
    let mut downloaded = 0_u64;
    let mut hasher = sha2::Sha256::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read native runtime body from {url}"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("write native runtime archive {}", path.display()))?;
        downloaded += chunk.len() as u64;
        sha2::Digest::update(&mut hasher, &chunk);
        emit_download_progress(artifact, url, downloaded, total, false, options);
    }
    file.flush()
        .await
        .with_context(|| format!("flush native runtime archive {}", path.display()))?;
    emit_download_progress(artifact, url, downloaded, total, true, options);
    verify_downloaded_archive(artifact, hasher)?;
    Ok(())
}

fn verify_download_policy_before_fetch(
    artifact: &NativeRuntimeArtifact,
    policy: NativeRuntimeVerificationPolicy,
) -> Result<()> {
    if artifact.sha256.is_none() {
        bail!(
            "native runtime {} is missing required sha256 verification metadata",
            artifact.native_runtime_id()
        );
    }
    if policy == NativeRuntimeVerificationPolicy::RequireChecksumAndSignature {
        let signature = artifact.signature.as_deref().unwrap_or_default();
        if signature.trim().is_empty() {
            bail!(
                "native runtime {} is missing required signature metadata",
                artifact.native_runtime_id()
            );
        }
        bail!("native runtime signature verification is not implemented yet");
    }
    Ok(())
}

fn verify_downloaded_archive(artifact: &NativeRuntimeArtifact, hasher: sha2::Sha256) -> Result<()> {
    let expected = artifact
        .sha256
        .as_deref()
        .context("native runtime sha256 missing after download")?;
    let expected = normalize_sha256(expected)?;
    let actual = hex::encode(sha2::Digest::finalize(hasher));
    if actual != expected {
        bail!("native runtime checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn emit_download_progress(
    artifact: &NativeRuntimeArtifact,
    url: &str,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    finished: bool,
    options: &NativeRuntimeInstallOptions,
) {
    let Some(progress) = &options.progress else {
        return;
    };
    progress(NativeRuntimeDownloadProgress {
        native_runtime_id: artifact.id.clone(),
        url: url.to_string(),
        downloaded_bytes,
        total_bytes,
        finished,
    });
}

fn manifest_url(options: &NativeRuntimeManifestOptions) -> Option<String> {
    options
        .manifest_url
        .clone()
        .or_else(|| {
            std::env::var(NATIVE_RUNTIME_MANIFEST_URL_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .or_else(|| {
            (options.allow_default_manifest_url && options.bundle_dirs.is_empty())
                .then(|| request_default_manifest_url(&options.mesh_version))
        })
}

fn append_bundle_artifacts(
    artifacts: &mut Vec<NativeRuntimeArtifact>,
    mesh_version: &mut String,
    skippy_abi: &mut String,
    bundle_dirs: &[PathBuf],
) -> Result<()> {
    for dir in bundle_dirs {
        let manifest = NativeRuntimeManifest::read_from_dir(dir)
            .with_context(|| format!("read bundled native runtime {}", dir.display()))?;
        if let Some(version) = &manifest.runtime.mesh_version {
            *mesh_version = version.clone();
        }
        *skippy_abi = manifest.runtime.skippy_abi.clone();
        artifacts.push(manifest.runtime);
    }
    Ok(())
}

fn normalize_sha256(value: &str) -> Result<String> {
    let trimmed = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    let digest = trimmed
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if digest.len() == 64 && digest.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Ok(digest)
    } else {
        bail!("native runtime manifest contains invalid sha256: {value}");
    }
}

fn extract_runtime_archive(archive: &Path, extracted: &Path) -> Result<()> {
    let file = fs::File::open(archive)
        .with_context(|| format!("open native runtime archive {}", archive.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(extracted).with_context(|| {
        format!(
            "extract native runtime archive into {}",
            extracted.display()
        )
    })
}

fn find_extracted_runtime_dir(extracted: &Path) -> Result<PathBuf> {
    let mut matches = Vec::new();
    collect_runtime_manifest_dirs(extracted, &mut matches)?;
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!("downloaded native runtime archive did not contain a manifest.json"),
        count => bail!("downloaded native runtime archive contained {count} manifest.json files"),
    }
}

fn collect_runtime_manifest_dirs(dir: &Path, matches: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if path
                .join(mesh_llm_native_runtime::NATIVE_RUNTIME_MANIFEST_FILE)
                .is_file()
            {
                matches.push(path);
            } else {
                collect_runtime_manifest_dirs(&path, matches)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_native_runtime::{NativeRuntimeBackend, NativeRuntimePlatform};
    use std::sync::Mutex;

    static MANIFEST_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn artifact_with_sha(signature: Option<&str>) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: "meshllm-runtime-linux-x86_64-cpu".to_string(),
            mesh_version: Some(CURRENT_MESH_VERSION.to_string()),
            skippy_abi: current_skippy_abi_version(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                target: Some("x86_64-unknown-linux-gnu".to_string()),
            },
            backend: NativeRuntimeBackend::cpu(),
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            url: Some("https://example.invalid/runtime.tar.gz".to_string()),
            sha256: Some("a".repeat(64)),
            signature: signature.map(str::to_string),
        }
    }

    #[test]
    fn checksum_policy_requires_sha256() {
        let mut artifact = artifact_with_sha(None);
        artifact.sha256 = None;

        let err = verify_download_policy_before_fetch(
            &artifact,
            NativeRuntimeVerificationPolicy::RequireChecksum,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("missing required sha256"),
            "{err:?}"
        );
    }

    #[test]
    fn signature_policy_fails_closed_until_implemented() {
        let artifact = artifact_with_sha(Some("signature"));

        let err = verify_download_policy_before_fetch(
            &artifact,
            NativeRuntimeVerificationPolicy::RequireChecksumAndSignature,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("signature verification is not implemented"),
            "{err:?}"
        );
    }

    #[test]
    fn default_manifest_url_is_skipped_for_bundle_only_resolution() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }

        let options = NativeRuntimeManifestOptions {
            bundle_dirs: vec![PathBuf::from("runtime-bundle")],
            ..Default::default()
        };

        assert!(manifest_url(&options).is_none());
    }

    #[test]
    fn explicit_manifest_url_wins_over_env_and_default() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                NATIVE_RUNTIME_MANIFEST_URL_ENV,
                "https://example.invalid/from-env.json",
            );
        }

        let options = NativeRuntimeManifestOptions {
            manifest_url: Some("https://example.invalid/from-arg.json".to_string()),
            ..Default::default()
        };

        assert_eq!(
            manifest_url(&options).as_deref(),
            Some("https://example.invalid/from-arg.json")
        );

        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }
    }

    #[test]
    fn env_manifest_url_wins_over_default() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                NATIVE_RUNTIME_MANIFEST_URL_ENV,
                "https://example.invalid/from-env.json",
            );
        }

        let url = manifest_url(&NativeRuntimeManifestOptions::default());

        assert_eq!(
            url.as_deref(),
            Some("https://example.invalid/from-env.json")
        );

        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }
    }

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
    fn non_default_mesh_version_request_uses_versioned_release_url() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }

        let options = NativeRuntimeManifestOptions {
            mesh_version: "0.67.0".to_string(),
            allow_default_manifest_url: true,
            ..Default::default()
        };

        assert_eq!(
            manifest_url(&options).as_deref(),
            Some(
                "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.67.0/native-runtimes.json"
            )
        );
    }

    #[test]
    fn current_mesh_version_uses_release_version() {
        assert_eq!(CURRENT_MESH_VERSION, mesh_llm_build_info::RELEASE_VERSION);
    }

    #[test]
    fn load_release_manifest_prefers_explicit_path_over_env_and_default() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                NATIVE_RUNTIME_MANIFEST_URL_ENV,
                "https://example.invalid/should-not-be-fetched.json",
            );
        }

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("native-runtimes.json");
        std::fs::write(
            &path,
            format!(
                r#"{{
  "mesh_version": "0.68.0",
  "skippy_abi": "{}",
  "artifacts": []
}}"#,
                current_skippy_abi_version()
            ),
        )
        .unwrap();

        let manifest = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(load_release_manifest(NativeRuntimeManifestOptions {
                mesh_version: "0.0.0+gLOCAL".to_string(),
                manifest_path: Some(path),
                manifest_url: Some("https://example.invalid/from-arg.json".to_string()),
                bundle_dirs: Vec::new(),
                allow_default_manifest_url: true,
            }))
            .unwrap();

        assert_eq!(manifest.mesh_version, "0.68.0");
        assert!(manifest.artifacts.is_empty());

        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }
    }
}
