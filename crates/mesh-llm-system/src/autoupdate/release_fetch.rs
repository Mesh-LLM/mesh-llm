use anyhow::{Context, Result};
use std::io;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};

use crate::backend;
use crate::release_target::ReleaseTarget;

#[path = "release_integrity.rs"]
mod release_integrity;

use release_integrity::{
    ReleaseAsset, github_release_asset_sha256, release_asset, verify_release_asset_bytes,
};

const DEFAULT_RELEASE_REPO: &str = "Mesh-LLM/mesh-llm";
const PRODUCT_MANIFEST_NAME: &str = "product-manifest.json";
const NATIVE_RUNTIMES_DIR_NAME: &str = "native-runtimes";
const PATH_WRITE_PROBE_PREFIX: &str = ".mesh-llm-write-probe";
#[cfg(not(windows))]
pub(super) const INSTALL_SCRIPT_URL: &str =
    "https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh";
pub(super) const RELEASES_URL: &str = "https://github.com/Mesh-LLM/mesh-llm/releases/latest";
const SELF_UPDATE_REPO_ENV: &str = "MESH_LLM_SELF_UPDATE_REPO";

pub(super) enum InstallOutcome {
    #[cfg_attr(windows, allow(dead_code))]
    RestartNow,
    #[cfg_attr(windows, allow(dead_code))]
    ExitNow,
    #[cfg_attr(not(windows), allow(dead_code))]
    HandoffAndExit,
}

#[derive(Clone, Copy)]
pub(super) enum PostInstallAction {
    RestartCurrentProcess,
    ExitAfterInstall,
}

pub(super) struct ReleaseInfo {
    pub(super) tag: String,
    pub(super) version: String,
    assets: Vec<ReleaseAsset>,
}

#[derive(Clone, Copy)]
pub(super) enum ReleaseAssetPreference {
    StableFirst,
    VersionedFirst,
}

pub(super) fn platform_has_release_assets() -> bool {
    platform_has_release_assets_for(std::env::consts::OS, std::env::consts::ARCH)
}

pub(super) fn platform_has_release_assets_for(os: &str, arch: &str) -> bool {
    backend::BinaryFlavor::ALL.into_iter().any(|flavor| {
        ReleaseTarget::from_raw(os, arch, flavor)
            .map(|target| target.support_status().is_supported())
            .unwrap_or(false)
    })
}

pub async fn latest_release_version() -> Option<String> {
    latest_release_info().await.map(|release| release.version)
}

pub(super) async fn latest_release_info() -> Option<ReleaseInfo> {
    fetch_release_info(&latest_release_api_url()).await
}

async fn release_info_for_tag(tag: &str) -> Option<ReleaseInfo> {
    fetch_release_info(&release_api_url_for_tag(tag)).await
}

async fn fetch_release_info(url: &str) -> Option<ReleaseInfo> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    release_info_from_json(&body)
}

pub fn version_newer(a: &str, b: &str) -> bool {
    let (Ok(a_parsed), Ok(b_parsed)) = (semver::Version::parse(a), semver::Version::parse(b))
    else {
        return false;
    };

    let a_is_sha = mesh_llm_build_info::is_sha_build(a);
    let b_is_sha = mesh_llm_build_info::is_sha_build(b);

    if !a_is_sha && b_is_sha {
        return false;
    }

    if a_is_sha && !b_is_sha {
        return true;
    }

    a_parsed > b_parsed
}

pub(super) fn current_release_target(flavor: backend::BinaryFlavor) -> Option<ReleaseTarget> {
    ReleaseTarget::from_raw(std::env::consts::OS, std::env::consts::ARCH, flavor).ok()
}

#[cfg(test)]
fn stable_release_asset_name_for(
    os: &str,
    arch: &str,
    flavor: backend::BinaryFlavor,
) -> Option<String> {
    ReleaseTarget::from_raw(os, arch, flavor)
        .ok()
        .and_then(ReleaseTarget::stable_asset_name)
}

fn push_release_asset_candidate(candidates: &mut Vec<String>, asset_name: Option<String>) {
    let Some(asset_name) = asset_name else {
        return;
    };
    if !candidates.iter().any(|candidate| candidate == &asset_name) {
        candidates.push(asset_name);
    }
}

pub(super) fn release_asset_candidates(
    target: ReleaseTarget,
    release_tag: &str,
    preference: ReleaseAssetPreference,
) -> Vec<String> {
    let mut candidates = Vec::new();
    match preference {
        ReleaseAssetPreference::StableFirst => {
            push_release_asset_candidate(&mut candidates, target.stable_asset_name());
            for name in target.stable_cuda_versioned_names() {
                push_release_asset_candidate(&mut candidates, Some(name));
            }
            push_release_asset_candidate(&mut candidates, target.versioned_asset_name(release_tag));
        }
        ReleaseAssetPreference::VersionedFirst => {
            push_release_asset_candidate(&mut candidates, target.versioned_asset_name(release_tag));
            for name in target.stable_cuda_versioned_names() {
                push_release_asset_candidate(&mut candidates, Some(name));
            }
            push_release_asset_candidate(&mut candidates, target.stable_asset_name());
        }
    }
    candidates
}

pub(super) fn resolve_release_asset_name(
    release: &ReleaseInfo,
    target: ReleaseTarget,
    preference: ReleaseAssetPreference,
) -> Option<String> {
    release_asset_candidates(target, &release.tag, preference)
        .into_iter()
        .find(|asset_name| {
            release
                .assets
                .iter()
                .any(|candidate| candidate.name == *asset_name)
        })
}

pub(super) fn release_has_any_platform_asset(release: &ReleaseInfo, os: &str, arch: &str) -> bool {
    backend::BinaryFlavor::ALL.into_iter().any(|flavor| {
        ReleaseTarget::from_raw(os, arch, flavor)
            .ok()
            .and_then(|target| {
                resolve_release_asset_name(release, target, ReleaseAssetPreference::StableFirst)
            })
            .is_some()
    })
}

pub(super) fn release_has_asset(release: &ReleaseInfo, asset_name: &str) -> bool {
    release.assets.iter().any(|asset| asset.name == asset_name)
}

pub(super) fn mesh_binary_name() -> String {
    backend::platform_bin_name("mesh-llm")
}

fn release_repo() -> String {
    match std::env::var(SELF_UPDATE_REPO_ENV) {
        Ok(repo) if repo.contains('/') && !repo.trim().is_empty() => repo,
        _ => DEFAULT_RELEASE_REPO.to_string(),
    }
}

fn latest_release_api_url() -> String {
    format!(
        "https://api.github.com/repos/{}/releases/latest",
        release_repo()
    )
}

fn release_api_url_for_tag(tag: &str) -> String {
    format!(
        "https://api.github.com/repos/{}/releases/tags/{tag}",
        release_repo()
    )
}

fn release_asset_url(tag: &str, asset_name: &str) -> String {
    format!(
        "https://github.com/{}/releases/download/{tag}/{asset_name}",
        release_repo()
    )
}

fn release_info_from_json(body: &serde_json::Value) -> Option<ReleaseInfo> {
    let tag = body["tag_name"].as_str()?.trim();
    let version = tag.trim_start_matches('v').trim();
    if tag.is_empty() || version.is_empty() {
        return None;
    }

    let assets = body["assets"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let name = item["name"].as_str()?.to_string();
                    let sha256 = github_release_asset_sha256(item)?;
                    Some(ReleaseAsset { name, sha256 })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Some(ReleaseInfo {
        tag: tag.to_string(),
        version: version.to_string(),
        assets,
    })
}

pub(super) async fn resolve_release_info(
    requested_version: Option<&str>,
) -> Result<Option<ReleaseInfo>> {
    let Some(requested_version) = requested_version else {
        return Ok(latest_release_info().await);
    };
    let tag = normalize_release_tag(requested_version)?;
    Ok(release_info_for_tag(&tag).await)
}

fn normalize_release_tag(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    anyhow::ensure!(!trimmed.is_empty(), "release version must not be empty");
    let version = trimmed.trim_start_matches('v');
    semver::Version::parse(version).with_context(|| format!("Invalid release version: {raw}"))?;
    Ok(format!("v{version}"))
}

pub(super) fn describe_requested_update(
    target_version: &str,
    current_version: &str,
    exact: bool,
) -> &'static str {
    if !exact {
        return "Updating";
    }

    match (
        semver::Version::parse(target_version),
        semver::Version::parse(current_version),
    ) {
        (Ok(target), Ok(current)) if target < current => "Downgrading",
        (Ok(target), Ok(current)) if target == current => "Reinstalling",
        _ => "Installing",
    }
}

pub(super) fn path_is_writable(path: &Path) -> bool {
    let probe = path.join(format!(
        "{PATH_WRITE_PROBE_PREFIX}-{}-{}",
        std::process::id(),
        PATH_WRITE_PROBE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));

    let Ok(file) = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    else {
        return false;
    };
    drop(file);

    std::fs::remove_file(&probe).is_ok()
}

static PATH_WRITE_PROBE_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) async fn install_latest_bundle(
    exe: &Path,
    install_dir: &Path,
    release: &ReleaseInfo,
    asset_name: &str,
    expected_flavor: backend::BinaryFlavor,
    action: PostInstallAction,
) -> Result<InstallOutcome> {
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let workspace = install_dir.join(format!(".mesh-llm-update-{unique}"));
    let extracted = workspace.join("bundle");
    let archive = workspace.join(asset_name);
    let backup = workspace.join("backup");

    std::fs::create_dir_all(&extracted)
        .with_context(|| format!("Failed to create update workspace {}", workspace.display()))?;

    let result = async {
        let asset = release_asset(release, asset_name)
            .with_context(|| format!("Release asset metadata missing for {asset_name}"))?;
        download_url(
            &release_asset_url(&release.tag, asset_name),
            &archive,
            &asset.sha256,
        )
        .await?;
        extract_bundle_archive(&archive, &extracted)?;
        let staged_files = collect_bundle_files(&extracted, expected_flavor)?;
        verify_staged_mesh_binary_version(&extracted, &release.version)?;
        finish_bundle_install(
            exe,
            install_dir,
            &workspace,
            &extracted,
            &backup,
            &staged_files,
            action,
        )?;
        install_native_runtime_after_update(install_dir, release, &workspace).await;
        Ok::<InstallOutcome, anyhow::Error>(install_outcome(action))
    }
    .await;

    if !matches!(result, Ok(InstallOutcome::HandoffAndExit)) {
        let _ = std::fs::remove_dir_all(&workspace);
    }
    result
}

#[cfg(not(windows))]
async fn install_native_runtime_after_update(
    install_dir: &Path,
    release: &ReleaseInfo,
    workspace: &Path,
) {
    let Some(asset) = release_asset(release, "native-runtimes.json") else {
        return;
    };
    let manifest_path = workspace.join("native-runtimes.json");
    match download_optional_url(
        &release_asset_url(&release.tag, "native-runtimes.json"),
        &manifest_path,
        &asset.sha256,
    )
    .await
    {
        Ok(true) => {}
        Ok(false) => return,
        Err(error) => {
            tracing::warn!(%error, "Failed to download native runtime release manifest");
            return;
        }
    }

    let binary = install_dir.join(mesh_binary_name());
    let install_status = std::process::Command::new(&binary)
        .arg("runtime")
        .arg("install")
        .arg("--manifest")
        .arg(&manifest_path)
        .status();
    match install_status {
        Ok(status) if status.success() => {
            let _ = std::process::Command::new(&binary)
                .arg("runtime")
                .arg("prune")
                .arg("--active-only")
                .status();
        }
        Ok(status) => {
            tracing::warn!(
                status = status.code(),
                "Native runtime install after update did not complete successfully"
            );
        }
        Err(error) => {
            tracing::warn!(%error, "Failed to run native runtime install after update");
        }
    }
}

#[cfg(windows)]
async fn install_native_runtime_after_update(
    _install_dir: &Path,
    _release: &ReleaseInfo,
    _workspace: &Path,
) {
}

// Only the non-Windows `install_native_runtime_after_update` calls this.
#[cfg(not(windows))]
async fn download_optional_url(url: &str, path: &Path, expected_sha256: &str) -> Result<bool> {
    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("Build optional release download HTTP client")?
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .with_context(|| format!("Download optional release asset {url}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(false);
    }
    let bytes = response
        .error_for_status()
        .with_context(|| format!("Optional release asset request failed for {url}"))?
        .bytes()
        .await
        .with_context(|| format!("Read optional release asset body from {url}"))?;
    verify_release_asset_bytes(&bytes, expected_sha256)
        .with_context(|| format!("Verify optional release asset {url}"))?;
    std::fs::write(path, &bytes)
        .with_context(|| format!("Write optional release asset {}", path.display()))?;
    Ok(true)
}

async fn download_url(url: &str, path: &Path, expected_sha256: &str) -> Result<()> {
    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("Build release download HTTP client")?
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .with_context(|| format!("Download release asset {url}"))?
        .error_for_status()
        .with_context(|| format!("Release asset request failed for {url}"))?;
    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("Read release asset body from {url}"))?;
    verify_release_asset_bytes(&bytes, expected_sha256)
        .with_context(|| format!("Verify release asset {url}"))?;
    std::fs::write(path, &bytes)
        .with_context(|| format!("Write release asset {}", path.display()))?;
    Ok(())
}

fn extract_bundle_archive(archive: &Path, extracted: &Path) -> Result<()> {
    match archive
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("zip") => extract_zip_archive(archive, extracted),
        _ => extract_tar_archive(archive, extracted),
    }
}

fn extract_tar_archive(archive: &Path, extracted: &Path) -> Result<()> {
    let status = std::process::Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(extracted)
        .arg("--strip-components=1")
        .status()
        .with_context(|| format!("Failed to extract {}", archive.display()))?;
    anyhow::ensure!(status.success(), "tar extraction failed");
    Ok(())
}

fn extract_zip_archive(archive: &Path, extracted: &Path) -> Result<()> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("Failed to open {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read ZIP archive {}", archive.display()))?;

    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        let enclosed = entry
            .enclosed_name()
            .context("ZIP archive contained an invalid path")?;
        let mut components = enclosed.components();
        let _top_level = components.next();
        let relative: PathBuf = components.collect();
        if relative.as_os_str().is_empty() {
            continue;
        }

        let output = extracted.join(&relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&output)
                .with_context(|| format!("Failed to create {}", output.display()))?;
            continue;
        }

        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let mut out = std::fs::File::create(&output)
            .with_context(|| format!("Failed to create {}", output.display()))?;
        io::copy(&mut entry, &mut out)
            .with_context(|| format!("Failed to extract {}", output.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod zip_tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{}_{}", prefix, nanos));
        // Best-effort cleanup in case something is left behind from a previous run.
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("failed to create temporary directory");
        dir
    }

    #[test]
    fn extract_zip_archive_strips_top_level_directory() -> Result<()> {
        let base_dir = unique_temp_dir("mesh_llm_extract_zip_test");
        let archive_path = base_dir.join("bundle.zip");
        let extracted_dir = base_dir.join("extracted");
        fs::create_dir_all(&extracted_dir)?;

        // Create a ZIP with a single top-level directory, similar to the release packager.
        let file = std::fs::File::create(&archive_path)
            .with_context(|| format!("Failed to create test archive {}", archive_path.display()))?;
        let mut writer = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

        // Top-level directory.
        writer.add_directory("bundle-1.0.0/", options)?;
        // Nested directory and file under the top-level directory.
        writer.add_directory("bundle-1.0.0/bin/", options)?;
        writer.start_file("bundle-1.0.0/bin/server", options)?;
        writer.write_all(b"dummy-server")?;

        writer
            .finish()
            .with_context(|| "Failed to finalize test ZIP archive")?;

        // Now extract and verify that the top-level directory is stripped.
        extract_zip_archive(&archive_path, &extracted_dir)?;

        let server_path = extracted_dir.join("bin").join("server");
        anyhow::ensure!(
            server_path.is_file(),
            "Expected extracted server file at {}",
            server_path.display()
        );

        let top_level = extracted_dir.join("bundle-1.0.0");
        anyhow::ensure!(
            !top_level.exists(),
            "Top-level directory should have been stripped, but {} exists",
            top_level.display()
        );

        Ok(())
    }
}
/// Files and directories a downloaded bundle installs into the install dir.
///
/// release bundles come in two layouts:
///
/// - legacy flat bundles (pre product-v2): only top-level files, including the
///   mesh-llm binary.
/// - composed product-v2 bundles (v0.75.0 and newer): the mesh-llm binary plus
///   product-manifest.json, host-imports.json, and a native-runtimes/ directory
///   holding the packaged runtime tree.
///
/// The returned paths are relative to the extracted bundle root, sorted so the
/// mesh binary installs first.
fn collect_bundle_files(
    extracted: &Path,
    expected_flavor: backend::BinaryFlavor,
) -> Result<Vec<String>> {
    let _ = expected_flavor;

    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(extracted)
        .with_context(|| format!("Failed to read {}", extracted.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name = name.to_string_lossy().to_string();
        if file_type.is_dir() {
            dirs.push(name.clone());
        }
        if file_type.is_file() {
            files.push(name);
        }
    }

    anyhow::ensure!(!files.is_empty(), "Downloaded bundle was empty");
    anyhow::ensure!(
        files.iter().any(|name| name == &mesh_binary_name()),
        "Downloaded bundle missing {}",
        mesh_binary_name()
    );
    if dirs.contains(&NATIVE_RUNTIMES_DIR_NAME.to_string()) {
        // Composed product-v2 bundle: the runtime tree ships inside the
        // bundle and installs recursively alongside the host.
        anyhow::ensure!(
            files.iter().any(|name| name == PRODUCT_MANIFEST_NAME),
            "Downloaded bundle has {} but no {PRODUCT_MANIFEST_NAME}",
            NATIVE_RUNTIMES_DIR_NAME
        );
    } else {
        // Legacy flat bundle: no directory is expected.
        anyhow::ensure!(
            dirs.is_empty(),
            "Unexpected directory in bundle: {}",
            dirs[0]
        );
    }

    let mut staged = files;
    if dirs.contains(&NATIVE_RUNTIMES_DIR_NAME.to_string()) {
        staged.push(NATIVE_RUNTIMES_DIR_NAME.to_string());
    }
    // The mesh binary must install first so a mid-install failure can never
    // leave the new runtime tree beside an old host.
    staged.sort_by_key(|name| (name != &mesh_binary_name(), name.clone()));
    Ok(staged)
}

fn verify_staged_mesh_binary_version(extracted: &Path, expected_version: &str) -> Result<()> {
    let binary = extracted.join(mesh_binary_name());
    let output = std::process::Command::new(&binary)
        .arg("--version")
        .output()
        .with_context(|| format!("Failed to run staged binary {}", binary.display()))?;

    anyhow::ensure!(
        output.status.success(),
        "Staged binary {} failed --version with status {}",
        binary.display(),
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let actual_version = stdout
        .split_whitespace()
        .last()
        .context("Staged binary --version output was empty")?;
    anyhow::ensure!(
        actual_version == expected_version,
        "Downloaded release v{} contains mesh-llm v{}; refusing to install mismatched bundle.",
        expected_version,
        actual_version
    );
    Ok(())
}

#[cfg(test)]
fn installed_runtime_tree(install_dir: &Path) -> PathBuf {
    install_dir.join(NATIVE_RUNTIMES_DIR_NAME).join("runtime")
}

/// Regression tests for the product-v2 self-update break: since v0.75.0 the
/// release archives carry a nested `native-runtimes/` directory that must
/// survive extraction (top-level stripped), pass `collect_bundle_files`, and
/// install recursively. The updater used to bail with "Unexpected directory
/// in bundle: native-runtimes", breaking every in-app update since v0.75.0.
#[cfg(all(unix, test))]
mod composed_product_regression_tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mesh-llm-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_extract_and_stage_composed_product_bundle_tar_gz() {
        use std::process::Command;

        let base = temp_dir("self-update-pv2-e2e-tar");
        let bundle_root = base.join("mesh-bundle");
        std::fs::create_dir_all(
            bundle_root
                .join("native-runtimes")
                .join("runtime")
                .join("lib"),
        )
        .unwrap();
        std::fs::write(bundle_root.join(mesh_binary_name()), b"binary").unwrap();
        std::fs::write(bundle_root.join("host-imports.json"), b"{}").unwrap();
        std::fs::write(bundle_root.join(PRODUCT_MANIFEST_NAME), b"{}").unwrap();
        std::fs::write(
            bundle_root
                .join("native-runtimes")
                .join("runtime")
                .join("lib")
                .join("libggml.dylib"),
            b"runtime",
        )
        .unwrap();

        let archive = base.join("bundle.tar.gz");
        let status = Command::new("tar")
            .arg("-C")
            .arg(&base)
            .arg("-czf")
            .arg(&archive)
            .arg("mesh-bundle")
            .status()
            .unwrap();
        assert!(status.success(), "test tar packaging failed");

        let extracted = base.join("extracted");
        std::fs::create_dir_all(&extracted).unwrap();
        extract_bundle_archive(&archive, &extracted).unwrap();

        let staged = collect_bundle_files(&extracted, backend::BinaryFlavor::Cpu).unwrap();
        assert!(staged.contains(&NATIVE_RUNTIMES_DIR_NAME.to_string()));

        let install_dir = base.join("install");
        let backup = base.join("backup");
        std::fs::create_dir_all(&install_dir).unwrap();
        replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap();

        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"binary"
        );
        assert_eq!(
            std::fs::read(
                install_dir
                    .join(NATIVE_RUNTIMES_DIR_NAME)
                    .join("runtime")
                    .join("lib")
                    .join("libggml.dylib")
            )
            .unwrap(),
            b"runtime"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    /// Same composed-product scenario for the Windows zip lane: the zip
    /// extractor strips the top-level directory and must not resurrect it as
    /// an unexpected nested directory that `collect_bundle_files` rejects.
    #[test]
    fn test_extract_and_stage_composed_product_bundle_zip() {
        use std::io::Write as _;
        use zip::CompressionMethod;
        use zip::write::SimpleFileOptions;

        let base = temp_dir("self-update-pv2-e2e-zip");
        let archive = base.join("bundle.zip");
        let file = std::fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

        writer.add_directory("mesh-bundle/", options).unwrap();
        writer
            .add_directory("mesh-bundle/native-runtimes/", options)
            .unwrap();
        writer
            .add_directory("mesh-bundle/native-runtimes/runtime/", options)
            .unwrap();
        writer.start_file("mesh-bundle/mesh-llm", options).unwrap();
        writer.write_all(b"binary").unwrap();
        writer
            .start_file("mesh-bundle/product-manifest.json", options)
            .unwrap();
        writer.write_all(b"{}").unwrap();
        writer
            .start_file("mesh-bundle/host-imports.json", options)
            .unwrap();
        writer.write_all(b"{}").unwrap();
        writer
            .start_file("mesh-bundle/native-runtimes/runtime/libggml.dylib", options)
            .unwrap();
        writer.write_all(b"runtime").unwrap();
        writer.finish().unwrap();

        let extracted = base.join("extracted");
        std::fs::create_dir_all(&extracted).unwrap();
        extract_bundle_archive(&archive, &extracted).unwrap();

        let staged = collect_bundle_files(&extracted, backend::BinaryFlavor::Cpu).unwrap();
        assert!(staged.contains(&NATIVE_RUNTIMES_DIR_NAME.to_string()));

        let install_dir = base.join("install");
        let backup = base.join("backup");
        std::fs::create_dir_all(&install_dir).unwrap();
        replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap();

        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"binary"
        );
        assert_eq!(
            std::fs::read(
                install_dir
                    .join(NATIVE_RUNTIMES_DIR_NAME)
                    .join("runtime")
                    .join("libggml.dylib")
            )
            .unwrap(),
            b"runtime"
        );
        let _ = std::fs::remove_dir_all(base);
    }
}

#[cfg(not(windows))]
fn backup_existing_path(
    install_dir: &Path,
    backup: &Path,
    relative: &str,
    backed_up: &mut Vec<String>,
) -> Result<()> {
    if backed_up.iter().any(|existing| existing == relative) {
        return Ok(());
    }

    let dest = install_dir.join(relative);
    if !dest.exists() {
        return Ok(());
    }

    let backup_path = backup.join(relative);
    if let Some(parent) = backup_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create backup directory {}", parent.display()))?;
    }
    std::fs::rename(&dest, &backup_path).with_context(|| {
        format!(
            "Failed to move {} to {}",
            dest.display(),
            backup_path.display()
        )
    })?;
    backed_up.push(relative.to_string());
    Ok(())
}

#[cfg(not(windows))]
fn replace_bundle_files(
    install_dir: &Path,
    extracted: &Path,
    backup: &Path,
    staged_files: &[String],
) -> Result<()> {
    use std::collections::BTreeSet;

    std::fs::create_dir_all(backup)
        .with_context(|| format!("Failed to create backup dir {}", backup.display()))?;

    let managed_names: BTreeSet<String> = BTreeSet::from([mesh_binary_name()]);

    let mut backed_up = Vec::new();
    let mut installed = Vec::new();

    let result = (|| {
        for name in managed_names {
            backup_existing_path(install_dir, backup, &name, &mut backed_up)?;
        }
        for relative in staged_files {
            backup_existing_path(install_dir, backup, relative, &mut backed_up)?;
        }

        for relative in staged_files {
            let source = extracted.join(relative);
            let dest = install_dir.join(relative);
            if let Err(err) = std::fs::rename(&source, &dest) {
                return Err(err).with_context(|| {
                    format!(
                        "Failed to install {} into {}",
                        source.display(),
                        dest.display()
                    )
                });
            }
            installed.push(relative.clone());
        }

        Ok(())
    })();

    if result.is_err() {
        rollback_bundle_replace(install_dir, backup, &installed, &backed_up);
    }

    result
}

#[cfg(not(windows))]
fn install_outcome(action: PostInstallAction) -> InstallOutcome {
    match action {
        PostInstallAction::RestartCurrentProcess => InstallOutcome::RestartNow,
        PostInstallAction::ExitAfterInstall => InstallOutcome::ExitNow,
    }
}

#[cfg(windows)]
fn install_outcome(_action: PostInstallAction) -> InstallOutcome {
    InstallOutcome::HandoffAndExit
}

#[cfg(not(windows))]
fn finish_bundle_install(
    _exe: &Path,
    install_dir: &Path,
    _workspace: &Path,
    extracted: &Path,
    backup: &Path,
    staged_files: &[String],
    _action: PostInstallAction,
) -> Result<()> {
    replace_bundle_files(install_dir, extracted, backup, staged_files)
}

#[cfg(windows)]
fn finish_bundle_install(
    exe: &Path,
    install_dir: &Path,
    workspace: &Path,
    extracted: &Path,
    backup: &Path,
    staged_files: &[String],
    action: PostInstallAction,
) -> Result<()> {
    use std::process::Command;

    let script = workspace.join("apply-update.ps1");
    let script_body = windows_update_script(
        exe,
        install_dir,
        workspace,
        extracted,
        backup,
        staged_files,
        action,
    )?;
    std::fs::write(&script, script_body)
        .with_context(|| format!("Failed to write {}", script.display()))?;

    Command::new("powershell")
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(&script)
        .spawn()
        .with_context(|| format!("Failed to launch Windows updater {}", script.display()))?;

    Ok(())
}

#[cfg(windows)]
fn windows_update_script(
    exe: &Path,
    install_dir: &Path,
    workspace: &Path,
    extracted: &Path,
    backup: &Path,
    staged_files: &[String],
    action: PostInstallAction,
) -> Result<String> {
    use std::collections::BTreeSet;

    let staged_json = serde_json::to_string(staged_files)?;
    let restart_after_update = matches!(action, PostInstallAction::RestartCurrentProcess);
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|arg| arg.to_string_lossy().to_string())
        .collect();
    let args_json = serde_json::to_string(&args)?;

    let managed_names: BTreeSet<String> = BTreeSet::from([mesh_binary_name()]);
    let managed_json = serde_json::to_string(&managed_names.into_iter().collect::<Vec<_>>())?;

    let quote = |path: &Path| path.to_string_lossy().replace('\'', "''");

    let args_json_ps = args_json.replace('\'', "''");
    let managed_json_ps = managed_json.replace('\'', "''");
    let staged_json_ps = staged_json.replace('\'', "''");

    Ok(format!(
        r#"$ErrorActionPreference = 'Stop'
$installDir = '{install_dir}'
$workspace = '{workspace}'
$stagingDir = '{staging_dir}'
$backupDir = '{backup_dir}'
$exePath = '{exe_path}'
$waitPid = {wait_pid}
$restartAfterUpdate = {restart_after_update}
$managedNames = @((ConvertFrom-Json '{managed_json_ps}'))
$stagedNames = @((ConvertFrom-Json '{staged_json_ps}'))
$args = @((ConvertFrom-Json '{args_json_ps}'))

function Ensure-Parent([string]$Path) {{
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path $parent)) {{
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }}
}}

function Move-ToBackup([string]$Name, [System.Collections.Generic.List[string]]$BackedUp) {{
    $dest = Join-Path $installDir $Name
    if (-not (Test-Path $dest)) {{
        return
    }}
    $backupPath = Join-Path $backupDir $Name
    Ensure-Parent $backupPath
    Move-Item -Force $dest $backupPath
    $BackedUp.Add($Name) | Out-Null
}}

function Restore-Backups([string[]]$BackedUpNames, [string[]]$InstalledNames) {{
    foreach ($name in $InstalledNames) {{
        $dest = Join-Path $installDir $name
        if (Test-Path $dest -PathType Container) {{
            Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
        }} else {{
            Remove-Item $dest -Force -ErrorAction SilentlyContinue
        }}
    }}
    foreach ($name in $BackedUpNames) {{
        $backupPath = Join-Path $backupDir $name
        $dest = Join-Path $installDir $name
        if (Test-Path $backupPath) {{
            Ensure-Parent $dest
            Move-Item -Force $backupPath $dest
        }}
    }}
}}

while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {{
    Start-Sleep -Milliseconds 200
}}

$backedUp = New-Object System.Collections.Generic.List[string]
$installed = New-Object System.Collections.Generic.List[string]

try {{
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

    foreach ($name in $managedNames) {{
        if ($stagedNames -contains $name) {{
            continue
        }}
        Move-ToBackup $name $backedUp
    }}

    foreach ($name in $stagedNames) {{
        Move-ToBackup $name $backedUp
    }}

    foreach ($name in $stagedNames) {{
        $source = Join-Path $stagingDir $name
        $dest = Join-Path $installDir $name
        Ensure-Parent $dest
        Move-Item -Force $source $dest
        $installed.Add($name) | Out-Null
    }}

    if ($restartAfterUpdate) {{
        $env:MESH_LLM_SELF_UPDATE_ATTEMPTED = '1'
        & $exePath @args
        exit $LASTEXITCODE
    }}
    exit 0
}} catch {{
    Restore-Backups $backedUp.ToArray() $installed.ToArray()
    throw
}} finally {{
    Remove-Item $workspace -Recurse -Force -ErrorAction SilentlyContinue
}}
"#,
        install_dir = quote(install_dir),
        workspace = quote(workspace),
        staging_dir = quote(extracted),
        backup_dir = quote(backup),
        exe_path = quote(exe),
        wait_pid = std::process::id(),
        restart_after_update = if restart_after_update {
            "$true"
        } else {
            "$false"
        },
        managed_json_ps = managed_json_ps,
        staged_json_ps = staged_json_ps,
        args_json_ps = args_json_ps
    ))
}

#[cfg(not(windows))]
fn rollback_bundle_replace(
    install_dir: &Path,
    backup: &Path,
    installed: &[String],
    backed_up: &[String],
) {
    // Staged entries can be whole trees (`native-runtimes`). `remove_file`
    // no-ops on a directory, which would leave the new tree in place and make
    // the restore rename below fail.
    for relative in installed.iter().rev() {
        let dest = install_dir.join(relative);
        match std::fs::symlink_metadata(&dest) {
            Ok(metadata) if metadata.is_dir() => {
                let _ = std::fs::remove_dir_all(&dest);
            }
            Ok(_) => {
                let _ = std::fs::remove_file(&dest);
            }
            Err(_) => {}
        }
    }
    for relative in backed_up.iter().rev() {
        let backup_path = backup.join(relative);
        let dest = install_dir.join(relative);
        let _ = std::fs::rename(&backup_path, &dest);
    }
}

#[cfg(unix)]
pub(super) fn exec_current_binary(exe: &Path, env_key: &str, env_value: &str) -> Result<()> {
    let mut args = std::env::args_os();
    let arg0 = args.next().unwrap_or_else(|| exe.as_os_str().to_owned());
    let error = std::process::Command::new(exe)
        .arg0(arg0)
        .args(args)
        .env(env_key, env_value)
        .exec();
    Err(error).context("Failed to restart updated mesh-llm")
}

#[cfg(not(unix))]
pub(super) fn exec_current_binary(_exe: &Path, _env_key: &str, _env_value: &str) -> Result<()> {
    anyhow::bail!("Self-update restart is only supported on Unix")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mesh-llm-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_release_asset(name: impl Into<String>) -> ReleaseAsset {
        ReleaseAsset {
            name: name.into(),
            sha256: "a".repeat(64),
        }
    }

    #[test]
    fn test_version_newer() {
        assert!(version_newer("0.33.1", "0.33.0"));
        assert!(!version_newer("0.33.0", "0.33.0"));
        assert!(!version_newer("0.32.0", "0.33.0"));
        assert!(version_newer("0.33.0", "0.33.0-rc.1"));
        assert!(!version_newer("0.33.0-rc.1", "0.33.0"));
        assert!(version_newer("0.33.0-rc.2", "0.33.0-rc.1"));
        assert!(!version_newer("0.99.0", "0.68.0+gAB131C"));
        assert!(version_newer("0.68.0+gAB131C", "0.99.0"));
        assert!(!version_newer("0.99.0", "0.68.0+gAB131C.dirty"));
        assert!(version_newer("0.68.0+gAB131C.dirty", "0.99.0"));
        assert!(version_newer("0.69.0", "0.68.0"));
        assert!(!version_newer("not-a-version", "0.68.0"));
        assert!(!version_newer("0.68.0", "not-a-version"));
        assert!(!version_newer("not-a-version+gAB131C", "0.99.0"));
        assert!(!version_newer("not-a-version+gAB131C.dirty", "0.99.0"));
        assert!(!version_newer("0.99.0", "not-a-version+gAB131C"));
    }

    #[test]
    #[serial]
    fn test_release_asset_url() {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var(SELF_UPDATE_REPO_ENV) };
        assert_eq!(
            release_asset_url("v0.60.0", "mesh-llm-aarch64-apple-darwin.tar.gz"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.60.0/mesh-llm-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    #[serial]
    fn test_release_repo_defaults_to_main_repo() {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var(SELF_UPDATE_REPO_ENV) };
        assert_eq!(release_repo(), "Mesh-LLM/mesh-llm");
        assert_eq!(
            latest_release_api_url(),
            "https://api.github.com/repos/Mesh-LLM/mesh-llm/releases/latest"
        );
    }

    #[test]
    #[serial]
    fn test_release_repo_can_be_overridden_for_testing() {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var(SELF_UPDATE_REPO_ENV, "jdumay/mesh-llm") };
        assert_eq!(release_repo(), "jdumay/mesh-llm");
        assert_eq!(
            latest_release_api_url(),
            "https://api.github.com/repos/jdumay/mesh-llm/releases/latest"
        );
        assert_eq!(
            release_api_url_for_tag("v0.60.0"),
            "https://api.github.com/repos/jdumay/mesh-llm/releases/tags/v0.60.0"
        );
        assert_eq!(
            release_asset_url("v0.60.0", "mesh-llm-x86_64-unknown-linux-gnu.tar.gz"),
            "https://github.com/jdumay/mesh-llm/releases/download/v0.60.0/mesh-llm-x86_64-unknown-linux-gnu.tar.gz"
        );
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var(SELF_UPDATE_REPO_ENV) };
    }

    #[test]
    fn test_normalize_release_tag() {
        assert_eq!(normalize_release_tag("v0.60.0").unwrap(), "v0.60.0");
        assert_eq!(
            normalize_release_tag("0.60.0-rc.1").unwrap(),
            "v0.60.0-rc.1"
        );
        assert!(normalize_release_tag("latest").is_err());
    }

    #[test]
    fn test_describe_requested_update() {
        assert_eq!(
            describe_requested_update("0.60.0", "0.68.0", false),
            "Updating"
        );
        assert_eq!(
            describe_requested_update("0.68.0", "0.68.0", true),
            "Reinstalling"
        );
        assert_eq!(
            describe_requested_update("0.0.1", "0.68.0", true),
            "Downgrading"
        );
        assert_eq!(
            describe_requested_update("999.0.0", "0.68.0", true),
            "Installing"
        );
    }

    #[test]
    fn test_stable_release_asset_name_matches_platform() {
        let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Some((
                backend::BinaryFlavor::Metal,
                "mesh-llm-aarch64-apple-darwin.tar.gz",
            )),
            ("linux", "x86_64") => Some((
                backend::BinaryFlavor::Cpu,
                "mesh-llm-x86_64-unknown-linux-gnu.tar.gz",
            )),
            _ => None,
        };

        let Some((flavor, asset)) = expected else {
            return;
        };
        assert_eq!(
            stable_release_asset_name_for(std::env::consts::OS, std::env::consts::ARCH, flavor),
            Some(asset.to_string())
        );
    }

    #[test]
    fn test_windows_release_asset_names() {
        assert!(platform_has_release_assets_for("windows", "x86_64"));
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Cpu),
            Some("mesh-llm-x86_64-pc-windows-msvc.zip".to_string())
        );
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Cuda),
            Some("mesh-llm-x86_64-pc-windows-msvc-cuda.zip".to_string())
        );
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Rocm),
            Some("mesh-llm-x86_64-pc-windows-msvc-rocm.zip".to_string())
        );
        assert_eq!(
            stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Vulkan),
            Some("mesh-llm-x86_64-pc-windows-msvc-vulkan.zip".to_string())
        );
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![test_release_asset(
                "mesh-llm-v0.60.0-x86_64-pc-windows-msvc.zip",
            )],
        };
        assert!(release_has_any_platform_asset(
            &release, "windows", "x86_64"
        ));
        let empty_release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: Vec::new(),
        };
        assert!(!release_has_any_platform_asset(
            &empty_release,
            "windows",
            "x86_64"
        ));
    }

    #[test]
    fn test_linux_arm64_release_asset_names() {
        let stable_asset = "mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string();
        assert!(platform_has_release_assets_for("linux", "aarch64"));
        assert_eq!(
            stable_release_asset_name_for("linux", "aarch64", backend::BinaryFlavor::Cpu),
            Some(stable_asset.clone())
        );

        let published_release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![test_release_asset(stable_asset)],
        };
        assert!(release_has_any_platform_asset(
            &published_release,
            "linux",
            "aarch64"
        ));

        let missing_release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: Vec::new(),
        };
        assert!(!release_has_any_platform_asset(
            &missing_release,
            "linux",
            "aarch64"
        ));
    }

    #[test]
    fn test_linux_arm64_aliases_resolve_identical_release_assets() {
        let arm64_asset =
            stable_release_asset_name_for("linux", "arm64", backend::BinaryFlavor::Cpu);
        let aarch64_asset =
            stable_release_asset_name_for("linux", "aarch64", backend::BinaryFlavor::Cpu);
        assert_eq!(arm64_asset, aarch64_asset);
        assert_eq!(
            arm64_asset,
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_prefers_stable_linux_arm64_asset() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![
                test_release_asset("mesh-llm-aarch64-unknown-linux-gnu.tar.gz"),
                test_release_asset("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz"),
            ],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "arm64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::StableFirst,
            ),
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_falls_back_to_versioned_linux_arm64_asset() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![test_release_asset(
                "mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz",
            )],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "aarch64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::StableFirst,
            ),
            Some("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_prefers_versioned_for_explicit_install() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![
                test_release_asset("mesh-llm-aarch64-unknown-linux-gnu.tar.gz"),
                test_release_asset("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz"),
            ],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "aarch64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::VersionedFirst,
            ),
            Some("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_resolve_release_asset_name_versioned_first_falls_back_to_stable() {
        let release = ReleaseInfo {
            tag: "v0.60.0".to_string(),
            version: "0.60.0".to_string(),
            assets: vec![test_release_asset(
                "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
            )],
        };

        assert_eq!(
            resolve_release_asset_name(
                &release,
                ReleaseTarget::from_raw("linux", "arm64", backend::BinaryFlavor::Cpu).unwrap(),
                ReleaseAssetPreference::VersionedFirst,
            ),
            Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
        );
    }

    #[test]
    fn test_path_is_writable_for_temp_dir() {
        let dir = temp_dir("self-update-writable");
        assert!(path_is_writable(&dir));
        assert!(std::fs::read_dir(&dir).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(PATH_WRITE_PROBE_PREFIX)
        }));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_path_is_writable_rejects_non_directory_path() {
        let dir = temp_dir("self-update-writable-file");
        let path = dir.join("mesh-llm");
        std::fs::write(&path, b"binary").unwrap();

        assert!(!path_is_writable(&path));
        assert_eq!(std::fs::read(&path).unwrap(), b"binary");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(not(windows))]
    #[test]
    fn test_replace_bundle_files_rolls_back_backup_failure() {
        let dir = temp_dir("self-update-backup-rollback");
        let install_dir = dir.join("install");
        let extracted = dir.join("extracted");
        let backup = dir.join("backup");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::create_dir_all(&extracted).unwrap();
        std::fs::create_dir_all(backup.join("sidecar")).unwrap();
        std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
        std::fs::write(install_dir.join("sidecar"), b"old-sidecar").unwrap();

        let err = replace_bundle_files(&install_dir, &extracted, &backup, &["sidecar".to_string()])
            .unwrap_err();

        assert!(err.to_string().contains("Failed to move"));
        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"old-binary"
        );
        assert_eq!(
            std::fs::read(install_dir.join("sidecar")).unwrap(),
            b"old-sidecar"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(not(windows))]
    #[test]
    fn test_replace_bundle_files_rolls_back_install_failure() {
        let dir = temp_dir("self-update-install-rollback");
        let install_dir = dir.join("install");
        let extracted = dir.join("extracted");
        let backup = dir.join("backup");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::create_dir_all(&extracted).unwrap();
        std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
        std::fs::write(install_dir.join("sidecar"), b"old-sidecar").unwrap();
        std::fs::write(extracted.join("sidecar"), b"new-sidecar").unwrap();

        let staged_files = ["sidecar".to_string(), mesh_binary_name()];
        let err =
            replace_bundle_files(&install_dir, &extracted, &backup, &staged_files).unwrap_err();

        assert!(err.to_string().contains("Failed to install"));
        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"old-binary"
        );
        assert_eq!(
            std::fs::read(install_dir.join("sidecar")).unwrap(),
            b"old-sidecar"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(not(windows))]
    #[test]
    fn test_replace_bundle_files_installs_and_backs_up_runtime_tree() {
        let dir = temp_dir("self-update-runtime-tree");
        let install_dir = dir.join("install");
        let extracted = dir.join("extracted");
        let backup = dir.join("backup");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::create_dir_all(installed_runtime_tree(&install_dir).join("lib")).unwrap();
        std::fs::create_dir_all(
            extracted
                .join("native-runtimes")
                .join("runtime")
                .join("lib"),
        )
        .unwrap();
        std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
        std::fs::write(
            installed_runtime_tree(&install_dir)
                .join("lib")
                .join("core.dylib"),
            b"old-runtime",
        )
        .unwrap();
        std::fs::write(install_dir.join(PRODUCT_MANIFEST_NAME), b"old-manifest").unwrap();
        std::fs::write(extracted.join(mesh_binary_name()), b"new-binary").unwrap();
        std::fs::write(extracted.join(PRODUCT_MANIFEST_NAME), b"new-manifest").unwrap();
        std::fs::write(
            extracted
                .join("native-runtimes")
                .join("runtime")
                .join("lib")
                .join("core.dylib"),
            b"new-runtime",
        )
        .unwrap();

        let staged = vec![
            PRODUCT_MANIFEST_NAME.to_string(),
            NATIVE_RUNTIMES_DIR_NAME.to_string(),
            mesh_binary_name(),
        ];
        replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap();

        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"new-binary"
        );
        assert_eq!(
            std::fs::read(install_dir.join(PRODUCT_MANIFEST_NAME)).unwrap(),
            b"new-manifest"
        );
        assert_eq!(
            std::fs::read(
                installed_runtime_tree(&install_dir)
                    .join("lib")
                    .join("core.dylib")
            )
            .unwrap(),
            b"new-runtime"
        );
        // Superseded files must be restorable from the backup.
        assert_eq!(
            std::fs::read(
                backup
                    .join("native-runtimes")
                    .join("runtime")
                    .join("lib")
                    .join("core.dylib")
            )
            .unwrap(),
            b"old-runtime"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(not(windows))]
    #[test]
    fn test_replace_bundle_files_rolls_back_runtime_tree() {
        let dir = temp_dir("self-update-runtime-rollback");
        let install_dir = dir.join("install");
        let extracted = dir.join("extracted");
        let backup = dir.join("backup");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::create_dir_all(extracted.join("native-runtimes").join("runtime")).unwrap();
        std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
        std::fs::write(extracted.join(mesh_binary_name()), b"new-binary").unwrap();
        std::fs::write(
            extracted
                .join("native-runtimes")
                .join("runtime")
                .join("core.dylib"),
            b"new-runtime",
        )
        .unwrap();

        let staged = vec![
            NATIVE_RUNTIMES_DIR_NAME.to_string(),
            mesh_binary_name(),
            "missing.bin".to_string(),
        ];
        let err = replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap_err();

        assert!(err.to_string().contains("Failed to install"));
        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"old-binary"
        );
        assert!(
            !install_dir.join(NATIVE_RUNTIMES_DIR_NAME).exists(),
            "runtime tree installed by a failed update must be rolled back"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(not(windows))]
    #[test]
    fn test_replace_bundle_files_restores_previous_runtime_tree() {
        let dir = temp_dir("self-update-runtime-restore");
        let install_dir = dir.join("install");
        let extracted = dir.join("extracted");
        let backup = dir.join("backup");
        std::fs::create_dir_all(installed_runtime_tree(&install_dir).join("lib")).unwrap();
        std::fs::create_dir_all(extracted.join("native-runtimes").join("runtime")).unwrap();
        std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
        std::fs::write(
            installed_runtime_tree(&install_dir)
                .join("lib")
                .join("core.dylib"),
            b"old-runtime",
        )
        .unwrap();
        std::fs::write(extracted.join(mesh_binary_name()), b"new-binary").unwrap();
        std::fs::write(
            extracted
                .join("native-runtimes")
                .join("runtime")
                .join("core.dylib"),
            b"new-runtime",
        )
        .unwrap();

        let staged = vec![
            NATIVE_RUNTIMES_DIR_NAME.to_string(),
            mesh_binary_name(),
            "missing.bin".to_string(),
        ];
        let err = replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap_err();

        assert!(err.to_string().contains("Failed to install"));
        assert_eq!(
            std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
            b"old-binary"
        );
        assert_eq!(
            std::fs::read(
                installed_runtime_tree(&install_dir)
                    .join("lib")
                    .join("core.dylib")
            )
            .unwrap(),
            b"old-runtime"
        );
        assert!(
            !installed_runtime_tree(&install_dir)
                .join("core.dylib")
                .exists(),
            "the failed update's runtime files must not survive rollback"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_collect_bundle_files_accepts_product_v2_layout() {
        let dir = temp_dir("self-update-pv2-collect");
        std::fs::create_dir_all(dir.join("native-runtimes").join("runtime").join("lib")).unwrap();
        std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();
        std::fs::write(dir.join("host-imports.json"), b"{}").unwrap();
        std::fs::write(dir.join(PRODUCT_MANIFEST_NAME), b"{}").unwrap();
        std::fs::write(
            dir.join("native-runtimes")
                .join("runtime")
                .join("core.dylib"),
            b"runtime",
        )
        .unwrap();

        let staged = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap();
        assert_eq!(
            staged,
            vec![
                mesh_binary_name(),
                "host-imports.json".to_string(),
                NATIVE_RUNTIMES_DIR_NAME.to_string(),
                PRODUCT_MANIFEST_NAME.to_string(),
            ]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_collect_bundle_files_rejects_product_v2_without_manifest() {
        let dir = temp_dir("self-update-pv2-manifest");
        std::fs::create_dir_all(dir.join("native-runtimes").join("runtime")).unwrap();
        std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();

        let err = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap_err();
        assert!(err.to_string().contains(PRODUCT_MANIFEST_NAME));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_collect_bundle_files_rejects_unexpected_directory() {
        let dir = temp_dir("self-update-unexpected-dir");
        std::fs::create_dir_all(dir.join("surprise")).unwrap();
        std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();

        let err = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap_err();
        assert!(err.to_string().contains("Unexpected directory in bundle"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_collect_bundle_files_accepts_legacy_flat_bundle() {
        let dir = temp_dir("self-update-legacy-collect");
        std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();
        std::fs::write(dir.join("sidecar.txt"), b"sidecar").unwrap();

        let staged = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap();
        assert_eq!(staged, vec![mesh_binary_name(), "sidecar.txt".to_string()]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn test_staged_binary_version_must_match_release() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_dir("self-update-version-check");
        let binary = dir.join(mesh_binary_name());
        std::fs::write(&binary, "#!/bin/sh\necho 'mesh-llm 0.68.0'\n").unwrap();
        let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&binary, permissions).unwrap();

        verify_staged_mesh_binary_version(&dir, "0.68.0").unwrap();
        let err = verify_staged_mesh_binary_version(&dir, "0.69.0").unwrap_err();
        assert!(
            err.to_string()
                .contains("contains mesh-llm v0.68.0; refusing to install")
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
