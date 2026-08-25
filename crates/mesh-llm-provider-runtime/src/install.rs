use crate::{
    InstalledProviderRuntime, ProviderRuntimeArtifact, ProviderRuntimeCache,
    ProviderRuntimeInstallStatus, ProviderRuntimeManifest, ProviderRuntimeReleaseManifest,
    ProviderRuntimeRequest, ProviderRuntimeResolution, ProviderRuntimeResolver,
    ProviderRuntimeSource,
};
use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::io::AsyncWriteExt;

const MAX_ARCHIVE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 4096;
const MAX_EXPANDED_BYTES: u64 = 2 * MAX_ARCHIVE_BYTES;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRuntimeBundlePolicy {
    #[default]
    UseInPlace,
    InstallIntoCache,
}

#[derive(Clone, Debug)]
pub struct ProviderRuntimeInstallOptions {
    pub host: crate::ProviderRuntimeHost,
    pub request: ProviderRuntimeRequest,
    pub release_manifest: ProviderRuntimeReleaseManifest,
    pub bundle_dirs: Vec<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub bundle_policy: ProviderRuntimeBundlePolicy,
    pub allow_download: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeInstallOutcome {
    pub status: ProviderRuntimeInstallStatus,
    pub runtime: InstalledProviderRuntime,
    pub resolution: ProviderRuntimeResolution,
}

impl Default for ProviderRuntimeInstallOptions {
    fn default() -> Self {
        Self {
            host: crate::ProviderRuntimeHost::current(),
            request: ProviderRuntimeRequest::default(),
            release_manifest: ProviderRuntimeReleaseManifest {
                schema_version: crate::PROVIDER_RUNTIME_SCHEMA_VERSION,
                artifacts: Vec::new(),
            },
            bundle_dirs: Vec::new(),
            cache_dir: None,
            bundle_policy: ProviderRuntimeBundlePolicy::UseInPlace,
            allow_download: false,
        }
    }
}

pub async fn install_provider_runtime(
    options: ProviderRuntimeInstallOptions,
) -> Result<ProviderRuntimeInstallOutcome> {
    options.release_manifest.validate()?;
    let cache = provider_runtime_cache(options.cache_dir.as_deref())?;
    let resolution = ProviderRuntimeResolver::new(
        options.host.clone(),
        options.release_manifest.clone(),
        cache.clone(),
    )
    .with_bundle_dirs(options.bundle_dirs.clone())
    .resolve(&options.request)?;
    let (status, runtime) = match &resolution.source {
        ProviderRuntimeSource::Bundle { path }
            if options.bundle_policy == ProviderRuntimeBundlePolicy::UseInPlace =>
        {
            in_place_runtime(path)?
        }
        ProviderRuntimeSource::Bundle { path } => cache.install_from_dir(path)?,
        ProviderRuntimeSource::Installed { path } => (
            ProviderRuntimeInstallStatus::AlreadyInstalled,
            installed_runtime(path)?,
        ),
        ProviderRuntimeSource::Download { url, sha256 } if options.allow_download => {
            download_and_install(&cache, &resolution.selected, url, sha256).await?
        }
        ProviderRuntimeSource::Download { .. } => {
            bail!("selected provider runtime requires a download, but downloads are disabled")
        }
        ProviderRuntimeSource::Missing => {
            bail!(
                "selected provider runtime {} is not bundled, installed, or downloadable",
                resolution.selected.id
            )
        }
    };
    Ok(ProviderRuntimeInstallOutcome {
        status,
        runtime,
        resolution,
    })
}

fn provider_runtime_cache(cache_dir: Option<&Path>) -> Result<ProviderRuntimeCache> {
    let root = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
            .context("cannot determine provider runtime cache directory")?
            .join("mesh-llm")
            .join("provider-runtimes"),
    };
    Ok(ProviderRuntimeCache::new(root))
}

fn in_place_runtime(
    path: &Path,
) -> Result<(ProviderRuntimeInstallStatus, InstalledProviderRuntime)> {
    Ok((
        ProviderRuntimeInstallStatus::AlreadyInstalled,
        installed_runtime(path)?,
    ))
}

fn installed_runtime(path: &Path) -> Result<InstalledProviderRuntime> {
    let manifest = ProviderRuntimeManifest::read_from_dir(path)?;
    Ok(InstalledProviderRuntime {
        path: path.to_path_buf(),
        manifest,
    })
}

async fn download_and_install(
    cache: &ProviderRuntimeCache,
    selected: &ProviderRuntimeArtifact,
    url: &str,
    expected_sha256: &str,
) -> Result<(ProviderRuntimeInstallStatus, InstalledProviderRuntime)> {
    let workspace = tempfile::Builder::new()
        .prefix("mesh-provider-runtime-")
        .tempdir()
        .context("create provider runtime download workspace")?;
    let archive_path = workspace.path().join("provider-runtime.zip");
    download_archive(url, &archive_path, expected_sha256).await?;
    install_provider_runtime_archive(cache, selected, &archive_path, expected_sha256)
}

pub fn install_provider_runtime_archive(
    cache: &ProviderRuntimeCache,
    selected: &ProviderRuntimeArtifact,
    archive_path: &Path,
    expected_sha256: &str,
) -> Result<(ProviderRuntimeInstallStatus, InstalledProviderRuntime)> {
    verify_archive_file(archive_path, expected_sha256)?;
    let workspace = tempfile::Builder::new()
        .prefix("mesh-provider-runtime-extract-")
        .tempdir()
        .context("create provider runtime extraction workspace")?;
    let extracted = workspace.path().join("extracted");
    fs::create_dir(&extracted)?;
    extract_archive(archive_path, &extracted)?;
    let bundle = find_provider_bundle(&extracted)?;
    let manifest = ProviderRuntimeManifest::read_from_dir(&bundle)?;
    if !manifest.runtime.payload_matches(selected) {
        bail!(
            "downloaded provider runtime payload does not match selected artifact {} {}",
            selected.id,
            selected.version
        );
    }
    cache.install_from_dir(&bundle)
}

async fn download_archive(url: &str, path: &Path, expected_sha256: &str) -> Result<()> {
    let diagnostic_url = url.split_once('?').map_or(url, |(base, _)| base);
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("build provider runtime download client")?
        .get(url)
        .header("User-Agent", "mesh-llm")
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("download provider runtime {diagnostic_url}"))?
        .error_for_status()
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("provider runtime request failed for {diagnostic_url}"))?;
    if response
        .content_length()
        .is_some_and(|size| size > MAX_ARCHIVE_BYTES)
    {
        bail!("provider runtime archive exceeds the download size limit");
    }
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("create provider runtime archive {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(reqwest::Error::without_url)
            .with_context(|| format!("read provider runtime response {diagnostic_url}"))?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded > MAX_ARCHIVE_BYTES {
            bail!("provider runtime archive exceeds the download size limit");
        }
        file.write_all(&chunk)
            .await
            .with_context(|| format!("write provider runtime archive {}", path.display()))?;
        hasher.update(&chunk);
    }
    file.flush().await?;
    let digest = hasher.finalize();
    verify_digest(&digest, expected_sha256, "archive")
}

fn extract_archive(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open provider runtime archive {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("open provider runtime ZIP")?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        bail!("provider runtime archive contains too many entries");
    }
    let mut remaining_budget = MAX_EXPANDED_BYTES;
    for index in 0..archive.len() {
        extract_entry(&mut archive, index, destination, &mut remaining_budget)?;
    }
    Ok(())
}

fn extract_entry(
    archive: &mut zip::ZipArchive<fs::File>,
    index: usize,
    destination: &Path,
    remaining_budget: &mut u64,
) -> Result<()> {
    let mut entry = archive.by_index(index)?;
    let relative = entry
        .enclosed_name()
        .with_context(|| format!("unsafe provider runtime ZIP entry {}", entry.name()))?;
    if zip_entry_is_symlink(&entry) {
        bail!("provider runtime ZIP contains a symlink: {}", entry.name());
    }
    let output = destination.join(relative);
    if entry.is_dir() {
        fs::create_dir_all(&output)?;
        return Ok(());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .with_context(|| {
            format!(
                "create extracted provider runtime file {}",
                output.display()
            )
        })?;
    let mut limited_reader = entry.take(*remaining_budget);
    let bytes_written = std::io::copy(&mut limited_reader, &mut file)?;
    if bytes_written > *remaining_budget {
        bail!("provider runtime archive exceeds the expanded size limit");
    }
    *remaining_budget = remaining_budget.saturating_sub(bytes_written);
    if limited_reader.limit() == 0 && entry.bytes().next().is_some() {
        bail!("provider runtime archive exceeds the expanded size limit");
    }
    file.flush()?;
    apply_zip_permissions(&output, entry.unix_mode())?;
    Ok(())
}

fn zip_entry_is_symlink(entry: &zip::read::ZipFile<'_>) -> bool {
    entry
        .unix_mode()
        .is_some_and(|mode| mode & 0o170000 == 0o120000)
}

#[cfg(unix)]
fn apply_zip_permissions(path: &Path, mode: Option<u32>) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(mode) = mode {
        fs::set_permissions(path, fs::Permissions::from_mode(mode & 0o777))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn apply_zip_permissions(_path: &Path, _mode: Option<u32>) -> Result<()> {
    Ok(())
}

fn find_provider_bundle(extracted: &Path) -> Result<PathBuf> {
    let mut matches = Vec::new();
    collect_provider_bundles(extracted, &mut matches)?;
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!("provider runtime archive contains no provider-runtime.json"),
        count => bail!("provider runtime archive contains {count} provider manifests"),
    }
}

fn collect_provider_bundles(directory: &Path, matches: &mut Vec<PathBuf>) -> Result<()> {
    if directory
        .join(crate::PROVIDER_RUNTIME_MANIFEST_FILE)
        .is_file()
    {
        matches.push(directory.to_path_buf());
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_provider_bundles(&entry.path(), matches)?;
        }
    }
    Ok(())
}

fn verify_digest(actual: &[u8], expected: &str, label: &str) -> Result<()> {
    let actual = actual
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let expected = crate::manifest::normalize_sha256(expected)?;
    if actual != expected {
        bail!("provider runtime {label} checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn verify_archive_file(path: &Path, expected: &str) -> Result<()> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("open provider runtime archive {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    verify_digest(&digest, expected, "archive")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PROVIDER_RUNTIME_SCHEMA_VERSION, ProviderRuntimeModel, ProviderRuntimePlatform};
    use std::collections::{BTreeMap, BTreeSet};
    use std::net::TcpListener;

    fn create_bundle(root: &Path) -> ProviderRuntimeManifest {
        let binary = root.join("bin/provider");
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, b"runtime").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let manifest = ProviderRuntimeManifest {
            schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
            runtime: ProviderRuntimeArtifact {
                id: "fixture-provider".to_string(),
                version: "1.0.0".to_string(),
                provider_kind: "fixture".to_string(),
                protocol_version: "1.0".to_string(),
                platform: ProviderRuntimePlatform {
                    os: std::env::consts::OS.to_string(),
                    arch: std::env::consts::ARCH.to_string(),
                    target: None,
                    minimum_os_version: None,
                },
                entrypoint: "bin/provider".to_string(),
                models: vec![ProviderRuntimeModel {
                    id: "fixture/model".to_string(),
                    kind: "fixture".to_string(),
                }],
                features: BTreeSet::new(),
                files: BTreeMap::from([(
                    "bin/provider".to_string(),
                    format!("{:x}", Sha256::digest(b"runtime")),
                )]),
                build: BTreeMap::new(),
                signature: None,
                url: None,
                archive_sha256: None,
            },
        };
        manifest.write_to_dir(root).unwrap();
        manifest
    }

    fn zip_bundle(bundle: &Path, output: &Path, symlink: bool) {
        use zip::write::SimpleFileOptions;
        let file = fs::File::create(output).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default().unix_permissions(0o755);
        if symlink {
            writer
                .add_symlink(
                    "bundle/bin/provider",
                    "outside",
                    SimpleFileOptions::default(),
                )
                .unwrap();
        } else {
            writer.start_file("bundle/bin/provider", options).unwrap();
            writer
                .write_all(&fs::read(bundle.join("bin/provider")).unwrap())
                .unwrap();
        }
        writer
            .start_file(
                "bundle/provider-runtime.json",
                SimpleFileOptions::default().unix_permissions(0o644),
            )
            .unwrap();
        writer
            .write_all(&fs::read(bundle.join("provider-runtime.json")).unwrap())
            .unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn extracts_and_installs_a_verified_archive() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("bundle-source");
        fs::create_dir(&bundle).unwrap();
        let expected = create_bundle(&bundle);
        let archive = temp.path().join("runtime.zip");
        zip_bundle(&bundle, &archive, false);
        let extracted = temp.path().join("extracted");
        fs::create_dir(&extracted).unwrap();

        extract_archive(&archive, &extracted).unwrap();
        let found = find_provider_bundle(&extracted).unwrap();
        let actual = ProviderRuntimeManifest::read_from_dir(&found).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_symlink_entries_before_installation() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("bundle-source");
        fs::create_dir(&bundle).unwrap();
        create_bundle(&bundle);
        let archive = temp.path().join("runtime.zip");
        zip_bundle(&bundle, &archive, true);
        let extracted = temp.path().join("extracted");
        fs::create_dir(&extracted).unwrap();

        let error = extract_archive(&archive, &extracted).unwrap_err();
        assert!(error.to_string().contains("symlink"), "{error:?}");
    }

    #[tokio::test]
    async fn downloads_verifies_and_installs_a_release_artifact() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("bundle-source");
        fs::create_dir(&bundle).unwrap();
        let manifest = create_bundle(&bundle);
        let archive = temp.path().join("runtime.zip");
        zip_bundle(&bundle, &archive, false);
        let archive_bytes = fs::read(&archive).unwrap();
        let archive_sha256 = format!("{:x}", Sha256::digest(&archive_bytes));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                archive_bytes.len()
            )
            .unwrap();
            stream.write_all(&archive_bytes).unwrap();
        });
        let mut release_artifact = manifest.runtime.clone();
        release_artifact.url = Some(format!("http://{address}/runtime.zip?token=redacted"));
        release_artifact.archive_sha256 = Some(archive_sha256);

        let outcome = install_provider_runtime(ProviderRuntimeInstallOptions {
            host: crate::ProviderRuntimeHost::current(),
            request: ProviderRuntimeRequest {
                artifact_id: Some(release_artifact.id.clone()),
                ..Default::default()
            },
            release_manifest: ProviderRuntimeReleaseManifest {
                schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
                artifacts: vec![release_artifact],
            },
            cache_dir: Some(temp.path().join("cache")),
            allow_download: true,
            ..Default::default()
        })
        .await
        .unwrap();
        server.join().unwrap();

        assert_eq!(outcome.status, ProviderRuntimeInstallStatus::Installed);
        assert!(outcome.runtime.entrypoint().is_file());
        assert_eq!(outcome.runtime.manifest.runtime, manifest.runtime);
    }
}
