use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use mesh_llm_native_runtime::{
    HostGpuProfile, HostRuntimeProfile, NativeRuntimeCache, NativeRuntimeFlavor,
    NativeRuntimeManifest, NativeRuntimePruneMode, NativeRuntimeReleaseManifest,
    NativeRuntimeResolver, NativeRuntimeSource, RuntimeSelection,
};
use serde::Serialize;
use serde_json::json;
use sha2::Digest;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;

const CURRENT_MESH_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn run_native_runtime_list(
    available: bool,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let cache = native_runtime_cache(cache_dir)?;
    if available {
        let manifest = load_release_manifest(manifest_path, bundle_dirs)?;
        let profile = host_runtime_profile();
        let rows = manifest
            .artifacts
            .iter()
            .map(|artifact| {
                let supported = artifact.mesh_version == CURRENT_MESH_VERSION
                    && artifact.os == profile.os
                    && artifact.arch == profile.arch
                    && profile.supports_flavor(&artifact.flavor);
                json!({
                    "id": artifact.native_runtime_id,
                    "mesh_version": artifact.mesh_version,
                    "flavor": artifact.flavor.to_string(),
                    "os": artifact.os,
                    "arch": artifact.arch,
                    "supported": supported,
                    "url": artifact.url.as_deref(),
                })
            })
            .collect::<Vec<_>>();
        if json_output {
            println!("{}", serde_json::to_string_pretty(&rows)?);
        } else {
            print_available_runtimes(&rows);
        }
        return Ok(());
    }

    let installed = cache.installed()?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&installed)?);
    } else {
        print_installed_runtimes(&installed, cache.root());
    }
    Ok(())
}

pub(crate) async fn run_native_runtime_install(
    requested_runtime: Option<&str>,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let selection = RuntimeSelection::parse(requested_runtime)?;
    let manifest = load_release_manifest(manifest_path, bundle_dirs)?;
    if manifest.artifacts.is_empty() {
        bail!(
            "no native runtime manifest entries found; pass --manifest or --bundle-dir before installing"
        );
    }

    if !json_output {
        eprintln!("🔎 Detecting host runtime profile");
    }
    let profile = host_runtime_profile();
    let cache = native_runtime_cache(cache_dir)?;
    let resolution =
        NativeRuntimeResolver::new(CURRENT_MESH_VERSION, profile, manifest, cache.clone())
            .with_bundle_dirs(bundle_dirs.to_vec())
            .resolve(&selection)?;

    match &resolution.source {
        NativeRuntimeSource::Installed { path } => {
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "already_installed",
                        "runtime": resolution.selected,
                        "path": path,
                    }))?
                );
            } else {
                eprintln!(
                    "✅ Native runtime already installed: {}",
                    resolution.selected.native_runtime_id
                );
                eprintln!("   path: {}", path.display());
            }
        }
        NativeRuntimeSource::Bundle { path } => {
            if !json_output {
                eprintln!(
                    "📦 Installing native runtime {}",
                    resolution.selected.native_runtime_id
                );
            }
            let installed = cache.install_from_dir(path)?;
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "installed",
                        "runtime": installed,
                    }))?
                );
            } else {
                eprintln!("✅ Installed {}", installed.native_runtime_id);
                eprintln!("   version: {}", installed.mesh_version);
                eprintln!("   flavor: {}", installed.flavor);
                eprintln!("   path: {}", installed.path.display());
            }
        }
        NativeRuntimeSource::Download { url } => {
            let installed =
                download_and_install_runtime(&cache, &resolution.selected, url, json_output)
                    .await?;
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "installed",
                        "runtime": installed,
                    }))?
                );
            } else {
                eprintln!("✅ Installed {}", installed.native_runtime_id);
                eprintln!("   version: {}", installed.mesh_version);
                eprintln!("   flavor: {}", installed.flavor);
                eprintln!("   path: {}", installed.path.display());
            }
        }
        NativeRuntimeSource::Missing => {
            bail!(
                "selected native runtime {} is not installed and no bundle or download URL was available",
                resolution.selected.native_runtime_id
            );
        }
    }
    Ok(())
}

async fn download_and_install_runtime(
    cache: &NativeRuntimeCache,
    artifact: &mesh_llm_native_runtime::NativeRuntimeArtifact,
    url: &str,
    json_output: bool,
) -> Result<mesh_llm_native_runtime::InstalledNativeRuntime> {
    let temp = tempfile::Builder::new()
        .prefix("mesh-native-runtime-")
        .tempdir()
        .context("create native runtime download workspace")?;
    let archive = temp
        .path()
        .join(format!("{}.tar.gz", artifact.native_runtime_id));
    if !json_output {
        eprintln!(
            "⬇️  Downloading native runtime {}",
            artifact.native_runtime_id
        );
    }
    download_runtime_archive(url, &archive, artifact.sha256.as_deref(), json_output).await?;
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
    expected_sha256: Option<&str>,
    json_output: bool,
) -> Result<()> {
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
    let mut progress = DownloadProgress::new(total, json_output);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read native runtime body from {url}"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("write native runtime archive {}", path.display()))?;
        downloaded += chunk.len() as u64;
        sha2::Digest::update(&mut hasher, &chunk);
        progress.tick(downloaded);
    }
    file.flush()
        .await
        .with_context(|| format!("flush native runtime archive {}", path.display()))?;
    progress.finish(downloaded);
    if let Some(expected) = expected_sha256 {
        let expected = normalize_sha256(expected)?;
        let actual = hex::encode(sha2::Digest::finalize(hasher));
        if actual != expected {
            bail!("native runtime checksum mismatch: expected {expected}, got {actual}");
        }
    }
    Ok(())
}

struct DownloadProgress {
    total: Option<u64>,
    quiet: bool,
    last_percent: Option<u64>,
    last_tick: Instant,
}

impl DownloadProgress {
    fn new(total: Option<u64>, quiet: bool) -> Self {
        Self {
            total,
            quiet,
            last_percent: None,
            last_tick: Instant::now(),
        }
    }

    fn tick(&mut self, downloaded: u64) {
        if self.quiet {
            return;
        }
        let should_print = match self.total {
            Some(total) if total > 0 => {
                let percent = downloaded.saturating_mul(100) / total;
                let crossed_step = self
                    .last_percent
                    .map(|last| percent >= last.saturating_add(5))
                    .unwrap_or(true);
                if crossed_step || percent == 100 {
                    self.last_percent = Some(percent);
                    true
                } else {
                    false
                }
            }
            _ => self.last_tick.elapsed() >= Duration::from_secs(1),
        };
        if should_print {
            self.last_tick = Instant::now();
            match self.total {
                Some(total) if total > 0 => eprintln!(
                    "   downloaded {} / {} ({})",
                    human_bytes(downloaded),
                    human_bytes(total),
                    format_args!("{}%", self.last_percent.unwrap_or(0))
                ),
                _ => eprintln!("   downloaded {}", human_bytes(downloaded)),
            }
        }
    }

    fn finish(&mut self, downloaded: u64) {
        if !self.quiet {
            eprintln!("   downloaded {}", human_bytes(downloaded));
        }
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for candidate in UNITS.iter().skip(1) {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
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

pub(crate) fn run_native_runtime_remove(
    native_runtime_id: &str,
    mesh_version: Option<&str>,
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let version = mesh_version.unwrap_or(CURRENT_MESH_VERSION);
    let cache = native_runtime_cache(cache_dir)?;
    let removed = cache.remove(version, native_runtime_id)?;
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "mesh_version": version,
                "native_runtime_id": native_runtime_id,
                "removed": removed,
            }))?
        );
    } else if removed {
        eprintln!("✅ Removed native runtime {native_runtime_id} for MeshLLM {version}");
    } else {
        eprintln!("🔎 Native runtime {native_runtime_id} for MeshLLM {version} was not installed");
    }
    Ok(())
}

pub(crate) fn run_native_runtime_prune(
    active_only: bool,
    mesh_version: Option<&str>,
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let version = mesh_version.unwrap_or(CURRENT_MESH_VERSION);
    let mode = if active_only {
        NativeRuntimePruneMode::ActiveOnly
    } else {
        NativeRuntimePruneMode::KeepActiveAndPrevious
    };
    let cache = native_runtime_cache(cache_dir)?;
    let plan = cache.prune(version, mode)?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else if plan.remove_dirs.is_empty() {
        eprintln!("✅ Native runtime cache already pruned");
    } else {
        eprintln!(
            "✅ Pruned {} native runtime cache version(s)",
            plan.remove_dirs.len()
        );
        for dir in plan.remove_dirs {
            eprintln!("   removed: {}", dir.display());
        }
    }
    Ok(())
}

pub(crate) fn run_native_runtime_doctor(json_output: bool) -> Result<()> {
    let cache = native_runtime_cache(None)?;
    let profile = host_runtime_profile();
    let installed = cache.installed()?;
    let current_version_runtimes = installed
        .iter()
        .filter(|runtime| runtime.mesh_version == CURRENT_MESH_VERSION)
        .collect::<Vec<_>>();
    let selected = current_version_runtimes
        .iter()
        .max_by_key(|runtime| runtime.manifest.artifact.flavor.default_rank());

    let report = NativeRuntimeDoctorReport {
        mesh_version: CURRENT_MESH_VERSION.to_string(),
        host: profile,
        cache_path: cache.root().to_path_buf(),
        selected_runtime_id: selected.map(|runtime| runtime.native_runtime_id.clone()),
        selected_runtime_flavor: selected.map(|runtime| runtime.flavor.clone()),
        selected_runtime_path: selected.map(|runtime| runtime.path.clone()),
        installed_count: installed.len(),
        current_version_installed_count: current_version_runtimes.len(),
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_doctor_report(&report);
    }
    Ok(())
}

#[derive(Serialize)]
struct NativeRuntimeDoctorReport {
    mesh_version: String,
    host: HostRuntimeProfile,
    cache_path: PathBuf,
    selected_runtime_id: Option<String>,
    selected_runtime_flavor: Option<String>,
    selected_runtime_path: Option<PathBuf>,
    installed_count: usize,
    current_version_installed_count: usize,
}

fn native_runtime_cache(cache_dir: Option<&Path>) -> Result<NativeRuntimeCache> {
    let root = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
            .context("cannot determine native runtime cache directory")?
            .join("mesh-llm")
            .join("native-runtimes"),
    };
    Ok(NativeRuntimeCache::new(root))
}

fn load_release_manifest(
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
) -> Result<NativeRuntimeReleaseManifest> {
    let mut artifacts = Vec::new();
    let mut mesh_version = CURRENT_MESH_VERSION.to_string();
    if let Some(path) = manifest_path {
        let manifest = NativeRuntimeReleaseManifest::read_from_path(path)?;
        mesh_version = manifest.mesh_version.clone();
        artifacts.extend(manifest.artifacts);
    }
    for dir in bundle_dirs {
        let manifest = NativeRuntimeManifest::read_from_dir(dir)
            .with_context(|| format!("read bundled native runtime {}", dir.display()))?;
        mesh_version = manifest.artifact.mesh_version.clone();
        artifacts.push(manifest.artifact);
    }
    Ok(NativeRuntimeReleaseManifest {
        mesh_version,
        artifacts,
    })
}

fn host_runtime_profile() -> HostRuntimeProfile {
    let survey = crate::system::hardware::survey();
    let gpus = survey
        .gpus
        .iter()
        .map(|gpu| HostGpuProfile {
            display_name: gpu.display_name.clone(),
            backend_device: gpu.backend_device.clone(),
            stable_id: gpu.stable_id.clone(),
            vram_bytes: Some(gpu.vram_bytes),
            unified_memory: gpu.unified_memory,
        })
        .collect::<Vec<_>>();
    HostRuntimeProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        target_triple: option_env!("TARGET").map(str::to_string),
        available_flavors: detected_native_runtime_flavors(&survey.gpus),
        gpus,
    }
}

fn detected_native_runtime_flavors(
    gpus: &[crate::system::hardware::GpuFacts],
) -> BTreeSet<NativeRuntimeFlavor> {
    let mut flavors = BTreeSet::from([NativeRuntimeFlavor::Cpu]);
    if cfg!(target_os = "macos") {
        flavors.insert(NativeRuntimeFlavor::Metal);
    }
    for gpu in gpus {
        let label = format!(
            "{} {}",
            gpu.display_name,
            gpu.backend_device.as_deref().unwrap_or_default()
        )
        .to_ascii_lowercase();
        if label.contains("cuda") || label.contains("nvidia") {
            flavors.insert(NativeRuntimeFlavor::Cuda);
        }
        if label.contains("blackwell")
            || label.contains("gb200")
            || label.contains("b200")
            || label.contains("rtx 50")
        {
            flavors.insert(NativeRuntimeFlavor::CudaBlackwell);
        }
        if label.contains("rocm") || label.contains("hip") || label.contains("amd") {
            flavors.insert(NativeRuntimeFlavor::Rocm);
        }
        if label.contains("vulkan") {
            flavors.insert(NativeRuntimeFlavor::Vulkan);
        }
    }
    flavors
}

fn print_available_runtimes(rows: &[serde_json::Value]) {
    if rows.is_empty() {
        println!("📦 No native runtime manifest entries found");
        println!("   Pass --manifest or --bundle-dir to inspect available runtimes.");
        return;
    }
    println!("📦 Available native runtimes");
    for row in rows {
        let status = if row["supported"].as_bool().unwrap_or(false) {
            "compatible"
        } else {
            "not compatible"
        };
        println!(
            "  - {} {} ({}, {}/{})",
            row["id"].as_str().unwrap_or("unknown"),
            status,
            row["flavor"].as_str().unwrap_or("unknown"),
            row["os"].as_str().unwrap_or("unknown"),
            row["arch"].as_str().unwrap_or("unknown")
        );
    }
}

fn print_installed_runtimes(
    installed: &[mesh_llm_native_runtime::InstalledNativeRuntime],
    cache_root: &Path,
) {
    if installed.is_empty() {
        println!("📦 No native runtimes installed");
        println!("   cache: {}", cache_root.display());
        return;
    }
    println!("📦 Installed native runtimes");
    println!("   cache: {}", cache_root.display());
    for runtime in installed {
        println!(
            "  - {} {} ({})",
            runtime.native_runtime_id, runtime.mesh_version, runtime.flavor
        );
        println!("    path: {}", runtime.path.display());
    }
}

fn print_doctor_report(report: &NativeRuntimeDoctorReport) {
    println!("🩺 MeshLLM doctor");
    println!();
    println!("Native runtime:");
    println!("  mesh version: {}", report.mesh_version);
    println!("  cache: {}", report.cache_path.display());
    println!("  host: {}/{}", report.host.os, report.host.arch);
    let flavors = report
        .host
        .available_flavors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  detected flavors: {flavors}");
    match &report.selected_runtime_id {
        Some(id) => {
            println!("  selected: {id}");
            if let Some(flavor) = &report.selected_runtime_flavor {
                println!("  flavor: {flavor}");
            }
            if let Some(path) = &report.selected_runtime_path {
                println!("  path: {}", path.display());
            }
        }
        None => {
            println!("  selected: none");
            println!("  status: no native runtime installed for this MeshLLM version");
        }
    }
    println!("  installed: {}", report.installed_count);
    println!(
        "  installed for current version: {}",
        report.current_version_installed_count
    );
}
