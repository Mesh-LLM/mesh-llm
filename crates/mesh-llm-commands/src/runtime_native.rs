use anyhow::Result;
use mesh_llm_native_runtime::{
    HostRuntimeProfile, NativeRuntimePruneMode, NativeRuntimeResolver, RuntimeSelection,
};
use mesh_llm_runtime_install::{
    CURRENT_MESH_VERSION, NativeRuntimeDownloadProgressCallback, NativeRuntimeInstallOptions,
    NativeRuntimeInstallStatus, NativeRuntimeManifestOptions, host_runtime_profile,
    install_native_runtime, load_release_manifest, native_runtime_cache,
};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub async fn run_native_runtime_list(
    available: bool,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let cache = native_runtime_cache(cache_dir)?;
    if available {
        if !json_output && manifest_path.is_none() && bundle_dirs.is_empty() {
            eprintln!("🔎 Loading native runtime release manifest");
        }
        let manifest = load_release_manifest(NativeRuntimeManifestOptions {
            manifest_path: manifest_path.map(Path::to_path_buf),
            bundle_dirs: bundle_dirs.to_vec(),
            ..Default::default()
        })
        .await?;
        let profile = host_runtime_profile();
        let cache = native_runtime_cache(cache_dir)?;
        let resolution = NativeRuntimeResolver::new(
            CURRENT_MESH_VERSION,
            profile.clone(),
            manifest.clone(),
            cache,
        )
        .resolve(&RuntimeSelection::Recommended)
        .ok();
        let rows = manifest
            .artifacts
            .iter()
            .map(|artifact| {
                let evaluation = resolution.as_ref().and_then(|resolution| {
                    resolution
                        .evaluated
                        .iter()
                        .find(|candidate| candidate.artifact.id == artifact.id)
                });
                let supported = evaluation.is_some_and(|candidate| candidate.compatible);
                json!({
                    "id": artifact.id,
                    "mesh_version": artifact.mesh_version.as_deref(),
                    "skippy_abi": artifact.skippy_abi,
                    "backend": artifact.backend.kind.to_string(),
                    "os": artifact.platform.os,
                    "arch": artifact.platform.arch,
                    "supported": supported,
                    "rejection_reasons": evaluation.map(|candidate| &candidate.rejection_reasons),
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

pub async fn run_native_runtime_install(
    requested_runtime: Option<&str>,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let selection = RuntimeSelection::parse(requested_runtime)?;
    if !json_output && manifest_path.is_none() && bundle_dirs.is_empty() {
        eprintln!("🔎 Loading native runtime release manifest");
    }
    if !json_output {
        eprintln!("🔎 Detecting host runtime profile");
    }
    let outcome = install_native_runtime(NativeRuntimeInstallOptions {
        selection,
        manifest_path: manifest_path.map(Path::to_path_buf),
        bundle_dirs: bundle_dirs.to_vec(),
        cache_dir: cache_dir.map(Path::to_path_buf),
        progress: cli_download_progress(json_output),
        ..Default::default()
    })
    .await?;
    match outcome.status {
        NativeRuntimeInstallStatus::AlreadyInstalled => {
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "already_installed",
                        "runtime": outcome.runtime,
                        "resolution": outcome.resolution,
                    }))?
                );
            } else {
                eprintln!(
                    "✅ Native runtime already installed: {}",
                    outcome.runtime.native_runtime_id
                );
                eprintln!("   path: {}", outcome.runtime.path.display());
            }
        }
        NativeRuntimeInstallStatus::Installed => {
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "installed",
                        "runtime": outcome.runtime,
                        "resolution": outcome.resolution,
                    }))?
                );
            } else {
                eprintln!("✅ Installed {}", outcome.runtime.native_runtime_id);
                eprintln!("   version: {}", outcome.runtime.mesh_version);
                eprintln!("   flavor: {}", outcome.runtime.flavor);
                eprintln!("   path: {}", outcome.runtime.path.display());
            }
        }
    }
    Ok(())
}

struct DownloadProgress {
    native_runtime_id: Option<String>,
    last_percent: Option<u64>,
    last_tick: Instant,
}

impl DownloadProgress {
    fn new() -> Self {
        Self {
            native_runtime_id: None,
            last_percent: None,
            last_tick: Instant::now(),
        }
    }

    fn tick(
        &mut self,
        native_runtime_id: &str,
        downloaded: u64,
        total: Option<u64>,
        finished: bool,
    ) {
        if self.native_runtime_id.is_none() {
            self.native_runtime_id = Some(native_runtime_id.to_string());
            eprintln!("⬇️  Downloading native runtime {native_runtime_id}");
        }
        if finished {
            self.finish(downloaded);
            return;
        }
        let should_print = match total {
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
            match total {
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
        eprintln!("   downloaded {}", human_bytes(downloaded));
    }
}

fn cli_download_progress(json_output: bool) -> Option<NativeRuntimeDownloadProgressCallback> {
    if json_output {
        return None;
    }
    let progress = Arc::new(Mutex::new(DownloadProgress::new()));
    Some(Arc::new(move |event| {
        let Ok(mut progress) = progress.lock() else {
            return;
        };
        progress.tick(
            &event.native_runtime_id,
            event.downloaded_bytes,
            event.total_bytes,
            event.finished,
        );
    }))
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

pub fn run_native_runtime_remove(
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

pub fn run_native_runtime_prune(
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

pub fn run_native_runtime_doctor(json_output: bool) -> Result<()> {
    let cache = native_runtime_cache(None)?;
    let profile = host_runtime_profile();
    let installed = cache.installed()?;
    let current_version_runtimes = installed
        .iter()
        .filter(|runtime| runtime.mesh_version == CURRENT_MESH_VERSION)
        .collect::<Vec<_>>();
    let selected = current_version_runtimes
        .iter()
        .max_by_key(|runtime| runtime.manifest.runtime.backend.kind.default_rank());

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
            row["backend"].as_str().unwrap_or("unknown"),
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
