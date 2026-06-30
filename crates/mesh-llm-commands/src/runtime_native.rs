mod formatters;
mod setup_helpers;

use anyhow::Result;
use mesh_llm_native_runtime::{NativeRuntimePruneMode, NativeRuntimeResolver, RuntimeSelection};
use mesh_llm_runtime_install::{
    CURRENT_MESH_VERSION, NativeRuntimeDownloadProgressCallback, NativeRuntimeManifestOptions,
    host_runtime_profile, install_native_runtime, load_release_manifest, native_runtime_cache,
};
use mesh_llm_tui::terminal_progress::{
    ratio_complete_u64, render_inline_gauge_with_reserved_width,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use formatters::{AvailableRuntimeRow, NativeRuntimeDoctorReport, runtime_native_formatter};
pub use setup_helpers::{
    SetupNativeRuntimeOptions, SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult,
    SetupNativeRuntimeStatus, install_and_prune_native_runtime_for_setup,
};
use setup_helpers::{
    native_runtime_install_options, prune_native_runtime_cache, resolve_runtime_selection,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeRuntimeDoctorReadiness {
    healthy: bool,
    status: String,
    blockers: Vec<String>,
    recommendations: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeRuntimeConfigSelection<'a> {
    pub mesh_version: Option<&'a str>,
    pub skippy_abi_version: Option<&'a str>,
    pub selection: Option<&'a str>,
}

impl<'a> NativeRuntimeConfigSelection<'a> {
    fn mesh_version_or_current(self) -> &'a str {
        self.mesh_version.unwrap_or(CURRENT_MESH_VERSION)
    }
}

pub async fn run_native_runtime_list(
    available: bool,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    configured: NativeRuntimeConfigSelection<'_>,
    json_output: bool,
) -> Result<()> {
    let mesh_version = configured.mesh_version_or_current();
    let selection = RuntimeSelection::parse(configured.selection)?;
    let cache = native_runtime_cache(cache_dir)?;
    let formatter = runtime_native_formatter(json_output);
    if available {
        print_configured_selector(configured, json_output);
        if !json_output && manifest_path.is_none() && bundle_dirs.is_empty() {
            eprintln!("🔎 Loading native runtime release manifest");
        }
        let manifest = load_release_manifest(NativeRuntimeManifestOptions {
            mesh_version: mesh_version.to_string(),
            manifest_path: manifest_path.map(Path::to_path_buf),
            bundle_dirs: bundle_dirs.to_vec(),
            ..Default::default()
        })
        .await?;
        let profile = host_runtime_profile();
        let cache = native_runtime_cache(cache_dir)?;
        let mut resolver =
            NativeRuntimeResolver::new(mesh_version, profile.clone(), manifest.clone(), cache)
                .with_bundle_dirs(bundle_dirs.to_vec());
        if let Some(skippy_abi_version) = configured.skippy_abi_version {
            resolver = resolver.with_skippy_abi_version(skippy_abi_version);
        }
        let evaluated = resolver.evaluate(&selection)?;
        let rows = manifest
            .artifacts
            .iter()
            .map(|artifact| {
                let evaluation = evaluated
                    .iter()
                    .find(|candidate| candidate.artifact.id == artifact.id);
                let supported = evaluation.is_some_and(|candidate| candidate.compatible);
                AvailableRuntimeRow {
                    id: artifact.id.clone(),
                    mesh_version: artifact.mesh_version.clone(),
                    skippy_abi: artifact.skippy_abi.clone(),
                    backend: artifact.backend.kind.to_string(),
                    os: artifact.platform.os.clone(),
                    arch: artifact.platform.arch.clone(),
                    supported,
                    rejection_reasons: evaluation
                        .map(|candidate| candidate.rejection_reasons.clone())
                        .unwrap_or_default(),
                    url: artifact.url.clone(),
                }
            })
            .collect::<Vec<_>>();
        return formatter.render_available(&rows);
    }

    let installed = cache.installed()?;
    formatter.render_installed(&installed, cache.root())
}

pub async fn run_native_runtime_install(
    requested_runtime: Option<&str>,
    manifest_path: Option<&Path>,
    bundle_dirs: &[PathBuf],
    cache_dir: Option<&Path>,
    configured: NativeRuntimeConfigSelection<'_>,
    json_output: bool,
) -> Result<()> {
    let resolved_selection = resolve_runtime_selection(requested_runtime, configured)?;
    if !json_output && manifest_path.is_none() && bundle_dirs.is_empty() {
        eprintln!("🔎 Loading native runtime release manifest");
    }
    if !json_output {
        eprintln!("🔎 Detecting host runtime profile");
    }
    print_configured_selector(
        NativeRuntimeConfigSelection {
            selection: resolved_selection.configured_selection,
            ..configured
        },
        json_output,
    );
    let formatter = runtime_native_formatter(json_output);
    let install_options = native_runtime_install_options(
        resolved_selection.selection,
        manifest_path,
        bundle_dirs,
        cache_dir,
        configured,
        cli_download_progress(json_output),
    );
    let outcome = match install_native_runtime(install_options).await {
        Ok(outcome) => outcome,
        Err(error) => {
            formatter.render_install_error(&error)?;
            return Err(error);
        }
    };
    formatter.render_install(&outcome)
}

fn print_configured_selector(configured: NativeRuntimeConfigSelection<'_>, json_output: bool) {
    if json_output || configured.mesh_version.is_none() {
        return;
    }
    let mesh_version = configured.mesh_version_or_current();
    eprintln!("🔒 Using native runtime selector from config");
    eprintln!("   mesh version: {mesh_version}");
    if let Some(skippy_abi_version) = configured.skippy_abi_version {
        eprintln!("   Skippy ABI: {skippy_abi_version}");
    }
    if let Some(configured_selection) = configured.selection {
        eprintln!("   selection: {configured_selection}");
    }
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
                Some(total) if total > 0 => {
                    let gauge = render_inline_gauge_with_reserved_width(
                        ratio_complete_u64(downloaded, total),
                        &format!(
                            "downloaded {} / {} ({}%)",
                            human_bytes(downloaded),
                            human_bytes(total),
                            self.last_percent.unwrap_or(0)
                        ),
                        3,
                    );
                    eprint!("\r\x1b[2K   {gauge}");
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
                _ => {
                    eprint!("\r\x1b[2K   downloaded {}", human_bytes(downloaded));
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
            }
        }
    }

    fn finish(&mut self, downloaded: u64) {
        eprintln!("\r\x1b[2K   downloaded {}", human_bytes(downloaded));
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
    runtime_native_formatter(json_output).render_remove(native_runtime_id, version, removed)
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
    let plan = prune_native_runtime_cache(version, mode, cache_dir)?;
    runtime_native_formatter(json_output).render_prune(&plan)
}

pub fn run_native_runtime_doctor(
    mesh_version: Option<&str>,
    skippy_abi_version: Option<&str>,
    configured_selection: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let cache = native_runtime_cache(None)?;
    let profile = host_runtime_profile();
    let installed = cache.installed()?;
    let selected_mesh_version = mesh_version.unwrap_or(CURRENT_MESH_VERSION);
    let runtime_selection = RuntimeSelection::parse(configured_selection)?;
    let selected_version_runtimes = installed
        .iter()
        .filter(|runtime| runtime.mesh_version == selected_mesh_version)
        .collect::<Vec<_>>();
    let installed_artifacts = selected_version_runtimes
        .iter()
        .map(|runtime| runtime.manifest.runtime.clone())
        .collect::<Vec<_>>();
    let selected_candidate = mesh_llm_native_runtime::select_native_runtime_from_artifacts(
        &installed_artifacts,
        &profile,
        selected_mesh_version,
        skippy_abi_version,
        &runtime_selection,
    );
    let selected = selected_candidate.as_ref().and_then(|candidate| {
        selected_version_runtimes.iter().find(|runtime| {
            runtime.native_runtime_id == candidate.artifact.id
                && runtime.manifest.runtime.skippy_abi == candidate.artifact.skippy_abi
        })
    });
    let readiness =
        native_runtime_doctor_readiness(selected.map(|runtime| runtime.native_runtime_id.as_str()));

    let report = NativeRuntimeDoctorReport {
        healthy: readiness.healthy,
        status: readiness.status,
        blockers: readiness.blockers,
        recommendations: readiness.recommendations,
        running_mesh_version: CURRENT_MESH_VERSION.to_string(),
        selected_mesh_version: selected_mesh_version.to_string(),
        configured_skippy_abi: skippy_abi_version.map(ToString::to_string),
        configured_selection: configured_selection.map(ToString::to_string),
        host: profile,
        cache_path: cache.root().to_path_buf(),
        selected_runtime_id: selected.map(|runtime| runtime.native_runtime_id.clone()),
        selected_runtime_flavor: selected.map(|runtime| runtime.flavor.clone()),
        selected_runtime_path: selected.map(|runtime| runtime.path.clone()),
        installed_count: installed.len(),
        selected_version_installed_count: selected_version_runtimes.len(),
    };

    runtime_native_formatter(json_output).render_doctor(&report)?;
    if !report.healthy {
        anyhow::bail!(
            "{}",
            report
                .blockers
                .first()
                .map(String::as_str)
                .unwrap_or("native runtime doctor found a blocking readiness issue")
        );
    }
    Ok(())
}

fn native_runtime_doctor_readiness(
    selected_runtime_id: Option<&str>,
) -> NativeRuntimeDoctorReadiness {
    if selected_runtime_id.is_some() {
        return NativeRuntimeDoctorReadiness {
            healthy: true,
            status: "ok".to_string(),
            blockers: Vec::new(),
            recommendations: Vec::new(),
        };
    }
    NativeRuntimeDoctorReadiness {
        healthy: false,
        status: "unhealthy".to_string(),
        blockers: vec![
            "No compatible native runtime is installed for the selected MeshLLM version and host."
                .to_string(),
        ],
        recommendations: vec![
            "Run `mesh-llm runtime install` to install the recommended native runtime.".to_string(),
            "Run `mesh-llm runtime list --available` to inspect compatible and rejected runtimes."
                .to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_readiness_blocks_missing_selected_runtime() {
        let readiness = native_runtime_doctor_readiness(None);

        assert!(!readiness.healthy);
        assert_eq!(readiness.status, "unhealthy");
        assert!(
            readiness
                .blockers
                .iter()
                .any(|item| item.contains("No compatible native runtime"))
        );
        assert!(
            readiness
                .recommendations
                .iter()
                .any(|item| item.contains("mesh-llm runtime install"))
        );
        assert!(
            readiness
                .recommendations
                .iter()
                .any(|item| item.contains("mesh-llm runtime list --available"))
        );
    }

    #[test]
    fn doctor_readiness_accepts_selected_runtime() {
        let readiness = native_runtime_doctor_readiness(Some("meshllm-native-runtime-test-cpu"));

        assert!(readiness.healthy);
        assert_eq!(readiness.status, "ok");
        assert!(readiness.blockers.is_empty());
    }
}
