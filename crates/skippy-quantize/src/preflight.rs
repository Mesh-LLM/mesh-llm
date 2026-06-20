use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::backend::{BackendKind, ensure_convert_backend, ensure_quant_backend};
use crate::manifest::{Manifest, manifest_progress, read_manifest};
use crate::splits::{Progress, SplitWindow, next_missing_window_in_range, split_status};
use crate::types::JobKind;

#[derive(Debug, Serialize)]
struct JobPreflight {
    kind: JobKind,
    manifest_path: PathBuf,
    manifest_exists: bool,
    manifest_matches: bool,
    source_complete: Option<bool>,
    source_shards: Option<usize>,
    expected_source_shards: Option<u32>,
    source_missing_ranges: Option<Vec<ProgressWindow>>,
    target_shards: usize,
    expected_target_shards: u32,
    first_missing_target: Option<u32>,
    target_missing_ranges: Vec<ProgressWindow>,
    next_window: Option<ProgressWindow>,
    requested_window: Option<ProgressWindow>,
    next_requested_window: Option<ProgressWindow>,
    backend_kind: String,
    backend_path: Option<PathBuf>,
    backend_ready: bool,
    backend_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProgressWindow {
    first_split: u32,
    last_split: u32,
}

pub fn run_job_preflight(
    manifest_path: &Path,
    manifest: &Manifest,
    source_split: Option<(&Path, &str)>,
    requested_window: Option<SplitWindow>,
    backend_kind: BackendKind,
    backend_path: Option<&Path>,
    json: bool,
) -> Result<()> {
    ensure_backend_supported(manifest.kind, backend_kind)?;
    let manifest_exists = manifest_path.exists();
    let manifest_matches = if manifest_exists {
        read_manifest(manifest_path)? == *manifest
    } else {
        true
    };
    let target_progress = manifest_progress(manifest)?;
    let source_progress = source_split
        .map(|(source, prefix)| split_status(source, prefix, None))
        .transpose()?;
    let next_requested_window = requested_window.and_then(|requested| {
        next_missing_window_in_range(&target_progress.missing_ranges, requested)
    });
    let backend_check = check_backend_ready(backend_kind, backend_path);
    let report = JobPreflight {
        kind: manifest.kind,
        manifest_path: manifest_path.to_path_buf(),
        manifest_exists,
        manifest_matches,
        source_complete: source_progress.as_ref().map(is_complete),
        source_shards: source_progress
            .as_ref()
            .map(|progress| progress.completed_count),
        expected_source_shards: source_progress
            .as_ref()
            .map(|progress| progress.expected_splits),
        source_missing_ranges: source_progress
            .as_ref()
            .map(|progress| progress_windows(&progress.missing_ranges)),
        target_shards: target_progress.completed_count,
        expected_target_shards: target_progress.expected_splits,
        first_missing_target: target_progress.first_missing,
        target_missing_ranges: progress_windows(&target_progress.missing_ranges),
        next_window: target_progress.next_window.map(|window| ProgressWindow {
            first_split: window.first_split,
            last_split: window.last_split,
        }),
        requested_window: requested_window.map(ProgressWindow::from),
        next_requested_window: next_requested_window.map(ProgressWindow::from),
        backend_kind: backend_kind.as_str().to_string(),
        backend_path: backend_path.map(Path::to_path_buf),
        backend_ready: backend_check.ready,
        backend_error: backend_check.error,
    };
    print_preflight(&report, json)?;
    ensure!(
        report.manifest_matches,
        "existing manifest does not match requested job"
    );
    ensure!(
        report.backend_ready,
        "backend is not ready for {}: {} ({})",
        backend_kind.as_str(),
        backend_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<missing>".to_string()),
        report
            .backend_error
            .as_deref()
            .unwrap_or("no additional error")
    );
    if let Some(false) = report.source_complete {
        ensure!(false, "source split artifact is incomplete");
    }
    Ok(())
}

struct BackendReady {
    ready: bool,
    error: Option<String>,
}

fn check_backend_ready(backend_kind: BackendKind, backend_path: Option<&Path>) -> BackendReady {
    match backend_kind {
        BackendKind::ExternalProcess => {
            let ready = backend_path.is_some_and(Path::is_file);
            BackendReady {
                ready,
                error: (!ready)
                    .then(|| "backend executable path is missing or not a file".to_string()),
            }
        }
        BackendKind::LlamaApi | BackendKind::SkippyAbi => check_native_runtime_ready(backend_path),
        BackendKind::NativeRust => BackendReady {
            ready: true,
            error: None,
        },
    }
}

fn check_native_runtime_ready(backend_path: Option<&Path>) -> BackendReady {
    if skippy_ffi::native_runtime_loaded() {
        return BackendReady {
            ready: true,
            error: None,
        };
    }
    let Some(path) = backend_path else {
        return BackendReady {
            ready: false,
            error: Some("native runtime library path is missing".to_string()),
        };
    };
    if !path.is_file() {
        return BackendReady {
            ready: false,
            error: Some(format!(
                "native runtime library is not a file: {}",
                path.display()
            )),
        };
    }
    let libraries = [path.to_path_buf()];
    match unsafe { skippy_ffi::load_native_runtime_libraries(&libraries) } {
        Ok(()) => BackendReady {
            ready: true,
            error: None,
        },
        Err(error) => BackendReady {
            ready: false,
            error: Some(error.to_string()),
        },
    }
}

fn ensure_backend_supported(kind: JobKind, backend_kind: BackendKind) -> Result<()> {
    match kind {
        JobKind::ConvertHf => ensure_convert_backend(backend_kind),
        JobKind::QuantizeGguf => ensure_quant_backend(backend_kind),
    }
}

fn is_complete(progress: &Progress) -> bool {
    progress.complete
}

fn progress_windows(ranges: &[crate::splits::ShardRange]) -> Vec<ProgressWindow> {
    ranges
        .iter()
        .map(|range| ProgressWindow {
            first_split: range.first_split,
            last_split: range.last_split,
        })
        .collect()
}

fn print_preflight(report: &JobPreflight, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        println!(
            "preflight kind={:?} backend={} manifest_exists={} manifest_matches={} backend_ready={} target={}/{} first_missing={:?}",
            report.kind,
            report.backend_kind,
            report.manifest_exists,
            report.manifest_matches,
            report.backend_ready,
            report.target_shards,
            report.expected_target_shards,
            report.first_missing_target
        );
        if let Some(error) = report.backend_error.as_deref() {
            println!("backend_error={error}");
        }
        if let (Some(source_shards), Some(expected), Some(complete)) = (
            report.source_shards,
            report.expected_source_shards,
            report.source_complete,
        ) {
            println!("source={source_shards}/{expected} complete={complete}");
            if let Some(ranges) = report.source_missing_ranges.as_deref()
                && !ranges.is_empty()
            {
                println!("source_missing_ranges={}", format_ranges(ranges));
            }
        }
        if !report.target_missing_ranges.is_empty() {
            println!(
                "target_missing_ranges={}",
                format_ranges(&report.target_missing_ranges)
            );
        }
        if let Some(requested) = report.requested_window.as_ref() {
            println!(
                "requested_window={} next_requested_window={}",
                format_ranges(std::slice::from_ref(requested)),
                report
                    .next_requested_window
                    .as_ref()
                    .map(|window| format_ranges(std::slice::from_ref(window)))
                    .unwrap_or_else(|| "complete".to_string())
            );
        }
    }
    Ok(())
}

impl From<SplitWindow> for ProgressWindow {
    fn from(window: SplitWindow) -> Self {
        Self {
            first_split: window.first_split,
            last_split: window.last_split,
        }
    }
}

fn format_ranges(ranges: &[ProgressWindow]) -> String {
    ranges
        .iter()
        .map(|range| {
            if range.first_split == range.last_split {
                range.first_split.to_string()
            } else {
                format!("{}..{}", range.first_split, range.last_split)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_rust_backend_is_ready_without_path() {
        let ready = check_backend_ready(BackendKind::NativeRust, None);

        assert!(ready.ready);
        assert!(ready.error.is_none());
    }

    #[test]
    fn external_backend_requires_file_path() {
        let missing = check_backend_ready(BackendKind::ExternalProcess, None);
        let executable = std::env::current_exe().unwrap();
        let present = check_backend_ready(BackendKind::ExternalProcess, Some(&executable));

        assert!(!missing.ready);
        assert!(missing.error.unwrap().contains("missing"));
        assert!(present.ready);
    }

    #[test]
    fn native_runtime_backend_rejects_missing_or_invalid_library() {
        let missing = check_backend_ready(BackendKind::SkippyAbi, None);
        let executable = std::env::current_exe().unwrap();
        let invalid_library = check_backend_ready(BackendKind::SkippyAbi, Some(&executable));

        assert!(!missing.ready);
        assert!(missing.error.unwrap().contains("missing"));
        assert!(!invalid_library.ready);
        assert!(invalid_library.error.is_some());
    }
}
