use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::backend::{BackendKind, ensure_convert_backend, ensure_quant_backend};
use crate::manifest::{Manifest, manifest_progress, read_manifest};
use crate::output::{
    print_info, print_json_pretty, print_progress_line, print_success, print_warn,
};
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
    backend_paths: Vec<PathBuf>,
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
    backend_paths: &[PathBuf],
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
    let backend_check = check_backend_ready(backend_kind, backend_paths);
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
        backend_paths: backend_paths.to_vec(),
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
        if backend_paths.is_empty() {
            "<missing>".to_string()
        } else {
            backend_paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        },
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

fn check_backend_ready(backend_kind: BackendKind, backend_paths: &[PathBuf]) -> BackendReady {
    match backend_kind {
        BackendKind::SkippyAbi => check_skippy_runtime_ready(backend_paths),
        BackendKind::NativeRust => BackendReady {
            ready: true,
            error: None,
        },
    }
}

fn check_skippy_runtime_ready(backend_paths: &[PathBuf]) -> BackendReady {
    if skippy_ffi::native_runtime_loaded() {
        return BackendReady {
            ready: true,
            error: None,
        };
    }
    if backend_paths.is_empty() {
        return BackendReady {
            ready: false,
            error: Some("native runtime library path is missing".to_string()),
        };
    }
    if let Some(path) = backend_paths.iter().find(|path| !path.is_file()) {
        return BackendReady {
            ready: false,
            error: Some(format!(
                "native runtime library is not a file: {}",
                path.display()
            )),
        };
    }
    match unsafe { skippy_ffi::load_native_runtime_libraries(backend_paths) } {
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
        print_json_pretty(report)?;
    } else {
        print_info(format!(
            "Preflight {:?} with backend {}",
            report.kind, report.backend_kind
        ));
        print_progress_line(
            "target",
            report.target_shards,
            report.expected_target_shards,
        );
        if report.manifest_matches {
            print_success("Manifest is compatible");
        } else {
            print_warn("Existing manifest does not match requested job");
        }
        if report.backend_ready {
            print_success("Backend is ready");
        } else {
            print_warn("Backend is not ready");
        }
        if let Some(error) = report.backend_error.as_deref() {
            print_warn(format!("Backend error: {error}"));
        }
        if let (Some(source_shards), Some(expected), Some(complete)) = (
            report.source_shards,
            report.expected_source_shards,
            report.source_complete,
        ) {
            print_progress_line("source", source_shards, expected);
            if complete {
                print_success("Source artifact is complete");
            } else {
                print_warn("Source artifact is incomplete");
            }
            if let Some(ranges) = report.source_missing_ranges.as_deref()
                && !ranges.is_empty()
            {
                print_info(format!("Source missing ranges: {}", format_ranges(ranges)));
            }
        }
        if !report.target_missing_ranges.is_empty() {
            print_info(format!(
                "Target missing ranges: {}",
                format_ranges(&report.target_missing_ranges)
            ));
        }
        if let Some(requested) = report.requested_window.as_ref() {
            print_info(format!(
                "Requested window: {}; next requested window: {}",
                format_ranges(std::slice::from_ref(requested)),
                report
                    .next_requested_window
                    .as_ref()
                    .map(|window| format_ranges(std::slice::from_ref(window)))
                    .unwrap_or_else(|| "complete".to_string())
            ));
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
        let ready = check_backend_ready(BackendKind::NativeRust, &[]);

        assert!(ready.ready);
        assert!(ready.error.is_none());
    }

    #[test]
    #[cfg(feature = "dynamic-skippy-runtime")]
    fn native_runtime_backend_rejects_missing_or_invalid_library() {
        let missing = check_backend_ready(BackendKind::SkippyAbi, &[]);
        let executable = std::env::current_exe().unwrap();
        let invalid_library = check_backend_ready(BackendKind::SkippyAbi, &[executable]);

        assert!(!missing.ready);
        assert!(missing.error.unwrap().contains("missing"));
        assert!(!invalid_library.ready);
        assert!(invalid_library.error.is_some());
    }

    #[test]
    #[cfg(not(feature = "dynamic-skippy-runtime"))]
    fn linked_native_runtime_is_ready_without_path() {
        let ready = check_backend_ready(BackendKind::SkippyAbi, &[]);
        assert!(ready.ready);
        assert!(ready.error.is_none());
    }
}
