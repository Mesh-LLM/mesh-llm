//! Mesh terminal and dashboard presentation of Skippy package acquisition.
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use mesh_llm_events::terminal_progress::{
    SpinnerHandle, ratio_complete_u64, render_inline_gauge_with_reserved_width, start_spinner,
};
use mesh_llm_events::{ModelProgressStatus, OutputEvent, emit_event, interactive_tui_active};
use skippy_api::package::acquisition::progress::{PackageFileProgress, PackageProgress};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
pub struct MeshPackageProgress {
    scope: Option<Arc<LayerPackageDownloadScope>>,
}

impl PackageProgress for MeshPackageProgress {
    fn batch(&self, label: &str, total_files: usize) -> Arc<dyn PackageProgress> {
        Arc::new(Self {
            scope: Some(Arc::new(LayerPackageDownloadScope::new(label, total_files))),
        })
    }

    fn file(
        &self,
        label: &str,
        file: &str,
        total_bytes: Option<u64>,
        completed_before: usize,
    ) -> Arc<dyn PackageFileProgress> {
        Arc::new(LayerPackageDownloadProgress::new(
            label.to_string(),
            file.to_string(),
            total_bytes,
            self.scope.clone(),
            completed_before,
        ))
    }
}

impl PackageFileProgress for LayerPackageDownloadProgress {
    fn ensuring(&self) {
        self.emit_ensuring();
    }
    fn ready(&self, path: &Path) {
        self.emit_ready(path);
    }
}

#[derive(Clone, Debug)]
struct LayerPackageDownloadProgressState {
    downloaded: u64,
    total: u64,
    bytes_per_sec: Option<f64>,
    last_draw: Option<Instant>,
    showed_progress: bool,
}

struct LayerPackageDownloadProgress {
    label: String,
    file: String,
    package_scope: Option<Arc<LayerPackageDownloadScope>>,
    completed_before: usize,
    preflight_spinner: Mutex<Option<SpinnerHandle>>,
    state: Mutex<LayerPackageDownloadProgressState>,
}

struct LayerPackageDownloadScope {
    package: String,
    total_files: usize,
    state: Mutex<LayerPackageDownloadScopeState>,
}

#[derive(Debug)]
struct LayerPackageDownloadScopeState {
    announced: bool,
    drawn_line: bool,
}

impl LayerPackageDownloadScope {
    fn new(label: &str, total_files: usize) -> Self {
        Self {
            package: layer_package_progress_package(label).to_string(),
            total_files,
            state: Mutex::new(LayerPackageDownloadScopeState {
                announced: false,
                drawn_line: false,
            }),
        }
    }

    fn has_drawn(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.announced || state.drawn_line)
            .unwrap_or(false)
    }

    fn complete_count(&self, completed: usize) -> usize {
        completed.min(self.total_files)
    }

    fn draw(
        &self,
        file: &str,
        completed_files: usize,
        downloaded: u64,
        total: u64,
        bytes_per_sec: Option<f64>,
        force: bool,
    ) {
        let mut err = mesh_llm_events::console_err();
        let Ok(mut scope_state) = self.state.lock() else {
            return;
        };
        if !scope_state.announced {
            let _ = writeln!(
                err,
                "\r\x1b[K📦 Downloading layer package {} ({} file(s))",
                self.package, self.total_files
            );
            scope_state.announced = true;
        }
        let percent = if total == 0 {
            0
        } else {
            ((downloaded as f64 / total as f64) * 1000.0).round() as usize
        };
        let percent_major = (percent.min(1000)) / 10;
        let percent_minor = (percent.min(1000)) % 10;
        let speed_suffix = bytes_per_sec
            .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
            .map(|bytes_per_sec| {
                format!(
                    " at {}/s",
                    format_layer_package_download_bytes(bytes_per_sec as u64)
                )
            })
            .unwrap_or_default();
        let (ratio, total_display) = match total {
            0 => (0.0, "?".to_string()),
            total => (
                ratio_complete_u64(downloaded, total),
                format_layer_package_download_bytes(total),
            ),
        };
        let gauge = render_inline_gauge_with_reserved_width(
            ratio,
            &format!(
                "⏬ {} {:>3}.{:01}% ({}/{}){}   files {}/{} complete",
                layer_package_artifact_display_for_package(&self.package, file),
                percent_major,
                percent_minor,
                format_layer_package_download_bytes(downloaded),
                total_display,
                speed_suffix,
                self.complete_count(completed_files),
                self.total_files,
            ),
            3,
        );
        let _ = write!(err, "\r\x1b[K   {gauge}");
        let _ = err.flush();
        scope_state.drawn_line = true;
        if force {
            let _ = writeln!(err);
            scope_state.drawn_line = false;
        }
    }
}

impl LayerPackageDownloadProgress {
    fn new(
        label: String,
        file: String,
        total_bytes: Option<u64>,
        package_scope: Option<Arc<LayerPackageDownloadScope>>,
        completed_before: usize,
    ) -> Self {
        let preflight_spinner = if interactive_tui_active()
            || package_scope
                .as_ref()
                .is_some_and(|scope| scope.has_drawn())
        {
            None
        } else {
            Some(start_spinner(&format!("Preparing download {file}")))
        };
        Self {
            label,
            file,
            package_scope,
            completed_before,
            preflight_spinner: Mutex::new(preflight_spinner),
            state: Mutex::new(LayerPackageDownloadProgressState {
                downloaded: 0,
                total: total_bytes.unwrap_or(0),
                bytes_per_sec: None,
                last_draw: None,
                showed_progress: false,
            }),
        }
    }

    fn emit(
        &self,
        downloaded_bytes: Option<u64>,
        total_bytes: Option<u64>,
        status: ModelProgressStatus,
    ) {
        let _ = emit_event(OutputEvent::ModelDownloadProgress {
            label: self.label.clone(),
            file: Some(self.file.clone()),
            downloaded_bytes,
            total_bytes,
            status,
        });
    }

    fn emit_ensuring(&self) {
        if !interactive_tui_active() {
            return;
        }
        let total = self
            .state
            .lock()
            .ok()
            .and_then(|state| (state.total > 0).then_some(state.total));
        self.emit(None, total, ModelProgressStatus::Ensuring);
    }

    fn emit_ready(&self, path: &Path) {
        let mut err = mesh_llm_events::console_err();
        let total = fs::metadata(path)
            .ok()
            .map(|metadata| metadata.len())
            .or_else(|| {
                self.state
                    .lock()
                    .ok()
                    .and_then(|state| (state.total > 0).then_some(state.total))
            });
        if interactive_tui_active() {
            self.emit(total, total, ModelProgressStatus::Ready);
            return;
        }
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
        let showed_progress = self
            .state
            .lock()
            .map(|state| state.showed_progress)
            .unwrap_or(false);
        if let Some(scope) = &self.package_scope {
            if !showed_progress {
                let total = total.unwrap_or(0);
                scope.draw(
                    &self.file,
                    self.completed_before + 1,
                    total,
                    total,
                    None,
                    true,
                );
            }
            return;
        }
        if !showed_progress {
            let file = layer_package_artifact_display(&self.label, &self.file);
            match total {
                Some(total) if total > 0 => {
                    let _ = writeln!(
                        err,
                        "   ✅ Ready {} ({})",
                        file,
                        format_layer_package_download_bytes(total)
                    );
                }
                _ => {
                    let _ = writeln!(err, "   ✅ Ready {}", file);
                }
            }
        }
    }

    fn draw(&self, state: &mut LayerPackageDownloadProgressState, force: bool) {
        if !force && state.downloaded == 0 && state.total == 0 {
            return;
        }
        let now = Instant::now();
        if !force
            && state
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(150))
        {
            return;
        }
        state.last_draw = Some(now);
        state.showed_progress = true;
        if interactive_tui_active() {
            self.emit(
                (state.downloaded > 0).then_some(state.downloaded),
                (state.total > 0).then_some(state.total),
                ModelProgressStatus::Downloading,
            );
            return;
        }
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
        if let Some(scope) = &self.package_scope {
            let completed = if force {
                self.completed_before + 1
            } else {
                self.completed_before
            };
            scope.draw(
                &self.file,
                completed,
                state.downloaded,
                state.total,
                state.bytes_per_sec,
                force,
            );
        } else {
            draw_layer_package_file_progress(
                &layer_package_artifact_display(&self.label, &self.file),
                state.downloaded,
                state.total,
                state.bytes_per_sec,
                force,
            );
        }
    }
}

impl Drop for LayerPackageDownloadProgress {
    fn drop(&mut self) {
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
    }
}

impl ProgressHandler for LayerPackageDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                if *total_bytes > 0 {
                    state.total = state.total.max(*total_bytes);
                }
            }
            DownloadEvent::Progress { files } => {
                if !files.is_empty() {
                    let downloaded: u64 = files.iter().map(|file| file.bytes_completed).sum();
                    state.downloaded = state.downloaded.max(downloaded);
                    let total: u64 = files.iter().map(|file| file.total_bytes).sum();
                    if total > 0 {
                        state.total = state.total.max(total);
                    }
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            } => {
                state.downloaded = state.downloaded.max(*bytes_completed);
                if *total_bytes > 0 {
                    state.total = state.total.max(*total_bytes);
                }
                state.bytes_per_sec = *bytes_per_sec;
            }
            DownloadEvent::Complete => {
                if state.total > 0 {
                    state.downloaded = state.total;
                }
                state.bytes_per_sec = None;
            }
        }
        let should_show_progress = state.downloaded > 0 || state.total > 0;
        let force = matches!(event, DownloadEvent::Complete) && should_show_progress;
        if should_show_progress {
            self.draw(&mut state, force);
        } else if matches!(event, DownloadEvent::Complete)
            && let Ok(mut spinner) = self.preflight_spinner.lock()
        {
            spinner.take();
        }
    }
}

fn format_layer_package_download_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}

fn layer_package_progress_package(label: &str) -> &str {
    label.strip_prefix("layer package ").unwrap_or(label)
}

fn layer_package_progress_repo(package: &str) -> &str {
    package
        .split_once('@')
        .map(|(repo, _)| repo)
        .unwrap_or(package)
}

fn layer_package_artifact_display(label: &str, file: &str) -> String {
    layer_package_artifact_display_for_package(layer_package_progress_package(label), file)
}

fn layer_package_artifact_display_for_package(package: &str, file: &str) -> String {
    let repo = layer_package_progress_repo(package);
    if file.starts_with(repo) || file.starts_with('/') {
        file.to_string()
    } else {
        format!("{repo}/{file}")
    }
}

fn draw_layer_package_file_progress(
    file: &str,
    downloaded: u64,
    total: u64,
    bytes_per_sec: Option<f64>,
    force: bool,
) {
    let mut err = mesh_llm_events::console_err();
    let percent = if total == 0 {
        0
    } else {
        ((downloaded as f64 / total as f64) * 1000.0).round() as usize
    };
    let percent_major = (percent.min(1000)) / 10;
    let percent_minor = (percent.min(1000)) % 10;
    let speed_suffix = bytes_per_sec
        .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
        .map(|bytes_per_sec| {
            format!(
                " at {}/s",
                format_layer_package_download_bytes(bytes_per_sec as u64)
            )
        })
        .unwrap_or_default();
    let (ratio, total_display) = match total {
        0 => (0.0, "?".to_string()),
        total => (
            ratio_complete_u64(downloaded, total),
            format_layer_package_download_bytes(total),
        ),
    };
    let gauge = render_inline_gauge_with_reserved_width(
        ratio,
        &format!(
            "⏬ {} {:>3}.{:01}% ({}/{}){}",
            file,
            percent_major,
            percent_minor,
            format_layer_package_download_bytes(downloaded),
            total_display,
            speed_suffix,
        ),
        3,
    );
    let _ = write!(err, "\r\x1b[K   {gauge}");
    let _ = err.flush();
    if force {
        let _ = writeln!(err);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_package_artifact_display_names_repo_and_file_without_revision() {
        assert_eq!(
            layer_package_artifact_display(
                "layer package meshllm/demo-package@abc123",
                "layers/layer-005.gguf"
            ),
            "meshllm/demo-package/layers/layer-005.gguf"
        );
        assert_eq!(
            layer_package_artifact_display(
                "layer package meshllm/demo-package",
                "model-package.json"
            ),
            "meshllm/demo-package/model-package.json"
        );
    }
}
