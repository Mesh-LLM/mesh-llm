use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, Sender},
};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::splits::{ShardRange, SplitWindow};

const BAR_WIDTH: usize = 24;
const JSON_EVENT_SCHEMA_VERSION: u32 = 1;

pub(crate) fn print_json_pretty(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub(crate) fn progress_bar(completed: usize, expected: u32) -> String {
    let expected = expected as usize;
    let filled = completed
        .saturating_mul(BAR_WIDTH)
        .checked_div(expected)
        .unwrap_or_default()
        .min(BAR_WIDTH);
    format!(
        "[{}{}]",
        "█".repeat(filled),
        "░".repeat(BAR_WIDTH.saturating_sub(filled))
    )
}

pub(crate) fn percent(completed: usize, expected: u32) -> f64 {
    if expected == 0 {
        0.0
    } else {
        (completed as f64 / f64::from(expected)) * 100.0
    }
}

pub(crate) fn print_progress_line(label: &str, completed: usize, expected: u32) {
    println!(
        "📊 {label}: {} {completed}/{expected} shards ({:.2}%)",
        progress_bar(completed, expected),
        percent(completed, expected)
    );
}

pub(crate) fn print_success(message: impl AsRef<str>) {
    println!("✅ {}", message.as_ref());
}

pub(crate) fn print_info(message: impl AsRef<str>) {
    println!("ℹ️  {}", message.as_ref());
}

pub(crate) fn print_warn(message: impl AsRef<str>) {
    println!("⚠️  {}", message.as_ref());
}

pub(crate) fn print_copy(source: &Path, target: &Path, size_bytes: Option<u64>) {
    match size_bytes {
        Some(size_bytes) => println!(
            "📤 Copying {} -> {} ({})",
            source.display(),
            target.display(),
            format_bytes(size_bytes)
        ),
        None => println!("📤 Copying {} -> {}", source.display(), target.display()),
    }
}

pub(crate) fn print_path_event(emoji: &str, label: &str, path: &Path) {
    println!("{emoji} {label}: {}", path.display());
}

pub(crate) fn print_window(label: &str, window: SplitWindow) {
    println!("🪟 {label}: {}", format_window(window));
}

pub(crate) fn format_window(window: SplitWindow) -> String {
    if window.first_split == window.last_split {
        window.first_split.to_string()
    } else {
        format!("{}..{}", window.first_split, window.last_split)
    }
}

pub(crate) fn format_shard_ranges(ranges: &[ShardRange]) -> String {
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
        .join(", ")
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.2} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.2} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[derive(Debug, Clone)]
pub(crate) struct JsonEventConfig {
    pub(crate) file: Option<PathBuf>,
    pub(crate) interval_seconds: u64,
    pub(crate) window_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonEventSnapshot {
    schema_version: u32,
    event: String,
    kind: String,
    phase: String,
    started_unix_ms: u128,
    updated_unix_ms: u128,
    interval_seconds: u64,
    window_size: usize,
    current_window: Option<SplitWindow>,
    recent_events: Vec<JsonRecentEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRecentEvent {
    unix_ms: u128,
    message: String,
}

#[derive(Debug)]
struct JsonEventState {
    kind: String,
    phase: String,
    started_unix_ms: u128,
    updated_unix_ms: u128,
    interval_seconds: u64,
    window_size: usize,
    current_window: Option<SplitWindow>,
    recent_events: VecDeque<JsonRecentEvent>,
}

impl JsonEventState {
    fn snapshot(&self) -> JsonEventSnapshot {
        JsonEventSnapshot {
            schema_version: JSON_EVENT_SCHEMA_VERSION,
            event: "skippy_quantize_periodic_status".to_string(),
            kind: self.kind.clone(),
            phase: self.phase.clone(),
            started_unix_ms: self.started_unix_ms,
            updated_unix_ms: self.updated_unix_ms,
            interval_seconds: self.interval_seconds,
            window_size: self.window_size,
            current_window: self.current_window,
            recent_events: self.recent_events.iter().cloned().collect(),
        }
    }
}

pub(crate) struct JsonEventReporter {
    path: Option<PathBuf>,
    state: Option<Arc<Mutex<JsonEventState>>>,
    stop: Option<Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl JsonEventReporter {
    pub(crate) fn start(
        config: JsonEventConfig,
        kind: impl Into<String>,
        window: Option<SplitWindow>,
    ) -> Result<Self> {
        let Some(path) = config.file else {
            return Ok(Self::disabled());
        };
        let interval_seconds = config.interval_seconds.max(1);
        let window_size = config.window_size.max(1);
        let now = unix_timestamp_ms();
        let state = Arc::new(Mutex::new(JsonEventState {
            kind: kind.into(),
            phase: "starting".to_string(),
            started_unix_ms: now,
            updated_unix_ms: now,
            interval_seconds,
            window_size,
            current_window: window,
            recent_events: VecDeque::new(),
        }));
        write_snapshot(&path, &state)?;
        let (stop, thread) = spawn_periodic_writer(path.clone(), Arc::clone(&state));
        Ok(Self {
            path: Some(path),
            state: Some(state),
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    fn disabled() -> Self {
        Self {
            path: None,
            state: None,
            stop: None,
            thread: None,
        }
    }

    pub(crate) fn record(&self, message: impl Into<String>) -> Result<()> {
        self.with_state(|state| {
            state.updated_unix_ms = unix_timestamp_ms();
            state.recent_events.push_back(JsonRecentEvent {
                unix_ms: state.updated_unix_ms,
                message: message.into(),
            });
            while state.recent_events.len() > state.window_size {
                state.recent_events.pop_front();
            }
        })
    }

    pub(crate) fn set_phase(&self, phase: impl Into<String>) -> Result<()> {
        self.with_state(|state| {
            state.updated_unix_ms = unix_timestamp_ms();
            state.phase = phase.into();
        })
    }

    pub(crate) fn finish(mut self, phase: impl Into<String>) -> Result<()> {
        self.set_phase(phase)?;
        self.write_now()?;
        self.stop_thread();
        Ok(())
    }

    pub(crate) fn write_now(&self) -> Result<()> {
        if let (Some(path), Some(state)) = (&self.path, &self.state) {
            write_snapshot(path, state)?;
        }
        Ok(())
    }

    fn with_state(&self, update: impl FnOnce(&mut JsonEventState)) -> Result<()> {
        if let Some(state) = &self.state {
            let mut state = state
                .lock()
                .map_err(|_| anyhow::anyhow!("json event state lock poisoned"))?;
            update(&mut state);
        }
        Ok(())
    }

    fn stop_thread(&mut self) {
        if let Some(stop) = &self.stop {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for JsonEventReporter {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

fn spawn_periodic_writer(
    path: PathBuf,
    state: Arc<Mutex<JsonEventState>>,
) -> (Sender<()>, thread::JoinHandle<()>) {
    let (stop_tx, stop_rx) = mpsc::channel();
    let interval = state
        .lock()
        .map(|state| Duration::from_secs(state.interval_seconds))
        .unwrap_or_else(|_| Duration::from_secs(120));
    let thread = thread::spawn(move || {
        loop {
            match stop_rx.recv_timeout(interval) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if let Ok(mut state) = state.lock() {
                state.updated_unix_ms = unix_timestamp_ms();
            }
            let _ = write_snapshot(&path, &state);
        }
    });
    (stop_tx, thread)
}

fn write_snapshot(path: &Path, state: &Arc<Mutex<JsonEventState>>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let snapshot = {
        let state = state
            .lock()
            .map_err(|_| anyhow::anyhow!("json event state lock poisoned"))?;
        state.snapshot()
    };
    let temp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    fs::write(&temp, serde_json::to_vec_pretty(&snapshot)?)
        .with_context(|| format!("write {}", temp.display()))?;
    fs::rename(&temp, path).with_context(|| {
        format!(
            "replace json event snapshot {} with {}",
            path.display(),
            temp.display()
        )
    })?;
    Ok(())
}

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_event_snapshot_keeps_bounded_recent_window() {
        let root = unique_temp_dir();
        let path = root.join("events").join("status.json");
        let reporter = JsonEventReporter::start(
            JsonEventConfig {
                file: Some(path.clone()),
                interval_seconds: 60,
                window_size: 2,
            },
            "quant",
            Some(SplitWindow {
                first_split: 3,
                last_split: 4,
            }),
        )
        .unwrap();

        reporter.set_phase("running").unwrap();
        reporter.record("first").unwrap();
        reporter.record("second").unwrap();
        reporter.record("third").unwrap();
        reporter.write_now().unwrap();
        reporter.finish("complete").unwrap();

        let snapshot: JsonEventSnapshot =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(snapshot.phase, "complete");
        assert_eq!(snapshot.recent_events.len(), 2);
        assert_eq!(snapshot.recent_events[0].message, "second");
        assert_eq!(snapshot.recent_events[1].message, "third");
        assert_eq!(snapshot.current_window.unwrap().first_split, 3);
        fs::remove_dir_all(root).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("skippy-quantize-json-events-{nanos}-{id}"))
    }
}
