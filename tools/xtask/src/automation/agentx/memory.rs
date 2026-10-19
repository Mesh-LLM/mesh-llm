//! Local benchmark memory observations. RSS and cgroup bytes are not GPU
//! residency on unified-memory systems; unavailable GPU counters stay absent.
use crate::command::DynResult;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
mod macos;

#[derive(Default, Serialize)]
struct Sample {
    elapsed_ms: u128,
    process_rss_bytes: Option<u64>,
    process_rss_high_water_bytes: Option<u64>,
    process_anonymous_rss_bytes: Option<u64>,
    process_physical_footprint_bytes: Option<u64>,
    container_memory_bytes: Option<u64>,
    host_available_bytes: Option<u64>,
}

#[derive(Default, Serialize)]
pub(super) struct Summary {
    samples: usize,
    sampled_peak_process_rss_bytes: Option<u64>,
    sampled_peak_process_physical_footprint_bytes: Option<u64>,
    process_rss_high_water_bytes: Option<u64>,
    sampled_peak_container_memory_bytes: Option<u64>,
    minimum_host_available_bytes: Option<u64>,
}

pub(super) struct Monitor {
    pub(super) pid: Arc<AtomicU32>,
    stop: mpsc::Sender<()>,
    worker: Option<JoinHandle<Result<Summary, String>>>,
}

impl Monitor {
    pub(super) fn start(output: &Path) -> DynResult<Self> {
        let file = File::create(output.join("memory.jsonl"))?;
        let pid = Arc::new(AtomicU32::new(0));
        let worker_pid = Arc::clone(&pid);
        let (stop, stopped) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("agentx-memory".into())
            .spawn(move || {
                sample_loop(file, &worker_pid, &stopped).map_err(|error| error.to_string())
            })?;
        Ok(Self {
            pid,
            stop,
            worker: Some(worker),
        })
    }

    pub(super) fn finish(mut self) -> DynResult<Summary> {
        let _ = self.stop.send(());
        self.worker
            .take()
            .ok_or("missing memory worker")?
            .join()
            .map_err(|_| "memory worker panicked")?
            .map_err(Into::into)
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(any(not(target_os = "macos"), test))]
fn kib_field(text: &str, name: &str) -> Option<u64> {
    let line = text.lines().find_map(|line| line.strip_prefix(name))?;
    let mut fields = line.split_whitespace();
    let value = fields.next()?.parse::<u64>().ok()?;
    (fields.next()? == "kB").then_some(value)?.checked_mul(1024)
}

fn cgroup_memory_path() -> Option<PathBuf> {
    // Docker's cgroup namespace normally mounts this container at the root.
    let root = PathBuf::from("/sys/fs/cgroup/memory.current");
    if root.is_file() {
        return Some(root);
    }
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = groups.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(
        Path::new("/sys/fs/cgroup")
            .join(relative.trim_start_matches('/'))
            .join("memory.current"),
    )
}

fn read_sample(pid: u32, elapsed_ms: u128, cgroup: Option<&Path>) -> Sample {
    #[cfg(target_os = "macos")]
    {
        let _ = cgroup;
        macos::sample(pid, elapsed_ms)
    }
    #[cfg(not(target_os = "macos"))]
    read_proc_sample(pid, elapsed_ms, cgroup)
}

#[cfg(not(target_os = "macos"))]
fn read_proc_sample(pid: u32, elapsed_ms: u128, cgroup: Option<&Path>) -> Sample {
    let status = if pid == 0 {
        String::new()
    } else {
        std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default()
    };
    let host = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    Sample {
        elapsed_ms,
        process_rss_bytes: kib_field(&status, "VmRSS:"),
        process_rss_high_water_bytes: kib_field(&status, "VmHWM:"),
        process_anonymous_rss_bytes: kib_field(&status, "RssAnon:"),
        process_physical_footprint_bytes: None,
        container_memory_bytes: cgroup
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|value| value.trim().parse().ok()),
        host_available_bytes: kib_field(&host, "MemAvailable:"),
    }
}

fn sample_loop(file: File, pid: &AtomicU32, stop: &mpsc::Receiver<()>) -> DynResult<Summary> {
    let mut writer = BufWriter::new(file);
    let mut summary = Summary::default();
    let started = Instant::now();
    let cgroup = cgroup_memory_path();
    loop {
        let sample = read_sample(
            pid.load(Ordering::Relaxed),
            started.elapsed().as_millis(),
            cgroup.as_deref(),
        );
        summary.samples += 1;
        summary.sampled_peak_process_rss_bytes = summary
            .sampled_peak_process_rss_bytes
            .max(sample.process_rss_bytes);
        summary.sampled_peak_process_physical_footprint_bytes = summary
            .sampled_peak_process_physical_footprint_bytes
            .max(sample.process_physical_footprint_bytes);
        summary.process_rss_high_water_bytes = summary
            .process_rss_high_water_bytes
            .max(sample.process_rss_high_water_bytes);
        summary.sampled_peak_container_memory_bytes = summary
            .sampled_peak_container_memory_bytes
            .max(sample.container_memory_bytes);
        if let Some(available) = sample.host_available_bytes {
            summary.minimum_host_available_bytes = Some(
                summary
                    .minimum_host_available_bytes
                    .map_or(available, |previous| previous.min(available)),
            );
        }
        serde_json::to_writer(&mut writer, &sample)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        match stop.recv_timeout(Duration::from_secs(1)) {
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proc_memory_requires_units_and_never_converts_missing_to_zero() {
        let status = "VmRSS:\t123 kB\nVmHWM:\t456 kB\nRssAnon:\t7 kB\n";
        assert_eq!(kib_field(status, "VmRSS:"), Some(123 * 1024));
        assert_eq!(kib_field(status, "VmHWM:"), Some(456 * 1024));
        assert_eq!(kib_field(status, "MemAvailable:"), None);
        assert_eq!(kib_field("VmRSS: 12 MB", "VmRSS:"), None);
        assert_eq!(kib_field("VmRSS: 18446744073709551615 kB", "VmRSS:"), None);
    }
}
