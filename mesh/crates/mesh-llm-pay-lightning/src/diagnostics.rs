//! Opt-in, payload-free timing records in a plugin-owned file (never stderr).
use std::{
    io::Write,
    path::PathBuf,
    sync::Mutex,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

// One plugin process owns its log path. Serialize appenders even if write_all
// needs multiple OS writes; separate processes must use separate paths.
static LOG_PATHS: std::sync::OnceLock<(Option<PathBuf>, Option<PathBuf>)> =
    std::sync::OnceLock::new();
pub fn configure(timing: Option<PathBuf>, error: Option<PathBuf>) -> anyhow::Result<()> {
    LOG_PATHS
        .set((timing, error))
        .map_err(|_| anyhow::anyhow!("payment runtime already configured"))
}
pub fn record_error(error: &anyhow::Error) {
    let Ok(_guard) = RECORD_WRITE.lock() else {
        return;
    };
    if let Some(path) = LOG_PATHS.get().and_then(|paths| paths.1.as_ref())
        && let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    {
        let _ = writeln!(file, "{}", serde_json::json!({"error":error.to_string()}));
    }
}
static RECORD_WRITE: Mutex<()> = Mutex::new(());

pub struct Timing {
    path: Option<PathBuf>,
    phase: &'static str,
    operation: String,
    started: Instant,
}

impl Timing {
    pub fn start(phase: &'static str, operation: impl Into<String>) -> Self {
        Self::with_path(
            LOG_PATHS.get().and_then(|paths| paths.0.clone()),
            phase,
            operation.into(),
        )
    }

    fn with_path(path: Option<PathBuf>, phase: &'static str, operation: String) -> Self {
        let timing = Self {
            path,
            phase,
            operation,
            started: Instant::now(),
        };
        timing.record("begin");
        timing
    }

    fn record(&self, boundary: &str) {
        let Some(path) = &self.path else {
            return;
        };
        // Diagnostics must not affect payment results. No invoices, hashes,
        // peer identities, wallet responses, or errors are serialized here.
        let Ok(_guard) = RECORD_WRITE.lock() else {
            return;
        };
        let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        else {
            return;
        };
        let record = serde_json::json!({
            "timestamp_us": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros(),
            "pid": std::process::id(), "phase": self.phase,
            "operation": self.operation, "boundary": boundary,
            "elapsed_us": self.started.elapsed().as_micros(),
        });
        let mut line = record.to_string().into_bytes();
        line.push(b'\n');
        let _ = file.write_all(&line);
    }
}

impl Drop for Timing {
    fn drop(&mut self) {
        self.record("end");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_records_remain_complete_json_lines() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("concurrent.jsonl");
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let path = &path;
                scope.spawn(move || {
                    for record in 0..100 {
                        let _timing = Timing::with_path(
                            Some(path.clone()),
                            "test",
                            format!("{worker}-{record}"),
                        );
                    }
                });
            }
        });
        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.ends_with('\n'));
        let mut boundaries = std::collections::HashMap::new();
        for line in contents.lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            let operation = record["operation"].as_str().unwrap().to_owned();
            boundaries
                .entry(operation)
                .or_insert_with(Vec::new)
                .push(record["boundary"].as_str().unwrap().to_owned());
        }
        assert_eq!(boundaries.len(), 800);
        for values in boundaries.values() {
            assert_eq!(values, &["begin", "end"]);
        }
    }

    #[test]
    fn records_boundaries_without_environment_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("timing.jsonl");
        {
            let _timing = Timing::with_path(Some(path.clone()), "wallet", "open".into());
        }
        let records: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["boundary"], "begin");
        assert_eq!(records[1]["boundary"], "end");
        assert!(records[1]["elapsed_us"].as_u64().is_some());
        let _disabled = Timing::with_path(None, "wallet", "open".into());
        let _unwritable = Timing::with_path(Some(directory.path().into()), "wallet", "open".into());
    }
}
