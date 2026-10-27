//! Opt-in, payload-free timing records in a plugin-owned file (never stderr).
use std::{
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub struct Timing {
    path: Option<PathBuf>,
    phase: &'static str,
    operation: String,
    started: Instant,
}

impl Timing {
    pub fn start(phase: &'static str, operation: impl Into<String>) -> Self {
        Self::with_path(
            std::env::var_os("PAY_LIGHTNING_TIMING_LOG").map(PathBuf::from),
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
        let _ = writeln!(file, "{record}");
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
