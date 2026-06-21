use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::backend::BackendRunStatus;
use crate::output::print_path_event;
use crate::splits::SplitWindow;
use crate::types::JobKind;

#[derive(Debug, Serialize)]
struct WindowRunRecord {
    schema_version: u32,
    kind: JobKind,
    started_unix_ms: u128,
    duration_ms: u128,
    first_split: u32,
    last_split: u32,
    output_prefix: PathBuf,
    command: Vec<String>,
    status_code: Option<i32>,
    success: bool,
}

pub struct WindowRunRecordInput<'a> {
    pub schema_version: u32,
    pub kind: JobKind,
    pub command: &'a [String],
    pub output_prefix: &'a Path,
    pub window: SplitWindow,
    pub status: BackendRunStatus,
    pub duration_ms: u128,
    pub started_unix_ms: u128,
}

pub fn write_window_record(
    record_dir: Option<&Path>,
    input: WindowRunRecordInput<'_>,
) -> Result<()> {
    let Some(record_dir) = record_dir else {
        return Ok(());
    };
    fs::create_dir_all(record_dir).with_context(|| format!("create {}", record_dir.display()))?;
    let record = WindowRunRecord {
        schema_version: input.schema_version,
        kind: input.kind,
        started_unix_ms: input.started_unix_ms,
        duration_ms: input.duration_ms,
        first_split: input.window.first_split,
        last_split: input.window.last_split,
        output_prefix: input.output_prefix.to_path_buf(),
        command: input.command.to_vec(),
        status_code: input.status.status_code,
        success: input.status.success,
    };
    let name = format!(
        "{:?}-{:05}-{:05}-{}.json",
        input.kind, input.window.first_split, input.window.last_split, input.started_unix_ms
    )
    .to_lowercase();
    let path = record_dir.join(name);
    fs::write(&path, serde_json::to_vec_pretty(&record)?)
        .with_context(|| format!("write {}", path.display()))?;
    print_path_event("🧾", "Wrote window record", &path);
    Ok(())
}

pub fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use crate::manifest::MANIFEST_VERSION;
    use crate::splits::SplitWindow;
    use crate::types::JobKind;

    use super::*;

    #[test]
    fn writes_window_run_records() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-record-test-{}",
            unix_timestamp_ms()
        ));
        write_window_record(
            Some(&root),
            WindowRunRecordInput {
                schema_version: MANIFEST_VERSION,
                kind: JobKind::QuantizeGguf,
                command: &["llama-quantize".to_string()],
                output_prefix: Path::new("/target/Q2_K/out.gguf"),
                window: SplitWindow {
                    first_split: 3,
                    last_split: 4,
                },
                status: BackendRunStatus {
                    status_code: Some(0),
                    success: true,
                },
                duration_ms: 10,
                started_unix_ms: 1234,
            },
        )
        .unwrap();

        let entries = fs::read_dir(&root).unwrap().count();
        assert_eq!(entries, 1);
        fs::remove_dir_all(root).unwrap();
    }
}
