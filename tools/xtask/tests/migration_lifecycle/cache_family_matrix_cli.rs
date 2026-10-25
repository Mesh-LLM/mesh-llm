//! Actual native matrix frontend/plan child admission; no model/runtime fixture.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};
struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    input: PathBuf,
    output: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let input = root.join("input.json");
        let output = root.join("matrix");
        let value = json!({"schema_version":1,"plan":{"schema_version":1,"cache_root":root,"cases":["qwen3_dense"],"use_cases":[],"corpus":null,"prefix_sweep":[64,128],"prefix_tokens":null,"n_gpu_layers":null,"cache_hit_repeats":null,"runtime_lane_count":null,"serving_ctx_size":null,"concurrency":[1,2],"concurrent_requests":2,"concurrent_output_tokens":32,"llama_parallel":1,"llama_repeats":1,"ttft_slo_ms":20,"tpot_slo_ms":10,"skip_llama_server":false,"old_server":null,"new_server":null,"model_sha256":{}},"profiles":{},"execution_seconds":30,"cell_seconds":15,"request_timeout_ms":1000});
        fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
        Self {
            directory,
            root,
            input,
            output,
        }
    }
    fn run(&self) -> process::ProcessReport {
        let report = process::supervise(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                arguments: [
                    "automation".into(),
                    "cache-family-run".into(),
                    "--input".into(),
                    self.input.clone().into_os_string(),
                    "--output".into(),
                    self.output.clone().into_os_string(),
                ]
                .into_iter()
                .map(Arg::Public)
                .collect(),
                cwd: self.root.clone(),
                environment: BTreeMap::new(),
            },
            &Limits {
                execution: Duration::from_secs(35),
                graceful_shutdown: Duration::from_secs(15),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none()
        );
        assert!(
            [&report.stdout, &report.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        report
    }
    fn receipt(&self) -> Value {
        serde_json::from_slice(&fs::read(self.output.join("cache-family-matrix.json")).unwrap())
            .unwrap()
    }
}
#[test]
fn cache_matrix_actual_cli_retains_all_missing_model_prefix_rows_without_launch() {
    let fixture = Fixture::new();
    let report = fixture.run();
    assert_eq!(report.status.and_then(|s| s.code()), Some(0));
    let value = fixture.receipt();
    assert_eq!(value["status"], "completed");
    assert_eq!(value["planned_cells"], 2);
    assert!(value["promotion"].is_null());
    let rows = value["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["prefix_tokens"], 64);
    assert_eq!(rows[1]["prefix_tokens"], 128);
    for row in rows {
        assert_eq!(row["result"]["status"], "missing-model");
        assert_eq!(row["result"]["launched"], false);
    }
    assert!(!fixture.output.join("cell-0000").exists());
    fixture.directory.close().unwrap();
}
#[test]
fn cache_matrix_actual_cli_plan_refusal_publishes_failed_evidence_without_cell() {
    let fixture = Fixture::new();
    let mut input: Value = serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
    input["plan"]["concurrency"] = json!([0]);
    fs::write(&fixture.input, serde_json::to_vec(&input).unwrap()).unwrap();
    let report = fixture.run();
    assert_eq!(report.status.and_then(|s| s.code()), Some(1));
    assert_eq!(fixture.receipt()["status"], "failed");
    assert!(!fixture.output.join("cell-0000").exists());
    fixture.directory.close().unwrap();
}
#[test]
fn cache_matrix_actual_cli_existing_output_refuses_without_overwriting_or_plan_child() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.output).unwrap();
    fs::write(fixture.output.join("sentinel"), b"preserved").unwrap();
    let report = fixture.run();
    assert_eq!(report.status.and_then(|s| s.code()), Some(1));
    assert_eq!(
        fs::read(fixture.output.join("sentinel")).unwrap(),
        b"preserved"
    );
    assert!(!fixture.output.join("plan").exists());
    assert!(!fixture.output.join("cache-family-matrix.json").exists());
    fixture.directory.close().unwrap();
}
