//! Actual matrix frontend across supplied MiniMax shards/DeepSeek package protocols.
use super::cache_family_full_matrix_cli::Fixture;
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};
fn setup(f: &Fixture, package: bool) -> PathBuf {
    let artifact = super::cache_family_artifact_cli::input(f, package);
    let mut correctness: Value = serde_json::from_slice(&fs::read(&artifact).unwrap()).unwrap();
    let mut input: Value = serde_json::from_slice(&fs::read(&f.input).unwrap()).unwrap();
    let key = if package { "deepseek3" } else { "minimax_m27" };
    let catalog: Value = serde_json::from_str(include_str!(
        "../../src/automation/cache_family_plan/catalog.json"
    ))
    .unwrap();
    let case = catalog
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["key"] == key)
        .unwrap();
    let requested = PathBuf::from(input["plan"]["cache_root"].as_str().unwrap())
        .join(case["snapshot_relative"].as_str().unwrap());
    let previous = PathBuf::from(correctness["model"].as_str().unwrap());
    let source_dir = if package {
        previous.clone()
    } else {
        previous.parent().unwrap().to_owned()
    };
    let destination = if package {
        requested.clone()
    } else {
        requested.parent().unwrap().to_owned()
    };
    fs::create_dir_all(&destination).unwrap();
    for file in fs::read_dir(source_dir).unwrap() {
        let file = file.unwrap();
        fs::copy(file.path(), destination.join(file.file_name())).unwrap();
    }
    correctness["model"] = json!(requested);
    let old_model = input["profiles"]["qwen3_dense"]["correctness"]["model"]
        .as_str()
        .unwrap()
        .to_owned();
    for side in ["native", "old", "new"] {
        let path = f.root.join(side);
        let body = fs::read_to_string(&path)
            .unwrap()
            .replace(&old_model, requested.to_str().unwrap());
        fs::write(&path, &body).unwrap();
        input["profiles"]["qwen3_dense"][side]["binary_sha256"] =
            json!(hex::encode(Sha256::digest(body.as_bytes())));
        if side != "native" {
            input["plan"][if side == "old" {
                "old_server"
            } else {
                "new_server"
            }]["sha256"] = json!(hex::encode(Sha256::digest(body.as_bytes())));
        }
    }
    correctness["stage_server_sha256"] = input["plan"]["new_server"]["sha256"].clone();
    fs::write(&artifact, serde_json::to_vec(&correctness).unwrap()).unwrap();
    let mut profile = input["profiles"]["qwen3_dense"].clone();
    profile["correctness"] = correctness.clone();
    for side in ["native", "old", "new"] {
        if package {
            profile[side] = Value::Null;
            continue;
        }
        profile[side]["model"] = correctness["model"].clone();
        profile[side]["model_sha256"] = correctness["model_sha256"].clone();
        profile[side]["model_id"] = correctness["model_id"].clone();
        profile[side]["layer_end"] = json!(62);
        profile[side]["artifact"] = correctness["artifact"].clone();
    }
    input["profiles"] = json!({(key):profile});
    input["plan"]["cases"] = json!([key]);
    input["plan"]["prefix_sweep"] = json!([]);
    input["plan"]["model_sha256"] = json!({(key):correctness["model_sha256"]});
    input["execution_seconds"] = json!(180);
    input["cell_seconds"] = json!(40);
    let path = f.root.join("artifact-matrix-input.json");
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    path
}
fn spec(f: &Fixture, path: PathBuf) -> ProcessSpec {
    ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: [
            "automation".into(),
            "cache-family-run".into(),
            "--input".into(),
            path.into_os_string(),
            "--output".into(),
            f.root.join("artifact-matrix").into_os_string(),
        ]
        .into_iter()
        .map(Arg::Public)
        .collect(),
        cwd: f.root.clone(),
        environment: BTreeMap::new(),
    }
}
fn run(spec: ProcessSpec, cancel: Cancellation) -> process::ProcessReport {
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(190),
            graceful_shutdown: Duration::from_secs(18),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &cancel,
        OutputFiles::default(),
    )
    .unwrap();
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
#[test]
fn artifact_matrix_actual_cli_completes_minimax_serving_and_deepseek_package_report_without_baseline()
 {
    for package in [false, true] {
        let f = Fixture::new();
        let input = setup(&f, package);
        let r = run(spec(&f, input), Cancellation::default());
        assert_eq!(r.outcome, process::Outcome::Exited);
        assert_eq!(r.status.and_then(|s| s.code()), Some(0));
        let summary: Value = serde_json::from_slice(
            &fs::read(f.root.join("artifact-matrix/cache-family-matrix.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["status"], "completed");
        let row: Value = serde_json::from_slice(
            &fs::read(f.root.join("artifact-matrix/cell-0000/row.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(row["status"], "completed");
        let observed = row["observations"].as_array().unwrap();
        if package {
            assert_eq!(observed.len(), 1);
            assert_eq!(observed[0]["status"], "unavailable");
            assert!(!f.root.join("post-native").exists());
            assert!(!f.root.join("post-old").exists());
        } else {
            assert_eq!(observed.len(), 5);
            assert_eq!(observed[4]["parity"]["matches"], true);
            assert!(f.root.join("post-native").exists());
            assert!(f.root.join("post-new").exists());
        }
        let report: Value = serde_json::from_slice(
            &fs::read(f.root.join("artifact-matrix/production-cache-bench.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report[0]["skippy"]["status"], "pass");
        if package {
            assert_eq!(report[0]["llama_server"]["status"], "unavailable");
        }
        assert!(
            f.root
                .join("artifact-matrix/production-cache-bench.md")
                .is_file()
        );
        f.directory.close().unwrap();
    }
}
#[test]
fn artifact_matrix_actual_cli_late_failure_preserves_old_measurement_and_refuses_parity() {
    let f = Fixture::new();
    let input = setup(&f, false);
    fs::write(f.root.join("mode"), b"later-failure").unwrap();
    let r = run(spec(&f, input), Cancellation::default());
    assert_eq!(r.status.and_then(|s| s.code()), Some(1));
    let row: Value = serde_json::from_slice(
        &fs::read(f.root.join("artifact-matrix/cell-0000/row.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(row["status"], "incomplete");
    assert_eq!(row["observations"][2]["status"], "completed");
    assert_ne!(row["observations"][3]["status"], "completed");
    assert!(
        f.root
            .join("artifact-matrix/production-cache-bench.md")
            .is_file()
    );
    f.directory.close().unwrap();
}
#[test]
fn artifact_matrix_actual_cli_causal_new_arm_cancel_preserves_prior_receipts_and_report() {
    let f = Fixture::new();
    let input = setup(&f, false);
    fs::write(f.root.join("mode"), b"hold").unwrap();
    let c = Cancellation::default();
    let child = c.clone();
    let spec = spec(&f, input);
    let task = thread::spawn(move || run(spec, child));
    let until = Instant::now() + Duration::from_secs(45);
    while !f.root.join("post-new").exists() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    let marker = f.root.join("post-new").exists();
    c.cancel();
    let r = task.join().unwrap();
    assert!(marker);
    assert_eq!(r.outcome, process::Outcome::Cancelled);
    assert!(
        f.root
            .join("artifact-matrix/cell-0000/sweep-skippy-old/output/cell.json")
            .is_file()
    );
    assert!(
        f.root
            .join("artifact-matrix/production-cache-bench.md")
            .is_file()
    );
    let receipt: Value = serde_json::from_slice(
        &fs::read(f.root.join("artifact-matrix/cache-family-matrix.json")).unwrap(),
    )
    .unwrap();
    assert_ne!(receipt["status"], "completed");
    f.directory.close().unwrap();
}

#[test]
fn artifact_matrix_actual_cli_serving_sibling_drift_retains_measurement_but_refuses_custody_parity()
{
    let f = Fixture::new();
    let input = setup(&f, false);
    fs::write(f.root.join("mode"), b"sibling-drift").unwrap();
    let report = run(spec(&f, input), Cancellation::default());
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert_eq!(report.status.and_then(|s| s.code()), Some(1));
    assert!(f.root.join("sibling-drift-observed").is_file());
    let cell = f.root.join("artifact-matrix/cell-0000");
    let row: Value = serde_json::from_slice(&fs::read(cell.join("row.json")).unwrap()).unwrap();
    assert_eq!(row["status"], "incomplete");
    assert_eq!(row["observations"][2]["status"], "completed");
    assert_eq!(row["observations"][3]["status"], "incomplete");
    assert_eq!(row["observations"][4]["parity"]["matches"], false);
    assert_eq!(
        row["observations"][4]["parity"]["unavailable_reason"],
        "custody_or_lifecycle_failure"
    );
    let new: Value =
        serde_json::from_slice(&fs::read(cell.join("sweep-skippy-new/output/cell.json")).unwrap())
            .unwrap();
    assert_eq!(new["status"], "incomplete");
    assert_eq!(new["measurement"]["status"], "completed");
    assert_eq!(
        new["measurement"]["sweep"][0]["measurement"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(
        new["measurement"]["sweep"][0]["measurement"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "completed")
    );
    assert_eq!(
        new["after_identity_error"],
        "owned post-run identity refusal"
    );
    assert!(new["after_identity"].is_null());
    let before = &new["before_identity"]["model_identity"]["ordered_files"];
    assert_eq!(before.as_array().unwrap().len(), 3);
    let primary = PathBuf::from(
        new["before_identity"]["admitted"]["model"]
            .as_str()
            .unwrap(),
    );
    let sibling = primary
        .parent()
        .unwrap()
        .join("MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf");
    assert_ne!(
        before[1]["sha256"],
        json!(hex::encode(Sha256::digest(fs::read(sibling).unwrap())))
    );
    let summary: Value = serde_json::from_slice(
        &fs::read(f.root.join("artifact-matrix/cache-family-matrix.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["status"], "incomplete");
    assert!(
        f.root
            .join("artifact-matrix/production-cache-bench.md")
            .is_file()
    );
    f.directory.close().unwrap();
}
