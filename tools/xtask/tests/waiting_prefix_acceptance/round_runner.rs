//! Complete-round fixtures prove orchestration, not native model performance.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

#[path = "metrics_collector_fixture.rs"]
mod metrics_collector;
use metrics_collector::{Collector, Mode};

fn fixture(directory: &Path, model: &str, mode: Mode) -> (Value, Collector) {
    let collector = Collector::start(mode);
    let mut input = input(directory, model);
    input["metrics_http"] = json!(collector.http);
    input["metrics_otlp_grpc"] = json!("http://127.0.0.1:14317");
    input["metrics_timeout_secs"] = json!(2);
    input["cell_timeout_secs"] = json!(12);
    (input, collector)
}

fn input(directory: &Path, model: &str) -> Value {
    let owner = super::server_cell::document(directory, model);
    let stage = &owner["stage"];
    let native = directory.join("provided-native");
    fs::create_dir(&native).unwrap();
    fs::write(native.join("fixture-native.bin"), b"fixture native bytes\n").unwrap();
    let catalog = directory.join("catalog.json");
    fs::write(&catalog,serde_json::to_vec(&json!({"schema_version":1,"datasets":{},"profiles":{"fixture":{
        "description":"local complete-round orchestration fixture",
        "model":{"id":model,"repo":"fixture/model","revision":"c".repeat(40),
            "filename":"model.gguf","sha256":stage["source_model_sha256"]},
        "corpus":{"kind":"synthetic","generator":"stable-prefix-v1"},
        "workload":{"rounds":2,"families":1,"requests_per_family":1,"prefix_blocks":2,"output_tokens":2,
            "ctx_size":512,"lanes":1,"admission_concurrency":1,"cache_entries":1,"stagger_ms":1.0},
        "ci_trace":{"family_order":[0],"prompt_tokens":1,"expected_fcfs_switches":0,"expected_dfs_switches":0},
        "hardware_acceptance":{"successful_requests_per_binary":2,"capacity_rejections_after_max":0}}}})).unwrap()).unwrap();
    json!({"schema_version":1,"old":{"path":owner["binary"],"sha256":owner["binary_sha256"],"supplied_commit":"a".repeat(40)},
        "new":{"path":owner["binary"],"sha256":owner["binary_sha256"],"supplied_commit":"b".repeat(40)},
        "model_id":model,"model_path":stage["model_path"],"model_sha256":stage["source_model_sha256"],
        "catalog":catalog,"profile":"fixture","contract":null,"prompt_manifest":null,
        "native_runtime_root":native,"native_runtime_sha256":"02d21a576ced558f4a83ee3541570622a3e68f746c347a0fff1377e572220912","payload":"resident-kv","n_gpu_layers":0,
        "request_timeout_secs":1.0,"startup_timeout_secs":2,"telemetry_timeout_secs":1,"cell_timeout_secs":8})
}

fn command(directory: &Path, input: &Value) -> Command {
    fs::write(
        directory.join("run-input.json"),
        serde_json::to_vec(input).unwrap(),
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.current_dir(directory).args([
        "automation",
        "waiting-prefix",
        "run",
        "--input",
        "run-input.json",
        "--output-directory",
        "comparison",
    ]);
    command
}

fn comparison(directory: &Path) -> Value {
    serde_json::from_slice(&fs::read(directory.join("comparison/comparison.json")).unwrap())
        .unwrap()
}

#[test]
fn native_round_runner_alternates_all_cells_and_accepts_only_complete_fixture_measurements() {
    let directory = tempfile::tempdir().unwrap();
    let (input, _collector) = fixture(directory.path(), "fixture", Mode::Good);
    let result = command(directory.path(), &input).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let document = comparison(directory.path());
    let cells = document["cells"].as_array().unwrap();
    let schedule: Vec<_> = cells
        .iter()
        .map(|cell| {
            (
                cell["round"].as_u64().unwrap(),
                cell["version"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(schedule, [(1, "old"), (1, "new"), (2, "new"), (2, "old")]);
    assert!(
        cells
            .iter()
            .all(|cell| cell["process_clean"] == true && cell["error"].is_null())
    );
    assert_eq!(document["acceptance"]["passed"], true);
    assert_eq!(document["aggregate"][0]["rounds"], 2);
    assert_eq!(document["aggregate"][1]["successful"], 2);
    assert_eq!(document["old"]["supplied_commit"], "a".repeat(40));
    assert_eq!(document["new"]["sha256"], input["new"]["sha256"]);
    assert!(directory.path().join("comparison/report.md").exists());
    #[cfg(unix)]
    for cell in cells {
        super::server_cell::no_survivor(Path::new(cell["directory"].as_str().unwrap()));
    }
    let prior = fs::read(directory.path().join("comparison/comparison.json")).unwrap();
    assert!(
        !command(directory.path(), &input)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(
        fs::read(directory.path().join("comparison/comparison.json")).unwrap(),
        prior
    );
}

#[test]
fn native_round_runner_retains_late_failed_cell_and_every_subsequent_attempt_without_acceptance() {
    let directory = tempfile::tempdir().unwrap();
    let (input, _collector) = fixture(directory.path(), "late-failure", Mode::Good);
    assert_eq!(
        command(directory.path(), &input)
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
    let document = comparison(directory.path());
    let cells = document["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 4);
    assert_eq!(cells[2]["round"], 2);
    assert_eq!(cells[2]["version"], "new");
    assert!(cells[2]["error"].is_string());
    assert!(cells[3]["error"].is_null());
    assert!(document["error"].is_string());
    assert!(document["acceptance"].is_null());
    #[cfg(unix)]
    for cell in cells {
        super::server_cell::no_survivor(Path::new(cell["directory"].as_str().unwrap()));
    }
}

#[test]
fn native_round_runner_failed_acceptance_retains_report_and_bad_pin_prevents_all_launches() {
    let directory = tempfile::tempdir().unwrap();
    let (mut input, _collector) = fixture(directory.path(), "fixture", Mode::Good);
    input["old"]["sha256"] = json!("0".repeat(64));
    assert_eq!(
        command(directory.path(), &input)
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
    assert!(!directory.path().join("comparison").exists());
    input["old"]["sha256"] = input["new"]["sha256"].clone();
    let path = Path::new(input["catalog"].as_str().unwrap());
    let mut catalog: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    catalog["profiles"]["fixture"]["hardware_acceptance"]["suffix_prefill_before_min"] =
        json!(99999);
    fs::write(path, serde_json::to_vec(&catalog).unwrap()).unwrap();
    assert_eq!(
        command(directory.path(), &input)
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
    let document = comparison(directory.path());
    assert_eq!(document["cells"].as_array().unwrap().len(), 4);
    assert_eq!(document["acceptance"]["passed"], false);
    assert!(directory.path().join("comparison/report.md").exists());
}

#[cfg(unix)]
#[test]
fn interrupted_round_runner_retains_attempt_and_never_launches_the_remaining_schedule() {
    let directory = tempfile::tempdir().unwrap();
    let (input, _collector) = fixture(directory.path(), "stall", Mode::Good);
    let child = command(directory.path(), &input)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let first = directory.path().join("comparison/round-1-old");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(4);
    while !first.join("fixture.pid").exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "round fixture did not start"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    assert!(!child.wait_with_output().unwrap().status.success());
    let document = comparison(directory.path());
    assert_eq!(document["cells"].as_array().unwrap().len(), 1);
    assert!(document["error"].is_string());
    assert!(!directory.path().join("comparison/round-1-new").exists());
    super::server_cell::no_survivor(&first);
}

#[test]
fn delayed_collector_delivery_keeps_distinct_cell_identities_and_finalized_reports() {
    let directory = tempfile::tempdir().unwrap();
    let (input, collector) = fixture(directory.path(), "fixture", Mode::Delayed);
    let result = command(directory.path(), &input).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let document = comparison(directory.path());
    let cells = document["cells"].as_array().unwrap();
    let mut ids = std::collections::BTreeSet::new();
    for cell in cells {
        let evidence = &cell["evidence"];
        let id = evidence["collector"]["endpoint"]["run_id"]
            .as_str()
            .unwrap();
        assert!(ids.insert(id));
        assert_eq!(evidence["collector"]["timings"][0]["request_id"], "1");
        assert_eq!(evidence["collector"]["timings"][0]["server_ttft_ms"], 1.0);
        let path = Path::new(cell["directory"].as_str().unwrap());
        let report: Value =
            serde_json::from_slice(&fs::read(path.join("metrics-report.json")).unwrap()).unwrap();
        assert_eq!(report["run"]["run_id"], id);
        assert_eq!(report["run"]["status"], "completed");
        let fixture: Value =
            serde_json::from_slice(&fs::read(path.join("fixture.metrics.json")).unwrap()).unwrap();
        assert_eq!(fixture["run_id"], id);
        assert_eq!(fixture["otlp_grpc"], input["metrics_otlp_grpc"]);
        #[cfg(unix)]
        super::server_cell::no_survivor(path);
    }
    assert_eq!(ids.len(), 4);
    let summaries = document["collector_summary"].as_array().unwrap();
    assert_eq!(summaries.len(), 2);
    for summary in summaries {
        assert_eq!(summary["measured_requests"], 2);
        assert_eq!(summary["server_ttft_ms_p50"], 1.0);
    }
    let markdown = fs::read_to_string(directory.path().join("comparison/report.md")).unwrap();
    assert!(markdown.contains("Client TTFT p50 ms"));
    assert!(markdown.contains("Server TTFT p50 ms"));
    assert_eq!(
        collector
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| *call == "POST /v1/runs")
            .count(),
        4
    );
    assert_eq!(document["acceptance"]["passed"], true);
}

#[test]
fn rejected_collectors_retain_all_attempts_and_request_results_without_acceptance() {
    for mode in [
        Mode::WrongRun,
        Mode::MissingDecode,
        Mode::Loss,
        Mode::FailFinalize,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let (input, _collector) = fixture(directory.path(), "fixture", mode);
        assert_eq!(
            command(directory.path(), &input)
                .output()
                .unwrap()
                .status
                .code(),
            Some(1)
        );
        let document = comparison(directory.path());
        let cells = document["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 4);
        assert!(document["acceptance"].is_null());
        for cell in cells {
            assert!(cell["error"].is_string());
            assert_eq!(cell["evidence"]["summary"]["summary"]["successful"], 1);
            let path = Path::new(cell["directory"].as_str().unwrap());
            assert!(path.join("metrics-report.json").is_file());
            assert!(!path.join("metrics-timing.json").exists());
            #[cfg(unix)]
            super::server_cell::no_survivor(path);
        }
    }
}

#[cfg(unix)]
#[test]
fn cancellation_during_collector_delivery_preserves_measurement_and_reaps_the_stage() {
    let directory = tempfile::tempdir().unwrap();
    let (input, collector) = fixture(directory.path(), "fixture", Mode::StallCollection);
    let child = command(directory.path(), &input)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(4);
    while !collector
        .calls
        .lock()
        .unwrap()
        .iter()
        .any(|call| call.ends_with("/report.json"))
    {
        assert!(
            std::time::Instant::now() < deadline,
            "collector delivery did not start"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    assert!(!child.wait_with_output().unwrap().status.success());
    let document = comparison(directory.path());
    assert_eq!(document["cells"].as_array().unwrap().len(), 1);
    assert_eq!(
        document["cells"][0]["evidence"]["summary"]["summary"]["successful"],
        1
    );
    assert!(document["acceptance"].is_null());
    assert!(!directory.path().join("comparison/round-1-new").exists());
    super::server_cell::no_survivor(&directory.path().join("comparison/round-1-old"));
}

fn manual(directory: &Path) -> (Value, Collector) {
    let (mut input, collector) = fixture(directory, "fixture", Mode::Good);
    input["catalog"] = Value::Null;
    input["profile"] = Value::Null;
    input["manual_workload"] = json!({"rounds":2,"families":2,"requests_per_family":2,"prefix_blocks":2,"output_tokens":2,"ctx_size":512,"lanes":4,"admission_concurrency":0,"cache_entries":1,"stagger_ms":0.0});
    (input, collector)
}

fn bounded_verb(directory: &Path, verb: &str, input: &Value, output: &str) -> std::process::Output {
    fs::write(
        directory.join("manual-input.json"),
        serde_json::to_vec(input).unwrap(),
    )
    .unwrap();
    bounded_existing(directory, verb, output)
}

fn bounded_existing(directory: &Path, verb: &str, output: &str) -> std::process::Output {
    let arguments = [
        "automation",
        "waiting-prefix",
        verb,
        "--input",
        "manual-input.json",
        if verb == "run" {
            "--output-directory"
        } else {
            "--output"
        },
        output,
    ];
    fs::write(directory.join("owned-timeout.json"),serde_json::to_vec(&json!({"label":"owned waiting-prefix manual CLI fixture","seconds":60,"cwd":directory,"executable":env!("CARGO_BIN_EXE_xtask"),"arguments":arguments})).unwrap()).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.env_clear().env("PATH", "/usr/bin:/bin");
    for name in ["SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let result = command
        .current_dir(directory)
        .args([
            "automation",
            "canary-timeout",
            "--input",
            "owned-timeout.json",
        ])
        .output()
        .unwrap();
    assert!(
        matches!(result.status.code(), Some(0 | 1)),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.len() < 65536 && result.stderr.len() < 65536);
    result
}

#[test]
fn manual_round_cli_prepares_and_executes_complete_synthetic_and_irregular_manifest_rosters() {
    for manifest in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let (mut input, collector) = manual(directory.path());
        if manifest {
            let path = directory.path().join("owned-prompts.json");
            fs::write(&path,br#"{"metadata":{"source":"owned-local-roster"},"prompts":[{"family":"a","prompt":"first"},{"family":"b","prompt":"second"},{"family":"a","prompt":"third"}]}"#).unwrap();
            input["prompt_manifest"] = json!(path);
        }
        let declared = input.clone();
        input["old"].as_object_mut().unwrap().remove("sha256");
        input["new"].as_object_mut().unwrap().remove("sha256");
        input.as_object_mut().unwrap().remove("model_sha256");
        input
            .as_object_mut()
            .unwrap()
            .remove("native_runtime_sha256");
        let prepared = bounded_verb(directory.path(), "prepare-run", &input, "prepared.json");
        assert!(
            prepared.status.success(),
            "{}",
            String::from_utf8_lossy(&prepared.stderr)
        );
        assert!(!directory.path().join("comparison").exists());
        let input: Value =
            serde_json::from_slice(&fs::read(directory.path().join("prepared.json")).unwrap())
                .unwrap();
        assert_eq!(input["prepared_plan_sha256"].as_str().unwrap().len(), 64);
        assert!(input["manual_workload"].is_object());
        assert_eq!(input["old"]["sha256"], declared["old"]["sha256"]);
        assert_eq!(input["new"]["sha256"], declared["new"]["sha256"]);
        assert_eq!(input["model_sha256"], declared["model_sha256"]);
        assert_eq!(
            input["native_runtime_sha256"],
            declared["native_runtime_sha256"]
        );
        let result = bounded_verb(directory.path(), "run", &input, "comparison");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let output = comparison(directory.path());
        assert_eq!(output["cells"].as_array().unwrap().len(), 4);
        assert!(output["error"].is_null() && output["acceptance"].is_null());
        let expected = if manifest { 3 } else { 4 };
        assert_eq!(output["workload_plan"]["requests_per_round"], expected);
        assert_eq!(output["aggregate"][0]["successful"], 2 * expected);
        assert_eq!(output["aggregate"][1]["successful"], 2 * expected);
        assert!(output["workload_plan"]["fixture_catalog_sha256"].is_null());
        if manifest {
            assert_eq!(
                output["workload_plan"]["family_request_counts"],
                json!({"a":2,"b":1})
            );
            assert_eq!(
                output["workload_plan"]["prompt_manifest_metadata"]["source"],
                "owned-local-roster"
            );
        }
        let report = fs::read_to_string(directory.path().join("comparison/report.md")).unwrap();
        assert!(report.contains("Acceptance: not requested") && report.contains("```mermaid"));
        assert!(!report.contains("Fixture acceptance: **PASS**"));
        #[cfg(unix)]
        for cell in output["cells"].as_array().unwrap() {
            super::server_cell::no_survivor(Path::new(cell["directory"].as_str().unwrap()));
        }
        drop(collector);
        directory.close().unwrap();
    }
}

#[test]
fn manual_round_cli_refuses_mixed_incomplete_or_drifted_preparation_before_any_cell_launch() {
    for failure in ["mixed", "short", "drift"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut input, collector) = manual(directory.path());
        if failure == "mixed" {
            input["profile"] = json!("fixture");
        }
        if failure == "short" {
            input["manual_workload"]["lanes"] = json!(1);
        }
        if failure == "drift" {
            let path = directory.path().join("owned-prompts.json");
            fs::write(&path, br#"{"prompts":[{"family":"a","prompt":"first"}]}"#).unwrap();
            input["prompt_manifest"] = json!(path);
            assert!(
                bounded_verb(directory.path(), "prepare-run", &input, "prepared.json")
                    .status
                    .success()
            );
            input =
                serde_json::from_slice(&fs::read(directory.path().join("prepared.json")).unwrap())
                    .unwrap();
            fs::write(&path, br#"{"prompts":[{"family":"a","prompt":"changed"}]}"#).unwrap();
        }
        let result = bounded_verb(directory.path(), "run", &input, "comparison");
        assert_eq!(result.status.code(), Some(1));
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            error.contains(match failure {
                "mixed" => "either catalog/profile",
                "short" => "complete prompt roster",
                _ => "provenance changed",
            }),
            "{error}"
        );
        assert!(!directory.path().join("comparison").exists());
        assert!(collector.calls.lock().unwrap().is_empty());
        drop(collector);
        directory.close().unwrap();
    }
}

#[test]
fn manual_round_cli_retains_late_failed_measurement_without_acceptance_or_complete_claim() {
    let directory = tempfile::tempdir().unwrap();
    let (mut input, collector) = manual(directory.path());
    input["model_id"] = json!("manual-late-failure");
    let result = bounded_verb(directory.path(), "run", &input, "comparison");
    assert_eq!(result.status.code(), Some(1));
    let output = comparison(directory.path());
    assert_eq!(output["cells"].as_array().unwrap().len(), 4);
    assert!(output["cells"][3]["error"].is_string());
    assert!(output["error"].is_string() && output["acceptance"].is_null());
    assert!(output["aggregate"].as_array().unwrap().is_empty());
    assert_eq!(output["workload_plan"]["workload_profile"], "manual");
    #[cfg(unix)]
    for cell in output["cells"].as_array().unwrap() {
        super::server_cell::no_survivor(Path::new(cell["directory"].as_str().unwrap()));
    }
    drop(collector);
    directory.close().unwrap();
}

#[cfg(unix)]
#[test]
fn manual_round_cli_cancels_actual_held_collector_and_preserves_attempt_without_next_launch() {
    let directory = tempfile::tempdir().unwrap();
    let (mut input, old_collector) = manual(directory.path());
    drop(old_collector);
    let collector = Collector::start(Mode::StallCollection);
    input["metrics_http"] = json!(collector.http);
    assert!(
        bounded_verb(directory.path(), "prepare-run", &input, "prepared.json")
            .status
            .success()
    );
    let path = directory.path().join("owned-timeout.json");
    let mut timeout: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    timeout["seconds"] = json!(10);
    timeout["arguments"][2] = json!("run");
    timeout["arguments"][5] = json!("--output-directory");
    timeout["arguments"][6] = json!("comparison");
    fs::write(&path, serde_json::to_vec(&timeout).unwrap()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .current_dir(directory.path())
        .args([
            "automation",
            "canary-timeout",
            "--input",
            "owned-timeout.json",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let observed = loop {
        if collector
            .calls
            .lock()
            .unwrap()
            .iter()
            .any(|c| c.ends_with("/report.json"))
        {
            break true;
        }
        if std::time::Instant::now() >= deadline {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    let result = child.wait_with_output().unwrap();
    assert!(
        observed,
        "held collector did not start: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.status.code(), Some(143));
    assert!(result.stdout.len() < 65536 && result.stderr.len() < 65536);
    let output = comparison(directory.path());
    assert_eq!(output["cells"].as_array().unwrap().len(), 1);
    assert!(output["error"].is_string() && output["acceptance"].is_null());
    assert!(!directory.path().join("comparison/round-1-new").exists());
    assert_eq!(output["workload_plan"]["workload_profile"], "manual");
    super::server_cell::no_survivor(&directory.path().join("comparison/round-1-old"));
    drop(collector);
    directory.close().unwrap();
}

#[cfg(unix)]
#[test]
fn manual_prepare_cli_refuses_static_fifo_input_before_publication_without_writer() {
    use std::os::unix::ffi::OsStrExt as _;
    let directory = tempfile::tempdir().unwrap();
    let (input, collector) = manual(directory.path());
    assert!(
        bounded_verb(directory.path(), "prepare-run", &input, "prepared.json")
            .status
            .success()
    );
    let input = directory.path().join("manual-input.json");
    fs::remove_file(&input).unwrap();
    let name = std::ffi::CString::new(input.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    fs::write(directory.path().join("refused.json"), b"preserve-existing").unwrap();
    let result = bounded_existing(directory.path(), "prepare-run", "refused.json");
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("must be a regular file"));
    assert_eq!(
        fs::read(directory.path().join("refused.json")).unwrap(),
        b"preserve-existing"
    );
    fs::create_dir(directory.path().join("existing-comparison")).unwrap();
    fs::write(
        directory.path().join("existing-comparison/prior"),
        b"preserve-existing",
    )
    .unwrap();
    for output in ["comparison", "existing-comparison"] {
        let result = bounded_existing(directory.path(), "run", output);
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&result.stderr).contains("must be a regular file"));
    }
    assert_eq!(
        fs::read(directory.path().join("existing-comparison/prior")).unwrap(),
        b"preserve-existing"
    );
    assert_eq!(
        fs::read_dir(directory.path().join("existing-comparison"))
            .unwrap()
            .count(),
        1
    );
    assert!(!directory.path().join("comparison").exists());
    assert!(collector.calls.lock().unwrap().is_empty());
    drop(collector);
    directory.close().unwrap();
}
