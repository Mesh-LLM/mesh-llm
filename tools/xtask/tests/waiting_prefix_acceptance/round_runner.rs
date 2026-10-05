//! Complete-round fixtures prove orchestration, not native model performance.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

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
    let input = input(directory.path(), "fixture");
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
    let input = input(directory.path(), "late-failure");
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
    let mut input = input(directory.path(), "fixture");
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
    let input = input(directory.path(), "stall");
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
