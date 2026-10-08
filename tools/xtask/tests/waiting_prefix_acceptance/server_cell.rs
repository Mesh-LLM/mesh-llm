//! Executable ownership evidence, using a local fixture without native models.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

pub(super) fn document(directory: &Path, model_id: &str) -> Value {
    let model = directory.join("model.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(4_u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "fixture");
    for (key, value) in [
        ("fixture.context_length", 1024_u32),
        ("fixture.block_count", 2),
        ("fixture.embedding_length", 16),
    ] {
        string(&mut bytes, key);
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    fs::write(&model, &bytes).unwrap();
    let binary = Path::new(env!("CARGO_BIN_EXE_waiting-prefix-fixture"))
        .canonicalize()
        .unwrap();
    let binary_sha = hex::encode(Sha256::digest(fs::read(&binary).unwrap()));
    json!({"schema_version":1,"binary":binary,"binary_sha256":binary_sha,
        "native_runtime_root":directory,"admission_concurrency":0,"execution_timeout_secs":8,
        "stage":{"model_id":model_id,"model_path":model,"source_model_sha256":hex::encode(Sha256::digest(&bytes)),
            "layer_end":2,"ctx_size":512,"lane_count":1,"n_gpu_layers":0,"payload":"resident-kv","cache_entries":1},
        "worker":{"schema_version":1,"server_log":directory.join("pending.log"),"cache_seed":null,
            "startup_timeout_secs":2,"telemetry_timeout_secs":1,
            "phase":{"schema_version":1,"round":1,"version":"new","base_url":"http://127.0.0.1:1/v1",
                "model":model_id,"output_tokens":2,"request_timeout_secs":1.0,"stagger_ms":0.0,
                "prompts":[{"family":"family-0","prompt":"task"}]}}})
}

fn run(directory: &Path, input: &Value) -> std::process::Output {
    fs::write(
        directory.join("input.json"),
        serde_json::to_vec(input).unwrap(),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory)
        .args([
            "automation",
            "waiting-prefix",
            "server-cell",
            "--input",
            "input.json",
            "--output-directory",
            "cell",
        ])
        .output()
        .unwrap()
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[cfg(unix)]
pub(super) fn no_survivor(directory: &Path) {
    let pid = fs::read_to_string(directory.join("fixture.pid"))
        .unwrap()
        .parse::<i32>()
        .unwrap();
    // The retained owner must reap its fixture leader, not merely return an exit code.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    if let Ok(address) = fs::read_to_string(directory.join("fixture.address")) {
        assert!(std::net::TcpListener::bind(address).is_ok());
    }
}

#[test]
fn native_server_owner_measures_then_stops_its_fixture_and_preserves_existing_directory() {
    let directory = tempfile::tempdir().unwrap();
    let input = document(directory.path(), "fixture");
    let result = run(directory.path(), &input);
    assert!(
        result.status.success(),
        "{}\nlifecycle: {}\nworker: {}\nserver: {}",
        String::from_utf8_lossy(&result.stderr),
        fs::read_to_string(directory.path().join("cell/lifecycle.json")).unwrap_or_default(),
        fs::read_to_string(directory.path().join("cell/worker.stderr.log")).unwrap_or_default(),
        fs::read_to_string(directory.path().join("cell/server.log")).unwrap_or_default()
    );
    let cell = directory.path().join("cell");
    let lifecycle = read(&cell.join("lifecycle.json"));
    assert_eq!(lifecycle["worker_status"], 0);
    assert_eq!(lifecycle["infrastructure_clean"], true);
    assert!(lifecycle["error"].is_null());
    assert_eq!(lifecycle["binary_sha256"], input["binary_sha256"]);
    assert_eq!(
        lifecycle["model_identity"]["sha256"],
        input["stage"]["source_model_sha256"]
    );
    assert_eq!(
        read(&cell.join("cell.json"))["summary"]["summary"]["successful"],
        1
    );
    let handoff = read(&cell.join("worker-input.json"));
    assert_ne!(
        handoff["phase"]["base_url"],
        input["worker"]["phase"]["base_url"]
    );
    assert!(
        fs::read_to_string(cell.join("server.log"))
            .unwrap()
            .contains("stage.openai_generation_summary")
    );
    #[cfg(unix)]
    no_survivor(&cell);
    let prior = fs::read(cell.join("lifecycle.json")).unwrap();
    assert!(!run(directory.path(), &input).status.success());
    assert_eq!(fs::read(cell.join("lifecycle.json")).unwrap(), prior);
}

#[test]
fn failed_worker_and_early_server_exit_retain_lifecycle_and_clean_owned_fixture() {
    for model in ["wrong-model", "early-exit"] {
        let directory = tempfile::tempdir().unwrap();
        let input = document(directory.path(), model);
        let result = run(directory.path(), &input);
        assert_eq!(result.status.code(), Some(1));
        let cell = directory.path().join("cell");
        let lifecycle = read(&cell.join("lifecycle.json"));
        assert!(lifecycle["error"].is_string());
        if model == "wrong-model" {
            assert_eq!(lifecycle["worker_status"], 1);
            assert!(
                read(&cell.join("cell.json"))["error"]
                    .as_str()
                    .unwrap()
                    .contains("differs")
            );
        } else {
            assert_eq!(lifecycle["infrastructure_clean"], false);
        }
        #[cfg(unix)]
        no_survivor(&cell);
    }
}

#[test]
fn forged_binary_model_hash_and_native_context_fail_before_any_output_or_launch() {
    for field in ["binary", "model", "context"] {
        let directory = tempfile::tempdir().unwrap();
        let mut input = document(directory.path(), "fixture");
        match field {
            "binary" => input["binary_sha256"] = json!("0".repeat(64)),
            "model" => input["stage"]["source_model_sha256"] = json!("0".repeat(64)),
            _ => input["stage"]["ctx_size"] = json!(2048),
        }
        assert_eq!(run(directory.path(), &input).status.code(), Some(1));
        assert!(!directory.path().join("cell").exists());
    }
}

#[cfg(unix)]
#[test]
fn interrupted_server_owner_reaps_only_its_owned_fixture_and_retains_failure() {
    let directory = tempfile::tempdir().unwrap();
    let input = document(directory.path(), "stall");
    fs::write(
        directory.path().join("input.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory.path())
        .args([
            "automation",
            "waiting-prefix",
            "server-cell",
            "--input",
            "input.json",
            "--output-directory",
            "cell",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let cell = directory.path().join("cell");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(4);
    while !cell.join("fixture.pid").exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "fixture did not start before interruption"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    let result = child.wait_with_output().unwrap();
    assert!(!result.status.success());
    assert!(read(&cell.join("lifecycle.json"))["error"].is_string());
    no_survivor(&cell);
}
