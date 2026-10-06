//! Real coordinator subprocess + inert Rust executable boundary, no model execution.
use crate::process;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};
fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples")
        .join(format!("l7_daemon_fixture{}", std::env::consts::EXE_SUFFIX))
}
fn tree(path: &Path) -> String {
    let mut entries: Vec<_> = std::fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    let mut hash = Sha256::new();
    for entry in entries {
        let name = entry.file_name().unwrap().to_str().unwrap().as_bytes();
        hash.update((name.len() as u64).to_be_bytes());
        hash.update(name);
        hash.update(Sha256::digest(std::fs::read(entry).unwrap()));
    }
    hex::encode(hash.finalize())
}
fn input(root: &Path, workload: &str, arm: &str) -> PathBuf {
    let binary = fixture();
    assert!(
        binary.is_file(),
        "existing example must be built by normal target prerequisites"
    );
    let model = root.join("fixture.gguf");
    std::fs::write(&model, b"inert model bytes").unwrap();
    let runtime = root.join("runtime");
    std::fs::create_dir(&runtime).unwrap();
    std::fs::write(runtime.join("runtime.data"), b"inert runtime").unwrap();
    let tokenizer = root.join("tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    std::fs::write(tokenizer.join("tokenizer.json"), b"{}").unwrap();
    let mut config: Value = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../evals/skippy-competitive-benchmark.json"
    )))
    .unwrap();
    config["models"][0]["sha256"] = digest(b"inert model bytes").into();
    config["models"][0]["model_id"] = "fixture-model".into();
    config["models"][0]["tokenizer_sha256"] = tree(&tokenizer).into();
    config["thoughtworks"]["minimum_prompts"] = 1.into();
    config["baseline"]["llama_benchy_version"] = "fixture version 1".into();
    let manifest = json!({"metadata":{"dataset_revision":config["thoughtworks"]["dataset"]["revision"],"rows":config["thoughtworks"]["selection"]["rows"]},"prompts":(0..256).map(|index|json!({"family":index%8,"prompt":"πabcdefghijkl"})).collect::<Vec<_>>()});
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let manifest_path = root.join("manifest.json");
    std::fs::write(&manifest_path, &bytes).unwrap();
    config["thoughtworks"]["selection"]["manifest_sha256"] = digest(&bytes).into();
    let bytes = serde_json::to_vec(&config).unwrap();
    let config_path = root.join("config.json");
    std::fs::write(&config_path, &bytes).unwrap();
    let cell = if workload == "thoughtworks" {
        json!({"platform":"metal","model":"llama32-dense","workload":workload,"arm":arm,"output_tokens":8,"context_size":131072,"active_lanes":16,"concurrency":1,"prompt_count":1})
    } else {
        json!({"platform":"metal","model":"llama32-dense","workload":workload,"arm":arm,"prompt_tokens":512,"output_tokens":8,"concurrency":1})
    };
    let binary_sha = digest(&std::fs::read(&binary).unwrap());
    let document = json!({"config":config_path,"config_sha256":digest(&bytes),"cell":cell,"model":{"path":model,"sha256":config["models"][0]["sha256"]},"backend":{"executable":{"path":binary,"sha256":binary_sha},"version_sha256":digest(b"fixture version 1"),"cwd":root,"runtime":{"path":runtime,"sha256":tree(&runtime)},"tokenizer":{"path":tokenizer,"sha256":tree(&tokenizer)},"hf_config":null,"comparison_model":null,"match_kv_capacity":false},"manifest":manifest_path,"benchy":{"path":binary,"sha256":binary_sha},"output":root.join("output"),"timeout_seconds":20,"request_timeout_seconds":1});
    let path = root.join("input.json");
    std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
    path
}
fn run(root: &Path, input: &Path) -> bool {
    run_command(root, input, "competitive-run-cell", Duration::from_secs(24))
}
fn run_command(root: &Path, input: &Path, command: &str, execution: Duration) -> bool {
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            arguments: [
                "automation",
                "replay-matrix",
                command,
                if command == "competitive-report" {
                    "--artifact"
                } else {
                    "--input"
                },
                input.to_str().unwrap(),
            ]
            .map(|value| process::Value::Public(value.into()))
            .into(),
            environment: BTreeMap::from([(
                "PATH".into(),
                process::Value::Public(std::env::var_os("PATH").expect("fixture Git PATH")),
            )]),
        },
        &process::Limits {
            execution,
            graceful_shutdown: Duration::from_millis(250),
            forced_shutdown: Duration::from_millis(250),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    std::fs::write(
        root.join("last-command-stderr"),
        raw.stderr.as_ref().unwrap().as_bytes(),
    )
    .unwrap();
    let report = raw.process;
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert!(
        report.failure.is_none()
            && report.cleanup.complete
            && !report.cleanup.forced
            && !report.cleanup.graceful_signal_failed
            && report.cleanup.failure.is_none(),
        "{report:?}"
    );
    assert!(!report.stdout.truncated && !report.stderr.truncated);
    assert_eq!(report.stdout.suppressed_lines, 0);
    assert_eq!(report.stderr.suppressed_lines, 0);
    assert_eq!(
        raw.stdout.unwrap().as_bytes().len() as u64,
        report.stdout.bytes_seen
    );
    assert_eq!(
        raw.stderr.unwrap().as_bytes().len() as u64,
        report.stderr.bytes_seen
    );
    report.status.unwrap().success()
}
fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn matrix_failure(root: &Path) -> String {
    let mut output = std::fs::read_to_string(root.join("last-command-stderr")).unwrap();
    for (concurrency, arm) in
        [1, 2, 4, 8, 16, 32, 64, 128, 256]
            .into_iter()
            .flat_map(|concurrency| {
                ["llama", "mesh"]
                    .into_iter()
                    .map(move |arm| (concurrency, arm))
            })
    {
        let directory = root.join(format!(
            "matrix/cells/llama32-dense/synthetic/tg-8-c-{concurrency}/{arm}"
        ));
        if directory.join("complete.json").is_file() {
            continue;
        }
        if let Ok(bytes) = std::fs::read(directory.join("worker/parity.json")) {
            let parity: Value = serde_json::from_slice(&bytes).unwrap();
            let failures: Vec<_> = parity["results"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|row| row["valid"] != true)
                .collect();
            output.push_str(&format!(
                "\n{arm}/c{concurrency}/parity-failures: {failures:?}"
            ));
        }
        for name in [
            "lifecycle.json",
            "worker.stderr.log",
            "server.stderr.log",
            "worker/worker-summary.json",
        ] {
            if let Ok(bytes) = std::fs::read_to_string(directory.join(name)) {
                output.push_str(&format!("\n{arm}/{name}\n{bytes}"));
            }
        }
    }
    for entry in std::fs::read_dir(root.join("matrix")).unwrap() {
        let path = entry.unwrap().path();
        if !path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("invocation-")
        {
            continue;
        }
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".stderr.log")
            {
                output.push_str(&std::fs::read_to_string(path).unwrap());
            }
        }
    }
    output
}
#[test]
fn competitive_retained_trace_cell_binds_capacity_and_completes_after_server_stop() {
    let root = tempfile::tempdir().unwrap();
    let path = input(root.path(), "thoughtworks", "mesh-adaptive");
    assert!(run(root.path(), &path));
    assert!(root.path().join("server-stopped").is_file());
    let stage = json_file(&root.path().join("output/stage.json"));
    assert_eq!(stage["ctx_size"], 131072);
    assert_eq!(stage["lane_count"], 16);
    assert_eq!(stage["kv_cache"]["shared_prefix_record_limit"], 2);
    let argv = json_file(&root.path().join("server-argv.json"));
    assert!(
        argv.as_array()
            .unwrap()
            .iter()
            .any(|value| value == "--adaptive-generation-concurrency")
    );
    let completed = json_file(&root.path().join("output/complete.json"));
    assert_eq!(completed["completed"], true);
    assert_eq!(completed["cell"]["arm"], "mesh-adaptive");
    assert_eq!(
        completed["worker_summary_sha256"],
        digest(&std::fs::read(root.path().join("output/worker/worker-summary.json")).unwrap())
    );
    let rows: Vec<Value> = std::fs::read_to_string(root.path().join("captured-requests.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["messages"][0]["content"], "πabcdef");
    assert_eq!(rows[2]["messages"][0]["content"], "πabcdefghijkl");
    root.close().unwrap();
}
#[test]
fn competitive_retained_short_trace_stops_owned_server_and_never_completes() {
    let root = tempfile::tempdir().unwrap();
    let path = input(root.path(), "thoughtworks", "llama");
    std::fs::write(root.path().join("short-response"), b"fixture").unwrap();
    assert!(!run(root.path(), &path));
    assert!(root.path().join("server-stopped").is_file());
    assert!(!root.path().join("output/complete.json").exists());
    assert_eq!(
        json_file(&root.path().join("output/worker/worker-summary.json"))["passed"],
        false
    );
    root.close().unwrap();
}
#[test]
fn competitive_retained_synthetic_tool_boundary_enforces_progress_result_and_cleanup() {
    version_refusal_cases();
    for hidden in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let path = input(root.path(), "synthetic", "mesh");
        if hidden {
            std::fs::write(root.path().join("benchy-hidden-error"), b"fixture").unwrap();
        }
        assert_eq!(run(root.path(), &path), !hidden);
        assert!(root.path().join("server-stopped").is_file());
        assert_eq!(root.path().join("output/complete.json").exists(), !hidden);
        let argv = json_file(&root.path().join("benchy-argv.json"));
        let args: Vec<_> = argv
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert!(args.windows(2).any(|pair| pair == ["--pp", "512"]));
        assert!(args.windows(2).any(|pair| pair == ["--concurrency", "1"]));
        assert!(args.contains(&"--no-results-on-fail"));
        root.close().unwrap();
    }
}
fn version_refusal_cases() {
    for name in ["version-held", "version-oversized"] {
        let root = tempfile::tempdir().unwrap();
        let path = input(root.path(), "synthetic", "mesh");
        std::fs::write(root.path().join(name), b"owned fixture selector").unwrap();
        let mut prepared = json_file(&path);
        prepared["timeout_seconds"] = 10.into();
        std::fs::write(&path, serde_json::to_vec(&prepared).unwrap()).unwrap();
        assert!(!run(root.path(), &path));
        assert!(!root.path().join("output/version.json").exists());
        assert!(!root.path().join("output/complete.json").exists());
        assert!(!root.path().join("server-argv.json").exists());
        root.close().unwrap();
    }
}

#[test]
fn competitive_matrix_actual_cli_executes_complete_ladder_and_refuses_corrupted_resume() {
    let root = tempfile::tempdir().unwrap();
    let path = input(root.path(), "synthetic", "mesh");
    let previous = json_file(&path);
    let config_path = Path::new(previous["config"].as_str().unwrap());
    let mut config = json_file(config_path);
    config["synthetic"]["output_tokens"] = json!([8]);
    let bytes = serde_json::to_vec(&config).unwrap();
    std::fs::write(config_path, &bytes).unwrap();
    let matrix = prepared_layout_fixture(root.path(), &path, &previous);
    std::fs::write(&path, serde_json::to_vec(&matrix).unwrap()).unwrap();
    assert!(
        run_command(
            root.path(),
            &path,
            "competitive-run",
            Duration::from_secs(124)
        ),
        "{}",
        matrix_failure(root.path())
    );
    let results = json_file(&root.path().join("matrix/matrix-results.json"));
    assert_eq!(results["completed"], true);
    assert_eq!(results["cells"].as_array().unwrap().len(), 18);
    for concurrency in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        for arm in ["llama", "mesh"] {
            let directory = root.path().join(format!(
                "matrix/cells/llama32-dense/synthetic/tg-8-c-{concurrency}/{arm}"
            ));
            assert!(directory.join("complete.json").is_file());
        }
    }
    assert!(root.path().join("matrix/summary/synthetic.csv").is_file());
    assert_eq!(
        digest(&std::fs::read(root.path().join("matrix/benchmark-config.source.json")).unwrap()),
        matrix["config_sha256"]
    );
    assert!(root.path().join("matrix/summary/REPORT.md").is_file());
    assert!(
        root.path()
            .join("matrix/summary/charts/metal-llama32-dense-synthetic-tg-8-throughput.svg")
            .is_file()
    );
    assert!(root.path().join("matrix/artifact-sha256.txt").is_file());
    assert!(!root.path().join("matrix/.matrix-active").exists());
    archived_report_refusal_cases(root.path());
    let mut resumed = matrix;
    resumed["resume"] = true.into();
    std::fs::write(&path, serde_json::to_vec(&resumed).unwrap()).unwrap();
    assert!(run_command(
        root.path(),
        &path,
        "competitive-run",
        Duration::from_secs(124)
    ));
    let results = json_file(&root.path().join("matrix/matrix-results.json"));
    assert!(
        results["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["resumed"] == true)
    );
    std::fs::write(
        root.path()
            .join("matrix/cells/llama32-dense/synthetic/tg-8-c-1/mesh/worker/result.json"),
        b"{}",
    )
    .unwrap();
    assert!(!run_command(
        root.path(),
        &path,
        "competitive-run",
        Duration::from_secs(124)
    ));
    assert!(!root.path().join("matrix/.matrix-active").exists());
    root.close().unwrap();
}
fn archived_report_refusal_cases(root: &Path) {
    let artifact = root.join("matrix");
    let path = artifact.join("matrix-plan.json");
    let original = std::fs::read(&path).unwrap();
    let plan: Value = serde_json::from_slice(&original).unwrap();
    let summary = std::fs::read(artifact.join("summary/report.json")).unwrap();
    let outside = root.join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(
        outside.join("complete.json"),
        b"not JSON; must never be consumed",
    )
    .unwrap();
    for field in ["workload", "platform"] {
        for bad in ["..", outside.to_str().unwrap()] {
            let mut changed = plan.clone();
            changed["cells"][0][field] = bad.into();
            std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(!run_command(
                root,
                &artifact,
                "competitive-report",
                Duration::from_secs(10)
            ));
            let stderr = std::fs::read_to_string(root.join("last-command-stderr")).unwrap();
            assert!(stderr.contains("archive cells differ"), "{stderr}");
            assert_eq!(
                std::fs::read(artifact.join("summary/report.json")).unwrap(),
                summary
            );
            assert_eq!(
                std::fs::read(outside.join("complete.json")).unwrap(),
                b"not JSON; must never be consumed"
            );
        }
    }
    std::fs::write(&path, &original).unwrap();
    #[cfg(unix)]
    {
        for relative in [
            "summary/REPORT.md",
            "summary/synthetic.csv",
            "summary/parity.json",
            "summary/charts/metal-llama32-dense-synthetic-tg-8-throughput.svg",
            "artifact-sha256.txt",
        ] {
            let leaf = artifact.join(relative);
            let original_leaf = std::fs::read(&leaf).unwrap();
            std::fs::remove_file(&leaf).unwrap();
            let sentinel = outside.join("must-not-overwrite");
            std::fs::write(&sentinel, b"outside sentinel unchanged").unwrap();
            std::os::unix::fs::symlink(&sentinel, &leaf).unwrap();
            assert!(!run_command(
                root,
                &artifact,
                "competitive-report",
                Duration::from_secs(10)
            ));
            let stderr = std::fs::read_to_string(root.join("last-command-stderr")).unwrap();
            assert!(
                stderr.contains("report output leaf refuses symlink"),
                "{stderr}"
            );
            assert_eq!(
                std::fs::read(&sentinel).unwrap(),
                b"outside sentinel unchanged"
            );
            std::fs::remove_file(&leaf).unwrap();
            std::fs::write(&leaf, original_leaf).unwrap();
        }
    }
    #[cfg(unix)]
    {
        let parent = artifact.join("cells/llama32-dense");
        let backup = artifact.join("saved-model-parent");
        std::fs::rename(&parent, &backup).unwrap();
        std::os::unix::fs::symlink(&outside, &parent).unwrap();
        assert!(!run_command(
            root,
            &artifact,
            "competitive-report",
            Duration::from_secs(10)
        ));
        let stderr = std::fs::read_to_string(root.join("last-command-stderr")).unwrap();
        assert!(
            stderr.contains("archive parent refuses symlink"),
            "{stderr}"
        );
        assert_eq!(
            std::fs::read(artifact.join("summary/report.json")).unwrap(),
            summary
        );
        std::fs::remove_file(&parent).unwrap();
        std::fs::rename(&backup, &parent).unwrap();
    }
}
fn local_git(root: &Path, arguments: &[&str]) -> String {
    let executable = PathBuf::from(
        std::env::var_os("MIGRATION_TEST_GIT").expect("normal target must provide real Git"),
    );
    let mut environment = BTreeMap::new();
    for key in ["PATH", "SystemRoot", "WINDIR", "TMP", "TEMP"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), process::Value::Public(value));
        }
    }
    for (key, value) in [
        ("GIT_MASTER", "1"),
        ("GIT_OPTIONAL_LOCKS", "0"),
        ("GIT_TERMINAL_PROMPT", "0"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        (
            "GIT_CONFIG_GLOBAL",
            if cfg!(windows) { "NUL" } else { "/dev/null" },
        ),
    ] {
        environment.insert(key.into(), process::Value::Public(value.into()));
    }
    let mut args = vec![
        "-c",
        "user.name=Native Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
    ];
    args.extend_from_slice(arguments);
    let report = process::supervise_raw(
        &process::ProcessSpec {
            executable,
            arguments: args
                .into_iter()
                .map(|arg| process::Value::Public(arg.into()))
                .collect(),
            cwd: root.into(),
            environment,
        },
        &process::Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(report.process.success(), "{:?}", report.process);
    assert_eq!(
        report.stdout.as_ref().unwrap().as_bytes().len() as u64,
        report.process.stdout.bytes_seen
    );
    assert_eq!(
        report.stderr.as_ref().unwrap().as_bytes().len() as u64,
        report.process.stderr.bytes_seen
    );
    String::from_utf8(report.stdout.unwrap().as_bytes().to_vec())
        .unwrap()
        .trim()
        .to_owned()
}
fn prepared_layout_fixture(root: &Path, path: &Path, previous: &Value) -> Value {
    let mesh = root.join("mesh-source");
    let raw = root.join("raw-source");
    for source in [&mesh, &raw] {
        std::fs::create_dir(source).unwrap();
        local_git(source, &["init", "--quiet"]);
        local_git(
            source,
            &[
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "owned source fixture",
            ],
        );
    }
    let expected = local_git(&raw, &["rev-parse", "HEAD"]);
    let config_path = Path::new(previous["config"].as_str().unwrap());
    let mut config = json_file(config_path);
    config["baseline"]["llama_cpp_revision"] = expected.clone().into();
    let bytes = serde_json::to_vec(&config).unwrap();
    std::fs::write(config_path, &bytes).unwrap();
    let models = root.join("models");
    let model_dir = models.join("llama32-dense");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::copy(
        previous["model"]["path"].as_str().unwrap(),
        model_dir.join(config["models"][0]["filename"].as_str().unwrap()),
    )
    .unwrap();
    let tokenizers = root.join("tokenizers");
    std::fs::create_dir(&tokenizers).unwrap();
    let tokenizer = tokenizers.join("llama32-dense");
    std::fs::create_dir(&tokenizer).unwrap();
    std::fs::copy(
        root.join("tokenizer/tokenizer.json"),
        tokenizer.join("tokenizer.json"),
    )
    .unwrap();
    #[cfg(unix)]
    {
        let model = model_dir.join(config["models"][0]["filename"].as_str().unwrap());
        let target = root.join("materialized-model.gguf");
        std::fs::rename(&model, &target).unwrap();
        std::os::unix::fs::symlink(&target, &model).unwrap();
    }
    let request = json!({"config":config_path,"platform":"metal","model_keys":["llama32-dense"],"workloads":["synthetic"],"model_root":models,"tokenizer_root":tokenizers,"mesh_root":mesh,"llama_root":raw,"mesh_binary":previous["backend"]["executable"]["path"],"llama_binary":previous["backend"]["executable"]["path"],"native_runtime":previous["backend"]["runtime"]["path"],"manifest":null,"benchy":previous["benchy"]["path"],"optional_backends":{},"required_comparisons":[],"adaptive":false,"output":root.join("matrix"),"prepared_output":root.join("prepared-run.json"),"timeout_seconds":120,"cell_timeout_seconds":20,"request_timeout_seconds":2,"preparation_timeout_seconds":60,"resume":false,"force":false});
    std::fs::write(path, serde_json::to_vec(&request).unwrap()).unwrap();
    assert!(
        run_command(root, path, "competitive-prepare", Duration::from_secs(64)),
        "{}",
        std::fs::read_to_string(root.join("last-command-stderr")).unwrap()
    );
    let prepared = json_file(&root.join("prepared-run.json"));
    assert_eq!(prepared["source_context"]["llama_head"], expected);
    assert_eq!(
        prepared["source_context"]["custody_scope"],
        "local_checkout_and_artifact_observation_no_build_attestation"
    );
    assert_eq!(
        prepared["models"][0]["backends"]["mesh"]["version_sha256"],
        digest(b"fixture version 1")
    );
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let model = model_dir.join(config["models"][0]["filename"].as_str().unwrap());
        let target = root.join("materialized-model.gguf");
        assert_eq!(
            prepared["models"][0]["model"]["path"],
            json!(target.canonicalize().unwrap())
        );
        std::fs::remove_file(&model).unwrap();
        let name = std::ffi::CString::new(model.as_os_str().as_bytes()).unwrap();
        // SAFETY: owned private pathname; CString remains valid for this call.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let mut fifo = request.clone();
        fifo["prepared_output"] = json!(root.join("fifo-must-not-prepare.json"));
        std::fs::write(path, serde_json::to_vec(&fifo).unwrap()).unwrap();
        assert!(!run_command(
            root,
            path,
            "competitive-prepare",
            Duration::from_secs(10)
        ));
        assert!(
            std::fs::read_to_string(root.join("last-command-stderr"))
                .unwrap()
                .contains("not a regular file")
        );
        assert!(!root.join("fifo-must-not-prepare.json").exists());
        std::fs::remove_file(&model).unwrap();
        std::os::unix::fs::symlink(&target, &model).unwrap();
    }
    let mut negative = request;
    negative["prepared_output"] = json!(root.join("must-not-prepare.json"));
    std::fs::write(path, serde_json::to_vec(&negative).unwrap()).unwrap();
    local_git(
        &raw,
        &["commit", "--allow-empty", "--quiet", "-m", "source drift"],
    );
    assert!(!run_command(
        root,
        path,
        "competitive-prepare",
        Duration::from_secs(64)
    ));
    assert!(!root.join("must-not-prepare.json").exists());
    local_git(&raw, &["reset", "--hard", "--quiet", &expected]);
    prepared
}
