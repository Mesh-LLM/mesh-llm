use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
pub(super) fn run(args: Vec<String>, cwd: &Path) -> process::RawProcessReport {
    run_spec(ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: args
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd: cwd.into(),
        environment: BTreeMap::new(),
    })
}
pub(super) fn run_spec(spec: ProcessSpec) -> process::RawProcessReport {
    let report = process::supervise_raw(
        &spec,
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    report
}

fn command(verb: &str, values: &[&str]) -> Vec<String> {
    ["automation", "system-one-smoke", verb]
        .into_iter()
        .chain(values.iter().copied())
        .map(str::to_owned)
        .collect()
}

#[test]
fn actual_smoke_stage_serializer_retains_native_contract_dimensions_and_optional_identity() {
    let scratch = tempfile::tempdir().unwrap();
    let output = scratch.path().join("stage.json");
    for sha in ["", &"a".repeat(64)] {
        let report = run(
            command(
                "stage",
                &[
                    output.to_str().unwrap(),
                    "fixture-model",
                    "/offline/model.gguf",
                    sha,
                    "24",
                    "127.0.0.1:9339",
                    "2",
                    "8192",
                    "128",
                    "-1",
                ],
            ),
            scratch.path(),
        );
        assert!(report.process.success());
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(config["run_id"], "skippy-system-one-smoke");
        assert_eq!(
            config["topology_id"],
            "skippy-system-one-smoke-single-stage"
        );
        assert_eq!(config["stage_index"], 0);
        assert_eq!(config["layer_start"], 0);
        assert_eq!(config["layer_end"], 24);
        assert_eq!(config["ctx_size"], 8192);
        assert_eq!(config["lane_count"], 2);
        assert_eq!(config["n_batch"], 128);
        assert_eq!(config["n_ubatch"], 128);
        assert_eq!(config["n_gpu_layers"], -1);
        assert_eq!(config["cache_type_k"], "f16");
        assert_eq!(config["cache_type_v"], "f16");
        assert_eq!(config["load_mode"], "runtime-slice");
        assert_eq!(config["execution_contract"], "");
        assert!(config["upstream"].is_null() && config["downstream"].is_null());
        if sha.is_empty() {
            assert!(config.get("source_model_sha256").is_none());
        } else {
            assert_eq!(config["source_model_sha256"], sha);
        }
    }
    fs::write(&output, "old").unwrap();
    let report = run(
        command(
            "stage",
            &[
                output.to_str().unwrap(),
                "fixture-model",
                "/offline/model.gguf",
                "bad",
                "24",
                "127.0.0.1:9339",
                "2",
                "8192",
                "128",
                "0",
            ],
        ),
        scratch.path(),
    );
    assert!(!report.process.success());
    assert_eq!(fs::read_to_string(&output).unwrap(), "old");
}

#[test]
fn actual_smoke_aggregate_serializer_preserves_qualified_unqualified_and_hard_failure_evidence() {
    let scratch = tempfile::tempdir().unwrap();
    let output = scratch.path().join("reports/aggregate.json");
    for (status, contract, read, backend, path, checked, required, skipped) in [
        (
            "pass",
            "pass",
            "pass",
            "cuda",
            "/offline/model.gguf",
            "true",
            "0",
            "0",
        ),
        (
            "unqualified",
            "skipped",
            "unqualified",
            "metal",
            "",
            "false",
            "0",
            "1",
        ),
        (
            "fail",
            "skipped",
            "unqualified",
            "metal",
            "",
            "false",
            "true",
            "true",
        ),
        ("fail", "skipped", "fail", "cuda", "", "true", "0", "1"),
        ("fail", "error", "unqualified", "", "", "false", "0", "0"),
    ] {
        let report = run(
            command(
                "report",
                &[
                    output.to_str().unwrap(),
                    status,
                    contract,
                    read,
                    "openjev-pinned",
                    backend,
                    path,
                    checked,
                    "cuda,,metal",
                    required,
                    skipped,
                    "first reason; second reason",
                ],
            ),
            scratch.path(),
        );
        assert!(report.process.success());
        let data: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(data["schema_version"], 1);
        assert_eq!(data["status"], status);
        assert_eq!(data["contract"]["status"], contract);
        let read_data = &data["full_model_read"];
        assert_eq!(read_data["status"], read);
        assert_eq!(read_data["artifact"], "openjev-pinned");
        assert_eq!(read_data["artifact_cache_checked"], checked == "true");
        assert_eq!(
            read_data["require_qualified"],
            matches!(required, "1" | "true")
        );
        assert_eq!(
            data["contract_part_skipped"],
            matches!(skipped, "1" | "true")
        );
        assert_eq!(
            read_data["certified_backends"],
            serde_json::json!(["cuda", "metal"])
        );
        assert_eq!(
            data["reasons"],
            serde_json::json!(["first reason", "second reason"])
        );
        assert_eq!(read_data["artifact_path"].is_null(), path.is_empty());
        assert_eq!(read_data["backend"].is_null(), backend.is_empty());
        let stdout: serde_json::Value =
            serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap();
        assert_eq!(stdout, data);
    }
}

#[test]
fn actual_whole_smoke_wrapper_preserves_gate_exits_without_python_or_native_execution() {
    let scratch = tempfile::tempdir().unwrap();
    let bin = scratch.path().join("bin");
    fs::create_dir(&bin).unwrap();
    let search = std::env::var_os("PATH").unwrap();
    for name in ["jq", "curl", "mkdir", "dirname", "cat", "bash"] {
        let executable = std::env::split_paths(&search)
            .map(|directory| directory.join(name))
            .find(|path| path.is_file())
            .unwrap_or_else(|| panic!("owning fixture requires {name}"));
        std::os::unix::fs::symlink(executable, bin.join(name)).unwrap();
    }
    assert!(!bin.join("python3").exists());
    let wrapper =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/skippy-system-one-smoke.sh");
    for (backend, required, skip, expected_status, expected_read, expected_code) in [
        ("metal", "0", "1", "unqualified", "unqualified", 0),
        ("metal", "1", "1", "fail", "unqualified", 1),
        ("cuda", "0", "1", "fail", "fail", 1),
        ("metal", "0", "0", "fail", "unqualified", 1),
    ] {
        let output = scratch.path().join("report.json");
        let mut environment = BTreeMap::from([
            ("PATH".into(), Value::Public(bin.clone().into_os_string())),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            (
                "LLAMA_STAGE_BUILD_DIR".into(),
                Value::Public(scratch.path().join("absent-native").into_os_string()),
            ),
            (
                "STAGE_SERVER_BIN".into(),
                Value::Public(scratch.path().join("absent-server").into_os_string()),
            ),
            (
                "MODEL_PACKAGE_BIN".into(),
                Value::Public(scratch.path().join("absent-package").into_os_string()),
            ),
            ("WORK_DIR".into(), Value::Public(scratch.path().into())),
            (
                "SYSTEMONE_SMOKE_REPORT".into(),
                Value::Public(output.clone().into_os_string()),
            ),
            (
                "SYSTEMONE_SMOKE_BUILD_BACKEND".into(),
                Value::Public(backend.into()),
            ),
            (
                "SYSTEMONE_SMOKE_REQUIRE_QUALIFIED".into(),
                Value::Public(required.into()),
            ),
            (
                "SYSTEMONE_SMOKE_SKIP_CONTRACT".into(),
                Value::Public(skip.into()),
            ),
        ]);
        environment.insert("GITHUB_ACTIONS".into(), Value::Public("true".into()));
        let report = run_spec(ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![Value::Public(wrapper.clone().into_os_string())],
            cwd: scratch.path().into(),
            environment,
        });
        assert_eq!(report.process.status.unwrap().code(), Some(expected_code));
        let data: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(data["status"], expected_status);
        if expected_status == "unqualified" {
            let text = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
            assert!(text.contains("NOT CERTIFIED"));
            assert!(text.contains("::warning title=System One smoke::"));
        }
        assert_eq!(data["full_model_read"]["status"], expected_read);
        assert_eq!(
            data["full_model_read"]["artifact_cache_checked"],
            backend == "cuda"
        );
        assert!(data["full_model_read"]["artifact_path"].is_null());
        if skip == "0" {
            assert_eq!(data["contract"]["status"], "error");
        }
    }
}
