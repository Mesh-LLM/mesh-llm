//! Execute supplied observations and the real wrapper with finite private tools only.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt, path::Path,
    time::Duration,
};
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn invoke(root: &Path, body: &str) -> process::RawProcessReport {
    let script = root.join("invoke.sh");
    fs::write(&script, body).unwrap();
    let environment = BTreeMap::from([
        (
            "PATH".into(),
            Value::Public(format!("{}:/usr/bin:/bin", root.join("bin").display()).into()),
        ),
        (
            "XTASK".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        (
            "INPUT".into(),
            Value::Public(root.join("input.json").into_os_string()),
        ),
        (
            "FIXTURE_ROOT".into(),
            Value::Public(root.as_os_str().to_owned()),
        ),
    ]);
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            environment,
            arguments: vec![Value::Public(script.into_os_string())],
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    let p = &result.process;
    assert_eq!(p.outcome, process::Outcome::Exited);
    assert!(p.failure.is_none() && p.cleanup.failure.is_none());
    assert!(p.cleanup.complete && !p.cleanup.forced && !p.cleanup.graceful_signal_failed);
    for (stream, raw) in [(&p.stdout, &result.stdout), (&p.stderr, &result.stderr)] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(
            raw.as_ref().unwrap().as_bytes().len() as u64,
            stream.bytes_seen
        );
    }
    result
}
fn stdout(report: &process::RawProcessReport) -> &[u8] {
    report.stdout.as_ref().unwrap().as_bytes()
}
const ENTRYPOINT: &str = include_str!("../../../../skippy/evals/wan-lab/entrypoint.sh");
fn function(name: &str, next: &str) -> String {
    let start = ENTRYPOINT.find(&format!("{name}() {{")).unwrap();
    let end = ENTRYPOINT[start..].find(&format!("\n{next}() {{")).unwrap() + start;
    ENTRYPOINT[start..end].replace("/usr/local/bin/mesh-llm-automation", "\"$XTASK\"")
}
fn stage_plan(root: &Path) {
    fs::write(root.join("input.json"), br#"{"stage_id":"stage-1","stage_index":1,"source_model_sha256":"admitted-source","resident_tensor_names":["exact-tensor"],"execution_contract":{"id":"preserved"},"activation_import_identities":["frontier-1"],"future":{"nested":[1,false,null]}}"#).unwrap();
}
#[test]
fn actual_wan_projection_cli_preserves_source_and_refuses_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    stage_plan(&root);
    let success = invoke(
        &root,
        "exec \"$XTASK\" automation wan-stage-deployment --input \"$INPUT\" --output \"$FIXTURE_ROOT/output.json\" --stage-index 1 --stage-count 4\n",
    );
    assert!(success.process.status.unwrap().success());
    assert!(stdout(&success).is_empty());
    let original: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("input.json")).unwrap()).unwrap();
    let bytes = fs::read(root.join("output.json")).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let output: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for (key, value) in original.as_object().unwrap() {
        assert_eq!(output.get(key), Some(value));
    }
    for extra in [
        "--stage-index 0 --stage-count 4",
        "--stage-index 1 --stage-count 1",
        "--stage-index 1 --stage-count 4 --bind-port 70000",
    ] {
        let refused = invoke(
            &root,
            &format!(
                "exec \"$XTASK\" automation wan-stage-deployment --input \"$INPUT\" --output \"$FIXTURE_ROOT/refused.json\" {extra}\n"
            ),
        );
        assert!(!refused.process.status.unwrap().success());
        assert!(!root.join("refused.json").exists());
    }
}
#[test]
fn actual_wan_config_function_preserves_supplied_config_and_directory_refusal() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    stage_plan(&root);
    let bytes = fs::read(root.join("input.json")).unwrap();
    let functions = format!(
        "{}\n{}",
        function("require_env", "has_cli_arg"),
        function("write_stage_config", "run_metrics")
    );
    let preserved = invoke(
        &root,
        &format!(
            "set -euo pipefail\nlog() {{ printf '%s\\n' \"$*\" >&2; }}\n{functions}\nCONFIG_PATH=$INPUT\nMODEL_PATH=$FIXTURE_ROOT/missing\nSTAGE_INDEX=1\nwrite_stage_config\n"
        ),
    );
    assert!(preserved.process.status.unwrap().success());
    assert_eq!(fs::read(root.join("input.json")).unwrap(), bytes);
    let refused = invoke(
        &root,
        &format!(
            "set -euo pipefail\nlog() {{ printf '%s\\n' \"$*\" >&2; }}\n{functions}\nMODEL_PATH=$FIXTURE_ROOT\nSTAGE_INDEX=1\nwrite_stage_config\n"
        ),
    );
    assert_eq!(refused.process.status.unwrap().code(), Some(65));
    assert!(!root.join("admission").exists());
}
#[test]
fn actual_wan_direct_file_calls_planner_before_native_projection() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::create_dir(root.join("bin")).unwrap();
    stage_plan(&root);
    executable(
        &root.join("bin/skippy"),
        "#!/bin/bash\nset -euo pipefail\n[[ $1 == plan-split ]]\nwhile [[ $# -gt 0 ]]; do if [[ $1 == --output-dir ]]; then output=$2; break; fi; shift; done\nmkdir \"$output\"\ncp \"$INPUT\" \"$output/stage-1.json\"\n",
    );
    let functions = format!(
        "{}\n{}",
        function("require_env", "has_cli_arg"),
        function("write_stage_config", "run_metrics")
    );
    let result = invoke(
        &root,
        &format!(
            "set -euo pipefail\nlog() {{ printf '%s\\n' \"$*\" >&2; }}\n{functions}\nMODEL_PATH=$INPUT\nSTAGE_INDEX=1\nSTAGE_COUNT=4\nCONFIG_DIR=$FIXTURE_ROOT/configs\nN_BATCH=1024\nwrite_stage_config\n"
        ),
    );
    assert!(result.process.status.unwrap().success());
    let output: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("configs/stage-1.json")).unwrap()).unwrap();
    assert_eq!(output["n_batch"], 1024);
    assert_eq!(output["execution_contract"]["id"], "preserved");
    assert_eq!(output["upstream"]["endpoint"], "tcp://stage0:19000");
}
#[test]
fn wan_image_delivers_exact_built_native_helper_without_executable_override() {
    let docker = include_str!("../../../../skippy/evals/wan-lab/Dockerfile");
    let recipes = include_str!("../../../../just/skippy.just");
    assert!(recipes.contains("just with-lld cargo build --release --locked -p xtask --bin xtask"));
    assert!(docker.contains("cp /src/target/release/xtask /out/mesh-llm-automation"));
    assert!(docker.contains(
        "COPY --from=builder /out/mesh-llm-automation /usr/local/bin/mesh-llm-automation"
    ));
    assert!(
        ENTRYPOINT.contains("/usr/local/bin/mesh-llm-automation automation wan-observation delay")
    );
    assert!(!ENTRYPOINT.contains("MESH_LLM_AUTOMATION_BIN"));
}
#[test]
fn actual_wan_shaping_uses_native_half_rtt_and_explicit_delay_precedence() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::create_dir(root.join("bin")).unwrap();
    executable(
        &root.join("bin/tc"),
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FIXTURE_ROOT/tc.log\"\n",
    );
    let functions = format!(
        "{}\n{}",
        function("float_half", "parse_hf_package_ref"),
        function("apply_linux_wan", "write_stage_config")
    );
    for (controls, expected) in [
        ("WAN_RTT_MS=20.25", "delay 10.125ms"),
        ("WAN_RTT_MS=NaN\nWAN_DELAY_MS=7", "delay 7ms"),
    ] {
        let result = invoke(
            &root,
            &format!(
                "set -euo pipefail\nlog() {{ :; }}\n{functions}\nWAN_IFACE=eth0\n{controls}\napply_linux_wan\n"
            ),
        );
        assert!(result.process.status.unwrap().success());
        assert!(
            fs::read_to_string(root.join("tc.log"))
                .unwrap()
                .contains(expected)
        );
        fs::remove_file(root.join("tc.log")).unwrap();
    }
    for invalid in ["NaN", "-1", "inf"] {
        let refused = invoke(
            &root,
            &format!(
                "set -euo pipefail\nlog() {{ :; }}\n{functions}\nWAN_IFACE=eth0\nWAN_RTT_MS={invalid}\napply_linux_wan\n"
            ),
        );
        assert!(!refused.process.status.unwrap().success());
        assert!(!root.join("tc.log").exists());
    }
}

#[test]
fn actual_wan_planner_refusal_stops_before_projection_publication() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::create_dir(root.join("bin")).unwrap();
    stage_plan(&root);
    executable(&root.join("bin/skippy"), "#!/bin/bash\nexit 43\n");
    let functions = format!(
        "{}\n{}",
        function("require_env", "has_cli_arg"),
        function("write_stage_config", "run_metrics")
    );
    let refused = invoke(
        &root,
        &format!(
            "set -euo pipefail\nlog() {{ :; }}\n{functions}\nMODEL_PATH=$INPUT\nSTAGE_INDEX=1\nCONFIG_DIR=$FIXTURE_ROOT/configs\nwrite_stage_config\n"
        ),
    );
    assert_eq!(refused.process.status.unwrap().code(), Some(43));
    assert!(!root.join("configs/stage-1.json").exists());
}

#[test]
fn actual_wan_generated_config_restarts_with_updated_controls_and_preserved_contract() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::create_dir(root.join("bin")).unwrap();
    stage_plan(&root);
    executable(
        &root.join("bin/skippy"),
        "#!/bin/bash\nset -euo pipefail\n[[ $1 == plan-split ]]\nwhile [[ $# -gt 0 ]]; do if [[ $1 == --output-dir ]]; then output=$2; break; fi; shift; done\nmkdir \"$output\"\ncp \"$INPUT\" \"$output/stage-1.json\"\n",
    );
    let functions = format!(
        "{}\n{}",
        function("require_env", "has_cli_arg"),
        function("write_stage_config", "run_metrics")
    );
    let result = invoke(
        &root,
        &format!(
            "set -euo pipefail\nlog() {{ :; }}\n{functions}\nMODEL_PATH=$INPUT\nSTAGE_INDEX=1\nSTAGE_COUNT=4\nCONFIG_DIR=$FIXTURE_ROOT/configs\nN_BATCH=1024\nwrite_stage_config\ncp \"$CONFIG_DIR/stage-1.json\" \"$FIXTURE_ROOT/first.json\"\nN_BATCH=2048\nwrite_stage_config\n"
        ),
    );
    assert!(result.process.status.unwrap().success());
    let first: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("first.json")).unwrap()).unwrap();
    let second: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("configs/stage-1.json")).unwrap()).unwrap();
    assert_eq!(first["n_batch"], 1024);
    assert_eq!(second["n_batch"], 2048);
    for field in [
        "source_model_sha256",
        "resident_tensor_names",
        "execution_contract",
        "activation_import_identities",
        "future",
    ] {
        assert_eq!(first[field], second[field]);
    }
}

#[path = "wan_stage_deployment/entrypoint.rs"]
mod entrypoint;
