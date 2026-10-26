//! Native generic materialization boundaries; no HF or real dataset download.
use crate::process;
use std::{
    collections::BTreeMap, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt as _, path::Path,
    time::Duration,
};

fn invoke(root: &Path, body: &str, extra: &[(&str, String)]) -> (bool, String, String) {
    let mut environment = BTreeMap::from([
        (
            "PATH".into(),
            process::Value::Public("/usr/bin:/bin".into()),
        ),
        (
            "AUTOMATION".into(),
            process::Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        ("ROOT".into(), process::Value::Public(root.into())),
    ]);
    for (key, value) in extra {
        environment.insert((*key).into(), process::Value::Public(value.as_str().into()));
    }
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            environment,
            arguments: ["-euo", "pipefail", "-c", body]
                .map(|v| process::Value::Public(v.into()))
                .into(),
        },
        &process::Limits {
            execution: Duration::from_secs(10),
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
    let out = raw.stdout.unwrap();
    let err = raw.stderr.unwrap();
    assert_eq!(out.as_bytes().len() as u64, report.stdout.bytes_seen);
    assert_eq!(err.as_bytes().len() as u64, report.stderr.bytes_seen);
    (
        report.status.unwrap().success(),
        String::from_utf8(out.as_bytes().to_vec()).unwrap(),
        String::from_utf8(err.as_bytes().to_vec()).unwrap(),
    )
}
fn materializer() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/materialize-competitive-inputs.sh")
        .canonicalize()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

#[test]
fn materializer_runs_actual_native_frontend_and_rejects_legacy_options() {
    let temporary = tempfile::tempdir().unwrap();
    let extra = [
        ("MATERIALIZER", materializer()),
        (
            "MESH_LLM_AUTOMATION_BIN",
            env!("CARGO_BIN_EXE_xtask").into(),
        ),
    ];
    let (ok, output, error) = invoke(
        temporary.path(),
        "exec /bin/bash \"$MATERIALIZER\" --help",
        &extra,
    );
    assert!(ok, "{error}");
    assert!(output.contains("competitive-inputs-prefetch --request ABS"));
    let (ok, _, error) = invoke(
        temporary.path(),
        "exec /bin/bash \"$MATERIALIZER\" --models obsolete",
        &extra,
    );
    assert!(!ok);
    assert!(
        error.contains("unrecognized arguments: --models"),
        "{error}"
    );
    temporary.close().unwrap();
}

#[test]
fn materializer_preserves_request_paths_and_refuses_wrong_helper_before_execution() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let helper = root.join("helper with spaces");
    fs::write(
        &helper,
        format!(
            "#!/bin/sh\nprintf invoked > '{}'/helper-called\n",
            root.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(
        root.join("request with spaces.json"),
        r#"{"timeout_seconds":10}"#,
    )
    .unwrap();
    let (ok, _, error) = invoke(
        &root,
        "exec /bin/bash \"$MATERIALIZER\" --request \"$ROOT/request with spaces.json\" --helper \"$ROOT/helper with spaces\" --helper-sha256 \"$HELPER_SHA\" --evidence-directory \"$ROOT/evidence\" --timeout-seconds 10",
        &[
            ("MATERIALIZER", materializer()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").into(),
            ),
            ("HELPER_SHA", "0".repeat(64)),
        ],
    );
    assert!(!ok);
    assert!(error.contains("helper byte identity mismatch"), "{error}");
    assert!(!root.join("helper-called").exists());
    assert!(!root.join("evidence").exists());
    temporary.close().unwrap();
}

#[test]
fn materializer_missing_prepared_executor_refuses_without_creating_inputs() {
    let temporary = tempfile::tempdir().unwrap();
    let (ok, _, error) = invoke(
        temporary.path(),
        "exec /bin/bash \"$MATERIALIZER\" --help",
        &[
            ("MATERIALIZER", materializer()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                temporary
                    .path()
                    .join("absent executor")
                    .display()
                    .to_string(),
            ),
        ],
    );
    assert!(!ok);
    assert!(error.contains("Prepare native automation"), "{error}");
    assert!(!temporary.path().join("bench-inputs").exists());
    temporary.close().unwrap();
}
