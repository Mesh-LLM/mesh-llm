//! Actual local xtask CLI with an admitted inert standalone product.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
fn artifact(path: &Path) -> Json {
    json!({"path":path,"sha256":hex::encode(Sha256::digest(std::fs::read(path).unwrap()))})
}
fn fixture(root: &Path, mode: &str, mtp: bool) -> Json {
    let canonical_root = root.canonicalize().unwrap();
    let root = canonical_root.as_path();
    use std::os::unix::fs::PermissionsExt as _;
    let example = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_daemon_fixture");
    assert!(example.is_file());
    let binary = root.join("skippy-quantize");
    std::fs::write(
        &binary,
        format!(
            "#!/bin/sh\nexec '{}' \"$@\"\n",
            example.to_str().unwrap().replace('\'', "'\\''")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let projector = root.join("projector.gguf");
    std::fs::write(&projector, format!("GGUF{mode}")).unwrap();
    let first = root.join("target-00001-of-00002.gguf");
    let last = root.join("target-00002-of-00002.gguf");
    let draft = root.join("mtp.gguf");
    for p in [&first, &last, &draft] {
        std::fs::write(p, b"GGUFinert").unwrap();
    }
    json!({"schema_version":1,"mode":if mtp{"mtp-attach"}else{"projector-only"},"binary":artifact(&binary),"supplied_mesh_revision":"a".repeat(40),"native_profile":"standalone-static-skippy-quantize-cpu","projector":artifact(&projector),"target_parts":if mtp{vec![artifact(&first),artifact(&last)]}else{vec![]},"expected_parts":if mtp{2}else{0},"mtp_draft":if mtp{artifact(&draft)}else{Json::Null},"layer_count":2,"mtp_layer_count":if mtp{json!(1)}else{Json::Null},"ctx_size":64,"timeout_secs":15})
}
fn command(root: &Path, cancel: &Cancellation) -> process::RawProcessReport {
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: [
            "automation",
            "hf-certify",
            "acquire",
            "--input",
            "input.json",
            "--output-directory",
            "evidence",
        ]
        .into_iter()
        .map(|s| Value::Public(s.into()))
        .collect(),
        cwd: root.into(),
        environment: [("PATH".into(), Value::Public("/usr/bin:/bin".into()))]
            .into_iter()
            .collect(),
    };
    let limits = Limits {
        execution: Duration::from_secs(22),
        graceful_shutdown: Duration::from_secs(4),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 1048576,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        cancel,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1048576),
            stderr: NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete
            && !report.process.cleanup.forced
            && !report.process.cleanup.graceful_signal_failed
            && report.process.cleanup.failure.is_none()
            && report.process.failure.is_none(),
        "{report:?}"
    );
    for (stream, raw) in [
        (&report.process.stdout, report.stdout.as_ref()),
        (&report.process.stderr, report.stderr.as_ref()),
    ] {
        assert!(!stream.truncated && stream.line_capture_complete);
        assert_eq!(stream.bytes_seen, raw.unwrap().as_bytes().len() as u64)
    }
    report
}
fn write(root: &Path, value: &Json) {
    std::fs::write(
        root.join("input.json"),
        serde_json::to_vec_pretty(&json!({"schema_version":1,"certification":value,"projector":{"kind":"supplied","artifact":value["projector"]}})).unwrap(),
    )
    .unwrap();
}
fn receipt(root: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(root.join("evidence/acquisition-report.json")).unwrap())
        .unwrap()
}
#[test]
fn hf_acquire_actual_supplied_projector_and_attach_keep_exact_pins_and_no_partial_pass() {
    for mtp in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let input = fixture(root.path(), "success", mtp);
        write(root.path(), &input);
        let result = command(root.path(), &Cancellation::default());
        assert_eq!(result.process.outcome, process::Outcome::Exited);
        assert_eq!(result.process.status.unwrap().code(), Some(0));
        let r = receipt(root.path());
        assert_eq!(r["status"], "PASS");
        assert_eq!(r["acquisition"]["status"], "PIN_OBSERVED_BEFORE_AND_AFTER");
        assert_eq!(r["certification"]["source_unchanged"], true);
        assert_eq!(
            r["certification"]["native_report"]["projector"],
            input["projector"]["path"]
        );
        assert_eq!(
            r["certification"]["admitted"]["projector"]["sha256"],
            input["projector"]["sha256"]
        );
        assert!(!root.path().join("evidence/projector.gguf").exists());
        root.close().unwrap();
    }
}
#[test]
fn hf_acquire_actual_origin_refusal_and_native_late_failure_never_publish_pass() {
    for mode in ["origin", "wrong-path", "mutate", "nonzero"] {
        let root = tempfile::tempdir().unwrap();
        let input = fixture(
            root.path(),
            if mode == "origin" { "success" } else { mode },
            true,
        );
        if mode == "origin" {
            let request = json!({"schema_version":1,"certification":input,"projector":{"kind":"hf-https","url":"https://localhost/projector","expected_sha256":input["projector"]["sha256"],"max_bytes":64}});
            std::fs::write(
                root.path().join("input.json"),
                serde_json::to_vec(&request).unwrap(),
            )
            .unwrap();
        } else {
            write(root.path(), &input);
        }
        let result = command(root.path(), &Cancellation::default());
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        if mode == "origin" {
            assert!(!root.path().join("evidence").exists());
        } else {
            let r = receipt(root.path());
            assert_eq!(r["status"], "FAILED");
            assert_eq!(r["certification"]["status"], "FAILED");
            assert_eq!(r["certification"]["source_unchanged"], false);
            assert!(
                root.path()
                    .join("evidence/certification/invoked.json")
                    .exists()
            );
        }
        root.close().unwrap();
    }
}
#[test]
fn hf_acquire_actual_inflight_tool_cancel_and_shared_deadline_reap_without_partial_pass() {
    for explicit in [true, false] {
        let root = tempfile::tempdir().unwrap();
        let mut input = fixture(root.path(), "held", true);
        if !explicit {
            input["timeout_secs"] = json!(5);
        }
        write(root.path(), &input);
        let token = Cancellation::default();
        let (result, seen) = std::thread::scope(|scope| {
            let worker = scope.spawn(|| command(root.path(), &token));
            let until = Instant::now() + Duration::from_secs(10);
            let marker = root.path().join("evidence/certification/invoked.json");
            while !marker.exists() && Instant::now() < until && !worker.is_finished() {
                std::thread::sleep(Duration::from_millis(5));
            }
            let seen = marker.exists();
            if explicit || !seen {
                token.cancel();
            }
            (worker.join().unwrap(), seen)
        });
        assert!(seen);
        assert_eq!(token.is_cancelled(), explicit);
        assert_eq!(receipt(root.path())["status"], "FAILED");
        assert!(root.path().join("evidence/certification/stopped").exists());
        if explicit {
            assert_eq!(result.process.outcome, process::Outcome::Cancelled);
        } else {
            assert_eq!(result.process.status.unwrap().code(), Some(1));
        }
        root.close().unwrap();
    }
}
