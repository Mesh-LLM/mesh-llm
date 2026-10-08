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
        serde_json::to_vec_pretty(value).unwrap(),
    )
    .unwrap();
}
fn receipt(root: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(root.join("evidence/report.json")).unwrap()).unwrap()
}
#[test]
fn hf_certification_cli_executes_exact_projector_and_ordered_mtp_modes() {
    for mtp in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let input = fixture(root.path(), "success", mtp);
        write(root.path(), &input);
        let result = command(root.path(), &Cancellation::default());
        assert_eq!(result.process.outcome, process::Outcome::Exited);
        assert_eq!(result.process.status.unwrap().code(), Some(0));
        let report = receipt(root.path());
        assert_eq!(report["status"], "PASS");
        assert_eq!(report["source_unchanged"], true);
        let args: Vec<String> = serde_json::from_slice(
            &std::fs::read(root.path().join("evidence/invoked.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            args[0],
            if mtp {
                "validate-mtp-attach"
            } else {
                "validate-projector"
            }
        );
        assert_eq!(args.last().unwrap(), "--json");
        assert_eq!(
            report["admitted"]["binary"]["sha256"],
            input["binary"]["sha256"]
        );
        if mtp {
            assert_eq!(
                report["native_report"]["model_parts"],
                json!([
                    input["target_parts"][0]["path"],
                    input["target_parts"][1]["path"]
                ])
            );
            assert_eq!(report["native_report"]["ctx_size"], 64);
            assert_eq!(report["native_report"]["mtp_layer_count"], 1);
        }
        root.close().unwrap();
    }
}
#[test]
fn hf_certification_cli_refuses_wrong_digest_magic_roster_and_prior_output_before_product() {
    for attack in ["digest", "magic", "roster", "duplicate", "prior"] {
        let root = tempfile::tempdir().unwrap();
        let mut input = fixture(root.path(), "success", true);
        match attack {
            "digest" => input["binary"]["sha256"] = json!("0".repeat(64)),
            "magic" => {
                std::fs::write(root.path().join("projector.gguf"), b"BAD!inert").unwrap();
                input["projector"] = artifact(&root.path().join("projector.gguf"));
            }
            "roster" => input["expected_parts"] = json!(3),
            "duplicate" => input["target_parts"][1] = input["target_parts"][0].clone(),
            "prior" => {
                std::fs::create_dir(root.path().join("evidence")).unwrap();
                std::fs::write(root.path().join("evidence/prior"), b"preserve").unwrap();
            }
            _ => unreachable!(),
        };
        write(root.path(), &input);
        let report = command(root.path(), &Cancellation::default());
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(!root.path().join("evidence/invoked.json").exists());
        if attack == "prior" {
            assert_eq!(
                std::fs::read(root.path().join("evidence/prior")).unwrap(),
                b"preserve"
            );
            assert!(!root.path().join("evidence/report.json").exists());
        }
        root.close().unwrap();
    }
}
#[test]
fn hf_certification_cli_retains_native_failure_and_refuses_report_or_source_drift() {
    for mode in [
        "nonzero",
        "malformed",
        "wrong-path",
        "zero-feature",
        "mutate",
    ] {
        let root = tempfile::tempdir().unwrap();
        let input = fixture(root.path(), mode, true);
        write(root.path(), &input);
        let result = command(root.path(), &Cancellation::default());
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let report = receipt(root.path());
        assert_eq!(report["status"], "FAILED");
        assert_eq!(report["source_unchanged"], false);
        assert!(report["error"].is_string());
        assert!(root.path().join("evidence/invoked.json").exists());
        assert!(root.path().join("evidence/validation-stderr.log").is_file());
        root.close().unwrap();
    }
}
#[test]
fn hf_certification_cli_cancel_and_deadline_stop_owned_inflight_product_and_publish_failure() {
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
            let deadline = Instant::now() + Duration::from_secs(10);
            let marker = root.path().join("evidence/invoked.json");
            while !marker.exists() && Instant::now() < deadline && !worker.is_finished() {
                std::thread::sleep(Duration::from_millis(5));
            }
            let seen = marker.exists();
            if explicit || !seen {
                token.cancel();
            }
            let result = worker.join().unwrap();
            (result, seen)
        });
        assert!(seen, "inflight product marker required");
        assert_eq!(token.is_cancelled(), explicit);
        if explicit {
            assert_eq!(result.process.outcome, process::Outcome::Cancelled)
        } else {
            assert_eq!(result.process.outcome, process::Outcome::Exited);
            assert_eq!(result.process.status.unwrap().code(), Some(1));
        }
        assert_eq!(receipt(root.path())["status"], "FAILED");
        assert!(root.path().join("evidence/stopped").exists());
        root.close().unwrap();
    }
}
#[test]
fn hf_certification_cli_static_fifo_input_refuses_without_writer_or_output() {
    use std::os::unix::ffi::OsStrExt as _;
    let root = tempfile::tempdir().unwrap();
    let name =
        std::ffi::CString::new(root.path().join("input.json").as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let result = command(root.path(), &Cancellation::default());
    assert_eq!(result.process.outcome, process::Outcome::Exited);
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    assert!(!root.path().join("evidence").exists());
    root.close().unwrap();
}
