//! Actual native local caller with inert product tool, no model or HF execution.
use crate::process;
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn sha(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
struct Fixture {
    root: tempfile::TempDir,
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    parts: Vec<std::path::PathBuf>,
}
impl Fixture {
    fn new(mode: &str, parts: usize) -> Self {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().canonicalize().unwrap();
        let binary = path.join("product");
        let source = Path::new(env!("CARGO_BIN_EXE_xtask"))
            .parent()
            .unwrap()
            .join("examples/l7_daemon_fixture");
        std::fs::copy(source, &binary).unwrap();
        let artifact =
            |path: &Path| json!({"path":path,"sha256":sha(&std::fs::read(path).unwrap())});
        let mtp = path.join("mtp.gguf");
        std::fs::write(&mtp, format!("GGUF{mode}")).unwrap();
        let paths = (1..=parts)
            .map(|i| {
                let p = path.join(format!("Target-{i:05}-of-{parts:05}.gguf"));
                std::fs::write(&p, format!("GGUFtarget-{i}")).unwrap();
                p
            })
            .collect::<Vec<_>>();
        let input = path.join("input.json");
        std::fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"binary":artifact(&binary),"target_parts":paths.iter().map(|p|artifact(p)).collect::<Vec<_>>(),"mtp_gguf":artifact(&mtp),"target_basename":"Target","composite_basename":"Composite","expected_parts":parts,"mtp_block":88,"supplied_mesh_revision":"a".repeat(40),"native_profile":"standalone-static-skippy-quantize-cpu","timeout_secs":10,"composite_repo":"fixture/composite"})).unwrap()).unwrap();
        Self {
            root,
            input,
            output: path.join("output"),
            parts: paths,
        }
    }
    fn spec(&self) -> process::ProcessSpec {
        process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation".into(),
                "hf-mtp-compose".into(),
                "--input".into(),
                self.input.as_os_str().into(),
                "--output-directory".into(),
                self.output.as_os_str().into(),
            ]
            .into_iter()
            .map(process::Value::Public)
            .collect(),
            cwd: self.root.path().canonicalize().unwrap(),
            environment: Default::default(),
        }
    }
    fn run(&self, cancel: &process::Cancellation) -> process::RawProcessReport {
        let raw = process::supervise_raw(
            &self.spec(),
            &process::Limits {
                execution: Duration::from_secs(15),
                graceful_shutdown: Duration::from_secs(4),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            cancel,
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        let p = &raw.process;
        assert!(
            p.failure.is_none()
                && p.cleanup.complete
                && !p.cleanup.forced
                && !p.cleanup.graceful_signal_failed
                && p.cleanup.failure.is_none()
        );
        for (s, r) in [
            (&p.stdout, raw.stdout.as_ref().unwrap()),
            (&p.stderr, raw.stderr.as_ref().unwrap()),
        ] {
            assert!(s.line_capture_complete);
            assert_eq!(s.bytes_seen, r.as_bytes().len() as u64);
        }
        raw
    }
    fn json(&self, name: &str) -> Value {
        serde_json::from_slice(&std::fs::read(self.output.join(name)).unwrap()).unwrap()
    }
}
#[test]
fn hf_compose_complete_split_publication_retains_middle_and_no_projector_dependency() {
    for count in [2, 3] {
        let f = Fixture::new("success", count);
        let before = f
            .parts
            .iter()
            .map(|p| std::fs::read(p).unwrap())
            .collect::<Vec<_>>();
        assert!(f.run(&process::Cancellation::default()).process.success());
        let report = f.json("report.json");
        assert_eq!(report["status"], "PASS");
        assert_eq!(report["source_unchanged"], true);
        let plan = f.json("publication-plan.json");
        let entries = plan["entries"].as_array().unwrap();
        assert_eq!(entries.len(), count);
        assert_eq!(plan["status"], "PLAN_ONLY_NOT_PUBLISHED");
        assert!(plan["upload_execution"].is_null());
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(
                entry["path_in_repo"],
                format!("Composite-{:05}-of-{count:05}.gguf", i + 1)
            );
            let path = entry["local"].as_str().unwrap();
            assert_eq!(entry["sha256"], sha(&std::fs::read(path).unwrap()));
            assert_eq!(std::fs::read(&f.parts[i]).unwrap(), before[i]);
            if i > 0 && i + 1 < count {
                assert_eq!(Path::new(path), f.parts[i]);
                assert_eq!(entry["source_kind"], "untouched-pinned-middle");
            }
        }
        let calls = std::fs::read_to_string(f.output.join("invocations.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Vec<String>>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0][0], "compose-mtp");
        assert_eq!(calls[1][0], "validate-mtp-attach");
        assert!(!calls[1].contains(&"--projector".into()));
        assert_eq!(
            report["validation_report"]["model_parts"]
                .as_array()
                .unwrap()
                .len(),
            count
        );
        f.root.close().unwrap();
    }
}
#[test]
fn hf_compose_bad_native_reports_missing_outputs_and_failures_never_publish() {
    for mode in [
        "nonzero",
        "malformed",
        "wrong-path",
        "missing-output",
        "no-feature",
    ] {
        let f = Fixture::new(mode, 3);
        let r = f.run(&process::Cancellation::default());
        assert_eq!(r.process.status.unwrap().code(), Some(1));
        assert_eq!(f.json("report.json")["status"], "FAILED");
        assert!(!f.output.join("publication-plan.json").exists());
        assert!(f.output.join("compose-process.json").is_file());
        f.root.close().unwrap();
    }
}
#[test]
fn hf_compose_source_and_composite_drift_after_attachment_refuse_publication() {
    for mode in ["mutate-middle", "mutate-output"] {
        let f = Fixture::new(mode, 3);
        assert_eq!(
            f.run(&process::Cancellation::default())
                .process
                .status
                .unwrap()
                .code(),
            Some(1)
        );
        let report = f.json("report.json");
        assert_eq!(report["status"], "FAILED");
        assert_eq!(report["source_unchanged"], false);
        assert!(!f.output.join("publication-plan.json").exists());
        assert!(f.output.join("validation-process.json").is_file());
        f.root.close().unwrap();
    }
}
#[test]
fn hf_compose_wrong_pin_incomplete_roster_and_fifo_refuse_before_product() {
    for mode in ["pin", "roster", "fifo"] {
        let mut f = Fixture::new("success", 3);
        let mut value: Value = serde_json::from_slice(&std::fs::read(&f.input).unwrap()).unwrap();
        if mode == "pin" {
            value["mtp_gguf"]["sha256"] = json!("0".repeat(64));
        }
        if mode == "roster" {
            value["target_parts"].as_array_mut().unwrap().remove(1);
        }
        std::fs::write(&f.input, serde_json::to_vec(&value).unwrap()).unwrap();
        if mode == "fifo" {
            let path = f.root.path().join("fifo");
            let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
            f.input = path;
        }
        assert_eq!(
            f.run(&process::Cancellation::default())
                .process
                .status
                .unwrap()
                .code(),
            Some(1)
        );
        assert!(!f.output.join("invocations.jsonl").exists());
        assert!(!f.output.join("publication-plan.json").exists());
        f.root.close().unwrap();
    }
}
#[test]
fn hf_compose_held_child_cancel_and_deadline_keep_failure_receipt_and_cleanup() {
    for cancellation in [true, false] {
        let f = Fixture::new("held", 3);
        if !cancellation {
            let mut value: Value =
                serde_json::from_slice(&std::fs::read(&f.input).unwrap()).unwrap();
            value["timeout_secs"] = json!(5);
            std::fs::write(&f.input, serde_json::to_vec(&value).unwrap()).unwrap();
        }
        let cancel = process::Cancellation::default();
        let observer = if cancellation {
            let marker = f.output.join("held-marker");
            let signal = cancel.clone();
            Some(std::thread::spawn(move || {
                let until = Instant::now() + Duration::from_secs(5);
                while !marker.exists() && Instant::now() < until {
                    std::thread::sleep(Duration::from_millis(5));
                }
                let seen = marker.exists();
                signal.cancel();
                seen
            }))
        } else {
            None
        };
        let raw = f.run(&cancel);
        let seen = observer.map(|t| t.join().unwrap());
        if cancellation {
            assert_eq!(seen, Some(true));
        }
        assert!(!raw.process.status.unwrap().success());
        assert_eq!(f.json("report.json")["status"], "FAILED");
        assert!(!f.output.join("publication-plan.json").exists());
        f.root.close().unwrap();
    }
}
