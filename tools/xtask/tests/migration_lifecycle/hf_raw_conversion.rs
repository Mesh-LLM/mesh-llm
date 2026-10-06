//! Real automation CLI with inert conversion/compose tools; no native model claim.
use crate::process;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
fn sha(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
struct Fixture {
    root: tempfile::TempDir,
    input: PathBuf,
    output: PathBuf,
    parts: Vec<PathBuf>,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().canonicalize().unwrap();
        let source = path.join("source");
        std::fs::create_dir(&source).unwrap();
        for (name, bytes) in [
            ("config.json", b"{}".as_slice()),
            ("tokenizer.json", b"{}".as_slice()),
            ("head.safetensors", b"inert supplied bytes".as_slice()),
            ("fixture-mode", mode.as_bytes()),
        ] {
            std::fs::write(source.join(name), bytes).unwrap();
        }
        let profile = path.join("profile.json");
        std::fs::write(&profile, b"{}").unwrap();
        let artifact =
            |path: &Path| json!({"path":path,"sha256":sha(&std::fs::read(path).unwrap())});
        let binary = path.join("product");
        let example = Path::new(env!("CARGO_BIN_EXE_xtask"))
            .parent()
            .unwrap()
            .join("examples/l7_daemon_fixture");
        std::fs::copy(example, &binary).unwrap();
        let parts = (1..=3)
            .map(|i| {
                let p = path.join(format!("Target-{i:05}-of-00003.gguf"));
                std::fs::write(&p, format!("GGUFpart{i}")).unwrap();
                p
            })
            .collect::<Vec<_>>();
        let files = [
            "config.json",
            "tokenizer.json",
            "head.safetensors",
            "fixture-mode",
        ]
        .iter()
        .map(|name| artifact(&source.join(name)))
        .collect::<Vec<_>>();
        let input = path.join("input.json");
        let value = json!({"schema_version":1,"binary":artifact(&binary),"supplied_mesh_revision":"b".repeat(40),"timeout_secs":30,"conversion":{"checkpoint_directory":source,"checkpoint_files":files,"tokenizer_profile":artifact(&profile),"target_parts":parts.iter().map(|p|artifact(p)).collect::<Vec<_>>(),"target_basename":"Target","composite_basename":"Composite","expected_parts":3,"mtp_block":3,"composite_repo":"fixture/composite"}});
        std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
        let output = path.join("output");
        Self {
            root,
            input,
            output,
            parts,
        }
    }
    fn run(&self, cancel: &process::Cancellation) -> process::RawProcessReport {
        let spec = process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation",
                "hf-mtp-compose",
                "raw-checkpoint",
                "--input",
                self.input.to_str().unwrap(),
                "--output-directory",
                self.output.to_str().unwrap(),
            ]
            .iter()
            .map(|s| process::Value::Public((*s).into()))
            .collect(),
            cwd: self.root.path().canonicalize().unwrap(),
            environment: Default::default(),
        };
        let raw = process::supervise_raw(
            &spec,
            &process::Limits {
                execution: Duration::from_secs(35),
                graceful_shutdown: Duration::from_secs(8),
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
            assert!(!s.truncated && s.line_capture_complete);
            assert_eq!(s.bytes_seen, r.as_bytes().len() as u64);
        }
        raw
    }
    fn report(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.output.join("report.json")).unwrap()).unwrap()
    }
}
#[test]
fn raw_native_cli_orders_conversion_verification_compose_attach_and_retains_middle() {
    let f = Fixture::new("success");
    let before = f
        .parts
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect::<Vec<_>>();
    assert!(f.run(&process::Cancellation::default()).process.success());
    let report = f.report();
    assert_eq!(report["status"], "PASS");
    assert_eq!(report["source_unchanged"], true);
    assert_eq!(report["native_conversion_verification"]["complete"], true);
    let calls = std::fs::read_to_string(
        f.output
            .join("native-conversion/conversion-invocations.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str::<Vec<String>>(line).unwrap())
    .collect::<Vec<_>>();
    assert_eq!(
        calls.iter().map(|a| a[0].as_str()).collect::<Vec<_>>(),
        ["convert", "verify-job"]
    );
    let composed = std::fs::read_to_string(f.output.join("invocations.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Vec<String>>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        composed.iter().map(|a| a[0].as_str()).collect::<Vec<_>>(),
        ["compose-mtp", "validate-mtp-attach"]
    );
    assert!(!composed[1].contains(&"--projector".into()));
    let plan: Value =
        serde_json::from_slice(&std::fs::read(f.output.join("publication-plan.json")).unwrap())
            .unwrap();
    assert_eq!(plan["status"], "PLAN_ONLY_NOT_PUBLISHED");
    assert!(plan["upload_execution"].is_null());
    assert_eq!(plan["entries"].as_array().unwrap().len(), 3);
    assert_eq!(
        plan["entries"][1]["local"],
        serde_json::to_value(&f.parts[1]).unwrap()
    );
    for (path, bytes) in f.parts.iter().zip(before) {
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
    f.root.close().unwrap();
}
#[test]
fn raw_native_cli_partial_conversion_bad_verification_and_source_drift_never_publish() {
    for mode in ["nonzero", "missing", "bad-verify", "drift"] {
        let f = Fixture::new(mode);
        let raw = f.run(&process::Cancellation::default());
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        assert_eq!(f.report()["status"], "FAILED");
        assert!(!f.output.join("publication-plan.json").exists());
        assert!(
            f.output
                .join("native-conversion/convert-process.json")
                .is_file()
        );
        assert!(!f.output.join("invocations.jsonl").exists());
        if mode == "nonzero" {
            assert!(f.output.join("native-conversion/partial-spool").is_file());
        }
        f.root.close().unwrap();
    }
}
#[test]
fn raw_native_cli_held_conversion_cancellation_joins_before_fixture_cleanup() {
    let f = Fixture::new("held");
    let marker = f.output.join("native-conversion/conversion-held-marker");
    let cancel = process::Cancellation::default();
    let signal = cancel.clone();
    let waiter = std::thread::spawn(move || {
        let end = Instant::now() + Duration::from_secs(20);
        while !marker.exists() && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(5));
        }
        let seen = marker.exists();
        signal.cancel();
        seen
    });
    let raw = f.run(&cancel);
    let seen = waiter.join().unwrap();
    assert!(seen, "actual conversion child never entered held path");
    assert!(!raw.process.success());
    assert!(!f.output.join("publication-plan.json").exists());
    if f.output.join("report.json").exists() {
        assert_eq!(f.report()["status"], "FAILED");
    }
    f.root.close().unwrap();
}
