//! Real native caller/process/ready-magic/report fixtures; no native model proof.
use crate::process;
use serde_json::Value;
use std::{path::Path, time::Duration};
struct Fixture {
    root: tempfile::TempDir,
    server: std::path::PathBuf,
    client: std::path::PathBuf,
    build: std::path::PathBuf,
    package: std::path::PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().canonicalize().unwrap();
        let fixture = Path::new(env!("CARGO_BIN_EXE_xtask"))
            .parent()
            .unwrap()
            .join("examples/l7_daemon_fixture");
        let server = path.join("server");
        let client = path.join("client");
        std::fs::copy(&fixture, &server).unwrap();
        std::fs::copy(&fixture, &client).unwrap();
        let build = path.join("native");
        let package = path.join("package");
        std::fs::create_dir(&build).unwrap();
        std::fs::create_dir(&package).unwrap();
        Self {
            root,
            server,
            client,
            build,
            package,
        }
    }
    fn arguments(&self, name: &str, extra: &[&str]) -> Vec<String> {
        let mut args = vec!["automation".to_owned(), "mtp-scheduler".into()];
        for (flag, path) in [
            ("--old-bin", &self.server),
            ("--new-bin", &self.server),
            ("--client-bin", &self.client),
            ("--native-build", &self.build),
            ("--package", &self.package),
            ("--output-dir", &self.root.path().join(name)),
        ] {
            args.extend([flag.into(), path.to_str().unwrap().into()]);
        }
        args.extend(
            [
                "--concurrency",
                "1,2",
                "--requests",
                "2",
                "--startup-seconds",
                "2",
                "--client-seconds",
                "2",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        args.extend(extra.iter().map(|s| (*s).into()));
        args
    }
    fn invoke(&self, name: &str, extra: &[&str]) -> process::RawProcessReport {
        self.raw(self.arguments(name, extra))
    }
    fn raw(&self, args: Vec<String>) -> process::RawProcessReport {
        self.raw_cancel(args, &process::Cancellation::default())
    }
    fn raw_cancel(
        &self,
        args: Vec<String>,
        cancellation: &process::Cancellation,
    ) -> process::RawProcessReport {
        let raw = process::supervise_raw(
            &process::ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                arguments: args
                    .into_iter()
                    .map(|s| process::Value::Public(s.into()))
                    .collect(),
                cwd: self.root.path().canonicalize().unwrap(),
                environment: std::collections::BTreeMap::new(),
            },
            &process::Limits {
                execution: Duration::from_secs(12),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            cancellation,
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
        assert!(p.stdout.line_capture_complete && p.stderr.line_capture_complete);
        assert_eq!(
            raw.stdout.as_ref().unwrap().as_bytes().len() as u64,
            p.stdout.bytes_seen
        );
        assert_eq!(
            raw.stderr.as_ref().unwrap().as_bytes().len() as u64,
            p.stderr.bytes_seen
        );
        raw
    }
    fn json(&self, path: &str) -> Value {
        serde_json::from_slice(&std::fs::read(self.root.path().join(path)).unwrap()).unwrap()
    }
}
#[test]
fn mtp_scheduler_full_old_new_sweeps_preserve_cpu_profile_metrics_and_exact_roster() {
    let f = Fixture::new();
    let raw = f.invoke("success", &[]);
    assert!(raw.process.success());
    let v = f.json("success/comparison.json");
    assert_eq!(v["status"], "completed");
    for label in ["old", "new"] {
        assert_eq!(
            f.json(&format!("success/{label}-cli-admission.json"))["dialect"],
            "Current"
        );
        let arm = &v[label];
        assert_eq!(arm["concurrency_sweep"].as_array().unwrap().len(), 2);
        for row in arm["concurrency_sweep"].as_array().unwrap() {
            assert_eq!(row["metrics"]["latency_p50_ms"], 3.0);
            assert_eq!(row["metrics"]["latency_p99_ms"], 4.0);
            assert_eq!(row["metrics"]["accepted"], 2);
        }
        let config = f.json(&format!("success/{label}-stage.json"));
        assert_eq!(config["ctx_size"], 512);
        assert_eq!(config["n_batch"], 8);
        assert_eq!(config["layer_start"], 74);
        assert_eq!(config["layer_end"], 78);
    }
    for row in v["parity"].as_array().unwrap() {
        assert_eq!(row["exact_field_matches"], 8);
        assert_eq!(row["complete_request_coverage"], true);
    }
    assert!(
        String::from_utf8_lossy(raw.stdout.as_ref().unwrap().as_bytes())
            .contains("comparison.json")
    );
    std::fs::write(f.build.join("default-signal"), b"").unwrap();
    assert!(f.invoke("signal-stop", &[]).process.success());
    for label in ["old", "new"] {
        let receipt = f.json(&format!("signal-stop/{label}-lifecycle.json"));
        assert_eq!(receipt["outcome"], "Ready");
        let members = receipt["members"].as_array().unwrap();
        let server = members
            .iter()
            .find(|member| member["disposition"] == "intentional-stop")
            .unwrap();
        assert_eq!(server["exit_code"], Value::Null);
        assert_eq!(server["unix_signal"], libc::SIGTERM);
        assert_eq!(server["cleanup_complete"], true);
        assert_eq!(server["cleanup_forced"], false);
        assert_eq!(server["stdout_complete"], true);
        assert_eq!(server["stderr_complete"], true);
        let worker = members
            .iter()
            .find(|member| member["disposition"] == "expected-exit")
            .unwrap();
        assert_eq!(worker["exit_code"], 0);
    }
    f.root.close().unwrap();
}
#[test]
fn mtp_scheduler_partial_client_failure_keeps_completed_cell_and_stops_server() {
    let f = Fixture::new();
    std::fs::write(f.build.join("client-fail"), b"").unwrap();
    let raw = f.invoke("partial", &[]);
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    let v = f.json("partial/old-result.json");
    assert_eq!(v["status"], "failed");
    assert_eq!(v["concurrency_sweep"].as_array().unwrap().len(), 1);
    assert_eq!(f.json("partial/comparison.json")["status"], "failed");
    assert!(!f.root.path().join("partial/new-stage.json").exists());
    f.root.close().unwrap();
}
#[test]
fn mtp_scheduler_duplicate_roster_and_invalid_profile_cannot_publish_success() {
    let f = Fixture::new();
    let raw = f.invoke("invalid", &["--concurrency", "0"]);
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert!(!f.root.path().join("invalid").exists());
    std::fs::write(f.build.join("duplicate"), b"").unwrap();
    let raw = f.invoke("duplicate", &[]);
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert_eq!(f.json("duplicate/comparison.json")["status"], "failed");
    f.root.close().unwrap();
}
#[test]
fn mtp_scheduler_wrong_or_held_ready_magic_and_early_exit_leave_owned_failure_evidence() {
    for mode in ["bad-magic", "held", "early"] {
        let f = Fixture::new();
        std::fs::write(f.build.join(mode), b"").unwrap();
        let raw = f.invoke("failed", &[]);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        assert_eq!(f.json("failed/comparison.json")["status"], "failed");
        assert!(!f.root.path().join("failed/new-stage.json").exists());
        f.root.close().unwrap();
    }
}
#[test]
fn mtp_scheduler_reports_request_failures_and_token_mismatch_without_claiming_parity() {
    let f = Fixture::new();
    std::fs::write(f.build.join("mismatch"), b"").unwrap();
    assert!(f.invoke("mismatch", &[]).process.success());
    let v = f.json("mismatch/comparison.json");
    for row in v["parity"].as_array().unwrap() {
        assert_eq!(row["exact_requests"], 0);
        assert_eq!(row["exact_field_matches"], 6);
        assert_eq!(row["exact_field_total"], 8);
    }
    std::fs::remove_file(f.build.join("mismatch")).unwrap();
    std::fs::write(f.build.join("error-row"), b"").unwrap();
    assert!(f.invoke("errors", &[]).process.success());
    let v = f.json("errors/comparison.json");
    assert_eq!(v["old"]["concurrency_sweep"][0]["metrics"]["failed"], 1);
    assert_eq!(v["parity"][0]["comparable_requests"], 1);
    assert_eq!(v["parity"][0]["complete_request_coverage"], false);
    std::fs::remove_file(f.build.join("error-row")).unwrap();
    std::fs::write(f.build.join("huge-latency"), b"").unwrap();
    assert!(f.invoke("finite-latency", &[]).process.success());
    let v = f.json("finite-latency/comparison.json");
    assert_eq!(
        v["old"]["concurrency_sweep"][0]["metrics"]["latency_p50_ms"].as_f64(),
        Some(f64::MAX)
    );

    f.root.close().unwrap();
}
#[test]
fn mtp_scheduler_held_ready_interruption_is_causal_and_cleans_owned_server() {
    let f = Fixture::new();
    std::fs::write(f.build.join("held"), b"").unwrap();
    let cancellation = process::Cancellation::default();
    let observer_cancel = cancellation.clone();
    let marker = f.build.join("ready-probed");
    let observer = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !marker.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let server = std::fs::read_to_string(marker)
            .ok()
            .and_then(|s| s.parse::<i32>().ok());
        observer_cancel.cancel();
        server
    });
    let raw = f.raw_cancel(
        f.arguments("cancelled", &["--startup-seconds", "20"]),
        &cancellation,
    );
    let observed = observer.join().unwrap();
    let server = observed.expect("actual readiness probe marker before interruption");
    let status = raw.process.status.unwrap();
    assert!(!status.success());
    assert_eq!(f.json("cancelled/comparison.json")["status"], "failed");
    assert!(!f.root.path().join("cancelled/new-stage.json").exists());
    assert_eq!(unsafe { libc::kill(server, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );

    f.root.close().unwrap();
}
#[test]
fn mtp_scheduler_direct_worker_refuses_unbounded_profiles_before_socket_or_client() {
    let f = Fixture::new();
    assert!(f.invoke("base", &[]).process.success());
    let input = f.root.path().join("base/input.json");
    let mut v = f.json("base/input.json");
    for (field, value) in [
        ("startup_seconds", serde_json::json!(u64::MAX)),
        ("client_seconds", serde_json::json!(0)),
        ("requests", serde_json::json!(10001)),
        ("concurrency", serde_json::json!([0])),
        ("activation_width", serde_json::json!(0)),
    ] {
        let old = v[field].clone();
        v[field] = value;
        std::fs::write(&input, serde_json::to_vec(&v).unwrap()).unwrap();
        let raw = f.raw(vec![
            "automation".into(),
            "mtp-scheduler-worker".into(),
            input.to_str().unwrap().into(),
            "old".into(),
            "127.0.0.1:1".into(),
        ]);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        let error = String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes());
        assert!(
            error.contains("invalid bounded MTP scheduler profile"),
            "{error}"
        );
        assert_eq!(f.json("base/old-result.json")["status"], "completed");
        v[field] = old;
    }
    use std::os::unix::ffi::OsStrExt;
    let fifo = f.root.path().join("worker-input-fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let raw = f.raw(vec![
        "automation".into(),
        "mtp-scheduler-worker".into(),
        fifo.to_str().unwrap().into(),
        "old".into(),
        "127.0.0.1:1".into(),
    ]);
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert!(
        String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes())
            .contains("bounded regular file")
    );
    f.root.close().unwrap();
}

#[test]
fn mtp_scheduler_admits_actual_legacy_help_without_changing_historical_arm_identity() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.build.join("cli-legacy"),
        b"finite legacy help fixture",
    )
    .unwrap();
    let raw = fixture.invoke("legacy", &[]);
    assert!(raw.process.success());
    assert_eq!(
        fixture.json("legacy/comparison.json")["status"],
        "completed"
    );
    for label in ["old", "new"] {
        let receipt = fixture.json(&format!("legacy/{label}-cli-admission.json"));
        assert_eq!(receipt["dialect"], "Legacy");
        assert_eq!(receipt["role"], "binary-worker");
        assert_eq!(
            fixture.json("legacy/comparison.json")[label]["label"],
            label
        );
    }
}
