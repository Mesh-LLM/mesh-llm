//! Actual xtask worker dispatch against a bounded local HTTP peer. Unix qualification.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde::Serialize;
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::Path, time::Duration};
#[path = "worker_http_fixture.rs"]
mod http_fixture;
use http_fixture::{Fixture, Mode, Request};

// Field order deliberately matches measurement_worker::Input, whose typed compact
// serialization (rather than raw input-file bytes) owns request correlation.
#[derive(Serialize)]
struct Input {
    schema_version: u64,
    port: u16,
    prompt: String,
    prompt_sha256: String,
    max_tokens: u64,
    readiness_timeout_ms: u64,
    request_timeout_ms: u64,
    readiness_poll_ms: u64,
}

impl Input {
    fn new(port: u16) -> Self {
        let prompt = "fixture prompt \"quoted\"\nλ".to_string();
        Self {
            schema_version: 1,
            port,
            prompt_sha256: hash(prompt.as_bytes()),
            prompt,
            max_tokens: 7,
            readiness_timeout_ms: 1000,
            request_timeout_ms: 1500,
            readiness_poll_ms: 5,
        }
    }
    fn publish(&self, path: &Path) {
        fs::write(path, serde_json::to_vec_pretty(self).unwrap()).unwrap();
    }
}

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn run(cwd: &Path, args: Vec<String>) -> process::ProcessReport {
    let result = process::supervise(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_xtask")).to_path_buf(),
            arguments: args
                .into_iter()
                .map(|arg| Value::Public(arg.into()))
                .collect(),
            cwd: cwd.into(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(12),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(result.outcome, process::Outcome::Exited, "{result:?}");
    assert!(
        result.cleanup.complete && !result.cleanup.forced,
        "{result:?}"
    );
    assert!(
        result.failure.is_none() && result.cleanup.failure.is_none(),
        "{result:?}"
    );
    result
}

fn worker(cwd: &Path, input: &Path, output: &Path) -> process::ProcessReport {
    run(
        cwd,
        [
            "automation",
            "event-benchmark-run",
            "measurement-worker",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

fn assert_public_help(cwd: &Path) {
    // Raw capture is restricted to this fixed public help argv and an empty
    // environment; user input, prompts, credentials and worker output stay on
    // the normal sanitized supervision path.
    let raw = process::supervise_raw(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_xtask")).to_path_buf(),
            arguments: ["automation", "event-benchmark-run", "--help"]
                .into_iter()
                .map(|argument| Value::Public(argument.into()))
                .collect(),
            cwd: cwd.into(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(12),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(4096),
            stderr: std::num::NonZeroUsize::new(4096),
        },
    )
    .unwrap();
    let report = &raw.process;
    assert_eq!(report.outcome, process::Outcome::Exited, "{report:?}");
    assert_eq!(report.status.unwrap().code(), Some(0));
    assert!(report.success(), "{report:?}");
    assert!(report.cleanup.complete && !report.cleanup.forced);
    assert!(report.failure.is_none() && report.cleanup.failure.is_none());
    let stdout = raw.stdout.as_ref().unwrap().as_bytes();
    let stderr = raw.stderr.as_ref().unwrap().as_bytes();
    assert!(!stdout.is_empty() && stdout.len() <= 4096);
    assert!(stderr.is_empty());
    assert_eq!(report.stdout.bytes_seen, stdout.len() as u64);
    assert_eq!(report.stderr.bytes_seen, 0);
    assert!(!report.stdout.truncated && !report.stderr.truncated);
    let help = std::str::from_utf8(stdout).unwrap();
    assert!(help.starts_with("cargo xtool automation event-benchmark-run --binary PATH "));
    assert!(help.ends_with('\n'));
    for expected in [
        "[--baseline-binary PATH]",
        "--model LOCAL_GGUF",
        "--output-dir PATH",
        "--pairs-primary N",
        "--pairs-scenario N",
        "--seed U64",
        "--mode production|event-disabled|off",
        "[--mode MODE]",
        "--scenario LABEL",
        "[--scenario LABEL]",
        "[--attempt 1|2]",
        "[--max-tokens N]",
        "[--readiness-timeout-secs N]",
        "[--request-timeout-secs N]",
        "[--shutdown-timeout-secs N]",
        "[--execution-timeout-secs N]",
    ] {
        assert!(
            help.contains(expected),
            "public help omits {expected}: {help}"
        );
    }
    // The single public USAGE line contains --max-tokens, so the normal
    // diagnostic channel must still apply its sensitive-line policy.
    assert_eq!(report.stdout.suppressed_lines, 1);
    assert_eq!(report.stdout.bytes_retained, b"[output line suppressed]\n");
    assert_eq!(report.stderr.suppressed_lines, 0);
}

fn receipt(directory: &Path, input: &Input) -> Json {
    let path = directory.join("receipt.json");
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_file());
    let mut bytes = Vec::new();
    fs::File::open(path)
        .unwrap()
        .take(65537)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= 65536);
    let evidence: Json = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(evidence["schema_version"], 1);
    assert_eq!(
        evidence["request_sha256"],
        hash(&serde_json::to_vec(input).unwrap())
    );
    assert_ne!(
        evidence["request_sha256"],
        hash(&fs::read(directory.join("input.json")).unwrap())
    );
    assert_eq!(evidence["prompt_sha256"], input.prompt_sha256);
    assert_eq!(evidence["model"], "served-fixture");
    assert!(evidence["readiness_ms"].as_f64().unwrap() >= 0.0);
    assert!(evidence["warmup_ms"].as_f64().unwrap() >= 0.0);
    let mut files = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(
        files,
        [
            std::ffi::OsString::from("input.json"),
            std::ffi::OsString::from("receipt.json")
        ]
    );
    evidence
}

fn requests(requests: &[Request], input: &Input) {
    assert_eq!(requests.len(), 3, "{requests:?}");
    assert_eq!(
        (&*requests[0].method, &*requests[0].path),
        ("GET", "/v1/models")
    );
    assert!(requests[0].body.is_null());
    for request in &requests[1..] {
        assert_eq!(
            (&*request.method, &*request.path),
            ("POST", "/v1/chat/completions")
        );
        assert_eq!(request.body["model"], "served-fixture");
        assert_eq!(request.body["messages"][0]["content"], input.prompt);
        assert_eq!(request.body["max_tokens"], 7);
        assert_eq!(request.body["temperature"], 0.0);
        assert_eq!(request.body["stream"], true);
        assert_eq!(request.body["stream_options"]["include_usage"], true);
    }
    assert_eq!(requests[1].body, requests[2].body);
}

fn measured(evidence: &Json) {
    let measurement = &evidence["measurement"];
    assert_eq!(measurement["completion_tokens"], 3); // Warmup returned 5.
    assert_eq!(measurement["malformed"], false);
    let elapsed = measurement["elapsed_ms"].as_f64().unwrap();
    let ttft = measurement["ttft_ms"].as_f64().unwrap();
    assert!(ttft >= 0.0 && elapsed >= ttft && elapsed > 0.0);
    let rate = measurement["decode_tok_s"].as_f64().unwrap();
    assert!((rate - 3.0 / (elapsed / 1000.0)).abs() < 1e-8);
    if elapsed > ttft {
        let decode_only = measurement["decode_only_tok_s"].as_f64().unwrap();
        assert!((decode_only - 3.0 / ((elapsed - ttft) / 1000.0).max(1e-6)).abs() < 1e-8);
    } else {
        // Scheduler delay can coalesce both finite frames into one observation.
        assert!(measurement["decode_only_tok_s"].is_null());
    }
}

#[test]
fn worker_cli_resolves_model_excludes_warmup_and_publishes_correlated_usage() {
    let fixture = Fixture::start(Mode::Healthy);
    let directory = tempfile::tempdir().unwrap();
    let input = Input::new(fixture.port);
    input.publish(&directory.path().join("input.json"));
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &directory.path().join("receipt.json"),
    );
    assert!(output.success(), "{output:?}");
    let evidence = receipt(directory.path(), &input);
    assert!(evidence["error"].is_null() && evidence["warmup_error"].is_null());
    measured(&evidence);
    requests(&fixture.finish(), &input);
}

#[test]
fn worker_cli_measurement_timeout_retains_atomic_partial_receipt_and_status_one() {
    let fixture = Fixture::start(Mode::MeasurementTimeout);
    let directory = tempfile::tempdir().unwrap();
    let mut input = Input::new(fixture.port);
    input.request_timeout_ms = 250;
    input.publish(&directory.path().join("input.json"));
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &directory.path().join("receipt.json"),
    );
    assert_eq!(output.status.unwrap().code(), Some(1));
    let evidence = receipt(directory.path(), &input);
    assert!(
        evidence["error"].as_str().unwrap().contains("deadline"),
        "{evidence}"
    );
    assert!(evidence["measurement"].is_null() && evidence["warmup_error"].is_null());
    requests(&fixture.finish(), &input);
}

#[test]
fn worker_cli_http_failure_retains_observed_readiness_and_warmup() {
    let fixture = Fixture::start(Mode::MeasurementError);
    let directory = tempfile::tempdir().unwrap();
    let input = Input::new(fixture.port);
    input.publish(&directory.path().join("input.json"));
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &directory.path().join("receipt.json"),
    );
    assert_eq!(output.status.unwrap().code(), Some(1));
    let evidence = receipt(directory.path(), &input);
    assert!(evidence["error"].as_str().unwrap().contains("500"));
    assert!(evidence["measurement"].is_null() && evidence["warmup_error"].is_null());
    requests(&fixture.finish(), &input);
}

#[test]
fn worker_cli_failed_warmup_still_attempts_measurement() {
    let fixture = Fixture::start(Mode::WarmupError);
    let directory = tempfile::tempdir().unwrap();
    let input = Input::new(fixture.port);
    input.publish(&directory.path().join("input.json"));
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &directory.path().join("receipt.json"),
    );
    assert!(output.success(), "{output:?}");
    let evidence = receipt(directory.path(), &input);
    assert!(evidence["error"].is_null());
    assert!(evidence["warmup_error"].as_str().unwrap().contains("500"));
    measured(&evidence);
    requests(&fixture.finish(), &input);
}

#[test]
fn worker_cli_existing_receipt_refuses_before_http_without_overwrite() {
    let fixture = Fixture::start(Mode::Healthy);
    let directory = tempfile::tempdir().unwrap();
    Input::new(fixture.port).publish(&directory.path().join("input.json"));
    let receipt = directory.path().join("receipt.json");
    fs::write(&receipt, b"existing sentinel").unwrap();
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &receipt,
    );
    assert_eq!(output.status.unwrap().code(), Some(1));
    assert_eq!(fs::read(receipt).unwrap(), b"existing sentinel");
    assert!(fixture.finish().is_empty());
}

#[test]
fn worker_cli_existing_symlink_refuses_without_touching_target_or_http() {
    let fixture = Fixture::start(Mode::Healthy);
    let directory = tempfile::tempdir().unwrap();
    Input::new(fixture.port).publish(&directory.path().join("input.json"));
    let target = directory.path().join("target");
    fs::write(&target, b"symlink sentinel").unwrap();
    let receipt = directory.path().join("receipt.json");
    std::os::unix::fs::symlink(&target, &receipt).unwrap();
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &receipt,
    );
    assert_eq!(output.status.unwrap().code(), Some(1));
    assert_eq!(fs::read(&target).unwrap(), b"symlink sentinel");
    assert_eq!(fs::read_link(receipt).unwrap(), target);
    assert!(fixture.finish().is_empty());
}

#[test]
fn worker_cli_dangling_symlink_refuses_before_http_without_materializing_target() {
    let fixture = Fixture::start(Mode::Healthy);
    let directory = tempfile::tempdir().unwrap();
    Input::new(fixture.port).publish(&directory.path().join("input.json"));
    let target = directory.path().join("absent-target");
    let receipt = directory.path().join("receipt.json");
    std::os::unix::fs::symlink(&target, &receipt).unwrap();
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &receipt,
    );
    assert_eq!(output.status.unwrap().code(), Some(1));
    assert!(!target.exists());
    assert_eq!(fs::read_link(receipt).unwrap(), target);
    assert!(fixture.finish().is_empty());
}

#[test]
fn worker_cli_help_and_argument_refusals_never_contact_http() {
    let fixture = Fixture::start(Mode::Healthy);
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.json");
    Input::new(fixture.port).publish(&input);
    let output = directory.path().join("receipt.json");
    assert_public_help(directory.path());
    for flags in [
        vec!["--input", input.to_str().unwrap()],
        vec![
            "--input",
            input.to_str().unwrap(),
            "--unknown",
            output.to_str().unwrap(),
        ],
        vec![
            "--input",
            "relative.json",
            "--output",
            output.to_str().unwrap(),
        ],
        vec![
            "--input",
            input.to_str().unwrap(),
            "--input",
            input.to_str().unwrap(),
        ],
    ] {
        let args = ["automation", "event-benchmark-run", "measurement-worker"]
            .into_iter()
            .chain(flags)
            .map(str::to_string)
            .collect();
        let report = run(directory.path(), args);
        assert_eq!(report.status.unwrap().code(), Some(1));
        assert!(!output.exists());
    }
    assert!(fixture.finish().is_empty());
}

#[test]
fn worker_cli_invalid_prompt_digest_refuses_before_any_http_or_receipt() {
    let fixture = Fixture::start(Mode::Healthy);
    let directory = tempfile::tempdir().unwrap();
    let mut input = Input::new(fixture.port);
    input.prompt_sha256 = "0".repeat(64);
    input.publish(&directory.path().join("input.json"));
    let receipt = directory.path().join("receipt.json");
    let output = worker(
        directory.path(),
        &directory.path().join("input.json"),
        &receipt,
    );
    assert_eq!(output.status.unwrap().code(), Some(1));
    assert!(!receipt.exists());
    assert!(fixture.finish().is_empty());
}
