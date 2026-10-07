//! Injected command orchestration proof; fake processes do not qualify a runtime or GGUF.
use super::super::{command, stream_metrics, trial_receipt};
use super::*;
use crate::process::{
    self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as ProcessValue,
};
use std::cell::RefCell;

struct Fixture {
    directory: tempfile::TempDir,
    root: std::path::PathBuf,
    binary: std::path::PathBuf,
    model: std::path::PathBuf,
}
impl Fixture {
    #[cfg(unix)]
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let binary = root.join("fake-host");
        std::fs::write(&binary,"#!/bin/sh\nprintf '%s\\n' \"$@\" > observed-argv\nif [ \"$1\" = event-disabled ]; then dropped=1; else dropped=0; fi\nprintf '{\"context\":\"event_system_health\",\"message\":\"version=1 dropped_progress=%s dropped_diagnostic=0 ingress_p99_us=4\"}\\n' \"$dropped\"\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let model = root.join("fixture-model-bytes");
        std::fs::write(
            &model,
            b"not a GGUF; admission boundary is explicitly injected",
        )
        .unwrap();
        Self {
            directory,
            root,
            binary,
            model,
        }
    }
    fn flags(&self, output: &str, attempt: Option<u64>) -> Vec<String> {
        let mut flags = vec![
            "--binary".into(),
            self.binary.to_string_lossy().into_owned(),
            "--model".into(),
            self.model.to_string_lossy().into_owned(),
            "--output-dir".into(),
            self.root.join(output).to_string_lossy().into_owned(),
            "--pairs-primary".into(),
            "1".into(),
            "--pairs-scenario".into(),
            "1".into(),
            "--seed".into(),
            "42".into(),
            "--mode".into(),
            "production".into(),
            "--mode".into(),
            "event-disabled".into(),
            "--scenario".into(),
            "fixture".into(),
        ];
        if let Some(attempt) = attempt {
            flags.extend(["--attempt".into(), attempt.to_string()]);
        }
        flags
    }
}
fn digest(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}
fn admitted(command: &options::Command) -> preflight::Prepared {
    let identities = std::array::from_fn(|index| manifest_output::Binary {
        path: command.sides[index].binary.clone(),
        sha256: digest(&command.sides[index].binary),
        version: Some("fake fixture".into()),
    });
    preflight::Prepared {
        metadata: admission::Metadata {
            sides: command.sides.clone(),
            binaries: identities,
            host: manifest_output::Host::classify("fixture".into(), "fixture".into()),
            model: command.model.clone(),
            source_model_sha256: digest(&command.model),
            model_metadata: json!({"architecture":"fixture","native_context_tokens":4096}),
            runtime_roots: [
                command.model.parent().unwrap().to_path_buf(),
                command.model.parent().unwrap().to_path_buf(),
            ],
            runtime_packages_verified: true,
        },
        thermal_state: json!({"available":false,"source":"fixture"}),
    }
}
fn fake_trial(
    input: &trial_cell::Input<'_>,
    cancellation: &Cancellation,
    fail: bool,
) -> DynResult<trial_cell::Trial> {
    assert_eq!(input.max_tokens, 64);
    assert_eq!(
        (input.readiness, input.request, input.shutdown),
        (
            Duration::from_secs(120),
            Duration::from_secs(120),
            Duration::from_secs(15)
        )
    );
    std::fs::create_dir(input.directory)?;
    let spec = ProcessSpec {
        executable: input.side.binary.clone(),
        cwd: input.directory.into(),
        arguments: [
            input.side.mode.label().to_string(),
            input.entry.prompt_sha256(),
            input.entry.pair_index.to_string(),
            input.entry.side_order_first.clone(),
        ]
        .into_iter()
        .map(|s| ProcessValue::Public(s.into()))
        .collect(),
        environment: BTreeMap::new(),
    };
    let report = process::supervise(
        &spec,
        &Limits {
            // This fixture proves forwarding/order, not scheduler timing. Keep its
            // finite printf child bounded without a two-second full-suite race.
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_millis(100),
            forced_shutdown: Duration::from_millis(100),
            retained_bytes_per_stream: 4096,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles {
            stdout: Some(input.directory.join("server.stdout.log")),
            stderr: Some(input.directory.join("server.stderr.log")),
        },
    )?;
    assert_eq!(
        report.outcome,
        process::Outcome::Exited,
        "finite printf-only fixture completion failed: {report:?}; observed argv: {:?}",
        std::fs::read_to_string(input.directory.join("observed-argv"))
    );
    assert_eq!(
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(0)
    );
    assert!(
        report.failure.is_none()
            && report.cleanup.complete
            && !report.cleanup.forced
            && !report.cleanup.graceful_signal_failed
            && report.cleanup.failure.is_none()
    );
    assert!(
        [&report.stdout, &report.stderr]
            .iter()
            .all(|stream| !stream.truncated && stream.suppressed_lines == 0)
    );
    let argv = std::fs::read_to_string(input.directory.join("observed-argv"))?;
    assert_eq!(
        argv.lines().collect::<Vec<_>>(),
        vec![
            input.side.mode.label(),
            &input.entry.prompt_sha256(),
            &input.entry.pair_index.to_string(),
            &input.entry.side_order_first
        ]
    );
    let (profile, inheritance) =
        trial_profile::apply(BTreeMap::new(), input.inherited_profile, input.side.mode);
    let effective = profile
        .into_iter()
        .map(|(key, value)| match value {
            ProcessValue::Public(value) | ProcessValue::Secret(value) => (key, value),
        })
        .collect();
    Ok(trial_cell::Trial {
        outcome: paired_execution::Outcome {
            launched: true,
            measurement: Some(stream_metrics::Measurement {
                completion_tokens: Some(2),
                elapsed_ms: 10.0,
                ttft_ms: Some(1.0),
                decode_tok_s: Some(200.0),
                decode_only_tok_s: Some(2.0 / 0.009),
                malformed: false,
            }),
            health: trial_receipt::final_health(input.directory)?,
            error: fail.then(|| "injected trial failure after owned child completion".into()),
            ..Default::default()
        },
        environment: trial_environment::snapshot(&effective),
        inheritance,
        worker_status: None,
        cleanup_complete: report.cleanup.complete,
        cleanup_forced: report.cleanup.forced,
        capture_complete: !report.stdout.truncated && !report.stderr.truncated,
        health_capture_complete: false,
        health_observation_error: None,
    })
}
#[cfg(unix)]
fn run(
    fixture: &Fixture,
    output: &str,
    attempt: Option<u64>,
    fail: bool,
) -> (command::ResultReceipt, Vec<(String, String, u64, String)>) {
    let observations = RefCell::new(Vec::new());
    let validations = RefCell::new(0usize);
    let result = command::drive(
        &fixture.flags(output, attempt),
        &Cancellation::default(),
        |command, directory, remaining, cancel| {
            assert!(
                directory.is_dir() && !cancel.is_cancelled() && remaining > Duration::from_secs(1)
            );
            assert_eq!(command.attempt, attempt.unwrap_or(1));
            assert_eq!((command.max_tokens, command.execution_secs), (64, 86400));
            Ok(admitted(command))
        },
        |command, metadata, start, cancellation| {
            execute_with(
                command,
                metadata,
                start,
                cancellation,
                |metadata, directory, remaining, cancel| {
                    assert!(
                        metadata.runtime_packages_verified
                            && directory.is_dir()
                            && !cancel.is_cancelled()
                            && !remaining.is_zero()
                    );
                    *validations.borrow_mut() += 1;
                    Ok(())
                },
                |input, cancel| {
                    observations.borrow_mut().push((
                        input.side.side_id.clone(),
                        input.entry.scenario.clone(),
                        input.entry.pair_index,
                        input.entry.side_order_first.clone(),
                    ));
                    fake_trial(
                        input,
                        cancel,
                        fail && input.side.mode == plan::Mode::EventDisabled,
                    )
                },
            )
        },
    )
    .unwrap();
    assert_eq!(*validations.borrow(), 4);
    (result, observations.into_inner())
}
#[cfg(unix)]
#[test]
fn paired_command_forwards_default_attempt_and_preserves_real_cell_order_and_manifests() {
    let fixture = Fixture::new();
    let (result, order) = run(&fixture, "default", None, false);
    assert!(!result.failed);
    assert_eq!(result.manifest_paths.len(), 2);
    assert_eq!(order.len(), 4);
    for pair in order.as_chunks::<2>().0 {
        assert_ne!(pair[0].0, pair[1].0);
        assert_eq!(pair[0].1, pair[1].1);
        assert_eq!(pair[0].2, pair[1].2);
        assert_eq!(pair[0].0, pair[0].3);
    }
    for (side, path) in result.manifest_paths {
        assert!(path.ends_with(&format!("manifest-{side}.json")));
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(manifest["attempt"], 1);
        assert_eq!(manifest["trials"].as_array().unwrap().len(), 2);
        assert_eq!(manifest["executed_order"].as_array().unwrap().len(), 2);
        assert_eq!(
            manifest["health"]["dropped_progress"],
            u64::from(side == "event-disabled")
        );
        assert!(manifest["environment"].is_object() && manifest["inherited_profile"].is_object());
    }
    assert!(fixture.directory.path().exists());
    fixture.directory.close().unwrap();
}
#[cfg(unix)]
#[test]
fn paired_command_forwards_retry_attempt_to_both_sides_and_keeps_partial_failure_evidence() {
    let fixture = Fixture::new();
    let (result, _) = run(&fixture, "retry", Some(2), true);
    assert!(result.failed);
    for (side, path) in result.manifest_paths {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(manifest["attempt"], 2);
        for trial in manifest["trials"].as_array().unwrap() {
            assert_eq!(
                trial["status"],
                if side == "event-disabled" {
                    "failed"
                } else {
                    "succeeded"
                }
            );
        }
    }
    fixture.directory.close().unwrap();
}
#[cfg(unix)]
#[test]
fn failed_command_admission_publishes_error_without_any_matrix_launch() {
    let fixture = Fixture::new();
    let result = command::drive(
        &fixture.flags("rejected", None),
        &Cancellation::default(),
        |_, _, _, _| Err("injected preflight rejection".into()),
        |_, _, _, _| panic!("matrix must not launch after rejected admission"),
    );
    assert!(result.is_err());
    assert!(fixture.root.join("rejected/preflight-error.json").is_file());
    assert!(
        !fixture
            .root
            .join("rejected/manifest-production.json")
            .exists()
    );
    fixture.directory.close().unwrap();
}
