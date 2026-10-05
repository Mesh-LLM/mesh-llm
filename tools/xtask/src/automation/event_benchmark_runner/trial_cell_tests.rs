use super::*;

fn trial() -> Trial {
    Trial {
        outcome: Outcome {
            measurement: Some(super::super::stream_metrics::Measurement {
                completion_tokens: Some(2),
                ttft_ms: Some(1.0),
                elapsed_ms: 10.0,
                decode_tok_s: Some(200.0),
                decode_only_tok_s: Some(2.0 / 0.009),
                malformed: false,
            }),
            ..Outcome::default()
        },
        environment: Default::default(),
        worker_status: None,
        cleanup_complete: false,
        cleanup_forced: false,
        capture_complete: false,
        health_observation_error: None,
        inheritance: Default::default(),
    }
}

#[cfg(unix)]
fn launch(member: MemberId, script: &str, directory: &Path) -> Launch {
    let label = if member == MemberId::Seed {
        "server"
    } else {
        "worker"
    };
    Launch {
        member,
        spec: ProcessSpec {
            executable: "/bin/sh".into(),
            cwd: directory.into(),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
            environment: Default::default(),
        },
        files: OutputFiles {
            stdout: Some(directory.join(format!("{label}.stdout.log"))),
            stderr: Some(directory.join(format!("{label}.stderr.log"))),
        },
        readiness_deadline: Duration::from_secs(5),
    }
}

#[cfg(unix)]
fn fixture(script: &str, retention: usize) -> (tempfile::TempDir, Trial, Report<String>) {
    let directory = tempfile::tempdir().unwrap();
    let mut owner = Owner {
        server: Some(launch(MemberId::Seed, script, directory.path())),
        worker: Some(launch(
            MemberId::WorkerOne,
            "while [ ! -f ready ]; do sleep 0.02; done; exit 0",
            directory.path(),
        )),
        worker_policy: ExpectedExit::new(&[0, 1], Duration::from_secs(5)).unwrap(),
        stopping: false,
        setup_ms: None,
        stop_started: None,
        shutdown_ms: None,
        api_readiness: None,
        listener_ready: true,
        host_readiness_timeout: Duration::from_secs(5),
        host_started: None,
    };
    let mut limits = limits(Duration::from_secs(5), Duration::from_millis(500));
    limits.retained_bytes_per_stream = retention;
    let report =
        crate::process::retained::run(&mut owner, &limits, &Cancellation::default()).unwrap();
    assert!(report.recovery_success());
    let mut trial = trial();
    observe(&mut trial, &owner, &report);
    collect_health(&mut trial, directory.path());
    (directory, trial, report)
}

const HEALTH: &str = r#"{"context":"event_system_health","message":"version=1 dropped_progress=9 ingress_p99_us=4"}"#;

#[cfg(unix)]
fn shutdown_script(prefix: &str) -> String {
    [
        format!("health='{HEALTH}'; trap 'printf \"%s\\n\" \"$health\" >&2; exit 0' TERM"),
        prefix.into(),
        ": > ready".into(),
        "while :; do sleep 1; done".into(),
    ]
    .join("; ")
}

#[cfg(unix)]
#[test]
fn health_emitted_only_during_owned_shutdown_is_recovered_from_complete_capture() {
    let (_directory, trial, report) = fixture(&shutdown_script(":"), 4096);
    assert!(trial.capture_complete);
    assert!(trial.health_observation_error.is_none());
    assert_eq!(
        trial.outcome.health.health.unwrap()["dropped_progress"],
        serde_json::json!(9)
    );
    assert!(trial.outcome.measurement.is_some());
    assert!(report.members.iter().all(|m| m.process.cleanup.complete));
}

#[cfg(unix)]
#[test]
fn truncated_actual_capture_cannot_qualify_prefix_health_and_preserves_throughput() {
    let prefix = format!("printf '%s\\n' '{HEALTH}'; printf '%01024d\\n' 0");
    let (_directory, trial, report) = fixture(&shutdown_script(&prefix), 256);
    assert!(
        report
            .members
            .iter()
            .filter(|m| m.member == MemberId::Seed)
            .any(|m| m.process.stdout.truncated)
    );
    assert!(!trial.capture_complete);
    assert!(trial.outcome.health.health.is_none());
    assert!(trial.health_observation_error.is_some());
    assert!(trial.outcome.measurement.is_some());
    assert!(trial.outcome.error.is_none());
}

#[cfg(unix)]
#[test]
fn suppressed_actual_capture_cannot_qualify_final_health_and_preserves_throughput() {
    let (_directory, trial, report) = fixture(
        &shutdown_script("printf '%s\\n' 'Authorization: Bearer private-value'"),
        4096,
    );
    assert!(
        report
            .members
            .iter()
            .filter(|m| m.member == MemberId::Seed)
            .any(|m| m.process.stdout.suppressed_lines > 0)
    );
    assert!(!trial.capture_complete);
    assert!(trial.outcome.health.health.is_none());
    assert!(trial.health_observation_error.is_some());
    assert!(trial.outcome.measurement.is_some());
    assert!(trial.outcome.error.is_none());
}

#[cfg(unix)]
#[test]
fn ambiguous_actual_capture_retains_measurement_without_claiming_final_health() {
    let prefix = format!("printf '%s\\n' '{HEALTH}'");
    let (_directory, trial, _report) = fixture(&shutdown_script(&prefix), 4096);
    assert!(trial.capture_complete);
    assert!(trial.outcome.health.health.is_none());
    assert!(
        trial
            .health_observation_error
            .as_ref()
            .unwrap()
            .contains("ambiguous")
    );
    assert!(trial.outcome.measurement.is_some());
    assert!(trial.outcome.error.is_none());
}

#[test]
fn failed_private_preparation_explicitly_deletes_owned_state_and_retains_prior_error() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "event-benchmark-fixture").unwrap();
    let root = state.root().to_path_buf();
    std::fs::create_dir(root.join("home")).unwrap();
    let error = state.prepare().unwrap_err();
    let receipt = preparation_failure(state, Box::new(error)).to_string();
    assert!(!root.exists());
    assert!(receipt.contains("Prior"));
    assert!(receipt.contains("create private directory"));
}

#[test]
fn cleanup_failure_after_preparation_retains_deletion_receipt_and_prior_error() {
    let parent = tempfile::tempdir().unwrap();
    let state = PrivateState::create(parent.path(), "event-benchmark-fixture").unwrap();
    std::fs::remove_dir(state.root()).unwrap();
    let receipt = preparation_failure(state, "launch construction failed".into()).to_string();
    assert!(receipt.contains("Deletion"));
    assert!(receipt.contains("launch construction failed"));
}
