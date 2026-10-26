use crate::automation::client_readiness::{event, receipt};
use crate::process::*;
use crate::protocol::{Behavior, Destination};
use crate::support::{Case, ready, record, repository};
use std::collections::BTreeMap;
use std::net::{Ipv4Addr, TcpListener};
use std::time::Duration;

fn specification(case: &Case) -> ProcessSpec {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    ProcessSpec {
        executable: case.binary.clone(),
        cwd: repository(),
        environment: BTreeMap::from([
            (
                "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR".into(),
                Value::Public(case.native.clone().into()),
            ),
            ("HOME".into(), Value::Public(case.state.clone().into())),
        ]),
        arguments: [
            "--log-format".into(),
            "json".into(),
            "--port".into(),
            port.to_string(),
            "--no-console".into(),
            "client".into(),
            "--mesh-discovery-mode".into(),
            "mdns".into(),
        ]
        .into_iter()
        .map(|value: String| Value::Public(value.into()))
        .collect(),
    }
}

fn limits(cap: usize) -> Limits {
    Limits {
        execution: Duration::from_secs(3),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: cap,
        readiness: Readiness::ObservedLines {
            deadline: Duration::from_secs(1),
            matcher: event::matches,
        },
        completion: Completion::StopAfterReady,
    }
}

fn outputs(case: &Case) -> OutputFiles {
    OutputFiles {
        stdout: Some(case.root.path().join("out")),
        stderr: Some(case.root.path().join("err")),
    }
}

#[test]
fn migration_lifecycle_raw_matching_survives_each_retention_boundary() {
    for cap in [0, 4, 26, 128] {
        let mut records = vec![record(
            Destination::Stderr,
            b"harmless diagnostic\n".repeat(20).as_slice(),
        )];
        records.push(record(
            Destination::Stderr,
            b"{\"message\":\"Client ready\"}\n",
        ));
        let case = Case::new(Behavior::Clean, records);

        let report = supervise(
            &specification(&case),
            &limits(cap),
            &Cancellation::default(),
            outputs(&case),
        )
        .unwrap();

        assert!(
            matches!(report.readiness_stop, ReadinessStop::Admitted { observation, request: GracefulRequest::RequestedAfterLiveObservation } if observation.stream == Stream::Stderr)
        );
        assert_eq!(report.stderr.bytes_retained.len(), cap);
        assert!(report.stderr.truncated);
        assert_eq!(
            std::fs::read(case.root.path().join("err")).unwrap(),
            report.stderr.bytes_retained
        );
        assert!(receipt::accept(report).is_ok());
    }
}

#[test]
fn migration_lifecycle_ready_record_cut_in_evidence_still_matches() {
    let frame = b"{\"message\":\"Client ready\"}\n";
    for cap in [4, frame.len() - 1] {
        let case = Case::new(Behavior::Clean, vec![record(Destination::Stdout, frame)]);

        let report = supervise(
            &specification(&case),
            &limits(cap),
            &Cancellation::default(),
            outputs(&case),
        )
        .unwrap();

        assert_eq!(report.stdout.bytes_retained.len(), cap);
        assert!(report.stdout.truncated);
        assert!(receipt::accept(report).is_ok());
    }
}

#[test]
fn migration_lifecycle_declared_secret_is_classified_before_masking() {
    let case = Case::new(
        Behavior::Clean,
        vec![record(
            Destination::Stderr,
            b"{\"message\":\"Client ready\"}\n",
        )],
    );
    let mut spec = specification(&case);
    spec.environment.insert(
        "FIXTURE_CREDENTIAL".into(),
        Value::Secret("Client ready".into()),
    );

    let report = supervise(
        &spec,
        &limits(128),
        &Cancellation::default(),
        outputs(&case),
    )
    .unwrap();

    assert!(!String::from_utf8_lossy(&report.stderr.bytes_retained).contains("Client ready"));
    assert!(
        !String::from_utf8_lossy(&std::fs::read(case.root.path().join("err")).unwrap())
            .contains("Client ready")
    );
    assert!(receipt::accept(report).is_ok());
}

#[test]
fn migration_lifecycle_sensitive_key_record_is_eligible_but_not_retained_raw() {
    let case = Case::new(Behavior::Clean, vec![record(Destination::Stdout, b"{\"role\":\"client\",\"event\":\"passive_mode\",\"status\":\"ready\",\"token\":\"private-generated-value\"}\n")]);

    let report = supervise(
        &specification(&case),
        &limits(128),
        &Cancellation::default(),
        outputs(&case),
    )
    .unwrap();

    assert_eq!(report.stdout.suppressed_lines, 1);
    assert!(
        !String::from_utf8_lossy(&report.stdout.bytes_retained).contains("private-generated-value")
    );
    assert!(
        !String::from_utf8_lossy(&std::fs::read(case.root.path().join("out")).unwrap())
            .contains("private-generated-value")
    );
    assert!(receipt::accept(report).is_ok());
}

#[test]
fn migration_lifecycle_masking_must_not_repair_malformed_json() {
    let case = Case::new(
        Behavior::Clean,
        vec![record(
            Destination::Stdout,
            b"{\"message\":\"Client ready a\"b\"}\n",
        )],
    );
    let mut spec = specification(&case);
    spec.environment
        .insert("FIXTURE_CREDENTIAL".into(), Value::Secret("a\"b".into()));

    let report = supervise(
        &spec,
        &limits(128),
        &Cancellation::default(),
        outputs(&case),
    )
    .unwrap();

    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(serde_json::from_slice::<serde_json::Value>(&report.stdout.bytes_retained).is_ok());
    assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
    assert!(!String::from_utf8_lossy(&report.stdout.bytes_retained).contains("a\"b"));
    assert!(receipt::accept(report).is_err());
}

#[test]
fn migration_lifecycle_raw_record_ceiling_includes_cr() {
    for (length, expected) in [(8192_usize, true), (8193, false)] {
        let mut bytes = b"{\"message\":\"Client ready\",\"pad\":\"".to_vec();
        bytes.resize(length - 3, b'x');
        bytes.extend_from_slice(b"\"}\r\n");
        let case = Case::new(Behavior::Clean, vec![record(Destination::Stdout, &bytes)]);

        let report = supervise(
            &specification(&case),
            &limits(128),
            &Cancellation::default(),
            outputs(&case),
        )
        .unwrap();

        assert_eq!(receipt::accept(report).is_ok(), expected);
    }
}

#[test]
fn migration_lifecycle_existing_evidence_is_not_overwritten_or_launched() {
    let case = Case::new(Behavior::Clean, vec![ready()]);
    std::fs::write(case.root.path().join("out"), b"existing evidence").unwrap();

    let result = supervise(
        &specification(&case),
        &limits(128),
        &Cancellation::default(),
        outputs(&case),
    );

    assert!(result.is_err());
    assert_eq!(
        std::fs::read(case.root.path().join("out")).unwrap(),
        b"existing evidence"
    );
    assert!(!case.native.join("audit.json").exists());
}

#[test]
fn migration_lifecycle_actual_adapter_function_is_exercised() {
    let case = Case::new(Behavior::Clean, vec![ready()]);

    let result = crate::automation::client_readiness::run(&repository(), &case.arguments());

    assert!(result.is_ok(), "{result:?}");
    assert!(case.native.join("handler").is_file());
    case.assert_removed();
}

#[test]
fn migration_lifecycle_spawn_failure_has_no_fixture_or_unsanitized_error() {
    let case = Case::new(Behavior::Clean, vec![ready()]);
    let mut spec = specification(&case);
    spec.executable = case.root.path().join("missing-binary");

    let result = supervise(
        &spec,
        &limits(128),
        &Cancellation::default(),
        outputs(&case),
    );

    assert!(matches!(
        result,
        Err(Failure::Io {
            operation: "spawn",
            ..
        })
    ));
    assert!(!case.native.join("audit.json").exists());
}

#[test]
fn migration_lifecycle_zero_cap_deadline_remains_failure() {
    let case = Case::new(Behavior::LateReady, vec![]);

    let report = supervise(
        &specification(&case),
        &limits(0),
        &Cancellation::default(),
        outputs(&case),
    )
    .unwrap();

    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(report.stdout.bytes_retained.is_empty() && report.stderr.bytes_retained.is_empty());
    assert!(report.cleanup.complete && !report.cleanup.forced);
    assert!(receipt::accept(report).is_err());
}
