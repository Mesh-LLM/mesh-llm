//! Pre-checkout Docker authority CLI using copied automation and inert config.
//! This does not qualify protected artifact delivery or real provider isolation.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap, ffi::OsString, fs, num::NonZeroUsize, path::PathBuf, time::Duration,
};

struct Fixture {
    temporary: tempfile::TempDir,
    binary: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let binary = temporary.path().join("copied-xtask");
        fs::copy(env!("CARGO_BIN_EXE_xtask"), &binary).unwrap();
        Self { temporary, binary }
    }

    fn config(&self) -> PathBuf {
        self.temporary.path().join("config.json")
    }

    fn run(&self, selected: &str, auth: Option<OsString>) -> process::RawProcessReport {
        let environment = auth.map_or_else(BTreeMap::new, |value| {
            // Deliberately public, synthetic data: supervisor redaction must not hide a CLI leak.
            BTreeMap::from([("DOCKER_AUTH_CONFIG".into(), Value::Public(value))])
        });
        let arguments = vec![
            Value::Public("ci-ops".into()),
            Value::Public("authority-audit".into()),
            Value::Public("docker".into()),
            Value::Public(self.config().into_os_string()),
            Value::Public("--depot-selected".into()),
            Value::Public(selected.into()),
        ];
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: self.binary.clone(),
                arguments,
                cwd: self.temporary.path().canonicalize().unwrap(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 16384,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(16384),
                stderr: NonZeroUsize::new(16384),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none());
        assert!(report.process.cleanup.complete);
        assert!(report.stdout.as_ref().unwrap().as_bytes().is_empty());
        assert!(
            !String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
                .contains("inert-private-marker")
        );
        report
    }

    fn sources(&self, raw: &str, selected: &str, expected: i32, reason: &str) {
        for file in [false, true] {
            let auth = if file {
                fs::write(self.config(), raw).unwrap();
                None
            } else {
                Some(raw.into())
            };
            let report = self.run(selected, auth);
            assert_eq!(report.process.status.unwrap().code(), Some(expected));
            let stderr = String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes());
            if expected == 0 {
                assert!(stderr.is_empty());
            } else {
                assert!(stderr.contains(reason), "{stderr}");
            }
            if file {
                assert_eq!(fs::read(self.config()).unwrap(), raw.as_bytes());
                fs::remove_file(self.config()).unwrap();
            }
        }
    }
}

#[test]
fn copied_docker_audit_without_checkout_or_environment_accepts_absent_auth_sources() {
    let fixture = Fixture::new();
    for selected in ["true", "false"] {
        for auth in [None, Some(OsString::new())] {
            let report = fixture.run(selected, auth);
            assert!(report.process.status.unwrap().success());
            assert!(report.stderr.unwrap().as_bytes().is_empty());
        }
    }
}

#[test]
fn copied_docker_audit_rejects_selected_depot_auth_even_when_empty() {
    let fixture = Fixture::new();
    for raw in [
        r#"{"auths":{}}"#,
        r#"{"credHelpers":{}}"#,
        r#"{"credsStore":""}"#,
    ] {
        fixture.sources(raw, "true", 1, "authentication");
    }
    assert_eq!(
        fixture
            .run("true", Some("{}".into()))
            .process
            .status
            .unwrap()
            .code(),
        Some(1)
    );
    fs::write(fixture.config(), b"{}").unwrap();
    assert!(fixture.run("true", None).process.status.unwrap().success());
}

#[test]
fn copied_docker_audit_hosted_context_accepts_non_depot_auth_sources() {
    let fixture = Fixture::new();
    for raw in [
        r#"{"auths":{"ghcr.io":{"auth":"inert-private-marker"}}}"#,
        r#"{"credHelpers":{"example.test":"inert-private-marker"}}"#,
        r#"{"credsStore":"inert-private-marker"}"#,
    ] {
        fixture.sources(raw, "false", 0, "");
    }
}

#[test]
fn copied_docker_audit_hosted_context_refuses_depot_auth_without_payload_leaks() {
    let fixture = Fixture::new();
    for raw in [
        r#"{"auths":{"ORG.REGISTRY.DEPOT.DEV":{"auth":"inert-private-marker"}}}"#,
        r#"{"credHelpers":{"depot.dev":"inert-private-marker"}}"#,
        r#"{"credsStore":"Depot.Dev-inert-private-marker"}"#,
    ] {
        fixture.sources(raw, "false", 1, "depot-authentication");
    }
}

#[test]
fn copied_docker_audit_refuses_malformed_json_and_section_types() {
    let fixture = Fixture::new();
    for raw in [
        "null",
        "[]",
        "{inert-private-marker",
        r#"{"auths":NaN}"#,
        r#"{"auths":[]}"#,
        r#"{"credHelpers":false}"#,
        r#"{"credsStore":{}}"#,
    ] {
        fixture.sources(raw, "false", 1, "malformed");
    }
}

#[test]
fn copied_docker_audit_refuses_invalid_utf8_file_and_preserves_bytes() {
    let fixture = Fixture::new();
    fs::write(fixture.config(), [0xff]).unwrap();
    for selected in ["true", "false"] {
        let report = fixture.run(selected, None);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("malformed"));
        assert_eq!(fs::read(fixture.config()).unwrap(), [0xff]);
    }
}

#[test]
fn copied_docker_audit_refuses_non_unicode_auth_environment() {
    use std::os::unix::ffi::OsStringExt;
    let fixture = Fixture::new();
    let report = fixture.run("false", Some(OsString::from_vec(vec![0xff])));
    assert_eq!(report.process.status.unwrap().code(), Some(1));
    assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("malformed"));
}

#[test]
fn copied_docker_audit_refuses_unbounded_provider_selection() {
    let fixture = Fixture::new();
    for selected in ["", "True", "yes", "1"] {
        let report = fixture.run(selected, None);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("usage:"));
    }
}
