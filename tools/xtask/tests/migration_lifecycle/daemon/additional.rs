use super::{
    cli::{case, command},
    protocol::{Behavior, Plan},
};
use crate::support::Sentinel;

#[test]
fn d14_endless_models_body_is_bounded() {
    let case = case(&Plan {
        behavior: Behavior::EndlessBody,
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let output = command(&case).output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostics.contains("response_limit") || diagnostics.contains("deadline expired"),
        "{diagnostics}"
    );
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn d22_raw_pipe_correlation_precedes_sensitive_line_suppression() {
    let mut plan = Plan::default();
    plan.record = plan.record.replace(
        "\"source\"",
        "\"token\":\"synthetic-pipe-secret\",\"source\"",
    );
    let case = case(&plan);
    let mut sentinel = Sentinel::new(&case);
    let output = command(&case).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("synthetic-pipe-secret"));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn d10_each_wrong_terminal_field_fails_cli() {
    for (from, to) in [
        ("models", "chat"),
        ("direct_http", "mesh"),
        ("GET", "POST"),
        ("model_listing", "other"),
        ("CODE", "201"),
        ("EVENT", "request_admitted"),
        ("ID", "00000000-0000-4000-8000-000000000000"),
    ] {
        let mut plan = Plan::default();
        plan.record = plan.record.replace(from, to);
        let case = case(&plan);
        let mut sentinel = Sentinel::new(&case);
        let output = command(&case).output().unwrap();
        assert!(!output.status.success(), "{from}");
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("attribution_unavailable"));
        assert!(sentinel.0.try_wait().unwrap().is_none());
        case.assert_removed();
    }
}

#[test]
fn d05_each_malformed_or_mismatched_status_fails_cli() {
    for (body, reason) in [
        ("42", "malformed_status"),
        (r#"{"api_port":API}"#, "malformed_status"),
        (
            r#"{"api_port":API,"local_instances":[{"pid":"PID","is_self":true}]}"#,
            "malformed_status",
        ),
        (
            r#"{"api_port":API,"api_port":API,"local_instances":[]}"#,
            "malformed_status",
        ),
        (
            r#"{"api_port":API,"local_instances":[{"pid":0,"is_self":true}]}"#,
            "ownership_mismatch",
        ),
        (
            r#"{"api_port":API,"local_instances":[{"pid":PID,"is_self":true},{"pid":PID,"is_self":true}]}"#,
            "ownership_mismatch",
        ),
        (
            r#"{"api_port":1,"local_instances":[{"pid":PID,"is_self":true}]}"#,
            "ownership_mismatch",
        ),
    ] {
        let case = case(&Plan {
            status: body.into(),
            ..Plan::default()
        });
        let mut sentinel = Sentinel::new(&case);
        let output = command(&case).output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(reason));
        assert!(!case.native.join("models.count").exists());
        assert!(sentinel.0.try_wait().unwrap().is_none());
        case.assert_removed();
    }
}

#[test]
fn d05_invalid_utf8_status_fails_cli() {
    let case = case(&Plan {
        status_wire: Some(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n\xff".to_vec()),
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let output = command(&case).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("malformed_status"));
    assert!(!case.native.join("models.count").exists());
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}
