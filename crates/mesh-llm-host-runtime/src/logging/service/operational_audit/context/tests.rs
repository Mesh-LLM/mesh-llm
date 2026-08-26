use super::*;

#[test]
fn typed_context_is_sanitized_bounded_and_cardinality_limited() {
    let long_subject = format!("{}?api_key=private", "model".repeat(80));
    let mut context = OperationalAuditContext::new()
        .subject(OperationalAuditSubjectKind::Model, &long_subject)
        .operation_id("runtime-7")
        .reason_code("load_failed")
        .outcome("failed")
        .duration_ms(42);
    for index in 0..12 {
        let key = match index {
            0 => "metric_0",
            1 => "metric_1",
            2 => "metric_2",
            3 => "metric_3",
            4 => "metric_4",
            5 => "metric_5",
            6 => "metric_6",
            7 => "metric_7",
            8 => "metric_8",
            9 => "metric_9",
            10 => "metric_10",
            _ => "metric_11",
        };
        context = context.numeric_summary(key, index);
    }

    let fields = context.fields();
    assert_eq!(fields["context_version"], 1);
    assert_eq!(fields["subject_kind"], "model");
    assert_eq!(fields["operation_id"], "runtime-7");
    assert_eq!(fields["reason_code"], "load_failed");
    assert_eq!(fields["outcome"], "failed");
    assert_eq!(fields["duration_ms"], 42);
    assert!(fields["subject_id"].as_str().unwrap().chars().count() <= 256);
    assert!(!fields["subject_id"].as_str().unwrap().contains("private"));
    assert_eq!(fields["numeric_summaries"].as_object().unwrap().len(), 8);
}

#[test]
fn invalid_static_codes_are_not_admitted() {
    let fields = OperationalAuditContext::new()
        .reason_code("NOT VALID")
        .outcome("also-not-valid")
        .numeric_summary("bad-key", 1)
        .fields();
    assert!(fields.get("reason_code").is_none());
    assert!(fields.get("outcome").is_none());
    assert!(fields.get("numeric_summaries").is_none());
}

#[test]
fn context_values_redact_url_credentials_and_query_secrets() {
    let fields = OperationalAuditContext::new()
        .subject(
            OperationalAuditSubjectKind::Model,
            "https://alice:top-secret@example.test/model?api_key=query-secret&safe=1",
        )
        .fields();
    let subject_id = fields["subject_id"].as_str().expect("subject id");

    assert!(!subject_id.contains("alice"));
    assert!(!subject_id.contains("top-secret"));
    assert!(!subject_id.contains("query-secret"));
    assert!(subject_id.contains("[REDACTED]@example.test"));
    assert!(subject_id.contains("api_key=[REDACTED]"));
    assert!(subject_id.contains("safe=1"));
}

#[test]
fn mesh_peer_direct_path_preserves_identity_address_and_path_type() {
    let remote_addr = "192.168.1.44:11204".parse().expect("socket address");

    let fields = OperationalAuditContext::new()
        .mesh_peer_subject("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        .network_path(OperationalAuditPathType::Direct, Some(remote_addr))
        .fields();

    assert_eq!(fields["context_version"], 1);
    assert_eq!(fields["subject_kind"], "mesh_peer");
    assert_eq!(
        fields["subject_id"],
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    );
    assert_eq!(fields["remote_addr"], "192.168.1.44:11204");
    assert_eq!(fields["path_type"], "direct");
}

#[test]
fn mesh_peer_relay_path_omits_observed_address() {
    let relay_addr = "203.0.113.10:443".parse().expect("socket address");

    let fields = OperationalAuditContext::new()
        .mesh_peer_subject("peer-hex")
        .network_path(OperationalAuditPathType::Relay, Some(relay_addr))
        .fields();

    assert_eq!(fields["path_type"], "relay");
    assert!(fields.get("remote_addr").is_none());
}

#[test]
fn command_summary_is_serialized_with_context_bounds_and_redaction() {
    let fields = OperationalAuditContext::new()
        .command_summary("mesh-llm load name [REDACTED]")
        .fields();
    let summary = fields["command_summary"].as_str().expect("command summary");
    assert!(summary.chars().count() <= 256);
    assert_eq!(summary, "mesh-llm load name [REDACTED]");
}

#[test]
fn command_summary_context_drops_malformed_values_and_overlong_token_lists() {
    let fields = OperationalAuditContext::new()
        .command_summary("mesh-llm load private-model-name")
        .fields();
    assert!(fields.get("command_summary").is_none());

    let fields = OperationalAuditContext::new()
        .command_summary(&format!("mesh-llm {}", "x ".repeat(32)))
        .fields();
    assert!(fields.get("command_summary").is_none());
}
