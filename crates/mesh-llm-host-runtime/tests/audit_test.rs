use mesh_llm_events::audit::AuditLevel;
use mesh_llm_events::audit::AuditLogFormat;
use mesh_llm_host_runtime::OperationalAuditSubjectKind;

#[test]
fn test_audit_types_exported() {
    let _ = AuditLogFormat::JsonLines;
    let _ = AuditLevel::Info;
}

#[test]
fn operational_audit_subject_kind_is_source_compatible() {
    let subjects = [
        (OperationalAuditSubjectKind::Runtime, "runtime"),
        (OperationalAuditSubjectKind::Model, "model"),
        (
            OperationalAuditSubjectKind::RuntimeInstance,
            "runtime_instance",
        ),
        (OperationalAuditSubjectKind::CliCommand, "cli_command"),
    ];

    for (subject, expected) in subjects {
        // This exhaustive match forces the table above to change when the enum grows.
        match subject {
            OperationalAuditSubjectKind::Runtime
            | OperationalAuditSubjectKind::Model
            | OperationalAuditSubjectKind::RuntimeInstance
            | OperationalAuditSubjectKind::CliCommand => {}
        }
        assert_eq!(subject.as_str(), expected);
    }
}
