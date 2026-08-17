use mesh_llm_events::audit::AuditLevel;
use mesh_llm_events::audit::AuditLogFormat;

#[test]
fn test_audit_types_exported() {
    let _ = AuditLogFormat::JsonLines;
    let _ = AuditLevel::Info;
}
