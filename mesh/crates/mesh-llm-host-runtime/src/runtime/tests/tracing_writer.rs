use super::*;

#[test]
fn noq_proto_tracing_messages_use_transport_context() {
    let message = "2026-06-11T03:49:18.033043Z  WARN noq_proto::connection: err=LastOpenPath failed closing path";

    let (message, context) = normalize_tracing_message("noq_proto::connection", message);

    assert_eq!(message, "failed closing path (err=LastOpenPath)");
    assert_eq!(context.as_deref(), Some("transport"));
}

#[test]
fn routed_tracing_messages_strip_ansi_sequences() {
    let formatted = "\u{1b}[2m2026-06-11T03:49:18.033043Z\u{1b}[0m \u{1b}[33m WARN\u{1b}[0m";

    assert_eq!(
        strip_ansi_escape_sequences(formatted),
        "2026-06-11T03:49:18.033043Z  WARN"
    );
}

#[test]
fn non_proto_tracing_messages_keep_stderr_context() {
    let (message, context) = normalize_tracing_message("mesh_llm::runtime", "runtime warning");

    assert_eq!(message, "runtime warning");
    assert_eq!(context.as_deref(), Some("stderr"));
}
