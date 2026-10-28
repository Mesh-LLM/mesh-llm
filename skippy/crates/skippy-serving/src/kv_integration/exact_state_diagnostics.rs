//! Opt-in, content-free diagnostics for exact-state admission and retention.

use std::sync::OnceLock;

use skippy_events::diagnostics::{self, ServingDiagnostic};

/// Numeric fields only: callers cannot accidentally log cache keys or tokens.
pub(crate) fn emit_exact_state_diagnostic(event: &'static str, fields: &[(&str, u64)]) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !ENABLED.get_or_init(|| {
        std::env::var("SKIPPY_EXACT_STATE_DIAGNOSTICS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
    }) {
        return;
    }
    let context = fields
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    let _ = diagnostics::emit(ServingDiagnostic::Info {
        message: format!("Skippy exact-state {event}"),
        context: Some(context),
    });
}
