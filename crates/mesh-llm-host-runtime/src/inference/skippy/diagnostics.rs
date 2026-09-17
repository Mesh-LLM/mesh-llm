//! Mesh presentation adapter for Skippy serving diagnostics.
use skippy_events::diagnostics::{DiagnosticSink, ServingDiagnostic, set_diagnostic_sink};
use std::{
    io::{self, Write},
    sync::Arc,
};

struct MeshDiagnostics;
impl DiagnosticSink for MeshDiagnostics {
    fn emit(&self, diagnostic: ServingDiagnostic) -> io::Result<()> {
        match diagnostic {
            ServingDiagnostic::Warning { message, context } => {
                mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
                    message,
                    context,
                })
            }
            ServingDiagnostic::Status { message } => {
                // Preserve JSON/TUI suppression from Mesh's console policy.
                writeln!(mesh_llm_events::console_out(), "{message}")
            }
        }
    }
}

pub(crate) fn install() {
    // Explicit routing preserves warnings in every Mesh output mode and keeps
    // listener status independent of the skippy_server=warn tracing filter.
    set_diagnostic_sink(Arc::new(MeshDiagnostics));
}
