//! Standalone command output formatting.
use skippy_events::diagnostics::{DiagnosticSink, ServingDiagnostic, set_diagnostic_sink};
use std::{
    io::{self, Write},
    sync::Arc,
};

struct StandaloneDiagnostics;
impl DiagnosticSink for StandaloneDiagnostics {
    fn emit(&self, diagnostic: ServingDiagnostic) -> io::Result<()> {
        render(&mut io::stderr().lock(), diagnostic)
    }
}

fn render(output: &mut impl Write, diagnostic: ServingDiagnostic) -> io::Result<()> {
    match diagnostic {
        ServingDiagnostic::Warning { message, context } => {
            write!(output, "warning: {message}")?;
            if let Some(context) = context {
                write!(output, " ({context})")?;
            }
            writeln!(output)
        }
        ServingDiagnostic::Status { message } => writeln!(output, "{message}"),
    }
}

pub fn install() {
    set_diagnostic_sink(Arc::new(StandaloneDiagnostics));
}

pub fn write_json(value: &impl serde::Serialize) -> anyhow::Result<()> {
    let mut output = io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostics_preserve_warning_context_and_status_text() {
        let mut output = Vec::new();
        render(
            &mut output,
            ServingDiagnostic::Warning {
                message: "cache disabled".into(),
                context: Some("stage=one".into()),
            },
        )
        .unwrap();
        render(
            &mut output,
            ServingDiagnostic::Status {
                message: "listening".into(),
            },
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "warning: cache disabled (stage=one)\nlistening\n"
        );
    }
}
