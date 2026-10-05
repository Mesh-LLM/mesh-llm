//! Standalone command output formatting.
use skippy_events::diagnostics::{DiagnosticSink, ServingDiagnostic, set_diagnostic_sink};
use std::{
    io::{self, IsTerminal, Write},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputMode {
    Auto,
    Human,
    Json,
    Jsonl,
}

impl OutputMode {
    fn resolved(self) -> Self {
        match self {
            Self::Auto if io::stdout().is_terminal() => Self::Human,
            Self::Auto => Self::Json,
            explicit => explicit,
        }
    }
}

static MODE: OnceLock<OutputMode> = OnceLock::new();
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn mode() -> OutputMode {
    *MODE.get().unwrap_or(&OutputMode::Json)
}

struct StandaloneDiagnostics(OutputMode);
impl DiagnosticSink for StandaloneDiagnostics {
    fn emit(&self, diagnostic: ServingDiagnostic) -> io::Result<()> {
        if self.0 == OutputMode::Jsonl {
            let (kind, message, context) = match diagnostic {
                ServingDiagnostic::Info { message, context } => ("info", message, context),
                ServingDiagnostic::Warning { message, context } => ("warning", message, context),
                ServingDiagnostic::Status { message } => ("status", message, None),
            };
            return event(
                kind,
                &serde_json::json!({"message":message,"context":context}),
            );
        }
        render(&mut io::stderr().lock(), diagnostic)
    }
}

fn render(output: &mut impl Write, diagnostic: ServingDiagnostic) -> io::Result<()> {
    match diagnostic {
        ServingDiagnostic::Info { message, context } => {
            writeln!(output, "{message}")?;
            if let Some(context) = context {
                writeln!(output, "  {context}")?;
            }
            Ok(())
        }
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

pub fn install(requested: OutputMode) {
    let resolved = requested.resolved();
    let _ = MODE.set(resolved);
    set_diagnostic_sink(Arc::new(StandaloneDiagnostics(resolved)));
}

pub fn write_json(value: &impl serde::Serialize) -> anyhow::Result<()> {
    if mode() == OutputMode::Jsonl {
        event("result", value)?;
        return Ok(());
    }
    let mut output = io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    Ok(())
}

pub fn present(
    value: &impl serde::Serialize,
    render_human: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> anyhow::Result<()> {
    if mode() == OutputMode::Human {
        render_human(&mut io::stdout().lock())?;
        Ok(())
    } else {
        write_json(value)
    }
}

/// Emit one versioned machine event. Human output stays on stderr.
pub fn event(kind: &str, data: &impl serde::Serialize) -> io::Result<()> {
    if mode() == OutputMode::Jsonl {
        let mut output = io::stdout().lock();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({
                "schema_version": 1, "sequence": sequence, "type": kind, "data": data
            }),
        )
        .map_err(io::Error::other)?;
        writeln!(output)?;
        output.flush()
    } else {
        Ok(())
    }
}

pub fn failure(error: &anyhow::Error) -> io::Result<()> {
    if mode() == OutputMode::Jsonl {
        event(
            "error",
            &serde_json::json!({"message": format!("{error:#}")}),
        )
    } else {
        writeln!(io::stderr().lock(), "error: {error:#}")
    }
}

/// Render native diagnostic fragments without corrupting machine output.
pub fn native_log(text: &str) -> io::Result<()> {
    if mode() == OutputMode::Jsonl {
        return event("native_log", &serde_json::json!({"message": text}));
    }
    let mut output = io::stderr().lock();
    output.write_all(text.as_bytes())?;
    output.flush()
}

/// Write streamed interactive output without waiting for a newline.
pub(crate) fn write_text(text: &str) -> io::Result<()> {
    let mut output = io::stdout().lock();
    output.write_all(text.as_bytes())?;
    output.flush()
}

pub(crate) fn write_line(text: &str) -> io::Result<()> {
    writeln!(io::stdout().lock(), "{text}")
}

pub(crate) fn write_status(message: &str) -> io::Result<()> {
    if mode() == OutputMode::Jsonl {
        return event("status", &serde_json::json!({"message":message}));
    }
    writeln!(io::stderr().lock(), "{message}")
}

/// Keep prompt statistics separate from response text and dim them on terminals.
pub(crate) fn write_prompt_stats(message: &str) -> io::Result<()> {
    let dim = io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    render_prompt_stats(&mut io::stderr().lock(), message, dim)
}

fn render_prompt_stats(output: &mut impl Write, message: &str, dim: bool) -> io::Result<()> {
    writeln!(output)?;
    if dim {
        writeln!(output, "\x1b[2m{message}\x1b[0m")
    } else {
        writeln!(output, "{message}")
    }
}

pub fn status(message: &str) -> io::Result<()> {
    write_status(message)
}

pub fn progress(label: &str, current: u64, total: u64) -> io::Result<()> {
    progress_with_unit(label, current, total, "bytes")
}

pub fn progress_with_unit(label: &str, current: u64, total: u64, unit: &str) -> io::Result<()> {
    if total == 0 {
        return Ok(());
    }
    if mode() == OutputMode::Jsonl {
        return event(
            "progress",
            &serde_json::json!({
                "phase": label, "current": current, "total": total, "unit": unit
            }),
        );
    }
    if mode() != OutputMode::Human {
        return Ok(());
    }
    let width = 24;
    let completed = ((current.min(total) as f64 / total as f64) * width as f64) as usize;
    let percent = current.min(total).saturating_mul(100) / total;
    let mut output = io::stderr().lock();
    let amount = if unit == "bytes" {
        format!("{:.1}/{:.1} MB", current as f64 / 1e6, total as f64 / 1e6)
    } else {
        format!("{current}/{total} {unit}")
    };
    if io::stderr().is_terminal() {
        write!(
            output,
            "\r📥 {label} [{}{}] {percent:>3}% ({amount})",
            "█".repeat(completed),
            "░".repeat(width - completed),
        )?;
        if current >= total {
            writeln!(output)?;
        }
        output.flush()
    } else {
        writeln!(output, "{label}: {percent}% ({amount})")
    }
}

pub fn stdout_is_terminal() -> bool {
    io::stdout().is_terminal()
}

pub fn stderr_is_terminal() -> bool {
    io::stderr().is_terminal()
}

pub(crate) fn model_console_out() -> Box<dyn Write + Send> {
    Box::new(io::stdout())
}
pub(crate) fn model_console_err() -> Box<dyn Write + Send> {
    if mode() == OutputMode::Jsonl {
        Box::new(ModelDiagnosticWriter(Vec::new()))
    } else {
        Box::new(io::stderr())
    }
}
pub(crate) fn model_machine_out() -> Box<dyn Write + Send> {
    if mode() == OutputMode::Jsonl {
        Box::new(ModelJsonWriter(Vec::new()))
    } else {
        Box::new(io::stdout())
    }
}
struct ModelJsonWriter(Vec<u8>);
impl Write for ModelJsonWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(bytes);
        if self.0.last() == Some(&b'\n')
            && let Ok(value) = serde_json::from_slice::<serde_json::Value>(&self.0)
        {
            event("result", &value)?;
            self.0.clear();
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
struct ModelDiagnosticWriter(Vec<u8>);
impl Write for ModelDiagnosticWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(bytes);
        while let Some(end) = self.0.iter().position(|byte| *byte == b'\n') {
            let line = self.0.drain(..=end).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            write_status(line.trim_end_matches('\n'))?;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if !self.0.is_empty() {
            write_status(&String::from_utf8_lossy(&self.0))?;
            self.0.clear();
        }
        Ok(())
    }
}

/// Page a long model table on interactive terminals; return false for direct output.
pub(crate) fn page_model_table(output: &str) -> anyhow::Result<bool> {
    use std::{
        ffi::OsStr,
        process::{Command, Stdio},
    };
    if !io::stdin().is_terminal()
        || !io::stdout().is_terminal()
        || std::env::var_os("TERM")
            .as_deref()
            .is_some_and(|term| term.eq_ignore_ascii_case(OsStr::new("dumb")))
    {
        return Ok(false);
    }
    let mut child = match Command::new("less")
        .args(["-F", "-R", "-X"])
        .stdin(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if let Some(mut input) = child.stdin.take() {
        match input.write_all(output.as_bytes()) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {}
            Err(error) => return Err(error.into()),
        }
    }
    child.wait()?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_stats_only_use_ansi_when_requested() {
        for dim in [false, true] {
            let mut output = Vec::new();
            render_prompt_stats(&mut output, "⚡ 43.2 tok/s · 📝 128 out", dim).unwrap();
            let output = String::from_utf8(output).unwrap();
            assert_eq!(output.contains("\x1b[2m"), dim);
            assert_eq!(output.contains("\x1b[0m"), dim);
            assert!(output.contains("⚡ 43.2 tok/s · 📝 128 out"));
        }
    }

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
