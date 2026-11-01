//! Bounded presentation events for parsed CLI command dispatch.
//!
//! Command handlers use this adapter before the host runtime exists, so its
//! fallback writes only static metadata to stderr. It never inspects command
//! arguments or error text, which keeps machine-readable command output on
//! stdout intact.

use super::{LogFormat, OutputEvent, OutputLevel, OutputSink, output_sink};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "command_summary_grammar.rs"]
mod command_summary_grammar;

/// Stable family for a parsed command. The enum intentionally groups commands
/// by responsibility instead of carrying user-supplied subcommand names or
/// arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliCommandFamily {
    Agent,
    Benchmark,
    Configuration,
    Diagnostics,
    Discovery,
    Hardware,
    Identity,
    Installation,
    Models,
    Plugin,
    Process,
    Runtime,
    Skills,
    Unknown,
}

/// A grammar-validated summary of a parsed CLI command.
///
/// External callers must use [`CliCommandSummary::sanitize`] to construct a
/// summary.
///
/// ```compile_fail
/// use mesh_llm_events::CliCommandSummary;
///
/// let _summary = CliCommandSummary::new("mesh-llm status");
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliCommandSummary(String);

impl CliCommandSummary {
    fn new(value: &str) -> Option<Self> {
        let token_count = value.split_whitespace().count();
        (!value.is_empty()
            && token_count > 0
            && token_count <= 32
            && value.chars().count() <= 256
            && !value.chars().any(char::is_control))
        .then(|| Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn sanitize(value: &str) -> Option<Self> {
        let summary = Self::new(value)?;
        is_safe_summary(&summary.0).then_some(summary)
    }
}

fn is_safe_summary(value: &str) -> bool {
    command_summary_grammar::is_safe_summary(value)
}

impl CliCommandFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Benchmark => "benchmark",
            Self::Configuration => "configuration",
            Self::Diagnostics => "diagnostics",
            Self::Discovery => "discovery",
            Self::Hardware => "hardware",
            Self::Identity => "identity",
            Self::Installation => "installation",
            Self::Models => "models",
            Self::Plugin => "plugin",
            Self::Process => "process",
            Self::Runtime => "runtime",
            Self::Skills => "skills",
            Self::Unknown => "unknown",
        }
    }
}

/// Static lifecycle outcome for a parsed CLI command dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliCommandOutcome {
    Started,
    Completed,
    Failed,
    Rejected,
    ParseFailed,
}

impl CliCommandOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::ParseFailed => "parse_failed",
        }
    }

    pub const fn code(self) -> &'static str {
        match self {
            Self::Started => "cli_command_started",
            Self::Completed => "cli_command_completed",
            Self::Failed => "cli_command_failed",
            Self::Rejected => "cli_command_rejected",
            Self::ParseFailed => "cli_parse_failed",
        }
    }

    pub(crate) const fn level(self) -> OutputLevel {
        match self {
            Self::Started | Self::Completed => OutputLevel::Info,
            Self::Failed | Self::Rejected | Self::ParseFailed => OutputLevel::Warn,
        }
    }
}

/// Emit a structured command event without contaminating command JSON output.
///
/// Presentation is silent unless the CLI enabled verbose command tracking with
/// `--debug`. One-shot commands keep the terminal clean by default; the raw
/// stderr fallback (and a pretty sink) only render these lines for debugging.
/// The durable audit bridge installed by the dispatcher is independent of this
/// toggle and keeps recording command outcomes either way.
///
/// A pretty sink receives the typed [`OutputEvent`]. A JSON sink is deliberately
/// bypassed because command handlers can independently write their JSON result
/// to stdout; the static command record is written to stderr instead. The same
/// stderr fallback is used before the runtime has initialized an output sink.
pub fn emit_cli_command_event(
    family: CliCommandFamily,
    outcome: CliCommandOutcome,
) -> io::Result<()> {
    if !cli_command_event_verbose() {
        return Ok(());
    }

    let event = OutputEvent::CliCommandLifecycle { family, outcome };
    let sink = output_sink();
    if emit_to_pretty_sink(&event, sink.as_deref()) {
        return Ok(());
    }

    let stderr_handle = io::stderr();
    let mut stderr = stderr_handle.lock();
    write_cli_command_event_to_stderr(&event, &mut stderr)
}

/// Process-global toggle for presenting CLI command lifecycle events to the
/// terminal. Defaults to off; the shipped binary enables it only when `--debug`
/// is provided.
static CLI_COMMAND_EVENT_VERBOSE: AtomicBool = AtomicBool::new(false);

/// Enable or disable the terminal presentation of CLI command lifecycle events.
///
/// This controls only the presentation emitted by [`emit_cli_command_event`]
/// (pretty sink or raw stderr fallback). It does not affect the durable
/// operational-audit bridge, which records command outcomes independently.
pub fn set_cli_command_event_verbose(enabled: bool) {
    CLI_COMMAND_EVENT_VERBOSE.store(enabled, Ordering::Relaxed);
}

fn cli_command_event_verbose() -> bool {
    CLI_COMMAND_EVENT_VERBOSE.load(Ordering::Relaxed)
}

#[cfg(test)]
fn emit_cli_command_event_with_sink<W: Write>(
    event: &OutputEvent,
    sink: Option<&dyn OutputSink>,
    stderr: &mut W,
) -> io::Result<()> {
    if emit_to_pretty_sink(event, sink) {
        return Ok(());
    }

    write_cli_command_event_to_stderr(event, stderr)
}

fn emit_to_pretty_sink(event: &OutputEvent, sink: Option<&dyn OutputSink>) -> bool {
    let Some(sink) = sink else {
        return false;
    };
    matches!(sink.mode(), LogFormat::Pretty) && sink.emit_event(event.clone()).is_ok()
}

fn write_cli_command_event_to_stderr<W: Write>(
    event: &OutputEvent,
    stderr: &mut W,
) -> io::Result<()> {
    let OutputEvent::CliCommandLifecycle { family, outcome } = event else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "CLI command event adapter received a different event variant",
        ));
    };

    writeln!(
        stderr,
        "mesh-llm command event: family={} code={} outcome={}",
        family.as_str(),
        outcome.code(),
        outcome.as_str(),
    )?;
    stderr.flush()
}

#[cfg(test)]
#[path = "command_lifecycle/tests.rs"]
mod tests;
