//! Bounded presentation events for parsed CLI command dispatch.
//!
//! Command handlers use this adapter before the host runtime exists, so its
//! fallback writes only static metadata to stderr. It never inspects command
//! arguments or error text, which keeps machine-readable command output on
//! stdout intact.

use super::{LogFormat, OutputEvent, OutputLevel, OutputSink, output_sink};
use std::io::{self, Write};

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
    Runtime,
    Skills,
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
            Self::Runtime => "runtime",
            Self::Skills => "skills",
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
}

impl CliCommandOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
        }
    }

    pub const fn code(self) -> &'static str {
        match self {
            Self::Started => "cli_command_started",
            Self::Completed => "cli_command_completed",
            Self::Failed => "cli_command_failed",
            Self::Rejected => "cli_command_rejected",
        }
    }

    pub(crate) const fn level(self) -> OutputLevel {
        match self {
            Self::Started | Self::Completed => OutputLevel::Info,
            Self::Failed | Self::Rejected => OutputLevel::Warn,
        }
    }
}

/// Emit a structured command event without contaminating command JSON output.
///
/// A pretty sink receives the typed [`OutputEvent`]. A JSON sink is deliberately
/// bypassed because command handlers can independently write their JSON result
/// to stdout; the static command record is written to stderr instead. The same
/// stderr fallback is used before the runtime has initialized an output sink.
pub fn emit_cli_command_event(
    family: CliCommandFamily,
    outcome: CliCommandOutcome,
) -> io::Result<()> {
    let event = OutputEvent::CliCommandLifecycle { family, outcome };
    let sink = output_sink();
    if emit_to_pretty_sink(&event, sink.as_deref()) {
        return Ok(());
    }

    let stderr_handle = io::stderr();
    let mut stderr = stderr_handle.lock();
    write_cli_command_event_to_stderr(&event, &mut stderr)
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
        unreachable!("CLI command event adapter received a different event variant");
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
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct RecordingSink {
        mode: LogFormat,
        events: Mutex<Vec<OutputEvent>>,
    }

    impl RecordingSink {
        fn new(mode: LogFormat) -> Self {
            Self {
                mode,
                events: Mutex::new(Vec::new()),
            }
        }
    }

    impl OutputSink for RecordingSink {
        fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
            self.events.lock().expect("recording sink lock").push(event);
            Ok(())
        }

        fn mode(&self) -> LogFormat {
            self.mode
        }
    }

    #[test]
    fn pretty_sink_receives_typed_command_event() {
        let sink = RecordingSink::new(LogFormat::Pretty);
        let mut stderr = Vec::new();
        let event = OutputEvent::CliCommandLifecycle {
            family: CliCommandFamily::Runtime,
            outcome: CliCommandOutcome::Started,
        };

        emit_cli_command_event_with_sink(&event, Some(&sink), &mut stderr)
            .expect("pretty sink should receive command event");

        assert_eq!(
            *sink.events.lock().expect("recording sink lock"),
            vec![event]
        );
        assert!(stderr.is_empty());
    }

    #[test]
    fn json_sink_keeps_command_event_off_stdout() {
        let sink = RecordingSink::new(LogFormat::Json);
        let mut stderr = Vec::new();
        let event = OutputEvent::CliCommandLifecycle {
            family: CliCommandFamily::Models,
            outcome: CliCommandOutcome::Completed,
        };

        emit_cli_command_event_with_sink(&event, Some(&sink), &mut stderr)
            .expect("stderr fallback should write command event");

        assert!(sink.events.lock().expect("recording sink lock").is_empty());
        assert_eq!(
            String::from_utf8(stderr).expect("stderr must be utf-8"),
            "mesh-llm command event: family=models code=cli_command_completed outcome=completed\n"
        );
    }

    #[test]
    fn command_event_vocabulary_is_bounded_and_static() {
        let families = [
            CliCommandFamily::Agent,
            CliCommandFamily::Benchmark,
            CliCommandFamily::Configuration,
            CliCommandFamily::Diagnostics,
            CliCommandFamily::Discovery,
            CliCommandFamily::Hardware,
            CliCommandFamily::Identity,
            CliCommandFamily::Installation,
            CliCommandFamily::Models,
            CliCommandFamily::Plugin,
            CliCommandFamily::Runtime,
            CliCommandFamily::Skills,
        ];
        let outcomes = [
            CliCommandOutcome::Started,
            CliCommandOutcome::Completed,
            CliCommandOutcome::Failed,
            CliCommandOutcome::Rejected,
        ];

        for family in families {
            assert!(family.as_str().len() <= 24);
            assert!(
                family
                    .as_str()
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            );
        }
        for outcome in outcomes {
            assert!(outcome.code().len() <= 48);
            assert!(
                outcome
                    .code()
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            );
        }
    }
}
