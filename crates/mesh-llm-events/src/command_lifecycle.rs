//! Bounded presentation events for parsed CLI command dispatch.
//!
//! Command handlers use this adapter before the host runtime exists, so its
//! fallback writes only static metadata to stderr. It never inspects command
//! arguments or error text, which keeps machine-readable command output on
//! stdout intact.

use super::{LogFormat, OutputEvent, OutputLevel, OutputSink, output_sink};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

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
    use crate::{clear_output_sink, set_output_sink};
    use std::sync::{Arc, Mutex};

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

    struct VerboseResetGuard;

    impl Drop for VerboseResetGuard {
        fn drop(&mut self) {
            set_cli_command_event_verbose(false);
        }
    }

    struct OutputSinkResetGuard;

    impl Drop for OutputSinkResetGuard {
        fn drop(&mut self) {
            clear_output_sink();
        }
    }

    #[test]
    fn public_emit_is_silent_unless_verbose_enabled() {
        let sink = Arc::new(RecordingSink::new(LogFormat::Pretty));
        let _sink_guard = OutputSinkResetGuard;
        let _verbose_guard = VerboseResetGuard;
        set_output_sink(sink.clone());

        set_cli_command_event_verbose(false);
        emit_cli_command_event(CliCommandFamily::Runtime, CliCommandOutcome::Started)
            .expect("silent emission must succeed");
        assert!(
            sink.events.lock().expect("recording sink lock").is_empty(),
            "without --debug the command event must not be presented"
        );

        set_cli_command_event_verbose(true);
        emit_cli_command_event(CliCommandFamily::Runtime, CliCommandOutcome::Completed)
            .expect("verbose emission must succeed");
        assert_eq!(
            *sink.events.lock().expect("recording sink lock"),
            vec![OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Runtime,
                outcome: CliCommandOutcome::Completed,
            }],
            "with --debug the command event must reach the pretty sink"
        );
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
            CliCommandFamily::Process,
            CliCommandFamily::Runtime,
            CliCommandFamily::Skills,
            CliCommandFamily::Unknown,
        ];
        let outcomes = [
            CliCommandOutcome::Started,
            CliCommandOutcome::Completed,
            CliCommandOutcome::Failed,
            CliCommandOutcome::Rejected,
            CliCommandOutcome::ParseFailed,
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
