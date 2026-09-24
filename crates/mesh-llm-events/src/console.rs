//! Sanctioned writers for user-facing console text.
//!
//! CLI presentation code renders prose, tables and prompts that are meant for a
//! human reading a terminal. That text is not telemetry, so it does not belong
//! in [`crate::OutputEvent`]; but it must still respect who currently owns the
//! terminal. Writing it with `println!` hard-codes that decision at every call
//! site, which is how free-form text ends up interleaved with `--json` payloads
//! or painted over the interactive dashboard.
//!
//! These writers move the decision to the sink. A call site asks for a handle
//! and writes into it; whether the bytes reach the terminal is decided here,
//! once, from the currently installed [`crate::OutputSink`]:
//!
//! * [`console_out`] / [`console_err`] carry human-facing text. They discard
//!   their input while a JSON sink is installed or while the interactive
//!   dashboard owns the screen.
//! * [`machine_out`] carries machine-readable payloads — the document a
//!   `--json` command was asked to produce. It always reaches stdout, because
//!   suppressing it would mean answering a request with nothing.
//! * [`disclosure_err`] carries a one-time disclosure the product is obliged
//!   to make before it acts — today, the analytics first-run notice. It always
//!   reaches stderr, for the same reason `machine_out` always reaches stdout:
//!   a disclosure that a sink happened to swallow is not a disclosure, and
//!   on-by-default reporting is only defensible if it actually happens.
//!
//! One-shot CLI commands run before any sink is installed, so both writers pass
//! through to the terminal there.

use std::io::{self, IsTerminal, Write};

use crate::{interactive_tui_active, json_mode_enabled};

/// Where a [`ConsoleWriter`] sends the bytes it is given.
enum ConsoleTarget {
    Stdout(io::Stdout),
    Stderr(io::Stderr),
    /// Another surface owns the terminal; bytes are accepted and dropped.
    Discard,
}

/// A console handle obtained from the output facility.
///
/// Implements [`Write`], so call sites use `write!` / `writeln!` exactly as
/// they would against any other stream.
pub struct ConsoleWriter {
    target: ConsoleTarget,
}

impl ConsoleWriter {
    fn stdout() -> Self {
        Self {
            target: ConsoleTarget::Stdout(io::stdout()),
        }
    }

    fn stderr() -> Self {
        Self {
            target: ConsoleTarget::Stderr(io::stderr()),
        }
    }

    fn discard() -> Self {
        Self {
            target: ConsoleTarget::Discard,
        }
    }

    /// Whether this handle is attached to a terminal.
    ///
    /// Use this instead of probing `io::stdout()` directly when deciding
    /// whether to emit ANSI styling: a suppressed handle reports `false`, so
    /// escape sequences are not built for output nobody will read.
    pub fn is_terminal(&self) -> bool {
        match &self.target {
            ConsoleTarget::Stdout(stream) => stream.is_terminal(),
            ConsoleTarget::Stderr(stream) => stream.is_terminal(),
            ConsoleTarget::Discard => false,
        }
    }

    /// Whether writes to this handle are being dropped.
    ///
    /// Useful for skipping expensive rendering (table layout, progress
    /// redraws) that would otherwise be built and thrown away.
    pub fn is_suppressed(&self) -> bool {
        matches!(self.target, ConsoleTarget::Discard)
    }
}

impl Write for ConsoleWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.target {
            ConsoleTarget::Stdout(stream) => stream.write(buf),
            ConsoleTarget::Stderr(stream) => stream.write(buf),
            ConsoleTarget::Discard => Ok(buf.len()),
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match &mut self.target {
            ConsoleTarget::Stdout(stream) => stream.write_all(buf),
            ConsoleTarget::Stderr(stream) => stream.write_all(buf),
            ConsoleTarget::Discard => Ok(()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.target {
            ConsoleTarget::Stdout(stream) => stream.flush(),
            ConsoleTarget::Stderr(stream) => stream.flush(),
            ConsoleTarget::Discard => Ok(()),
        }
    }
}

/// Whether a surface other than plain console output currently owns the
/// terminal.
fn terminal_owned_elsewhere() -> bool {
    json_mode_enabled() || interactive_tui_active()
}

/// Human-facing console output on stdout.
///
/// Suppressed while a JSON sink is installed or the interactive dashboard is
/// active.
pub fn console_out() -> ConsoleWriter {
    if terminal_owned_elsewhere() {
        ConsoleWriter::discard()
    } else {
        ConsoleWriter::stdout()
    }
}

/// Human-facing console output on stderr — diagnostics, warnings and prompts
/// that should not contaminate a piped stdout.
///
/// Suppressed under the same conditions as [`console_out`].
pub fn console_err() -> ConsoleWriter {
    if terminal_owned_elsewhere() {
        ConsoleWriter::discard()
    } else {
        ConsoleWriter::stderr()
    }
}

/// Machine-readable command output on stdout.
///
/// This is the payload a `--json` command was invoked to produce, so it is
/// never suppressed.
pub fn machine_out() -> ConsoleWriter {
    ConsoleWriter::stdout()
}

/// A one-time disclosure the product must make on stderr.
///
/// Never suppressed. This is deliberately *not* [`console_err`]: the notice is
/// not prose a caller may discard when another surface owns the screen, it is
/// the statement that makes on-by-default behaviour defensible. Dropping it
/// because the first run happened to be `--json` or under the dashboard would
/// reintroduce exactly the "disclosure never happens" hole the marker file
/// exists to close.
///
/// stderr, not stdout, so a `--json` payload stays a clean document; and the
/// dashboard redirects fd 2 into itself (see the TUI's console capture), so a
/// write here surfaces as a dashboard event rather than painting the frame.
///
/// Reserved for that category. Ordinary warnings and diagnostics belong in
/// [`console_err`], which respects the sink.
pub fn disclosure_err() -> ConsoleWriter {
    ConsoleWriter::stderr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discarding_writer_accepts_and_drops_bytes() {
        let mut writer = ConsoleWriter::discard();
        assert_eq!(writer.write(b"hello").expect("write succeeds"), 5);
        writer.write_all(b"world").expect("write_all succeeds");
        writer.flush().expect("flush succeeds");
    }

    #[test]
    fn discarding_writer_reports_no_terminal_and_suppression() {
        let writer = ConsoleWriter::discard();
        assert!(writer.is_suppressed());
        assert!(!writer.is_terminal());
    }

    #[test]
    fn stream_writers_are_not_suppressed() {
        assert!(!ConsoleWriter::stdout().is_suppressed());
        assert!(!ConsoleWriter::stderr().is_suppressed());
    }

    #[test]
    fn machine_output_is_never_suppressed() {
        assert!(!machine_out().is_suppressed());
    }

    struct JsonSink;

    impl crate::OutputSink for JsonSink {
        fn emit_event(&self, _event: crate::OutputEvent) -> io::Result<()> {
            Ok(())
        }

        fn mode(&self) -> crate::LogFormat {
            crate::LogFormat::Json
        }
    }

    #[test]
    fn disclosure_survives_a_sink_that_suppresses_console_text() {
        let _sink_lock = crate::OUTPUT_SINK_TEST_LOCK
            .lock()
            .expect("output sink test lock");
        // The property the analytics first-run notice rests on: a sink owning
        // the terminal must not swallow the disclosure, because a run that
        // reports without disclosing is the bug the notice exists to prevent.
        crate::set_output_sink(std::sync::Arc::new(JsonSink));
        assert!(console_err().is_suppressed(), "sink must own console prose");
        assert!(!disclosure_err().is_suppressed());
        crate::clear_output_sink();
    }

    #[test]
    fn writers_pass_through_without_an_installed_sink() {
        let _sink_lock = crate::OUTPUT_SINK_TEST_LOCK
            .lock()
            .expect("output sink test lock");
        // One-shot CLI commands run before any sink exists; console text must
        // still reach the terminal there.
        crate::clear_output_sink();
        assert!(!console_out().is_suppressed());
        assert!(!console_err().is_suppressed());
    }
}
