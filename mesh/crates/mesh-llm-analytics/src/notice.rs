//! The first-run disclosure.
//!
//! mesh-llm reports anonymous usage by default, so the first run has to say
//! so plainly, in the same breath as how to turn it off. The notice goes to
//! stderr so it never contaminates machine-readable stdout.

use std::fs;
use std::io::Write;
use std::path::Path;

use mesh_llm_events::disclosure_err;

/// Marker recording that the disclosure has been shown.
///
/// Deliberately separate from the install identifier. Tying disclosure to
/// identifier creation meant a first run under `DO_NOT_TRACK`, or a first
/// `analytics status`, consumed the first-run signal and the notice was then
/// never shown on any later run.
pub const NOTICE_MARKER_FILE: &str = "analytics-notice-shown";

/// The disclosure text shown once per install.
pub const NOTICE: &str = "\
mesh-llm reports anonymous usage data (version, platform, which commands run,
how many nodes and models are in play) so we can see what to improve. It never
includes prompts, completions, model contents, file paths, IP addresses, or
anything about your mesh peers.

  Turn it off:  mesh-llm analytics disable
  What we send: https://meshllm.cloud/docs/pages/analytics/

This notice is shown once.";

/// Whether the disclosure has already been shown for this install.
#[must_use]
pub fn was_shown(dir: &Path) -> bool {
    dir.join(NOTICE_MARKER_FILE).exists()
}

/// Record that the disclosure has been shown.
fn mark_shown(dir: &Path) {
    let _ = fs::create_dir_all(dir);
    let _ = fs::write(dir.join(NOTICE_MARKER_FILE), b"1\n");
}

/// Print the notice to stderr and record that it was shown.
///
/// Emitted whether or not stderr is a terminal, and whether or not a JSON sink
/// or the dashboard owns it. A notice in a service log is still disclosure; a
/// notice suppressed because the process happened to be daemonized is not, and
/// on-by-default reporting is only defensible if the disclosure actually
/// happens.
///
/// That is why this uses [`disclosure_err`] rather than `console_err`: the
/// console writers discard while another surface owns the terminal, which for
/// a once-per-install notice would mean never showing it at all. Routing it
/// through the output facility still keeps the terminal handle where it
/// belongs — this module does not hold one.
///
/// Returns whether the notice reached stderr, and records it only when it did.
/// A write that fails — closed or full stderr, a pipe with no reader — must not
/// spend the once-only marker, because the marker is the record that disclosure
/// *happened*. Callers fail closed on `false` instead of reporting without
/// having disclosed.
#[must_use]
pub fn print_notice(dir: &Path) -> bool {
    let mut handle = disclosure_err();
    show_notice(&mut handle, dir)
}

/// Write the notice to `sink` and, only if the bytes went out, record it.
///
/// Separate from [`print_notice`] so the failure path can be tested: the handle
/// it really uses is stderr, which a test cannot be made to fail.
fn show_notice(sink: &mut dyn Write, dir: &Path) -> bool {
    if writeln!(sink, "\n{NOTICE}\n").is_err() || sink.flush().is_err() {
        return false;
    }
    mark_shown(dir);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use tempfile::TempDir;

    /// A sink that always fails, standing in for a closed or full stderr.
    struct FailingSink;

    impl Write for FailingSink {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "no reader"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "no reader"))
        }
    }

    #[test]
    fn marker_is_absent_until_the_notice_is_printed() {
        let dir = TempDir::new().expect("tempdir");
        assert!(!was_shown(dir.path()));
        assert!(print_notice(dir.path()), "stderr accepted the notice");
        assert!(was_shown(dir.path()));
    }

    #[test]
    fn an_undelivered_notice_is_not_recorded_as_shown() {
        // The bug this guards: recording the marker before the bytes reached
        // stderr spends the once-only disclosure on a notice nobody saw, so
        // every later run stays silent while reporting continues.
        let dir = TempDir::new().expect("tempdir");
        let mut sink = FailingSink;

        assert!(!show_notice(&mut sink, dir.path()));
        assert!(
            !was_shown(dir.path()),
            "a notice that was not delivered must stay pending"
        );
    }

    #[test]
    fn a_delivered_notice_is_recorded_and_states_the_opt_out() {
        let dir = TempDir::new().expect("tempdir");
        let mut sink = Vec::new();

        assert!(show_notice(&mut sink, dir.path()));
        assert!(was_shown(dir.path()));

        let text = String::from_utf8(sink).expect("notice is utf-8");
        assert!(
            text.contains("mesh-llm analytics disable"),
            "delivered notice omits the opt-out: {text}"
        );
    }

    #[test]
    fn marker_is_independent_of_the_install_identifier() {
        // The bug this guards: creating the install id used to consume the
        // first-run signal, so a first run under DO_NOT_TRACK meant the
        // notice was never shown on any later run.
        let dir = TempDir::new().expect("tempdir");
        crate::load_or_create(dir.path()).expect("create id");
        assert!(
            !was_shown(dir.path()),
            "install id must not mark disclosure"
        );
    }

    #[test]
    fn notice_states_the_opt_out_and_the_exclusions() {
        assert!(NOTICE.contains("mesh-llm analytics disable"));
        assert!(NOTICE.contains("anonymous"));
        for excluded in ["prompts", "completions", "file paths", "IP addresses"] {
            assert!(NOTICE.contains(excluded), "notice omits {excluded}");
        }
    }

    #[test]
    fn notice_has_no_trailing_whitespace() {
        for line in NOTICE.lines() {
            assert_eq!(line, line.trim_end(), "trailing whitespace in notice");
        }
    }
}
