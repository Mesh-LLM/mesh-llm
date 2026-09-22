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
pub fn print_notice(dir: &Path) {
    let mut handle = disclosure_err();
    let _ = writeln!(handle, "\n{NOTICE}\n");
    let _ = handle.flush();
    mark_shown(dir);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn marker_is_absent_until_the_notice_is_printed() {
        let dir = TempDir::new().expect("tempdir");
        assert!(!was_shown(dir.path()));
        print_notice(dir.path());
        assert!(was_shown(dir.path()));
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
