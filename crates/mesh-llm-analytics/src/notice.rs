//! The first-run disclosure.
//!
//! mesh-llm reports anonymous usage by default, so the first run has to say
//! so plainly, in the same breath as how to turn it off. The notice goes to
//! stderr so it never contaminates machine-readable stdout.

use std::io::{self, IsTerminal, Write};

/// The disclosure text shown once per install.
pub const NOTICE: &str = "\
mesh-llm reports anonymous usage data (version, platform, which commands run,
how many nodes and models are in play) so we can see what to improve. It never
includes prompts, completions, model contents, file paths, IP addresses, or
anything about your mesh peers.

  Turn it off:  mesh-llm analytics disable
  What we send: https://meshllm.cloud/docs/pages/analytics/

This notice is shown once.";

/// Print the notice to stderr, unless stderr is redirected.
///
/// A redirected stderr usually means a log file or a pipe, where a one-time
/// human-facing notice is noise rather than disclosure. The documented
/// opt-out remains available either way.
pub fn print_notice() {
    let stderr = io::stderr();
    if !stderr.is_terminal() {
        return;
    }
    let mut handle = stderr.lock();
    let _ = writeln!(handle, "\n{NOTICE}\n");
    let _ = handle.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

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
