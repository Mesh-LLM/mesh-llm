//! `Gh` of `scripts/release-notes-link.py`: a best-effort `gh` caller that
//! stops after its budget or its first timeout and reports each failure on
//! stderr with native process and JSON diagnostics.

use crate::ci_operations::ci_metrics_value::{Value, parse};
use crate::prepared_input::text_io::os_error;
use crate::release::command_failure::{argv_repr, called_process_error};
use crate::release::link_host::{Exit, ReleaseHost, text_streams};
use std::time::Duration;

/// `Gh(timeout=30)`: the per-call timeout in seconds.
const TIMEOUT_SECONDS: u64 = 30;

pub(crate) struct Gh<'h> {
    host: &'h mut dyn ReleaseHost,
    remaining: i64,
    pub(crate) exhausted: bool,
    pub(crate) failures: u64,
    /// Everything the legacy script printed to stderr so far.
    pub(crate) stderr: String,
}

impl<'h> Gh<'h> {
    pub(crate) fn new(host: &'h mut dyn ReleaseHost, budget: i64) -> Self {
        Self {
            host,
            remaining: budget,
            exhausted: false,
            failures: 0,
            stderr: String::new(),
        }
    }

    /// Best-effort JSON response, with failures recorded for the caller.
    pub(crate) fn json(&mut self, args: &[String]) -> Option<Value> {
        if self.remaining <= 0 {
            if !self.exhausted {
                self.stderr.push_str(
                    "release-notes-link: API budget exhausted; \
                     remaining commits keep their published entries\n",
                );
            }
            self.exhausted = true;
            return None;
        }
        self.remaining -= 1;
        match self.call(args) {
            Ok(value) => Some(value),
            Err(message) => {
                self.failures += 1;
                self.stderr.push_str(&format!(
                    "release-notes-link: gh {} failed: {message}\n",
                    args.join(" ")
                ));
                None
            }
        }
    }

    fn call(&mut self, args: &[String]) -> Result<Value, String> {
        let argv = argv_repr("gh", args);
        let output = match self.host.gh(args, Duration::from_secs(TIMEOUT_SECONDS)) {
            Ok(output) => output,
            Err(failure) => return Err(os_error(&failure.filename, &failure.error)),
        };
        if output.exit == Exit::TimedOut {
            self.remaining = 0;
            return Err(format!(
                "command {argv} timed out after {TIMEOUT_SECONDS} seconds"
            ));
        }
        let stdout = match text_streams(&output) {
            Ok((stdout, _)) => stdout,
            Err(message) => return Err(message),
        };
        match output.exit {
            Exit::Code(0) | Exit::TimedOut => {}
            Exit::Code(code) => return Err(called_process_error(&argv, Some(code), None)),
            Exit::Signal(signal) => {
                return Err(called_process_error(&argv, None, Some(signal)));
            }
        }
        parse(stdout.as_bytes())
    }
}
