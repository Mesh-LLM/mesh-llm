//! `Gh` of `scripts/release-notes-link.py`: a best-effort `gh` caller that
//! stops after its budget or its first timeout and reports each failure on
//! stderr with Python's exception text.

use crate::ci_operations::ci_metrics_value::{Value, parse};
use crate::prepared_input::python_io::os_error;
use crate::release::link_host::{Exit, ReleaseHost, text_streams};
use crate::release::python_failure::{Uncaught, argv_repr, called_process_error};
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

    /// `Gh.json(args)`: parsed JSON, or `None` if it did not work out. An
    /// exception the legacy method does not catch is returned as an error.
    pub(crate) fn json(&mut self, args: &[String]) -> Result<Option<Value>, Uncaught> {
        if self.remaining <= 0 {
            if !self.exhausted {
                self.stderr.push_str(
                    "release-notes-link: API budget exhausted; \
                     remaining commits keep their published entries\n",
                );
            }
            self.exhausted = true;
            return Ok(None);
        }
        self.remaining -= 1;
        match self.call(args)? {
            Ok(value) => Ok(Some(value)),
            Err(message) => {
                self.failures += 1;
                self.stderr.push_str(&format!(
                    "release-notes-link: gh {} failed: {message}\n",
                    args.join(" ")
                ));
                Ok(None)
            }
        }
    }

    /// The outer error escapes `Gh.json`; the inner one is caught there.
    fn call(&mut self, args: &[String]) -> Result<Result<Value, String>, Uncaught> {
        let argv = argv_repr("gh", args);
        let output = match self.host.gh(args, Duration::from_secs(TIMEOUT_SECONDS)) {
            Ok(output) => output,
            Err(failure) => return Ok(Err(os_error(&failure.filename, &failure.error))),
        };
        if output.exit == Exit::TimedOut {
            self.remaining = 0;
            return Ok(Err(format!(
                "Command '{argv}' timed out after {TIMEOUT_SECONDS} seconds"
            )));
        }
        let stdout = match text_streams(&output) {
            Ok((stdout, _)) => stdout,
            Err(message) => return Ok(Err(message)),
        };
        match output.exit {
            Exit::Code(0) | Exit::TimedOut => {}
            Exit::Code(code) => return Ok(Err(called_process_error(&argv, Some(code), None))),
            Exit::Signal(signal) => {
                return Ok(Err(called_process_error(&argv, None, Some(signal))));
            }
        }
        match parse(stdout.as_bytes()) {
            Ok(value) => Ok(Ok(value)),
            Err(message) if message.starts_with("maximum recursion depth") => {
                Err(Uncaught::new("RecursionError", message))
            }
            Err(message) => Ok(Err(message)),
        }
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Int(int) => *int != 0,
        Value::BigInt(_) => true,
        Value::Float(float) => *float != 0.0,
        Value::Str(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

/// Python's type name for `TypeError`/`AttributeError` text.
pub(crate) fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::BigInt(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}
