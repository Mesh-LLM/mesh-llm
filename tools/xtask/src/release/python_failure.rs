//! The uncaught exceptions `release-notes-link.py` can die with. A Python
//! traceback's frames are not reproduced; the report keeps its header, the
//! final `Class: message` line, earlier stderr and the status of 1.

use crate::repository::check_report::CheckReport;
use std::path::Path;

/// An exception the legacy script does not catch.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Uncaught {
    class: &'static str,
    message: String,
}

impl Uncaught {
    pub(crate) fn new(class: &'static str, message: String) -> Self {
        Self { class, message }
    }

    /// `OSError` raised for `path`, with its errno-specific subclass.
    pub(crate) fn os(path: &Path, error: &std::io::Error) -> Self {
        let class = match error.raw_os_error() {
            Some(2) => "FileNotFoundError",
            Some(1 | 13) => "PermissionError",
            Some(17) => "FileExistsError",
            Some(20) => "NotADirectoryError",
            Some(21) => "IsADirectoryError",
            _ => "OSError",
        };
        Self::new(
            class,
            crate::prepared_input::python_io::os_error(path, error),
        )
    }

    pub(crate) fn decode(message: String) -> Self {
        Self::new("UnicodeDecodeError", message)
    }

    /// The failing run: stderr written so far, then the traceback.
    pub(crate) fn report(&self, stderr: String) -> CheckReport {
        CheckReport {
            stdout: String::new(),
            stderr: format!(
                "{stderr}Traceback (most recent call last):\n{}: {}\n",
                self.class, self.message
            ),
            code: 1,
        }
    }
}

/// Python `repr(list_of_str)`, as `subprocess` errors quote their argv.
pub(crate) fn argv_repr(program: &str, args: &[String]) -> String {
    let quoted: Vec<String> = std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .map(crate::repository::python_text::repr)
        .collect();
    format!("[{}]", quoted.join(", "))
}

/// `str(signal.Signals(number))` where the platform defines it.
fn signal_name(number: i32) -> Option<&'static str> {
    let common = match number {
        1 => "SIGHUP",
        2 => "SIGINT",
        3 => "SIGQUIT",
        4 => "SIGILL",
        5 => "SIGTRAP",
        6 => "SIGABRT",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        14 => "SIGALRM",
        15 => "SIGTERM",
        _ => "",
    };
    if !common.is_empty() {
        return Some(common);
    }
    if cfg!(target_os = "macos") {
        match number {
            7 => Some("SIGEMT"),
            10 => Some("SIGBUS"),
            12 => Some("SIGSYS"),
            _ => None,
        }
    } else {
        match number {
            7 => Some("SIGBUS"),
            10 => Some("SIGUSR1"),
            12 => Some("SIGUSR2"),
            _ => None,
        }
    }
}

/// `str(CalledProcessError)` for a nonzero status or a fatal signal.
pub(crate) fn called_process_error(argv: &str, code: Option<i32>, signal: Option<i32>) -> String {
    match (code, signal) {
        (_, Some(number)) => match signal_name(number) {
            Some(name) => format!("Command '{argv}' died with <Signals.{name}: {number}>."),
            None => format!("Command '{argv}' died with unknown signal {number}."),
        },
        (code, None) => format!(
            "Command '{argv}' returned non-zero exit status {}.",
            code.unwrap_or(1)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_release_subprocess_errors_match_python() {
        let argv = argv_repr("gh", &["api".to_owned(), "it's".to_owned()]);
        assert_eq!(argv, "['gh', 'api', \"it's\"]");
        assert_eq!(
            called_process_error("['false']", Some(1), None),
            "Command '['false']' returned non-zero exit status 1."
        );
        assert_eq!(
            called_process_error("['x']", None, Some(9)),
            "Command '['x']' died with <Signals.SIGKILL: 9>."
        );
        assert_eq!(
            called_process_error("['x']", None, Some(64)),
            "Command '['x']' died with unknown signal 64."
        );
    }

    #[test]
    fn migration_release_traceback_keeps_earlier_stderr() {
        let report = Uncaught::new("KeyError", "0".to_owned()).report("warn\n".to_owned());
        assert_eq!(report.code, 1);
        assert_eq!(
            report.stderr,
            "warn\nTraceback (most recent call last):\nKeyError: 0\n"
        );
    }
}
