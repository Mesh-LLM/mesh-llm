use crate::repository::check_report::CheckReport;
use std::path::Path;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub(crate) struct Uncaught {
    message: String,
}

impl Uncaught {
    pub(crate) fn new(_class: &'static str, message: String) -> Self {
        Self { message }
    }

    pub(crate) fn os(path: &Path, error: &std::io::Error) -> Self {
        Self {
            message: format!("{}: {error}", path.display()),
        }
    }

    pub(crate) fn decode(message: String) -> Self {
        Self { message }
    }

    pub(crate) fn report(&self, stderr: String) -> CheckReport {
        CheckReport {
            stdout: String::new(),
            stderr: format!("{stderr}{self}\n"),
            code: 1,
        }
    }
}

pub(crate) fn argv_repr(program: &str, args: &[String]) -> String {
    format!("{program:?} {args:?}")
}

pub(crate) fn called_process_error(argv: &str, code: Option<i32>, signal: Option<i32>) -> String {
    match signal {
        Some(signal) => format!("command {argv} terminated by signal {signal}"),
        None => format!("command {argv} exited with status {code:?}"),
    }
}
