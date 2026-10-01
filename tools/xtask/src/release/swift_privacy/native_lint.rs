use super::{adapter_args::Options, adapter_error::Error};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, io::Write, path::Path};

#[derive(Debug, thiserror::Error)]
pub(super) enum NativeFailure {
    #[error(transparent)]
    Setup(#[from] process::Failure),
    #[error(
        "native privacy lint failed: outcome={outcome:?}, exit={exit:?}, cleanup={cleanup:?}, failure={failure:?}, complete_raw_stderr={complete_raw_stderr}"
    )]
    Child {
        outcome: process::Outcome,
        exit: Option<i32>,
        cleanup: process::Cleanup,
        failure: Option<process::Failure>,
        complete_raw_stderr: bool,
    },
    #[error("native privacy lint did not provide complete stdout and stderr EOF")]
    Incomplete,
}

pub(super) struct NativeLint<'a> {
    pub(super) options: &'a Options<'a>,
    pub(super) cwd: &'a Path,
    pub(super) cancellation: &'a Cancellation,
}

type LintResult = Result<process::RawBytes, (NativeFailure, Option<process::RawBytes>)>;

impl NativeLint<'_> {
    pub(super) fn lint(&self, path: &Path, stderr: &mut impl Write) -> Result<(), Error> {
        let mut environment = BTreeMap::new();
        for key in ["SYSTEMROOT", "WINDIR"] {
            if let Some(value) = std::env::var_os(key) {
                environment.insert(key.into(), Value::Public(value));
            }
        }
        let spec = ProcessSpec {
            executable: self.options.plutil.to_path_buf(),
            arguments: vec![Value::Public("-lint".into()), Value::Public(path.into())],
            cwd: self.cwd.to_path_buf(),
            environment,
        };
        let budget = &self.options.budget;
        let limits = Limits {
            execution: budget.execution,
            graceful_shutdown: budget.grace,
            forced_shutdown: budget.cleanup,
            retained_bytes_per_stream: budget.output.get(),
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            self.cancellation,
            RawCaptureOptions {
                stdout: Some(budget.output),
                stderr: Some(budget.output),
            },
        )
        .map_err(NativeFailure::from)?;
        deliver(classify(report), stderr)
    }
}

pub(super) fn classify(report: process::RawProcessReport) -> LintResult {
    let diagnostic = report.stderr;
    if !report.process.success() {
        return Err((
            NativeFailure::Child {
                outcome: report.process.outcome,
                exit: report.process.status.and_then(|status| status.code()),
                cleanup: report.process.cleanup,
                failure: report.process.failure,
                complete_raw_stderr: diagnostic.is_some(),
            },
            diagnostic,
        ));
    }
    match (report.stdout, diagnostic) {
        (Some(_), Some(diagnostic)) => Ok(diagnostic),
        (None, diagnostic) | (Some(_), diagnostic @ None) => {
            Err((NativeFailure::Incomplete, diagnostic))
        }
    }
}

fn deliver(classified: LintResult, stderr: &mut impl Write) -> Result<(), Error> {
    let (primary, diagnostic) = match classified {
        Ok(diagnostic) => (Ok(()), Some(diagnostic)),
        Err((error, diagnostic)) => (Err(error), diagnostic),
    };
    let delivery = match diagnostic {
        Some(bytes) => stderr
            .write_all(bytes.as_bytes())
            .and_then(|()| stderr.flush()),
        None => Ok(()),
    };
    match (primary, delivery) {
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(output)) => Err(Error::Output(output)),
        (Err(primary), Ok(())) => Err(Error::Native(primary)),
        (Err(primary), Err(output)) => Err(Error::Diagnostic { primary, output }),
    }
}
