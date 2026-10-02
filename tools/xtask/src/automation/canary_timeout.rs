//! Canary repair command deadlines, with live inherited streams and tree ownership.
use crate::command::DynResult;
use crate::command_interrupt::Interrupt;
use crate::process::{
    Completion, InheritedReport, Limits, Outcome, ProcessSpec, Readiness, Value,
    supervise_inherited,
};
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Deserialize;
use std::{fs, path::PathBuf, time::Duration};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    label: String,
    seconds: u64,
    cwd: PathBuf,
    executable: PathBuf,
    arguments: Vec<String>,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation canary-timeout --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input =
        serde_json::from_slice(&fs::read(parsed.last("--input").ok_or("missing --input")?)?)?;
    execute(&input).emit()
}

fn execute(input: &Input) -> CheckReport {
    match supervise(input) {
        Ok(report) => report,
        Err(error) => CheckReport {
            stdout: String::new(),
            stderr: format!(
                "{} infrastructure supervision failed: {error}\n",
                input.label
            ),
            code: 125,
        },
    }
}

fn supervise(input: &Input) -> DynResult<CheckReport> {
    if input.seconds == 0
        || input.seconds > 86400
        || input.label.is_empty()
        || input.label.contains(['\n', '\r'])
    {
        return Err("invalid canary command label or deadline".into());
    }
    let spec = ProcessSpec {
        executable: input.executable.clone(),
        cwd: input.cwd.clone(),
        arguments: input
            .arguments
            .iter()
            .map(|arg| Value::Secret(arg.into()))
            .collect(),
        environment: std::env::vars_os()
            .map(|(key, value)| (key, Value::Secret(value)))
            .collect(),
    };
    let limits = Limits {
        execution: Duration::from_secs(input.seconds),
        graceful_shutdown: Duration::from_secs(10),
        forced_shutdown: Duration::from_secs(10),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let interrupt = Interrupt::install()?;
    let result = supervise_inherited(&spec, &limits, &interrupt.cancellation());
    // Restore handlers only after owned-tree cleanup, then freeze first signal.
    #[cfg(unix)]
    let signal = interrupt.finish_signal()?;
    #[cfg(windows)]
    let signal = match interrupt.finish() {
        Ok(()) => None,
        Err(crate::command_interrupt::Reason::Interrupted) => Some(2),
        Err(error) => return Err(error.into()),
    };
    let report = match result {
        Ok(report) => report,
        Err(_) if signal.is_some() => {
            return Ok(CheckReport {
                stdout: String::new(),
                stderr: format!("{} interrupted before command spawn\n", input.label),
                code: 128 + signal.unwrap_or(0),
            });
        }
        Err(error) => return Err(error.into()),
    };
    let code = status(&report, signal);
    let stderr = match code {
        125 => format!(
            "{} infrastructure cleanup failed for pid {} after {:?}: {:?}; {:?}\n",
            input.label, report.pid, report.elapsed, report.failure, report.cleanup.failure
        ),
        124 => format!(
            "{} timed out after {}s; terminating process group\n",
            input.label, input.seconds
        ),
        _ if signal.is_some() => format!(
            "{} received signal {}; terminating process group\n",
            input.label,
            signal.unwrap_or(0)
        ),
        _ => String::new(),
    };
    Ok(CheckReport {
        stdout: String::new(),
        stderr,
        code,
    })
}

fn status(report: &InheritedReport, signal: Option<i32>) -> i32 {
    if !report.cleanup.complete || report.failure.is_some() || report.cleanup.failure.is_some() {
        return 125;
    }
    if let Some(signal) = signal {
        return 128 + signal;
    }
    match report.outcome {
        Outcome::Deadline => 124,
        Outcome::Exited => report.status.map_or(125, child_status),
        _ => 125,
    }
}

fn child_status(status: std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status
            .code()
            .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
    }
    #[cfg(windows)]
    {
        status.code().unwrap_or(125)
    }
}

#[cfg(all(test, unix))]
#[path = "canary_timeout_tests.rs"]
mod tests;
