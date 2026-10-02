#[path = "build/contract.rs"]
mod contract;
#[path = "build/environment.rs"]
mod environment;
#[path = "build/logs.rs"]
mod logs;
#[path = "build/package.rs"]
mod package;
#[cfg(test)]
#[path = "build/tests.rs"]
mod tests;

use crate::automation::{canary_source_plan, command_interrupt::Interrupt};
use crate::command::DynResult;
use crate::process::{Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value};
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use contract::Input;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation canary-receipts build --input PATH",
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
    let mut input: Input =
        serde_json::from_slice(&fs::read(parsed.last("--input").ok_or("missing --input")?)?)?;
    if let Err(error) = input.validate() {
        failure("infrastructure", "build-contract")?;
        return Err(error);
    }
    execute(&input)
}

fn execute(input: &Input) -> DynResult<()> {
    let previous = match package::previous(input) {
        Ok(previous) => previous,
        Err(error) => {
            failure("infrastructure", "previous-package")?;
            return Err(error);
        }
    };
    let wrapper = input
        .controller_root
        .join("scripts/llama-canary-agent-repair.sh")
        .canonicalize()?;
    if !wrapper.starts_with(&input.controller_root) || !wrapper.is_file() {
        return Err("repair wrapper escapes frozen controller checkout".into());
    }
    fs::create_dir(&input.evidence)?;
    admit_selected_contract(input)?;
    let previous_identity = previous.as_ref().map(|(_, identity)| identity);
    let spec = ProcessSpec {
        executable: PathBuf::from("/bin/bash"),
        arguments: vec![Value::Public(wrapper.into())],
        cwd: input.controller_root.clone(),
        environment: environment::wrapper(
            input,
            previous
                .as_ref()
                .map(|(path, identity)| (path.as_path(), identity)),
        ),
    };
    let interrupt = Interrupt::install()?;
    let report = logs::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(
                input.agent_timeout_seconds + input.verification_timeout_seconds + 600,
            ),
            graceful_shutdown: Duration::from_secs(20),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: 64 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &interrupt.cancellation(),
        OutputFiles {
            stdout: Some(input.evidence.join("wrapper.stdout.log")),
            stderr: Some(input.evidence.join("wrapper.stderr.log")),
        },
    );
    let interrupted = interrupt.finish();
    match report {
        Ok(report) if report.success() && interrupted.is_ok() => {
            if let Err(error) = package::exported(input, previous_identity) {
                failure("infrastructure", "artifact-export")?;
                return Err(error);
            }
            Ok(())
        }
        Ok(report) => {
            if interrupted.is_err()
                || !report.cleanup.complete
                || report.cleanup.failure.is_some()
                || report.failure.is_some()
                || !matches!(report.outcome, crate::process::Outcome::Exited)
            {
                failure("infrastructure", "process-supervision")?;
            }
            Err(format!(
                "canary build wrapper failed: {:?}, status {:?}",
                report.outcome, report.status
            )
            .into())
        }
        Err(error) => {
            failure("infrastructure", "process-supervision")?;
            Err(error.into())
        }
    }
}

fn admit_selected_contract(input: &Input) -> DynResult<()> {
    let request = serde_json::json!({
        "controller_root":input.controller_root,
        "source_root":input.source_root,
        "controller_revision":input.controller_revision,
        "selected_revision":input.selected_revision,
        "manifest":"ci/llama-canary/family-certified.json",
        "output":input.evidence.join("source-plan"),
        "cache":{"mode":"not_checked"}
    });
    if let Err(error) = canary_source_plan::generate_document(&serde_json::to_vec(&request)?) {
        failure("infrastructure", "selected-plan-contract")?;
        return Err(error);
    }
    Ok(())
}

fn failure(class: &str, stage: &str) -> DynResult<()> {
    if let Some(path) = std::env::var_os("GITHUB_OUTPUT") {
        let mut output = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(Path::new(&path))?;
        writeln!(output, "failure_class={class}\nfailure_stage={stage}")?;
    }
    Ok(())
}

#[cfg(all(test, unix))]
#[path = "build/wrapper_main_tests.rs"]
mod wrapper_main_tests;
