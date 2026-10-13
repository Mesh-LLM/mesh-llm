use crate::automation::command_interrupt::Interrupt;
use crate::command::DynResult;
use crate::process::{Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value, supervise};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::Path;
use std::time::Duration;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix trajectory-reader --reader PATH --timeout SECONDS --dataset-file PATH --dataset-revision SHA --output PATH --cohort NAME... --framework NAME... --source-dataset NAME... (--sessions-per-cohort N | --trajectories-per-framework N) [--min-isl N] [--max-isl N] [--min-turns N]",
    values: &[
        "--reader",
        "--timeout",
        "--dataset-file",
        "--dataset-revision",
        "--output",
        "--cohort",
        "--framework",
        "--source-dataset",
        "--sessions-per-cohort",
        "--trajectories-per-framework",
        "--min-isl",
        "--max-isl",
        "--min-turns",
    ],
    flags: &["--help"],
};

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
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
    let root = crate::repository::RepositoryRoot::resolve(root)?;
    let reader = Path::new(parsed.last("--reader").ok_or("missing --reader")?);
    if !reader.is_absolute() {
        return GRAMMAR
            .error("--reader must name an absolute executable")
            .emit();
    }
    let seconds = parsed
        .last("--timeout")
        .ok_or("missing --timeout")?
        .parse::<u64>()?;
    if !(1..=86400).contains(&seconds) {
        return GRAMMAR.error("timeout must be in 1..=86400 seconds").emit();
    }
    let mut arguments = vec![Value::Public("cohorts".into())];
    for option in GRAMMAR
        .values
        .iter()
        .copied()
        .filter(|option| !["--reader", "--timeout"].contains(option))
    {
        for value in parsed.all(option) {
            arguments.extend([Value::Public(option.into()), Value::Public(value.into())]);
        }
    }
    let spec = ProcessSpec {
        executable: super::executable_resolution::executable(reader)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?,
        arguments,
        cwd: root.as_path().to_path_buf(),
        environment: ["SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|name| {
                std::env::var_os(name)
                    .filter(|value| !value.is_empty())
                    .map(|value| (name.into(), Value::Public(value)))
            })
            .collect(),
    };
    let interrupt = Interrupt::install()?;
    let result = supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(seconds),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &interrupt.cancellation(),
        OutputFiles::default(),
    );
    interrupt.finish()?;
    let report = result?;
    if report.success() {
        Ok(())
    } else {
        Err(format!(
            "trajectory reader failed: {:?}; stderr={}",
            report.outcome,
            String::from_utf8_lossy(&report.stderr.bytes_retained)
        )
        .into())
    }
}
