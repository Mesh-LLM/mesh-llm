use super::run_workload::Input;
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use std::path::Path;

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix execute-run --input PATH [--engine-config PATH]",
        values: &["--input", "--engine-config"],
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
    let mut input: Input = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    if let Some(path) = parsed.last("--engine-config") {
        input.engine_config = Some(std::path::absolute(path)?);
    }
    execute(root, input)
}

fn execute(root: Option<&Path>, input: Input) -> DynResult<()> {
    let mut prepared = super::run_setup::prepare(root, input)?;
    let completed = if prepared.input.resume {
        super::run_resume::restore(
            &mut prepared.document,
            (&prepared.run_path, &prepared.manifest),
            prepared.input.passes,
        )?
    } else {
        std::collections::BTreeSet::new()
    };
    super::run_arms::record(&prepared.input, &prepared.document)?;
    super::run_snapshot::write(&prepared.run_path, &prepared.document)?;
    let qualification = super::run_qualification::qualify(
        root,
        &prepared.input,
        &mut prepared.document,
        &prepared.manifest,
        &prepared.run_path,
        &prepared.budget,
    )?;
    let passed = super::run_measurement::measure(
        root,
        &prepared.input,
        &mut prepared.document,
        (&prepared.manifest, &prepared.run_path),
        &completed,
        &qualification,
        &prepared.budget,
    )?;
    super::run_completion::complete(
        root,
        &prepared.input,
        &mut prepared.document,
        &prepared.run_path,
        passed,
    )
}
