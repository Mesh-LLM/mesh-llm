use super::session_evidence::{Request, Runtime, Trajectory, complete};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    trajectories: Vec<Trajectory>,
    requests: Vec<Request>,
    runtime: Runtime,
    required_context: u64,
    recurrent: Option<Recurrent>,
    eligibility: Option<super::context_eligibility::Budget>,
}

#[derive(Deserialize)]
struct Recurrent {
    minimum_restored_tokens: u64,
    #[serde(default)]
    lookups: Vec<super::recurrent_evidence::Lookup>,
    #[serde(default)]
    log_paths: Vec<std::path::PathBuf>,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix session-evidence --input PATH --output PATH",
        values: &["--input", "--output"],
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
    let input = parsed.last("--input").ok_or("missing --input")?;
    let output = parsed.last("--output").ok_or("missing --output")?;
    let mut input: Input = serde_json::from_slice(&std::fs::read(input)?)?;
    if let Some(recurrent) = &mut input.recurrent {
        recurrent
            .lookups
            .extend(super::recurrent_evidence::read_logs(&recurrent.log_paths)?);
    }
    let mut completeness = complete(&input.trajectories, &input.requests);
    if let Err(problem) = input.runtime.context(input.required_context) {
        completeness.passed = false;
        completeness.problems.push(problem.into());
    }
    let recurrent = input.recurrent.as_ref().map(|recurrent| {
        super::recurrent_evidence::evaluate(
            &input.requests,
            &recurrent.lookups,
            recurrent.minimum_restored_tokens,
        )
    });
    if let Some(recurrent) = &recurrent {
        completeness.passed &= recurrent.passed;
        completeness
            .problems
            .extend(recurrent.problems.iter().cloned());
    }
    let mut report = serde_json::to_value(&completeness)?;
    if let Some(budget) = &input.eligibility {
        let eligibility =
            super::context_eligibility::evaluate(&input.trajectories, &input.requests, budget);
        completeness.passed &= eligibility.passed;
        completeness
            .problems
            .extend(eligibility.problems.iter().cloned());
        report["eligibility"] = serde_json::to_value(eligibility)?;
        report["passed"] = completeness.passed.into();
        report["problems"] = serde_json::to_value(&completeness.problems)?;
    }
    if let Some(recurrent) = recurrent {
        report["recurrent_state"] = serde_json::to_value(recurrent)?;
    }
    crate::command::write_json_file(std::path::Path::new(output), &report)?;
    if completeness.passed {
        Ok(())
    } else {
        Err("replay session evidence failed".into())
    }
}
