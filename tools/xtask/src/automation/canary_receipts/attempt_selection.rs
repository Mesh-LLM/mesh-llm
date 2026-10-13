//! Select bounded distributed canary attempts from protected workflow outputs.
//! These projections do not verify producer packages or feedback on disk.
use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::fmt::Write as _;
use std::{collections::BTreeMap, fs::OpenOptions, io::Write};

#[path = "attempt_selection/decision.rs"]
mod decision;
#[path = "attempt_selection/input.rs"]
mod input;
use decision::{AttemptDecision, FinalDecision};
use input::{Job, JobOutputs, JobResult};

const INPUT_LIMIT: usize = 1024 * 1024;
const LINE_LIMIT: usize = 4096;
const ATTEMPT: Grammar = Grammar {
    usage: "cargo xtool automation canary-receipts select-attempt --changed <true|false> --repair-json <job> --verification-json <job>",
    values: &["--changed", "--repair-json", "--verification-json"],
    flags: &["--help"],
};
const FINAL: Grammar = Grammar {
    usage: "cargo xtool automation canary-receipts select-final --certify <true|false> --changed <true|false> --mesh-source <sha-or-empty> --preflight <result> --attempts-json <needs>",
    values: &[
        "--certify",
        "--changed",
        "--mesh-source",
        "--preflight",
        "--attempts-json",
    ],
    flags: &["--help"],
};

pub(super) fn run_attempt(args: &[String]) -> DynResult<()> {
    run(args, &ATTEMPT, |args| {
        let changed = boolean(required(args, "--changed")?)?;
        let repair: Job = decode(required(args, "--repair-json")?)?;
        let verification: Job = decode(required(args, "--verification-json")?)?;
        repair.validate()?;
        verification.validate()?;
        Ok(select_attempt(changed, &repair, &verification)?.outputs())
    })
}

pub(super) fn run_final(args: &[String]) -> DynResult<()> {
    run(args, &FINAL, |args| {
        let certify = boolean(required(args, "--certify")?)?;
        let changed = boolean(required(args, "--changed")?)?;
        let source = required(args, "--mesh-source")?;
        if !source.is_empty() {
            input::head(source)?;
        }
        let preflight = JobResult::parse(required(args, "--preflight")?)?;
        let attempts = decode(required(args, "--attempts-json")?)?;
        Ok(select_final(certify, changed, source, preflight, &attempts)?.outputs())
    })
}

fn run(
    args: &[String],
    grammar: &Grammar,
    decide: impl FnOnce(&ParsedArgs) -> DynResult<BTreeMap<&'static str, String>>,
) -> DynResult<()> {
    let parsed = match grammar.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("usage: {}\n", grammar.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return Err("canary selection accepts only named inputs".into());
    }
    for name in grammar.values {
        if parsed.all(name).len() != 1 {
            return Err(format!("canary selection requires exactly one {name}").into());
        }
    }
    let outputs = decide(&parsed)?;
    emit(&outputs)
}

fn required<'a>(args: &'a ParsedArgs, name: &str) -> DynResult<&'a str> {
    args.last(name)
        .ok_or_else(|| format!("missing canary selection input {name}").into())
}
fn boolean(value: &str) -> DynResult<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err("canary selection boolean must be true or false".into()),
    }
}
fn decode<T: serde::de::DeserializeOwned>(value: &str) -> DynResult<T> {
    if value.len() > INPUT_LIMIT {
        return Err("canary selection JSON exceeds 1 MiB".into());
    }
    Ok(serde_json::from_str(value)?)
}
fn safe_line(value: &str, name: &str) -> DynResult<String> {
    if value.trim().is_empty() || value.len() > LINE_LIMIT || value.chars().any(char::is_control) {
        return Err(format!("canary selection {name} must be a nonempty safe single line").into());
    }
    Ok(value.to_owned())
}

fn select_attempt(changed: bool, repair: &Job, verification: &Job) -> DynResult<AttemptDecision> {
    // Reusable job failure may precede a successful infrastructure recheck.
    // The protected aggregate/reconcile outputs own its normalized result.
    if repair.outputs.green != "true" {
        return decision::failed_or_resume(
            changed,
            &repair.outputs,
            "candidate-build-or-family-certification",
        );
    }
    let candidate = decision::Candidate::from_outputs(&repair.outputs)?;
    if !changed {
        return Ok(AttemptDecision::Green(candidate));
    }
    if verification.outputs.green == "true" {
        let verified = decision::Candidate::from_outputs(&verification.outputs)?;
        return Ok(if candidate.head == verified.head {
            AttemptDecision::Green(verified)
        } else {
            decision::identity_failure()
        });
    }
    let result =
        decision::failed_or_resume(true, &verification.outputs, "independent-verification")?;
    if matches!(&result, AttemptDecision::Repairable(resume) if resume.head != candidate.head) {
        return Ok(decision::identity_failure());
    }
    Ok(result)
}

fn select_final(
    certify: bool,
    changed: bool,
    mesh_source: &str,
    preflight: JobResult,
    attempts: &input::Attempts,
) -> DynResult<FinalDecision> {
    attempts.validate_slots()?;
    if !certify {
        return Ok(FinalDecision::Noop);
    }
    if preflight != JobResult::Success {
        return Err(
            "canary environment preflight failed; candidate source was not evaluated".into(),
        );
    }
    let selected = attempts.latest()?;
    if selected.result != JobResult::Success || selected.outputs.state != "green" {
        let failure =
            decision::Failure::from_outputs(&selected.outputs, "candidate", "distributed-repair")?;
        return Err(format!(
            "{} failure during {}; publication denied",
            failure.class, failure.stage
        )
        .into());
    }
    if selected.outputs.green != "true" || selected.outputs.repairable == "true" {
        return Err("green attempt has contradictory selection outputs".into());
    }
    let candidate = decision::Candidate::from_outputs(&selected.outputs)?;
    if !mesh_source.is_empty() {
        if changed || candidate.head != mesh_source {
            return Err("selected MeshLLM revision certification failed".into());
        }
        return Ok(FinalDecision::Certified);
    }
    Ok(if changed {
        FinalDecision::Publish(candidate)
    } else {
        FinalDecision::Certified
    })
}

fn emit(outputs: &BTreeMap<&str, String>) -> DynResult<()> {
    // Admit the entire decision before touching either output destination.
    let mut rendered = String::new();
    for (key, value) in outputs {
        safe_line(value, key)?;
        if *key != "publish" {
            writeln!(rendered, "{key}={value}")?;
        }
    }
    if let Some(value) = outputs.get("publish") {
        writeln!(rendered, "publish={value}")?;
    }
    let document = serde_json::to_vec(outputs)?;
    if let Some(path) = std::env::var_os("GITHUB_OUTPUT").filter(|path| !path.is_empty()) {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(rendered.as_bytes())?;
        file.flush()?;
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&document)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

#[cfg(test)]
#[path = "attempt_selection/tests.rs"]
mod tests;
