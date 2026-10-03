use super::canary_receipts::{
    Digest, PackageVerification, ReceiptContext, aggregate, verify_package,
};
use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::{fs::OpenOptions, io::Write, path::Path};
#[path = "result_gate.rs"]
mod result_gate;

pub(crate) const USAGE: &str = "cargo xtool automation canary-receipts aggregate --package <path> --identity <sha256> --evidence <path> --run-id <id> --run-attempt <attempt> [--controller-revision <sha>] [--selected-source <sha>]";

const GRAMMAR: Grammar = Grammar {
    usage: USAGE,
    values: &[
        "--package",
        "--identity",
        "--evidence",
        "--run-id",
        "--run-attempt",
        "--controller-revision",
        "--selected-source",
    ],
    flags: &["--help"],
};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [verb, rest @ ..] if verb == "verification-source-admit" => {
            return super::canary_package_closure::verification_source::run(rest);
        }
        [verb, rest @ ..]
            if matches!(
                verb.as_str(),
                "verification-manifest-policy"
                    | "verification-parity-inventory"
                    | "verification-split-roster-check"
            ) =>
        {
            return super::canary_package_closure::verification_source::run_inspection(rest, verb);
        }
        [verb, rest @ ..] if verb == "result" => return result_gate::run(rest),
        [verb, rest @ ..] if verb == "receipt" || verb == "publication" => {
            return super::canary_handoff::run(rest, verb == "publication");
        }
        [verb, rest @ ..]
            if matches!(
                verb.as_str(),
                "producer-receipt"
                    | "candidate-plan"
                    | "pack"
                    | "restore"
                    | "certify"
                    | "split-roster"
                    | "manifest-policy"
                    | "parity-inventory"
                    | "local-manifest-policy"
                    | "local-parity-inventory"
                    | "local-split-roster"
            ) =>
        {
            return super::canary_package_closure::transaction(rest, verb);
        }
        [verb, rest @ ..] if verb == "workload-manifest" => {
            return super::canary_package_closure::run(rest, true);
        }
        [verb, rest @ ..] if verb == "prepared-source" => {
            return super::canary_package_closure::prepared_source::run(rest);
        }
        [verb, rest @ ..] if verb == "verify-package-closure" => {
            return super::canary_package_closure::run(rest, false);
        }
        [verb, rest @ ..] if verb == "build" => {
            return super::canary_build::run(rest);
        }
        [verb, rest @ ..] if verb == "preflight" => {
            return super::canary_source_plan::preflight(rest);
        }
        [verb, rest @ ..] if verb == "source-plan" => {
            return super::canary_source_plan::run(rest, false);
        }
        [verb, rest @ ..] if verb == "verify-source-plan" => {
            return super::canary_source_plan::run(rest, true);
        }
        _ => (),
    }
    let rest = match args {
        [verb, rest @ ..] if verb == "aggregate" => rest,
        _ => return GRAMMAR.error("an aggregate command is required").emit(),
    };
    let parsed = match GRAMMAR.parse(rest) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("usage: {USAGE}\n")).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unrecognized positional arguments").emit();
    }
    for name in [
        "--package",
        "--identity",
        "--evidence",
        "--run-id",
        "--run-attempt",
    ] {
        if parsed.last(name).is_none() {
            return GRAMMAR
                .error(&format!("the following arguments are required: {name}"))
                .emit();
        }
    }
    match execute(&parsed) {
        Ok(report) => report.emit(),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("canary aggregate rejected: {error}\n"),
        )
        .emit(),
    }
}

fn execute(args: &ParsedArgs) -> DynResult<CheckReport> {
    let identity = Digest::try_from(value(args, "--identity").to_owned())?;
    let package = verify_package(
        Path::new(value(args, "--package")),
        PackageVerification {
            expected_identity_sha256: identity,
            current_run_id: value(args, "--run-id").to_owned(),
            current_run_attempt: value(args, "--run-attempt").to_owned(),
            controller_revision: args.last("--controller-revision").map(str::to_owned),
            selected_source: value(args, "--selected-source").to_owned(),
        },
    )?;
    let context = ReceiptContext::from_verified_package(package);
    let report = aggregate(&context, Path::new(value(args, "--evidence")))?;
    CheckReport::success(format!("{}\n", report.summary())).emit()?;
    append_environment("GITHUB_STEP_SUMMARY", &format!("{}\n", report.summary()))?;
    match report.github_outputs() {
        Some(outputs) => {
            append_environment("GITHUB_OUTPUT", &outputs)?;
            Ok(CheckReport::success(format!(
                "All {} families passed for {} ({})\n",
                report.passed.len(),
                report.candidate,
                report.pass_id
            )))
        }
        None => {
            let errors = report
                .failures
                .iter()
                .map(|failure| {
                    if failure.family.is_empty() {
                        failure.error.message.clone()
                    } else {
                        format!("{}: {}", failure.family, failure.error.message)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(CheckReport::failure(
                String::new(),
                format!("family aggregation failed:\n{errors}\n"),
            ))
        }
    }
}

fn append_environment(name: &str, bytes: &str) -> DynResult<()> {
    if let Some(path) = std::env::var_os(name).filter(|path| !path.is_empty()) {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(bytes.as_bytes())?;
    }
    Ok(())
}

fn value<'a>(args: &'a ParsedArgs, name: &str) -> &'a str {
    args.last(name).unwrap_or_default()
}
