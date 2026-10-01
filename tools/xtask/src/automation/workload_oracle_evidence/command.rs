use super::{Error, VerifyRequest, WorkloadClass, WriteRequest, verify_evidence, write_evidence};
use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

const WRITE_USAGE: &str = "cargo xtool automation workload-oracle-evidence write --output <path> --comparison-log <path> --class <class> --smoke-lane <lane> --model-id <id> --model-sha256 <sha256> [--projector-path <path>] --candidate-executable <path> --oracle-executable <path> --pinned-patch-sha <sha> --work-dir <path>";
const VERIFY_USAGE: &str = "cargo xtool automation workload-oracle-evidence verify --evidence <path> --class <class> --smoke-lane <lane> --oracle-lane <lane> --model-id <id> --model-path <path> [--projector-path <path>] --candidate-executable <path> --oracle-executable <path> --pinned-patch-sha <sha>";

const WRITE_GRAMMAR: Grammar = Grammar {
    usage: WRITE_USAGE,
    values: &[
        "--output",
        "--comparison-log",
        "--class",
        "--smoke-lane",
        "--model-id",
        "--model-sha256",
        "--projector-path",
        "--candidate-executable",
        "--oracle-executable",
        "--pinned-patch-sha",
        "--work-dir",
    ],
    flags: &["--help"],
};

const VERIFY_GRAMMAR: Grammar = Grammar {
    usage: VERIFY_USAGE,
    values: &[
        "--evidence",
        "--class",
        "--smoke-lane",
        "--oracle-lane",
        "--model-id",
        "--model-path",
        "--projector-path",
        "--candidate-executable",
        "--oracle-executable",
        "--pinned-patch-sha",
    ],
    flags: &["--help"],
};

const WRITE_REQUIRED: &[&str] = &[
    "--output",
    "--comparison-log",
    "--class",
    "--smoke-lane",
    "--model-id",
    "--model-sha256",
    "--candidate-executable",
    "--oracle-executable",
    "--pinned-patch-sha",
    "--work-dir",
];

const VERIFY_REQUIRED: &[&str] = &[
    "--evidence",
    "--class",
    "--smoke-lane",
    "--oracle-lane",
    "--model-id",
    "--model-path",
    "--candidate-executable",
    "--oracle-executable",
    "--pinned-patch-sha",
];

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let report = match args {
        [verb, rest @ ..] if verb == "write" => write_command(rest),
        [verb, rest @ ..] if verb == "verify" => verify_command(rest),
        _ => CheckReport::usage(
            "cargo xtool automation workload-oracle-evidence {write|verify} ...",
            "a write or verify command is required",
        ),
    };
    report.emit()
}

enum CommandOptions {
    Help(String),
    Parsed(ParsedArgs),
}

fn parse(
    grammar: &'static Grammar,
    required: &'static [&'static str],
    args: &[String],
) -> Result<CommandOptions, CheckReport> {
    let parsed = grammar.parse(args)?;
    if parsed.flag("--help") {
        return Ok(CommandOptions::Help(format!("usage: {}\n", grammar.usage)));
    }
    if !parsed.positionals.is_empty() {
        return Err(grammar.error(&format!(
            "unrecognized arguments: {}",
            parsed.positionals.join(" ")
        )));
    }
    for name in required {
        if parsed.last(name).is_none() {
            return Err(grammar.error(&format!("the following arguments are required: {name}")));
        }
    }
    Ok(CommandOptions::Parsed(parsed))
}

fn verifier_class_error(name: &str) -> CheckReport {
    VERIFY_GRAMMAR.error(&format!(
        "argument --class: invalid choice: '{name}' (choose from 'embedding', 'rerank', 'encoder_decoder', 'ocr', 'speech_synthesis', 'speech_recognition')"
    ))
}

fn write_command(args: &[String]) -> CheckReport {
    match parse(&WRITE_GRAMMAR, WRITE_REQUIRED, args) {
        Ok(CommandOptions::Help(output)) => CheckReport::success(output),
        Ok(CommandOptions::Parsed(args)) => write(&args),
        Err(report) => report,
    }
}

fn verify_command(args: &[String]) -> CheckReport {
    match parse(&VERIFY_GRAMMAR, VERIFY_REQUIRED, args) {
        Ok(CommandOptions::Help(output)) => CheckReport::success(output),
        Ok(CommandOptions::Parsed(args)) => match validate_verifier_classes(&args) {
            Ok(()) => verify(&args),
            Err(report) => report,
        },
        Err(report) => report,
    }
}

fn validate_verifier_classes(args: &ParsedArgs) -> Result<(), CheckReport> {
    for model_class in args.all("--class") {
        if WorkloadClass::parse(model_class).is_err() {
            return Err(verifier_class_error(model_class));
        }
    }
    Ok(())
}

fn write(args: &ParsedArgs) -> CheckReport {
    let request = WriteRequest {
        output: path(args, "--output"),
        comparison_log: path(args, "--comparison-log"),
        model_class: value(args, "--class"),
        smoke_lane: value(args, "--smoke-lane"),
        model_id: value(args, "--model-id"),
        model_sha256: value(args, "--model-sha256"),
        projector_path: optional_path(args, "--projector-path"),
        candidate_executable: path(args, "--candidate-executable"),
        oracle_executable: path(args, "--oracle-executable"),
        pinned_patch_sha: value(args, "--pinned-patch-sha"),
        work_dir: path(args, "--work-dir"),
    };
    match write_evidence(&request) {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("workload oracle evidence not written: {error}\n"),
        ),
    }
}

fn verify(args: &ParsedArgs) -> CheckReport {
    let model_class = match WorkloadClass::parse(args.last("--class").unwrap_or_default()) {
        Ok(model_class) => model_class,
        Err(Error::UnknownClass(name)) => {
            return verifier_class_error(&name);
        }
        Err(error) => {
            return CheckReport::failure(
                String::new(),
                format!("workload oracle evidence rejected: {error}\n"),
            );
        }
    };
    let request = VerifyRequest {
        evidence: path(args, "--evidence"),
        model_class,
        smoke_lane: value(args, "--smoke-lane"),
        oracle_lane: value(args, "--oracle-lane"),
        model_id: value(args, "--model-id"),
        model_path: path(args, "--model-path"),
        projector_path: optional_path(args, "--projector-path"),
        candidate_executable: path(args, "--candidate-executable"),
        oracle_executable: path(args, "--oracle-executable"),
        pinned_patch_sha: value(args, "--pinned-patch-sha"),
    };
    match verify_evidence(&request) {
        Ok(()) => CheckReport::success(format!(
            "verified {} local-monolithic oracle evidence\n",
            model_class.name()
        )),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("workload oracle evidence rejected: {error}\n"),
        ),
    }
}

fn value(args: &ParsedArgs, name: &str) -> String {
    args.last(name).unwrap_or_default().to_owned()
}

fn path(args: &ParsedArgs, name: &str) -> PathBuf {
    Path::new(args.last(name).unwrap_or_default()).to_path_buf()
}

fn optional_path(args: &ParsedArgs, name: &str) -> Option<PathBuf> {
    args.last(name).map(PathBuf::from)
}
