pub(super) mod generator;
mod input;
mod policy;
mod types;

use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::path::PathBuf;

const USAGE: &str = "cargo xtool automation rewriter-report --report <json> [--mode validate|idempotence] [--patch-check pass|fail] [--patch-drift-gate warn|fail] [--compile-result pass|fail|skipped] [--graph-verify-result pass|fail|skipped]";
const HELP: &str = "usage: skippy-rewriter-harness.py [-h] --report REPORT\n                                  [--mode {validate,idempotence}]\n                                  [--patch-check {pass,fail}]\n                                  [--patch-drift-gate {warn,fail}]\n                                  [--compile-result {pass,fail,skipped}]\n                                  [--graph-verify-result {pass,fail,skipped}]\n\nCI harness for the Skippy stage-rewriter (Clang Transformer generator).\nConsumes the rewriter's outputs and enforces the contract described in\nPLANS/SKIPPY_REWRITER_HARNESS_CONTRACT_V0.md (workspace copy; repo copy lands\nwith the generator commit): - report validation: per-builder verdicts, refusal\nreasons, proof blocks - patch-drift checking: `--check` result forwarded as\npass/fail - idempotence assertion: second-run report must contain an empty\nedit set The generator (scama's tool) owns producing these artifacts; this\nscript only validates and gates. It never invokes the compiler or the graph\nverifier directly -- those results arrive as pre-computed pass/fail inputs so\nthe harness stays decoupled from the generator's toolchain.\n\noptional arguments:\n  -h, --help            show this help message and exit\n  --report REPORT       rewriter JSON report to validate\n  --mode {validate,idempotence}\n                        validate a first-run report, or assert a second run\n                        edited nothing\n  --patch-check {pass,fail}\n                        forwarded result of the generator's --check byte-\n                        compare\n  --patch-drift-gate {warn,fail}\n                        policy for patch drift while the queue regeneration is\n                        in flight\n  --compile-result {pass,fail,skipped}\n                        compile result of the transformed tree\n  --graph-verify-result {pass,fail,skipped}\n                        no-allocation graph verifier result for the\n                        transformed tree\n";

#[derive(Clone, Copy)]
enum Mode {
    Validate,
    Idempotence,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail,
    Skipped,
}
#[derive(Clone, Copy)]
enum DriftGate {
    Warn,
    Fail,
}

struct Options {
    report: PathBuf,
    mode: Mode,
    patch_check: Option<Outcome>,
    drift_gate: DriftGate,
    compile: Outcome,
    graph: Outcome,
}

fn parse(args: &[String]) -> Result<Options, CheckReport> {
    let (mut report, mut mode, mut patch_check, mut drift_gate, mut compile, mut graph) = (
        None,
        Mode::Validate,
        None,
        DriftGate::Warn,
        Outcome::Skipped,
        Outcome::Skipped,
    );
    let mut args = args.iter();
    let mut unknown = None;
    while let Some(flag) = args.next() {
        if flag == "-h" {
            return Err(CheckReport::success(HELP.into()));
        }
        if flag == "--" {
            return Err(CheckReport::usage(USAGE, "unexpected positional arguments"));
        }
        let (flag, inline) = flag
            .split_once('=')
            .map_or((flag.as_str(), None), |(flag, value)| (flag, Some(value)));
        let flag = match resolve(flag)? {
            Some(flag) => flag,
            None => {
                unknown.get_or_insert(flag);
                continue;
            }
        };
        if flag == "--help" {
            return Err(if inline.is_none() {
                CheckReport::success(HELP.into())
            } else {
                CheckReport::usage(USAGE, "--help does not accept a value")
            });
        }
        let Some(value) = inline.or_else(|| args.next().map(String::as_str)) else {
            return Err(CheckReport::usage(
                USAGE,
                &format!("missing value for {flag}"),
            ));
        };
        if inline.is_none() && value.starts_with('-') {
            return Err(CheckReport::usage(
                USAGE,
                &format!("missing value for {flag}"),
            ));
        }
        match flag {
            "--report" => report = Some(PathBuf::from(value)),
            "--mode" => {
                mode = match value {
                    "validate" => Mode::Validate,
                    "idempotence" => Mode::Idempotence,
                    _ => return Err(invalid(flag, value)),
                }
            }
            "--patch-check" => patch_check = Some(outcome(flag, value, false)?),
            "--patch-drift-gate" => {
                drift_gate = match value {
                    "warn" => DriftGate::Warn,
                    "fail" => DriftGate::Fail,
                    _ => return Err(invalid(flag, value)),
                }
            }
            "--compile-result" => compile = outcome(flag, value, true)?,
            "--graph-verify-result" => graph = outcome(flag, value, true)?,
            _ => return Err(CheckReport::usage(USAGE, &format!("unknown option {flag}"))),
        }
    }
    if let Some(flag) = unknown {
        return Err(CheckReport::usage(USAGE, &format!("unknown option {flag}")));
    }
    let report = report.ok_or_else(|| CheckReport::usage(USAGE, "missing required --report"))?;
    Ok(Options {
        report,
        mode,
        patch_check,
        drift_gate,
        compile,
        graph,
    })
}

fn resolve(flag: &str) -> Result<Option<&'static str>, CheckReport> {
    const FLAGS: [&str; 7] = [
        "--report",
        "--mode",
        "--patch-check",
        "--patch-drift-gate",
        "--compile-result",
        "--graph-verify-result",
        "--help",
    ];
    if !flag.starts_with("--") {
        return Ok(None);
    }
    Ok(FLAGS.into_iter().find(|name| *name == flag))
}

fn invalid(flag: &str, value: &str) -> CheckReport {
    CheckReport::usage(USAGE, &format!("invalid {flag} value: {value}"))
}
fn outcome(flag: &str, value: &str, skipped: bool) -> Result<Outcome, CheckReport> {
    match value {
        "pass" => Ok(Outcome::Pass),
        "fail" => Ok(Outcome::Fail),
        "skipped" if skipped => Ok(Outcome::Skipped),
        _ => Err(invalid(flag, value)),
    }
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let options = match parse(args) {
        Ok(options) => options,
        Err(report) => return report.emit(),
    };
    let report = match input::load(&options.report, options.mode) {
        Ok(report) => report,
        Err(error) => {
            return CheckReport {
                stderr: format!("error: cannot load report: {error}\n"),
                code: 2,
                ..CheckReport::default()
            }
            .emit();
        }
    };
    policy::check(&report, &options).emit()
}
