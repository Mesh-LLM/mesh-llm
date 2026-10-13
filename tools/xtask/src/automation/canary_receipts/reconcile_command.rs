//! CLI boundary for the single same-candidate infrastructure reconciliation.
use super::{append_environment, feedback_command, value};
use crate::{
    automation::canary_receipts::{
        Digest, FamilyJobResult, PackageVerification, ReceiptContext,
        reconcile::{ReconcileState, reconcile},
        verify_package,
    },
    command::DynResult,
    repository::{
        check_args::{Grammar, ParsedArgs},
        check_report::CheckReport,
    },
};
use std::path::Path;

const USAGE: &str = "cargo xtool automation canary-receipts reconcile --package <path> --identity <sha256> --previous-feedback <path> --evidence <path> --run-id <id> --run-attempt <attempt> --family-result <success|failure|cancelled|skipped> [--controller-revision <sha>] [--selected-source <sha>] [--feedback-output <path>]";
const GRAMMAR: Grammar = Grammar {
    usage: USAGE,
    values: &[
        "--package",
        "--identity",
        "--previous-feedback",
        "--evidence",
        "--run-id",
        "--run-attempt",
        "--family-result",
        "--controller-revision",
        "--selected-source",
        "--feedback-output",
    ],
    flags: &["--help"],
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
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
        "--previous-feedback",
        "--evidence",
        "--run-id",
        "--run-attempt",
        "--family-result",
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
            format!("canary reconciliation rejected: {error}\n"),
        )
        .emit(),
    }
}
fn execute(args: &ParsedArgs) -> DynResult<CheckReport> {
    let graph = FamilyJobResult::parse(value(args, "--family-result"))?;
    let package = verify_package(
        Path::new(value(args, "--package")),
        PackageVerification {
            expected_identity_sha256: Digest::try_from(value(args, "--identity").to_owned())?,
            current_run_id: value(args, "--run-id").to_owned(),
            current_run_attempt: value(args, "--run-attempt").to_owned(),
            controller_revision: args.last("--controller-revision").map(str::to_owned),
            selected_source: value(args, "--selected-source").to_owned(),
        },
    )?;
    let context = ReceiptContext::from_verified_package(package);
    let report = reconcile(
        &context,
        Path::new(value(args, "--previous-feedback")),
        Path::new(value(args, "--evidence")),
        graph,
    )?;
    let state = report.state;
    let summary = report.summary();
    let mut outputs = match state {
        ReconcileState::Green => format!("green=true\nstate=green\nrepairable=false\nfeedback_ready=false\ncandidate={}\nbranch={}\n", report.candidate, report.branch),
        ReconcileState::CandidateRepairable => String::new(),
        ReconcileState::InfrastructureExhausted => "green=false\nstate=infrastructure_exhausted\nrepairable=false\nfeedback_ready=false\nfailure_class=infrastructure\nfailure_stage=infrastructure-recheck\n".into(),
        ReconcileState::TerminalContract => "green=false\nstate=terminal_contract\nrepairable=false\nfeedback_ready=false\nfailure_class=contract\nfailure_stage=infrastructure-recheck\n".into(),
    };
    if state == ReconcileState::CandidateRepairable {
        let feedback = args
            .last("--feedback-output")
            .map(|destination| report.publish_feedback(&context, Path::new(destination)))
            .transpose()?
            .flatten();
        outputs = if feedback.is_some() {
            feedback_command::outputs(feedback.as_ref())
        } else {
            "green=false\nstate=candidate_repairable\nrepairable=false\nfeedback_ready=false\nfailure_class=candidate\nfailure_stage=family-certification\n".into()
        };
    }
    append_environment("GITHUB_OUTPUT", &outputs)?;
    append_environment("GITHUB_STEP_SUMMARY", &summary)?;
    if state == ReconcileState::Green {
        Ok(CheckReport::success(summary))
    } else {
        Ok(CheckReport::failure(
            summary,
            "family reconciliation did not pass\n".into(),
        ))
    }
}
