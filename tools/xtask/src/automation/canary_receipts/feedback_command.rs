//! Export classified aggregation evidence through its admitted snapshot owner.
use super::{
    Error, ErrorKind, ReceiptContext,
    aggregate::{AggregateReport, AggregateState},
    feedback::{FamilyEvidence, FeedbackDraft, FeedbackState, PublishedFeedback},
};
use std::{collections::BTreeSet, path::Path};

pub(crate) fn publish(
    context: &ReceiptContext,
    report: &AggregateReport,
    destination: &Path,
) -> Result<Option<PublishedFeedback>, Error> {
    let state = match report.state {
        AggregateState::CandidateRepairable => FeedbackState::CandidateRepairable,
        AggregateState::InfrastructureRetryable => FeedbackState::InfrastructureRetryable,
        AggregateState::Green | AggregateState::TerminalContract => return Ok(None),
    };
    let mut admitted = Vec::new();
    let mut missing = BTreeSet::new();
    for family in report
        .candidate_failures
        .union(&report.infrastructure_failures)
    {
        let Some(attempt) = report.selected_attempts.get(family) else {
            if report.infrastructure_failures.contains(family) {
                missing.insert(family.clone());
                continue;
            }
            return Err(contract_error("candidate feedback has no selected receipt"));
        };
        let directory = report
            .selected_evidence()
            .get(family)
            .ok_or_else(|| contract_error("selected feedback evidence directory missing"))?;
        let digest = report
            .selected_receipt_digests()
            .get(family)
            .ok_or_else(|| contract_error("selected feedback receipt preimage missing"))?;
        let evidence = FamilyEvidence::admit(context, family, directory)?;
        if !evidence.matches_selection(attempt, digest)
            || evidence.is_candidate() != report.candidate_failures.contains(family)
        {
            return Err(contract_error(
                "selected feedback receipt or failure class changed",
            ));
        }
        admitted.push(evidence);
    }
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
        .collect();
    FeedbackDraft::new(context, state, admitted, missing, errors)?
        .publish(context, destination)
        .map(Some)
}

pub(crate) fn aggregate_outputs(
    report: &AggregateReport,
    published: Option<&PublishedFeedback>,
) -> String {
    match report.state {
        AggregateState::Green => "green=true\nstate=green\nrepairable=false\nfeedback_ready=false\n".into(),
        AggregateState::TerminalContract => "green=false\nstate=terminal_contract\nrepairable=false\nfailure_class=contract\nfailure_stage=family-certification\nfeedback_ready=false\n".into(),
        AggregateState::CandidateRepairable | AggregateState::InfrastructureRetryable => outputs(published),
    }
}

pub(crate) fn outputs(published: Option<&PublishedFeedback>) -> String {
    let Some(feedback) = published else {
        return "feedback_ready=false\n".into();
    };
    let (state, repairable, class) = match feedback.state() {
        FeedbackState::CandidateRepairable => ("candidate_repairable", "true", "candidate"),
        FeedbackState::InfrastructureRetryable => {
            ("infrastructure_retryable", "false", "infrastructure")
        }
    };
    format!(
        "green=false\nstate={state}\nrepairable={repairable}\nfailure_class={class}\nfailure_stage=family-certification\nfeedback_ready=true\n"
    )
}
fn contract_error(message: &str) -> Error {
    Error::new(ErrorKind::ReceiptIdentity, message)
}

#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
#[path = "feedback_command/tests.rs"]
mod tests;
