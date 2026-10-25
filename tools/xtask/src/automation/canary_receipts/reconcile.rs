//! One bounded same-candidate infrastructure retry, retaining admitted failures.
mod admission;
#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests;

use super::aggregate::{FamilyJobResult, ReceiptFailure};
use super::feedback::{self, FamilyEvidence, FeedbackDraft, FeedbackState, PublishedFeedback};
use super::{Error, ErrorKind, Family, ReceiptContext, RunAttempt};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReconcileState {
    Green,
    CandidateRepairable,
    InfrastructureExhausted,
    TerminalContract,
}

#[derive(Serialize)]
pub(crate) struct ReconcileReport {
    pub(crate) candidate: String,
    pub(crate) branch: String,
    pub(crate) pass_id: String,
    pub(crate) retry_families: BTreeSet<Family>,
    pub(crate) passed: Vec<Family>,
    pub(crate) selected_attempts: BTreeMap<Family, RunAttempt>,
    pub(crate) candidate_failures: BTreeSet<Family>,
    pub(crate) infrastructure_failures: BTreeSet<Family>,
    pub(crate) failures: Vec<ReceiptFailure>,
    pub(crate) state: ReconcileState,
    #[serde(skip)]
    candidate_evidence: Vec<FamilyEvidence>,
}
impl ReconcileReport {
    pub(crate) fn summary(&self) -> String {
        let mut output = format!(
            "Canary {} infrastructure recheck: {}/{} retry families passed for {}\n",
            self.pass_id,
            self.passed.len(),
            self.retry_families.len(),
            self.candidate
        );
        for failure in &self.failures {
            output.push_str("- ");
            output.push_str(&failure.family);
            output.push_str(": ");
            output.push_str(&failure.error.message);
            output.push('\n');
        }
        output
    }

    /// Only candidate feedback can be emitted after the single infrastructure retry.
    pub(crate) fn publish_feedback(
        self,
        context: &ReceiptContext,
        destination: &Path,
    ) -> Result<Option<PublishedFeedback>, Error> {
        if self.state != ReconcileState::CandidateRepairable {
            return Ok(None);
        }
        FeedbackDraft::new(
            context,
            FeedbackState::CandidateRepairable,
            self.candidate_evidence,
            BTreeSet::new(),
            self.failures
                .into_iter()
                .map(|failure| format!("{}: {}", failure.family, failure.error))
                .collect(),
        )?
        .publish(context, destination)
        .map(Some)
    }
}

pub(crate) fn reconcile(
    context: &ReceiptContext,
    previous_feedback: &Path,
    retries: &Path,
    job_result: FamilyJobResult,
) -> Result<ReconcileReport, Error> {
    let previous = feedback::verify(
        context,
        previous_feedback,
        FeedbackState::InfrastructureRetryable,
    )?;
    let retained = previous.candidate_evidence(context)?;
    let identity = &context.package.identity;
    let mut report = ReconcileReport {
        candidate: identity.candidate.clone(),
        branch: identity.branch.clone(),
        pass_id: identity.pass_id.clone(),
        retry_families: previous.infrastructure_failures().iter().cloned().collect(),
        passed: Vec::new(),
        selected_attempts: BTreeMap::new(),
        candidate_failures: previous.candidate_failures().iter().cloned().collect(),
        infrastructure_failures: BTreeSet::new(),
        failures: Vec::new(),
        state: ReconcileState::Green,
        candidate_evidence: retained,
    };
    match admission::collect(context, retries, &report.retry_families) {
        Ok(selected) => {
            for (family, evidence) in selected.receipts {
                report
                    .selected_attempts
                    .insert(family.clone(), evidence.attempt.clone());
                match admission::check(context, &family, &evidence) {
                    Ok(None) => report.passed.push(family),
                    Ok(Some(failed)) => {
                        report.failures.push(ReceiptFailure {
                            family: family.to_string(),
                            error: Error::new(
                                ErrorKind::WorkerOutcome,
                                "retry certification did not pass",
                            ),
                        });
                        if failed.is_candidate() {
                            report.candidate_failures.insert(family);
                            report.candidate_evidence.push(failed);
                        } else {
                            report.infrastructure_failures.insert(family);
                        }
                    }
                    Err(error) => terminal(&mut report, family.to_string(), error),
                }
            }
        }
        Err(error) => terminal(&mut report, String::new(), error),
    }
    finish(&mut report, job_result);
    Ok(report)
}

fn terminal(report: &mut ReconcileReport, family: String, error: Error) {
    report.state = ReconcileState::TerminalContract;
    report.failures.push(ReceiptFailure { family, error });
}

fn finish(report: &mut ReconcileReport, job_result: FamilyJobResult) {
    let missing: BTreeSet<_> = report
        .retry_families
        .iter()
        .filter(|family| !report.selected_attempts.contains_key(*family))
        .cloned()
        .collect();
    if !missing.is_empty() {
        report
            .infrastructure_failures
            .extend(missing.iter().cloned());
        report.failures.push(ReceiptFailure {
            family: String::new(),
            error: Error::new(
                ErrorKind::MissingReceipts,
                format!("missing retry receipts: {missing:?}"),
            ),
        });
    }
    if job_result != FamilyJobResult::Success {
        report.failures.push(ReceiptFailure {
            family: String::new(),
            error: Error::new(
                ErrorKind::WorkerOutcome,
                format!("retry family job graph result: {job_result:?}"),
            ),
        });
        if matches!(
            job_result,
            FamilyJobResult::Cancelled | FamilyJobResult::Skipped
        ) && missing.is_empty()
        {
            report.state = ReconcileState::TerminalContract;
        } else if report.infrastructure_failures.is_empty() && report.candidate_failures.is_empty()
        {
            report
                .infrastructure_failures
                .extend(report.retry_families.iter().cloned());
        }
    }
    if report.state != ReconcileState::TerminalContract {
        report.state = if !report.infrastructure_failures.is_empty() {
            ReconcileState::InfrastructureExhausted
        } else if !report.candidate_failures.is_empty() {
            ReconcileState::CandidateRepairable
        } else {
            ReconcileState::Green
        };
    }
}
