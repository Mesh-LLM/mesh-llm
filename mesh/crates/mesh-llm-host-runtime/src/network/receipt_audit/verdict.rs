//! What a watched turn says about the peer that served it.
//!
//! Four verdicts, and each calls for something else (`enforcement`):
//!
//! - **passed:** every check ran and held. Nothing happens.
//! - **failed:** a defect lies in something the peer signed, its receipt or
//!   an answer kept as evidence, so anyone can be shown it and
//!   `skippy recheck-receipt` fails the same way. The peer is blocked.
//! - **strike:** the peer gave no usable signed answer to a challenge, or
//!   did something only this node saw: the response is not the text of the
//!   sampled tokens, the sampler statement contradicts the request, the
//!   logits fail the head key. Nobody else can be shown it, and one lost
//!   answer must not block an honest peer, so it counts as one strike
//!   (`strikes`).
//! - **no audit:** this node did not or could not audit the turn. That
//!   says nothing about the peer.

use skippy_receipt_audit::outcome::Outcome;
use skippy_receipt_audit::output::AuditOutput;
use skippy_receipt_audit::turn_audit::TurnAudit;
use skippy_receipts::evidence::Evidence;
use skippy_receipts::wire::Receipt;

use super::opener::Late;
use super::peer_text::Shown;

/// The most failures a verdict lists.
const LISTED_FAILURES: usize = 8;

/// What shows a failure to anyone.
#[derive(Debug)]
pub(super) enum Proof {
    /// The signed receipt, the challenge, the signed opening and both VRF
    /// proofs: what `skippy recheck-receipt --evidence` checks again.
    Evidence(Box<Evidence>),
    /// A receipt the audit refused before any challenge: the signed
    /// receipt alone shows that it describes no turn of the model it names.
    RefusedReceipt(Box<Receipt>),
}

#[derive(Debug)]
pub(super) enum Verdict {
    Passed,
    Failed { failures: Vec<Shown>, proof: Proof },
    Strike(Shown),
    NoAudit(Shown),
}

impl Verdict {
    pub(super) fn strike(reason: impl AsRef<str>) -> Self {
        Self::Strike(Shown::of(reason.as_ref()))
    }

    pub(super) fn no_audit(reason: impl AsRef<str>) -> Self {
        Self::NoAudit(Shown::of(reason.as_ref()))
    }

    /// The verdict of an audit that ran. `late` is how this node itself was
    /// late with a request, if it was: an answer it did not ask for in time
    /// is not one the peer failed to give.
    pub(super) fn of(audit: TurnAudit, late: Option<Late>) -> Self {
        match (audit, late) {
            (TurnAudit::NoAudit(output), _) => Self::no_audit(output.reason),
            (TurnAudit::Unanswered(_), Some(late)) => Self::no_audit(late.to_string()),
            (TurnAudit::Unanswered(output), None) => Self::strike(format!(
                "no usable signed answer to /v1/receipts/{}: {}",
                output.request, output.reason
            )),
            (TurnAudit::Audited { output, evidence }, late) => {
                Self::of_checks(*output, evidence, late)
            }
        }
    }

    fn of_checks(output: AuditOutput, evidence: Option<Box<Evidence>>, late: Option<Late>) -> Self {
        let failures: Vec<Shown> = output
            .failures
            .iter()
            .take(LISTED_FAILURES)
            .map(|failure| Shown::of(&failure.check))
            .collect();
        if output.provable() {
            let proof = evidence
                .map(Proof::Evidence)
                .unwrap_or_else(|| Proof::RefusedReceipt(Box::new(output.receipt)));
            return Self::Failed { failures, proof };
        }
        if !failures.is_empty() {
            let listed: Vec<&str> = failures.iter().map(Shown::as_str).collect();
            return Self::strike(format!(
                "failed checks only this node saw: {}",
                listed.join("; ")
            ));
        }
        match (output.outcome, late) {
            (Outcome::Pass, _) => Self::Passed,
            (_, Some(late)) => Self::no_audit(late.to_string()),
            (Outcome::Unanswered, None) => {
                Self::strike("no usable signed answer to /v1/receipts/rows")
            }
            (Outcome::NoAudit | Outcome::Fail, None) => Self::no_audit(format!(
                "checks could not run here: {}",
                output.unchecked.join("; ")
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use skippy_receipt_audit::output::{NoAuditOutput, UnansweredOutput};

    use super::*;

    fn unanswered(request: &'static str) -> TurnAudit {
        TurnAudit::Unanswered(UnansweredOutput {
            outcome: Outcome::Unanswered,
            pass: false,
            receipt_id: "r".into(),
            request,
            reason: "503: busy\nAUDIT PASS".into(),
        })
    }

    #[test]
    fn a_request_the_peer_did_not_answer_is_a_strike() {
        for request in ["open", "rows"] {
            let Verdict::Strike(why) = Verdict::of(unanswered(request), None) else {
                panic!("an unanswered {request} request is no strike");
            };
            let why = why.to_string();
            assert!(why.contains(&format!("/v1/receipts/{request}")), "{why}");
            assert!(why.ends_with("503: busy\\nAUDIT PASS"), "{why}");
        }
    }

    /// What this node did not ask for in time, the peer did not fail to
    /// answer.
    #[test]
    fn a_request_this_node_was_late_with_is_no_audit() {
        let late_challenge = Late::Challenge(Duration::from_secs(12));
        let Verdict::NoAudit(why) = Verdict::of(unanswered("open"), Some(late_challenge)) else {
            panic!("this node's lateness is held against the peer");
        };
        assert!(
            why.to_string().contains("12.0 s after the receipt"),
            "{why}"
        );
        // the opening was checked and held; the rows were asked for too late
        let late_rows = Late::Rows(Duration::from_secs(45));
        let Verdict::NoAudit(why) = Verdict::of(unanswered("rows"), Some(late_rows)) else {
            panic!("this node's lateness is held against the peer");
        };
        let why = why.to_string();
        assert!(why.contains("the attention rows were not checked"), "{why}");
        assert!(why.contains("45.0 s after its challenge"), "{why}");
    }

    #[test]
    fn an_audit_that_could_not_run_is_no_audit() {
        let output = NoAuditOutput {
            outcome: Outcome::NoAudit,
            pass: false,
            receipt_id: "r".into(),
            reason: "no rounding profiles for backend Abacus yet".into(),
        };
        let verdict = Verdict::of(TurnAudit::NoAudit(output), None);
        let Verdict::NoAudit(why) = verdict else {
            panic!("an audit this node could not run says something of the peer");
        };
        assert!(why.to_string().contains("Abacus"), "{why}");
    }
}
