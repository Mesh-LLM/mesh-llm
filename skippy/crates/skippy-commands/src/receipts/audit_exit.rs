//! How `skippy audit-receipt` and `skippy recheck-receipt` end: an audit
//! that did not pass is an error that carries its outcome, and the process
//! exits with the outcome's code.

use std::fmt;

use skippy_receipt_audit::outcome::Outcome;

/// An audit that did not pass, as the error a command ends with. The
/// report is printed before; this carries the exit code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditExit {
    pub outcome: Outcome,
    reason: String,
}

impl AuditExit {
    pub(super) fn new(outcome: Outcome, reason: impl Into<String>) -> Self {
        Self {
            outcome,
            reason: reason.into(),
        }
    }

    /// How a command ends after an audit with `outcome`: `Ok` for a pass,
    /// else the error that carries the outcome's exit code.
    pub(super) fn result(outcome: Outcome, reason: impl Into<String>) -> anyhow::Result<()> {
        match outcome {
            Outcome::Pass => Ok(()),
            outcome => Err(Self::new(outcome, reason).into()),
        }
    }
}

impl fmt::Display for AuditExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.outcome {
            Outcome::Pass => "the receipt audit passed",
            Outcome::Fail => "the receipt audit failed",
            Outcome::NoAudit => "the receipt was not audited",
            Outcome::Unanswered => "the node did not answer the audit",
        };
        if self.reason.is_empty() {
            f.write_str(what)
        } else {
            write!(f, "{what}: {}", self.reason)
        }
    }
}

impl std::error::Error for AuditExit {}

/// The exit code of a command that ended with `error`, when the error is
/// the outcome of an audit: 1 for a failed audit, 2 for no audit, 3 for a
/// node that did not answer. `None` for any other error.
pub fn exit_code(error: &anyhow::Error) -> Option<u8> {
    error
        .downcast_ref::<AuditExit>()
        .map(|exit| exit.outcome.exit_code())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_exits_with_its_audits_outcome() {
        for outcome in [Outcome::Fail, Outcome::NoAudit, Outcome::Unanswered] {
            let error: anyhow::Error = AuditExit::new(outcome, "why").into();
            assert_eq!(exit_code(&error), Some(outcome.exit_code()));
            // context a caller adds does not hide it
            let error = error.context("audit receipt");
            assert_eq!(exit_code(&error), Some(outcome.exit_code()));
        }
        assert_eq!(exit_code(&anyhow::anyhow!("no such file")), None);
        // a pass ends the command well; every other outcome names its code
        assert!(AuditExit::result(Outcome::Pass, "").is_ok());
        for (outcome, code) in [
            (Outcome::Fail, 1),
            (Outcome::NoAudit, 2),
            (Outcome::Unanswered, 3),
        ] {
            let error = AuditExit::result(outcome, "why").unwrap_err();
            assert_eq!(exit_code(&error), Some(code));
        }
        let exit = AuditExit::new(Outcome::Unanswered, "timed out");
        assert_eq!(
            exit.to_string(),
            "the node did not answer the audit: timed out"
        );
    }
}
