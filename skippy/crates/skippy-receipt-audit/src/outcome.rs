//! What an audit says about a node, and what one check of it found.
//!
//! An audit ends one of four ways, and a caller must be able to tell them
//! apart, because they call for different things:
//!
//! - **pass:** every check ran and held.
//! - **fail:** the node's receipt or answer is defective, or contradicts
//!   what the client sent or received. A failure is *provable* when it
//!   lies in something the node signed, so anyone can be shown it: the
//!   receipt, or an answer kept as evidence. It is not when only this
//!   client can see it: the response text, the request.
//! - **unanswered:** the node gave no usable signed answer to the
//!   challenge. Nothing was checked, and that is the node's doing.
//! - **no audit:** the client could not run a check. That says nothing
//!   about the node.
//!
//! A check that stops with an error is sorted by the mark the error
//! carries ([`skippy_receipts::node_fault`]), never by its text.

use serde::Serialize;
use skippy_receipts::node_fault::NodeFault;

/// How an audit of a turn ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Pass,
    Fail,
    NoAudit,
    Unanswered,
}

impl Outcome {
    /// The exit code of `skippy audit-receipt` and `skippy recheck-receipt`.
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Fail => 1,
            Self::NoAudit => 2,
            Self::Unanswered => 3,
        }
    }

    /// The last line of a text report.
    pub fn line(self) -> &'static str {
        match self {
            Self::Pass => "AUDIT PASS",
            Self::Fail => "AUDIT FAIL",
            Self::NoAudit => "NO AUDIT",
            Self::Unanswered => "NO ANSWER",
        }
    }
}

/// What one check of an audit found.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Check<T> {
    /// The check ran: its verdict.
    Verdict(T),
    /// The node's answer is defective: a failure, whatever a verdict would
    /// have been.
    NodeFault(String),
    /// The check could not run here, which says nothing about the node.
    Unavailable(String),
}

impl<T> Check<T> {
    /// Sorts the result of a check: a verdict, an error that blames the
    /// node, or any other error, which is the client's.
    pub fn of(result: anyhow::Result<T>) -> Self {
        match result {
            Ok(verdict) => Self::Verdict(verdict),
            Err(error) if NodeFault::find(&error).is_some() => {
                Self::NodeFault(format!("{error:#}"))
            }
            Err(error) => Self::Unavailable(format!("{error:#}")),
        }
    }

    pub fn verdict(&self) -> Option<&T> {
        match self {
            Self::Verdict(verdict) => Some(verdict),
            _ => None,
        }
    }

    pub fn node_fault(&self) -> Option<&str> {
        match self {
            Self::NodeFault(reason) => Some(reason),
            _ => None,
        }
    }

    pub fn unavailable(&self) -> Option<&str> {
        match self {
            Self::Unavailable(reason) => Some(reason),
            _ => None,
        }
    }
}

/// A check that failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Failure {
    /// The check, and why it failed, in a few words.
    pub check: String,
    /// The failure lies in something the node signed, so anyone can be
    /// shown it: the receipt, or an answer the audit keeps as evidence,
    /// which `skippy recheck-receipt` fails the same way.
    pub provable: bool,
}

impl Failure {
    pub(crate) fn provable(check: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            provable: true,
        }
    }

    pub(crate) fn unprovable(check: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            provable: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use skippy_receipts::node_fault::NodeFault;

    use super::*;

    #[test]
    fn each_outcome_has_its_exit_code_line_and_name() {
        let expected = [
            (Outcome::Pass, 0, "AUDIT PASS", "\"pass\""),
            (Outcome::Fail, 1, "AUDIT FAIL", "\"fail\""),
            (Outcome::NoAudit, 2, "NO AUDIT", "\"no_audit\""),
            (Outcome::Unanswered, 3, "NO ANSWER", "\"unanswered\""),
        ];
        for (outcome, code, line, name) in expected {
            assert_eq!(outcome.exit_code(), code);
            assert_eq!(outcome.line(), line);
            assert_eq!(serde_json::to_string(&outcome).unwrap(), name);
        }
    }

    #[test]
    fn an_error_is_sorted_by_its_mark() {
        let fault: Check<u8> = Check::of(Err(NodeFault::error("a short column")));
        assert_eq!(fault.node_fault(), Some("a short column"));
        // an error nobody marked is the client's: it could not run the check
        let read: anyhow::Result<u8> = Err(anyhow::anyhow!("connection reset"));
        let unmarked = Check::of(read.map_err(|e| e.context("fetch a weight row")));
        assert_eq!(
            unmarked.unavailable(),
            Some("fetch a weight row: connection reset")
        );
        assert_eq!(Check::of(Ok(3u8)).verdict(), Some(&3));
        // the three serialize apart
        let json = |check: &Check<u8>| serde_json::to_string(check).unwrap();
        assert_eq!(json(&Check::Verdict(1)), r#"{"verdict":1}"#);
        assert_eq!(json(&fault), r#"{"node_fault":"a short column"}"#);
        assert!(json(&unmarked).starts_with(r#"{"unavailable":"#));
    }
}
