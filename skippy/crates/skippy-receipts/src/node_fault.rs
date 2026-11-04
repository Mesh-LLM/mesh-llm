//! Who an audit error blames.
//!
//! A check of an opening can stop for two reasons that a client must not
//! mix up. Either the node's answer is defective: a value is missing,
//! malformed, of the wrong length, not a number, or not the one the receipt
//! commits to. That is the node's fault, and when the node signed the
//! answer, anyone can see it. Or the client could not run the check: the
//! claimed GGUF is out of reach, a weight read failed, its own evaluator
//! cannot load the layer. That says nothing about the node.
//!
//! An error is marked as the node's where it is created ([`NodeFault`]).
//! An error without the mark is the client's. So a new error source is
//! "could not run here" until someone decides otherwise, and a node cannot
//! turn a failure into no audit by provoking an error the code did not
//! foresee as its own.
//!
//! The mark is in the shared layer because the shared decoders of opened
//! values set it ([`crate::cache_types`], [`crate::attention`]). Only a
//! client reads it. A node that runs the same decoders on its own rows gets
//! an ordinary error.

use std::fmt;

use anyhow::Result;

/// An error the serving node caused with what it sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeFault {
    reason: String,
}

impl NodeFault {
    /// An error that blames the node, for `reason`.
    pub fn error(reason: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(Self {
            reason: reason.into(),
        })
    }

    /// `error`, blamed on the node. An error that already blames it is
    /// returned as it is.
    pub fn blame(error: anyhow::Error) -> anyhow::Error {
        if Self::find(&error).is_some() {
            return error;
        }
        Self::error(format!("{error:#}"))
    }

    /// The mark of `error`, when it blames the node.
    pub fn find(error: &anyhow::Error) -> Option<&Self> {
        error.downcast_ref::<Self>()
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for NodeFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for NodeFault {}

/// Blames the node for the error of a result.
pub(crate) trait BlameNode<T> {
    /// The result with its error marked as the node's fault.
    fn or_node_fault(self) -> Result<T>;
}

impl<T> BlameNode<T> for Result<T> {
    fn or_node_fault(self) -> Result<T> {
        self.map_err(NodeFault::blame)
    }
}

/// Fails with an error that blames the node, unless `condition` holds.
macro_rules! ensure_node {
    ($condition:expr, $($reason:tt)+) => {
        if !$condition {
            return Err($crate::node_fault::NodeFault::error(format!($($reason)+)));
        }
    };
}
pub(crate) use ensure_node;

/// Fails with an error that blames the node.
macro_rules! node_fault {
    ($($reason:tt)+) => {
        return Err($crate::node_fault::NodeFault::error(format!($($reason)+)))
    };
}
pub(crate) use node_fault;

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    fn defective() -> Result<()> {
        ensure_node!(1 + 1 == 3, "position {}: the residual is short", 4);
        Ok(())
    }

    #[test]
    fn a_marked_error_blames_the_node_through_added_context() {
        let error = defective().context("layer 3").unwrap_err();
        let fault = NodeFault::find(&error).expect("the node's fault");
        assert_eq!(fault.reason(), "position 4: the residual is short");
        assert_eq!(
            format!("{error:#}"),
            "layer 3: position 4: the residual is short"
        );
    }

    #[test]
    fn an_unmarked_error_blames_nobody() {
        let error = anyhow::anyhow!("read past the end").context("fetch a weight row");
        assert!(NodeFault::find(&error).is_none());
        // blaming it keeps the whole message
        let blamed = NodeFault::blame(error);
        assert_eq!(
            NodeFault::find(&blamed).unwrap().reason(),
            "fetch a weight row: read past the end"
        );
        // and blaming twice changes nothing
        let again = NodeFault::blame(blamed);
        assert_eq!(
            format!("{again:#}"),
            "fetch a weight row: read past the end"
        );
    }

    #[test]
    fn a_result_is_blamed_in_one_call() {
        let decoded: Result<u8> = Err(anyhow::anyhow!("not base64"));
        let error = decoded.or_node_fault().unwrap_err();
        assert!(NodeFault::find(&error).is_some());
        let fine: Result<u8> = Ok(1);
        assert_eq!(fine.or_node_fault().unwrap(), 1);
    }
}
