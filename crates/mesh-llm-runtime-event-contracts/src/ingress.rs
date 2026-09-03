//! Synchronous producer boundary for dependency-leaf runtime facts.
//!
//! Implementations must return immediately without waiting for consumers.
//! Delivery class is derived from the fact, never supplied by a caller.

use crate::RuntimeFact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    Coalesced,
    DroppedProgress,
    DroppedDiagnostic,
    RejectedShuttingDown,
    TerminalDeliveryFailed,
}

pub trait RuntimeEventIngress: Send + Sync {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome;
}
