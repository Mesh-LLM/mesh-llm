//! Trusted, in-process isolation metadata. Never decoded from an HTTP body.

use std::fmt;

use uuid::Uuid;

/// An opaque cache/session domain selected by the owner of a shared backend.
///
/// Allocate a fresh scope for each independently authorized serving attachment.
/// Reusing a scope deliberately permits reuse of that attachment's cached state.
/// This is logical isolation, not a hardware or timing side-channel boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InferenceExecutionScope(Uuid);

impl InferenceExecutionScope {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl Default for InferenceExecutionScope {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InferenceExecutionScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.simple().fmt(formatter)
    }
}

/// An admission lease that must outlive native execution, including cancellation.
///
/// Backends retain the request context inside their actual worker, not only in
/// the HTTP future or response stream. Dropping the final lease releases the
/// owner's shared capacity; a disconnected client alone does not release it.
pub trait InferenceExecutionLease: fmt::Debug + Send + Sync + 'static {}
