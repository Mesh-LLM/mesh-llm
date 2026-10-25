//! The error for media a request is not allowed to send.
//!
//! Media past the size limits fails in the runtime, after the request has
//! been accepted. Callers tell this error apart from runtime failures so the
//! client gets an invalid-request response rather than a server error it
//! might retry.

use std::fmt;

/// A media part, or a request's media together, exceeded a limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRejected(String);

impl MediaRejected {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for MediaRejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MediaRejected {}
