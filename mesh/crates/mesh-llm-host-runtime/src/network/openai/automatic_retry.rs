//! An uncommitted exhausted model route may return its downstream to directive ingress.
//!
//! This is host-only state, never reconstructed from forwarded HTTP headers.
use super::client_stream::ClientStream;
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct AutomaticRetryHandoff {
    stream: Mutex<Option<ClientStream>>,
}

impl AutomaticRetryHandoff {
    pub(crate) fn retain(&self, stream: ClientStream) {
        *self.stream.lock().unwrap() = Some(stream);
    }

    pub(crate) fn take(&self) -> Option<ClientStream> {
        self.stream.lock().unwrap().take()
    }
}
