//! An uncommitted exhausted model route may return its downstream to directive ingress.
//!
//! This is host-only state, never reconstructed from forwarded HTTP headers.
use super::client_stream::ClientStream;
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct AutomaticRetryHandoff {
    stream: Mutex<Option<ClientStream>>,
    repick: bool,
}

impl AutomaticRetryHandoff {
    /// The handoff for a re-picked model, whose dispatch needs a ready host.
    pub(crate) fn for_repick() -> Self {
        Self {
            repick: true,
            ..Self::default()
        }
    }

    /// Whether a cooling host is out of this dispatch. The first dispatch
    /// keeps the ordinary availability rule, so a model whose only host is
    /// cooling is still tried rather than locked out; a re-pick must not
    /// resurrect an all-cooling model the spill already moved away from.
    pub(crate) fn requires_ready_host(&self) -> bool {
        self.repick
    }

    pub(crate) fn retain(&self, stream: ClientStream) {
        *self.stream.lock().unwrap() = Some(stream);
    }

    pub(crate) fn take(&self) -> Option<ClientStream> {
        self.stream.lock().unwrap().take()
    }
}
