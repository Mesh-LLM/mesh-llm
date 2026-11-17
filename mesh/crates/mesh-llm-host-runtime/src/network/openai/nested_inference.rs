//! One-use host capabilities for nested single-model inference.
//!
//! The plugin bridge mints a fresh request identity before its loopback call.
//! Only an identity registered here can bypass an automatic virtual override;
//! neither a JSON field nor an arbitrary HTTP header grants that authority.
use mesh_llm_events::logging::identifiers::RequestId;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

fn pending() -> &'static Mutex<HashSet<RequestId>> {
    static PENDING: OnceLock<Mutex<HashSet<RequestId>>> = OnceLock::new();
    PENDING.get_or_init(Mutex::default)
}

pub(super) struct SingleModelCapability(RequestId);

impl SingleModelCapability {
    pub(super) fn new() -> Self {
        let id = RequestId::new();
        pending().lock().unwrap().insert(id);
        Self(id)
    }

    pub(super) fn request_id(&self) -> RequestId {
        self.0
    }
}

impl Drop for SingleModelCapability {
    fn drop(&mut self) {
        pending().lock().unwrap().remove(&self.0);
    }
}

pub(super) fn consume(id: RequestId) -> bool {
    pending().lock().unwrap().remove(&id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_identity_does_not_grant_override_bypass() {
        assert!(!consume(RequestId::new()));
        let capability = SingleModelCapability::new();
        assert!(consume(capability.request_id()));
        assert!(!consume(capability.request_id()));
    }

    #[test]
    fn abandoned_nested_call_revokes_its_capability() {
        let capability = SingleModelCapability::new();
        let id = capability.request_id();
        drop(capability);
        assert!(!consume(id));
    }
}
