//! Lookup and retention policy for Responses continuation state.

use std::sync::Arc;

use super::store::{
    CallerScope, ResponseSessionCache, ResponseStore, StoredResponse, previous_response_not_found,
};
use crate::errors::InferenceError;

/// Where a request may look up and save earlier responses.
///
/// Storage is a startup setting: when it is off no shared response is saved.
/// A persistent transport may additionally install a latest-only local cache;
/// its shared-store fallback policy is selected independently for every turn.
#[derive(Debug, Clone, Default)]
pub struct ResponseContinuation {
    store: Option<Arc<ResponseStore>>,
    scope: Option<CallerScope>,
    pub(super) session: Option<Arc<ResponseSessionCache>>,
    shared_fallback: bool,
}

impl ResponseContinuation {
    pub fn new(store: Option<Arc<ResponseStore>>, scope: Option<CallerScope>) -> Self {
        Self {
            store,
            scope,
            session: None,
            shared_fallback: true,
        }
    }

    /// No lookup and no save.
    pub fn disabled() -> Self {
        Self::default()
    }

    pub(super) fn available(&self) -> Option<(&ResponseStore, &CallerScope)> {
        Some((self.store.as_deref()?, self.scope.as_ref()?))
    }

    pub(super) fn for_session(
        store: Option<Arc<ResponseStore>>,
        scope: Option<CallerScope>,
        session: Arc<ResponseSessionCache>,
        shared_fallback: bool,
    ) -> Self {
        Self {
            store,
            scope,
            session: Some(session),
            shared_fallback,
        }
    }

    pub(super) fn lookup(&self, id: &str) -> Result<Arc<StoredResponse>, InferenceError> {
        if let Some(response) = self.session.as_ref().and_then(|session| session.get(id)) {
            return Ok(response);
        }
        if !self.shared_fallback {
            return Err(previous_response_not_found(id));
        }
        let (store, scope) = self
            .available()
            .ok_or_else(|| previous_response_not_found(id))?;
        store.get(scope, id)
    }
}
