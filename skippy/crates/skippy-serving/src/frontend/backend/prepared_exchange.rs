//! Admission observes the fully prepared typed request immediately before generation.
use super::*;
use skippy_inference_api::InferenceHookPolicy;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct PreparedExchangeState {
    request: Option<ChatCompletionRequest>,
    denied: bool,
}

#[derive(Clone)]
pub(super) struct PreparedExchangeAdmission {
    hooks: Option<Arc<dyn InferenceHookPolicy>>,
    exchange_id: String,
    state: Arc<Mutex<PreparedExchangeState>>,
}

impl PreparedExchangeAdmission {
    pub(super) fn new(hooks: Option<Arc<dyn InferenceHookPolicy>>, exchange_id: String) -> Self {
        Self {
            hooks,
            exchange_id,
            state: Arc::new(Mutex::new(PreparedExchangeState::default())),
        }
    }
    pub(super) async fn admit(&self, request: &ChatCompletionRequest) -> InferenceResult<()> {
        if let Some(hooks) = &self.hooks {
            if hooks.observes_dispatched_request() {
                self.state.lock().unwrap().request = Some(request.clone());
            }
            let route = ChatExchangeRoute::for_request(request, self.exchange_id.clone());
            if let Err(error) = hooks.admit_effective_chat_completion(request, &route).await {
                self.state.lock().unwrap().denied = true;
                return Err(error);
            }
            hooks.on_effective_chat_completion(request, &route).await;
        }
        Ok(())
    }
    pub(super) fn request(&self) -> Option<ChatCompletionRequest> {
        if self
            .hooks
            .as_ref()
            .is_some_and(|h| h.observes_dispatched_request())
        {
            self.state.lock().unwrap().request.clone()
        } else if self.hooks.is_some() {
            Some(ChatCompletionRequest::default())
        } else {
            None
        }
    }
    pub(super) fn denied(&self) -> bool {
        self.state.lock().unwrap().denied
    }
}

impl StageOpenAiBackend {
    pub(super) async fn admit_prepared_completion(
        &self,
        request: &CompletionRequest,
        _context: &InferenceRequestContext,
    ) -> InferenceResult<Option<super::completion_exchange::CompletionTerminalGuard>> {
        let Some(hooks) = self
            .hook_policy
            .as_ref()
            .filter(|h| h.requires_exchange_lifecycle())
        else {
            return Ok(None);
        };
        let id = uuid::Uuid::new_v4().to_string();
        let mut guard =
            super::completion_exchange::CompletionTerminalGuard::new(hooks.clone(), id.clone());
        if let Err(error) = hooks.admit_effective_completion(request, &id).await {
            guard.finish(if error.status().as_u16() == 403 {
                "policy_denied"
            } else {
                "internal_hook_failure"
            });
            return Err(error);
        }
        Ok(Some(guard))
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct AdmissionPolicy {
        observe: bool,
        calls: AtomicUsize,
    }
    #[async_trait]
    impl InferenceHookPolicy for AdmissionPolicy {
        fn observes_dispatched_request(&self) -> bool {
            self.observe
        }
        async fn admit_effective_chat_completion(
            &self,
            _request: &ChatCompletionRequest,
            _route: &ChatExchangeRoute,
        ) -> InferenceResult<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn prepared_request_retention_requires_observation_without_skipping_admission() {
        let request = ChatCompletionRequest {
            model: "prepared-model".into(),
            ..Default::default()
        };
        let absent = PreparedExchangeAdmission::new(None, "absent".into());
        absent.admit(&request).await.unwrap();
        assert!(absent.state.lock().unwrap().request.is_none());
        for observe in [false, true] {
            let policy = Arc::new(AdmissionPolicy {
                observe,
                calls: AtomicUsize::new(0),
            });
            let admission = PreparedExchangeAdmission::new(Some(policy.clone()), "id".into());
            admission.admit(&request).await.unwrap();
            assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
            let retained = admission.state.lock().unwrap().request.clone();
            assert_eq!(retained.is_some(), observe);
            if let Some(retained) = retained {
                assert_eq!(retained.model, "prepared-model");
            }
        }
    }
}
