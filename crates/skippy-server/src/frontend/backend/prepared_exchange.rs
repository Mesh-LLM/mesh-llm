//! Admission observes the fully prepared typed request immediately before generation.
use super::*;
use openai_frontend::OpenAiHookPolicy;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct PreparedExchangeState {
    request: Option<ChatCompletionRequest>,
    denied: bool,
}

#[derive(Clone)]
pub(super) struct PreparedExchangeAdmission {
    hooks: Option<Arc<dyn OpenAiHookPolicy>>,
    exchange_id: String,
    state: Arc<Mutex<PreparedExchangeState>>,
}

impl PreparedExchangeAdmission {
    pub(super) fn new(hooks: Option<Arc<dyn OpenAiHookPolicy>>, exchange_id: String) -> Self {
        Self {
            hooks,
            exchange_id,
            state: Arc::new(Mutex::new(PreparedExchangeState::default())),
        }
    }
    pub(super) async fn admit(&self, request: &ChatCompletionRequest) -> OpenAiResult<()> {
        self.state.lock().unwrap().request = Some(request.clone());
        if let Some(hooks) = &self.hooks {
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
        _context: &OpenAiRequestContext,
    ) -> OpenAiResult<Option<super::completion_exchange::CompletionTerminalGuard>> {
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
