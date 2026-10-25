//! Ordered core transformations composed with read-only exchange policies.
use crate::{
    CapsuleMarker, ChatCompletionOutcome, ChatCompletionRequest, ChatCompletionResponse,
    ChatExchangeRoute, ChatHookOutcome, GenerationHookSignals, InferenceHookPolicy,
    InferenceResult, PrefillHookSignals, apply_chat_hook_outcome,
};
use async_trait::async_trait;
use std::sync::Arc;

pub struct CompositeOpenAiHookPolicy {
    policies: Vec<Arc<dyn InferenceHookPolicy>>,
}
impl CompositeOpenAiHookPolicy {
    pub fn new(policies: Vec<Arc<dyn InferenceHookPolicy>>) -> Arc<Self> {
        Arc::new(Self { policies })
    }
}
#[async_trait]
impl InferenceHookPolicy for CompositeOpenAiHookPolicy {
    async fn before_chat_completion(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> InferenceResult<ChatHookOutcome> {
        for policy in &self.policies {
            let outcome = policy.before_chat_completion(request).await?;
            apply_chat_hook_outcome(request, &outcome);
        }
        Ok(ChatHookOutcome::none())
    }
    async fn after_prefill(
        &self,
        request: &mut ChatCompletionRequest,
        signals: PrefillHookSignals,
    ) -> InferenceResult<ChatHookOutcome> {
        for policy in &self.policies {
            let outcome = policy.after_prefill(request, signals.clone()).await?;
            apply_chat_hook_outcome(request, &outcome);
        }
        Ok(ChatHookOutcome::none())
    }
    async fn mid_generation(
        &self,
        request: &mut ChatCompletionRequest,
        signals: GenerationHookSignals,
    ) -> InferenceResult<ChatHookOutcome> {
        for policy in &self.policies {
            let outcome = policy.mid_generation(request, signals.clone()).await?;
            apply_chat_hook_outcome(request, &outcome);
        }
        Ok(ChatHookOutcome::none())
    }
    async fn admit_effective_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) -> InferenceResult<()> {
        for policy in &self.policies {
            policy
                .admit_effective_chat_completion(request, route)
                .await?;
        }
        Ok(())
    }
    async fn admit_effective_completion(
        &self,
        request: &crate::CompletionRequest,
        exchange_id: &str,
    ) -> InferenceResult<()> {
        for policy in &self.policies {
            policy
                .admit_effective_completion(request, exchange_id)
                .await?;
        }
        Ok(())
    }
    async fn on_effective_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) {
        for policy in &self.policies {
            policy.on_effective_chat_completion(request, route).await;
        }
    }
    async fn on_completion_terminal(&self, exchange_id: &str, outcome: &str) {
        for policy in &self.policies {
            policy.on_completion_terminal(exchange_id, outcome).await;
        }
    }
    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        for policy in &self.policies {
            policy
                .on_chat_completion_terminal(request, id, outcome)
                .await;
        }
    }
    async fn capsule_marker_for_response(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        for policy in &self.policies {
            if let Some(marker) = policy.capsule_marker_for_response(request, response).await {
                return Some(marker);
            }
        }
        None
    }
    fn observes_dispatched_request(&self) -> bool {
        self.policies
            .iter()
            .any(|p| p.observes_dispatched_request())
    }
    fn requires_exchange_lifecycle(&self) -> bool {
        self.policies
            .iter()
            .any(|p| p.requires_exchange_lifecycle())
    }
    fn http_exchange_policy(&self) -> Option<Arc<dyn crate::http_exchange::HttpExchangePolicy>> {
        self.policies.iter().find_map(|p| p.http_exchange_policy())
    }
}
