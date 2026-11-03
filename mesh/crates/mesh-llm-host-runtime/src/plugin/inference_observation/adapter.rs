//! Turns one local frontend's lifecycle events into observation events.
//!
//! The adapter is composed into the observer of each local inference
//! frontend, so it sees only what this node executes: local API requests and
//! requests that arrived from peers, which the model-aware proxy also serves
//! on the local frontend. A request this node only forwards never reaches one.
//! The lifecycle context has no model, so each adapter is built with its
//! frontend's.

use super::hub::ObservationHub;
use super::proto;
use skippy_inference_api::{
    InferenceBackendOperation, InferenceLifecycleEvent, InferenceLifecycleObserver,
    InferenceTerminalResult, InferenceUsage,
};
use std::sync::Arc;

pub(crate) struct ObservationAdapter {
    hub: Arc<ObservationHub>,
    model: String,
}

impl ObservationAdapter {
    pub(crate) fn new(hub: Arc<ObservationHub>, model: impl Into<String>) -> Self {
        let model = model.into();
        hub.model_ready(&model);
        Self { hub, model }
    }
}

impl Drop for ObservationAdapter {
    fn drop(&mut self) {
        self.hub.model_unloaded(&self.model);
    }
}

/// What a backend operation runs, or `None` when it runs no model (listing
/// models, counting tokens).
fn executed_operation(
    operation: InferenceBackendOperation,
) -> Option<(proto::InferenceOperation, bool)> {
    use InferenceBackendOperation as Backend;
    use proto::InferenceOperation as Op;
    Some(match operation {
        Backend::Models | Backend::MessagesCountTokens => return None,
        Backend::ChatCompletion => (Op::ChatCompletion, false),
        Backend::ChatCompletionStream => (Op::ChatCompletion, true),
        Backend::Completion => (Op::Completion, false),
        Backend::CompletionStream => (Op::Completion, true),
        Backend::Responses => (Op::Responses, false),
        Backend::ResponsesStream => (Op::Responses, true),
        Backend::Messages => (Op::Messages, false),
        Backend::MessagesStream => (Op::Messages, true),
        Backend::Embeddings => (Op::Embeddings, false),
        Backend::Rerank => (Op::Rerank, false),
        Backend::AudioSpeech => (Op::AudioSpeech, false),
        Backend::AudioTranscription => (Op::AudioTranscription, false),
        Backend::AudioTranslation => (Op::AudioTranslation, false),
        Backend::SystemOne => (Op::SystemOne, false),
    })
}

fn usage_from_response(usage: &InferenceUsage) -> proto::InferenceOutputUsage {
    proto::InferenceOutputUsage {
        prompt_tokens: usage.prompt_tokens,
        cached_prompt_tokens: usage.cached_tokens,
        completion_tokens: usage.completion_tokens,
    }
}

fn usage_from_terminal(result: &InferenceTerminalResult) -> Option<proto::InferenceOutputUsage> {
    let InferenceTerminalResult::CompletedWithUsage { usage, .. } = result else {
        return None;
    };
    let tokens = |value: Option<u64>| value.map(|value| u32::try_from(value).unwrap_or(u32::MAX));
    Some(proto::InferenceOutputUsage {
        prompt_tokens: tokens(usage.prompt_tokens).unwrap_or_default(),
        cached_prompt_tokens: tokens(usage.cached_prompt_tokens),
        completion_tokens: tokens(usage.completion_tokens)?,
    })
}

fn outcome(result: &InferenceTerminalResult) -> proto::InferenceExecutionOutcome {
    match result {
        InferenceTerminalResult::Completed { .. }
        | InferenceTerminalResult::CompletedWithUsage { .. } => {
            proto::InferenceExecutionOutcome::Completed
        }
        InferenceTerminalResult::Failed { .. } => proto::InferenceExecutionOutcome::Failed,
    }
}

impl InferenceLifecycleObserver for ObservationAdapter {
    fn observe(&self, event: &InferenceLifecycleEvent) {
        use proto::InferenceExecutionOutcome as Outcome;
        let hub = &self.hub;
        match event {
            InferenceLifecycleEvent::BackendDispatched { context, operation } => {
                if let Some((operation, streaming)) = executed_operation(*operation) {
                    hub.started(context.request_id, &self.model, operation, streaming);
                }
            }
            InferenceLifecycleEvent::StreamFirstItem { context, .. } => {
                hub.first_output(&context.request_id);
            }
            // A stream's backend call returns when the stream opens: its
            // compute ends with the stream, unless opening it failed.
            InferenceLifecycleEvent::BackendTerminal {
                context,
                operation,
                result,
            } => {
                let streaming = executed_operation(*operation).is_some_and(|(_, stream)| stream);
                if !streaming || matches!(result, InferenceTerminalResult::Failed { .. }) {
                    hub.compute_finished(&context.request_id);
                }
            }
            InferenceLifecycleEvent::ResponseCompleted { context, usage, .. } => {
                hub.usage(&context.request_id, usage_from_response(usage));
            }
            InferenceLifecycleEvent::StreamTerminal { context, result } => {
                if let Some(usage) = usage_from_terminal(result) {
                    hub.usage(&context.request_id, usage);
                }
                hub.compute_finished(&context.request_id);
                hub.ended(&context.request_id, outcome(result));
            }
            InferenceLifecycleEvent::NonStreamTerminal {
                context, result, ..
            } => {
                hub.compute_finished(&context.request_id);
                hub.ended(&context.request_id, outcome(result));
            }
            InferenceLifecycleEvent::StreamCancelled { context }
            | InferenceLifecycleEvent::RequestCancelled { context } => {
                hub.compute_finished(&context.request_id);
                hub.ended(&context.request_id, Outcome::Cancelled);
            }
            InferenceLifecycleEvent::StreamDropped { context } => {
                hub.compute_finished(&context.request_id);
                hub.ended(&context.request_id, Outcome::Dropped);
            }
            InferenceLifecycleEvent::Rejected { context, .. } => {
                hub.ended(&context.request_id, Outcome::Failed);
            }
            InferenceLifecycleEvent::Admitted { .. }
            | InferenceLifecycleEvent::ExchangeIdentified { .. } => {}
        }
    }
}

/// Fans one frontend's lifecycle events out to the existing observer and the
/// observation adapter.
struct WithObservation {
    base: Option<Arc<dyn InferenceLifecycleObserver>>,
    adapter: ObservationAdapter,
}

impl InferenceLifecycleObserver for WithObservation {
    fn observe(&self, event: &InferenceLifecycleEvent) {
        if let Some(base) = &self.base {
            base.observe(event);
        }
        self.adapter.observe(event);
    }
}

/// `base` plus this node's observation of `model`'s frontend. The model counts
/// as ready while the returned observer lives.
pub(crate) fn observe_frontend(
    base: Option<Arc<dyn InferenceLifecycleObserver>>,
    model: &str,
) -> Option<Arc<dyn InferenceLifecycleObserver>> {
    Some(Arc::new(WithObservation {
        base,
        adapter: ObservationAdapter::new(super::hub(), model),
    }))
}
