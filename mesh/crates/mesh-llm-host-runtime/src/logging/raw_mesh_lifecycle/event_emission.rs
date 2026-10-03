use mesh_llm_events::logging::{
    events::{LifecycleEvent, TokenUsage},
    identifiers::RequestId,
    replay::ReplayChannel,
};

use super::{
    LifecycleOwnerEntry, LoggingService, RawMeshLifecycleOwners, RawMeshRequestLifecycle,
    lock_recover,
};

const MAX_LOGGED_COMPLETION_TOKENS: u64 = u32::MAX as u64;
const MAX_ROUTE_METADATA_CHARS: usize = 64;

impl RawMeshLifecycleOwners {
    fn emit_stream_started(
        &self,
        service: &LoggingService,
        request_id: RequestId,
        token: u64,
        model: Option<&str>,
    ) {
        let should_emit = {
            let mut coordination = lock_recover(&self.coordination);
            let Some(LifecycleOwnerEntry::Raw(entry)) = coordination.owners.get_mut(&request_id)
            else {
                return;
            };
            if entry.token != token || entry.stream_started || entry.stream_completed {
                false
            } else {
                entry.stream_started = true;
                entry.first_token_recorded = false;
                entry.stream_completed = false;
                entry.stream_error = false;
                true
            }
        };
        if should_emit {
            enqueue_stream_event(
                service,
                request_id,
                LifecycleEvent::StreamStarted {
                    model: bounded_route_metadata(model),
                },
            );
        }
    }

    fn emit_stream_chunk(&self, service: &LoggingService, request_id: RequestId, token: u64) {
        let should_emit = {
            let mut coordination = lock_recover(&self.coordination);
            let Some(LifecycleOwnerEntry::Raw(entry)) = coordination.owners.get_mut(&request_id)
            else {
                return;
            };
            if entry.token != token
                || !entry.stream_started
                || entry.stream_completed
                || entry.stream_error
                || entry.first_token_recorded
            {
                false
            } else {
                entry.first_token_recorded = true;
                true
            }
        };
        if should_emit {
            enqueue_stream_event(
                service,
                request_id,
                LifecycleEvent::StreamChunk { tokens: None },
            );
        }
    }

    fn emit_stream_completed(
        &self,
        service: &LoggingService,
        request_id: RequestId,
        token: u64,
        usage: Option<TokenUsage>,
    ) {
        let usage = usage.and_then(|usage| {
            TokenUsage::from_counts(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            .map(|normalized| normalized.with_cached_prompt_tokens(usage.cached_prompt_tokens))
        });
        let should_emit = {
            let mut coordination = lock_recover(&self.coordination);
            let Some(LifecycleOwnerEntry::Raw(entry)) = coordination.owners.get_mut(&request_id)
            else {
                return;
            };
            if entry.token != token
                || !entry.stream_started
                || entry.stream_completed
                || entry.stream_error
            {
                false
            } else {
                entry.stream_completed = true;
                true
            }
        };
        if should_emit {
            enqueue_stream_event(
                service,
                request_id,
                LifecycleEvent::StreamCompleted {
                    tokens: bounded_completion_tokens(
                        usage.and_then(|usage| usage.completion_tokens),
                    ),
                    usage,
                },
            );
        }
    }

    fn emit_stream_error(
        &self,
        service: &LoggingService,
        request_id: RequestId,
        token: u64,
        label: &'static str,
    ) {
        let should_emit = {
            let mut coordination = lock_recover(&self.coordination);
            let Some(LifecycleOwnerEntry::Raw(entry)) = coordination.owners.get_mut(&request_id)
            else {
                return;
            };
            if entry.token != token || entry.stream_completed || entry.stream_error {
                false
            } else {
                entry.stream_error = true;
                entry.stream_started = false;
                true
            }
        };
        if should_emit {
            enqueue_stream_event(
                service,
                request_id,
                LifecycleEvent::StreamError {
                    error: Some(label.to_owned()),
                },
            );
        }
    }
}

impl RawMeshRequestLifecycle {
    pub(crate) fn route_selected(&self, model: Option<&str>) {
        self.route_selected_with_metadata(model, Some("mesh"), Some("raw_ingress"));
    }

    pub(crate) fn route_selected_with_metadata(
        &self,
        model: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
    ) {
        self.owners.emit_route_selected(
            &self.service,
            self.request_id,
            self.token,
            model,
            provider,
            engine,
        );
    }

    pub(crate) fn stream_started(&self, model: Option<&str>) {
        self.owners
            .emit_stream_started(&self.service, self.request_id, self.token, model);
    }

    /// Record the first produced stream chunk. The canonical event contract
    /// deliberately keeps this metadata-only and represents the first-token
    /// boundary as the first `stream_chunk` event.
    pub(crate) fn stream_first_token(&self) {
        self.owners
            .emit_stream_chunk(&self.service, self.request_id, self.token);
    }

    pub(crate) fn stream_chunk(&self) {
        self.owners
            .emit_stream_chunk(&self.service, self.request_id, self.token);
    }

    pub(crate) fn stream_completed(&self, usage: Option<TokenUsage>) {
        self.owners
            .emit_stream_completed(&self.service, self.request_id, self.token, usage);
    }

    pub(crate) fn stream_error(&self, label: &'static str) {
        self.owners
            .emit_stream_error(&self.service, self.request_id, self.token, label);
    }

    pub(crate) fn stream_cancelled(&self) {
        self.stream_error("client_disconnected");
    }
}

fn enqueue_stream_event(service: &LoggingService, request_id: RequestId, event: LifecycleEvent) {
    if let Ok(payload) = serde_json::to_string(&event) {
        let _ = service.enqueue_event(request_id, ReplayChannel::Operations, payload);
    }
}

fn bounded_route_metadata(value: Option<&str>) -> Option<String> {
    let value = value?;
    let (value, _) = super::super::policy::apply_redaction(value);
    let bounded: String = value
        .chars()
        .take(MAX_ROUTE_METADATA_CHARS)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect();
    (!bounded.is_empty()).then_some(bounded)
}

fn bounded_completion_tokens(tokens: Option<u64>) -> Option<u64> {
    tokens.filter(|value| *value <= MAX_LOGGED_COMPLETION_TOKENS)
}
