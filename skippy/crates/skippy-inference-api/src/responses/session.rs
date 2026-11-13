//! Transport-independent state for a persistent Responses connection.
//!
//! The session validates `response.create`, owns the connection-local latest
//! response, and prepares the same [`ResponsesTurn`] and generation request
//! used by HTTP. Socket framing, queueing, and routing remain adapter-owned so
//! Mesh can reuse this state without tunnelling a WebSocket to a worker.

use std::{sync::Arc, time::Duration};

use serde::Serialize;
use serde_json::Value;

use super::{
    CallerScope, ResponseAdapterMode, ResponseContinuation, ResponseStore, ResponsesTurn,
    normalize_openai_compat_request, store::ResponseSessionCache,
};
use crate::{
    AgentSessionIdentity, AgentSessionSource, ChatCompletionRequest, InferenceError,
    generation::GenerationRequest,
};

/// The protocol and memory bounds for one persistent Responses connection.
#[derive(Debug, Clone)]
pub struct ResponsesSessionConfig {
    max_pending_turns: usize,
    max_outbound_events: usize,
    max_transcript_bytes: usize,
    connection_lifetime: Duration,
}

impl ResponsesSessionConfig {
    /// The specification's absolute connection lifetime.
    pub const MAX_CONNECTION_LIFETIME: Duration = Duration::from_secs(60 * 60);
    pub const DEFAULT_MAX_PENDING_TURNS: usize = 8;
    pub const DEFAULT_MAX_OUTBOUND_EVENTS: usize = 64;
    pub const DEFAULT_MAX_TRANSCRIPT_BYTES: usize = 16 * 1024 * 1024;

    pub fn with_max_pending_turns(mut self, max_pending_turns: usize) -> Self {
        self.max_pending_turns = max_pending_turns.max(1);
        self
    }

    pub fn with_max_outbound_events(mut self, max_outbound_events: usize) -> Self {
        self.max_outbound_events = max_outbound_events.max(1);
        self
    }

    pub fn with_max_transcript_bytes(mut self, max_transcript_bytes: usize) -> Self {
        self.max_transcript_bytes = max_transcript_bytes;
        self
    }

    /// Shorter lifetimes are useful to embedders and tests; the public limit
    /// can never be raised beyond the specification's 60 minutes.
    pub fn with_connection_lifetime(mut self, connection_lifetime: Duration) -> Self {
        self.connection_lifetime = connection_lifetime.min(Self::MAX_CONNECTION_LIFETIME);
        self
    }

    pub fn max_pending_turns(&self) -> usize {
        self.max_pending_turns
    }

    pub fn max_outbound_events(&self) -> usize {
        self.max_outbound_events
    }

    pub fn max_transcript_bytes(&self) -> usize {
        self.max_transcript_bytes
    }

    pub fn connection_lifetime(&self) -> Duration {
        self.connection_lifetime
    }
}

impl Default for ResponsesSessionConfig {
    fn default() -> Self {
        Self {
            max_pending_turns: Self::DEFAULT_MAX_PENDING_TURNS,
            max_outbound_events: Self::DEFAULT_MAX_OUTBOUND_EVENTS,
            max_transcript_bytes: Self::DEFAULT_MAX_TRANSCRIPT_BYTES,
            connection_lifetime: Self::MAX_CONNECTION_LIFETIME,
        }
    }
}

/// A normalized Responses turn ready for an ingress-owned executor.
#[derive(Debug)]
pub struct PreparedResponsesTurn {
    turn: Arc<ResponsesTurn>,
    request: GenerationRequest,
    trusted_agent_session: bool,
    previous_response_id: Option<String>,
    response_adapter: ResponseAdapterMode,
}

impl PreparedResponsesTurn {
    pub fn turn(&self) -> &Arc<ResponsesTurn> {
        &self.turn
    }

    pub fn request(&self) -> &GenerationRequest {
        &self.request
    }

    pub fn previous_response_id(&self) -> Option<&str> {
        self.previous_response_id.as_deref()
    }

    pub fn response_adapter(&self) -> ResponseAdapterMode {
        self.response_adapter
    }

    pub fn has_trusted_agent_session(&self) -> bool {
        self.trusted_agent_session
    }

    /// Transfer the prepared turn and expanded generation request to an
    /// ingress-owned router or executor.
    pub fn into_parts(self) -> (Arc<ResponsesTurn>, GenerationRequest) {
        (self.turn, self.request)
    }

    pub(crate) fn into_execution_parts(
        self,
    ) -> (Arc<ResponsesTurn>, GenerationRequest, bool, Option<String>) {
        (
            self.turn,
            self.request,
            self.trusted_agent_session,
            self.previous_response_id,
        )
    }
}

/// Connection-local continuation state, independent of any socket library.
#[derive(Debug, Clone)]
pub struct ResponsesSession {
    config: ResponsesSessionConfig,
    shared_store: Option<Arc<ResponseStore>>,
    caller_scope: Option<CallerScope>,
    local: Arc<ResponseSessionCache>,
}

impl ResponsesSession {
    pub fn new(
        config: ResponsesSessionConfig,
        shared_store: Option<Arc<ResponseStore>>,
        caller_scope: Option<CallerScope>,
    ) -> Self {
        let local = Arc::new(ResponseSessionCache::new(config.max_transcript_bytes));
        Self {
            config,
            shared_store,
            caller_scope,
            local,
        }
    }

    pub fn config(&self) -> &ResponsesSessionConfig {
        &self.config
    }

    /// Validate and normalize one WebSocket `response.create` event.
    ///
    /// Transport-only fields are rejected by presence, even when `null` or
    /// `false`. Streaming is then enabled internally so the ordinary
    /// Responses normalization and backend boundary remain authoritative.
    pub fn prepare_create(
        &self,
        mut event: Value,
        header_session: Option<AgentSessionIdentity>,
    ) -> Result<PreparedResponsesTurn, InferenceError> {
        let referenced = event
            .get("previous_response_id")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let result = self.prepare_create_inner(&mut event, header_session, referenced.clone());
        if result.is_err() {
            self.evict_referenced(referenced.as_deref());
        }
        result
    }

    fn prepare_create_inner(
        &self,
        event: &mut Value,
        header_session: Option<AgentSessionIdentity>,
        previous_response_id: Option<String>,
    ) -> Result<PreparedResponsesTurn, InferenceError> {
        let object = event.as_object_mut().ok_or_else(|| {
            InferenceError::invalid_request("WebSocket message must be a JSON object")
        })?;
        if object.get("type").and_then(Value::as_str) != Some("response.create") {
            return Err(InferenceError::invalid_request(
                "WebSocket message type must be \"response.create\"",
            )
            .with_param("type"));
        }
        for field in ["stream", "stream_options", "background"] {
            if object.contains_key(field) {
                return Err(InferenceError::invalid_request(format!(
                    "{field} must not be sent on a Responses WebSocket request"
                ))
                .with_param(field));
            }
        }
        object.remove("type");
        let allow_shared_fallback = match object.get("store") {
            None => true,
            Some(Value::Bool(store)) => *store,
            Some(_) => {
                return Err(
                    InferenceError::invalid_request("store must be true or false")
                        .with_param("store"),
                );
            }
        };
        object.insert("stream".into(), Value::Bool(true));
        let continuation = ResponseContinuation::for_session(
            self.shared_store.clone(),
            self.caller_scope.clone(),
            self.local.clone(),
            allow_shared_fallback,
        );
        prepare_request(
            std::mem::take(event),
            continuation,
            header_session,
            previous_response_id,
        )
    }

    /// Evict only the referenced connection-local response. The caller-scoped
    /// shared store is deliberately untouched.
    pub fn evict_referenced(&self, previous_response_id: Option<&str>) {
        if let Some(id) = previous_response_id {
            self.local.evict(id);
        }
    }
}

pub(crate) fn prepare_request(
    mut value: Value,
    continuation: ResponseContinuation,
    header_session: Option<AgentSessionIdentity>,
    previous_response_id: Option<String>,
) -> Result<PreparedResponsesTurn, InferenceError> {
    let turn = Arc::new(ResponsesTurn::prepare(&mut value, continuation)?);
    let normalization = normalize_openai_compat_request("/v1/responses", &mut value)?;
    let request: ChatCompletionRequest = serde_json::from_value(value).map_err(|error| {
        InferenceError::invalid_request(format!("invalid Responses request: {error}"))
    })?;
    let trusted_agent_session = header_session.is_some();
    let mut request = GenerationRequest::from(request);
    let responses_session = normalization
        .agent_session_id
        .map(|id| AgentSessionIdentity::new(id, AgentSessionSource::ResponsesConversation))
        .transpose()?;
    request.set_agent_session(resolve_agent_session(header_session, responses_session)?);
    request.validate()?;
    Ok(PreparedResponsesTurn {
        turn,
        request,
        trusted_agent_session,
        previous_response_id,
        response_adapter: normalization.response_adapter,
    })
}

fn resolve_agent_session(
    header: Option<AgentSessionIdentity>,
    protocol: Option<AgentSessionIdentity>,
) -> Result<Option<AgentSessionIdentity>, InferenceError> {
    match (header, protocol) {
        (Some(header), Some(protocol)) if header.id() != protocol.id() => {
            Err(InferenceError::invalid_request(
                "trusted agent-session header conflicts with Responses conversation identity",
            ))
        }
        (Some(header), _) => Ok(Some(header)),
        (None, protocol) => Ok(protocol),
    }
}

/// The required WebSocket error envelope.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WebSocketErrorEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    status: u16,
    error: crate::errors::ErrorBody,
}

impl WebSocketErrorEvent {
    pub fn from_error(error: &InferenceError) -> Self {
        Self {
            event_type: "error",
            status: error.status().as_u16(),
            error: error.body().error,
        }
    }

    pub fn into_value(self) -> Value {
        serde_json::to_value(self).expect("WebSocket error event serializes")
    }
}
