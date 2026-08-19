//! Host-owned bridge from OpenAI frontend lifecycle boundaries to logging.
//!
//! The frontend emits bounded metadata. This adapter owns the matching request
//! guards and hands the production transport a narrow, body-only capture view.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::Instant,
};

use mesh_llm_events::logging::{
    events::{LifecycleEvent, TokenUsage},
    identifiers::{AttemptId, RequestId},
    replay::ReplayChannel,
};
use openai_frontend::{
    OpenAiBackendOperation, OpenAiFailure, OpenAiLifecycleContext, OpenAiLifecycleEvent,
    OpenAiLifecycleObserver, OpenAiRejection, OpenAiTerminalResult, OpenAiUsage,
};

use super::{
    ArtifactUnavailableReason, LifecycleGuard, LoggingService, ProxyAttemptFinish,
    RawMeshLifecycleOwners, RawMeshProxyAttempt, RawMeshRequestLifecycle, RequestSummaryMetadata,
    TerminalOutcome,
};

/// Fixed upper bound for requests owned by one embedded frontend observer.
const MAX_TRACKED_REQUESTS: usize = 1_024;
const LEGACY_OBSERVABILITY_TARGET: &str = "mesh_openai_observability";

/// The host-owned lifecycle attachment for one parsed OpenAI ingress request.
///
/// The attachment keeps terminal ownership at the ingress boundary. Routing
/// receives only the [`OpenAiRouteObserver`] view below, which can publish
/// bounded route/attempt metadata but cannot admit or terminalize a request.
pub(crate) struct OpenAiLifecycleAttachment {
    parent: Option<RawMeshRequestLifecycle>,
    capture: Option<Arc<dyn OpenAiArtifactCapture>>,
    terminalized: bool,
}

/// Narrow production artifact boundary for a parsed OpenAI request. The
/// transport can only submit body bytes and fixed artifact kinds; it never
/// receives filesystem or store access.
pub(crate) trait OpenAiArtifactCapture: Send + Sync {
    fn body_limit_bytes(&self) -> usize {
        256 * 1024
    }

    fn capture_body(
        &self,
        request_id: RequestId,
        kind: &'static str,
        content: &[u8],
        media_kind: Option<&str>,
    );
    fn capture_unavailable(
        &self,
        request_id: RequestId,
        kind: &'static str,
        reason: ArtifactUnavailableReason,
    );
}

/// Request-local bounded assembly for a client-visible SSE response body.
/// The configured per-artifact limit is enforced while frames arrive, before
/// the persistence queue is offered any body bytes.
pub(crate) struct OpenAiStreamArtifactCapture {
    capture: Arc<dyn OpenAiArtifactCapture>,
    request_id: RequestId,
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
    finalized: bool,
}

impl OpenAiStreamArtifactCapture {
    fn new(capture: Arc<dyn OpenAiArtifactCapture>, request_id: RequestId, limit: usize) -> Self {
        Self {
            capture,
            request_id,
            bytes: Vec::new(),
            limit,
            overflowed: false,
            finalized: false,
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        if self.overflowed {
            return;
        }
        let Some(next_len) = self.bytes.len().checked_add(bytes.len()) else {
            self.overflowed = true;
            self.bytes.clear();
            return;
        };
        if next_len > self.limit {
            self.overflowed = true;
            self.bytes.clear();
            return;
        }
        self.bytes.extend_from_slice(bytes);
    }

    fn complete(mut self) {
        self.finalized = true;
        if self.overflowed {
            self.capture.capture_unavailable(
                self.request_id,
                "response",
                ArtifactUnavailableReason::CaptureContentLimitExceeded,
            );
        } else {
            self.capture.capture_body(
                self.request_id,
                "response",
                &self.bytes,
                Some("text/event-stream"),
            );
        }
    }
}

impl Drop for OpenAiStreamArtifactCapture {
    fn drop(&mut self) {
        if !self.finalized {
            self.capture.capture_unavailable(
                self.request_id,
                "response",
                ArtifactUnavailableReason::StreamingResponseNotAssembled,
            );
        }
    }
}

/// Metadata-only route view handed to downstream dispatch code.
///
/// An empty view is the normal fail-open value when logging is disabled,
/// retired, or bounded admission cannot allocate a parent.
#[derive(Clone, Copy, Default)]
pub(crate) struct OpenAiRouteObserver<'a> {
    parent: Option<&'a RawMeshRequestLifecycle>,
    capture: Option<&'a Arc<dyn OpenAiArtifactCapture>>,
    request_id: Option<RequestId>,
}

/// An existing lifecycle attempt plus its private logging timestamp.
///
/// Downstream transport code can only finish this through its observer; it
/// cannot terminalize the request parent or enqueue arbitrary proxy records.
pub(crate) struct OpenAiRouteAttempt(RawMeshProxyAttempt);

impl OpenAiLifecycleAttachment {
    pub(crate) fn new(parent: Option<RawMeshRequestLifecycle>) -> Self {
        Self {
            parent,
            capture: None,
            terminalized: false,
        }
    }

    pub(crate) fn with_capture(
        parent: Option<RawMeshRequestLifecycle>,
        capture: Arc<dyn OpenAiArtifactCapture>,
    ) -> Self {
        Self {
            parent,
            capture: Some(capture),
            terminalized: false,
        }
    }

    pub(crate) fn unowned() -> Self {
        Self::new(None)
    }

    pub(crate) fn owns_parent(&self) -> bool {
        self.parent.is_some()
    }

    pub(crate) fn route_observer(&self) -> OpenAiRouteObserver<'_> {
        OpenAiRouteObserver {
            parent: self.parent.as_ref(),
            capture: self.capture.as_ref(),
            request_id: self
                .parent
                .as_ref()
                .map(RawMeshRequestLifecycle::request_id),
        }
    }

    /// Capture only the decoded request body. Raw HTTP headers are never made
    /// available here, so credentials cannot enter artifact capture.
    pub(crate) fn capture_request_body(&self, content: &[u8], media_kind: Option<&str>) {
        if let (Some(parent), Some(capture)) = (self.parent.as_ref(), self.capture.as_ref()) {
            // The parsed body is the only request data admitted here. Do not
            // carry raw request headers into artifact persistence. The caller
            // may provide only the closed, ingress-validated semantic media
            // kind for a recognized JSON endpoint; opaque bodies stay opaque.
            capture.capture_body(parent.request_id(), "request", content, media_kind);
        }
    }

    /// Terminalize exactly once from the owning ingress boundary.
    pub(crate) fn terminal(&mut self, outcome: TerminalOutcome) {
        if self.terminalized {
            return;
        }
        self.terminalized = true;
        if let Some(parent) = self.parent.as_ref() {
            parent.terminal(outcome);
        }
    }
}

impl Drop for OpenAiLifecycleAttachment {
    fn drop(&mut self) {
        if !self.terminalized {
            self.terminal(TerminalOutcome::Dropped(Some(
                "openai_ingress_scope_dropped".into(),
            )));
        }
    }
}

impl<'a> OpenAiRouteObserver<'a> {
    #[cfg(test)]
    pub(crate) fn capture_test_observer(
        request_id: RequestId,
        capture: &'a Arc<dyn OpenAiArtifactCapture>,
    ) -> Self {
        Self {
            parent: None,
            capture: Some(capture),
            request_id: Some(request_id),
        }
    }

    /// Capture a client-visible non-streaming response body. The capture
    /// implementation applies the canonical redactor and configured byte
    /// limit; HTTP headers are intentionally never passed here.
    pub(crate) fn capture_response_body(&self, content: &[u8], media_kind: Option<&str>) {
        if let (Some(request_id), Some(capture)) = (self.request_id, self.capture) {
            capture.capture_body(request_id, "response", content, media_kind);
        }
    }

    /// Persist an explicit state marker when a streaming response cannot be
    /// represented as a bounded semantic payload. This is deliberately not a
    /// fabricated status-only response body.
    pub(crate) fn capture_response_unavailable(&self, reason: ArtifactUnavailableReason) {
        if let (Some(request_id), Some(capture)) = (self.request_id, self.capture) {
            capture.capture_unavailable(request_id, "response", reason);
        }
    }

    pub(crate) fn begin_stream_response_capture(&self) -> Option<OpenAiStreamArtifactCapture> {
        let capture = self.capture?;
        let request_id = self.request_id?;
        Some(OpenAiStreamArtifactCapture::new(
            Arc::clone(capture),
            request_id,
            capture.body_limit_bytes(),
        ))
    }

    pub(crate) fn complete_stream_response_capture(
        &self,
        capture: Option<OpenAiStreamArtifactCapture>,
    ) {
        let Some(capture) = capture else {
            return;
        };
        capture.complete();
    }

    pub(crate) fn abandon_stream_response_capture(
        &self,
        capture: Option<OpenAiStreamArtifactCapture>,
    ) {
        drop(capture);
    }

    pub(crate) fn route_selected(&self, model: Option<&str>) {
        if let Some(parent) = self.parent {
            parent.route_selected(model);
        }
    }

    /// Record one bounded route selection with provider/engine metadata.
    ///
    /// The ingress owner remains responsible for the parent lifecycle; this
    /// observer only exposes the metadata-only route boundary to downstream
    /// transports. The raw lifecycle owner bounds and sanitizes both labels.
    pub(crate) fn route_selected_with_metadata(
        &self,
        model: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
    ) {
        if let Some(parent) = self.parent {
            parent.route_selected_with_metadata(model, provider, engine);
        }
    }

    pub(crate) fn stream_started(&self, model: Option<&str>) {
        if let Some(parent) = self.parent {
            parent.stream_started(model);
        }
    }

    /// Record the first bounded stream chunk. The canonical event envelope
    /// has no separate first-token variant, so the first `stream_chunk` marks
    /// that boundary without capturing token text or usage.
    pub(crate) fn stream_first_token(&self) {
        if let Some(parent) = self.parent {
            parent.stream_first_token();
        }
    }

    pub(crate) fn stream_chunk(&self) {
        if let Some(parent) = self.parent {
            parent.stream_chunk();
        }
    }

    pub(crate) fn stream_completed(
        &self,
        usage: Option<mesh_llm_events::logging::events::TokenUsage>,
    ) {
        if let Some(parent) = self.parent {
            parent.stream_completed(usage);
        }
    }

    /// Record a bounded static stream error/cancellation label.
    pub(crate) fn stream_error(&self, label: &'static str) {
        if let Some(parent) = self.parent {
            parent.stream_error(label);
        }
    }

    pub(crate) fn stream_cancelled(&self) {
        if let Some(parent) = self.parent {
            parent.stream_cancelled();
        }
    }

    pub(crate) fn start_attempt(&self) -> Option<AttemptId> {
        self.parent.map(RawMeshRequestLifecycle::start_attempt)
    }

    /// Start a lifecycle attempt that may later persist one bounded proxy
    /// record. Empty observers remain fully fail-open.
    pub(crate) fn start_proxy_attempt(&self) -> Option<OpenAiRouteAttempt> {
        self.parent
            .map(RawMeshRequestLifecycle::start_proxy_attempt)
            .map(OpenAiRouteAttempt)
    }

    pub(crate) fn complete_attempt(&self, attempt_id: Option<AttemptId>, status_code: u16) {
        if let (Some(parent), Some(attempt_id)) = (self.parent, attempt_id) {
            parent.complete_attempt(attempt_id, status_code);
        }
    }

    pub(crate) fn fail_attempt(&self, attempt_id: Option<AttemptId>, label: &'static str) {
        if let (Some(parent), Some(attempt_id)) = (self.parent, attempt_id) {
            parent.fail_attempt(attempt_id, label);
        }
    }

    /// Finish one transport attempt and enqueue a metadata-only proxy record.
    /// This retains the ingress attachment as the sole parent terminal owner.
    pub(crate) fn finish_proxy_attempt(
        &self,
        attempt: Option<OpenAiRouteAttempt>,
        finish: ProxyAttemptFinish,
    ) {
        if let (Some(parent), Some(OpenAiRouteAttempt(attempt))) = (self.parent, attempt) {
            parent.finish_proxy_attempt(attempt, finish);
        }
    }
}

/// Metadata-only OpenAI frontend lifecycle observer owned by the host runtime.
pub(crate) struct OpenAiLifecycleLoggingAdapter {
    service: Arc<LoggingService>,
    raw_mesh_owners: Arc<RawMeshLifecycleOwners>,
    tracked: Mutex<TrackedRequests>,
}

#[derive(Default)]
struct TrackedRequests {
    requests: HashMap<RequestId, TrackedRequest>,
    insertion_order: VecDeque<RequestId>,
}

enum TrackedRequest {
    Admitting,
    Active(ActiveRequest),
    Terminal,
}

struct ActiveRequest {
    /// `None` means raw Mesh ingress owns canonical lifecycle state. The
    /// embedded frontend still emits the legacy custody projection, but must
    /// not register or terminalize a second canonical request.
    guard: Option<LifecycleGuard>,
    legacy: Arc<LegacyCustody>,
    started_at: Instant,
    operation: OpenAiBackendOperation,
    agent_session_id: Option<String>,
    agent_session_source: Option<String>,
    backend_operation: Option<OpenAiBackendOperation>,
    backend_attempt: Option<(OpenAiBackendOperation, AttemptId)>,
    backend_stream_first_item: bool,
    usage: Option<OpenAiUsage>,
}

type LegacyAction = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct LegacyCustody {
    state: Mutex<LegacyCustodyState>,
}

#[derive(Default)]
struct LegacyCustodyState {
    actions: VecDeque<LegacyAction>,
    draining: bool,
    started: bool,
    terminalized: bool,
}

impl LegacyCustody {
    /// Queue the two events that establish legacy custody as one atomic
    /// per-request action sequence. The caller may hold the adapter's map
    /// mutex while this method queues; logging itself is always drained after
    /// that lock is released.
    fn start(
        &self,
        request_id: RequestId,
        operation: OpenAiBackendOperation,
        agent_session_id: Option<String>,
        agent_session_source: Option<String>,
    ) -> bool {
        let backend_session_id = agent_session_id.clone();
        let backend_session_source = agent_session_source.clone();
        self.start_with(
            Box::new(move || {
                log_legacy_request_started(
                    request_id,
                    operation,
                    agent_session_id.as_deref(),
                    agent_session_source.as_deref(),
                );
            }),
            Box::new(move || {
                log_legacy_backend_started(
                    request_id,
                    operation,
                    backend_session_id.as_deref(),
                    backend_session_source.as_deref(),
                );
            }),
        )
    }

    fn start_with(&self, request_started: LegacyAction, backend_started: LegacyAction) -> bool {
        let mut state = lock_recover(&self.state);
        if state.started || state.terminalized {
            return false;
        }
        state.started = true;
        state.actions.push_back(request_started);
        state.actions.push_back(backend_started);
        Self::begin_draining(&mut state)
    }

    fn enqueue<F>(&self, action: F) -> bool
    where
        F: FnOnce() + Send + 'static,
    {
        let mut state = lock_recover(&self.state);
        if state.terminalized {
            return false;
        }
        state.actions.push_back(Box::new(action));
        Self::begin_draining(&mut state)
    }

    fn finish<F>(&self, action: F) -> bool
    where
        F: FnOnce() + Send + 'static,
    {
        let mut state = lock_recover(&self.state);
        if state.terminalized {
            return false;
        }
        state.terminalized = true;
        if !state.started {
            return false;
        }
        state.actions.push_back(Box::new(action));
        Self::begin_draining(&mut state)
    }

    fn begin_draining(state: &mut LegacyCustodyState) -> bool {
        if state.draining {
            false
        } else {
            state.draining = true;
            true
        }
    }

    fn drain(&self) {
        loop {
            let action = {
                let mut state = lock_recover(&self.state);
                match state.actions.pop_front() {
                    Some(action) => action,
                    None => {
                        state.draining = false;
                        return;
                    }
                }
            };
            action();
        }
    }
}

impl ActiveRequest {
    fn new(guard: Option<LifecycleGuard>, context: &OpenAiLifecycleContext) -> Self {
        Self {
            guard,
            legacy: Arc::new(LegacyCustody::default()),
            started_at: Instant::now(),
            operation: route_operation(context.route),
            agent_session_id: context.agent_session_id.clone(),
            agent_session_source: context.agent_session_source.clone(),
            backend_operation: None,
            backend_attempt: None,
            backend_stream_first_item: false,
            usage: None,
        }
    }
}

impl OpenAiLifecycleLoggingAdapter {
    pub(crate) fn new(
        service: Arc<LoggingService>,
        raw_mesh_owners: Arc<RawMeshLifecycleOwners>,
    ) -> Self {
        Self {
            service,
            raw_mesh_owners,
            tracked: Mutex::new(TrackedRequests::default()),
        }
    }

    fn admit(&self, context: &OpenAiLifecycleContext) {
        let request_id = context.request_id;
        // Both local raw ownership and remote-tunnel suppression mean that a
        // canonical parent already exists elsewhere. Retain a legacy-only
        // record here so the target's authenticated frontend events remain
        // observable without creating a second canonical lifecycle.
        let raw_mesh_owned = self.raw_mesh_owners.is_claimed(request_id);

        // Reserve the request under the small state mutex, then perform
        // service registration outside it. This prevents persistence and
        // tracing work from blocking every other OpenAI request.
        {
            let mut tracked = lock_recover(&self.tracked);
            if tracked.requests.contains_key(&request_id) || !tracked.make_room() {
                return;
            }
            tracked
                .requests
                .insert(request_id, TrackedRequest::Admitting);
            tracked.insertion_order.push_back(request_id);
        }

        let guard = if raw_mesh_owned {
            None
        } else {
            Some(
                self.service
                    .register_request_with_metadata(
                        request_id,
                        RequestSummaryMetadata::from_openai_frontend_route(context.route),
                    )
                    .0,
            )
        };
        let active = ActiveRequest::new(guard, context);

        let mut tracked = lock_recover(&self.tracked);
        let Some(entry) = tracked.requests.get_mut(&request_id) else {
            return;
        };
        if !matches!(entry, TrackedRequest::Admitting) {
            return;
        }
        *entry = TrackedRequest::Active(active);
    }

    fn backend_dispatched(&self, request_id: RequestId, operation: OpenAiBackendOperation) {
        let guard = {
            let tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get(&request_id) else {
                return;
            };
            active.guard.clone()
        };

        let metadata = RequestSummaryMetadata::from_parts(
            None,
            None,
            Some("openai_frontend"),
            Some(operation_label(operation)),
        );
        let attempt_id = guard.as_ref().map(|guard| {
            self.service
                .merge_request_metadata(request_id, metadata.clone());
            self.enqueue_operation_event(
                request_id,
                LifecycleEvent::RouteSelected {
                    model: None,
                    provider: metadata.provider().map(str::to_owned),
                    engine: metadata.engine().map(str::to_owned),
                },
            );
            self.service.start_attempt(request_id, guard)
        });

        let legacy_drain = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.backend_operation.is_some() {
                return;
            }
            active.operation = operation;
            active.backend_operation = Some(operation);
            active.backend_attempt = attempt_id.map(|attempt_id| (operation, attempt_id));
            let legacy = Arc::clone(&active.legacy);
            let should_drain = legacy.start(
                request_id,
                operation,
                active.agent_session_id.clone(),
                active.agent_session_source.clone(),
            );
            (legacy, should_drain)
        };
        if legacy_drain.1 {
            legacy_drain.0.drain();
        }
    }

    fn backend_terminal(
        &self,
        request_id: RequestId,
        operation: OpenAiBackendOperation,
        result: OpenAiTerminalResult,
    ) {
        let attempt = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.backend_operation == Some(operation) {
                active.backend_operation = None;
                let backend_attempt = active.backend_attempt.take();
                let elapsed = active.started_at.elapsed();
                let agent_session_id = active.agent_session_id.clone();
                let agent_session_source = active.agent_session_source.clone();
                let legacy = Arc::clone(&active.legacy);
                let should_drain = match result {
                    OpenAiTerminalResult::Completed { .. }
                    | OpenAiTerminalResult::CompletedWithUsage { .. } => {
                        legacy.enqueue(move || {
                            log_legacy_backend_returned(
                                request_id,
                                operation,
                                elapsed,
                                agent_session_id.as_deref(),
                                agent_session_source.as_deref(),
                            );
                        })
                    }
                    OpenAiTerminalResult::Failed { failure, .. }
                        if failure == OpenAiFailure::Timeout =>
                    {
                        legacy.enqueue(move || {
                            log_legacy_backend_timeout(
                                request_id,
                                operation,
                                elapsed,
                                agent_session_id.as_deref(),
                                agent_session_source.as_deref(),
                            );
                        })
                    }
                    OpenAiTerminalResult::Failed { .. } => legacy.enqueue(move || {
                        log_legacy_backend_error(
                            request_id,
                            operation,
                            elapsed,
                            agent_session_id.as_deref(),
                            agent_session_source.as_deref(),
                        );
                    }),
                };
                Some((
                    backend_attempt.map(|(_, attempt_id)| attempt_id),
                    legacy,
                    should_drain,
                ))
            } else {
                None
            }
        };
        let Some((attempt_id, legacy, should_drain)) = attempt else {
            return;
        };
        match result {
            OpenAiTerminalResult::Completed { status_code }
            | OpenAiTerminalResult::CompletedWithUsage { status_code, .. } => {
                if let Some(attempt_id) = attempt_id {
                    self.service
                        .complete_attempt(request_id, attempt_id, Some(status_code));
                }
            }
            OpenAiTerminalResult::Failed { failure, .. } => {
                if let Some(attempt_id) = attempt_id {
                    self.service.fail_attempt(
                        request_id,
                        attempt_id,
                        failure_label(failure).to_owned(),
                    );
                }
            }
        }
        if should_drain {
            legacy.drain();
        }
    }

    fn backend_stream_first_item(&self, request_id: RequestId, operation: OpenAiBackendOperation) {
        let first_item = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.backend_stream_first_item {
                None
            } else {
                active.backend_stream_first_item = true;
                let elapsed = active.started_at.elapsed();
                let agent_session_id = active.agent_session_id.clone();
                let agent_session_source = active.agent_session_source.clone();
                let legacy = Arc::clone(&active.legacy);
                let should_drain = legacy.enqueue(move || {
                    log_legacy_stream_first_item(
                        request_id,
                        operation,
                        elapsed,
                        agent_session_id.as_deref(),
                        agent_session_source.as_deref(),
                    );
                });
                Some((active.guard.is_some(), legacy, should_drain))
            }
        };
        if let Some((canonical, legacy, should_drain)) = first_item {
            if canonical {
                self.enqueue_operation_event(request_id, LifecycleEvent::BackendStreamFirstItem);
            }
            if should_drain {
                legacy.drain();
            }
        }
    }

    fn response_completed(&self, request_id: RequestId, usage: OpenAiUsage) {
        let metadata = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.usage.is_some() {
                None
            } else {
                active.usage = Some(usage);
                let elapsed = active.started_at.elapsed();
                let operation = active.operation;
                let agent_session_id = active.agent_session_id.clone();
                let agent_session_source = active.agent_session_source.clone();
                let legacy = Arc::clone(&active.legacy);
                let should_drain = legacy.enqueue(move || {
                    log_legacy_response_completed(
                        request_id,
                        operation,
                        elapsed,
                        usage,
                        agent_session_id.as_deref(),
                        agent_session_source.as_deref(),
                    );
                });
                Some((active.guard.is_some(), legacy, should_drain))
            }
        };
        if let Some((canonical, legacy, should_drain)) = metadata {
            if canonical {
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::UsageRecorded {
                        prompt_tokens: Some(u64::from(usage.prompt_tokens)),
                        cached_prompt_tokens: Some(u64::from(usage.cached_tokens)),
                        completion_tokens: Some(u64::from(usage.completion_tokens)),
                        total_tokens: Some(u64::from(usage.total_tokens)),
                    },
                );
            }
            if should_drain {
                legacy.drain();
            }
        }
    }

    fn stream_terminal(&self, request_id: RequestId, result: OpenAiTerminalResult) {
        let (usage, canonical, legacy, should_drain) = {
            let tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get(&request_id) else {
                return;
            };
            let usage = active.usage;
            let canonical = active.guard.is_some();
            let legacy = Arc::clone(&active.legacy);
            let should_drain = match result {
                OpenAiTerminalResult::Failed { .. } => {
                    let elapsed = active.started_at.elapsed();
                    let operation = active.operation;
                    let agent_session_id = active.agent_session_id.clone();
                    let agent_session_source = active.agent_session_source.clone();
                    legacy.enqueue(move || {
                        log_legacy_stream_item_error(
                            request_id,
                            operation,
                            elapsed,
                            agent_session_id.as_deref(),
                            agent_session_source.as_deref(),
                        );
                    })
                }
                OpenAiTerminalResult::Completed { .. }
                | OpenAiTerminalResult::CompletedWithUsage { .. } => false,
            };
            (usage, canonical, legacy, should_drain)
        };
        match result {
            OpenAiTerminalResult::Completed { .. } => {
                let terminal_usage = usage.and_then(|value| {
                    TokenUsage::from_counts(
                        Some(u64::from(value.prompt_tokens)),
                        Some(u64::from(value.completion_tokens)),
                        Some(u64::from(value.total_tokens)),
                    )
                });
                if canonical {
                    self.enqueue_operation_event(
                        request_id,
                        LifecycleEvent::StreamCompleted {
                            tokens: usage.map(|value| u64::from(value.completion_tokens)),
                            usage: terminal_usage,
                        },
                    );
                }
            }
            OpenAiTerminalResult::CompletedWithUsage { usage, .. } => {
                if canonical {
                    self.enqueue_operation_event(
                        request_id,
                        LifecycleEvent::StreamCompleted {
                            tokens: usage.completion_tokens,
                            usage: Some(usage),
                        },
                    );
                }
            }
            OpenAiTerminalResult::Failed { failure, .. } => {
                if canonical {
                    self.enqueue_operation_event(
                        request_id,
                        LifecycleEvent::StreamError {
                            error: Some(failure_label(failure).to_owned()),
                        },
                    );
                }
            }
        }
        if should_drain {
            legacy.drain();
        }
    }

    fn stream_interrupted(&self, request_id: RequestId, label: &'static str) {
        let metadata = {
            let tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get(&request_id) else {
                return;
            };
            let legacy = Arc::clone(&active.legacy);
            let operation = active.operation;
            let elapsed = active.started_at.elapsed();
            let agent_session_id = active.agent_session_id.clone();
            let agent_session_source = active.agent_session_source.clone();
            let should_drain = legacy.enqueue(move || {
                log_legacy_stream_item_error(
                    request_id,
                    operation,
                    elapsed,
                    agent_session_id.as_deref(),
                    agent_session_source.as_deref(),
                );
            });
            (active.guard.is_some(), legacy, should_drain)
        };
        if metadata.0 {
            self.enqueue_operation_event(
                request_id,
                LifecycleEvent::StreamError {
                    error: Some(label.to_owned()),
                },
            );
        }
        if metadata.2 {
            metadata.1.drain();
        }
    }

    fn enqueue_operation_event(&self, request_id: RequestId, event: LifecycleEvent) {
        if let Ok(payload) = serde_json::to_string(&event) {
            let _ = self
                .service
                .enqueue_event(request_id, ReplayChannel::Operations, payload);
        }
    }

    fn terminal(&self, request_id: RequestId, outcome: TerminalOutcome) {
        let (guard, legacy, should_drain) = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(entry) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            let TrackedRequest::Active(active) = entry else {
                return;
            };
            let guard = active.guard.clone();
            let elapsed = active.started_at.elapsed();
            let operation = active.operation;
            let agent_session_id = active.agent_session_id.clone();
            let agent_session_source = active.agent_session_source.clone();
            let legacy = Arc::clone(&active.legacy);
            let outcome_label = legacy_outcome(&outcome);
            let should_drain = legacy.finish(move || {
                log_legacy_request_finished(
                    request_id,
                    operation,
                    elapsed,
                    outcome_label,
                    agent_session_id.as_deref(),
                    agent_session_source.as_deref(),
                );
            });
            *entry = TrackedRequest::Terminal;
            (guard, legacy, should_drain)
        };

        if should_drain {
            legacy.drain();
        }

        // A stale or externally-terminalized guard is harmless: request
        // serving and later frontend events must never depend on this write.
        if let Some(guard) = guard.as_ref() {
            let _ = self.service.transition_terminal(request_id, guard, outcome);
        }
    }

    #[cfg(test)]
    fn tracked_len(&self) -> usize {
        lock_recover(&self.tracked).requests.len()
    }
}

impl TrackedRequests {
    fn make_room(&mut self) -> bool {
        while self.requests.len() >= MAX_TRACKED_REQUESTS {
            let Some(oldest) = self.insertion_order.pop_front() else {
                return false;
            };
            if matches!(self.requests.get(&oldest), Some(TrackedRequest::Terminal)) {
                self.requests.remove(&oldest);
                continue;
            }
            self.insertion_order.push_front(oldest);
            return false;
        }
        true
    }

    fn is_active(&self, request_id: RequestId) -> bool {
        matches!(
            self.requests.get(&request_id),
            Some(TrackedRequest::Active(_))
        )
    }
}

impl OpenAiLifecycleObserver for OpenAiLifecycleLoggingAdapter {
    fn observe(&self, event: &OpenAiLifecycleEvent) {
        match event {
            OpenAiLifecycleEvent::Admitted { context } => self.admit(context),
            OpenAiLifecycleEvent::BackendDispatched { context, operation } => {
                self.backend_dispatched(context.request_id, *operation)
            }
            OpenAiLifecycleEvent::BackendTerminal {
                context,
                operation,
                result,
            } => self.backend_terminal(context.request_id, *operation, *result),
            OpenAiLifecycleEvent::StreamFirstItem {
                context, operation, ..
            } => self.backend_stream_first_item(context.request_id, *operation),
            OpenAiLifecycleEvent::ResponseCompleted { context, usage, .. } => {
                self.response_completed(context.request_id, *usage)
            }
            OpenAiLifecycleEvent::Rejected {
                context, rejection, ..
            } => self.terminal(
                context.request_id,
                TerminalOutcome::Rejected(Some(rejection_label(*rejection).into())),
            ),
            OpenAiLifecycleEvent::NonStreamTerminal { context, result } => {
                self.terminal(context.request_id, terminal_outcome(*result))
            }
            OpenAiLifecycleEvent::StreamTerminal { context, result } => {
                self.stream_terminal(context.request_id, *result);
                self.terminal(context.request_id, terminal_outcome(*result));
            }
            OpenAiLifecycleEvent::StreamCancelled { context } => {
                self.stream_interrupted(context.request_id, "stream_cancelled");
                self.terminal(
                    context.request_id,
                    TerminalOutcome::Cancelled(Some("stream_cancelled".into())),
                );
            }
            OpenAiLifecycleEvent::StreamDropped { context } => {
                self.stream_interrupted(context.request_id, "stream_dropped");
                self.terminal(
                    context.request_id,
                    TerminalOutcome::Dropped(Some("stream_dropped".into())),
                );
            }
            OpenAiLifecycleEvent::RequestCancelled { context } => self.terminal(
                context.request_id,
                TerminalOutcome::Cancelled(Some("request_cancelled".into())),
            ),
        }
    }
}

fn terminal_outcome(result: OpenAiTerminalResult) -> TerminalOutcome {
    match result {
        OpenAiTerminalResult::Completed { status_code } => {
            TerminalOutcome::CompletedWithStatus(status_code)
        }
        OpenAiTerminalResult::CompletedWithUsage { status_code, usage } => {
            TerminalOutcome::CompletedWithUsage { status_code, usage }
        }
        OpenAiTerminalResult::Failed {
            failure: OpenAiFailure::Cancelled,
            ..
        } => TerminalOutcome::Cancelled(Some("request_cancelled".into())),
        OpenAiTerminalResult::Failed {
            status_code,
            failure,
        } => TerminalOutcome::FailedWithStatus {
            error: failure_label(failure).into(),
            status_code,
        },
    }
}

const fn operation_label(operation: OpenAiBackendOperation) -> &'static str {
    match operation {
        OpenAiBackendOperation::Models => "models",
        OpenAiBackendOperation::ChatCompletion => "chat_completion",
        OpenAiBackendOperation::ChatCompletionStream => "chat_completion_stream",
        OpenAiBackendOperation::Completion => "completion",
        OpenAiBackendOperation::CompletionStream => "completion_stream",
        OpenAiBackendOperation::Responses => "responses",
        OpenAiBackendOperation::ResponsesStream => "responses_stream",
    }
}

const fn route_operation(route: openai_frontend::OpenAiFrontendRoute) -> OpenAiBackendOperation {
    match route {
        openai_frontend::OpenAiFrontendRoute::Completions => OpenAiBackendOperation::Completion,
        openai_frontend::OpenAiFrontendRoute::Responses => OpenAiBackendOperation::Responses,
        openai_frontend::OpenAiFrontendRoute::Models
        | openai_frontend::OpenAiFrontendRoute::Health
        | openai_frontend::OpenAiFrontendRoute::Healthz
        | openai_frontend::OpenAiFrontendRoute::Readyz => OpenAiBackendOperation::Models,
        openai_frontend::OpenAiFrontendRoute::ChatCompletions
        | openai_frontend::OpenAiFrontendRoute::Unknown => OpenAiBackendOperation::ChatCompletion,
    }
}

fn log_legacy_request_started(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let operation = operation_label(operation);
    if let (Some(agent_session_id), Some(agent_session_source)) =
        (agent_session_id, agent_session_source)
    {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "request_started",
            request_id = %request_id.as_ref(),
            operation,
            agent_session_id,
            agent_session_source,
            "OpenAI request accepted"
        );
    } else {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "request_started",
            request_id = %request_id.as_ref(),
            operation,
            "OpenAI request accepted"
        );
    }
}

fn log_legacy_backend_started(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let operation = operation_label(operation);
    let request_id = request_id.as_ref();
    if let (Some(agent_session_id), Some(agent_session_source)) =
        (agent_session_id, agent_session_source)
    {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "backend_started",
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            "OpenAI backend operation started"
        );
    } else {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "backend_started",
            request_id = %request_id,
            operation,
            "OpenAI backend operation started"
        );
    }
}

fn log_legacy_backend_returned(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    log_legacy_backend_result(
        "backend_returned",
        request_id,
        operation,
        elapsed,
        agent_session_id,
        agent_session_source,
    );
}

fn log_legacy_backend_error(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    log_legacy_backend_result(
        "backend_error",
        request_id,
        operation,
        elapsed,
        agent_session_id,
        agent_session_source,
    );
}

fn log_legacy_backend_timeout(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    log_legacy_backend_result(
        "backend_timeout",
        request_id,
        operation,
        elapsed,
        agent_session_id,
        agent_session_source,
    );
}

fn log_legacy_backend_result(
    event: &'static str,
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let operation = operation_label(operation);
    let request_id = request_id.as_ref();
    let elapsed_us = elapsed.as_micros() as u64;
    match (event, agent_session_id, agent_session_source) {
        ("backend_returned", Some(agent_session_id), Some(agent_session_source)) => tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event,
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            elapsed_us,
            "OpenAI backend operation returned"
        ),
        ("backend_timeout", Some(agent_session_id), Some(agent_session_source)) => tracing::warn!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event,
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            elapsed_us,
            "OpenAI backend operation timed out"
        ),
        ("backend_error", Some(agent_session_id), Some(agent_session_source)) => tracing::warn!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event,
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            elapsed_us,
            "OpenAI backend operation failed"
        ),
        ("backend_returned", _, _) => tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event,
            request_id = %request_id,
            operation,
            elapsed_us,
            "OpenAI backend operation returned"
        ),
        ("backend_timeout", _, _) => tracing::warn!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event,
            request_id = %request_id,
            operation,
            elapsed_us,
            "OpenAI backend operation timed out"
        ),
        _ => tracing::warn!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event,
            request_id = %request_id,
            operation,
            elapsed_us,
            "OpenAI backend operation failed"
        ),
    }
}

fn log_legacy_stream_first_item(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let request_id = request_id.as_ref();
    let operation = operation_label(operation);
    let time_to_first_item_us = elapsed.as_micros() as u64;
    if let (Some(agent_session_id), Some(agent_session_source)) =
        (agent_session_id, agent_session_source)
    {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "stream_first_item",
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            elapsed_us = time_to_first_item_us,
            time_to_first_item_us,
            "OpenAI stream emitted its first item"
        );
    } else {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "stream_first_item",
            request_id = %request_id,
            operation,
            elapsed_us = time_to_first_item_us,
            time_to_first_item_us,
            "OpenAI stream emitted its first item"
        );
    }
}

fn log_legacy_stream_item_error(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let operation = operation_label(operation);
    let request_id = request_id.as_ref();
    let elapsed_us = elapsed.as_micros() as u64;
    match (agent_session_id, agent_session_source) {
        (Some(agent_session_id), Some(agent_session_source)) => tracing::warn!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "stream_item_error",
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            elapsed_us,
            "OpenAI stream item failed"
        ),
        _ => tracing::warn!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "stream_item_error",
            request_id = %request_id,
            operation,
            elapsed_us,
            "OpenAI stream item failed"
        ),
    }
}

fn log_legacy_response_completed(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    usage: OpenAiUsage,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let operation = operation_label(operation);
    let request_id = request_id.as_ref();
    let elapsed_us = elapsed.as_micros() as u64;
    if let (Some(agent_session_id), Some(agent_session_source)) =
        (agent_session_id, agent_session_source)
    {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "response_completed",
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            elapsed_us,
            prompt_tokens = usage.prompt_tokens,
            cached_tokens = usage.cached_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "OpenAI response completed"
        );
    } else {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "response_completed",
            request_id = %request_id,
            operation,
            elapsed_us,
            prompt_tokens = usage.prompt_tokens,
            cached_tokens = usage.cached_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "OpenAI response completed"
        );
    }
}

fn log_legacy_request_finished(
    request_id: RequestId,
    operation: OpenAiBackendOperation,
    elapsed: std::time::Duration,
    outcome: &'static str,
    agent_session_id: Option<&str>,
    agent_session_source: Option<&str>,
) {
    let operation = operation_label(operation);
    let request_id = request_id.as_ref();
    let elapsed_us = elapsed.as_micros() as u64;
    if let (Some(agent_session_id), Some(agent_session_source)) =
        (agent_session_id, agent_session_source)
    {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "request_finished",
            request_id = %request_id,
            operation,
            agent_session_id,
            agent_session_source,
            outcome,
            elapsed_us,
            "OpenAI request finished"
        );
    } else {
        tracing::info!(
            target: LEGACY_OBSERVABILITY_TARGET,
            event = "request_finished",
            request_id = %request_id,
            operation,
            outcome,
            elapsed_us,
            "OpenAI request finished"
        );
    }
}

fn legacy_outcome(outcome: &TerminalOutcome) -> &'static str {
    match outcome {
        TerminalOutcome::Completed
        | TerminalOutcome::CompletedWithStatus(_)
        | TerminalOutcome::CompletedWithUsage { .. } => "success",
        TerminalOutcome::Failed(error) | TerminalOutcome::FailedWithStatus { error, .. }
            if error == "timeout" =>
        {
            "timeout"
        }
        TerminalOutcome::Failed(_) | TerminalOutcome::FailedWithStatus { .. } => "backend_error",
        TerminalOutcome::Rejected(_) | TerminalOutcome::RejectedWithStatus { .. } => "client_error",
        TerminalOutcome::Cancelled(_) => "cancelled",
        TerminalOutcome::Dropped(_) => "client_disconnect",
    }
}

const fn rejection_label(rejection: OpenAiRejection) -> &'static str {
    match rejection {
        OpenAiRejection::InvalidRequest => "invalid_request",
        OpenAiRejection::PayloadTooLarge => "payload_too_large",
        OpenAiRejection::MethodNotAllowed => "method_not_allowed",
        OpenAiRejection::NotFound => "not_found",
        OpenAiRejection::AdmissionDenied => "admission_denied",
    }
}

const fn failure_label(failure: OpenAiFailure) -> &'static str {
    match failure {
        OpenAiFailure::Backend => "backend",
        OpenAiFailure::Timeout => "timeout",
        OpenAiFailure::Internal => "internal",
        OpenAiFailure::Cancelled => "cancelled",
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        sync::{Arc, Mutex},
    };

    use mesh_llm_events::logging::{events::LifecycleEvent, identifiers::RequestId};
    use openai_frontend::{
        OpenAiBackendOperation, OpenAiFrontendRoute, OpenAiLifecycleContext, OpenAiLifecycleEvent,
        OpenAiRequestMethod, OpenAiTerminalResult, OpenAiUsage,
    };

    use super::*;

    #[derive(Default)]
    struct ArtifactCaptureProbe {
        bodies: Mutex<Vec<Vec<u8>>>,
        unavailable: Mutex<Vec<ArtifactUnavailableReason>>,
    }

    impl OpenAiArtifactCapture for ArtifactCaptureProbe {
        fn capture_body(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            content: &[u8],
            _media_kind: Option<&str>,
        ) {
            self.bodies.lock().unwrap().push(content.to_vec());
        }

        fn capture_unavailable(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            reason: ArtifactUnavailableReason,
        ) {
            self.unavailable.lock().unwrap().push(reason);
        }
    }

    #[test]
    fn stream_artifact_capture_enforces_the_limit_before_persistence() {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let capture: Arc<dyn OpenAiArtifactCapture> = probe.clone();
        let mut stream = OpenAiStreamArtifactCapture::new(capture, RequestId::new(), 4);

        stream.push(b"1234");
        stream.push(b"5");
        stream.complete();

        assert!(probe.bodies.lock().unwrap().is_empty());
        assert_eq!(
            *probe.unavailable.lock().unwrap(),
            vec![ArtifactUnavailableReason::CaptureContentLimitExceeded]
        );
    }

    #[test]
    fn dropped_stream_artifact_capture_records_an_explicit_unavailable_state() {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let capture: Arc<dyn OpenAiArtifactCapture> = probe.clone();
        let mut stream = OpenAiStreamArtifactCapture::new(capture, RequestId::new(), 64);

        stream.push(b"partial");
        drop(stream);

        assert!(probe.bodies.lock().unwrap().is_empty());
        assert_eq!(
            *probe.unavailable.lock().unwrap(),
            vec![ArtifactUnavailableReason::StreamingResponseNotAssembled]
        );
    }

    fn context(request_id: RequestId) -> OpenAiLifecycleContext {
        OpenAiLifecycleContext::new(
            request_id,
            OpenAiRequestMethod::Post,
            OpenAiFrontendRoute::ChatCompletions,
        )
    }

    #[derive(Clone)]
    struct TraceWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for TraceWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for TraceWriter {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn adapter() -> (Arc<LoggingService>, OpenAiLifecycleLoggingAdapter) {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let adapter = OpenAiLifecycleLoggingAdapter::new(
            Arc::clone(&service),
            Arc::new(RawMeshLifecycleOwners::default()),
        );
        (service, adapter)
    }

    #[test]
    fn legacy_custody_serializes_dispatch_and_terminal_race() {
        use std::thread;

        let custody = Arc::new(LegacyCustody::default());
        let order = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(std::sync::Barrier::new(2));
        let first_gate = Arc::clone(&gate);
        let first_order = Arc::clone(&order);
        assert!(custody.start_with(
            Box::new(move || {
                first_order.lock().unwrap().push("request_started");
                first_gate.wait();
            }),
            Box::new({
                let order = Arc::clone(&order);
                move || order.lock().unwrap().push("backend_started")
            }),
        ));

        let drain_custody = Arc::clone(&custody);
        let drain_thread = thread::spawn(move || drain_custody.drain());
        gate.wait();

        let finish_order = Arc::clone(&order);
        assert!(!custody.finish(move || { finish_order.lock().unwrap().push("request_finished") }));
        drain_thread.join().unwrap();
        assert_eq!(
            *order.lock().unwrap(),
            vec!["request_started", "backend_started", "request_finished"]
        );
    }

    #[test]
    fn legacy_custody_serializes_response_and_terminal_race() {
        use std::thread;

        let custody = Arc::new(LegacyCustody::default());
        let order = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(std::sync::Barrier::new(2));
        let first_gate = Arc::clone(&gate);
        let first_order = Arc::clone(&order);
        assert!(custody.start_with(
            Box::new(move || {
                first_order.lock().unwrap().push("request_started");
                first_gate.wait();
            }),
            Box::new({
                let order = Arc::clone(&order);
                move || order.lock().unwrap().push("backend_started")
            }),
        ));

        let drain_custody = Arc::clone(&custody);
        let drain_thread = thread::spawn(move || drain_custody.drain());
        gate.wait();

        let response_order = Arc::clone(&order);
        assert!(
            !custody.enqueue(move || { response_order.lock().unwrap().push("response_completed") })
        );
        let finish_order = Arc::clone(&order);
        assert!(!custody.finish(move || { finish_order.lock().unwrap().push("request_finished") }));
        drain_thread.join().unwrap();
        assert_eq!(
            *order.lock().unwrap(),
            vec![
                "request_started",
                "backend_started",
                "response_completed",
                "request_finished"
            ]
        );
    }

    #[test]
    fn admitted_requests_map_route_and_terminal_once() {
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);

        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation: OpenAiBackendOperation::ChatCompletion,
        });
        adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context: context.clone(),
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });

        assert!(
            service
                .registry_ref()
                .get_active(&request_id.as_uuid().to_string())
                .is_none()
        );
        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal request summary");
        assert_eq!(summary.metadata.route(), Some("chat_completions"));
        assert_eq!(summary.metadata.provider(), Some("openai_frontend"));
        assert_eq!(summary.metadata.engine(), Some("chat_completion"));
        let records = service.bus_ref().replay_window().records;
        assert_eq!(
            records
                .iter()
                .filter(|record| record.entry.payload.contains("route_selected"))
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.entry.payload.contains("completed"))
                .count(),
            1
        );
        assert_eq!(adapter.tracked_len(), 1);
    }

    #[test]
    fn legacy_observability_retains_authenticated_request_custody() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_target(true)
            .with_writer(TraceWriter(Arc::clone(&output)))
            .finish();
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let mut context = context(request_id);
        context.agent_session_id = Some("cacheline-eval-developer-session-1".to_owned());
        context.agent_session_source = Some("trusted_header".to_owned());
        let operation = OpenAiBackendOperation::ChatCompletion;
        let usage = OpenAiUsage {
            prompt_tokens: 21,
            cached_tokens: 13,
            completion_tokens: 8,
            total_tokens: 29,
        };

        tracing::subscriber::with_default(subscriber, || {
            adapter.observe(&OpenAiLifecycleEvent::Admitted {
                context: context.clone(),
            });
            adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
                context: context.clone(),
                operation,
            });
            adapter.observe(&OpenAiLifecycleEvent::BackendTerminal {
                context: context.clone(),
                operation,
                result: OpenAiTerminalResult::Completed { status_code: 200 },
            });
            adapter.observe(&OpenAiLifecycleEvent::ResponseCompleted {
                context: context.clone(),
                operation,
                usage,
            });
            adapter.observe(&OpenAiLifecycleEvent::NonStreamTerminal {
                context,
                result: OpenAiTerminalResult::CompletedWithUsage {
                    status_code: 200,
                    usage: TokenUsage {
                        prompt_tokens: Some(21),
                        completion_tokens: Some(8),
                        total_tokens: Some(29),
                    },
                },
            });
        });

        let lines = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        let records = lines
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|record| record["target"] == LEGACY_OBSERVABILITY_TARGET)
            .collect::<Vec<_>>();
        assert_eq!(
            records
                .iter()
                .map(|record| record["fields"]["event"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "request_started",
                "backend_started",
                "backend_returned",
                "response_completed",
                "request_finished",
            ]
        );
        for record in records {
            assert_eq!(record["fields"]["operation"], "chat_completion");
            assert_eq!(
                record["fields"]["agent_session_id"],
                "cacheline-eval-developer-session-1"
            );
            assert_eq!(record["fields"]["agent_session_source"], "trusted_header");
            assert_eq!(
                record["fields"]["request_id"],
                request_id.as_uuid().to_string()
            );
        }
        assert!(
            service
                .registry_ref()
                .get_recent(&request_id.as_uuid().to_string())
                .is_some()
        );
    }

    #[test]
    fn legacy_observability_uses_timeout_and_stream_error_events() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_target(true)
            .with_writer(TraceWriter(Arc::clone(&output)))
            .finish();
        let (_service, adapter) = adapter();
        let request_id = RequestId::new();
        let mut context = context(request_id);
        context.agent_session_id = Some("cacheline-eval-developer-session-1".to_owned());
        context.agent_session_source = Some("trusted_header".to_owned());
        let operation = OpenAiBackendOperation::ChatCompletionStream;

        tracing::subscriber::with_default(subscriber, || {
            adapter.observe(&OpenAiLifecycleEvent::Admitted {
                context: context.clone(),
            });
            adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
                context: context.clone(),
                operation,
            });
            adapter.observe(&OpenAiLifecycleEvent::BackendTerminal {
                context: context.clone(),
                operation,
                result: OpenAiTerminalResult::Failed {
                    status_code: 504,
                    failure: OpenAiFailure::Timeout,
                },
            });
            adapter.observe(&OpenAiLifecycleEvent::StreamTerminal {
                context,
                result: OpenAiTerminalResult::Failed {
                    status_code: 504,
                    failure: OpenAiFailure::Timeout,
                },
            });
        });

        let lines = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        let records = lines
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|record| record["target"] == LEGACY_OBSERVABILITY_TARGET)
            .collect::<Vec<_>>();
        assert_eq!(
            records
                .iter()
                .map(|record| record["fields"]["event"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "request_started",
                "backend_started",
                "backend_timeout",
                "stream_item_error",
                "request_finished",
            ]
        );
        for record in records {
            assert_eq!(record["fields"]["agent_session_source"], "trusted_header");
            if matches!(
                record["fields"]["event"].as_str(),
                Some("backend_timeout" | "stream_item_error" | "request_finished")
            ) {
                assert!(record["fields"]["elapsed_us"].is_u64());
            }
        }
    }

    #[test]
    fn pre_dispatch_rejection_does_not_create_legacy_custody() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_target(true)
            .with_writer(TraceWriter(Arc::clone(&output)))
            .finish();
        let (_service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);

        tracing::subscriber::with_default(subscriber, || {
            adapter.observe(&OpenAiLifecycleEvent::Admitted {
                context: context.clone(),
            });
            adapter.observe(&OpenAiLifecycleEvent::Rejected {
                context,
                status_code: 404,
                rejection: OpenAiRejection::NotFound,
            });
        });

        let lines = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert!(
            lines
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .all(|record| record["target"] != LEGACY_OBSERVABILITY_TARGET)
        );
    }

    #[test]
    fn backend_stream_and_usage_events_map_to_canonical_children_once() {
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);
        let operation = OpenAiBackendOperation::ChatCompletionStream;
        let usage = OpenAiUsage {
            prompt_tokens: 21,
            cached_tokens: 13,
            completion_tokens: 8,
            total_tokens: 29,
        };

        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context: context.clone(),
            operation,
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendTerminal {
            context: context.clone(),
            operation,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });
        adapter.observe(&OpenAiLifecycleEvent::StreamFirstItem {
            context: context.clone(),
            operation,
        });
        adapter.observe(&OpenAiLifecycleEvent::StreamFirstItem {
            context: context.clone(),
            operation,
        });
        adapter.observe(&OpenAiLifecycleEvent::ResponseCompleted {
            context: context.clone(),
            operation,
            usage,
        });
        adapter.observe(&OpenAiLifecycleEvent::ResponseCompleted {
            context: context.clone(),
            operation,
            usage,
        });
        adapter.observe(&OpenAiLifecycleEvent::StreamTerminal {
            context,
            result: OpenAiTerminalResult::Completed { status_code: 200 },
        });

        let events = canonical_events(&service);
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::AttemptStarted { .. }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::AttemptCompleted { .. }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::BackendStreamFirstItem
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::UsageRecorded { .. }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::StreamCompleted {
                    tokens: Some(8),
                    usage: Some(TokenUsage {
                        prompt_tokens: Some(21),
                        completion_tokens: Some(8),
                        total_tokens: Some(29),
                    }),
                }
            )),
            1
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::Completed { .. }
            )),
            1
        );
    }

    fn canonical_events(service: &LoggingService) -> Vec<LifecycleEvent> {
        service
            .bus_ref()
            .replay_window()
            .records
            .into_iter()
            .filter_map(|record| {
                let envelope =
                    serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                serde_json::from_str(envelope.get("payload")?.as_str()?).ok()
            })
            .collect()
    }

    fn count_events(
        events: &[LifecycleEvent],
        predicate: impl Fn(&LifecycleEvent) -> bool,
    ) -> usize {
        events.iter().filter(|event| predicate(event)).count()
    }

    #[test]
    fn terminal_before_admission_and_unknown_route_do_not_create_requests() {
        let (service, adapter) = adapter();
        let request_id = RequestId::new();
        let context = context(request_id);

        adapter.observe(&OpenAiLifecycleEvent::StreamDropped {
            context: context.clone(),
        });
        adapter.observe(&OpenAiLifecycleEvent::BackendDispatched {
            context,
            operation: OpenAiBackendOperation::Responses,
        });

        assert!(
            service
                .registry_ref()
                .get_active(&request_id.as_uuid().to_string())
                .is_none()
        );
        assert!(service.bus_ref().replay_window().records.is_empty());
    }

    #[test]
    fn raw_mesh_owner_prevents_a_competing_embedded_frontend_parent() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let raw_mesh_owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter =
            OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&raw_mesh_owners));
        let request_id = RequestId::new();
        let _raw = super::super::RawMeshRequestLifecycle::register(
            Arc::clone(&service),
            raw_mesh_owners,
            request_id,
        )
        .unwrap();

        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });

        // Raw ingress owns canonical state, while this adapter retains a
        // legacy-only record for authenticated custody.
        assert_eq!(adapter.tracked_len(), 1);
        assert_eq!(
            service
                .bus_ref()
                .replay_window()
                .records
                .into_iter()
                .filter_map(|record| {
                    let envelope =
                        serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                    serde_json::from_str(envelope.get("payload")?.as_str()?).ok()
                })
                .filter(|event| matches!(event, LifecycleEvent::Admitted { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn ingress_attachment_keeps_direct_frontend_on_one_parent() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let raw_mesh_owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter =
            OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&raw_mesh_owners));
        let request_id = RequestId::new();
        let parent =
            RawMeshRequestLifecycle::register(Arc::clone(&service), raw_mesh_owners, request_id)
                .expect("direct ingress should claim one parent");
        let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));

        // A direct host ingress can pass through the embedded frontend, but
        // the shared owner registry prevents it from creating a second parent.
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });
        // The raw parent remains canonical; the adapter's entry is legacy-only.
        assert_eq!(adapter.tracked_len(), 1);

        let observer = attachment.route_observer();
        observer.route_selected(Some("safe-model"));
        let attempt_id = observer
            .start_attempt()
            .expect("owned attachment should allocate attempts");
        observer.complete_attempt(Some(attempt_id), 200);
        attachment.terminal(TerminalOutcome::Completed);

        let events = service.bus_ref().replay_window().records;
        assert_eq!(
            events
                .iter()
                .filter(|record| record.entry.payload.contains("\"type\":\"admitted\""))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|record| record.entry.payload.contains("\"type\":\"completed\""))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|record| record
                    .entry
                    .payload
                    .contains("\"type\":\"attempt_started\""))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|record| record
                    .entry
                    .payload
                    .contains("\"type\":\"attempt_completed\""))
                .count(),
            1
        );
    }

    #[test]
    fn remote_tunnel_suppression_skips_canonical_parent_but_retains_legacy() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let raw_mesh_owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter =
            OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&raw_mesh_owners));
        let request_id = RequestId::new();

        let lease = super::super::RawMeshRemoteSuppressionLease::acquire(
            Arc::clone(&raw_mesh_owners),
            request_id,
        )
        .unwrap();
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });
        // Suppression skips a second canonical parent, but the adapter retains
        // a legacy-only record for the target's custody events.
        assert_eq!(adapter.tracked_len(), 1);

        drop(lease);
        adapter.observe(&OpenAiLifecycleEvent::Admitted {
            context: context(request_id),
        });
        assert_eq!(adapter.tracked_len(), 1);
    }

    #[test]
    fn terminal_labels_are_bounded_metadata() {
        assert_eq!(
            terminal_outcome(OpenAiTerminalResult::Failed {
                status_code: 504,
                failure: OpenAiFailure::Timeout,
            }),
            TerminalOutcome::FailedWithStatus {
                error: "timeout".into(),
                status_code: 504,
            }
        );
        assert_eq!(
            terminal_outcome(OpenAiTerminalResult::Failed {
                status_code: 499,
                failure: OpenAiFailure::Cancelled,
            }),
            TerminalOutcome::Cancelled(Some("request_cancelled".into()))
        );
        assert_eq!(
            serde_json::to_string(&LifecycleEvent::RouteSelected {
                model: None,
                provider: Some("openai_frontend".into()),
                engine: Some(operation_label(OpenAiBackendOperation::Responses).into()),
            })
            .expect("event should serialize"),
            r#"{"type":"route_selected","provider":"openai_frontend","engine":"responses"}"#
        );
    }

    #[test]
    fn bounded_tracking_rejects_new_admission_when_all_slots_are_active() {
        let (service, adapter) = adapter();
        for _ in 0..MAX_TRACKED_REQUESTS {
            adapter.admit(&context(RequestId::new()));
        }
        let overflow = RequestId::new();
        adapter.admit(&context(overflow));

        assert_eq!(adapter.tracked_len(), MAX_TRACKED_REQUESTS);
        assert!(
            service
                .registry_ref()
                .get_active(&overflow.as_uuid().to_string())
                .is_none()
        );
    }
}
