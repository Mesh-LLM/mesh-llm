//! Host-owned bridge from OpenAI frontend lifecycle boundaries to logging.
//!
//! The frontend emits bounded metadata. This adapter owns the matching request
//! guards and hands the production transport a narrow, body-only capture view.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use mesh_llm_events::logging::{
    events::{LifecycleEvent, TokenUsage},
    identifiers::{AttemptId, RequestId},
    replay::ReplayChannel,
};
use skippy_inference_api::{
    InferenceBackendOperation, InferenceFailure, InferenceLifecycleContext,
    InferenceLifecycleEvent, InferenceLifecycleObserver, InferenceRejection,
    InferenceRequestMethod, InferenceTerminalResult, InferenceUsage,
};

use super::raw_mesh_lifecycle::FrontendAdmissionDecision;
use super::{
    ArtifactUnavailableReason, LifecycleGuard, LoggingService, ProxyAttemptFinish,
    RawMeshLifecycleOwners, RawMeshProxyAttempt, RawMeshRequestLifecycle, RequestSummaryMetadata,
    TerminalOutcome,
};

/// Fixed upper bound for requests owned by one embedded frontend observer.
///
/// `pub(crate)` (review defect D12) so `logging::raw_mesh_lifecycle` and
/// `network::openai::runtime_events` import this single definition instead
/// of each carrying their own copy; pinned against
/// `runtime_events::config::REQUEST_ROOT_BOUND` by the `request_root_bound`
/// contract test in this module's `tests` block.
pub(crate) const MAX_TRACKED_REQUESTS: usize = 1_024;

/// The host-owned lifecycle attachment for one parsed OpenAI ingress request.
///
/// The attachment keeps terminal ownership at the ingress boundary. Routing
/// receives only the [`OpenAiRouteObserver`] view below, which can publish
/// bounded route/attempt metadata but cannot admit or terminalize a request.
pub(crate) struct OpenAiLifecycleAttachment {
    parent: Option<RawMeshRequestLifecycle>,
    capture: Option<Arc<dyn OpenAiArtifactCapture>>,
    terminalized: bool,
    response_capture_guard: Option<Arc<AtomicBool>>,
    pending_request: Option<PendingRequestCapture>,
}

enum PendingRequestCapture {
    Body(Vec<u8>, Option<String>),
    LimitExceeded,
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
    response_capture_guard: Option<Arc<AtomicBool>>,
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
            response_capture_guard: None,
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        if self.guarded() {
            self.bytes.clear();
            return;
        }
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
        if self.guarded() {
            return;
        }
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

    fn guarded(&self) -> bool {
        self.response_capture_guard
            .as_ref()
            .is_some_and(|guard| guard.load(Ordering::Acquire))
    }
}

impl Drop for OpenAiStreamArtifactCapture {
    fn drop(&mut self) {
        if !self.finalized && !self.guarded() {
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
    response_capture_guard: Option<&'a Arc<AtomicBool>>,
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
            response_capture_guard: None,
            pending_request: None,
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
            response_capture_guard: None,
            pending_request: None,
        }
    }

    pub(crate) fn unowned() -> Self {
        Self::new(None)
    }

    pub(crate) fn owns_parent(&self) -> bool {
        self.parent.is_some()
    }

    /// Guarded exchanges retain no raw input or output artifact. The atomic is
    /// armed at selected-backend admission, before generation starts.
    pub(crate) fn with_response_capture_guard(&mut self, guard: Arc<AtomicBool>) {
        self.response_capture_guard = Some(guard);
    }

    pub(crate) fn defer_request_body(&mut self, content: &[u8], media_kind: Option<&str>) {
        let Some(capture) = self.capture.as_ref() else {
            return;
        };
        self.pending_request = Some(if content.len() <= capture.body_limit_bytes() {
            PendingRequestCapture::Body(content.to_vec(), media_kind.map(str::to_owned))
        } else {
            PendingRequestCapture::LimitExceeded
        });
    }

    pub(crate) fn route_observer(&self) -> OpenAiRouteObserver<'_> {
        OpenAiRouteObserver {
            parent: self.parent.as_ref(),
            capture: self.capture.as_ref(),
            request_id: self
                .parent
                .as_ref()
                .map(RawMeshRequestLifecycle::request_id),
            response_capture_guard: self.response_capture_guard.as_ref(),
        }
    }

    /// Capture only the decoded request body. Raw HTTP headers are never made
    /// available here, so credentials cannot enter artifact capture.
    pub(crate) fn capture_request_body(&self, content: &[u8], media_kind: Option<&str>) {
        if self
            .response_capture_guard
            .as_ref()
            .is_some_and(|guard| guard.load(Ordering::Acquire))
        {
            return;
        }
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
        if let Some(pending) = self.pending_request.take()
            && !self
                .response_capture_guard
                .as_ref()
                .is_some_and(|guard| guard.load(Ordering::Acquire))
        {
            match pending {
                PendingRequestCapture::Body(content, media_kind) => {
                    self.capture_request_body(&content, media_kind.as_deref())
                }
                PendingRequestCapture::LimitExceeded => {
                    if let (Some(parent), Some(capture)) =
                        (self.parent.as_ref(), self.capture.as_ref())
                    {
                        capture.capture_unavailable(
                            parent.request_id(),
                            "request",
                            ArtifactUnavailableReason::CaptureContentLimitExceeded,
                        );
                    }
                }
            }
        }
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
            response_capture_guard: None,
        }
    }

    /// Capture a client-visible non-streaming response body. The capture
    /// implementation applies the canonical redactor and configured byte
    /// limit; HTTP headers are intentionally never passed here.
    pub(crate) fn capture_response_body(&self, content: &[u8], media_kind: Option<&str>) {
        if self
            .response_capture_guard
            .is_some_and(|guard| guard.load(Ordering::Acquire))
        {
            return;
        }
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
        if self
            .response_capture_guard
            .is_some_and(|guard| guard.load(Ordering::Acquire))
        {
            return None;
        }
        let capture = self.capture?;
        let request_id = self.request_id?;
        let mut stream = OpenAiStreamArtifactCapture::new(
            Arc::clone(capture),
            request_id,
            capture.body_limit_bytes(),
        );
        stream.response_capture_guard = self.response_capture_guard.cloned();
        Some(stream)
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
    Active(ActiveRequest),
    Terminal,
}

struct ActiveRequest {
    guard: LifecycleGuard,
    backend_attempt: Option<(InferenceBackendOperation, AttemptId)>,
    backend_stream_first_item: bool,
    usage: Option<InferenceUsage>,
}

impl ActiveRequest {
    fn new(guard: LifecycleGuard) -> Self {
        Self {
            guard,
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

    fn admit(&self, context: &InferenceLifecycleContext) {
        let request_id = context.request_id;
        self.raw_mesh_owners.admit_frontend(request_id, || {
            let mut tracked = lock_recover(&self.tracked);
            if tracked.requests.contains_key(&request_id) {
                return FrontendAdmissionDecision::Rejected;
            }
            let Ok(evicted) = tracked.make_room() else {
                return FrontendAdmissionDecision::Rejected;
            };

            let (guard, _) = self.service.register_request_with_metadata(
                request_id,
                RequestSummaryMetadata::from_openai_frontend_route(context.route)
                    .with_source(Some("direct_http"))
                    .with_method(Some(openai_method_label(context.method))),
            );
            tracked.requests.insert(
                request_id,
                TrackedRequest::Active(ActiveRequest::new(guard)),
            );
            tracked.insertion_order.push_back(request_id);
            FrontendAdmissionDecision::Registered { evicted }
        });
    }

    fn backend_dispatched(&self, request_id: RequestId, operation: InferenceBackendOperation) {
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
        self.service
            .merge_request_metadata(request_id, metadata.clone());
        let event = LifecycleEvent::RouteSelected {
            model: None,
            provider: metadata.provider().map(str::to_owned),
            engine: metadata.engine().map(str::to_owned),
        };
        self.enqueue_operation_event(request_id, event);
        let attempt_id = self.service.start_attempt(request_id, &guard);
        let mut tracked = lock_recover(&self.tracked);
        if let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) {
            active.backend_attempt = Some((operation, attempt_id));
        }
    }

    fn backend_terminal(
        &self,
        request_id: RequestId,
        operation: InferenceBackendOperation,
        result: InferenceTerminalResult,
    ) {
        let attempt_id = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            match active.backend_attempt {
                Some((attempt_operation, attempt_id)) if attempt_operation == operation => {
                    active.backend_attempt = None;
                    Some(attempt_id)
                }
                _ => None,
            }
        };
        let Some(attempt_id) = attempt_id else {
            return;
        };
        match result {
            InferenceTerminalResult::Completed { status_code }
            | InferenceTerminalResult::CompletedWithUsage { status_code, .. } => {
                self.service
                    .complete_attempt(request_id, attempt_id, Some(status_code));
            }
            InferenceTerminalResult::Failed { failure, .. } => {
                self.service.fail_attempt(
                    request_id,
                    attempt_id,
                    failure_label(failure).to_owned(),
                );
            }
        }
    }

    fn backend_stream_first_item(&self, request_id: RequestId) {
        let should_emit = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.backend_stream_first_item {
                false
            } else {
                active.backend_stream_first_item = true;
                true
            }
        };
        if should_emit {
            self.enqueue_operation_event(request_id, LifecycleEvent::BackendStreamFirstItem);
        }
    }

    fn response_completed(&self, request_id: RequestId, usage: InferenceUsage) {
        let should_emit = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            if active.usage.is_some() {
                false
            } else {
                active.usage = Some(usage);
                true
            }
        };
        if should_emit {
            self.enqueue_operation_event(
                request_id,
                LifecycleEvent::UsageRecorded {
                    prompt_tokens: Some(u64::from(usage.prompt_tokens)),
                    cached_prompt_tokens: usage.cached_tokens.map(u64::from),
                    completion_tokens: Some(u64::from(usage.completion_tokens)),
                    total_tokens: Some(u64::from(usage.total_tokens)),
                },
            );
        }
    }

    fn stream_terminal(&self, request_id: RequestId, result: InferenceTerminalResult) {
        let usage = {
            let tracked = lock_recover(&self.tracked);
            let Some(TrackedRequest::Active(active)) = tracked.requests.get(&request_id) else {
                return;
            };
            active.usage
        };
        match result {
            InferenceTerminalResult::Completed { .. } => {
                let terminal_usage = usage.and_then(|value| {
                    TokenUsage::from_counts(
                        Some(u64::from(value.prompt_tokens)),
                        Some(u64::from(value.completion_tokens)),
                        Some(u64::from(value.total_tokens)),
                    )
                    .map(|usage| {
                        usage.with_cached_prompt_tokens(value.cached_tokens.map(u64::from))
                    })
                });
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::StreamCompleted {
                        tokens: usage.map(|value| u64::from(value.completion_tokens)),
                        usage: terminal_usage,
                    },
                );
            }
            InferenceTerminalResult::CompletedWithUsage { usage, .. } => {
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::StreamCompleted {
                        tokens: usage.completion_tokens,
                        usage: Some(usage),
                    },
                );
            }
            InferenceTerminalResult::Failed { failure, .. } => {
                self.enqueue_operation_event(
                    request_id,
                    LifecycleEvent::StreamError {
                        error: Some(failure_label(failure).to_owned()),
                    },
                );
            }
        }
    }

    fn stream_interrupted(&self, request_id: RequestId, label: &'static str) {
        if lock_recover(&self.tracked).is_active(request_id) {
            self.enqueue_operation_event(
                request_id,
                LifecycleEvent::StreamError {
                    error: Some(label.to_owned()),
                },
            );
        }
    }

    fn enqueue_operation_event(&self, request_id: RequestId, event: LifecycleEvent) {
        if let Ok(payload) = serde_json::to_string(&event) {
            let _ = self
                .service
                .enqueue_event(request_id, ReplayChannel::Operations, payload);
        }
    }

    /// Merge the exchange join-key into the request's summary metadata, when
    /// this terminal request carried one — a chat/responses completion
    /// dispatched through the exchange-tracked path. Merged via
    /// [`RequestSummaryMetadata::merge_missing`] semantics like every other
    /// summary field: the first truthful value wins, so this is safe to call
    /// even if some future caller ever tries to set it twice.
    fn merge_exchange_id(&self, request_id: RequestId, exchange_id: Option<&str>) {
        let Some(exchange_id) = exchange_id else {
            return;
        };
        self.service.merge_request_metadata(
            request_id,
            RequestSummaryMetadata::with_exchange_id_only(exchange_id),
        );
    }

    fn terminal(&self, request_id: RequestId, outcome: TerminalOutcome) {
        let guard = {
            let mut tracked = lock_recover(&self.tracked);
            let Some(entry) = tracked.requests.get_mut(&request_id) else {
                return;
            };
            let TrackedRequest::Active(active) = entry else {
                return;
            };
            let guard = active.guard.clone();
            *entry = TrackedRequest::Terminal;
            guard
        };

        // A stale or externally-terminalized guard is harmless: request
        // serving and later frontend events must never depend on this write.
        let _ = self
            .service
            .transition_terminal(request_id, &guard, outcome);
        self.raw_mesh_owners.release_frontend(request_id);
    }

    #[cfg(test)]
    fn tracked_len(&self) -> usize {
        lock_recover(&self.tracked).requests.len()
    }
}

const fn openai_method_label(method: InferenceRequestMethod) -> &'static str {
    match method {
        InferenceRequestMethod::Get => "GET",
        InferenceRequestMethod::Post => "POST",
        InferenceRequestMethod::Other => "OTHER",
    }
}

impl TrackedRequests {
    fn make_room(&mut self) -> Result<Option<RequestId>, ()> {
        if self.requests.len() < MAX_TRACKED_REQUESTS {
            return Ok(None);
        }
        let Some(oldest) = self.insertion_order.pop_front() else {
            return Err(());
        };
        if matches!(self.requests.get(&oldest), Some(TrackedRequest::Terminal)) {
            self.requests.remove(&oldest);
            Ok(Some(oldest))
        } else {
            self.insertion_order.push_front(oldest);
            Err(())
        }
    }

    fn is_active(&self, request_id: RequestId) -> bool {
        matches!(
            self.requests.get(&request_id),
            Some(TrackedRequest::Active(_))
        )
    }
}

impl InferenceLifecycleObserver for OpenAiLifecycleLoggingAdapter {
    fn observe(&self, event: &InferenceLifecycleEvent) {
        match event {
            InferenceLifecycleEvent::Admitted { context } => self.admit(context),
            InferenceLifecycleEvent::BackendDispatched { context, operation } => {
                self.backend_dispatched(context.request_id, *operation)
            }
            InferenceLifecycleEvent::BackendTerminal {
                context,
                operation,
                result,
            } => self.backend_terminal(context.request_id, *operation, *result),
            InferenceLifecycleEvent::StreamFirstItem { context, .. } => {
                self.backend_stream_first_item(context.request_id)
            }
            InferenceLifecycleEvent::ExchangeIdentified {
                context,
                exchange_id,
            } => self.merge_exchange_id(context.request_id, Some(exchange_id)),
            InferenceLifecycleEvent::ResponseCompleted { context, usage, .. } => {
                self.response_completed(context.request_id, *usage)
            }
            InferenceLifecycleEvent::Rejected {
                context, rejection, ..
            } => self.terminal(
                context.request_id,
                TerminalOutcome::Rejected(Some(rejection_label(*rejection).into())),
            ),
            InferenceLifecycleEvent::NonStreamTerminal {
                context,
                result,
                exchange_id,
            } => {
                self.merge_exchange_id(context.request_id, exchange_id.as_deref());
                self.terminal(context.request_id, terminal_outcome(*result))
            }
            InferenceLifecycleEvent::StreamTerminal { context, result } => {
                self.stream_terminal(context.request_id, *result);
                self.terminal(context.request_id, terminal_outcome(*result));
            }
            InferenceLifecycleEvent::StreamCancelled { context } => {
                self.stream_interrupted(context.request_id, "stream_cancelled");
                self.terminal(
                    context.request_id,
                    TerminalOutcome::Cancelled(Some("stream_cancelled".into())),
                );
            }
            InferenceLifecycleEvent::StreamDropped { context } => {
                self.stream_interrupted(context.request_id, "stream_dropped");
                self.terminal(
                    context.request_id,
                    TerminalOutcome::Dropped(Some("stream_dropped".into())),
                );
            }
            InferenceLifecycleEvent::RequestCancelled { context } => self.terminal(
                context.request_id,
                TerminalOutcome::Cancelled(Some("request_cancelled".into())),
            ),
        }
    }
}

fn terminal_outcome(result: InferenceTerminalResult) -> TerminalOutcome {
    match result {
        InferenceTerminalResult::Completed { status_code } => {
            TerminalOutcome::CompletedWithStatus(status_code)
        }
        InferenceTerminalResult::CompletedWithUsage { status_code, usage } => {
            TerminalOutcome::CompletedWithUsage { status_code, usage }
        }
        InferenceTerminalResult::Failed {
            failure: InferenceFailure::Cancelled,
            ..
        } => TerminalOutcome::Cancelled(Some("request_cancelled".into())),
        InferenceTerminalResult::Failed {
            status_code,
            failure,
        } => TerminalOutcome::FailedWithStatus {
            error: failure_label(failure).into(),
            status_code,
        },
    }
}

/// Map backend operations to bounded labels, separating non-chat and streaming workloads.
const fn operation_label(operation: InferenceBackendOperation) -> &'static str {
    match operation {
        InferenceBackendOperation::Models => "models",
        InferenceBackendOperation::Embeddings => "embeddings",
        InferenceBackendOperation::Rerank => "rerank",
        InferenceBackendOperation::AudioSpeech => "audio_speech",
        InferenceBackendOperation::AudioTranscription => "audio_transcription",
        InferenceBackendOperation::AudioTranslation => "audio_translation",
        InferenceBackendOperation::ChatCompletion => "chat_completion",
        InferenceBackendOperation::ChatCompletionStream => "chat_completion_stream",
        InferenceBackendOperation::Completion => "completion",
        InferenceBackendOperation::CompletionStream => "completion_stream",
        InferenceBackendOperation::Responses => "responses",
        InferenceBackendOperation::ResponsesStream => "responses_stream",
        InferenceBackendOperation::Messages => "messages",
        InferenceBackendOperation::MessagesStream => "messages_stream",
        InferenceBackendOperation::MessagesCountTokens => "messages_count_tokens",
        InferenceBackendOperation::SystemOne => "system_one",
    }
}

const fn rejection_label(rejection: InferenceRejection) -> &'static str {
    match rejection {
        InferenceRejection::InvalidRequest => "invalid_request",
        InferenceRejection::PayloadTooLarge => "payload_too_large",
        InferenceRejection::MethodNotAllowed => "method_not_allowed",
        InferenceRejection::NotFound => "not_found",
        InferenceRejection::AdmissionDenied => "admission_denied",
    }
}

const fn failure_label(failure: InferenceFailure) -> &'static str {
    match failure {
        InferenceFailure::Backend => "backend",
        InferenceFailure::Timeout => "timeout",
        InferenceFailure::Internal => "internal",
        InferenceFailure::Cancelled => "cancelled",
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
