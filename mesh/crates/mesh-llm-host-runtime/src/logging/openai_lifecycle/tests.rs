use std::sync::{Arc, Mutex};

use mesh_llm_events::logging::{events::LifecycleEvent, identifiers::RequestId};
use skippy_inference_api::{
    InferenceBackendOperation, InferenceFrontendRoute, InferenceLifecycleContext,
    InferenceLifecycleEvent, InferenceRequestMethod, InferenceTerminalResult, InferenceUsage,
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

fn capturing_attachment(probe: &Arc<ArtifactCaptureProbe>) -> OpenAiLifecycleAttachment {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let parent = RawMeshRequestLifecycle::register(service, owners, RequestId::new())
        .expect("test request owns lifecycle");
    let capture: Arc<dyn OpenAiArtifactCapture> = probe.clone();
    OpenAiLifecycleAttachment::with_capture(Some(parent), capture)
}

#[test]
fn unguarded_deferred_request_and_stream_are_captured_once() {
    let probe = Arc::new(ArtifactCaptureProbe::default());
    let mut attachment = capturing_attachment(&probe);
    attachment.defer_request_body(b"request body", Some("application/json"));
    assert!(probe.bodies.lock().unwrap().is_empty());
    let observer = attachment.route_observer();
    let mut stream = observer.begin_stream_response_capture().unwrap();
    stream.push(b"stream response");
    observer.complete_stream_response_capture(Some(stream));
    attachment.terminal(TerminalOutcome::Completed);
    attachment.terminal(TerminalOutcome::Completed);
    assert_eq!(
        *probe.bodies.lock().unwrap(),
        vec![b"stream response".to_vec(), b"request body".to_vec()]
    );
    assert!(probe.unavailable.lock().unwrap().is_empty());
}

#[test]
fn oversized_deferred_request_preserves_unavailable_marker_without_retaining_body() {
    for explicit_terminal in [true, false] {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let mut attachment = capturing_attachment(&probe);
        let limit = probe.body_limit_bytes();
        attachment.defer_request_body(&vec![b'x'; limit + 1], Some("application/json"));
        assert!(matches!(
            attachment.pending_request,
            Some(PendingRequestCapture::LimitExceeded)
        ));
        assert!(probe.unavailable.lock().unwrap().is_empty());
        if explicit_terminal {
            attachment.terminal(TerminalOutcome::Completed);
        }
        drop(attachment);
        assert!(probe.bodies.lock().unwrap().is_empty());
        assert_eq!(
            *probe.unavailable.lock().unwrap(),
            vec![ArtifactUnavailableReason::CaptureContentLimitExceeded]
        );
    }
}

#[test]
fn guarded_exchange_never_persists_request_or_response_bodies() {
    for oversized in [false, true] {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let mut attachment = capturing_attachment(&probe);
        let guard = Arc::new(AtomicBool::new(false));
        attachment.with_response_capture_guard(Arc::clone(&guard));
        if oversized {
            attachment.defer_request_body(
                &vec![b'x'; probe.body_limit_bytes() + 1],
                Some("application/json"),
            );
        } else {
            attachment.defer_request_body(b"private request", Some("application/json"));
        }
        guard.store(true, Ordering::Release);
        attachment.capture_request_body(b"private request", Some("application/json"));
        let observer = attachment.route_observer();
        observer.capture_response_body(b"private image", Some("application/json"));
        assert!(observer.begin_stream_response_capture().is_none());
        attachment.terminal(TerminalOutcome::Completed);
        drop(attachment);
        assert!(probe.bodies.lock().unwrap().is_empty());
        assert!(probe.unavailable.lock().unwrap().is_empty());
    }
}

#[test]
fn stream_capture_created_before_guard_arming_is_suppressed_on_complete_and_drop() {
    for complete in [true, false] {
        let probe = Arc::new(ArtifactCaptureProbe::default());
        let mut attachment = capturing_attachment(&probe);
        let guard = Arc::new(AtomicBool::new(false));
        attachment.with_response_capture_guard(Arc::clone(&guard));
        let observer = attachment.route_observer();
        let mut stream = observer.begin_stream_response_capture().unwrap();
        stream.push(b"private response before guard");
        guard.store(true, Ordering::Release);
        if complete {
            observer.complete_stream_response_capture(Some(stream));
        } else {
            observer.abandon_stream_response_capture(Some(stream));
        }
        drop(attachment);
        assert!(probe.bodies.lock().unwrap().is_empty());
        assert!(probe.unavailable.lock().unwrap().is_empty());
    }
}

#[test]
fn guarded_stream_discards_previously_buffered_raw_bytes_when_pushed_again() {
    let probe = Arc::new(ArtifactCaptureProbe::default());
    let mut attachment = capturing_attachment(&probe);
    let guard = Arc::new(AtomicBool::new(false));
    attachment.with_response_capture_guard(Arc::clone(&guard));
    let observer = attachment.route_observer();
    let mut stream = observer.begin_stream_response_capture().unwrap();
    stream.push(b"private response before guard");
    guard.store(true, Ordering::Release);
    stream.push(b"private response after guard");
    assert!(stream.bytes.is_empty());
    observer.complete_stream_response_capture(Some(stream));
    drop(attachment);
    assert!(probe.bodies.lock().unwrap().is_empty());
    assert!(probe.unavailable.lock().unwrap().is_empty());
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

fn context(request_id: RequestId) -> InferenceLifecycleContext {
    InferenceLifecycleContext::new(
        request_id,
        InferenceRequestMethod::Post,
        InferenceFrontendRoute::ChatCompletions,
    )
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
fn admitted_requests_map_route_and_terminal_once() {
    let (service, adapter) = adapter();
    let request_id = RequestId::new();
    let context = context(request_id);

    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::BackendDispatched {
        context: context.clone(),
        operation: InferenceBackendOperation::ChatCompletion,
    });
    adapter.observe(&InferenceLifecycleEvent::NonStreamTerminal {
        context: context.clone(),
        result: InferenceTerminalResult::Completed { status_code: 200 },
        exchange_id: None,
    });
    adapter.observe(&InferenceLifecycleEvent::NonStreamTerminal {
        context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
        exchange_id: None,
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
    assert_eq!(summary.metadata.source(), Some("direct_http"));
    assert_eq!(summary.metadata.method(), Some("POST"));
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

/// a `NonStreamTerminal` carrying the exchange
/// join-key merges it into the request's summary metadata, so the Logs
/// record and other consumers of the exchange can be correlated by id.
#[test]
fn non_stream_terminal_with_exchange_id_merges_it_into_summary_metadata() {
    let (service, adapter) = adapter();
    let request_id = RequestId::new();
    let context = context(request_id);

    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::NonStreamTerminal {
        context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
        exchange_id: Some("exch-abc123".to_string()),
    });

    let summary = service
        .registry_ref()
        .get_recent(&request_id.as_uuid().to_string())
        .expect("terminal request summary");
    assert_eq!(summary.metadata.exchange_id(), Some("exch-abc123"));
}

/// A terminal request that never carried an exchange (e.g. `/v1/completions`,
/// or chat completions dispatched without mesh hooks enabled) must leave
/// the summary's exchange id absent rather than fabricating one.
#[test]
fn non_stream_terminal_without_exchange_id_leaves_summary_metadata_absent() {
    let (service, adapter) = adapter();
    let request_id = RequestId::new();
    let context = context(request_id);

    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::NonStreamTerminal {
        context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
        exchange_id: None,
    });

    let summary = service
        .registry_ref()
        .get_recent(&request_id.as_uuid().to_string())
        .expect("terminal request summary");
    assert_eq!(summary.metadata.exchange_id(), None);
}

#[test]
fn stream_exchange_id_survives_terminalization() {
    let (service, adapter) = adapter();
    let request_id = RequestId::new();
    let context = context(request_id);

    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::ExchangeIdentified {
        context: context.clone(),
        exchange_id: "exch-stream-1".to_string(),
    });
    adapter.observe(&InferenceLifecycleEvent::StreamTerminal {
        context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
    });

    let summary = service
        .registry_ref()
        .get_recent(&request_id.as_uuid().to_string())
        .expect("terminal request summary");
    assert_eq!(summary.metadata.exchange_id(), Some("exch-stream-1"));
}

#[test]
fn terminalization_releases_frontend_owner_without_disturbing_raw_owner() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let adapter = OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&owners));
    let frontend_request_id = RequestId::new();
    let frontend_context = context(frontend_request_id);
    let raw_request_id = RequestId::new();
    let raw = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::clone(&owners),
        raw_request_id,
    )
    .expect("raw request should own its lifecycle");
    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: frontend_context.clone(),
    });
    assert!(owners.is_claimed(frontend_request_id));
    assert!(owners.is_claimed(raw_request_id));

    adapter.observe(&InferenceLifecycleEvent::NonStreamTerminal {
        context: frontend_context.clone(),
        result: InferenceTerminalResult::Completed { status_code: 200 },
        exchange_id: None,
    });
    adapter.observe(&InferenceLifecycleEvent::NonStreamTerminal {
        context: frontend_context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
        exchange_id: None,
    });

    assert!(!owners.is_claimed(frontend_request_id));
    assert!(owners.is_claimed(raw_request_id));
    assert_eq!(adapter.tracked_len(), 1);
    raw.terminal(TerminalOutcome::Completed);
}

#[test]
fn backend_stream_and_usage_events_map_to_canonical_children_once() {
    let (service, adapter) = adapter();
    let request_id = RequestId::new();
    let context = context(request_id);
    let operation = InferenceBackendOperation::ChatCompletionStream;
    let usage = InferenceUsage {
        prompt_tokens: 21,
        cached_tokens: Some(13),
        completion_tokens: 8,
        total_tokens: 29,
    };

    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::BackendDispatched {
        context: context.clone(),
        operation,
    });
    adapter.observe(&InferenceLifecycleEvent::BackendTerminal {
        context: context.clone(),
        operation,
        result: InferenceTerminalResult::Completed { status_code: 200 },
    });
    adapter.observe(&InferenceLifecycleEvent::StreamFirstItem {
        context: context.clone(),
        operation,
    });
    adapter.observe(&InferenceLifecycleEvent::StreamFirstItem {
        context: context.clone(),
        operation,
    });
    adapter.observe(&InferenceLifecycleEvent::ResponseCompleted {
        context: context.clone(),
        operation,
        usage,
    });
    adapter.observe(&InferenceLifecycleEvent::ResponseCompleted {
        context: context.clone(),
        operation,
        usage,
    });
    adapter.observe(&InferenceLifecycleEvent::StreamTerminal {
        context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
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
                    cached_prompt_tokens: Some(13),
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
            let envelope = serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
            serde_json::from_str(envelope.get("payload")?.as_str()?).ok()
        })
        .collect()
}

fn count_events(events: &[LifecycleEvent], predicate: impl Fn(&LifecycleEvent) -> bool) -> usize {
    events.iter().filter(|event| predicate(event)).count()
}

#[test]
fn terminal_before_admission_and_unknown_route_do_not_create_requests() {
    let (service, adapter) = adapter();
    let request_id = RequestId::new();
    let context = context(request_id);

    adapter.observe(&InferenceLifecycleEvent::StreamDropped {
        context: context.clone(),
    });
    adapter.observe(&InferenceLifecycleEvent::BackendDispatched {
        context,
        operation: InferenceBackendOperation::Responses,
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

    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context(request_id),
    });

    assert_eq!(adapter.tracked_len(), 0);
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
fn markerless_authenticated_attribution_and_frontend_admission_choose_one_parent_in_both_orders() {
    for attribution_first in [true, false] {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let owners = Arc::new(RawMeshLifecycleOwners::default());
        let adapter = OpenAiLifecycleLoggingAdapter::new(Arc::clone(&service), Arc::clone(&owners));
        let request_id = RequestId::new();
        let endpoint_id = "83".repeat(32);
        let authenticated = RequestSummaryMetadata::default().with_caller_identity(
            Some(&endpoint_id),
            Some("192.0.2.83:11204"),
            Some(super::super::CallerPathType::RemoteQuicHttp),
        );
        let request_metadata = RequestSummaryMetadata::from_parts(
            Some("chat_completions"),
            Some("model-a"),
            Some("openai_frontend"),
            Some("chat_completion"),
        )
        .with_source(Some("direct_http"))
        .with_method(Some("POST"));

        let attribution = if attribution_first {
            super::super::RawMeshRemoteAttributionLease::acquire(
                &service,
                Arc::clone(&owners),
                request_id,
                authenticated.clone(),
            )
            .expect("authenticated attribution")
        } else {
            adapter.admit(&context(request_id));
            service.merge_request_metadata(request_id, request_metadata.clone());
            super::super::RawMeshRemoteAttributionLease::acquire(
                &service,
                Arc::clone(&owners),
                request_id,
                authenticated,
            )
            .expect("authenticated attribution")
        };
        if attribution_first {
            adapter.admit(&context(request_id));
        }

        let raw = RawMeshRequestLifecycle::register_with_metadata(
            Arc::clone(&service),
            owners,
            request_id,
            request_metadata,
        );
        match raw {
            Some(raw) => raw.terminal(TerminalOutcome::Completed),
            None => adapter.terminal(request_id, TerminalOutcome::Completed),
        }
        drop(attribution);

        let events = canonical_events(&service);
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::Admitted { .. }
            )),
            1,
            "admission order: attribution_first={attribution_first}"
        );
        assert_eq!(
            count_events(&events, |event| matches!(
                event,
                LifecycleEvent::Completed { .. }
            )),
            1,
            "terminal order: attribution_first={attribution_first}"
        );
        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal summary");
        assert_eq!(summary.metadata.route(), Some("chat_completions"));
        assert_eq!(summary.metadata.model(), Some("model-a"));
        assert_eq!(summary.metadata.provider(), Some("openai_frontend"));
        assert_eq!(summary.metadata.engine(), Some("chat_completion"));
        assert_eq!(summary.metadata.source(), Some("direct_http"));
        assert_eq!(summary.metadata.method(), Some("POST"));
        assert_eq!(
            summary.metadata.caller_endpoint_id(),
            Some(endpoint_id.as_str())
        );
        assert_eq!(summary.metadata.caller_addr(), Some("192.0.2.83:11204"));
        assert_eq!(
            summary.metadata.caller_path_type(),
            Some("remote_quic_http")
        );
    }
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
    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context(request_id),
    });
    assert_eq!(adapter.tracked_len(), 0);

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
fn remote_tunnel_suppression_skips_only_the_active_frontend_admission() {
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
    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context(request_id),
    });
    assert_eq!(adapter.tracked_len(), 0);

    drop(lease);
    adapter.observe(&InferenceLifecycleEvent::Admitted {
        context: context(request_id),
    });
    assert_eq!(adapter.tracked_len(), 1);
}

#[test]
fn terminal_labels_are_bounded_metadata() {
    assert_eq!(
        terminal_outcome(InferenceTerminalResult::Failed {
            status_code: 504,
            failure: InferenceFailure::Timeout,
        }),
        TerminalOutcome::FailedWithStatus {
            error: "timeout".into(),
            status_code: 504,
        }
    );
    assert_eq!(
        terminal_outcome(InferenceTerminalResult::Failed {
            status_code: 499,
            failure: InferenceFailure::Cancelled,
        }),
        TerminalOutcome::Cancelled(Some("request_cancelled".into()))
    );
    assert_eq!(
        serde_json::to_string(&LifecycleEvent::RouteSelected {
            model: None,
            provider: Some("openai_frontend".into()),
            engine: Some(operation_label(InferenceBackendOperation::Responses).into()),
        })
        .expect("event should serialize"),
        r#"{"type":"route_selected","provider":"openai_frontend","engine":"responses"}"#
    );
    assert_eq!(
        operation_label(InferenceBackendOperation::SystemOne),
        "system_one"
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

/// Review defect D12: pins `runtime_events::config::REQUEST_ROOT_BOUND`
/// to this module's `MAX_TRACKED_REQUESTS` so the runtime-event
/// reservation sizing cannot silently desynchronize from the logging
/// request-tracking bound. Imports both constants through their real
/// crate-visible paths (not `super::*`) so the test also proves
/// `MAX_TRACKED_REQUESTS` is reachable crate-wide, matching how
/// `logging::raw_mesh_lifecycle` and `network::openai::runtime_events`
/// consume it.
#[test]
fn request_root_bound() {
    use crate::logging::MAX_TRACKED_REQUESTS;
    use crate::runtime_events::config::REQUEST_ROOT_BOUND;

    assert_eq!(
        REQUEST_ROOT_BOUND, MAX_TRACKED_REQUESTS,
        "runtime_events::config::REQUEST_ROOT_BOUND must stay pinned to \
         logging::openai_lifecycle::MAX_TRACKED_REQUESTS"
    );
}
