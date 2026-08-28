use super::*;

#[tokio::test]
async fn route_observer_fails_open_without_a_parent_or_proxy_record() {
    let sink = Arc::new(TransportProxySink::default());
    let service = LoggingService::new(
        Default::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(crate::logging::SystemClock),
    );
    let observer = OpenAiRouteObserver::default();
    let attempt = observer.start_proxy_attempt();
    assert!(attempt.is_none());
    finish_route_attempt(
        observer,
        attempt,
        "remote",
        ResponseAdapter::OpenAiResponsesJson,
        &RouteAttemptResult::ClientDisconnected,
    );
    assert_eq!(service.pump_sync().await, 0);
    assert!(sink.proxy_records().is_empty());
}

#[test]
fn passive_moa_chat_and_responses_streams_record_usage_lifecycle() {
    let usage = TokenUsage {
        prompt_tokens: Some(8),
        cached_prompt_tokens: None,
        completion_tokens: Some(5),
        total_tokens: Some(13),
    };
    for adapter in [
        ResponseAdapter::OpenAiChatCompletionsStream,
        ResponseAdapter::OpenAiResponsesStream,
    ] {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let parent = RawMeshRequestLifecycle::register(
            Arc::clone(&service),
            Arc::new(RawMeshLifecycleOwners::default()),
            RequestId::new(),
        )
        .expect("passive MoA test should claim one parent");
        let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
        let outcome = RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200,
            usage,
        };

        record_moa_stream_lifecycle(attachment.route_observer(), adapter, outcome);
        attachment.terminal(outcome.terminal_outcome());

        let events = recorded_lifecycle_events(&service);
        assert!(events.iter().any(|event| matches!(
            event,
            mesh_llm_events::logging::events::LifecycleEvent::StreamStarted { model }
                if model.as_deref() == Some(mesh_mixture_of_agents::VIRTUAL_MODEL_NAME)
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            mesh_llm_events::logging::events::LifecycleEvent::StreamCompleted {
                tokens: Some(5),
                usage: Some(recorded),
            } if *recorded == usage
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            mesh_llm_events::logging::events::LifecycleEvent::Completed {
                status_code: Some(200),
                usage: Some(recorded),
                ..
            } if *recorded == usage
        )));
    }
}

#[test]
fn passive_moa_stream_failure_records_stream_error_before_terminal_failure() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        RequestId::new(),
    )
    .expect("passive MoA test should claim one parent");
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let outcome = RouteDispatchOutcome::FailedWithStatus {
        status_code: 200,
        reason: "moa_turn_failed_after_commit",
    };

    record_moa_stream_lifecycle(
        attachment.route_observer(),
        ResponseAdapter::OpenAiResponsesStream,
        outcome,
    );
    attachment.terminal(outcome.terminal_outcome());

    let events = recorded_lifecycle_events(&service);
    let stream_error = events
        .iter()
        .position(|event| {
            matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::StreamError { .. }
            )
        })
        .expect("stream error");
    let terminal_failure = events
        .iter()
        .position(|event| {
            matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Failed {
                    status_code: Some(200),
                    ..
                }
            )
        })
        .expect("terminal failure");
    assert!(stream_error < terminal_failure);
}

#[tokio::test]
async fn transport_attempt_records_reuse_lifecycle_ids_and_keep_one_parent_terminal() {
    let sink = Arc::new(TransportProxySink::default());
    let service = Arc::new(LoggingService::new(
        Default::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(DeterministicClock::default()),
    ));
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        RequestId::new(),
    )
    .expect("transport test should claim one parent");
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let observer = attachment.route_observer();

    for (target, adapter, result) in [
        (
            "local",
            ResponseAdapter::OpenAiChatCompletionsJson,
            RouteAttemptResult::Delivered {
                status_code: 200,
                usage: None,
            },
        ),
        (
            "remote",
            ResponseAdapter::OpenAiResponsesStream,
            RouteAttemptResult::RetryableTimeout,
        ),
        (
            "remote",
            ResponseAdapter::OpenAiResponsesStream,
            RouteAttemptResult::Delivered {
                status_code: 502,
                usage: None,
            },
        ),
        (
            "external",
            ResponseAdapter::OpenAiChatCompletionsStream,
            RouteAttemptResult::RetryableUnavailable,
        ),
        (
            "none",
            ResponseAdapter::None,
            RouteAttemptResult::RetryableUnavailable,
        ),
    ] {
        let attempt = observer.start_proxy_attempt();
        finish_route_attempt(observer, attempt, target, adapter, &result);
    }
    attachment.terminal(crate::logging::TerminalOutcome::Completed);
    let lifecycle_attempt_ids: Vec<_> = recorded_lifecycle_events(&service)
        .into_iter()
        .filter_map(|event| match event {
            mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { attempt_id } => {
                attempt_id
            }
            _ => None,
        })
        .collect();
    assert_eq!(lifecycle_attempt_ids.len(), 5);
    assert!(
        lifecycle_attempt_ids
            .iter()
            .enumerate()
            .all(|(index, id)| !lifecycle_attempt_ids[..index].contains(id)),
        "each remote retry must retain a distinct lifecycle attempt ID"
    );

    let _ = service.pump_sync().await;
    let records = sink.proxy_records();
    assert_eq!(records.len(), 5, "one record per real transport attempt");
    assert_eq!(
        records
            .iter()
            .map(|record| record.attempt_id)
            .collect::<Vec<_>>(),
        lifecycle_attempt_ids
    );
    assert_eq!(sink.summary_count(), 1, "the parent owns the sole terminal");
    assert_eq!(
        records
            .iter()
            .map(|record| {
                (
                    record.target.as_str(),
                    record.provider.as_deref(),
                    record.engine.as_deref(),
                    record.status_code,
                    record.error.as_deref(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                "local",
                Some("local"),
                Some("chat_completion"),
                Some(200),
                None
            ),
            (
                "remote",
                Some("remote"),
                Some("responses_stream"),
                None,
                Some("timeout")
            ),
            (
                "remote",
                Some("remote"),
                Some("responses_stream"),
                Some(502),
                Some("upstream_status"),
            ),
            (
                "external",
                Some("external"),
                Some("chat_completion_stream"),
                None,
                Some("unavailable"),
            ),
            ("none", None, None, None, Some("unavailable")),
        ]
    );
    for record in records {
        // attempt_id/request_id are random UUIDs and the timestamps are clock output,
        // so they carry arbitrary hex and digits that collide with short numeric
        // tokens like the port below. Scan every other field — including any added
        // later — for leaked request data.
        let mut scanned = serde_json::to_value(&record).expect("serialize bounded record");
        let fields = scanned
            .as_object_mut()
            .expect("proxy record serializes to a JSON object");
        for generated in ["attempt_id", "request_id", "started_at", "completed_at"] {
            assert!(
                fields.remove(generated).is_some(),
                "expected generated field {generated} on the bounded record"
            );
        }
        let serialized = serde_json::to_string(&scanned).expect("serialize scanned fields");
        for forbidden in [
            "9337",
            "peer-id",
            "https://plugin.example/private/path",
            "plugin-name",
            "request body",
            "prompt text",
            "completion text",
            "connection refused",
        ] {
            assert!(!serialized.contains(forbidden), "record leaked {forbidden}");
        }
        assert!(!record.started_at.is_empty());
        assert!(record.completed_at.is_some());
    }
}

#[tokio::test]
async fn retry_then_stream_cancellation_keeps_one_metadata_only_parent() {
    let sink = Arc::new(TransportProxySink::default());
    let service = Arc::new(LoggingService::new(
        Default::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(crate::logging::SystemClock),
    ));
    let request_id = RequestId::new();
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        request_id,
    )
    .expect("transport test should claim one parent");
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let observer = attachment.route_observer();

    let first_attempt = observer.start_proxy_attempt();
    finish_route_attempt(
        observer,
        first_attempt,
        "remote",
        ResponseAdapter::OpenAiResponsesStream,
        &RouteAttemptResult::RetryableTimeout,
    );
    let second_attempt = observer.start_proxy_attempt();
    observer.stream_started(Some("safe-model"));
    observer.stream_first_token();
    finish_route_attempt(
        observer,
        second_attempt,
        "remote",
        ResponseAdapter::OpenAiResponsesStream,
        &RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
        },
    );
    observer.stream_cancelled();
    attachment.terminal(crate::logging::TerminalOutcome::Cancelled(Some(
        "client_disconnected".into(),
    )));
    attachment.terminal(crate::logging::TerminalOutcome::Completed);

    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Cancelled { .. }
            ))
            .count(),
        1,
        "the ingress parent emits one terminal cancellation"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Completed { .. }
            ))
            .count(),
        0,
        "a late terminal result cannot replace the client cancellation"
    );
    let attempt_events = events
        .iter()
        .filter_map(|event| match event {
            mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { attempt_id } => {
                Some(("started", *attempt_id))
            }
            mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed {
                attempt_id, ..
            } => Some(("failed", *attempt_id)),
            mesh_llm_events::logging::events::LifecycleEvent::AttemptCompleted {
                attempt_id,
                ..
            } => Some(("completed", *attempt_id)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        attempt_events
            .iter()
            .map(|(kind, _)| *kind)
            .collect::<Vec<_>>(),
        ["started", "failed", "started", "completed"]
    );
    assert_eq!(attempt_events[0].1, attempt_events[1].1);
    assert_eq!(attempt_events[2].1, attempt_events[3].1);
    assert_ne!(attempt_events[0].1, attempt_events[2].1);

    let _ = service.pump_sync().await;
    let records = sink.proxy_records();
    assert_eq!(
        sink.summary_count(),
        1,
        "only the parent persists a summary"
    );
    assert!(
        sink.artifact_pointers().is_empty(),
        "metadata-only retry and stream lifecycle paths never persist raw body artifacts"
    );
    assert_eq!(records.len(), 2, "both transport attempts are durable");
    assert!(records.iter().all(|record| record.request_id == request_id));
    assert!(records.iter().all(|record| record.completed_at.is_some()));
    assert_eq!(records[0].error.as_deref(), Some("timeout"));
    assert_eq!(records[1].status_code, Some(200));

    for record in service.bus_ref().replay_window().records {
        for forbidden in ["request body", "prompt text", "completion text"] {
            assert!(
                !record.entry.payload.contains(forbidden),
                "metadata-only replay leaked {forbidden}"
            );
        }
    }
}

#[test]
fn local_inference_attempt_failure_stays_under_one_parent() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        RequestId::new(),
    )
    .unwrap();
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let observer = attachment.route_observer();
    observer.route_selected(Some("safe-model"));

    let result = record_local_inference_attempt(observer, RouteAttemptResult::RetryableUnavailable);
    assert_eq!(result, RouteAttemptResult::RetryableUnavailable);

    // The ingress attachment remains the sole terminal owner even after
    // the local attempt has failed; a late terminal call is ignored.
    attachment.terminal(crate::logging::TerminalOutcome::Failed(
        "local_inference_unavailable".into(),
    ));
    attachment.terminal(crate::logging::TerminalOutcome::Completed);

    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Admitted { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::RouteSelected { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Failed { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Completed { .. }
            ))
            .count(),
        0
    );

    let attempt_id = events.iter().find_map(|event| match event {
        mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { attempt_id } => {
            *attempt_id
        }
        _ => None,
    });
    match events.iter().find(|event| {
        matches!(
            event,
            mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed { .. }
        )
    }) {
        Some(mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed {
            attempt_id: failed_id,
            error,
        }) => {
            assert_eq!(*failed_id, attempt_id);
            assert_eq!(error.as_deref(), Some("retryable_unavailable"));
        }
        other => panic!("expected one local attempt failure, got {other:?}"),
    }
    for record in service.bus_ref().replay_window().records {
        assert!(!record.entry.payload.contains("body"));
        assert!(!record.entry.payload.contains("prompt"));
        assert!(!record.entry.payload.contains("completion"));
    }
}

#[test]
fn local_inference_attempt_success_stays_under_one_parent() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        RequestId::new(),
    )
    .unwrap();
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let observer = attachment.route_observer();
    observer.route_selected(Some("safe-model"));

    let result = record_local_inference_attempt(
        observer,
        RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
        },
    );
    assert_eq!(
        result,
        RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
        }
    );

    // The ingress attachment remains the sole terminal owner after a
    // successful local attempt; a late failure call is ignored.
    attachment.terminal(crate::logging::TerminalOutcome::Completed);
    attachment.terminal(crate::logging::TerminalOutcome::Failed(
        "late_local_failure".into(),
    ));

    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Admitted { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::RouteSelected { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptCompleted { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed { .. }
            ))
            .count(),
        0
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Completed { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Failed { .. }
            ))
            .count(),
        0
    );

    let attempt_id = events.iter().find_map(|event| match event {
        mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { attempt_id } => {
            *attempt_id
        }
        _ => None,
    });
    match events.iter().find(|event| {
        matches!(
            event,
            mesh_llm_events::logging::events::LifecycleEvent::AttemptCompleted { .. }
        )
    }) {
        Some(mesh_llm_events::logging::events::LifecycleEvent::AttemptCompleted {
            attempt_id: completed_id,
            status_code,
        }) => {
            assert_eq!(*completed_id, attempt_id);
            assert_eq!(*status_code, Some(200));
        }
        other => panic!("expected one local attempt completion, got {other:?}"),
    }
    for record in service.bus_ref().replay_window().records {
        assert!(!record.entry.payload.contains("body"));
        assert!(!record.entry.payload.contains("prompt"));
        assert!(!record.entry.payload.contains("completion"));
    }
}

#[test]
fn remote_transports_record_target_failover_and_retry_under_one_parent() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        RequestId::new(),
    )
    .unwrap();
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let observer = attachment.route_observer();
    observer.route_selected(Some("safe-model"));

    // Deterministic target A failure followed by target B success.
    record_remote_transport_attempt(observer, RouteAttemptResult::RetryableUnavailable);
    record_remote_transport_attempt(
        observer,
        RouteAttemptResult::Delivered {
            status_code: 202,
            usage: None,
        },
    );

    // A fresh transport retry on the same remote target has its own
    // attempt ID rather than being hidden inside the retry loop.
    record_remote_transport_attempt(observer, RouteAttemptResult::RetryableTimeout);
    record_remote_transport_attempt(
        observer,
        RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
        },
    );

    attachment.terminal(crate::logging::TerminalOutcome::Completed);
    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Admitted { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::Completed { .. }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { .. }
            ))
            .count(),
        4
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed { .. }
            ))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                mesh_llm_events::logging::events::LifecycleEvent::AttemptCompleted { .. }
            ))
            .count(),
        2
    );
    let attempt_ids: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { attempt_id } => {
                *attempt_id
            }
            _ => None,
        })
        .collect();
    assert_eq!(attempt_ids.len(), 4);
    assert!(
        attempt_ids
            .iter()
            .enumerate()
            .all(|(index, attempt_id)| !attempt_ids[..index].contains(attempt_id))
    );
    let attempt_events: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            mesh_llm_events::logging::events::LifecycleEvent::AttemptStarted { attempt_id } => {
                Some(("started", *attempt_id))
            }
            mesh_llm_events::logging::events::LifecycleEvent::AttemptFailed {
                attempt_id, ..
            } => Some(("failed", *attempt_id)),
            mesh_llm_events::logging::events::LifecycleEvent::AttemptCompleted {
                attempt_id,
                ..
            } => Some(("completed", *attempt_id)),
            _ => None,
        })
        .collect();
    assert_eq!(
        attempt_events
            .iter()
            .map(|(kind, _)| *kind)
            .collect::<Vec<_>>(),
        [
            "started",
            "failed",
            "started",
            "completed",
            "started",
            "failed",
            "started",
            "completed",
        ]
    );
    for pair in attempt_events.chunks_exact(2) {
        assert_eq!(pair[0].1, pair[1].1);
    }
    for record in service.bus_ref().replay_window().records {
        assert!(!record.entry.payload.contains("body"));
        assert!(!record.entry.payload.contains("prompt"));
        assert!(!record.entry.payload.contains("completion"));
    }
}
