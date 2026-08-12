use super::*;
use crate::logging::{OpenAiLifecycleAttachment, ServiceConfig};

fn recorded_events(service: &LoggingService) -> Vec<LifecycleEvent> {
    service
        .bus_ref()
        .replay_window()
        .records
        .into_iter()
        .filter_map(|record| {
            let envelope = serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
            let payload = envelope.get("payload")?.as_str()?;
            serde_json::from_str::<LifecycleEvent>(payload).ok()
        })
        .collect()
}

#[test]
fn raw_mesh_lifecycle_orders_metadata_events_without_payloads() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let lifecycle = RawMeshRequestLifecycle::register_with_metadata(
        service.clone(),
        owners,
        request_id,
        RequestSummaryMetadata::from_openai_ingress_path("/v1/chat/completions?ignored"),
    )
    .unwrap();

    lifecycle.route_selected(Some("safe-model"));
    lifecycle.terminal(TerminalOutcome::Completed);

    let events = recorded_events(&service);
    assert!(matches!(events[0], LifecycleEvent::Admitted { .. }));
    assert!(matches!(events[1], LifecycleEvent::RouteSelected { .. }));
    assert!(matches!(events[2], LifecycleEvent::Completed { .. }));
    let route_event = serde_json::to_value(&events[1]).unwrap();
    assert_eq!(route_event["model"], "safe-model");
    let summary = service
        .registry_ref()
        .get_recent(&request_id.as_uuid().to_string())
        .expect("terminal request summary");
    assert_eq!(summary.metadata.route(), Some("chat_completions"));
    assert_eq!(summary.metadata.model(), Some("safe-model"));
    assert_eq!(summary.metadata.provider(), Some("mesh"));
    assert_eq!(summary.metadata.engine(), Some("raw_ingress"));
    for payload_field in ["body", "headers", "prompt", "artifacts"] {
        assert!(route_event.get(payload_field).is_none());
    }
}

#[test]
fn stream_phases_record_one_first_token_and_metadata_only() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let parent = RawMeshRequestLifecycle::register(service.clone(), owners, request_id).unwrap();
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    {
        let observer = attachment.route_observer();
        observer.route_selected(Some("safe-model"));
        observer.stream_started(Some("safe model/secret"));
        observer.stream_started(Some("secret-model-name"));
        observer.stream_first_token();
        for _ in 0..260 {
            observer.stream_chunk();
        }
        observer.stream_completed(None);
        observer.stream_completed(None);
    }
    attachment.terminal(TerminalOutcome::Completed);
    {
        let observer = attachment.route_observer();
        observer.stream_started(Some("post_terminal_model"));
        observer.stream_chunk();
        observer.stream_completed(None);
        observer.stream_error("post_terminal_error");
    }
    attachment.terminal(TerminalOutcome::Failed("late_raw_failure".into()));

    let events = recorded_events(&service);
    assert!(matches!(events[0], LifecycleEvent::Admitted { .. }));
    assert!(matches!(events[1], LifecycleEvent::RouteSelected { .. }));
    assert!(matches!(events[2], LifecycleEvent::StreamStarted { .. }));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::StreamStarted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::StreamChunk { .. }))
            .count(),
        1
    );
    let completed_index = events
        .iter()
        .position(|event| matches!(event, LifecycleEvent::StreamCompleted { .. }))
        .expect("stream completion should be recorded");
    let terminal_index = events
        .iter()
        .position(|event| matches!(event, LifecycleEvent::Completed { .. }))
        .expect("request completion should be recorded");
    assert!(completed_index > 2);
    assert!(terminal_index > completed_index);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::StreamCompleted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
            .count(),
        1
    );

    let serialized = serde_json::to_string(&events).expect("phase events should serialize");
    for forbidden in [
        "raw_prompt",
        "secret_chunk",
        "authorization",
        "late_raw_failure",
        "safe model/secret",
        "secret-model-name",
        "post_terminal_model",
        "post_terminal_error",
    ] {
        assert!(
            !serialized.contains(forbidden),
            "found forbidden data: {forbidden}"
        );
    }
    assert!(events.iter().all(|event| {
        !matches!(
            event,
            LifecycleEvent::StreamChunk { tokens: Some(_) }
                | LifecycleEvent::StreamCompleted {
                    tokens: Some(_),
                    ..
                }
        )
    }));
}

#[test]
fn stream_completion_tokens_propagate_only_when_bounded_and_available() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    for usage in [
        Some(mesh_llm_events::logging::events::TokenUsage {
            prompt_tokens: Some(0),
            completion_tokens: Some(42),
            total_tokens: Some(42),
        }),
        None,
        Some(mesh_llm_events::logging::events::TokenUsage {
            prompt_tokens: Some(u64::MAX),
            completion_tokens: Some(1),
            total_tokens: Some(u64::MAX),
        }),
    ] {
        let request_id = RequestId::new();
        let parent =
            RawMeshRequestLifecycle::register(service.clone(), owners.clone(), request_id).unwrap();
        let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
        let observer = attachment.route_observer();
        observer.stream_started(None);
        observer.stream_first_token();
        observer.stream_completed(usage);
        attachment.terminal(TerminalOutcome::Completed);
    }

    let events = recorded_events(&service);
    let completion_tokens = events
        .iter()
        .filter_map(|event| match event {
            LifecycleEvent::StreamCompleted { tokens, .. } => Some(*tokens),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(completion_tokens, [Some(42), None, None]);
    let structured_usage = events
        .iter()
        .filter_map(|event| match event {
            LifecycleEvent::StreamCompleted { usage, .. } => Some(*usage),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        structured_usage,
        [
            Some(mesh_llm_events::logging::events::TokenUsage {
                prompt_tokens: Some(0),
                completion_tokens: Some(42),
                total_tokens: Some(42),
            }),
            None,
            None,
        ]
    );

    let serialized = serde_json::to_string(&events).expect("phase events should serialize");
    assert!(serialized.contains("\"tokens\":42"));
    assert!(serialized.contains("\"usage\""));
    assert!(!serialized.contains(&u64::MAX.to_string()));
    for forbidden in ["completion_text", "authorization"] {
        assert!(!serialized.contains(forbidden));
    }
}

#[test]
fn stream_cancellation_emits_one_bounded_error_before_terminal_cancel() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let parent = RawMeshRequestLifecycle::register(service.clone(), owners, request_id).unwrap();
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    {
        let observer = attachment.route_observer();
        observer.stream_started(None);
        observer.stream_first_token();
        observer.stream_cancelled();
        observer.stream_error("raw_upstream_error");
    }
    attachment.terminal(TerminalOutcome::Cancelled(Some(
        "client_disconnected".into(),
    )));
    {
        let observer = attachment.route_observer();
        observer.stream_error("post_terminal_error");
        observer.stream_completed(None);
    }
    attachment.terminal(TerminalOutcome::Completed);

    let events = recorded_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::StreamError { .. }))
            .count(),
        1
    );
    assert!(matches!(
        events
            .iter()
            .find(|event| matches!(event, LifecycleEvent::StreamError { .. }))
            .expect("stream error"),
        LifecycleEvent::StreamError {
            error: Some(label)
        } if label == "client_disconnected"
    ));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Cancelled { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
            .count(),
        0
    );
    let serialized = serde_json::to_string(&events).expect("phase events should serialize");
    assert!(!serialized.contains("raw_upstream_error"));
    assert!(!serialized.contains("post_terminal_error"));
}

#[test]
fn stream_retry_resets_phase_state_without_replacing_parent() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let parent = RawMeshRequestLifecycle::register(service.clone(), owners, request_id).unwrap();
    let mut attachment = OpenAiLifecycleAttachment::new(Some(parent));
    let observer = attachment.route_observer();

    observer.stream_started(Some("target-a"));
    observer.stream_first_token();
    observer.stream_error("upstream_status");
    observer.stream_started(Some("target-b"));
    observer.stream_first_token();
    observer.stream_chunk();
    observer.stream_completed(None);
    attachment.terminal(TerminalOutcome::Completed);

    let events = recorded_events(&service);
    let phase_types = events
        .iter()
        .filter_map(|event| match event {
            LifecycleEvent::StreamStarted { .. } => Some("started"),
            LifecycleEvent::StreamChunk { .. } => Some("chunk"),
            LifecycleEvent::StreamError { .. } => Some("error"),
            LifecycleEvent::StreamCompleted { .. } => Some("completed"),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        phase_types,
        ["started", "chunk", "error", "started", "chunk", "completed"]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Admitted { .. }))
            .count(),
        1
    );
}

#[test]
fn raw_mesh_owner_reuses_one_guard_and_terminalizes_once() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let first =
        RawMeshRequestLifecycle::register(service.clone(), owners.clone(), request_id).unwrap();
    let second =
        RawMeshRequestLifecycle::register(service.clone(), owners.clone(), request_id).unwrap();

    first.route_selected(None);
    second.route_selected(None);
    first.terminal(TerminalOutcome::Completed);
    second.terminal(TerminalOutcome::Failed("late".into()));

    let events = recorded_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Admitted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
            .count(),
        1
    );
    assert!(!owners.is_claimed(request_id));
}

#[test]
fn claimed_plan_failure_terminalizes_without_route_selected_event() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let lifecycle = RawMeshRequestLifecycle::register(service.clone(), owners, request_id).unwrap();

    // The raw handler claims after parsing; a later planning failure must
    // still finish the admitted parent without fabricating route metadata.
    lifecycle.terminal(TerminalOutcome::Failed("no_hosts_available".into()));

    let events = recorded_events(&service);
    assert!(matches!(events[0], LifecycleEvent::Admitted { .. }));
    assert!(matches!(events[1], LifecycleEvent::Failed { .. }));
    assert!(
        events
            .iter()
            .all(|event| !matches!(event, LifecycleEvent::RouteSelected { .. }))
    );
}

#[test]
fn bounded_owner_registry_drops_overflow_without_partial_lifecycle() {
    let service = Arc::new(LoggingService::new_disabled(ServiceConfig {
        replay_capacity: MAX_RAW_MESH_LIFECYCLE_OWNERS + 1,
        ..Default::default()
    }));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let mut lifecycles = Vec::with_capacity(MAX_RAW_MESH_LIFECYCLE_OWNERS);
    for _ in 0..MAX_RAW_MESH_LIFECYCLE_OWNERS {
        lifecycles.push(
            RawMeshRequestLifecycle::register(service.clone(), owners.clone(), RequestId::new())
                .unwrap(),
        );
    }

    let overflow_request_id = RequestId::new();
    assert!(
        RawMeshRequestLifecycle::register(service.clone(), owners.clone(), overflow_request_id,)
            .is_none()
    );
    assert!(!owners.is_claimed(overflow_request_id));
    assert_eq!(
        recorded_events(&service)
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Admitted { .. }))
            .count(),
        MAX_RAW_MESH_LIFECYCLE_OWNERS
    );

    for lifecycle in lifecycles {
        lifecycle.terminal(TerminalOutcome::Completed);
    }
}

#[test]
fn stale_duplicate_handle_cannot_release_a_newer_owner() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();
    let first =
        RawMeshRequestLifecycle::register(service.clone(), owners.clone(), request_id).unwrap();
    let stale_duplicate =
        RawMeshRequestLifecycle::register(service.clone(), owners.clone(), request_id).unwrap();

    first.terminal(TerminalOutcome::Completed);
    let replacement =
        RawMeshRequestLifecycle::register(service, owners.clone(), request_id).unwrap();
    stale_duplicate.terminal(TerminalOutcome::Failed("late".into()));
    assert!(owners.is_claimed(request_id));

    replacement.terminal(TerminalOutcome::Completed);
    assert!(!owners.is_claimed(request_id));
}

#[test]
fn remote_suppression_is_metadata_free_and_releases_after_the_last_lease() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let request_id = RequestId::new();

    let first = RawMeshRemoteSuppressionLease::acquire(Arc::clone(&owners), request_id)
        .expect("first suppression lease should fit");
    let second = RawMeshRemoteSuppressionLease::acquire(Arc::clone(&owners), request_id)
        .expect("duplicate suppression lease should share the marker");

    assert!(owners.is_claimed(request_id));
    assert!(recorded_events(&service).is_empty());

    drop(first);
    assert!(owners.is_claimed(request_id));
    drop(second);
    assert!(!owners.is_claimed(request_id));
    assert!(recorded_events(&service).is_empty());
}

#[test]
fn remote_suppression_cap_fails_open_without_registering_parents() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let owners = Arc::new(RawMeshLifecycleOwners::default());
    let mut leases = Vec::with_capacity(MAX_RAW_MESH_LIFECYCLE_OWNERS);
    for _ in 0..MAX_RAW_MESH_LIFECYCLE_OWNERS {
        leases.push(
            RawMeshRemoteSuppressionLease::acquire(Arc::clone(&owners), RequestId::new())
                .expect("bounded lease should fit"),
        );
    }

    let overflow_request_id = RequestId::new();
    assert!(
        RawMeshRemoteSuppressionLease::acquire(Arc::clone(&owners), overflow_request_id,).is_none()
    );
    assert!(!owners.is_claimed(overflow_request_id));
    assert!(recorded_events(&service).is_empty());

    drop(leases);
}
