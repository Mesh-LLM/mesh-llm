//! Request lifecycle, retry, replay, and registry invariant tests.

use super::*;

#[test]
fn test_one_terminal_per_outcome() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let outcomes = [
        TerminalOutcome::Completed,
        TerminalOutcome::Failed("timeout".into()),
        TerminalOutcome::Rejected(Some("invalid model".into())),
        TerminalOutcome::Cancelled(None),
        TerminalOutcome::Dropped(Some("queue full".into())),
    ];

    for outcome in &outcomes {
        let rid = RequestId::new();
        let (guard, _) = svc.register_request(rid);

        // First terminal transition should succeed.
        assert!(
            svc.transition_terminal(rid, &guard, outcome.clone())
                .is_ok()
        );

        assert!(
            svc.transition_terminal(rid, &guard, outcome.clone())
                .is_err(),
            "a second terminal transition must not emit another record"
        );
    }

    let events = recorded_lifecycle_events(&svc);
    assert_eq!(events.len(), outcomes.len());
    assert!(events.iter().all(|(_, event)| {
        matches!(
            event,
            LifecycleEvent::Completed { .. }
                | LifecycleEvent::Failed { .. }
                | LifecycleEvent::Rejected { .. }
                | LifecycleEvent::Cancelled { .. }
                | LifecycleEvent::Dropped { .. }
        )
    }));
    assert_eq!(svc.registry_ref().active_count(), 0);
    assert_eq!(svc.registry_ref().recent_count(), outcomes.len());
}

#[test]
fn test_duplicate_terminal_rejected() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    assert!(
        svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
            .is_ok()
    );

    // Second terminal → DuplicateTerminalError.
    let err = svc
        .transition_terminal(rid, &guard, TerminalOutcome::Failed("x".into()))
        .unwrap_err();
    assert_eq!(*err.existing, TerminalOutcome::Completed);
}

// ---------------------------------------------------------------------------
// Test Scenario 2: One summary with multiple retry attempts (parent not terminated by per-attempt)
// ---------------------------------------------------------------------------

#[test]
fn test_retry_attempts_under_one_summary() {
    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    // Simulate 3 retry attempts — each is typed and does NOT terminate the parent.
    let mut attempt_ids = Vec::new();
    for (index, status_code) in [502, 503, 200].into_iter().enumerate() {
        let attempt_id = svc.start_attempt(rid, &guard);
        svc.complete_attempt(rid, attempt_id, Some(status_code));
        attempt_ids.push(attempt_id);

        // Guard still active after each attempt.
        assert!(
            guard.is_active(),
            "guard should remain active during retry {}",
            index
        );
    }

    // Now terminate the parent request — exactly one terminal transition.
    assert!(
        svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
            .is_ok()
    );
    assert!(!guard.is_active());

    let events = recorded_lifecycle_events(&svc);
    assert_eq!(events.len(), 7);
    for (index, attempt_id) in attempt_ids.iter().enumerate() {
        assert_eq!(events[index * 2].0["request_id"], rid.as_uuid().to_string());
        assert_eq!(
            events[index * 2].1,
            LifecycleEvent::AttemptStarted {
                attempt_id: Some(*attempt_id)
            }
        );
        assert_eq!(
            events[index * 2 + 1].0["request_id"],
            rid.as_uuid().to_string()
        );
    }
    assert!(matches!(
        events.last().unwrap().1,
        LifecycleEvent::Completed { .. }
    ));
}

#[test]
fn retry_failure_then_success_does_not_terminalize_parent() {
    let svc = make_service();
    let request_id = RequestId::new();
    let (guard, _) = svc.register_request(request_id);

    let failed_attempt = svc.start_attempt(request_id, &guard);
    svc.fail_attempt(request_id, failed_attempt, "upstream timeout".into());
    assert!(guard.is_active());
    assert_eq!(svc.registry_ref().active_count(), 1);
    assert_eq!(svc.registry_ref().recent_count(), 0);

    let successful_attempt = svc.start_attempt(request_id, &guard);
    svc.complete_attempt(request_id, successful_attempt, Some(200));
    assert!(guard.is_active());

    svc.transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .unwrap();
    let events = recorded_lifecycle_events(&svc);
    assert!(matches!(events[1].1, LifecycleEvent::AttemptFailed { .. }));
    assert!(matches!(
        events[3].1,
        LifecycleEvent::AttemptCompleted { .. }
    ));
    assert!(matches!(events[4].1, LifecycleEvent::Completed { .. }));
    assert_eq!(svc.registry_ref().active_count(), 0);
    assert_eq!(svc.registry_ref().recent_count(), 1);
}

#[test]
fn dropping_intermediate_guard_clone_does_not_terminalize_request() {
    let svc = make_service();
    let request_id = RequestId::new();
    let (guard, _) = svc.register_request(request_id);
    let intermediate = guard.clone();

    drop(intermediate);

    assert!(guard.is_active());
    assert!(recorded_lifecycle_events(&svc).is_empty());
    assert_eq!(svc.registry_ref().active_count(), 1);
    assert_eq!(svc.registry_ref().recent_count(), 0);
    svc.transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .unwrap();
}

#[test]
fn dropping_final_guard_emits_one_dropped_terminal_record() {
    let svc = make_service();
    let request_id = RequestId::new();
    let (guard, _) = svc.register_request(request_id);

    drop(guard);

    let events = recorded_lifecycle_events(&svc);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0["request_id"], request_id.as_uuid().to_string());
    assert!(matches!(events[0].1, LifecycleEvent::Dropped { .. }));
    assert_eq!(svc.registry_ref().active_count(), 0);
    assert_eq!(svc.registry_ref().recent_count(), 1);
}

#[test]
fn concurrent_terminal_and_final_drop_emit_exactly_one_terminal_record() {
    let service = Arc::new(make_service());
    let request_id = RequestId::new();
    let (guard, _) = service.register_request(request_id);
    let thread_guard = guard.clone();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let thread_service = Arc::clone(&service);
    let thread_barrier = Arc::clone(&barrier);

    let worker = std::thread::spawn(move || {
        thread_barrier.wait();
        let _ = thread_service.transition_terminal(
            request_id,
            &thread_guard,
            TerminalOutcome::Completed,
        );
        drop(thread_guard);
    });

    barrier.wait();
    drop(guard);
    worker.join().expect("terminal worker must join");

    let events = recorded_lifecycle_events(&service);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0].1,
        LifecycleEvent::Completed { .. } | LifecycleEvent::Dropped { .. }
    ));
    assert_eq!(service.registry_ref().active_count(), 0);
    assert_eq!(service.registry_ref().recent_count(), 1);
}

#[tokio::test]
async fn failed_recorder_during_final_drop_is_fail_open_and_counted() {
    let (sink, mut attempts) = TestSink::failing_with_attempt_notifications();
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    assert!(service.spawn());
    let request_id = RequestId::new();
    let (guard, _) = service.register_request(request_id);

    let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(guard)));
    assert!(dropped.is_ok(), "Drop must not propagate recorder failures");
    for _ in 0..3 {
        attempts
            .recv()
            .await
            .expect("original or fallback persistence attempt");
    }
    assert!(service.shutdown().await);
    assert_eq!(
        service.persistence_failures(),
        6,
        "each failed summary/admitted/terminal delivery produces one failed fallback, never a loop"
    );
    assert_eq!(service.registry_ref().active_count(), 0);
    assert_eq!(service.registry_ref().recent_count(), 1);
}

// ---------------------------------------------------------------------------
// Test Scenario 3: Monotonic channel sequences across many events
// ---------------------------------------------------------------------------

#[test]
fn test_monotonic_channel_sequences() {
    let svc = make_service();

    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    // Emit 100 events on each channel. Sequences must be strictly increasing
    // per channel and leave every other channel unchanged.
    for ch in [
        ReplayChannel::Requests,
        ReplayChannel::Operations,
        ReplayChannel::System,
    ] {
        let before = [
            (
                ReplayChannel::Requests,
                svc.sequences_ref().current(ReplayChannel::Requests),
            ),
            (
                ReplayChannel::Operations,
                svc.sequences_ref().current(ReplayChannel::Operations),
            ),
            (
                ReplayChannel::System,
                svc.sequences_ref().current(ReplayChannel::System),
            ),
        ];
        let mut prev_seq = svc.sequences_ref().current(ch);
        for _i in 0..100 {
            svc.enqueue_event(rid, ch, "test".into()).unwrap();
            // The sequence generator is internal to the service — verify via sequences_ref.
            let current = svc.sequences_ref().current(ch);
            assert!(
                current > prev_seq,
                "sequence must be strictly increasing on {:?}",
                ch
            );
            prev_seq = current;
        }

        for (other_ch, other_before) in before {
            let other_current = svc.sequences_ref().current(other_ch);
            let expected = if other_ch == ch {
                other_before + 100
            } else {
                other_before
            };
            assert_eq!(
                other_current, expected,
                "events on {:?} must not change {:?}",
                ch, other_ch
            );
        }
    }

    // Verify sequences survive guard cloning.
}

#[test]
fn test_sequences_survive_guard_clone() {
    let svc = make_service();

    let rid = RequestId::new();
    let (guard1, _) = svc.register_request(rid);
    let _guard2 = guard1.clone(); // Clone the guard — sequences are independent of guards.

    // Emit events via service after cloning.
    for i in 0..5 {
        let _payload = serde_json::json!({ "i": i }).to_string();
        svc.enqueue_event(rid, ReplayChannel::Requests, _payload)
            .unwrap();
    }

    assert_eq!(svc.sequences_ref().current(ReplayChannel::Requests), 6);
}

// ---------------------------------------------------------------------------
// Test Scenario 4: Bounded replay eviction (overflow drops + counter increments)
// ---------------------------------------------------------------------------

#[test]
fn test_bounded_replay_eviction() {
    let svc = make_service();

    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    // Emit more events than the bus capacity (128). This triggers drop-oldest evictions.
    for i in 0..200 {
        let payload = serde_json::json!({ "i": i }).to_string();
        assert!(
            svc.enqueue_event(rid, ReplayChannel::Requests, payload)
                .is_ok()
        );
    }

    // Bus should be at capacity.
    assert_eq!(svc.bus_ref().len(), 128);

    // The admitted event is present before the 200 explicit events, so 73 old
    // replay entries were evicted; all 200 new events were accepted.
    let evictions = svc.bus_ref().evictions.load(AtomicOrdering::Relaxed);
    assert_eq!(evictions, 73);
    assert_eq!(svc.bus_ref().drops.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(svc.total_drops(), 0);

    // Queue never exceeds capacity.
}

#[test]
fn test_queue_never_exceeds_capacity() {
    let svc = make_service();
    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    for i in 0..10_000 {
        let payload = serde_json::json!({ "i": i }).to_string();
        assert!(
            svc.enqueue_event(rid, ReplayChannel::Requests, payload)
                .is_ok()
        );
        // Invariant: bus never exceeds capacity.
        assert!(
            svc.bus_ref().len() <= 128,
            "bus exceeded capacity at iteration {}",
            i
        );
    }

    // Request path completes despite overflow — no blocking or panic.
}

// ---------------------------------------------------------------------------
// Test Scenario 5: Active → recent movement on terminal transition
// ---------------------------------------------------------------------------

#[test]
fn test_active_to_recent_movement() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    assert_eq!(svc.registry_ref().active_count(), 1);
    assert_eq!(svc.registry_ref().recent_count(), 0);

    // Transition to terminal → moves from active to recent.
    svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
        .unwrap();

    assert_eq!(svc.registry_ref().active_count(), 0);
    assert_eq!(svc.registry_ref().recent_count(), 1);
}

#[test]
fn test_active_to_recent_preserves_created_at() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    // Get the active entry's created_at.
    let rid_str = rid.as_uuid().to_string();
    let active_entry = svc.registry_ref().get_active(&rid_str).unwrap();
    let original_created_at = active_entry.created_at.clone();

    // Transition to terminal.
    svc.transition_terminal(rid, &guard, TerminalOutcome::Failed("err".into()))
        .unwrap();

    // Recent entry should preserve created_at.
    let recent_entry = svc.registry_ref().get_recent(&rid_str).unwrap();
    assert_eq!(recent_entry.created_at, original_created_at);
}

// ---------------------------------------------------------------------------
// Test Scenario 6: No registry leak (registry empties when all entries evict)
// ---------------------------------------------------------------------------

#[test]
fn test_no_registry_leak() {
    use crate::logging::lifecycle::TerminalOutcome;

    let config = ServiceConfig {
        queue_capacity: 10,
        event_buffer_size: 10,
        artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
        registry_config: RegistryConfig {
            max_active: 2,
            max_recent: 3,
        },
        ..ServiceConfig::default()
    };

    let svc = LoggingService::new(
        config.clone(),
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    );

    // Register many requests — all should eventually evict from both sets.
    for i in 0..50 {
        let rid = RequestId::new();
        let (guard, _) = svc.register_request(rid);

        if i % 2 == 0 {
            // Every other request transitions to terminal → moves active→recent.
            svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
                .unwrap();
        }
    }

    assert!(svc.registry_ref().active_count() <= config.registry_config.max_active);
    assert!(svc.registry_ref().recent_count() <= config.registry_config.max_recent);

    // Clear the registry — should become empty.
    svc.registry_ref().clear();
    assert!(svc.registry_ref().is_empty());
}

#[test]
fn test_registry_eviction_counters_increment() {
    use crate::logging::lifecycle::TerminalOutcome;

    let config = ServiceConfig {
        queue_capacity: 10,
        event_buffer_size: 10,
        artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
        registry_config: RegistryConfig {
            max_active: 2,
            max_recent: 2,
        },
        ..ServiceConfig::default()
    };

    let svc = LoggingService::new(
        config.clone(),
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    );

    for i in 0..20 {
        let rid = RequestId::new();
        let (guard, _) = svc.register_request(rid);
        if i % 3 == 0 {
            svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
                .unwrap();
        }
    }

    // Every loop-local guard now terminalizes as Dropped when it is the final
    // handle, so bounded pressure can land in recent rather than active.
    assert!(
        svc.registry_ref()
            .active_evictions
            .load(AtomicOrdering::Relaxed)
            + svc
                .registry_ref()
                .recent_evictions
                .load(AtomicOrdering::Relaxed)
            > 0
    );
}
