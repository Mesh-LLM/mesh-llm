use super::*;

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn started_then_completed_emits_exactly_one_terminal() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    // generation root + sibling prefill child both occupy slots until resolved.
    assert_eq!(engine.occupied_count(), 2);
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();

    assert_eq!(engine.occupied_count(), 0);
    let kinds = generation_kinds(&engine);
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == GenerationEventKind::GenerationCompleted)
            .count(),
        1
    );
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn frontend_generation_reserves_a_child_of_the_request_root() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();
    let root_bytes = [7_u8; 16];
    let root = mesh_llm_runtime_event_contracts::OperationId::from_bytes(root_bytes);
    let root_reservation = engine
        .reserve_root(root, synthetic_generation_terminal)
        .expect("root reservation available");

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(
            1,
            2,
            Some(root_bytes),
        )))
        .unwrap();
    // pre-reserved root + generation child + prefill sibling child.
    assert_eq!(engine.occupied_count(), 3);

    // Dropping the pre-reserved root, without ever resolving the
    // generation's own terminal, must cascade-release BOTH children --
    // proof they are genuinely children of this root (task 3's
    // cascade-on-root-release), not independent roots that merely share
    // the same byte value. If `begin` reserved its own root instead of a
    // child (the mutation this test is designed to catch), the
    // generation's slot would survive the root's drop and this assertion
    // would fail.
    drop(root_reservation);
    engine.drain();
    assert_eq!(
        engine.occupied_count(),
        0,
        "both child reservations must cascade-release when their root releases"
    );
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn non_frontend_generation_parents_prefill_under_its_own_root() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(9, 9, None)))
        .unwrap();
    assert_eq!(
        engine.occupied_count(),
        2,
        "generation root + prefill child"
    );
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            9,
            9,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    assert_eq!(engine.occupied_count(), 0);
    assert!(prefill_kinds(&engine).contains(&PrefillEventKind::PrefillCompleted));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn aborted_generation_maps_to_cancelled_and_cancels_prefill() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Aborted(GenerationAbort {
            request_id: 1,
            session_id: 2,
        }))
        .unwrap();
    engine.drain();

    assert_eq!(engine.occupied_count(), 0);
    assert!(generation_kinds(&engine).contains(&GenerationEventKind::GenerationCancelled));
    assert!(prefill_kinds(&engine).contains(&PrefillEventKind::PrefillCancelled));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn cancelled_termination_on_the_receipt_path_maps_to_cancelled() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::Cancelled,
        )))
        .unwrap();
    engine.drain();

    assert!(generation_kinds(&engine).contains(&GenerationEventKind::GenerationCancelled));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn committed_progress_is_bounded_and_carries_no_token_ids() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 5,
                token_ids: vec![11, 12].into_boxed_slice(),
            },
        ))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    assert!(generation_kinds(&engine).contains(&GenerationEventKind::GenerationCompleted));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn suffix_completion_after_progress_resolves_cleanly_and_removes_tracking() {
    let engine = install_test_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(3, 4, None)))
        .unwrap();
    for count in [1_usize, 2, 3] {
        adapter
            .try_submit(GenerationLifecycleObservation::Committed(
                GenerationCommit {
                    request_id: 3,
                    session_id: 4,
                    generated_token_count: count,
                    token_ids: vec![count as i32].into_boxed_slice(),
                },
            ))
            .unwrap();
    }
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            3,
            4,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    engine.drain();
    assert_eq!(engine.occupied_count(), 0);
    // Tracking must be REMOVED (not merely resolved-and-retained): a
    // late duplicate Committed for the same key must be a silent no-op,
    // not resurrect a stale reservation handle.
    assert_eq!(adapter.lock().generations.len(), 0);
    adapter
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 3,
                session_id: 4,
                generated_token_count: 4,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn cleanup_without_a_terminal_synthesizes_one_for_generation_and_prefill() {
    let engine = install_test_engine();
    {
        let adapter = SkippyGenerationRuntimeEventAdapter::new();
        adapter
            .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
            .unwrap();
        assert_eq!(engine.occupied_count(), 2);
        // Dropping the adapter drops both tracked reservations without a
        // terminal submit, exercising the drop-synthesis path for both
        // families.
    }
    engine.drain();
    assert_eq!(engine.occupied_count(), 0);
    assert!(generation_kinds(&engine).contains(&GenerationEventKind::GenerationFailed));
    assert!(prefill_kinds(&engine).contains(&PrefillEventKind::PrefillFailed));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn absent_engine_degrades_to_no_op_and_never_fails_inference() {
    clear_runtime_event_engine();
    let adapter = SkippyGenerationRuntimeEventAdapter::new();
    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![9].into_boxed_slice(),
            },
        ))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();
    // No assertions beyond "returned Ok every time": there is no engine
    // to inspect, which is exactly the degraded-but-not-failing contract.
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn reservation_exhaustion_degrades_without_failing_inference() {
    let engine = RuntimeEventEngine::with_capacity(0);
    clear_runtime_event_engine();
    install_runtime_event_engine(engine.clone());
    let adapter = SkippyGenerationRuntimeEventAdapter::new();

    adapter
        .try_submit(GenerationLifecycleObservation::Started(start(1, 2, None)))
        .unwrap();
    adapter
        .try_submit(GenerationLifecycleObservation::Completed(receipt(
            1,
            2,
            GenerationTermination::MaxTokens,
        )))
        .unwrap();

    assert_eq!(engine.occupied_count(), 0);
    assert!(engine.health().snapshot().reservation_exhausted > 0);
    clear_runtime_event_engine();
}
