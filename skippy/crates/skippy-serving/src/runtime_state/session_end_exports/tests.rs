//! The modelless test runtime has no native sessions, so these cover the
//! bookkeeping around owed exports; parking a live lane is exercised by the
//! split benchmarks.

use std::sync::{Arc, Mutex};

use super::super::panic_recovery::PanicRecovery;
use super::*;

/// What each owed export's sink saw: `Some(session)` when it ran with the
/// session's rows resident, `None` when it was abandoned.
type Seen = Arc<Mutex<Vec<Option<String>>>>;

fn sink(seen: &Seen) -> SessionEndKvExportSink {
    let seen = Arc::clone(seen);
    Box::new(move |session| {
        // A failed assertion poisons `seen` while the runtime still owes
        // this export; dropping the runtime must not then abort the run.
        seen.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(session.map(|(_, session_id)| session_id.to_string()));
    })
}

/// A runtime tracking `request` at `tokens` that owes an export of rows
/// `[0, boundary)`.
fn runtime_owing_export(tokens: u64, boundary: u64, seen: &Seen) -> RuntimeState {
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    runtime.track_session_tokens_for_test("request", tokens);
    runtime.export_kv_at_session_end("request", boundary, sink(seen));
    runtime
}

#[test]
fn completing_runs_the_export_while_the_rows_are_resident() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(130, 100, &seen);

    runtime.complete_session_end_kv_exports("request");

    assert_eq!(*seen.lock().unwrap(), vec![Some("request".to_string())]);
    assert!(runtime.session_end_kv_exports.is_empty());
}

#[test]
fn a_session_shorter_than_its_boundary_abandons_the_export() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(90, 100, &seen);

    runtime.complete_session_end_kv_exports("request");

    assert_eq!(*seen.lock().unwrap(), vec![None]);
}

#[test]
fn an_error_drop_abandons_the_export() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(130, 100, &seen);

    runtime.drop_session_timed("request").unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![None]);
    assert!(runtime.session_end_kv_exports.is_empty());
}

#[test]
fn ending_a_session_without_a_lane_drops_it_and_abandons_the_export() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(130, 100, &seen);

    runtime.end_session_timed("request").unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![None]);
    assert!(runtime.parked_session_ends.is_empty());
}

#[test]
fn a_trim_abandons_only_the_exports_whose_rows_it_removes() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(130, 100, &seen);
    runtime.export_kv_at_session_end("request", 120, sink(&seen));

    runtime.abandon_session_end_kv_exports_above("request", 110);
    assert_eq!(*seen.lock().unwrap(), vec![None]);

    runtime.complete_session_end_kv_exports("request");
    assert_eq!(
        *seen.lock().unwrap(),
        vec![None, Some("request".to_string())]
    );
}

#[test]
fn the_longest_export_runs_first() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    runtime.track_session_tokens_for_test("request", 130);
    for boundary in [100, 120, 110] {
        let order = Arc::clone(&order);
        runtime.export_kv_at_session_end(
            "request",
            boundary,
            Box::new(move |_| order.lock().unwrap().push(boundary)),
        );
    }

    runtime.complete_session_end_kv_exports("request");

    assert_eq!(*order.lock().unwrap(), vec![120, 110, 100]);
}

#[test]
fn a_restore_abandons_the_export() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(130, 100, &seen);

    let _ = runtime.restore_transaction("request", |_| Ok(()));

    assert_eq!(*seen.lock().unwrap(), vec![None]);
}

#[test]
fn panic_recovery_abandons_every_owed_export() {
    let seen = Seen::default();
    let mut runtime = runtime_owing_export(130, 100, &seen);
    runtime.track_session_tokens_for_test("other", 50);
    runtime.export_kv_at_session_end("other", 40, sink(&seen));

    runtime.reset_after_panic();

    assert_eq!(*seen.lock().unwrap(), vec![None, None]);
    assert!(runtime.session_end_kv_exports.is_empty());
}

/// Parks `session_id` on a lane with no native state that owes an export of
/// rows `[0, 100)`.
fn park_with_export(runtime: &mut RuntimeState, session_id: &str, sink: SessionEndKvExportSink) {
    let index = runtime.next_lane_index;
    runtime.next_lane_index += 1;
    runtime.sessions.insert(
        session_id.to_string(),
        RuntimeLaneSession {
            index,
            session: skippy_runtime::StageSession::detached_dflash_for_test(),
            resident_prefix: None,
        },
    );
    runtime.track_session_tokens_for_test(session_id, 130);
    runtime.export_kv_at_session_end(session_id, 100, sink);
    runtime.end_session_timed(session_id).unwrap();
}

#[test]
fn a_panicking_sink_leaves_the_other_parked_lanes_for_panic_recovery() {
    let seen = Seen::default();
    let mut runtime = RuntimeState::new_modelless_for_test(2);
    park_with_export(&mut runtime, "first", sink(&seen));
    park_with_export(
        &mut runtime,
        "second",
        Box::new(|_| panic!("session-end export sink panicked")),
    );
    assert_eq!(runtime.parked_session_ends.len(), 2);

    let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.complete_parked_session_ends();
    }));
    assert!(completed.is_err(), "precondition: the sink panicked");
    // What the runtime lock does when it recovers from the poison.
    runtime.reset_after_panic();

    assert_eq!(*seen.lock().unwrap(), vec![None]);
    assert!(runtime.parked_session_ends.is_empty());
    assert!(runtime.sessions.is_empty());
    let mut free = runtime.free_lane_indices.clone();
    free.sort_unstable();
    assert_eq!(free, vec![0, 1], "both lane indices are reclaimed");
    for _ in 0..2 {
        super::super::lane_lifecycle::create_indexed_lane_resource(
            &mut runtime.next_lane_index,
            &mut runtime.free_lane_indices,
            runtime.lane_count,
            || Ok(()),
        )
        .expect("a later admission finds a free lane");
    }
}

#[test]
fn a_panicking_sink_for_a_parked_duplicate_of_a_live_session_keeps_its_lane() {
    let mut runtime = RuntimeState::new_modelless_for_test(2);
    park_with_export(
        &mut runtime,
        "request",
        Box::new(|_| panic!("session-end export sink panicked")),
    );
    // A live session under the same id; the parked one is abandoned.
    let index = runtime.next_lane_index;
    runtime.next_lane_index += 1;
    runtime.sessions.insert(
        "request".to_string(),
        RuntimeLaneSession {
            index,
            session: skippy_runtime::StageSession::detached_dflash_for_test(),
            resident_prefix: None,
        },
    );

    let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.complete_parked_session_ends();
    }));
    assert!(completed.is_err(), "precondition: the sink panicked");
    assert_eq!(
        runtime.free_lane_indices,
        vec![0],
        "the parked lane is back before its sink runs"
    );
    assert!(
        runtime.sessions.contains_key("request"),
        "the live session stays"
    );

    runtime.reset_after_panic();
    let mut free = runtime.free_lane_indices.clone();
    free.sort_unstable();
    assert_eq!(free, vec![0, 1]);
}

/// Puts `session_id` on a lane with no native state, as a live session.
fn activate(runtime: &mut RuntimeState, session_id: &str) -> usize {
    let index = runtime.next_lane_index;
    runtime.next_lane_index += 1;
    runtime.sessions.insert(
        session_id.to_string(),
        RuntimeLaneSession {
            index,
            session: skippy_runtime::StageSession::detached_dflash_for_test(),
            resident_prefix: None,
        },
    );
    runtime.track_session_tokens_for_test(session_id, 10);
    index
}

/// A parked session's cells stay claimed in the shared pool until its
/// exports run. An operation on another, already active session must run
/// them and release the lane first, or that session's decode can find the
/// pool full (`decode: failed to find a memory slot`).
#[test]
fn an_operation_on_an_active_session_completes_parked_sessions_first() {
    let seen = Seen::default();
    let mut runtime = RuntimeState::new_modelless_for_test(2);
    activate(&mut runtime, "active");
    park_with_export(&mut runtime, "parked", sink(&seen));
    let parked_index = runtime.parked_session_ends[0].lane.index;

    runtime.ensure_session_active("active").unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![Some("parked".to_string())]);
    assert!(runtime.parked_session_ends.is_empty());
    assert!(!runtime.sessions.contains_key("parked"));
    assert!(runtime.free_lane_indices.contains(&parked_index));
    assert!(runtime.sessions.contains_key("active"));
}

#[test]
fn an_active_session_lookup_completes_parked_sessions_first() {
    let seen = Seen::default();
    let mut runtime = RuntimeState::new_modelless_for_test(2);
    activate(&mut runtime, "active");
    park_with_export(&mut runtime, "parked", sink(&seen));

    let _ = runtime.active_session("active").unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![Some("parked".to_string())]);
    assert!(runtime.parked_session_ends.is_empty());
}

/// A sink reaches the runtime through the same session lookup; that must
/// not start completing the other parked sessions inside it.
#[test]
fn a_sink_using_the_runtime_does_not_complete_other_parked_sessions() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = RuntimeState::new_modelless_for_test(2);
    let first = Arc::clone(&order);
    park_with_export(
        &mut runtime,
        "first",
        Box::new(move |_| first.lock().unwrap().push("first")),
    );
    let second = Arc::clone(&order);
    park_with_export(
        &mut runtime,
        "second",
        Box::new(move |session| {
            let (runtime, session_id) = session.expect("rows resident");
            runtime.ensure_session_active(session_id).unwrap();
            second.lock().unwrap().push("second");
        }),
    );

    runtime.complete_parked_session_ends();

    assert_eq!(*order.lock().unwrap(), vec!["second", "first"]);
    assert!(runtime.parked_session_ends.is_empty());
    assert!(!runtime.completing_parked_session_ends);
}

#[test]
fn ending_a_parked_session_again_keeps_its_export() {
    let seen = Seen::default();
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    park_with_export(&mut runtime, "request", sink(&seen));

    runtime.end_session_timed("request").unwrap();

    assert!(seen.lock().unwrap().is_empty(), "the export is still owed");
    assert_eq!(runtime.parked_session_ends.len(), 1);
    runtime.complete_parked_session_ends();
    assert_eq!(*seen.lock().unwrap(), vec![Some("request".to_string())]);
}
