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
        seen.lock()
            .unwrap()
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
