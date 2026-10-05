//! `sessions` category.

use mesh_llm_runtime_event_contracts::SessionEventKind;

use super::super::fixtures::{input, scope as root, session_fact};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, apply};

#[test]
fn sessions_track_active_count_and_bounded_recent() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            session_fact(SessionEventKind::SessionCreated, "sess-1"),
        ),
    ) else {
        panic!("session_created must apply");
    };
    assert_eq!(snapshot.domain().sessions_active_count(), 1);

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            session_fact(SessionEventKind::SessionClosed, "sess-1"),
        ),
    ) else {
        panic!("session_closed must apply");
    };
    assert_eq!(
        snapshot.domain().sessions_active_count(),
        0,
        "a closed session must leave the active count"
    );
    let recent = snapshot.domain().sessions_recent();
    assert!(
        recent
            .iter()
            .any(|entry| entry.id == "sess-1" && entry.state == "closed"),
        "a closed session must appear in the bounded recent list"
    );
}
