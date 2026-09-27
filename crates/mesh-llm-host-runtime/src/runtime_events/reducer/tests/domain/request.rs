//! `requests` category: in-flight-only rows owned by the root scope.

use mesh_llm_runtime_event_contracts::{
    ChildOperationId, OperationId, OperationScope, RequestEventKind,
};

use super::super::fixtures::{input, request_fact, scope as root};
use crate::runtime_events::config::REQUEST_ROOT_BOUND;
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, apply};

#[test]
fn requests_are_in_flight_only_and_bounded() {
    let mut snapshot = ReducerSnapshot::empty();
    for index in 0..(REQUEST_ROOT_BOUND + 50) {
        let request_id = format!("req-{index}");
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(
                root(),
                index as u64,
                request_fact(RequestEventKind::RequestReceived, &request_id),
            ),
        ) else {
            panic!("request_received must apply for {request_id}");
        };
        snapshot = next;
    }
    assert!(
        snapshot.domain().requests().len() <= REQUEST_ROOT_BOUND,
        "in-flight requests must stay bounded by REQUEST_ROOT_BOUND"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            (REQUEST_ROOT_BOUND + 50) as u64,
            request_fact(RequestEventKind::RequestCompleted, "req-0"),
        ),
    ) else {
        panic!("request_completed must apply even for an evicted-from-domain request");
    };
    assert!(
        !snapshot
            .domain()
            .requests()
            .iter()
            .any(|request| request.id == "req-0"),
        "a completed request must never appear as in-flight"
    );
}

#[test]
fn a_child_terminal_does_not_remove_its_streaming_root_request() {
    let root_operation = OperationId::new();
    let root_scope = OperationScope::root_only(root_operation);
    let child_scope = OperationScope::with_child(root_operation, ChildOperationId::new());

    let ReduceOutcome::Applied(snapshot) = apply(
        &ReducerSnapshot::empty(),
        input(
            root_scope,
            0,
            request_fact(RequestEventKind::RequestReceived, "streaming-request"),
        ),
    ) else {
        panic!("request_received must apply");
    };

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            child_scope,
            1,
            request_fact(
                RequestEventKind::RequestExecutionStarted,
                "streaming-request",
            ),
        ),
    ) else {
        panic!("a backend child progress fact must apply");
    };
    assert_eq!(
        snapshot
            .domain()
            .requests()
            .iter()
            .find(|request| request.id == "streaming-request")
            .and_then(|request| request.state.as_deref()),
        Some("received"),
        "child observations must not overwrite the root request status"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            child_scope,
            2,
            request_fact(RequestEventKind::RequestCompleted, "streaming-request"),
        ),
    ) else {
        panic!("a backend child terminal must apply");
    };
    assert!(
        snapshot
            .domain()
            .requests()
            .iter()
            .any(|request| request.id == "streaming-request"),
        "a child terminal must not hide the still-streaming root request"
    );
    assert_eq!(
        snapshot
            .domain()
            .requests()
            .iter()
            .find(|request| request.id == "streaming-request")
            .and_then(|request| request.state.as_deref()),
        Some("received"),
        "a child terminal must not overwrite the root request status"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root_scope,
            3,
            request_fact(RequestEventKind::RequestCompleted, "streaming-request"),
        ),
    ) else {
        panic!("the root terminal must apply");
    };
    assert!(
        snapshot
            .domain()
            .requests()
            .iter()
            .all(|request| request.id != "streaming-request"),
        "only the root terminal may remove the request row"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            child_scope,
            4,
            request_fact(RequestEventKind::RequestCompleted, "streaming-request"),
        ),
    ) else {
        panic!("a late child terminal must still apply to the child operation");
    };
    assert!(
        snapshot
            .domain()
            .requests()
            .iter()
            .all(|request| request.id != "streaming-request"),
        "a child terminal after root completion must not resurrect the request row"
    );
}
