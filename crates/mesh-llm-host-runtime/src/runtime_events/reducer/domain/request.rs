//! `requests` category: in-flight root requests only; a row is removed the
//! moment its root terminal settles.

use mesh_llm_runtime_event_contracts::{FactData, OperationScope, RequestEventKind};

use super::DomainState;
use super::bounded::{remove_bounded, touch};
use super::execution::{RequestGenerationState, RequestPrefillState};
use crate::runtime_events::config::REQUEST_ROOT_BOUND;

/// One in-flight request's reduced domain view: a `requests` category
/// row. Removed the moment its terminal-class outcome settles --
/// "in-flight only" per the plan text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RequestDomainState {
    pub id: String,
    pub state: Option<String>,
    pub prefill: Option<RequestPrefillState>,
    pub generation: Option<RequestGenerationState>,
}

fn request_id(data: &FactData) -> Option<String> {
    data.scope
        .request_id
        .as_ref()
        .map(|id| id.as_str().to_string())
}

fn is_terminal_request_kind(kind: RequestEventKind) -> bool {
    matches!(
        kind,
        RequestEventKind::RequestCompleted
            | RequestEventKind::RequestCancelled
            | RequestEventKind::RequestTimedOut
            | RequestEventKind::RequestFailed
            | RequestEventKind::RequestRejected
    )
}

fn request_state_label(kind: RequestEventKind) -> &'static str {
    match kind {
        RequestEventKind::RequestReceived => "received",
        RequestEventKind::RequestQueued => "queued",
        RequestEventKind::RequestAdmitted => "admitted",
        RequestEventKind::RequestExecutionStarted => "executing",
        // Terminal-class kinds never reach this arm: `apply_request` below
        // removes the entry before calling this function for any of them.
        // The match stays exhaustive so a new `RequestEventKind` variant is
        // a compile error here, not a silent no-op.
        RequestEventKind::RequestRejected
        | RequestEventKind::RequestCompleted
        | RequestEventKind::RequestCancelled
        | RequestEventKind::RequestTimedOut
        | RequestEventKind::RequestFailed => "terminal",
    }
}

pub(super) fn apply_request(
    state: &mut DomainState,
    scope: OperationScope,
    kind: RequestEventKind,
    data: &FactData,
) {
    // Request-domain rows represent the root request's public lifetime. A
    // child operation may share the request id while it streams, but none of
    // its observations owns that row: a child terminal must not remove it,
    // and a late child update must not recreate or overwrite a row after the
    // root has settled.
    if matches!(scope, OperationScope::Child { .. }) {
        return;
    }
    let Some(id) = request_id(data) else {
        return;
    };
    if is_terminal_request_kind(kind) {
        remove_bounded(&mut state.requests_order, &mut state.requests, &id);
        return;
    }
    touch(
        &mut state.requests_order,
        &mut state.requests,
        &id,
        REQUEST_ROOT_BOUND,
    );
    let label = request_state_label(kind);
    let row = state
        .requests
        .entry(id.clone())
        .or_insert_with(|| RequestDomainState {
            id,
            ..RequestDomainState::default()
        });
    row.state = Some(label.to_string());
}
