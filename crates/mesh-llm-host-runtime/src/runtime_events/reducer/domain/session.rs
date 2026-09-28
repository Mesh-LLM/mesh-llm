//! `sessions` category: an active-session count plus a bounded FIFO of
//! recently settled sessions -- deliberately not a per-entity collection.

use mesh_llm_runtime_event_contracts::{FactData, SessionEventKind};

use super::DomainState;
use super::bounded::{push_recent, touch};
use crate::runtime_events::config::LIFECYCLE_OPERATION_BOUND;

/// One recently-settled session, retained after leaving the active set:
/// a `sessions.recent` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRecentEntry {
    pub id: String,
    pub state: String,
}

fn session_id(data: &FactData) -> Option<String> {
    data.scope
        .session_id
        .as_ref()
        .map(|id| id.as_str().to_string())
}

fn session_state_label(kind: SessionEventKind) -> &'static str {
    use SessionEventKind::{
        SessionAbandoned, SessionActive, SessionClosed, SessionCreated, SessionDraining,
        SessionFailed, SessionIdle, SessionReclaimed, SessionRequested, SessionReset,
        SessionRestoredFromCheckpoint, SessionRestoredFromPrefixCache, SessionReusable,
        SessionTrimmed,
    };
    match kind {
        SessionRequested => "requested",
        SessionCreated => "created",
        SessionActive => "active",
        SessionIdle => "idle",
        SessionReusable => "reusable",
        SessionReset => "reset",
        SessionTrimmed => "trimmed",
        SessionRestoredFromPrefixCache | SessionRestoredFromCheckpoint => "restored",
        SessionDraining => "draining",
        SessionClosed => "closed",
        SessionFailed => "failed",
        SessionAbandoned => "abandoned",
        SessionReclaimed => "reclaimed",
    }
}

fn is_terminal_session_kind(kind: SessionEventKind) -> bool {
    matches!(
        kind,
        SessionEventKind::SessionClosed
            | SessionEventKind::SessionFailed
            | SessionEventKind::SessionAbandoned
            | SessionEventKind::SessionReclaimed
    )
}

fn settle_session(state: &mut DomainState, id: String, label: &str) {
    if let Some(position) = state
        .sessions_order
        .iter()
        .position(|existing| existing == &id)
    {
        state.sessions_order.remove(position);
    }
    state.sessions_active.remove(&id);
    push_recent(
        &mut state.sessions_recent,
        SessionRecentEntry {
            id,
            state: label.to_string(),
        },
        LIFECYCLE_OPERATION_BOUND,
    );
}

pub(super) fn apply_session(state: &mut DomainState, kind: SessionEventKind, data: &FactData) {
    let Some(id) = session_id(data) else {
        return;
    };
    let label = session_state_label(kind);
    if is_terminal_session_kind(kind) {
        settle_session(state, id, label);
        return;
    }
    touch(
        &mut state.sessions_order,
        &mut state.sessions_active,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    state.sessions_active.insert(id, label.to_string());
}
