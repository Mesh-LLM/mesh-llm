//! `node.diagnostics`: bounded active warnings, degraded toggle, a pinned
//! fatal entry, and monotonic failure counters.
//!
//! Producer today: the native-family mapping in
//! `system::native_runtime_events` (warning raised/cleared, recoverable,
//! fatal, invariant violation), plus the reservation table's own
//! `InvariantProtocolViolation`. `FallbackApplied` and
//! `DegradedOperationEntered`/`Exited` have no production producer yet.

use std::collections::VecDeque;

use mesh_llm_runtime_event_contracts::{DiagnosticEventKind, FactData};

use super::{is_undelivered_terminal, reason_label};

/// Active warnings retained; raising a new key past the bound evicts the
/// oldest-raised warning and counts it in `evicted_warnings`.
pub const ACTIVE_WARNING_BOUND: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticEntry {
    /// Correlation key: reason code plus every scope identity present, so a
    /// `WarningCleared` for the same source clears exactly this entry.
    pub key: String,
    pub reason_code: Option<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiagnosticDomainState {
    pub active_warnings: VecDeque<DiagnosticEntry>,
    pub evicted_warnings: u64,
    pub degraded: bool,
    pub fatal: Option<DiagnosticEntry>,
    pub recoverable_failures: u64,
    pub fallbacks: u64,
    pub invariant_violations: u64,
}

impl DiagnosticDomainState {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

fn warning_key(data: &FactData) -> String {
    let scope = &data.scope;
    let parts = [
        reason_label(data),
        scope
            .model_id
            .as_ref()
            .map(|id| format!("model={}", id.as_str())),
        scope
            .topology_id
            .as_ref()
            .map(|id| format!("topology={}", id.as_str())),
        scope
            .stage
            .as_ref()
            .map(|stage| format!("stage={}#{}", stage.id.as_str(), stage.index)),
        scope
            .session_id
            .as_ref()
            .map(|id| format!("session={}", id.as_str())),
        scope
            .request_id
            .as_ref()
            .map(|id| format!("request={}", id.as_str())),
        scope
            .device_id
            .as_ref()
            .map(|id| format!("device={}", id.as_str())),
    ];
    parts.into_iter().flatten().collect::<Vec<_>>().join("|")
}

fn entry(data: &FactData) -> DiagnosticEntry {
    DiagnosticEntry {
        key: warning_key(data),
        reason_code: reason_label(data),
        summary: data
            .summary
            .as_ref()
            .map(|summary| summary.as_str().to_string()),
    }
}

fn raise_warning(state: &mut DiagnosticDomainState, warning: DiagnosticEntry) {
    if let Some(existing) = state
        .active_warnings
        .iter_mut()
        .find(|existing| existing.key == warning.key)
    {
        *existing = warning;
        return;
    }
    if state.active_warnings.len() >= ACTIVE_WARNING_BOUND {
        state.active_warnings.pop_front();
        state.evicted_warnings = state.evicted_warnings.saturating_add(1);
    }
    state.active_warnings.push_back(warning);
}

fn clear_warning(state: &mut DiagnosticDomainState, key: &str) {
    state.active_warnings.retain(|warning| warning.key != key);
}

pub(super) fn apply_diagnostic(
    state: &mut DiagnosticDomainState,
    kind: DiagnosticEventKind,
    data: &FactData,
) {
    match kind {
        DiagnosticEventKind::WarningRaised => raise_warning(state, entry(data)),
        DiagnosticEventKind::WarningCleared => clear_warning(state, &warning_key(data)),
        DiagnosticEventKind::RecoverableNativeFailure => {
            state.recoverable_failures = state.recoverable_failures.saturating_add(1);
        }
        DiagnosticEventKind::FallbackApplied => {
            state.fallbacks = state.fallbacks.saturating_add(1);
        }
        DiagnosticEventKind::DegradedOperationEntered => state.degraded = true,
        DiagnosticEventKind::DegradedOperationExited => state.degraded = false,
        // Pinned: the first real fatal stays; the engine's synthesized
        // `terminal_not_delivered` fatal is not evidence of a failure.
        DiagnosticEventKind::FatalNativeFailure => {
            if state.fatal.is_none() && !is_undelivered_terminal(data) {
                state.fatal = Some(entry(data));
            }
        }
        DiagnosticEventKind::InvariantProtocolViolation => {
            state.invariant_violations = state.invariant_violations.saturating_add(1);
        }
    }
}
