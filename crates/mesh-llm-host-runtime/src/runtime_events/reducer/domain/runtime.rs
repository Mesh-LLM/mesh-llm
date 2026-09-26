//! `node.runtime`: latest-wins native-runtime lifecycle status.
//!
//! Fed by `system::native_runtime_events::NativeRuntimeResolution`
//! (resolution started/library loaded/initialized/completed/failed/
//! unavailable). `NativeLibraryRejected`, the ABI-compatibility kinds, and
//! `RuntimeStopping`/`RuntimeStopped`/`RuntimeCrashed` have no production
//! producer today; the reducer handles them so a future producer needs no
//! reducer change.

use mesh_llm_runtime_event_contracts::{FactData, NativeRuntimeEventKind};

use super::{is_undelivered_terminal, outcome_label, reason_label};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NativeRuntimeDomainState {
    pub status: Option<&'static str>,
    pub abi_compatible: Option<bool>,
    pub last_outcome: Option<&'static str>,
    pub last_reason_code: Option<String>,
}

impl NativeRuntimeDomainState {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The status a kind moves to, or `None` when the kind does not by itself
/// change status (see [`apply_native_runtime`] for `ResolutionCompleted`).
const fn status_label(kind: NativeRuntimeEventKind) -> Option<&'static str> {
    use NativeRuntimeEventKind::{
        AbiFeatureCompatibilityEstablished, AbiFeatureCompatibilityFailed, NativeLibraryLoaded,
        NativeLibraryRejected, NativeLibraryUnavailable, RuntimeCrashed, RuntimeInitialized,
        RuntimeResolutionCompleted, RuntimeResolutionFailed, RuntimeResolutionStarted,
        RuntimeStopped, RuntimeStopping,
    };
    match kind {
        RuntimeResolutionStarted => Some("resolving"),
        RuntimeResolutionCompleted => None,
        RuntimeResolutionFailed | NativeLibraryUnavailable => Some("unavailable"),
        NativeLibraryLoaded => Some("library_loaded"),
        NativeLibraryRejected => Some("library_rejected"),
        AbiFeatureCompatibilityEstablished => Some("abi_compatible"),
        AbiFeatureCompatibilityFailed => Some("abi_incompatible"),
        RuntimeInitialized => Some("initialized"),
        RuntimeStopping => Some("stopping"),
        RuntimeStopped => Some("stopped"),
        RuntimeCrashed => Some("crashed"),
    }
}

const fn abi_compatibility(kind: NativeRuntimeEventKind) -> Option<bool> {
    match kind {
        NativeRuntimeEventKind::AbiFeatureCompatibilityEstablished => Some(true),
        NativeRuntimeEventKind::AbiFeatureCompatibilityFailed => Some(false),
        _ => None,
    }
}

/// `ResolutionCompleted` is the resolution operation's terminal, emitted
/// AFTER `RuntimeInitialized` in production; it reports "resolved" only
/// when the runtime has not already advanced to `initialized`.
fn next_status(
    current: Option<&'static str>,
    kind: NativeRuntimeEventKind,
) -> Option<&'static str> {
    match status_label(kind) {
        Some(label) => Some(label),
        None if current == Some("initialized") => current,
        None => Some("resolved"),
    }
}

pub(super) fn apply_native_runtime(
    state: &mut NativeRuntimeDomainState,
    kind: NativeRuntimeEventKind,
    data: &FactData,
) {
    if let Some(outcome) = data.outcome {
        state.last_outcome = Some(outcome_label(outcome));
    }
    if let Some(reason) = reason_label(data) {
        state.last_reason_code = Some(reason);
    }
    // An engine-synthesized `terminal_not_delivered` only says the producer
    // never reported a terminal; it carries no authority over the status.
    if is_undelivered_terminal(data) {
        return;
    }
    state.status = next_status(state.status, kind);
    if let Some(compatible) = abi_compatibility(kind) {
        state.abi_compatible = Some(compatible);
    }
}
