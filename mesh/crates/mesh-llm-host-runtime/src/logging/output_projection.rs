//! Canonical logging projection into process-local output sinks.
//!
//! This module is deliberately downstream of the replay bus: accepted
//! lifecycle envelopes can be rendered for JSONL, pretty output, and the TUI,
//! but output is never fed back into logging persistence or replay.

use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::{OutputEvent, emit_event};

use super::bus::PushOutcome;

/// Emit an accepted canonical envelope through the existing presentation
/// surface. Output failures remain fail-open, matching persistence failures.
pub(super) fn emit_accepted_canonical_event(
    outcome: PushOutcome,
    canonical_envelope: Option<&CanonicalEnvelope>,
) {
    if matches!(outcome, PushOutcome::Rejected) {
        return;
    }
    if let Some(envelope) = canonical_envelope {
        let _ = emit_event(OutputEvent::CanonicalLog(Box::new(envelope.clone())));
    }
}
