use crate::binary_transport::stage_execution::elapsed_ms;
use crate::runtime_state::{RuntimeSessionAlignStats, RuntimeState};
use crate::telemetry::Telemetry;
use anyhow::{Context, Result};
use serde_json::json;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Default)]
pub(super) struct SessionAutoAlignObservation {
    pub(super) count: usize,
    pub(super) elapsed_ms: f64,
    pub(super) trimmed_tokens: u64,
}

/// Emit the auto-align debug event for a completed trim and build its
/// observation. Shared so every alignment path reports the same counter and the
/// same event attributes, whether the trim ran through an explicit
/// [`align_session_to_target`] call (lookup/compute paths, which time it) or was
/// fused into a batched decode frame (which reports no standalone elapsed).
pub(super) fn record_session_auto_align(
    telemetry: &Telemetry,
    session_key: &str,
    align: &RuntimeSessionAlignStats,
    elapsed_ms: f64,
) -> SessionAutoAlignObservation {
    let trimmed_tokens = align
        .before_token_count
        .saturating_sub(align.after_token_count);
    telemetry.emit_debug(
        "stage.binary_session_auto_align",
        BTreeMap::from([
            ("skippy.session_id".to_string(), json!(session_key)),
            (
                "llama_stage.session_auto_align_before_tokens".to_string(),
                json!(align.before_token_count),
            ),
            (
                "llama_stage.session_auto_align_after_tokens".to_string(),
                json!(align.after_token_count),
            ),
            (
                "llama_stage.session_auto_align_trimmed_tokens".to_string(),
                json!(trimmed_tokens),
            ),
            ("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms)),
        ]),
    );
    SessionAutoAlignObservation {
        count: 1,
        elapsed_ms,
        trimmed_tokens,
    }
}

/// Trims a stage session that is ahead of the message's authoritative
/// position. Callers run this inside a scheduler runtime closure they already
/// own for the message, so alignment does not cost its own scheduler
/// round-trip on the per-token path.
pub(super) fn align_session_to_target(
    runtime: &mut RuntimeState,
    telemetry: &Telemetry,
    session_key: &str,
    target_token_count: Option<u64>,
) -> Result<SessionAutoAlignObservation> {
    let Some(target_token_count) = target_token_count else {
        return Ok(SessionAutoAlignObservation::default());
    };
    let started = Instant::now();
    let align = runtime
        .align_session_to_token_count_if_ahead(session_key, target_token_count)
        .context("auto-align binary stage session")?;
    let Some(align) = align else {
        return Ok(SessionAutoAlignObservation::default());
    };
    Ok(record_session_auto_align(
        telemetry,
        session_key,
        &align,
        elapsed_ms(started),
    ))
}
