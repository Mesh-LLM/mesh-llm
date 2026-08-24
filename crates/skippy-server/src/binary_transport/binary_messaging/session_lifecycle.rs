use crate::binary_transport::stage_execution::{binary_message_attrs, elapsed_ms};
use crate::runtime_state::RuntimeState;
use crate::telemetry::Telemetry;
use anyhow::{Context, Result};
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageWireMessage;
use std::time::Instant;

#[derive(Default)]
pub(super) struct SessionAutoAlignObservation {
    pub(super) count: usize,
    pub(super) elapsed_ms: f64,
    pub(super) trimmed_tokens: u64,
}

pub(super) fn align_session_to_message(
    config: &StageConfig,
    runtime: &mut RuntimeState,
    telemetry: &Telemetry,
    session_key: &str,
    session_id: u64,
    message: &StageWireMessage,
) -> Result<SessionAutoAlignObservation> {
    let Some(target_token_count) = message.authoritative_session_position() else {
        return Ok(SessionAutoAlignObservation::default());
    };
    let started = Instant::now();
    let align = runtime
        .align_session_to_token_count_if_ahead(session_key, target_token_count)
        .context("auto-align binary stage session")?;
    let Some(align) = align else {
        return Ok(SessionAutoAlignObservation::default());
    };
    let elapsed_ms = elapsed_ms(started);
    let mut attrs = binary_message_attrs(config, session_id, message);
    attrs.insert(
        "llama_stage.session_auto_align_before_tokens".to_string(),
        json!(align.before_token_count),
    );
    attrs.insert(
        "llama_stage.session_auto_align_after_tokens".to_string(),
        json!(align.after_token_count),
    );
    attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
    telemetry.emit_debug("stage.binary_session_auto_align", attrs);
    Ok(SessionAutoAlignObservation {
        count: 1,
        elapsed_ms,
        trimmed_tokens: align
            .before_token_count
            .saturating_sub(align.after_token_count),
    })
}
