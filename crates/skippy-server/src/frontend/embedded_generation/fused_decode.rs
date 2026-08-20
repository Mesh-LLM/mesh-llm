use super::*;

pub(super) fn apply_fused_first_decode(
    backend: &StageOpenAiBackend,
    request: &EmbeddedStageZeroGeneration<'_>,
    fused_first_decode: &mut Option<EmbeddedFusedFirstDecode>,
    native_mtp: &mut NativeMtpVerifier,
    current: &mut i32,
    decoded_tokens: &mut usize,
    exact_replay_tokens: &mut Vec<i32>,
    context_tokens: &mut Vec<i32>,
    decode_stage0_compute_ms: &mut f64,
    decode_runtime_lock_wait_ms: &mut f64,
    decode_runtime_lock_wait_max_ms: &mut f64,
    decode_runtime_lock_hold_ms: &mut f64,
    decode_runtime_lock_hold_max_ms: &mut f64,
    decode_runtime_lock_acquires: &mut usize,
    decode_forward_activation_encode_ms: &mut f64,
    decode_output_activation_bytes: &mut usize,
    decode_forward_activation_bytes: &mut usize,
    decode_forward_write_ms: &mut f64,
    decode_downstream_wait_ms: &mut f64,
    on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
) -> OpenAiResult<bool> {
    let mut fused_reached_stop = false;
    if let Some(mut fused) = fused_first_decode.take() {
        *current = fused.predicted;
        let mut fused_native_mtp_draft = fused.native_mtp_draft.take();
        *decode_stage0_compute_ms += fused.execution.stage0_compute_ms;
        *decode_runtime_lock_wait_ms += fused.execution.runtime_lock_wait_ms;
        *decode_runtime_lock_wait_max_ms =
            (*decode_runtime_lock_wait_max_ms).max(fused.execution.runtime_lock_wait_ms);
        *decode_runtime_lock_hold_ms += fused.execution.runtime_lock_hold_ms;
        *decode_runtime_lock_hold_max_ms =
            (*decode_runtime_lock_hold_max_ms).max(fused.execution.runtime_lock_hold_ms);
        *decode_runtime_lock_acquires += 1;
        *decode_forward_activation_encode_ms += fused.execution.activation_encode_ms;
        *decode_output_activation_bytes = (*decode_output_activation_bytes)
            .saturating_add(fused.execution.output_activation_bytes);
        *decode_forward_activation_bytes = (*decode_forward_activation_bytes)
            .saturating_add(fused.execution.forward_activation_bytes);
        *decode_forward_write_ms += fused.execution.forward_write_ms;
        *decode_downstream_wait_ms += fused.execution.downstream_wait_ms;
        for (index, token) in fused.predicted_tokens.iter().copied().enumerate() {
            if *decoded_tokens >= request.max_tokens as usize {
                break;
            }
            *current = token;
            exact_replay_tokens.push(*current);
            context_tokens.push(*current);
            let native_mtp_decision = native_mtp.observe_target_token(
                *current,
                if index == 0 {
                    ms_to_us(fused.execution.downstream_wait_ms)
                } else {
                    0
                },
                if index == 0 {
                    fused_native_mtp_draft.take()
                } else {
                    None
                },
                NativeMtpDraftOrigin::InitialSerial,
            );
            *decoded_tokens += 1;
            if backend.telemetry.is_debug_enabled() {
                let mut token_attrs = backend.openai_attrs(request.ids);
                token_attrs.insert("llama_stage.decode_step".to_string(), json!(index));
                token_attrs.insert(
                    "llama_stage.decode_token_phase".to_string(),
                    json!(fused.token_phase),
                );
                token_attrs.insert(
                    "llama_stage.message_kind".to_string(),
                    json!(fused.message_kind),
                );
                token_attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(if index == 0 { fused.elapsed_ms } else { 0.0 }),
                );
                token_attrs.insert(
                    "llama_stage.cached_replay_token_index".to_string(),
                    json!(index),
                );
                token_attrs.insert(
                    "llama_stage.cached_replay_token_count".to_string(),
                    json!(fused.predicted_tokens.len()),
                );
                token_attrs.insert(
                    "llama_stage.stage0_compute_ms".to_string(),
                    json!(if index == 0 {
                        fused.execution.stage0_compute_ms
                    } else {
                        0.0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(if index == 0 {
                        fused.execution.runtime_lock_wait_ms
                    } else {
                        0.0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(if index == 0 {
                        fused.execution.runtime_lock_hold_ms
                    } else {
                        0.0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(if index == 0 {
                        fused.execution.output_activation_bytes
                    } else {
                        0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.forward_activation_bytes".to_string(),
                    json!(if index == 0 {
                        fused.execution.forward_activation_bytes
                    } else {
                        0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.activation_encode_ms".to_string(),
                    json!(if index == 0 {
                        fused.execution.activation_encode_ms
                    } else {
                        0.0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(if index == 0 {
                        fused.execution.forward_write_ms
                    } else {
                        0.0
                    }),
                );
                token_attrs.insert(
                    "llama_stage.downstream_wait_ms".to_string(),
                    json!(if index == 0 {
                        fused.execution.downstream_wait_ms
                    } else {
                        0.0
                    }),
                );
                token_attrs.insert("llama_stage.predicted_token".to_string(), json!(*current));
                token_attrs.insert(
                    "llama_stage.native_mtp.verification".to_string(),
                    json!(native_mtp_decision.label()),
                );
                backend
                    .telemetry
                    .emit_debug("stage.openai_decode_token", token_attrs);
            }
            if on_token(*current)? == TokenControl::Stop {
                fused_reached_stop = true;
                break;
            }
        }
    }
    Ok(fused_reached_stop)
}
