use std::time::Instant;

use anyhow::{Context, Result, bail};
use skippy_protocol::{
    StageConfig,
    binary::{StageWireMessage, WireActivationDType, activation_state_flags_from_frame_flags},
};
use skippy_runtime::{ActivationFrame, RuntimeActivationDType};

pub(crate) fn forwarded_stage_message(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> Result<StageWireMessage> {
    Ok(
        forwarded_stage_message_timed(config, incoming, output, wire_dtype, activation_width)?
            .message,
    )
}

pub(crate) struct ForwardedStageMessage {
    pub message: StageWireMessage,
    pub activation_encode_ms: f64,
}

pub(crate) fn forwarded_stage_message_timed(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> Result<ForwardedStageMessage> {
    // A non-first stage must always compute the *full* incoming token range.
    // Suffix-only execution after a partial cache hit is permitted only on the
    // stage that owns layer 0 (see `executable_token_ids` in
    // `binary_messaging/connection.rs`), because the forwarded header keeps
    // `incoming.token_count` and downstream stages attend over the whole
    // range. If a later stage ever emitted a short frame, the next stage would
    // attend over a prefix it never received and produce plausible-looking but
    // wrong tokens.
    //
    // Today the payload-size check inside the encoder happens to catch this,
    // but it fails for the wrong reason and only for some dtypes. Assert the
    // invariant directly so a future change to the restore path fails loudly
    // and specifically instead of silently corrupting output.
    if config.layer_start != 0
        && i64::from(output.desc.token_count) != i64::from(incoming.token_count)
    {
        bail!(
            "stage {} (layers {}..{}) produced {} activation tokens for {} incoming tokens; \
             non-first stages must execute the full range",
            config.stage_index,
            config.layer_start,
            config.layer_end,
            output.desc.token_count,
            incoming.token_count,
        );
    }
    let mut state = incoming.state;
    state.source_stage_index = config.stage_index as i32;
    state.reserved = wire_dtype as i32;
    state.flags |= activation_state_flags_from_frame_flags(output.desc.flags);
    let encode_started = Instant::now();
    let activation =
        encode_output_activation_payload(wire_dtype, incoming, output, activation_width, state.flags)
            .with_context(|| {
                format!(
                    "encode output activation payload; wire_dtype={wire_dtype:?} frame_dtype={:?} incoming_tokens={} output_tokens={} activation_width={} payload_bytes={} frame_payload_bytes={} state_flags={}",
                    output.desc.dtype,
                    incoming.token_count,
                    output.desc.token_count,
                    activation_width,
                    output.payload.len(),
                    output.desc.payload_bytes,
                    state.flags,
                )
            })?;
    Ok(ForwardedStageMessage {
        message: StageWireMessage {
            kind: incoming.kind,
            pos_start: incoming.pos_start,
            token_count: incoming.token_count,
            state,
            request_id: incoming.request_id,
            session_id: incoming.session_id,
            sampling: incoming.sampling.clone(),
            chat_sampling_metadata: None,
            tokens: incoming.tokens.clone(),
            positions: incoming.positions.clone(),
            activation,
            raw_bytes: Vec::new(),
        },
        activation_encode_ms: encode_started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn encode_output_activation_payload(
    wire_dtype: WireActivationDType,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    state_flags: i32,
) -> Result<Vec<u8>> {
    match (output.desc.dtype, wire_dtype) {
        (RuntimeActivationDType::F32, _) => Ok(
            skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
                wire_dtype,
                incoming.token_count,
                activation_width,
                &output.payload,
                state_flags,
            )?,
        ),
        (RuntimeActivationDType::F16, WireActivationDType::F16) => {
            validate_f16_passthrough_payload(incoming, output, activation_width, state_flags)?;
            Ok(output.payload.clone())
        }
        (dtype, wire_dtype) => {
            bail!("unsupported activation dtype conversion: {dtype:?} to {wire_dtype:?}")
        }
    }
}

fn validate_f16_passthrough_payload(
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    state_flags: i32,
) -> Result<()> {
    if output.payload.len() as u64 != output.desc.payload_bytes {
        bail!("F16 activation payload length does not match frame descriptor");
    }
    let expected = skippy_protocol::binary::activation_wire_bytes_with_state_flags(
        WireActivationDType::F16,
        incoming.token_count,
        activation_width,
        state_flags,
    )
    .context("compute expected F16 activation payload size")?;
    if output.payload.len() != expected {
        bail!("F16 activation payload size mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{
        FlashAttentionType, LoadMode, PeerConfig, StageDevice, StageKvCacheConfig,
        binary::{
            StageStateHeader, WireMessageKind, activation_frame_flags_from_state_flags, state_flags,
        },
    };
    use skippy_runtime::{ActivationDesc, RuntimeActivationDType, RuntimeActivationLayout};

    fn stage_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "model".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            layer_start: 4,
            layer_end: 8,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None::<StageDevice>,
            kv_cache: None::<StageKvCacheConfig>,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                endpoint: "tcp://127.0.0.1:19000".to_string(),
            }),
            downstream: None,
        }
    }

    fn incoming_message() -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 7,
            token_count: 1,
            state: StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32),
            request_id: 42,
            session_id: 99,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn f32_frame(flags: u64, token_count: u32, values: &[f32]) -> ActivationFrame {
        let mut payload = Vec::new();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 1,
                layer_start: 4,
                layer_end: 8,
                token_count,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags,
            },
            payload,
        }
    }

    fn rwkv7_sideband_frame() -> ActivationFrame {
        f32_frame(
            skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST,
            1,
            &[1.0_f32, 2.0, 3.0, 4.0],
        )
    }

    fn f16_frame() -> ActivationFrame {
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F16,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 1,
                layer_start: 4,
                layer_end: 8,
                token_count: 2,
                sequence_count: 1,
                payload_bytes: 8,
                flags: 0,
            },
            payload: vec![0, 1, 2, 3, 4, 5, 6, 7],
        }
    }

    #[test]
    fn forwarded_stage_message_preserves_rwkv7_sideband_shape() {
        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming_message(),
            &rwkv7_sideband_frame(),
            WireActivationDType::F32,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation.len(), 16);
        assert_ne!(
            forwarded.message.state.flags & state_flags::RWKV7_V_FIRST_SIDEBAND,
            0
        );
        assert_eq!(
            activation_frame_flags_from_state_flags(forwarded.message.state.flags),
            skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST
        );
    }

    #[test]
    fn forwarded_stage_message_reencodes_rwkv7_sideband_for_wire_dtype() {
        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming_message(),
            &rwkv7_sideband_frame(),
            WireActivationDType::Q8,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation.len(), 12);
        assert_ne!(
            forwarded.message.state.flags & state_flags::RWKV7_V_FIRST_SIDEBAND,
            0
        );
    }

    #[test]
    fn forwarded_stage_message_passes_through_f16_activation_for_f16_wire() {
        let mut incoming = incoming_message();
        incoming.token_count = 2;

        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming,
            &f16_frame(),
            WireActivationDType::F16,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            forwarded.message.state.reserved,
            WireActivationDType::F16 as i32
        );
    }

    #[test]
    fn forwarded_stage_message_rejects_bad_f16_passthrough_size() {
        let mut incoming = incoming_message();
        incoming.token_count = 2;
        let mut output = f16_frame();
        output.payload.pop();
        output.desc.payload_bytes = output.payload.len() as u64;

        let error = match forwarded_stage_message_timed(
            &stage_config(),
            &incoming,
            &output,
            WireActivationDType::F16,
            2,
        ) {
            Ok(_) => panic!("expected bad F16 passthrough payload to fail"),
            Err(error) => error,
        };

        assert!(format!("{error:#}").contains("F16 activation payload size mismatch"));
    }

    /// A non-first stage must execute the full incoming token range.
    ///
    /// Suffix-only execution after a partial cache hit is legal only on the
    /// stage owning layer 0. If a later stage emitted a short frame, the next
    /// stage would attend over a prefix it never received -- plausible-looking
    /// but wrong output. This must fail loudly and by name.
    #[test]
    fn non_first_stage_must_not_emit_a_short_activation_frame() {
        let config = stage_config();
        assert_ne!(config.layer_start, 0, "fixture must be a non-first stage");
        let mut incoming = incoming_message();
        incoming.token_count = 4;
        // Frame covers only 1 of the 4 incoming tokens, as a suffix-only
        // execution after a 3-token restore would produce.
        let output = f32_frame(0, 1, &[1.0, 2.0, 3.0, 4.0]);

        let error =
            forwarded_stage_message_timed(&config, &incoming, &output, WireActivationDType::F32, 4)
                .err()
                .expect("short frame from a non-first stage must be rejected");

        let text = format!("{error:#}");
        assert!(
            text.contains("must execute the full range"),
            "expected the named invariant, got: {text}"
        );
    }

    /// The first stage is explicitly allowed to emit a short frame: that is
    /// how suffix-only prefill after a cache hit works.
    #[test]
    fn first_stage_may_emit_a_short_activation_frame() {
        let mut config = stage_config();
        config.stage_index = 0;
        config.layer_start = 0;
        let mut incoming = incoming_message();
        incoming.token_count = 4;

        // Keep the encoded payload valid for the four-token wire header while
        // the frame descriptor exercises the first-stage short-frame branch.
        assert!(
            forwarded_stage_message_timed(
                &config,
                &incoming,
                &f32_frame(0, 1, &[1.0; 16]),
                WireActivationDType::F32,
                4,
            )
            .is_ok()
        );
    }
}
