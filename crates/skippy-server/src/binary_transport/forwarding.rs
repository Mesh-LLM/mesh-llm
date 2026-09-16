use std::time::Instant;

use anyhow::{Context, Result, bail};
use skippy_protocol::{
    StageActivationCodec, StageActivationCodecPolicy, StageConfig,
    binary::{StageWireMessage, encode_activation_frame, select_lossless_activation_codec},
};
use skippy_runtime::ActivationFrame;

use super::prefill_execution::stage_activation_desc;

pub(crate) fn forwarded_stage_message(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
) -> Result<StageWireMessage> {
    Ok(forwarded_stage_message_timed(config, incoming, output, activation_width)?.message)
}

pub(crate) struct ForwardedStageMessage {
    pub message: StageWireMessage,
    pub activation_encode_ms: f64,
}

pub(crate) fn forwarded_stage_message_timed(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    _activation_width: i32,
) -> Result<ForwardedStageMessage> {
    // A stage with a downstream consumer must compute the *full* incoming token
    // range unless it owns layer 0. Suffix-only execution after a partial cache
    // hit is also safe on a final stage with no downstream consumer (see
    // `prefill_execution.rs`). The forwarded header keeps
    // `incoming.token_count` and downstream stages attend over the whole
    // range. If a later stage ever emitted a short frame, the next stage would
    // attend over a prefix it never received and produce plausible-looking but
    // wrong tokens.
    //
    // Today the payload-size check inside the encoder happens to catch this,
    // but it fails for the wrong reason. Assert the
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
    let activation_desc = stage_activation_desc(&output.desc)?;
    let activation_codec = select_output_activation_codec(
        config.activation_codec,
        config.activation_codec_policy,
        &activation_desc,
        output,
    )?;
    state.activation_codec = activation_codec;
    let encode_started = Instant::now();
    let activation = encode_activation_frame(activation_codec, &activation_desc, &output.payload)
        .with_context(|| {
            format!(
                "encode multipart output activation; incoming_tokens={} output_tokens={} parts={} payload_bytes={} frame_payload_bytes={}",
                incoming.token_count,
                output.desc.token_count,
                output.desc.part_count,
                output.payload.len(),
                output.desc.payload_bytes,
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

fn select_output_activation_codec(
    configured_codec: StageActivationCodec,
    policy: StageActivationCodecPolicy,
    desc: &skippy_protocol::binary::StageActivationDesc,
    output: &ActivationFrame,
) -> Result<StageActivationCodec> {
    if !policy.compatible(configured_codec) {
        bail!(
            "activation codec policy {policy:?} is incompatible with configured codec {configured_codec:?}"
        );
    }
    match policy {
        StageActivationCodecPolicy::Fixed => Ok(configured_codec),
        StageActivationCodecPolicy::AutoLosslessV1 => Ok(select_lossless_activation_codec(
            desc,
            &output.payload,
            &[
                StageActivationCodec::RawF32V1,
                StageActivationCodec::Bf16RneV1,
                StageActivationCodec::F16RneV1,
            ],
        )?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{
        FlashAttentionType, LoadMode, PeerConfig, StageDevice, StageKvCacheConfig,
        binary::{StageStateHeader, WireMessageKind, activation_frame_wire_bytes},
    };
    use skippy_runtime::{ACTIVATION_PART_OPTIONAL, GGML_TYPE_F32, GGML_TYPE_I32};

    use crate::test_activation::{PartBytes, f32_frame, frame};

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
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
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
            ..StageConfig::default()
        }
    }

    fn incoming_message() -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 7,
            token_count: 1,
            state: StageStateHeader::new(WireMessageKind::DecodeEmbd),
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

    fn two_f32_part_frame(values: &[f32]) -> ActivationFrame {
        let midpoint = values.len() / 2;
        frame(
            1,
            vec![
                PartBytes {
                    identity: 1,
                    ggml_type: GGML_TYPE_F32,
                    flags: 0,
                    bytes: values[..midpoint]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                },
                PartBytes {
                    identity: 2,
                    ggml_type: GGML_TYPE_F32,
                    flags: ACTIVATION_PART_OPTIONAL,
                    bytes: values[midpoint..]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                },
            ],
        )
    }

    #[test]
    fn forwarded_stage_message_preserves_two_f32_parts() {
        let mut config = stage_config();
        config.activation_codec = skippy_protocol::StageActivationCodec::F16RneV1;
        let source = two_f32_part_frame(&[1.0_f32, 2.0, 3.0, 4.0]);
        let forwarded =
            forwarded_stage_message_timed(&config, &incoming_message(), &source, 2).unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            skippy_protocol::StageActivationCodec::F16RneV1
        );

        let mut wire = Vec::new();
        skippy_protocol::binary::write_stage_message(&mut wire, &forwarded.message).unwrap();
        let decoded = skippy_protocol::binary::read_stage_message_for_codec(
            std::io::Cursor::new(wire),
            2,
            skippy_protocol::StageActivationCodec::F16RneV1,
        )
        .unwrap();
        let decoded = decoded.activation_frame().unwrap().unwrap();
        assert_eq!(decoded.desc, stage_activation_desc(&source.desc).unwrap());
        assert_eq!(decoded.payload, source.payload);
    }

    #[test]
    fn forwarded_stage_message_preserves_glm_dsa_mixed_dtype_sideband() {
        let mut config = stage_config();
        config.activation_codec = skippy_protocol::StageActivationCodec::F16RneV1;
        let top_k = [7_i32, 11, 13]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let frame = frame(
            1,
            vec![
                PartBytes {
                    identity: 1,
                    ggml_type: GGML_TYPE_F32,
                    flags: 0,
                    bytes: [1.0_f32, 2.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                },
                PartBytes {
                    identity: 2,
                    ggml_type: GGML_TYPE_I32,
                    flags: 0,
                    bytes: top_k.clone(),
                },
            ],
        );

        let forwarded =
            forwarded_stage_message_timed(&config, &incoming_message(), &frame, 2).unwrap();

        let mut wire = Vec::new();
        skippy_protocol::binary::write_stage_message(&mut wire, &forwarded.message).unwrap();
        let decoded = skippy_protocol::binary::read_stage_message_for_codec(
            std::io::Cursor::new(wire),
            2,
            skippy_protocol::StageActivationCodec::F16RneV1,
        )
        .unwrap();
        let decoded = decoded.activation_frame().unwrap().unwrap();
        assert_eq!(decoded.desc, stage_activation_desc(&frame.desc).unwrap());
        assert_eq!(&decoded.payload[8..], top_k);
    }

    #[test]
    fn auto_lossless_selects_bf16_when_both_half_formats_are_exact() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let frame = f32_frame(1, &[1.0, 2.0]);
        let forwarded =
            forwarded_stage_message_timed(&config, &incoming_message(), &frame, 2).unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::Bf16RneV1
        );
        assert_eq!(
            forwarded.message.activation.len(),
            activation_frame_wire_bytes(
                StageActivationCodec::Bf16RneV1,
                &stage_activation_desc(&frame.desc).unwrap(),
            )
            .unwrap()
        );
    }

    #[test]
    fn auto_lossless_selects_f16_when_only_f16_is_exact() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let forwarded = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(1, &[1.000_976_6, 2.0]),
            2,
        )
        .unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::F16RneV1
        );
    }

    #[test]
    fn auto_lossless_uses_raw_for_mixed_primary_and_sideband_requirements() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let frame = two_f32_part_frame(&[1.000_976_6, 2.0, 65_536.0, 4.0]);
        let forwarded =
            forwarded_stage_message_timed(&config, &incoming_message(), &frame, 2).unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::RawF32V1
        );
        assert_eq!(
            forwarded.message.activation.len(),
            activation_frame_wire_bytes(
                StageActivationCodec::RawF32V1,
                &stage_activation_desc(&frame.desc).unwrap(),
            )
            .unwrap()
        );
    }

    #[test]
    fn auto_lossless_rejects_a_non_raw_fallback() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::F16RneV1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let error = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(1, &[1.0, 2.0]),
            2,
        )
        .err()
        .expect("invalid AutoLosslessV1 fallback must fail");

        assert!(format!("{error:#}").contains("incompatible with configured codec"));
    }

    #[test]
    fn first_stage_applies_auto_lossless_selection() {
        let mut config = stage_config();
        config.stage_index = 0;
        config.layer_start = 0;
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let forwarded = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(1, &[1.0, 2.0]),
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.state.source_stage_index, 0);
        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::Bf16RneV1
        );
    }

    #[test]
    fn compact_activation_encode_failure_does_not_fall_back_to_raw() {
        let mut config = stage_config();
        config.activation_codec = skippy_protocol::StageActivationCodec::F16RneV1;
        let error = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(1, &[f32::MAX, 1.0]),
            2,
        )
        .err()
        .expect("F16 overflow must fail the stage connection");

        assert!(format!("{error:#}").contains("F16 activation value is out of range"));
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
        let output = f32_frame(1, &[1.0, 2.0, 3.0, 4.0]);

        let error = forwarded_stage_message_timed(&config, &incoming, &output, 4)
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
            forwarded_stage_message_timed(&config, &incoming, &f32_frame(1, &[1.0; 16]), 4,)
                .is_ok()
        );
    }
}
