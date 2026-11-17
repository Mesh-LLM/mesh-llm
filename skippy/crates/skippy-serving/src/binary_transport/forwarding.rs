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
    /// Bytes of the message's activation that are DFlash feature rows.
    pub dflash_feature_bytes: usize,
}

/// Forwards one computed prefill chunk as one or more stage messages, each
/// within `frame_budget` wire bytes once every later stage has added
/// `downstream_bytes_per_token` to each of its tokens (the DFlash feature rows
/// they capture). A frame is sized by the larger of its encoded and raw
/// activation bytes, since a later stage may forward the same tokens under a
/// codec that compresses them less. The chunk computes as one batch whatever
/// its size; only its output is cut along the token axis, into consecutive
/// prefill messages that downstream stages run in order, as they would
/// smaller chunks. Every part of the frame is cut by token, so feature rows
/// stay with their tokens.
pub(crate) fn forwarded_prefill_frames(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    frame_budget: usize,
    downstream_bytes_per_token: usize,
) -> Result<Vec<ForwardedStageMessage>> {
    let tokens = usize::try_from(output.desc.token_count).context("output token count")?;
    let downstream_bytes = tokens.saturating_mul(downstream_bytes_per_token);
    let sliceable_positions = incoming.positions.is_empty() || incoming.positions.len() == tokens;
    let splittable = tokens > 1 && incoming.tokens.len() == tokens && sliceable_positions;
    // A frame too large for the encoder itself is split in halves; one over
    // the budget is split into as many pieces as its size needs. A piece
    // that still overflows (an exact lossless codec's size depends on the
    // values) is split again.
    let downstream_frame_bytes = |message: &StageWireMessage| {
        let wire = message.estimated_wire_bytes();
        let raw = wire
            .saturating_sub(message.activation.len())
            .saturating_add(output.payload.len());
        wire.max(raw).saturating_add(downstream_bytes)
    };
    let pieces = match forwarded_stage_message_timed(config, incoming, output, activation_width) {
        Ok(whole) if downstream_frame_bytes(&whole.message) <= frame_budget => {
            return Ok(vec![whole]);
        }
        Ok(whole) if !splittable => bail!(
            "a {tokens}-token prefill frame of {} bytes ({} with its raw activation and what later stages add) exceeds the {frame_budget}-byte stage frame budget",
            whole.message.estimated_wire_bytes(),
            downstream_frame_bytes(&whole.message)
        ),
        Ok(whole) => downstream_frame_bytes(&whole.message)
            .div_ceil(frame_budget.max(1))
            .clamp(2, tokens),
        Err(error) if !splittable => return Err(error),
        Err(_) => 2,
    };
    let piece_tokens = tokens.div_ceil(pieces);
    let mut frames = Vec::new();
    for start in (0..tokens).step_by(piece_tokens) {
        let count = piece_tokens.min(tokens - start);
        let piece_output = crate::runtime_state::slice_activation_frame(output, start, count)?;
        let mut piece = incoming.clone();
        piece.pos_start = incoming
            .pos_start
            .checked_add(i32::try_from(start).context("prefill slice start")?)
            .context("prefill slice position")?;
        piece.token_count = i32::try_from(count).context("prefill slice tokens")?;
        piece.tokens = incoming.tokens[start..start + count].to_vec();
        if !incoming.positions.is_empty() {
            piece.positions = incoming.positions[start..start + count].to_vec();
        }
        if let Some(&last) = piece.tokens.last() {
            piece.state.current_token = last;
        }
        frames.extend(forwarded_prefill_frames(
            config,
            &piece,
            &piece_output,
            activation_width,
            frame_budget,
            downstream_bytes_per_token,
        )?);
    }
    Ok(frames)
}

/// The budget for each prefill frame stage 0 sends: the stage frame budget
/// less a quarter. `forwarded_prefill_frames` also counts the DFlash feature
/// rows later stages append to the same tokens; the quarter is headroom for
/// what stage 0 cannot count.
pub(crate) const PREFILL_FRAME_BUDGET: usize = skippy_protocol::MAX_STAGE_FRAME_BYTES / 4 * 3;

/// Forward stage execution metadata while substituting the produced activation frame.
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
            // Fused restore/decode samples before the later ConfigureGeneration
            // message; every hop must carry its grammar and reasoning metadata.
            chat_sampling_metadata: incoming.chat_sampling_metadata.clone(),
            tokens: incoming.tokens.clone(),
            positions: incoming.positions.clone(),
            activation,
            readout_tokens: incoming.readout_tokens.clone(),
            raw_bytes: Vec::new(),
        },
        activation_encode_ms: encode_started.elapsed().as_secs_f64() * 1000.0,
        dflash_feature_bytes: skippy_runtime::dflash_feature_payload_bytes(output),
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
            host_expert_layers: None,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            resident_tensor_names: Vec::new(),
            selected_device: None::<StageDevice>,
            kv_cache: None::<StageKvCacheConfig>,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                persistent_lane_marker: false,
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                endpoint: "tcp://127.0.0.1:19000".to_string(),
            }),
            downstream: None,
            ..StageConfig::default()
        }
    }

    /// Construct an incoming stage request for forwarding regression tests.
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
            readout_tokens: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn prefill_message(tokens: &[i32], pos_start: i32) -> StageWireMessage {
        let mut message = incoming_message();
        message.kind = WireMessageKind::PrefillEmbd;
        message.state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        message.pos_start = pos_start;
        message.token_count = tokens.len() as i32;
        message.tokens = tokens.to_vec();
        message
    }

    /// A `tokens`-token frame: a 64-wide F32 part and a 2-wide I32 optional
    /// part, every value distinct.
    fn wide_prefill_frame(tokens: u32) -> ActivationFrame {
        let rows = tokens as usize;
        frame(
            tokens,
            vec![
                PartBytes {
                    identity: 1,
                    ggml_type: GGML_TYPE_F32,
                    flags: 0,
                    bytes: (0..rows * 64)
                        .flat_map(|index| (index as f32 * 0.37).to_le_bytes())
                        .collect(),
                },
                PartBytes {
                    identity: 2,
                    ggml_type: GGML_TYPE_I32,
                    flags: ACTIVATION_PART_OPTIONAL,
                    bytes: (0..rows * 2)
                        .flat_map(|index| (index as i32).to_le_bytes())
                        .collect(),
                },
            ],
        )
    }

    /// Each part's bytes of the activation frame `message` decodes to.
    fn decoded_parts(message: &StageWireMessage, config: &StageConfig) -> Vec<Vec<u8>> {
        let mut wire = Vec::new();
        skippy_protocol::binary::write_stage_message(&mut wire, message).unwrap();
        let decoded = skippy_protocol::binary::read_stage_message_for_codec(
            std::io::Cursor::new(wire),
            2,
            config.activation_codec,
        )
        .unwrap();
        let frame = decoded.activation_frame().unwrap().unwrap();
        frame
            .desc
            .parts
            .iter()
            .map(|part| {
                frame.payload[part.payload_offset as usize..][..part.payload_bytes as usize]
                    .to_vec()
            })
            .collect()
    }

    #[test]
    fn a_prefill_chunk_within_the_budget_leaves_as_one_frame() {
        let config = stage_config();
        let frames = forwarded_prefill_frames(
            &config,
            &prefill_message(&[1, 2, 3, 4, 5, 6], 10),
            &wide_prefill_frame(6),
            2,
            skippy_protocol::MAX_STAGE_FRAME_BYTES,
            0,
        )
        .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].message.token_count, 6);
    }

    #[test]
    fn a_prefill_chunk_over_the_budget_leaves_in_consecutive_frames() {
        let config = stage_config();
        let incoming = prefill_message(&[1, 2, 3, 4, 5, 6, 7], 10);
        let source = wide_prefill_frame(7);
        let whole = forwarded_stage_message(&config, &incoming, &source, 2).unwrap();
        // Room for about two tokens a frame.
        let budget = whole.estimated_wire_bytes() * 2 / 7 + 64;

        let frames = forwarded_prefill_frames(&config, &incoming, &source, 2, budget, 0).unwrap();

        assert!(frames.len() > 2);
        let mut next_position = 10;
        let mut tokens = Vec::new();
        let mut primary = Vec::new();
        let mut optional = Vec::new();
        for forwarded in &frames {
            let message = &forwarded.message;
            assert!(message.estimated_wire_bytes() <= budget);
            assert_eq!(message.pos_start, next_position);
            assert_eq!(message.token_count as usize, message.tokens.len());
            assert_eq!(message.state.current_token, *message.tokens.last().unwrap());
            next_position += message.token_count;
            tokens.extend_from_slice(&message.tokens);
            let parts = decoded_parts(message, &config);
            primary.extend_from_slice(&parts[0]);
            optional.extend_from_slice(&parts[1]);
        }
        assert_eq!(tokens, incoming.tokens);
        let source_parts = source.desc.parts().unwrap();
        assert_eq!(
            primary,
            source.payload[..source_parts[0].payload_bytes as usize]
        );
        assert_eq!(
            optional,
            source.payload[source_parts[1].payload_offset as usize..]
        );
    }

    #[test]
    fn a_single_token_over_the_budget_is_an_error() {
        let config = stage_config();
        assert!(
            forwarded_prefill_frames(
                &config,
                &prefill_message(&[1], 3),
                &wide_prefill_frame(1),
                2,
                16,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn a_frame_is_sized_by_its_raw_activation_when_the_codec_shrinks_it() {
        let mut config = stage_config();
        config.activation_codec = skippy_protocol::StageActivationCodec::F16RneV1;
        let incoming = prefill_message(&[1, 2, 3, 4, 5, 6, 7], 10);
        let source = wide_prefill_frame(7);
        let whole = forwarded_stage_message(&config, &incoming, &source, 2).unwrap();
        // The F16 frame fits; a later stage forwarding the same tokens as F32
        // would not.
        let budget = whole.estimated_wire_bytes();
        assert!(source.payload.len() > whole.activation.len());

        let frames = forwarded_prefill_frames(&config, &incoming, &source, 2, budget, 0).unwrap();

        assert!(frames.len() > 1);
        let tokens: Vec<_> = frames
            .iter()
            .flat_map(|forwarded| forwarded.message.tokens.clone())
            .collect();
        assert_eq!(tokens, incoming.tokens);
    }

    #[test]
    fn feature_rows_later_stages_add_count_against_the_budget() {
        let config = stage_config();
        let incoming = prefill_message(&[1, 2, 3, 4, 5, 6, 7], 10);
        let source = wide_prefill_frame(7);
        let whole = forwarded_stage_message(&config, &incoming, &source, 2).unwrap();
        let budget = whole.estimated_wire_bytes();
        // The frame alone fits; with what later stages add to each token it
        // does not.
        let added_per_token = budget / 7;

        let frames =
            forwarded_prefill_frames(&config, &incoming, &source, 2, budget, added_per_token)
                .unwrap();

        assert!(frames.len() > 1);
        let mut tokens = Vec::new();
        for forwarded in &frames {
            let message = &forwarded.message;
            let grown =
                message.estimated_wire_bytes() + message.token_count as usize * added_per_token;
            assert!(grown <= budget);
            tokens.extend_from_slice(&message.tokens);
        }
        assert_eq!(tokens, incoming.tokens);
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

    /// Fused restore requests retain sampling metadata across downstream hops.
    #[test]
    fn fused_restore_preserves_sampling_metadata_across_stage_hops() {
        let mut incoming = incoming_message();
        incoming.kind = WireMessageKind::TryRestorePrefillDecode;
        incoming.state = StageStateHeader::new(incoming.kind);
        incoming.state.prompt_token_count = 3;
        incoming.pos_start = 2;
        incoming.tokens = vec![7, 9, 11];
        incoming.chat_sampling_metadata =
            Some(r#"{"grammar":"root ::= \"B\" | \"S\"","grammar_lazy":false}"#.into());
        let metadata = incoming.chat_sampling_metadata.clone();
        let mut output = two_f32_part_frame(&[1.0, 2.0, 3.0, 4.0]);
        for stage_index in 0..2 {
            let mut config = stage_config();
            config.stage_index = stage_index;
            output.desc.producer_stage_index = stage_index as i32;
            let forwarded = forwarded_stage_message(&config, &incoming, &output, 2).unwrap();
            let mut wire = Vec::new();
            skippy_protocol::binary::write_stage_message(&mut wire, &forwarded).unwrap();
            incoming = skippy_protocol::binary::read_stage_message_for_codec(
                std::io::Cursor::new(wire),
                2,
                config.activation_codec,
            )
            .unwrap();
            assert_eq!(incoming.chat_sampling_metadata, metadata);
            assert_eq!(incoming.tokens, [7, 9, 11]);
            assert_eq!(incoming.state.prompt_token_count, 3);
        }
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
    fn dflash_feature_rows_cross_every_codec_unchanged_and_are_counted() {
        let mut frame = f32_frame(1, &[1.0, -2.0]);
        let mut rows = skippy_runtime::DFlashFeatureRows::new(2, 1);
        rows.insert(17, vec![3.0, -1.0]).unwrap();
        rows.append_to_frame_as(&mut frame, skippy_runtime::DFlashFeatureEncoding::Q8)
            .unwrap();
        let features = frame.payload[8..].to_vec();
        assert_eq!(features.len(), 4);
        for codec in [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::F16RneV1,
            StageActivationCodec::S8RowF32RneV1,
            StageActivationCodec::Fp8E4m3RowF32RneV1,
            StageActivationCodec::Fp8E4m3Block32E8m0RneV1,
        ] {
            let mut config = stage_config();
            config.activation_codec = codec;

            let forwarded =
                forwarded_stage_message_timed(&config, &incoming_message(), &frame, 2).unwrap();

            assert_eq!(forwarded.dflash_feature_bytes, features.len());
            let mut wire = Vec::new();
            skippy_protocol::binary::write_stage_message(&mut wire, &forwarded.message).unwrap();
            let decoded = skippy_protocol::binary::read_stage_message_for_codec(
                std::io::Cursor::new(wire),
                2,
                codec,
            )
            .unwrap();
            let decoded = decoded.activation_frame().unwrap().unwrap();
            assert_eq!(&decoded.payload[8..], &features[..], "{codec:?}");
        }
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
