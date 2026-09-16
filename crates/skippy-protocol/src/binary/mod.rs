mod activation;
mod activation_codec;
mod codec;
mod types;

pub use activation::{
    activation_frame_wire_bytes, decode_activation_frame, decode_raw_activation_frame,
    encode_activation_frame, encode_raw_activation_frame, select_lossless_activation_codec,
};
pub use codec::{
    read_stage_message, read_stage_message_for_codec, read_stage_message_for_codec_policy,
    recv_ready, recv_reply, send_ready, send_reply_ack, send_reply_ack_with_stats,
    send_reply_message, send_reply_predicted, send_reply_predicted_tokens_with_stats,
    send_reply_predicted_tokens_with_window_and_stats, send_reply_predicted_with_stats,
    send_reply_predicted_with_tokens_and_stats, send_reply_predicted_with_tokens_window_and_stats,
    write_stage_message,
};
pub use types::sampling_flags;
pub use types::{
    LLAMA_TOKEN_NULL, MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_ACTIVATION_DIMS,
    MAX_STAGE_ACTIVATION_PARTS, MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
    MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_DRY_SEQUENCE_BREAKERS, MAX_STAGE_LOGIT_BIAS,
    MAX_STAGE_PREDICTED_TOKENS, MAX_STAGE_SAMPLERS, MAX_STAGE_SAMPLING_STRING_BYTES,
    MAX_STAGE_SIDEBAND_VALUES, MAX_STAGE_STATE_IMPORT_BYTES, READY_MAGIC,
    STAGE_ACTIVATION_FRAME_VERSION, STAGE_ACTIVATION_IDENTITY_BYTES,
    STAGE_ACTIVATION_PART_OPTIONAL, STAGE_LOGIT_BIAS_WIRE_BYTES, STAGE_SAMPLING_CONFIG_BASE_BYTES,
    STAGE_STATE_HEADER_BYTES, STAGE_STATE_VERSION, STAGE_WIRE_FIXED_HEADER_BYTES,
    StageActivationDesc, StageActivationFrame, StageActivationPartDesc, StageLogitBias,
    StageNativeMtpDraft, StageReply, StageReplyStats, StageReplyWindow, StageRequestEpoch,
    StageSamplingConfig, StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind,
    WireStagePhase, state_flags,
};

pub(crate) fn invalid_data(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

pub(crate) fn invalid_input(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::{Cursor, Read, Write};
    use std::rc::Rc;

    fn push_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_state_header(bytes: &mut Vec<u8>, state: StageStateHeader) {
        push_i32(bytes, state.version);
        push_i32(bytes, state.seq_id);
        push_i32(bytes, state.phase);
        push_i32(bytes, state.flags);
        push_i32(bytes, state.checkpoint_generation);
        push_i32(bytes, state.prompt_token_count);
        push_i32(bytes, state.decode_step);
        push_i32(bytes, state.current_token);
        push_i32(bytes, state.source_stage_index);
        push_i32(bytes, state.activation_codec.binary_wire_id());
    }

    fn stage_frame_prefix(
        kind: WireMessageKind,
        token_count: i32,
        token_sideband_count: i32,
        position_sideband_count: i32,
        state: StageStateHeader,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, kind as i32);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, token_count);
        push_i32(&mut bytes, token_sideband_count);
        push_i32(&mut bytes, position_sideband_count);
        push_state_header(&mut bytes, state);
        push_i32(&mut bytes, 0);
        push_u64(&mut bytes, 7);
        push_u64(&mut bytes, 11);
        bytes
    }

    fn assert_invalid_data<T: std::fmt::Debug>(result: std::io::Result<T>, expected: &str) {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), expected);
    }

    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        calls: Rc<Cell<usize>>,
    }

    impl Read for CountingReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            self.inner.read(output)
        }
    }

    struct CountingWriter {
        bytes: Vec<u8>,
        calls: Rc<Cell<usize>>,
    }

    impl Write for CountingWriter {
        fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            self.bytes.extend_from_slice(input);
            Ok(input.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn ready_round_trips() {
        let mut bytes = Vec::new();
        send_ready(&mut bytes).unwrap();
        recv_ready(Cursor::new(bytes)).unwrap();
    }

    #[test]
    fn reply_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted(&mut bytes, 42).unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        assert_eq!(reply.predicted_tokens, vec![42]);
        assert_eq!(reply.native_mtp_draft, None);
    }

    #[test]
    fn reply_round_trips_typed_native_mtp_draft() {
        let reply = StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: 42,
            predicted_tokens: vec![42],
            native_mtp_draft: Some(StageNativeMtpDraft {
                token_ids: vec![43, 44],
                proposal_compute_us: 12_345,
            }),
            window: StageReplyWindow::default(),
            stats: StageReplyStats::default(),
        };
        let mut bytes = Vec::new();
        send_reply_message(&mut bytes, &reply).unwrap();

        assert_eq!(recv_reply(Cursor::new(bytes)).unwrap(), reply);
    }

    #[test]
    fn predicted_token_reply_preserves_sideband_tokens() {
        let mut bytes = Vec::new();
        send_reply_predicted_with_tokens_and_stats(
            &mut bytes,
            42,
            &[42, 43, 123],
            StageReplyStats::default(),
        )
        .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        assert_eq!(reply.predicted_tokens, vec![42, 43, 123]);
    }

    #[test]
    fn reply_stats_preserve_prefill_edge_transport() {
        let mut stats = StageReplyStats::default();
        stats.observe_prefill_edge_transport(2, 12_000, 3_000, 1_048_576);
        stats.observe_prefill_edge_transport(1, 4_000, 40_000, 524_288);
        stats.observe_prefill_compute(2, 18_000, 128);
        stats.observe_prefill_compute(1, 42_000, 128);
        stats.observe_prefill_compute(3, 25_000, 64);

        let mut bytes = Vec::new();
        send_reply_predicted_with_stats(&mut bytes, 42, stats).unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();

        assert_eq!(reply.stats.prefill_edge_write_us_max, 12_000);
        assert_eq!(reply.stats.prefill_edge_wait_us_max, 40_000);
        assert_eq!(reply.stats.prefill_edge_total_us_max, 44_000);
        assert_eq!(reply.stats.prefill_edge_stage_index, 1);
        assert_eq!(reply.stats.prefill_edge_activation_bytes_max, 524_288);
        assert_eq!(reply.stats.prefill_edge_observation_count, 2);
        assert_eq!(reply.stats.prefill_compute_us_at_slowest_rate, 42_000);
        assert_eq!(reply.stats.prefill_compute_stage_index, 1);
        assert_eq!(reply.stats.prefill_compute_token_count_at_slowest_rate, 128);
        assert_eq!(reply.stats.prefill_compute_observation_count, 3);
    }

    #[test]
    fn prefill_compute_calibration_ignores_short_tail_overhead() {
        let mut stats = StageReplyStats::default();
        stats.observe_prefill_compute(1, 48_000, 128);
        stats.observe_prefill_compute(1, 30_000, 9);

        assert_eq!(stats.prefill_compute_us_at_slowest_rate, 48_000);
        assert_eq!(stats.prefill_compute_stage_index, 1);
        assert_eq!(stats.prefill_compute_token_count_at_slowest_rate, 128);
        assert_eq!(stats.prefill_compute_observation_count, 2);
    }

    #[test]
    fn token_vector_reply_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted_tokens_with_stats(&mut bytes, &[1, 2, 3], StageReplyStats::default())
            .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedTokens);
        assert_eq!(reply.predicted, 1);
        assert_eq!(reply.predicted_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn reply_window_metadata_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted_tokens_with_window_and_stats(
            &mut bytes,
            &[1, 2, 3],
            StageReplyWindow { window_id: 42 },
            StageReplyStats::default(),
        )
        .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();

        assert_eq!(reply.kind, WireReplyKind::PredictedTokens);
        assert_eq!(reply.predicted_tokens, vec![1, 2, 3]);
        assert_eq!(reply.window.window_id, 42);
    }

    #[test]
    fn reply_rejects_predicted_token_count_over_limit() {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, WireReplyKind::PredictedTokens as i32);
        push_i32(&mut bytes, 1);
        push_i32(
            &mut bytes,
            i32::try_from(MAX_STAGE_PREDICTED_TOKENS + 1).unwrap(),
        );

        assert_invalid_data(
            recv_reply(Cursor::new(bytes)),
            "predicted token count exceeds maximum",
        );
    }

    fn multipart_activation_frame(token_count: u32) -> StageActivationFrame {
        let hidden_values = (0..token_count * 2)
            .map(|value| value as f32 * 0.5)
            .collect::<Vec<_>>();
        let hidden = hidden_values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let routes = (0..token_count * 3)
            .map(|value| i32::try_from(value).unwrap())
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let hidden_bytes = u64::try_from(hidden.len()).unwrap();
        let route_bytes = u64::try_from(routes.len()).unwrap();
        let mut payload = hidden;
        payload.extend_from_slice(&routes);
        StageActivationFrame {
            desc: StageActivationDesc {
                version: STAGE_ACTIVATION_FRAME_VERSION,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                token_count,
                sequence_count: 1,
                payload_bytes: hidden_bytes + route_bytes,
                frontier_identity: [0x11; STAGE_ACTIVATION_IDENTITY_BYTES],
                parts: vec![
                    StageActivationPartDesc {
                        identity: [0x22; STAGE_ACTIVATION_IDENTITY_BYTES],
                        ggml_type: 0,
                        rank: 2,
                        token_axis: 1,
                        flags: 0,
                        dimensions: [2, i64::from(token_count), 1, 1],
                        byte_strides: [
                            4,
                            8,
                            8 * u64::from(token_count),
                            8 * u64::from(token_count),
                        ],
                        payload_offset: 0,
                        payload_bytes: hidden_bytes,
                    },
                    StageActivationPartDesc {
                        identity: [0x33; STAGE_ACTIVATION_IDENTITY_BYTES],
                        ggml_type: 26,
                        rank: 2,
                        token_axis: 1,
                        flags: STAGE_ACTIVATION_PART_OPTIONAL,
                        dimensions: [3, i64::from(token_count), 1, 1],
                        byte_strides: [
                            4,
                            12,
                            12 * u64::from(token_count),
                            12 * u64::from(token_count),
                        ],
                        payload_offset: hidden_bytes,
                        payload_bytes: route_bytes,
                    },
                ],
            },
            payload,
        }
    }

    fn activation_message(
        frame: &StageActivationFrame,
        codec: crate::StageActivationCodec,
    ) -> StageWireMessage {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.checkpoint_generation = 3;
        state.prompt_token_count = i32::try_from(frame.desc.token_count).unwrap();
        state.decode_step = 0;
        state.current_token = 11;
        state.source_stage_index = frame.desc.producer_stage_index;
        state.activation_codec = codec;
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 1,
            token_count: i32::try_from(frame.desc.token_count).unwrap(),
            state,
            request_id: 7,
            session_id: 11,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11],
            positions: Vec::new(),
            activation: encode_activation_frame(codec, &frame.desc, &frame.payload).unwrap(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn stage_message_round_trips_multipart_activation_and_sampling() {
        let frame = multipart_activation_frame(1);
        let mut message = activation_message(&frame, crate::StageActivationCodec::RawF32V1);
        message.sampling = Some(StageSamplingConfig {
            flags: 1,
            seed: 42,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            logit_bias: vec![StageLogitBias {
                token_id: 123,
                bias: -50.0,
            }],
            ..StageSamplingConfig::default()
        });

        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::DecodeEmbd);
        assert_eq!(decoded.tokens, vec![11]);
        assert_eq!(decoded.activation_frame().unwrap(), Some(frame));
        assert_eq!(decoded.state.source_stage_index, 0);
        assert_eq!(
            decoded.request_epoch(),
            StageRequestEpoch {
                request_id: 7,
                session_id: 11,
                checkpoint_generation: 3,
                prompt_token_count: 1,
                decode_step: 0,
            }
        );
        assert_ne!(decoded.state.flags & state_flags::SAMPLING, 0);
        let sampling = decoded.sampling.expect("sampling extension round-tripped");
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.top_k, 40);
        assert_eq!(sampling.logit_bias[0].token_id, 123);
        assert_eq!(sampling.logit_bias[0].bias, -50.0);
    }

    #[test]
    fn stage_message_round_trips_every_activation_codec_with_typed_parts() {
        let frame = multipart_activation_frame(2);
        let i32_part = frame.desc.parts[1];
        let i32_start = usize::try_from(i32_part.payload_offset).unwrap();
        for codec in [
            crate::StageActivationCodec::RawF32V1,
            crate::StageActivationCodec::F16RneV1,
            crate::StageActivationCodec::Bf16RneV1,
            crate::StageActivationCodec::S8RowF32RneV1,
        ] {
            let message = activation_message(&frame, codec);
            assert_eq!(
                message.activation.len(),
                activation_frame_wire_bytes(codec, &frame.desc).unwrap()
            );
            let mut bytes = Vec::new();
            write_stage_message(&mut bytes, &message).unwrap();
            let decoded = read_stage_message_for_codec(Cursor::new(bytes), 2, codec).unwrap();
            let decoded_frame = decoded.activation_frame().unwrap().unwrap();
            assert_eq!(decoded_frame.desc, frame.desc);
            assert_eq!(
                &decoded_frame.payload[i32_start..],
                &frame.payload[i32_start..],
                "non-F32 parts must stay byte-exact for {codec:?}"
            );
        }
    }

    #[test]
    fn stage_message_rejects_codec_mismatch_and_malformed_header() {
        let frame = multipart_activation_frame(1);
        let codec = crate::StageActivationCodec::F16RneV1;
        let message = activation_message(&frame, codec);
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_invalid_data(
            read_stage_message_for_codec(
                Cursor::new(bytes.clone()),
                2,
                crate::StageActivationCodec::RawF32V1,
            ),
            "stage activation codec mismatch",
        );

        let mut wrong_size = bytes.clone();
        wrong_size[60..64].copy_from_slice(&5_i32.to_le_bytes());
        assert_eq!(
            read_stage_message_for_codec(Cursor::new(wrong_size), 2, codec)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );

        bytes[56..60].copy_from_slice(&99_i32.to_le_bytes());
        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2),
            "unknown stage activation codec",
        );
    }

    #[test]
    fn stage_message_auto_lossless_policy_admits_only_exact_codec_set() {
        let frame = multipart_activation_frame(1);
        for codec in [
            crate::StageActivationCodec::RawF32V1,
            crate::StageActivationCodec::Bf16RneV1,
            crate::StageActivationCodec::F16RneV1,
        ] {
            let message = activation_message(&frame, codec);
            let mut bytes = Vec::new();
            write_stage_message(&mut bytes, &message).unwrap();
            let decoded = read_stage_message_for_codec_policy(
                Cursor::new(bytes),
                2,
                crate::StageActivationCodec::RawF32V1,
                crate::StageActivationCodecPolicy::AutoLosslessV1,
            )
            .unwrap();
            assert_eq!(decoded.state.activation_codec, codec);
        }

        let message = activation_message(&frame, crate::StageActivationCodec::S8RowF32RneV1);
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_invalid_data(
            read_stage_message_for_codec_policy(
                Cursor::new(bytes),
                2,
                crate::StageActivationCodec::RawF32V1,
                crate::StageActivationCodecPolicy::AutoLosslessV1,
            ),
            "stage activation codec mismatch",
        );
    }

    #[test]
    fn verify_window_message_round_trips_window_metadata() {
        let mut state = StageStateHeader::new(WireMessageKind::VerifyWindow);
        state.seq_id = 42;
        state.prompt_token_count = 128;
        state.decode_step = 7;
        state.current_token = 1001;
        let message = StageWireMessage {
            kind: WireMessageKind::VerifyWindow,
            pos_start: 135,
            token_count: 4,
            state,
            request_id: 7,
            session_id: 11,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![1001, 1002, 1003, 1004],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };

        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();

        assert_eq!(decoded.kind, WireMessageKind::VerifyWindow);
        assert_eq!(decoded.verify_window_id(), Some(42));
        assert_eq!(decoded.verify_window_base_position(), Some(135));
        assert_eq!(decoded.verify_window_token_count(), Some(4));
        assert_eq!(decoded.authoritative_session_position(), Some(135));
        assert_eq!(decoded.tokens, vec![1001, 1002, 1003, 1004]);
        assert_eq!(decoded.state.decode_step, 7);
    }

    #[test]
    fn only_decode_messages_carry_authoritative_session_positions() {
        let mut decode = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 17,
            token_count: 1,
            state: StageStateHeader::new(WireMessageKind::DecodeEmbd),
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![3],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        assert_eq!(decode.authoritative_session_position(), Some(17));

        decode.pos_start = -1;
        assert_eq!(decode.authoritative_session_position(), None);

        decode.kind = WireMessageKind::PrefillEmbd;
        assert_eq!(decode.authoritative_session_position(), None);
    }

    #[test]
    fn stage_message_rejects_old_state_version() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.version = STAGE_STATE_VERSION - 1;
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, 1, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2),
            "unsupported stage state version",
        );
    }

    #[test]
    fn stage_message_rejects_legacy_kind_10() {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, 10);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, 1);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, 0);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2),
            "unknown stage message kind",
        );
    }

    #[test]
    fn stage_message_estimates_full_wire_transfer_bytes() {
        let frame = multipart_activation_frame(2);
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = 0;
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 2,
            state,
            request_id: 7,
            session_id: 11,
            sampling: Some(StageSamplingConfig {
                flags: 1,
                logit_bias: vec![
                    StageLogitBias {
                        token_id: 1,
                        bias: -1.0,
                    },
                    StageLogitBias {
                        token_id: 2,
                        bias: 1.0,
                    },
                ],
                ..StageSamplingConfig::default()
            }),
            chat_sampling_metadata: Some("{}".to_string()),
            tokens: vec![1, 2],
            positions: vec![0],
            activation: encode_raw_activation_frame(&frame).unwrap(),
            raw_bytes: Vec::new(),
        };

        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_eq!(message.estimated_wire_bytes(), bytes.len());
    }

    #[test]
    fn request_epoch_orders_only_matching_flows() {
        let older = StageRequestEpoch {
            request_id: 7,
            session_id: 11,
            checkpoint_generation: 1,
            prompt_token_count: 8,
            decode_step: 2,
        };
        let newer = StageRequestEpoch {
            request_id: 7,
            session_id: 11,
            checkpoint_generation: 1,
            prompt_token_count: 8,
            decode_step: 3,
        };
        let different_session = StageRequestEpoch {
            session_id: 12,
            ..newer
        };

        assert!(older.same_flow(newer));
        assert!(older.is_stale_for(newer));
        assert!(!newer.is_stale_for(older));
        assert!(!older.same_flow(different_session));
        assert!(!older.is_stale_for(different_session));
    }

    #[test]
    fn request_epoch_staleness_orders_generation_before_prompt_before_decode() {
        let base = StageRequestEpoch {
            request_id: 7,
            session_id: 11,
            checkpoint_generation: 1,
            prompt_token_count: 8,
            decode_step: 3,
        };
        let newer_checkpoint = StageRequestEpoch {
            checkpoint_generation: 2,
            prompt_token_count: 0,
            decode_step: 0,
            ..base
        };
        let newer_prompt = StageRequestEpoch {
            prompt_token_count: 9,
            decode_step: 0,
            ..base
        };
        let newer_decode = StageRequestEpoch {
            decode_step: 4,
            ..base
        };

        assert!(base.same_flow(newer_checkpoint));
        assert!(base.is_stale_for(newer_checkpoint));
        assert!(!newer_checkpoint.is_stale_for(base));
        assert!(base.is_stale_for(newer_prompt));
        assert!(!newer_prompt.is_stale_for(base));
        assert!(base.is_stale_for(newer_decode));
        assert!(!newer_decode.is_stale_for(base));
    }

    #[test]
    fn generation_config_round_trips_sampling_metadata() {
        let message = StageWireMessage::configure_generation(
            7,
            11,
            123,
            Some(StageSamplingConfig {
                flags: 1,
                seed: 42,
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                ..StageSamplingConfig::default()
            }),
            Some("{\"grammar\":\"root ::= \\\"x\\\"\"}".to_string()),
        );
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::ConfigureGeneration);
        assert_eq!(decoded.token_count, 0);
        assert_eq!(decoded.tokens, Vec::<i32>::new());
        assert_eq!(decoded.activation, Vec::<u8>::new());
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.session_id, 11);
        assert_eq!(decoded.state.prompt_token_count, 123);
        assert_ne!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert_ne!(decoded.state.flags & state_flags::CHAT_SAMPLING_METADATA, 0);
        assert_eq!(
            decoded.chat_sampling_metadata.as_deref(),
            Some("{\"grammar\":\"root ::= \\\"x\\\"\"}")
        );
        let sampling = decoded.sampling.expect("sampling extension round-tripped");
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.top_k, 40);
    }

    #[test]
    fn stage_message_rejects_sampling_metadata_length_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::ConfigureGeneration);
        state.flags |= state_flags::CHAT_SAMPLING_METADATA;
        let mut bytes = stage_frame_prefix(WireMessageKind::ConfigureGeneration, 0, 0, 0, state);
        push_u32(
            &mut bytes,
            u32::try_from(MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES + 1).unwrap(),
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "chat sampling metadata length exceeds maximum",
        );
    }

    #[test]
    fn driver_origin_message_round_trips_without_activation() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.prompt_token_count = 2;
        state.current_token = 22;
        state.source_stage_index = -1;
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 2,
            state,
            request_id: 13,
            session_id: 17,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11, 22],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.tokens, vec![11, 22]);
        assert!(decoded.activation.is_empty());
        assert_eq!(decoded.state.source_stage_index, -1);
        assert_eq!(decoded.request_id, 13);
        assert_eq!(decoded.session_id, 17);
        assert_eq!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert!(decoded.sampling.is_none());
    }

    #[test]
    fn stop_message_round_trips_without_activation_from_a_stage() {
        let mut state = StageStateHeader::new(WireMessageKind::Stop);
        state.source_stage_index = 0;
        let message = StageWireMessage {
            kind: WireMessageKind::Stop,
            pos_start: 0,
            token_count: 0,
            state,
            request_id: 19,
            session_id: 23,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };

        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();

        assert_eq!(decoded.kind, WireMessageKind::Stop);
        assert_eq!(decoded.state.source_stage_index, 0);
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn stage_message_bulk_reads_sidebands() {
        let value_count = 4_096;
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = -1;
        let mut bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            value_count,
            value_count,
            value_count,
            state,
        );
        for value in 0..value_count {
            push_i32(&mut bytes, value);
        }
        for value in 0..value_count {
            push_i32(&mut bytes, value + 10_000);
        }
        let calls = Rc::new(Cell::new(0));
        let reader = CountingReader {
            inner: Cursor::new(bytes),
            calls: calls.clone(),
        };

        let decoded = read_stage_message(reader, 2_048).unwrap();

        assert_eq!(decoded.tokens.len(), value_count as usize);
        assert_eq!(decoded.positions.len(), value_count as usize);
        assert!(
            calls.get() < 128,
            "sideband decoding used {} reads",
            calls.get()
        );
    }

    #[test]
    fn stage_message_bulk_writes_sidebands() {
        let value_count = 4_096;
        let kind = WireMessageKind::TrimSession;
        let message = StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(kind),
            request_id: 23,
            session_id: 29,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: (0..value_count).collect(),
            positions: (10_000..10_000 + value_count).collect(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let calls = Rc::new(Cell::new(0));
        let writer = CountingWriter {
            bytes: Vec::new(),
            calls: calls.clone(),
        };

        write_stage_message(writer, &message).unwrap();

        assert!(
            calls.get() < 128,
            "sideband encoding used {} writes",
            calls.get()
        );
    }

    #[test]
    fn stage_message_rejects_token_sideband_count_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = -1;
        let bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            0,
            i32::try_from(MAX_STAGE_SIDEBAND_VALUES + 1).unwrap(),
            0,
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "token sideband count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_rejects_position_sideband_count_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = -1;
        let bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            0,
            0,
            i32::try_from(MAX_STAGE_SIDEBAND_VALUES + 1).unwrap(),
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "position sideband count exceeds maximum",
        );
    }

    #[test]
    fn prefill_wire_overhead_is_fixed_and_bounded() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.prompt_token_count = 128;
        state.current_token = 127;
        state.source_stage_index = -1;
        let tokens: Vec<i32> = (0..128).collect();
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: tokens.len() as i32,
            state,
            request_id: u64::MAX - 1,
            session_id: u64::MAX,
            sampling: None,
            chat_sampling_metadata: None,
            tokens,
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();

        assert_eq!(STAGE_STATE_HEADER_BYTES, 40);
        assert_eq!(STAGE_SAMPLING_CONFIG_BASE_BYTES, 108);
        assert_eq!(STAGE_WIRE_FIXED_HEADER_BYTES, 80);
        assert_eq!(
            bytes.len(),
            STAGE_WIRE_FIXED_HEADER_BYTES + message.tokens.len() * 4
        );
        const { assert!(STAGE_WIRE_FIXED_HEADER_BYTES <= 80) };
    }

    #[test]
    fn verify_retirement_round_trips_exact_identity() {
        let kind = WireMessageKind::RetireVerifyWindow;
        let message = StageWireMessage {
            kind,
            pos_start: 128,
            token_count: 8,
            state: StageStateHeader::new(kind),
            request_id: 23,
            session_id: 29,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();

        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.pos_start, 128);
        assert_eq!(decoded.token_count, 8);
        assert!(decoded.state.matches_kind(kind));
    }

    #[test]
    fn session_control_messages_are_fixed_header_only() {
        let kind = WireMessageKind::TrimSession;
        let message = StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(kind),
            request_id: 23,
            session_id: 29,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_eq!(bytes.len(), STAGE_WIRE_FIXED_HEADER_BYTES);
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.request_id, 23);
        assert_eq!(decoded.session_id, 29);
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn state_import_message_round_trips_raw_bytes() {
        let state = StageStateHeader::new(WireMessageKind::StateImport);
        let message = StageWireMessage {
            kind: WireMessageKind::StateImport,
            pos_start: 0,
            token_count: 4,
            state,
            request_id: 31,
            session_id: 37,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::StateImport);
        assert_eq!(decoded.raw_bytes, vec![1, 2, 3, 4]);
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn state_import_rejects_raw_byte_count_over_limit() {
        let state = StageStateHeader::new(WireMessageKind::StateImport);
        let bytes = stage_frame_prefix(
            WireMessageKind::StateImport,
            i32::try_from(MAX_STAGE_STATE_IMPORT_BYTES + 1).unwrap(),
            0,
            0,
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "state import byte count exceeds maximum",
        );
    }

    #[test]
    fn state_import_writer_rejects_raw_byte_count_mismatch() {
        let state = StageStateHeader::new(WireMessageKind::StateImport);
        let message = StageWireMessage {
            kind: WireMessageKind::StateImport,
            pos_start: 0,
            token_count: 8,
            state,
            request_id: 31,
            session_id: 37,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        let error = write_stage_message(&mut bytes, &message)
            .expect_err("mismatched state import byte count should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "state import raw byte count mismatch");
    }

    #[test]
    fn state_export_message_round_trips_without_payload() {
        let state = StageStateHeader::new(WireMessageKind::StateExport);
        let message = StageWireMessage {
            kind: WireMessageKind::StateExport,
            pos_start: 0,
            token_count: 0,
            state,
            request_id: 41,
            session_id: 43,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::StateExport);
        assert!(decoded.raw_bytes.is_empty());
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn activation_descriptor_rejects_duplicate_identity() {
        let frame = multipart_activation_frame(1);
        let mut duplicate = frame.desc.clone();
        duplicate.parts[1].identity = duplicate.parts[0].identity;

        assert_invalid_data(
            encode_activation_frame(
                crate::StageActivationCodec::RawF32V1,
                &duplicate,
                &frame.payload,
            ),
            "activation frame has duplicate part identities",
        );
    }

    #[test]
    fn activation_descriptor_rejects_token_dimension_mismatch() {
        let frame = multipart_activation_frame(2);
        let mut mismatched = frame.desc.clone();
        mismatched.parts[0].dimensions[1] = 1;

        assert_invalid_data(
            encode_activation_frame(
                crate::StageActivationCodec::RawF32V1,
                &mismatched,
                &frame.payload,
            ),
            "activation part token dimension does not match frame",
        );
    }

    #[test]
    fn activation_descriptor_rejects_payload_over_limit_before_allocation() {
        let mut desc = multipart_activation_frame(1).desc;
        let elements = u64::try_from(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 + 1).unwrap();
        desc.parts.truncate(1);
        desc.parts[0].dimensions = [i64::try_from(elements).unwrap(), 1, 1, 1];
        desc.parts[0].byte_strides = [4, elements * 4, elements * 4, elements * 4];
        desc.parts[0].payload_bytes = elements * 4;
        desc.payload_bytes = elements * 4;

        assert_invalid_data(
            activation_frame_wire_bytes(crate::StageActivationCodec::RawF32V1, &desc),
            "decoded activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn multipart_activation_frame_can_be_taken() {
        let frame = multipart_activation_frame(1);
        let mut message = activation_message(&frame, crate::StageActivationCodec::RawF32V1);

        assert_eq!(message.take_activation_frame().unwrap(), Some(frame));
        assert!(message.activation.is_empty());
    }

    #[test]
    fn lossless_selection_checks_every_f32_part_and_preserves_raw_fallback() {
        let frame = multipart_activation_frame(2);
        let selected = select_lossless_activation_codec(
            &frame.desc,
            &frame.payload,
            &[
                crate::StageActivationCodec::Bf16RneV1,
                crate::StageActivationCodec::F16RneV1,
                crate::StageActivationCodec::RawF32V1,
            ],
        )
        .unwrap();

        assert!(
            [
                crate::StageActivationCodec::Bf16RneV1,
                crate::StageActivationCodec::F16RneV1,
                crate::StageActivationCodec::RawF32V1,
            ]
            .contains(&selected)
        );
    }
}
