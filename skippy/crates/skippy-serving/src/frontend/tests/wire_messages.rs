use super::*;

#[test]
fn multimodal_final_prefill_message_requests_downstream_prediction() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };

    let message = multimodal_prefill_message(MultimodalPrefillArgs {
        request_id: 11,
        session_id: 13,
        prompt_token_count: 17,
        pos_start: 0,
        token_count: 17,
        tokens: vec![3; 17],
        positions: Vec::new(),
        sampling: Some(sampling.clone()),
        final_chunk: true,
        request_class: Default::default(),
    })
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::PrefillFinalEmbd);
    assert!(message.kind.requires_predicted_reply());
    assert_eq!(message.token_count, 17);
    assert_eq!(message.state.current_token, LLAMA_TOKEN_NULL);
    assert_eq!(message.tokens, vec![3; 17]);
    assert_eq!(message.sampling, Some(sampling));
}

#[test]
fn restore_prefill_decode_message_carries_chat_sampling_metadata() {
    let metadata = r#"{"grammar":"chat","prompt_tokens":4}"#;
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };

    let message = embedded_restore_prefill_decode_message(RestorePrefillDecodeMessageArgs {
        request_id: 11,
        session_id: 13,
        prompt_token_count: 4,
        pos_start: 3,
        decode_step: 0,
        prefix_tokens: &[101, 102, 103],
        current: 104,
        sampling: Some(sampling.clone()),
        chat_sampling_metadata: Some(metadata),
        native_mtp_draft_tokens: 0,
        request_class: Default::default(),
    })
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::TryRestorePrefillDecode);
    assert_eq!(message.tokens, vec![101, 102, 103, 104]);
    assert_eq!(message.sampling, Some(sampling.clone()));
    assert_eq!(message.chat_sampling_metadata.as_deref(), Some(metadata));

    let mut encoded = Vec::new();
    write_stage_message(&mut encoded, &message).unwrap();
    let decoded = skippy_protocol::binary::read_stage_message(Cursor::new(encoded), 2816).unwrap();
    assert_eq!(decoded.kind, WireMessageKind::TryRestorePrefillDecode);
    assert_eq!(decoded.tokens, vec![101, 102, 103, 104]);
    assert_eq!(decoded.sampling, Some(sampling));
    assert_eq!(decoded.chat_sampling_metadata.as_deref(), Some(metadata));
    assert_eq!(decoded.state.native_mtp_draft_tokens(), Some(0));
}

#[test]
fn reusable_decode_message_updates_hot_path_fields() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };
    let mut message = ReusableDecodeMessage::new(ReusableDecodeMessageArgs {
        request_id: 11,
        session_id: 13,
        prompt_token_count: 4,
        base_pos_start: 4,
        sampling: Some(sampling.clone()),
        sideband_capacity: 4,
        native_mtp_draft_tokens: 3,
        request_class: skippy_inference_api::InferenceRequestClass::default(),
    })
    .unwrap();

    let first = message.update(0, 104).unwrap();
    assert_eq!(first.kind, WireMessageKind::DecodeEmbd);
    assert_eq!(first.state.native_mtp_draft_tokens(), Some(3));
    assert!(first.state.matches_kind(WireMessageKind::DecodeEmbd));
    assert_eq!(first.request_id, 11);
    assert_eq!(first.session_id, 13);
    assert_eq!(first.sampling, Some(sampling.clone()));
    assert_eq!(first.pos_start, 4);
    assert_eq!(first.token_count, 1);
    assert_eq!(first.state.prompt_token_count, 4);
    assert_eq!(first.state.decode_step, 0);
    assert_eq!(first.state.current_token, 104);
    assert_eq!(first.tokens, vec![104]);

    let second = message
        .update_with_tokens(1, 105, &[101, 102, 104, 105])
        .unwrap();
    assert_eq!(second.request_id, 11);
    assert_eq!(second.session_id, 13);
    assert_eq!(second.sampling, Some(sampling));
    assert_eq!(second.pos_start, 5);
    assert_eq!(second.token_count, 1);
    assert_eq!(second.state.prompt_token_count, 4);
    assert_eq!(second.state.decode_step, 1);
    assert_eq!(second.state.current_token, 105);
    assert_eq!(second.tokens, vec![101, 102, 104, 105]);
    assert!(second.positions.is_empty());
    assert!(second.activation.is_empty());
    assert!(second.raw_bytes.is_empty());
}

/// A downstream stage caches prompts from every prefill, restore, decode,
/// verify window and stop of a session, so each must carry the request's
/// class.
#[test]
fn decode_verify_and_stop_messages_carry_the_request_class() {
    use skippy_inference_api::{InferenceRequestClass, InferenceRequestTier};
    let class = InferenceRequestClass::new(InferenceRequestTier::Mesh, true);
    let served = |state| {
        crate::stage_request_class::stage_request_class(
            &skippy_protocol::StageConfig::default(),
            state,
        )
    };

    let decode = embedded_decode_message(DecodeMessageArgs {
        request_id: 11,
        session_id: 13,
        prompt_token_count: 4,
        pos_start: 4,
        decode_step: 0,
        current: 104,
        sampling: None,
        native_mtp_draft_tokens: 0,
        request_class: class,
    })
    .unwrap();
    assert_eq!(served(decode.state), class);

    let verify = embedded_verify_window_message(VerifyWindowMessageArgs {
        window_id: 1,
        request_id: 11,
        session_id: 13,
        prompt_token_count: 4,
        pos_start: 5,
        decode_step: 1,
        tokens: &[105, 106],
        sampling: None,
        native_mtp_draft_tokens: 0,
        request_class: class,
    })
    .unwrap();
    assert_eq!(served(verify.state), class);

    let stop = stop_message(11, 13, class);
    assert_eq!(stop.kind, WireMessageKind::Stop);
    assert_eq!(served(stop.state), class);

    let prefill = embedded_prefill_message(OpenAiPrefillChunk {
        seq_id: 0,
        pos_start: 0,
        prefill_token_count: 2,
        tokens: &[1, 2],
        request_id: 11,
        session_id: 13,
        request_class: class,
    })
    .unwrap();
    assert_eq!(served(prefill.state), class);

    let restore =
        embedded_prefix_cache_message(WireMessageKind::TryRestorePrefill, &[1, 2], 11, 13, class)
            .unwrap();
    assert_eq!(served(restore.state), class);

    let multimodal = multimodal_prefill_message(MultimodalPrefillArgs {
        request_id: 11,
        session_id: 13,
        prompt_token_count: 2,
        pos_start: 0,
        token_count: 2,
        tokens: vec![1, 2],
        positions: Vec::new(),
        sampling: None,
        final_chunk: false,
        request_class: class,
    })
    .unwrap();
    assert_eq!(served(multimodal.state), class);

    let fused = embedded_restore_prefill_decode_message(RestorePrefillDecodeMessageArgs {
        request_id: 11,
        session_id: 13,
        prompt_token_count: 3,
        pos_start: 2,
        decode_step: 0,
        prefix_tokens: &[1, 2],
        current: 3,
        sampling: None,
        chat_sampling_metadata: None,
        native_mtp_draft_tokens: 0,
        request_class: class,
    })
    .unwrap();
    assert_eq!(served(fused.state), class);
}
