use super::*;

#[test]
fn multimodal_final_prefill_message_requests_downstream_prediction() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };

    let message = multimodal_prefill_message(
        WireActivationDType::F16,
        MultimodalPrefillArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 17,
            pos_start: 0,
            token_count: 17,
            positions: Vec::new(),
            sampling: Some(sampling.clone()),
            final_chunk: true,
        },
    )
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::PrefillFinalEmbd);
    assert!(message.kind.requires_predicted_reply());
    assert_eq!(message.token_count, 17);
    assert_eq!(message.state.current_token, LLAMA_TOKEN_NULL);
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

    let message = embedded_restore_prefill_decode_message(
        WireActivationDType::F16,
        RestorePrefillDecodeMessageArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 4,
            pos_start: 3,
            decode_step: 0,
            prefix_tokens: &[101, 102, 103],
            current: 104,
            sampling: Some(sampling.clone()),
            chat_sampling_metadata: Some(metadata),
        },
    )
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::TryRestorePrefillDecode);
    assert_eq!(message.tokens, vec![101, 102, 103, 104]);
    assert_eq!(message.sampling, Some(sampling.clone()));
    assert_eq!(message.chat_sampling_metadata.as_deref(), Some(metadata));

    let mut encoded = Vec::new();
    write_stage_message(&mut encoded, &message, WireActivationDType::F16).unwrap();
    let decoded = skippy_protocol::binary::read_stage_message(Cursor::new(encoded), 2816).unwrap();
    assert_eq!(decoded.kind, WireMessageKind::TryRestorePrefillDecode);
    assert_eq!(decoded.tokens, vec![101, 102, 103, 104]);
    assert_eq!(decoded.sampling, Some(sampling));
    assert_eq!(decoded.chat_sampling_metadata.as_deref(), Some(metadata));
}

#[test]
fn reusable_decode_message_updates_hot_path_fields() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };
    let mut message = ReusableDecodeMessage::new(
        WireActivationDType::F16,
        ReusableDecodeMessageArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 4,
            base_pos_start: 4,
            sampling: Some(sampling.clone()),
            sideband_capacity: 4,
        },
    )
    .unwrap();

    let first = message.update(0, 104).unwrap();
    assert_eq!(first.kind, WireMessageKind::DecodeEmbd);
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
