use super::*;
use std::io::Cursor;

#[test]
fn writer_rejects_total_frame_over_budget_before_writing() {
    let mut message = StageWireMessage::stop_with_identity(7, 11);
    message.kind = WireMessageKind::ConfigureGeneration;
    message.state = StageStateHeader::new(message.kind);
    message.chat_sampling_metadata = Some("x".repeat(crate::MAX_STAGE_FRAME_BYTES));
    let mut bytes = Vec::new();

    let result = write_stage_message(&mut bytes, &message);

    assert!(result.is_err());
    assert!(bytes.is_empty());
}

#[test]
fn sampling_uses_actual_bytes_at_the_frame_boundary() {
    let mut message = StageWireMessage::stop_with_identity(7, 11);
    message.kind = WireMessageKind::ConfigureGeneration;
    message.state = StageStateHeader::new(message.kind);
    message.sampling = Some(StageSamplingConfig::default());
    message.chat_sampling_metadata = Some("x".repeat(MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES));
    let mut bytes = Vec::new();
    write_stage_message(&mut bytes, &message).unwrap();
    assert_eq!(bytes.len(), message.estimated_wire_bytes());

    // A budget of exactly the frame's size admits it.
    let budget = bytes.len();
    let decoded = read_stage_message_for_codec_policy_with_frame_budget(
        Cursor::new(bytes),
        1,
        crate::StageActivationCodec::RawF32V1,
        crate::StageActivationCodecPolicy::Fixed,
        budget,
    )
    .unwrap();

    assert_eq!(
        decoded.chat_sampling_metadata,
        message.chat_sampling_metadata
    );
}

#[test]
fn sampling_lengths_are_rejected_before_their_payload_is_read() {
    let message =
        StageWireMessage::configure_generation(1, 2, 1, Some(StageSamplingConfig::default()), None);
    let mut bytes = Vec::new();
    write_stage_message(&mut bytes, &message).unwrap();
    // No bias entries: the first dry-breaker length follows 40 + 4 + 32 + 4 bytes.
    let length_offset = STAGE_WIRE_FIXED_HEADER_BYTES + 80;
    bytes[length_offset..length_offset + 4].copy_from_slice(&64_u32.to_le_bytes());
    bytes.truncate(length_offset + 4);

    let result = read_stage_message_for_codec_policy_with_frame_budget(
        Cursor::new(bytes),
        1,
        crate::StageActivationCodec::RawF32V1,
        crate::StageActivationCodecPolicy::Fixed,
        STAGE_WIRE_FIXED_HEADER_BYTES + STAGE_SAMPLING_CONFIG_BASE_BYTES + 16,
    );

    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
}
