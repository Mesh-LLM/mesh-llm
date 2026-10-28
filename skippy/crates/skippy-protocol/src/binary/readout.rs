//! Bounded wire payloads for unsampled causal readouts.

use super::{invalid_data, invalid_input};
use std::io::{self, Read, Write};

const MAX_SELECTED_TOKENS: usize = 256;

/// Write a bounded, ordered list of selected token IDs.
pub(super) fn write_tokens(mut writer: impl Write, tokens: &[i32]) -> io::Result<()> {
    if tokens.len() > MAX_SELECTED_TOKENS || tokens.iter().any(|token| *token < 0) {
        return Err(invalid_input("invalid selected-token readout"));
    }
    writer.write_all(&(tokens.len() as u32).to_le_bytes())?;
    for token in tokens {
        writer.write_all(&token.to_le_bytes())?;
    }
    Ok(())
}

/// Read a bounded element count before allocating or consuming its payload.
fn read_count(mut reader: impl Read) -> io::Result<usize> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    let count = u32::from_le_bytes(bytes) as usize;
    if count > MAX_SELECTED_TOKENS {
        return Err(invalid_data("invalid readout count"));
    }
    Ok(count)
}

/// Decode selected token IDs while preserving caller order.
pub(super) fn read_tokens(mut reader: impl Read) -> io::Result<Vec<i32>> {
    let count = read_count(&mut reader)?;
    let mut tokens = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        let token = i32::from_le_bytes(bytes);
        if token < 0 {
            return Err(invalid_data("negative readout token"));
        }
        tokens.push(token);
    }
    Ok(tokens)
}

/// Serialize a bounded vector of valid conditional log probabilities.
pub(super) fn write_scores(mut writer: impl Write, scores: &[f32]) -> io::Result<()> {
    if scores.len() > MAX_SELECTED_TOKENS
        || scores
            .iter()
            .any(|score| !score.is_finite() || *score > 0.0)
    {
        return Err(invalid_input("invalid readout log probabilities"));
    }
    writer.write_all(&(scores.len() as u32).to_le_bytes())?;
    for score in scores {
        writer.write_all(&score.to_le_bytes())?;
    }
    Ok(())
}

/// Decode bounded scores and reject invalid probability evidence.
pub(super) fn read_scores(mut reader: impl Read) -> io::Result<Vec<f32>> {
    let count = read_count(&mut reader)?;
    let mut scores = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        let score = f32::from_le_bytes(bytes);
        if !score.is_finite() || score > 0.0 {
            return Err(invalid_data("invalid readout log probability"));
        }
        scores.push(score);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::{
        STAGE_STATE_VERSION, StageReply, StageStateHeader, StageWireMessage, WireMessageKind,
        WireReplyKind, read_stage_message, recv_reply, send_reply_message, write_stage_message,
    };

    /// Build a current-generation readout request for transport tests.
    fn message(selected: Vec<i32>) -> StageWireMessage {
        let mut message = StageWireMessage::stop_with_identity(7, 11);
        message.kind = WireMessageKind::PrefillReadout;
        message.state = StageStateHeader::new(message.kind);
        message.state.prompt_token_count = 2;
        message.tokens = vec![1, 42];
        message.token_count = 2;
        message.readout_tokens = selected;
        message
    }

    /// Readouts preserve identity and selection order, including empty prefill selections.
    #[test]
    fn readout_roundtrip_preserves_identity_and_selected_order_including_prefill_only() {
        for selected in [vec![], vec![42, 8, 1]] {
            let request = message(selected.clone());
            let mut bytes = Vec::new();
            write_stage_message(&mut bytes, &request).unwrap();
            let parsed = read_stage_message(bytes.as_slice(), 0).unwrap();
            assert_eq!(parsed.kind, WireMessageKind::PrefillReadout);
            assert_eq!(parsed.readout_tokens, selected);
            assert_eq!(parsed.request_id, 7);
            assert_eq!(parsed.session_id, 11);
            assert_eq!(parsed.tokens, vec![1, 42]);
            assert!(parsed.state.matches_kind(parsed.kind));
            assert!(!parsed.kind.requires_predicted_reply());
        }
    }

    /// Score replies remain distinguishable from sampled-token replies.
    #[test]
    fn readout_replies_cannot_be_mistaken_for_sampled_tokens() {
        for scores in [vec![], vec![-1.0, -2.0]] {
            let mut reply = StageReply {
                kind: WireReplyKind::TokenScores,
                predicted: -1,
                predicted_tokens: vec![],
                native_mtp_draft: None,
                window: Default::default(),
                stats: Default::default(),
                log_probabilities: scores.clone(),
            };
            let mut bytes = Vec::new();
            send_reply_message(&mut bytes, &reply).unwrap();
            assert_eq!(
                recv_reply(bytes.as_slice()).unwrap().log_probabilities,
                scores
            );
            reply.predicted = 42;
            assert!(send_reply_message(Vec::new(), &reply).is_err());
        }
    }

    /// Reject obsolete stage generations and selections on incompatible message kinds.
    #[test]
    fn rejects_previous_stage_version_and_misplaced_selection() {
        let mut request = message(vec![1]);
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &request).unwrap();
        // Five i32 fields precede the version in the fixed wire prefix.
        bytes[20..24].copy_from_slice(&(STAGE_STATE_VERSION - 1).to_le_bytes());
        assert!(
            read_stage_message(bytes.as_slice(), 0)
                .unwrap_err()
                .to_string()
                .contains("version")
        );
        request.kind = WireMessageKind::PrefillEmbd;
        request.state = StageStateHeader::new(request.kind);
        assert!(write_stage_message(Vec::new(), &request).is_err());
    }

    /// Readout scores retain selection order and reject invalid numeric evidence.
    #[test]
    fn preserves_selected_order_and_rejects_invalid_evidence() {
        let mut bytes = Vec::new();
        write_tokens(&mut bytes, &[41, 3, 41]).unwrap();
        assert_eq!(read_tokens(bytes.as_slice()).unwrap(), vec![41, 3, 41]);
        bytes.clear();
        write_scores(&mut bytes, &[-0.1, -90.0]).unwrap();
        assert_eq!(read_scores(bytes.as_slice()).unwrap(), vec![-0.1, -90.0]);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0] {
            assert!(write_scores(Vec::new(), &[value]).is_err());
            let bytes = [1u32.to_le_bytes(), value.to_le_bytes()].concat();
            assert!(read_scores(bytes.as_slice()).is_err());
        }
    }

    /// Oversized counts fail before payload allocation or reads.
    #[test]
    fn rejects_oversized_counts_before_reading_or_allocating_payload() {
        for count in [257u32, u32::MAX] {
            assert!(read_tokens(count.to_le_bytes().as_slice()).is_err());
            assert!(read_scores(count.to_le_bytes().as_slice()).is_err());
        }
        assert!(read_tokens(&[1u8, 0, 0, 0][..]).is_err());
        assert!(write_tokens(Vec::new(), &[-1]).is_err());
    }
}
