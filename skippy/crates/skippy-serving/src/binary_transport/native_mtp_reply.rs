//! Native MTP drafts on a final stage's reply: the sideband the decode ABI
//! returns, the depth stage 0 asks for, and which verify drafts it may use.

use anyhow::{Result, anyhow, bail};
use skippy_protocol::binary::{StageNativeMtpDraft, StageWireMessage, WireMessageKind};
use skippy_runtime::NativeMtpDraft;

use super::stage_execution::BinaryStageExecutionOptions;

pub(in crate::binary_transport) fn native_mtp_prediction_tokens(
    predicted: i32,
    draft: Option<NativeMtpDraft>,
) -> Vec<i32> {
    let Some(draft) = draft else {
        return vec![predicted];
    };
    let token_count = i32::try_from(draft.token_ids.len()).unwrap_or(i32::MAX);
    let mut tokens = Vec::with_capacity(draft.token_ids.len() + 3);
    tokens.push(predicted);
    tokens.push(token_count);
    tokens.extend(draft.token_ids);
    tokens.push(draft.proposal_compute_us.clamp(0, i64::from(i32::MAX)) as i32);
    tokens
}

/// Converts the temporary llama-stage sideband into the typed stage reply field.
///
/// The C ABI still returns the proposal as a trailer. The network boundary is
/// authoritative: consumers receive only target predictions plus a separate
/// native-MTP draft. A malformed trailer is rejected instead of being exposed
/// as target-model output.
pub(in crate::binary_transport) fn split_native_mtp_reply(
    message: &StageWireMessage,
    prediction_tokens: &mut Vec<i32>,
) -> Result<Option<StageNativeMtpDraft>> {
    let sideband_offset = match message.kind {
        WireMessageKind::DecodeEmbd
        | WireMessageKind::DecodeReadout
        | WireMessageKind::DecodeLightCtx
        | WireMessageKind::DecodeReplayEmbd
        | WireMessageKind::DecodeReplayFinalEmbd => 1,
        _ => return Ok(None),
    };
    if prediction_tokens.len() <= sideband_offset {
        return Ok(None);
    }
    let draft_token_count = usize::try_from(prediction_tokens[sideband_offset])
        .map_err(|_| anyhow!("negative native MTP draft token count"))?;
    let draft_start = sideband_offset + 1;
    let draft_end = draft_start
        .checked_add(draft_token_count)
        .ok_or_else(|| anyhow!("native MTP draft token count overflow"))?;
    let compute_index = draft_end;
    if prediction_tokens.len() != compute_index + 1 {
        bail!(
            "malformed native MTP sideband: expected {} values, got {}",
            compute_index + 1,
            prediction_tokens.len()
        );
    }
    let proposal_compute_us = i64::from(prediction_tokens[compute_index].max(0));
    let token_ids = prediction_tokens[draft_start..draft_end].to_vec();
    prediction_tokens.truncate(sideband_offset);
    Ok((!token_ids.is_empty()).then_some(StageNativeMtpDraft {
        token_ids,
        proposal_compute_us,
        follows_correction: false,
    }))
}

/// Stage 0 requests a draft depth on each decode and verify message; older
/// senders leave it unset and get this stage's configured depth.
pub(in crate::binary_transport) fn native_mtp_draft_tokens(
    message: &StageWireMessage,
    options: BinaryStageExecutionOptions,
) -> usize {
    message
        .state
        .native_mtp_draft_tokens()
        .unwrap_or(options.native_mtp_max_tokens)
}

/// The runtime drafts after the first verify row that rejected its proposal
/// (or after the last row when none did). A draft from a rejected window is
/// only meaningful to a stage 0 that commits the correction and asked for it.
pub(in crate::binary_transport) fn verify_reply_native_mtp_draft(
    draft: Option<StageNativeMtpDraft>,
    window_tokens: &[i32],
    predicted_tokens: &[i32],
    accepts_draft_after_correction: bool,
) -> Option<StageNativeMtpDraft> {
    let mut draft = draft?;
    let rejected = predicted_tokens.len() < window_tokens.len()
        || predicted_tokens
            .iter()
            .zip(window_tokens.iter().skip(1))
            .any(|(predicted, proposed)| predicted != proposed);
    if rejected {
        if !accepts_draft_after_correction {
            return None;
        }
        draft.follows_correction = true;
    }
    Some(draft)
}

pub(in crate::binary_transport) fn stage_native_mtp_draft(
    draft: NativeMtpDraft,
) -> StageNativeMtpDraft {
    StageNativeMtpDraft {
        token_ids: draft.token_ids,
        proposal_compute_us: draft.proposal_compute_us,
        follows_correction: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::StageStateHeader;

    fn test_message(kind: WireMessageKind, token_count: i32) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count,
            state: StageStateHeader::new(kind),
            request_id: 11,
            session_id: 13,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn native_mtp_sideband_is_removed_from_decode_predictions() {
        let message = test_message(WireMessageKind::DecodeEmbd, 1);
        let mut predictions = vec![11, 2, 14, 15, 123];

        let draft = split_native_mtp_reply(&message, &mut predictions).unwrap();

        assert_eq!(predictions, vec![11]);
        assert_eq!(
            draft,
            Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![14, 15],
                proposal_compute_us: 123,
                follows_correction: false,
            })
        );
    }
    #[test]
    fn native_mtp_sideband_is_not_read_from_verify_predictions() {
        let mut message = test_message(WireMessageKind::VerifyWindow, 3);
        message.tokens = vec![10, 11, 12];
        let mut predictions = vec![11, 12, 13, 2, 14, 15, 123];

        let draft = split_native_mtp_reply(&message, &mut predictions).unwrap();

        assert_eq!(predictions, vec![11, 12, 13, 2, 14, 15, 123]);
        assert_eq!(draft, None);
    }
    #[test]
    fn malformed_native_mtp_sideband_is_rejected() {
        let message = test_message(WireMessageKind::DecodeEmbd, 1);
        let mut predictions = vec![11, 2, 12];

        let error = split_native_mtp_reply(&message, &mut predictions).unwrap_err();

        assert!(error.to_string().contains("malformed native MTP sideband"));
    }
    #[test]
    fn verify_reply_keeps_rejected_window_drafts_only_on_request() {
        let draft = || {
            Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![5, 6],
                proposal_compute_us: 1,
                follows_correction: false,
            })
        };
        // Every proposal held: the draft follows the last prediction.
        let accepted =
            verify_reply_native_mtp_draft(draft(), &[1, 2, 3], &[2, 3, 4], false).unwrap();
        assert!(!accepted.follows_correction);
        // Row 1 rejected proposal 3 in favour of 9.
        assert_eq!(
            verify_reply_native_mtp_draft(draft(), &[1, 2, 3], &[2, 9, 4], false),
            None
        );
        let corrected =
            verify_reply_native_mtp_draft(draft(), &[1, 2, 3], &[2, 9, 4], true).unwrap();
        assert!(corrected.follows_correction);
        // A stateful sampler stops at the first rejected row.
        let stopped = verify_reply_native_mtp_draft(draft(), &[1, 2, 3], &[7], true).unwrap();
        assert!(stopped.follows_correction);
        assert_eq!(
            verify_reply_native_mtp_draft(None, &[1, 2], &[2, 3], true),
            None
        );
    }
}
