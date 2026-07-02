#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpDraft {
    pub(in crate::frontend) tokens: Vec<i32>,
    pub(in crate::frontend) proposal_compute_us: i64,
}

impl NativeMtpDraft {
    pub(in crate::frontend) fn from_prediction_tokens(tokens: &[i32]) -> Option<Self> {
        Self::from_sideband(tokens, 1)
    }

    pub(in crate::frontend) fn from_verify_prediction_tokens(
        tokens: &[i32],
        verified_token_count: usize,
    ) -> Option<Self> {
        Self::from_sideband(tokens, verified_token_count)
    }

    fn from_sideband(tokens: &[i32], offset: usize) -> Option<Self> {
        let token_count = usize::try_from(*tokens.get(offset)?).ok()?;
        if token_count == 0 {
            return None;
        }
        let start = offset.saturating_add(1);
        let end = start.checked_add(token_count)?;
        let draft_tokens = tokens.get(start..end)?.to_vec();
        let proposal_compute_us = tokens.get(end).copied().unwrap_or_default();
        Some(Self {
            tokens: draft_tokens,
            proposal_compute_us: i64::from(proposal_compute_us.max(0)),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct PendingNativeMtpDraft {
    pub(in crate::frontend) tokens: Vec<i32>,
    pub(in crate::frontend) origin: NativeMtpDraftOrigin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) enum NativeMtpDraftOrigin {
    InitialSerial,
    SerialAfterGap,
    VerifyNext,
}

impl NativeMtpDraftOrigin {
    pub(in crate::frontend) fn label(self) -> &'static str {
        match self {
            Self::InitialSerial => "initial_serial",
            Self::SerialAfterGap => "serial_after_gap",
            Self::VerifyNext => "verify_next",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prediction_token_sideband() {
        assert_eq!(
            NativeMtpDraft::from_prediction_tokens(&[11, 1, 12, 34]),
            Some(NativeMtpDraft {
                tokens: vec![12],
                proposal_compute_us: 34,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_prediction_tokens(&[11, 2, 34, 35, 567]),
            Some(NativeMtpDraft {
                tokens: vec![34, 35],
                proposal_compute_us: 567,
            })
        );
        assert_eq!(NativeMtpDraft::from_prediction_tokens(&[11]), None);
    }

    #[test]
    fn parses_verify_prediction_token_sideband_after_verified_tokens() {
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 1, 12, 34], 2),
            Some(NativeMtpDraft {
                tokens: vec![12],
                proposal_compute_us: 34,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 1, 12, -3], 2),
            Some(NativeMtpDraft {
                tokens: vec![12],
                proposal_compute_us: 0,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 2, 12, 13, 567], 2),
            Some(NativeMtpDraft {
                tokens: vec![12, 13],
                proposal_compute_us: 567,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11], 2),
            None
        );
    }

    #[test]
    fn pending_draft_keeps_origin_label() {
        let pending = PendingNativeMtpDraft {
            tokens: vec![12, 13],
            origin: NativeMtpDraftOrigin::VerifyNext,
        };

        assert_eq!(pending.tokens, vec![12, 13]);
        assert_eq!(pending.origin.label(), "verify_next");
    }
}
