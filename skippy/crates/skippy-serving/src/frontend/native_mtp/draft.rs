use skippy_protocol::binary::StageNativeMtpDraft;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpDraft {
    pub(in crate::frontend) tokens: Vec<i32>,
    pub(in crate::frontend) proposal_compute_us: i64,
}

impl NativeMtpDraft {
    pub(in crate::frontend) fn from_stage_draft(draft: StageNativeMtpDraft) -> Self {
        Self {
            tokens: draft.token_ids,
            proposal_compute_us: draft.proposal_compute_us.max(0),
        }
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
    fn converts_typed_stage_draft() {
        assert_eq!(
            NativeMtpDraft::from_stage_draft(StageNativeMtpDraft {
                token_ids: vec![12],
                proposal_compute_us: 34,
            }),
            NativeMtpDraft {
                tokens: vec![12],
                proposal_compute_us: 34,
            }
        );
    }

    #[test]
    fn clamps_negative_typed_proposal_time() {
        let draft = NativeMtpDraft::from_stage_draft(StageNativeMtpDraft {
            token_ids: vec![12],
            proposal_compute_us: -3,
        });

        assert_eq!(draft.proposal_compute_us, 0);
    }

    #[test]
    fn pending_draft_keeps_origin() {
        let pending = PendingNativeMtpDraft {
            tokens: vec![12, 13],
            origin: NativeMtpDraftOrigin::VerifyNext,
        };

        assert_eq!(pending.tokens, vec![12, 13]);
        assert_eq!(pending.origin, NativeMtpDraftOrigin::VerifyNext);
    }
}
