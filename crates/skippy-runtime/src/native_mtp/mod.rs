use skippy_ffi::NativeMtpDraft as RawNativeMtpDraft;

const NATIVE_MTP_DRAFT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeMtpDraft {
    pub token_ids: Vec<i32>,
    pub proposal_compute_us: i64,
}

impl NativeMtpDraft {
    pub(crate) fn from_raw(raw: RawNativeMtpDraft) -> Option<Self> {
        if !raw.available || raw.version != NATIVE_MTP_DRAFT_VERSION {
            return None;
        }
        let token_count = usize::try_from(raw.token_count)
            .ok()?
            .min(skippy_ffi::NATIVE_MTP_MAX_DRAFT_TOKENS);
        if token_count == 0 {
            return None;
        }
        Some(Self {
            token_ids: raw.token_ids[..token_count].to_vec(),
            proposal_compute_us: raw.proposal_compute_us,
        })
    }
}
