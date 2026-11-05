//! Ergonomic aliases for the generated sampling ABI types.

use crate::bindings;

pub const MAX_SAMPLERS: usize = bindings::skippy::SKIPPY_MAX_SAMPLERS as usize;
pub const MAX_DRY_SEQUENCE_BREAKERS: usize =
    bindings::skippy::SKIPPY_MAX_DRY_SEQUENCE_BREAKERS as usize;
pub const MAX_DRY_SEQUENCE_BREAKER_BYTES: usize =
    bindings::skippy::SKIPPY_MAX_DRY_SEQUENCE_BREAKER_BYTES as usize;

pub type SamplingConfig = bindings::skippy::skippy_sampling_config;

/// Record each sampled token's log-probability and top alternatives.
pub const SAMPLING_FLAG_LOGPROBS: u32 = bindings::skippy::SKIPPY_SAMPLING_FLAG_LOGPROBS;

/// Alternatives recorded per token with [`SAMPLING_FLAG_LOGPROBS`].
pub const MAX_TOP_LOGPROBS: usize = bindings::skippy::SKIPPY_MAX_TOP_LOGPROBS as usize;

pub type LogprobEntry = bindings::skippy::skippy_logprob_entry;
pub type TokenLogprobs = bindings::skippy::skippy_token_logprobs;

#[allow(clippy::derivable_impls)]
impl Default for LogprobEntry {
    fn default() -> Self {
        Self {
            token: 0,
            logprob: 0.0,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for TokenLogprobs {
    fn default() -> Self {
        Self {
            token: 0,
            logprob: 0.0,
            top_count: 0,
            top: [LogprobEntry::default(); MAX_TOP_LOGPROBS],
        }
    }
}

pub const NATIVE_MTP_MAX_DRAFT_TOKENS: usize =
    bindings::skippy::SKIPPY_NATIVE_MTP_MAX_DRAFT_TOKENS as usize;

pub type NativeMtpDraft = bindings::skippy::skippy_native_mtp_draft;

#[allow(clippy::derivable_impls)]
impl Default for NativeMtpDraft {
    fn default() -> Self {
        Self {
            version: 0,
            available: false,
            token_count: 0,
            token_ids: [0; NATIVE_MTP_MAX_DRAFT_TOKENS],
            proposal_compute_us: 0,
        }
    }
}

pub type TokenSignal = bindings::skippy::skippy_token_signal;
pub type GenerationSignalWindow = bindings::skippy::skippy_generation_signal_window;

#[allow(clippy::derivable_impls)]
impl Default for TokenSignal {
    fn default() -> Self {
        Self {
            entropy: 0.0,
            top_logprob: 0.0,
            second_logprob: 0.0,
            margin: 0.0,
            top_token: 0,
            second_token: 0,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for GenerationSignalWindow {
    fn default() -> Self {
        Self {
            token_count: 0,
            mean_entropy: 0.0,
            max_entropy: 0.0,
            mean_margin: 0.0,
            min_margin: 0.0,
            high_entropy_count: 0,
            repetition_count: 0,
        }
    }
}
