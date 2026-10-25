use crate::LogitBias;

pub const MAX_SAMPLERS: usize = 16;
pub const MAX_DRY_SEQUENCE_BREAKERS: usize = 8;
pub const MAX_DRY_SEQUENCE_BREAKER_BYTES: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub version: u32,
    pub flags: u32,
    pub seed: u32,
    pub top_k: i32,
    pub penalty_last_n: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub logit_bias_count: u32,
    pub min_p: f32,
    pub typical_p: f32,
    pub top_nsigma: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub mirostat_mode: i32,
    pub mirostat_entropy: f32,
    pub mirostat_learning_rate: f32,
    pub sampler_count: u32,
    pub samplers: [u32; MAX_SAMPLERS],
    pub ignore_eos: u32,
    pub dry_sequence_breaker_count: u32,
    pub dry_sequence_breakers: [[u8; MAX_DRY_SEQUENCE_BREAKER_BYTES]; MAX_DRY_SEQUENCE_BREAKERS],
    pub logit_bias: [LogitBias; 256],
    pub reasoning_budget_tokens: i32,
    /// Alternatives recorded per token with [`SAMPLING_FLAG_LOGPROBS`], up to [`MAX_TOP_LOGPROBS`].
    pub top_logprobs: u32,
}

/// Mirrors `SKIPPY_SAMPLING_FLAG_LOGPROBS`: record each sampled token's log-probability.
pub const SAMPLING_FLAG_LOGPROBS: u32 = 1 << 2;

/// Mirrors `SKIPPY_MAX_TOP_LOGPROBS` (`skippy/logprobs.h`).
pub const MAX_TOP_LOGPROBS: usize = 20;

/// Mirrors `struct skippy_logprob_entry`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogprobEntry {
    pub token: i32,
    pub logprob: f32,
}

/// Mirrors `struct skippy_token_logprobs`: one sampled token, its log-probability
/// under the raw distribution and its most likely alternatives.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenLogprobs {
    pub token: i32,
    pub logprob: f32,
    pub top_count: u32,
    pub top: [LogprobEntry; MAX_TOP_LOGPROBS],
}

pub const NATIVE_MTP_MAX_DRAFT_TOKENS: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeMtpDraft {
    pub version: u32,
    pub available: bool,
    pub token_count: i32,
    pub token_ids: [i32; NATIVE_MTP_MAX_DRAFT_TOKENS],
    pub proposal_compute_us: i64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenSignal {
    pub entropy: f32,
    pub top_logprob: f32,
    pub second_logprob: f32,
    pub margin: f32,
    pub top_token: i32,
    pub second_token: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationSignalWindow {
    pub token_count: u32,
    pub mean_entropy: f32,
    pub max_entropy: f32,
    pub mean_margin: f32,
    pub min_margin: f32,
    pub high_entropy_count: u32,
    pub repetition_count: u32,
}
