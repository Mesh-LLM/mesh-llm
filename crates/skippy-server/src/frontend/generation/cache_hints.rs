use crate::frontend::NativeMtpStats;
use crate::frontend::util::now_unix_millis;
use crate::frontend::util::stable_wire_id;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::CompletionRequest;
use skippy_protocol::binary::StageReplyStats;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

pub(in crate::frontend) static OPENAI_GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Sentinel meaning "no caller-specified max completion length; let the
/// request consume the entire remaining context window when the client
/// also omits max_tokens".
///
/// This is opt-in and should only be wired in by callers that have made
/// a deliberate decision to allow unbounded chat completions. The
/// embedded mesh-llm wiring uses [`DEFAULT_EMBEDDED_MAX_TOKENS`] instead.
pub const CONTEXT_BUDGET_MAX_TOKENS: u32 = u32::MAX;

/// Default max completion tokens for embedded mesh-llm chat serving when
/// the client omits max_tokens. Bounded so that an adversarial or
/// non-terminating generation cannot run for the full context window.
/// Clients can still request more by sending max_tokens explicitly, up
/// to the remaining context budget.
///
/// When the configured context window is smaller than this value, the
/// request is silently clamped to whatever remaining budget exists
/// rather than rejected — see [`GenerationTokenLimit::resolve`].
pub const DEFAULT_EMBEDDED_MAX_TOKENS: u32 = 4096;
pub(in crate::frontend) const GENERATION_ADMISSION_TIMEOUT: Duration = Duration::from_secs(10);
pub(in crate::frontend) const GENERATION_RETRY_AFTER_SECS: u64 = 1;
pub(in crate::frontend) const MAX_EXACT_REPLAY_TOKENS: usize = 8;

#[derive(Clone)]
pub(in crate::frontend) struct OpenAiGenerationIds {
    pub(in crate::frontend) session_label: String,
    pub(in crate::frontend) session_id: u64,
    pub(in crate::frontend) request_id: u64,
    pub(in crate::frontend) cache: OpenAiCacheHints,
}

impl OpenAiGenerationIds {
    pub(in crate::frontend) fn new(cache: OpenAiCacheHints) -> Self {
        let sequence = OPENAI_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let session_label = format!("openai-session-{}-{sequence}", now_unix_millis());
        Self {
            session_id: stable_wire_id(&[session_label.as_bytes()]),
            request_id: stable_wire_id(&[session_label.as_bytes(), b"request"]),
            session_label,
            cache,
        }
    }

    pub(in crate::frontend) fn session_id_string(&self) -> String {
        self.session_id.to_string()
    }

    pub(in crate::frontend) fn request_id_string(&self) -> String {
        self.request_id.to_string()
    }
}

#[derive(Clone, Default)]
pub(in crate::frontend) struct OpenAiCacheHints {
    pub(in crate::frontend) prompt_cache_key: Option<String>,
    pub(in crate::frontend) prompt_cache_retention: Option<String>,
}

impl OpenAiCacheHints {
    pub(in crate::frontend) fn from_chat_request(request: &ChatCompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    pub(in crate::frontend) fn from_completion_request(request: &CompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    pub(in crate::frontend) fn namespace(&self) -> Option<String> {
        self.prompt_cache_key
            .as_ref()
            .map(|key| format!("openai:prompt_cache_key:{key}"))
    }
}

pub(in crate::frontend) fn prompt_cache_retention_label(
    retention: openai_frontend::PromptCacheRetention,
) -> &'static str {
    match retention {
        openai_frontend::PromptCacheRetention::InMemory => "in_memory",
        openai_frontend::PromptCacheRetention::TwentyFourHours => "24h",
    }
}

pub(in crate::frontend) struct GenerationCacheStats {
    pub(in crate::frontend) status: &'static str,
    pub(in crate::frontend) cached_prompt_tokens: u32,
    pub(in crate::frontend) matched_prefix_tokens: u32,
    pub(in crate::frontend) suffix_prefill_tokens: u32,
    pub(in crate::frontend) hit_kind: Option<&'static str>,
    pub(in crate::frontend) native_mtp_stats: NativeMtpStats,
    pub(in crate::frontend) native_mtp_decode_telemetry:
        Option<crate::frontend::native_mtp::NativeMtpDecodeTelemetry>,
    pub(in crate::frontend) verify_window_pipeline_stats:
        Option<crate::frontend::decode_scheduler::VerifyWindowPipelineStats>,
    pub(in crate::frontend) speculative_stats:
        Option<crate::frontend::speculative::OpenAiSpeculativeStats>,
    pub(in crate::frontend) prompt_ms: f64,
    pub(in crate::frontend) predicted_ms: f64,
}

impl Default for GenerationCacheStats {
    fn default() -> Self {
        Self {
            status: "disabled",
            cached_prompt_tokens: 0,
            matched_prefix_tokens: 0,
            suffix_prefill_tokens: 0,
            hit_kind: None,
            native_mtp_stats: NativeMtpStats::default(),
            native_mtp_decode_telemetry: None,
            verify_window_pipeline_stats: None,
            speculative_stats: None,
            prompt_ms: 0.0,
            predicted_ms: 0.0,
        }
    }
}

pub(in crate::frontend) struct ChainPrefixRestore {
    pub(in crate::frontend) restored_tokens: usize,
    pub(in crate::frontend) stats: StageReplyStats,
}
