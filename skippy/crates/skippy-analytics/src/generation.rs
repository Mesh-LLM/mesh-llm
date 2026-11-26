//! The `generation` event: one served generation, from the frontend's own
//! measurement.
//!
//! skippy-serving measures every generation and hands the figures to one
//! process-wide observer, [`skippy_serving::generation_stats`]. [`install`]
//! makes this crate that observer. Only counts and timings cross that
//! boundary; prompt text, generated text and token IDs never do.
//!
//! Events are rate limited per process. An event that goes out after others
//! were dropped carries `weight`, the number of generations it stands for, so
//! totals stay right when summed by weight.
//!
//! The model is described by [`crate::models`]: a catalog name when the
//! loader registered one, otherwise `redacted` plus architecture and size.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use skippy_serving::generation_stats::{GenerationStats, GenerationStatsObserver, RequestShape};

use crate::event::{Event, Label, Properties};
use crate::models::{self, ModelIdentity};

/// `generation` events one process may send per minute. A busy server
/// generates many times a second; the distribution survives sampling, the
/// bill does not.
const EVENTS_PER_MINUTE: u32 = 30;

static LIMITER: RateLimiter = RateLimiter::new(EVENTS_PER_MINUTE);

/// Report `generation` for every generation in this process.
///
/// Installing is cheap when analytics is off: the observer returns before
/// building anything. Returns `false` if another observer is already set.
pub fn install() -> bool {
    skippy_serving::generation_stats::set_generation_stats_observer(Arc::new(AnalyticsObserver))
}

struct AnalyticsObserver;

impl GenerationStatsObserver for AnalyticsObserver {
    fn observe(&self, stats: &GenerationStats) {
        if !crate::is_reporting() {
            return;
        }
        let Some(weight) = LIMITER.admit(Instant::now()) else {
            return;
        };
        let model = models::lookup(&stats.model_id);
        let mut properties = request_properties(stats, &model, weight);
        if let Some(hardware) = crate::serving_hardware() {
            properties = hardware.apply(properties, request_backend(&model, loaded_backend));
        }
        crate::capture(Event::Generation, properties);
    }
}

/// The backend a `generation` reports. The device the model was
/// loaded onto decides it; only a load that selected no device takes the
/// runtime's, since the runtime then offloads to its own GPU devices.
fn request_backend(
    model: &ModelIdentity,
    runtime_backend: impl FnOnce() -> Option<&'static str>,
) -> Option<&'static str> {
    model.backend.or_else(runtime_backend)
}

/// The backend a model loaded with no selected device runs on: the first GPU
/// device the loaded runtime registers, which is where it offloads, or `cpu`
/// when it registers none. Read once, on the first reported request, when a
/// runtime is certainly loaded. `None` if the devices cannot be read.
fn loaded_backend() -> Option<&'static str> {
    static LOADED_BACKEND: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *LOADED_BACKEND.get_or_init(|| {
        let devices = skippy_runtime::backend_devices().ok()?;
        Some(
            devices
                .iter()
                .find_map(|device| crate::backend_of_device(&device.name))
                .unwrap_or("cpu"),
        )
    })
}

/// Throughput in tokens per second, or `None` when it would be noise.
fn tokens_per_second(tokens: u64, ms: f64) -> Option<f64> {
    (tokens > 0 && ms > 0.0).then(|| round2(tokens as f64 * 1000.0 / ms))
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| round4(numerator as f64 / denominator as f64))
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// The coarse kind of request: `agentic` when tools are in play (offered, or
/// results in the history), `chat` for other message requests, `completion`
/// for a raw prompt.
fn request_kind(shape: &RequestShape) -> &'static str {
    if shape.tools_offered || shape.tool_results {
        "agentic"
    } else if shape.chat {
        "chat"
    } else {
        "completion"
    }
}

/// Context length in coarse bands, so acceptance and speed can be compared
/// at like context without an exact-length breakdown.
fn context_band(tokens: u64) -> &'static str {
    match tokens {
        0..2_048 => "lt2k",
        2_048..8_192 => "2k-8k",
        8_192..32_768 => "8k-32k",
        32_768..131_072 => "32k-128k",
        _ => "128k+",
    }
}

fn with_request_shape(properties: Properties, shape: &RequestShape) -> Properties {
    properties
        .with("request_kind", request_kind(shape))
        .with("tools_offered", shape.tools_offered)
        .with("tool_results", shape.tool_results)
        .with("system_prompt", shape.system_prompt)
        .with(
            "user_turns",
            crate::bucket_count(u64::from(shape.user_turns)),
        )
}

/// The speculation plan a request ran under: why it stood down, and the caps
/// that bound how much it could draft.
fn with_speculation_plan(mut properties: Properties, stats: &GenerationStats) -> Properties {
    if let Some(reason) = stats.speculation_stand_down {
        properties = properties.with("spec_stand_down", reason.as_str());
    }
    if let Some(cap) = stats.draft_token_cap {
        properties = properties.with("spec_draft_cap", cap);
    }
    properties
        .with("spec_window_max", stats.verify_window_max_tokens)
        .with("spec_runahead_tokens", stats.runahead_budget_tokens)
}

fn request_properties(stats: &GenerationStats, model: &ModelIdentity, weight: u64) -> Properties {
    let prompt = u64::from(stats.prompt_tokens);
    let completion = u64::from(stats.completion_tokens);
    let context = prompt + completion;
    let proposed = stats.draft_proposed_tokens + stats.mtp_proposed_tokens;
    let accepted = stats.draft_accepted_tokens + stats.mtp_accepted_tokens;
    let windows = stats.draft_windows + stats.mtp_verifications;

    let mut properties = model
        .apply(Properties::new())
        .with("weight", weight)
        .with("serving", if stats.split { "split" } else { "local" })
        .with("concurrency_limit", stats.concurrency_limit)
        .with("context_size", stats.context_size)
        .with("prompt_tokens", prompt)
        .with("completion_tokens", completion)
        .with("context_tokens", context)
        .with("context_band", context_band(context))
        .with(
            "cached_prompt_tokens",
            u64::from(stats.cached_prompt_tokens),
        )
        .with("prefill_tokens", u64::from(stats.prefill_tokens))
        .with("cache_hit_kind", stats.cache_hit_kind)
        .with("queue_wait_ms", round2(stats.queue_wait_ms))
        .with("restore_ms", round2(stats.restore_ms))
        .with("prefill_ms", round2(stats.prefill_ms))
        .with("decode_ms", round2(stats.decode_ms))
        .with("total_ms", round2(stats.total_ms))
        .with(
            "ttft_ms",
            round2(stats.queue_wait_ms + stats.restore_ms + stats.prefill_ms),
        )
        .with("finish_reason", stats.finish_reason)
        .with(
            "spec_strategy",
            Label::sanitize_or_redact(&stats.speculative_strategy),
        )
        .with("spec_allowed", stats.speculation_allowed)
        .with("spec_proposed_tokens", proposed)
        .with("spec_accepted_tokens", accepted)
        .with("spec_windows", windows)
        .with("spec_draft_proposed_tokens", stats.draft_proposed_tokens)
        .with("spec_draft_accepted_tokens", stats.draft_accepted_tokens)
        .with("spec_mtp_proposed_tokens", stats.mtp_proposed_tokens)
        .with("spec_mtp_accepted_tokens", stats.mtp_accepted_tokens);
    properties = with_request_shape(properties, &stats.request_shape);
    properties = with_speculation_plan(properties, stats);

    if let Some(fill) = ratio(context, stats.context_size as u64) {
        properties = properties.with("context_fill", fill);
    }
    if let Some(rate) = tokens_per_second(u64::from(stats.prefill_tokens), stats.prefill_ms) {
        properties = properties.with("prefill_tps", rate);
    }
    if let Some(rate) = tokens_per_second(completion, stats.decode_ms) {
        properties = properties.with("decode_tps", rate);
    }
    if let Some(rate) = ratio(accepted, proposed) {
        properties = properties.with("spec_acceptance", rate);
    }
    if windows > 0 {
        // Tokens committed per verify step: the figure speculation exists
        // to raise above one.
        properties = properties.with(
            "spec_accepted_per_window",
            round2(accepted as f64 / windows as f64),
        );
    }
    if let Some(lanes) = stats.downstream_lanes {
        properties = properties.with("downstream_lanes", lanes);
    }
    if let Some(codec) = stats.activation_codec {
        properties = properties
            .with("activation_codec", Label::sanitize_or_redact(codec))
            .with("activation_prefill_bytes", stats.activation_prefill_bytes)
            .with("activation_decode_bytes", stats.activation_decode_bytes);
    }
    properties
}

/// A fixed-window limiter that remembers how much it dropped.
struct RateLimiter {
    per_window: u32,
    state: Mutex<LimiterState>,
}

struct LimiterState {
    window_start: Option<Instant>,
    sent: u32,
    skipped: u64,
}

impl RateLimiter {
    const WINDOW: Duration = Duration::from_secs(60);

    const fn new(per_window: u32) -> Self {
        Self {
            per_window,
            state: Mutex::new(LimiterState {
                window_start: None,
                sent: 0,
                skipped: 0,
            }),
        }
    }

    /// `Some(weight)` when this request may be reported, where `weight` is
    /// itself plus every request dropped since the last one that was.
    fn admit(&self, now: Instant) -> Option<u64> {
        let mut state = self.state.lock().ok()?;
        let expired = state
            .window_start
            .is_none_or(|start| now.duration_since(start) >= Self::WINDOW);
        if expired {
            state.window_start = Some(now);
            state.sent = 0;
        }
        if state.sent >= self.per_window {
            state.skipped += 1;
            return None;
        }
        state.sent += 1;
        let weight = state.skipped + 1;
        state.skipped = 0;
        Some(weight)
    }
}

#[cfg(test)]
#[path = "generation/tests.rs"]
mod tests;
