//! Soft peer-advertised model throughput hints.
//!
//! These are the dependency-neutral throughput-advertisement contracts used by
//! peer gossip. The routing metric collection that produces these hints stays
//! in `mesh-llm-host-runtime` (`network::metrics`); this module only owns the
//! wire-shape data type, its sanitization, and the bounds that keep gossip
//! deterministic.

use serde::Serialize;
use std::collections::HashSet;

pub const THROUGHPUT_SCALE_MILLI: u64 = 1000;
pub const MAX_ADVERTISED_MODEL_THROUGHPUT_HINTS: usize = 64;
pub const MAX_ADVERTISED_MODEL_NAME_BYTES: usize = 256;
pub const MAX_ADVERTISED_TPS_MILLI: u64 = 100_000 * THROUGHPUT_SCALE_MILLI;
pub const MAX_ADVERTISED_THROUGHPUT_SAMPLES: u64 = 256;
pub const MAX_ADVERTISED_STAGE_US_PER_LAYER: u64 = 10_000_000;
pub const MAX_ADVERTISED_STAGE_TIMING_AGE_MS: u64 = 30 * 60 * 1_000;

/// Soft peer-advertised model performance hint.
///
/// Values are fixed-point milli tokens/second to keep gossip deterministic and
/// avoid protobuf floating-point edge cases. Staged runtimes can additionally
/// attach observed steady-decode work normalized per loaded layer; placement
/// uses that as a measured floor on the analytical stage model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelThroughputHint {
    pub model_name: String,
    pub avg_tokens_per_second_milli: u64,
    pub throughput_samples: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_stage_us_per_layer: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_timing_samples: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_timing_age_ms: Option<u64>,
}

pub fn sanitize_model_throughput_hints<I>(hints: I) -> Vec<ModelThroughputHint>
where
    I: IntoIterator<Item = ModelThroughputHint>,
{
    let mut seen = HashSet::new();
    let mut sanitized = Vec::new();
    for mut hint in hints {
        hint.model_name = hint.model_name.trim().to_string();
        let throughput_valid = hint.avg_tokens_per_second_milli > 0 && hint.throughput_samples > 0;
        let stage_timing_valid = hint
            .observed_stage_us_per_layer
            .is_some_and(|value| value > 0)
            && hint.stage_timing_samples.is_some_and(|samples| samples > 0)
            && hint
                .stage_timing_age_ms
                .is_some_and(|age| age <= MAX_ADVERTISED_STAGE_TIMING_AGE_MS);
        if hint.model_name.is_empty()
            || hint.model_name.len() > MAX_ADVERTISED_MODEL_NAME_BYTES
            || (!throughput_valid && !stage_timing_valid)
            || !seen.insert(hint.model_name.clone())
        {
            continue;
        }
        hint.avg_tokens_per_second_milli = hint
            .avg_tokens_per_second_milli
            .min(MAX_ADVERTISED_TPS_MILLI);
        hint.throughput_samples = hint
            .throughput_samples
            .min(MAX_ADVERTISED_THROUGHPUT_SAMPLES);
        if stage_timing_valid {
            hint.observed_stage_us_per_layer = hint
                .observed_stage_us_per_layer
                .map(|value| value.min(MAX_ADVERTISED_STAGE_US_PER_LAYER));
            hint.stage_timing_samples = hint
                .stage_timing_samples
                .map(|samples| samples.min(MAX_ADVERTISED_THROUGHPUT_SAMPLES));
        } else {
            hint.observed_stage_us_per_layer = None;
            hint.stage_timing_samples = None;
            hint.stage_timing_age_ms = None;
        }
        sanitized.push(hint);
        if sanitized.len() >= MAX_ADVERTISED_MODEL_THROUGHPUT_HINTS {
            break;
        }
    }
    sanitized
}
