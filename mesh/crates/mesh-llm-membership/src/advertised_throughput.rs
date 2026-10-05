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

/// Soft peer-advertised model throughput hint.
///
/// Values are fixed-point milli tokens/second to keep gossip deterministic and
/// avoid protobuf floating-point edge cases. They are advisory only; routing
/// clamps and local observations take precedence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelThroughputHint {
    pub model_name: String,
    pub avg_tokens_per_second_milli: u64,
    pub throughput_samples: u64,
}

pub fn sanitize_model_throughput_hints<I>(hints: I) -> Vec<ModelThroughputHint>
where
    I: IntoIterator<Item = ModelThroughputHint>,
{
    let mut seen = HashSet::new();
    let mut sanitized = Vec::new();
    for mut hint in hints {
        hint.model_name = hint.model_name.trim().to_string();
        if hint.model_name.is_empty()
            || hint.model_name.len() > MAX_ADVERTISED_MODEL_NAME_BYTES
            || hint.avg_tokens_per_second_milli == 0
            || hint.throughput_samples == 0
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
        sanitized.push(hint);
        if sanitized.len() >= MAX_ADVERTISED_MODEL_THROUGHPUT_HINTS {
            break;
        }
    }
    sanitized
}
