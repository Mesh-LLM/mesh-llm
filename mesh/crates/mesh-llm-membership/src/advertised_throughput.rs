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
/// The longest evidence window a receiver honours, whatever the sender says.
pub const MAX_ADVERTISED_THROUGHPUT_WINDOW_MS: u64 = 15 * 60 * 1000;
/// How far in the future a measurement time may be (clock skew between
/// nodes) and still count as evidence.
pub const MAX_THROUGHPUT_FUTURE_SKEW_MS: u64 = 2 * 60 * 1000;

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
    /// When and how the samples were measured. `None` from nodes that predate
    /// it, or when the sender's metadata was malformed: the age is unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurement: Option<ThroughputMeasurement>,
}

/// The measuring node's account of the samples behind a hint. Relays forward
/// it unchanged, so receiving a hint again never makes it younger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ThroughputMeasurement {
    /// Wall-clock time of the newest sample behind the average.
    pub observed_at_unix_ms: u64,
    /// How long the measuring node keeps a sample as evidence.
    pub window_ms: u64,
    pub kind: ThroughputMeasurementKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputMeasurementKind {
    /// Completion tokens over the whole routed attempt's duration.
    EndToEndAttempt,
    /// Absent, unspecified, or a kind this build does not know.
    Unknown,
}

impl ModelThroughputHint {
    /// Whether a receiver can rank on this hint at `now_unix_ms`: a known kind,
    /// measured no later than the allowed clock skew, and no older than the
    /// shorter of the sender's window and [`MAX_ADVERTISED_THROUGHPUT_WINDOW_MS`].
    /// A hint without a measurement is never fresh.
    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        let Some(measurement) = self.measurement else {
            return false;
        };
        if measurement.kind != ThroughputMeasurementKind::EndToEndAttempt
            || measurement.observed_at_unix_ms
                > now_unix_ms.saturating_add(MAX_THROUGHPUT_FUTURE_SKEW_MS)
        {
            return false;
        }
        let window_ms = measurement
            .window_ms
            .min(MAX_ADVERTISED_THROUGHPUT_WINDOW_MS);
        now_unix_ms.saturating_sub(measurement.observed_at_unix_ms) <= window_ms
    }
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
        // A zero time or window is malformed: keep the hint, drop the claim
        // about its age.
        if hint.measurement.is_some_and(|measurement| {
            measurement.observed_at_unix_ms == 0 || measurement.window_ms == 0
        }) {
            hint.measurement = None;
        }
        sanitized.push(hint);
        if sanitized.len() >= MAX_ADVERTISED_MODEL_THROUGHPUT_HINTS {
            break;
        }
    }
    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    fn hint(measurement: Option<ThroughputMeasurement>) -> ModelThroughputHint {
        ModelThroughputHint {
            model_name: "qwen".to_string(),
            avg_tokens_per_second_milli: 40_000,
            throughput_samples: 3,
            measurement,
        }
    }

    fn measured(observed_at_unix_ms: u64, window_ms: u64) -> Option<ThroughputMeasurement> {
        Some(ThroughputMeasurement {
            observed_at_unix_ms,
            window_ms,
            kind: ThroughputMeasurementKind::EndToEndAttempt,
        })
    }

    #[test]
    fn a_hint_is_fresh_up_to_its_window_and_not_one_millisecond_after() {
        let window = 60_000;
        assert!(hint(measured(NOW - window, window)).is_fresh_at(NOW));
        assert!(!hint(measured(NOW - window - 1, window)).is_fresh_at(NOW));
    }

    #[test]
    fn the_receiver_caps_a_sender_window() {
        let observed = NOW - MAX_ADVERTISED_THROUGHPUT_WINDOW_MS - 1;
        assert!(!hint(measured(observed, u64::MAX)).is_fresh_at(NOW));
    }

    #[test]
    fn future_measurements_count_only_within_the_clock_skew() {
        let window = 60_000;
        let at_skew = NOW + MAX_THROUGHPUT_FUTURE_SKEW_MS;
        assert!(hint(measured(at_skew, window)).is_fresh_at(NOW));
        assert!(!hint(measured(at_skew + 1, window)).is_fresh_at(NOW));
    }

    #[test]
    fn hints_without_a_known_measurement_are_never_fresh() {
        assert!(!hint(None).is_fresh_at(NOW));
        let unknown_kind = Some(ThroughputMeasurement {
            observed_at_unix_ms: NOW,
            window_ms: 60_000,
            kind: ThroughputMeasurementKind::Unknown,
        });
        assert!(!hint(unknown_kind).is_fresh_at(NOW));
    }

    #[test]
    fn sanitizing_keeps_the_hint_but_drops_malformed_measurements() {
        let sanitized = sanitize_model_throughput_hints([
            hint(measured(0, 60_000)),
            ModelThroughputHint {
                model_name: "other".to_string(),
                ..hint(measured(NOW, 0))
            },
            ModelThroughputHint {
                model_name: "kept".to_string(),
                ..hint(measured(NOW, 60_000))
            },
        ]);

        assert_eq!(sanitized.len(), 3);
        assert_eq!(sanitized[0].measurement, None);
        assert_eq!(sanitized[1].measurement, None);
        assert_eq!(sanitized[2].measurement, measured(NOW, 60_000));
    }
}
