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
    /// When and how the samples were measured. On a peer's hint, `None` means
    /// the age is unknown (a node that predates it, or malformed metadata) and
    /// [`Self::is_fresh_at`] is false. A node's own local observation also
    /// carries `None`; it is read straight from the local window and must not
    /// be filtered through `is_fresh_at`.
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

/// Merge the hints a relay forwards for a peer into the ones already held for
/// it. The relay's set decides which models are listed, but a held hint is
/// kept when the relay's would leave worse evidence: one dated later than the
/// relay's, a dated one the relay forwards undated (an older relay drops the
/// fields), an equal-timestamp copy with a window no longer than the held
/// one's, or any copy that is not fresh at `now_unix_ms` while the held hint
/// still is (a too-future timestamp must not clobber usable evidence). A relay
/// can therefore never age, strip, shrink, or spoil a hint; only the peer
/// itself, through direct gossip, replaces its hints outright.
pub fn merge_relayed_throughput_hints(
    held: &[ModelThroughputHint],
    relayed: &[ModelThroughputHint],
    now_unix_ms: u64,
) -> Vec<ModelThroughputHint> {
    relayed
        .iter()
        .map(|incoming| {
            match held
                .iter()
                .find(|current| current.model_name == incoming.model_name)
            {
                Some(current)
                    if relayed_hint_is_not_newer(current, incoming)
                        || (current.is_fresh_at(now_unix_ms)
                            && !incoming.is_fresh_at(now_unix_ms)) =>
                {
                    current.clone()
                }
                _ => incoming.clone(),
            }
        })
        .collect()
}

fn relayed_hint_is_not_newer(
    current: &ModelThroughputHint,
    incoming: &ModelThroughputHint,
) -> bool {
    match (current.measurement, incoming.measurement) {
        (Some(_), None) => true,
        (Some(current), Some(incoming)) => {
            incoming.observed_at_unix_ms < current.observed_at_unix_ms
                || (incoming.observed_at_unix_ms == current.observed_at_unix_ms
                    && incoming.window_ms <= current.window_ms)
        }
        (None, _) => false,
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

    fn named(model: &str, measurement: Option<ThroughputMeasurement>) -> ModelThroughputHint {
        ModelThroughputHint {
            model_name: model.to_string(),
            ..hint(measurement)
        }
    }

    #[test]
    fn a_relay_can_neither_age_nor_strip_a_held_hint() {
        let held = [
            named("older-relayed", measured(NOW, 60_000)),
            named("stripped", measured(NOW, 60_000)),
            named("newer-relayed", measured(NOW - 1_000, 60_000)),
            named("undated", None),
            named("shrunk", measured(NOW, 60_000)),
            named("gone", measured(NOW, 60_000)),
        ];
        let relayed = [
            named("older-relayed", measured(NOW - 5_000, 60_000)),
            named("stripped", None),
            named("newer-relayed", measured(NOW, 60_000)),
            named("undated", measured(NOW, 60_000)),
            named("shrunk", measured(NOW, 30_000)),
            named("added", measured(NOW, 60_000)),
        ];

        let merged = merge_relayed_throughput_hints(&held, &relayed, NOW);
        let by_model = |model: &str| {
            merged
                .iter()
                .find(|hint| hint.model_name == model)
                .map(|hint| hint.measurement)
        };
        assert_eq!(by_model("older-relayed"), Some(measured(NOW, 60_000)));
        assert_eq!(by_model("stripped"), Some(measured(NOW, 60_000)));
        assert_eq!(by_model("newer-relayed"), Some(measured(NOW, 60_000)));
        assert_eq!(by_model("undated"), Some(measured(NOW, 60_000)));
        assert_eq!(
            by_model("shrunk"),
            Some(measured(NOW, 60_000)),
            "an equal-timestamp relay with a shorter window does not shrink the held hint"
        );
        assert_eq!(by_model("added"), Some(measured(NOW, 60_000)));
        assert_eq!(by_model("gone"), None, "the relay's set decides the models");
    }

    #[test]
    fn repeated_relaying_of_the_same_hint_does_not_change_it() {
        let held = [named("qwen", measured(NOW - 10_000, 60_000))];
        let merged = merge_relayed_throughput_hints(&held, &held, NOW);
        assert_eq!(merged, held);
    }

    #[test]
    fn a_not_fresh_relayed_hint_does_not_clobber_a_fresh_held_hint() {
        let held = [named("qwen", measured(NOW - 30_000, 60_000))];
        let beyond_skew = NOW + MAX_THROUGHPUT_FUTURE_SKEW_MS + 1;
        let relayed = [named("qwen", measured(beyond_skew, 60_000))];
        assert!(!relayed[0].is_fresh_at(NOW));
        assert!(held[0].is_fresh_at(NOW));

        let merged = merge_relayed_throughput_hints(&held, &relayed, NOW);
        assert_eq!(
            merged[0].measurement, held[0].measurement,
            "a too-future timestamp must not spoil usable evidence"
        );
    }

    #[test]
    fn a_not_fresh_held_hint_is_replaced_by_a_fresh_relayed_one() {
        let held = [named("qwen", measured(NOW - 10 * 60_000, 60_000))];
        let relayed = [named("qwen", measured(NOW, 60_000))];
        assert!(!held[0].is_fresh_at(NOW));

        let merged = merge_relayed_throughput_hints(&held, &relayed, NOW);
        assert_eq!(
            merged[0].measurement, relayed[0].measurement,
            "the freshness guard only protects a still-usable held hint"
        );
    }
}
