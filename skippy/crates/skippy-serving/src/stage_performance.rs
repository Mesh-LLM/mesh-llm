//! Process-local observations of steady decode work performed by staged runtimes.
//!
//! Embedded stages share a process with the host, so retaining a bounded,
//! model-keyed timing hint here lets the host advertise real stage behavior
//! without changing the stage execution wire protocol.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use skippy_protocol::{StageConfig, binary::StageWireMessage};

const MAX_OBSERVATION_AGE: Duration = Duration::from_secs(30 * 60);
const MAX_EFFECTIVE_SAMPLES: u64 = 256;
const MAX_TRACKED_STAGE_TIMINGS: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageDecodeTimingHint {
    pub model_id: String,
    /// Mean steady-decode runtime work, normalized by the loaded layer count.
    pub observed_us_per_layer: u64,
    pub sample_count: u64,
    pub sample_age_ms: u64,
}

#[derive(Clone, Debug)]
struct StageDecodeTimingObservation {
    observed_us_per_layer: u64,
    sample_count: u64,
    observed_at: Instant,
    layer_start: u32,
    layer_end: u32,
}

static STAGE_DECODE_TIMINGS: OnceLock<Mutex<HashMap<String, StageDecodeTimingObservation>>> =
    OnceLock::new();

pub(crate) fn record_stage_decode_timing(
    config: &StageConfig,
    message: &StageWireMessage,
    compute_ms: f64,
) {
    if !matches!(
        message.kind,
        skippy_protocol::binary::WireMessageKind::DecodeEmbd
    ) || message.state.decode_step < 8
        || !compute_ms.is_finite()
        || compute_ms <= 0.0
    {
        return;
    }
    let layer_count = u64::from(config.layer_end.saturating_sub(config.layer_start));
    let Some(executed_tokens) = u64::try_from(message.token_count)
        .ok()
        .filter(|count| *count > 0)
    else {
        return;
    };
    if layer_count == 0 {
        return;
    }
    let compute_us = (compute_ms * 1_000.0).round().max(1.0) as u64;
    let sample = compute_us.div_ceil(layer_count.saturating_mul(executed_tokens));
    let mut timings = STAGE_DECODE_TIMINGS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    timings.retain(|_, observation| {
        now.duration_since(observation.observed_at) <= MAX_OBSERVATION_AGE
    });
    let incompatible = timings.get(&config.model_id).is_some_and(|observation| {
        observation.layer_start != config.layer_start || observation.layer_end != config.layer_end
    });
    if incompatible {
        timings.remove(&config.model_id);
    }
    if !timings.contains_key(&config.model_id)
        && timings.len() >= MAX_TRACKED_STAGE_TIMINGS
        && let Some(oldest) = timings
            .iter()
            .min_by_key(|(_, observation)| observation.observed_at)
            .map(|(model_id, _)| model_id.clone())
    {
        timings.remove(&oldest);
    }
    let observation =
        timings
            .entry(config.model_id.clone())
            .or_insert(StageDecodeTimingObservation {
                observed_us_per_layer: sample,
                sample_count: 0,
                observed_at: now,
                layer_start: config.layer_start,
                layer_end: config.layer_end,
            });
    if observation.sample_count < MAX_EFFECTIVE_SAMPLES {
        let next_count = observation.sample_count + 1;
        observation.observed_us_per_layer = observation
            .observed_us_per_layer
            .saturating_mul(observation.sample_count)
            .saturating_add(sample)
            / next_count;
        observation.sample_count = next_count;
    } else {
        // Retain a bounded EWMA after the initial arithmetic-mean window.
        observation.observed_us_per_layer = observation
            .observed_us_per_layer
            .saturating_mul(7)
            .saturating_add(sample)
            / 8;
    }
    observation.observed_at = now;
}

pub fn stage_decode_timing_hints() -> Vec<StageDecodeTimingHint> {
    let Some(timings) = STAGE_DECODE_TIMINGS.get() else {
        return Vec::new();
    };
    let mut timings = timings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    timings.retain(|_, observation| observation.observed_at.elapsed() <= MAX_OBSERVATION_AGE);
    let mut hints = timings
        .iter()
        .filter_map(|(model_id, observation)| {
            let age = observation.observed_at.elapsed();
            (age <= MAX_OBSERVATION_AGE).then(|| StageDecodeTimingHint {
                model_id: model_id.clone(),
                observed_us_per_layer: observation.observed_us_per_layer,
                sample_count: observation.sample_count,
                sample_age_ms: u64::try_from(age.as_millis()).unwrap_or(u64::MAX),
            })
        })
        .collect::<Vec<_>>();
    hints.sort_by(|left, right| left.model_id.cmp(&right.model_id));
    hints
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{StageStateHeader, WireMessageKind};

    fn config(model_id: &str) -> StageConfig {
        StageConfig {
            model_id: model_id.to_string(),
            layer_start: 10,
            layer_end: 20,
            ..StageConfig::default()
        }
    }

    fn message(decode_step: i32) -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state: StageStateHeader {
                decode_step,
                ..StageStateHeader::new(WireMessageKind::DecodeEmbd)
            },
            request_id: 1,
            session_id: 1,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![1],
            positions: vec![0],
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn records_only_steady_decode_and_normalizes_by_layers() {
        let model_id = format!("timing-test-{}", std::process::id());
        let config = config(&model_id);
        record_stage_decode_timing(&config, &message(7), 10.0);
        assert!(
            stage_decode_timing_hints()
                .iter()
                .all(|hint| hint.model_id != model_id)
        );

        record_stage_decode_timing(&config, &message(8), 10.0);
        let hint = stage_decode_timing_hints()
            .into_iter()
            .find(|hint| hint.model_id == model_id)
            .expect("steady timing hint");
        assert_eq!(hint.observed_us_per_layer, 1_000);
        assert_eq!(hint.sample_count, 1);
    }

    #[test]
    fn normalizes_batched_decode_by_executed_token_count() {
        let model_id = format!("timing-batch-test-{}", std::process::id());
        let config = config(&model_id);
        let mut batched = message(8);
        batched.token_count = 4;

        record_stage_decode_timing(&config, &batched, 40.0);
        let hint = stage_decode_timing_hints()
            .into_iter()
            .find(|hint| hint.model_id == model_id)
            .expect("batched timing hint");
        assert_eq!(hint.observed_us_per_layer, 1_000);
    }

    #[test]
    fn changing_stage_range_resets_the_observation_window() {
        let model_id = format!("timing-range-test-{}", std::process::id());
        let first = config(&model_id);
        record_stage_decode_timing(&first, &message(8), 10.0);

        let mut changed = first.clone();
        changed.layer_end = 15;
        record_stage_decode_timing(&changed, &message(8), 20.0);

        let hint = stage_decode_timing_hints()
            .into_iter()
            .find(|hint| hint.model_id == model_id)
            .expect("changed-range timing hint");
        assert_eq!(hint.observed_us_per_layer, 4_000);
        assert_eq!(hint.sample_count, 1);
    }
}
