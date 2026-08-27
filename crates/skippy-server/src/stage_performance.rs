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
    if layer_count == 0 {
        return;
    }
    let compute_us = (compute_ms * 1_000.0).round().max(1.0) as u64;
    let sample = compute_us.div_ceil(layer_count);
    let mut timings = STAGE_DECODE_TIMINGS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let observation =
        timings
            .entry(config.model_id.clone())
            .or_insert(StageDecodeTimingObservation {
                observed_us_per_layer: sample,
                sample_count: 0,
                observed_at: Instant::now(),
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
    observation.observed_at = Instant::now();
}

pub fn stage_decode_timing_hints() -> Vec<StageDecodeTimingHint> {
    let Some(timings) = STAGE_DECODE_TIMINGS.get() else {
        return Vec::new();
    };
    let timings = timings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    use skippy_protocol::{
        LoadMode,
        binary::{StageStateHeader, WireActivationDType, WireMessageKind},
    };

    fn config(model_id: &str) -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: model_id.to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage".to_string(),
            stage_index: 0,
            layer_start: 10,
            layer_end: 20,
            ctx_size: 1_024,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: Default::default(),
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn message(decode_step: i32) -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state: StageStateHeader {
                decode_step,
                ..StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32)
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
}
