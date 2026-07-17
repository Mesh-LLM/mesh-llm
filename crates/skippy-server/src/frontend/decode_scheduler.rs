use std::collections::{BTreeMap, VecDeque};

use super::{OpenAiError, OpenAiResult};

const PIPELINE_PROFILE_MIN_OBSERVATIONS: usize = 8;
const PIPELINE_PROFILE_MAX_OBSERVATIONS: usize = 32;
const PIPELINE_PROFIT_MARGIN: f64 = 1.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowPipelineConfig {
    depth: usize,
}

impl VerifyWindowPipelineConfig {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
        }
    }

    pub(super) fn depth(self) -> usize {
        self.depth
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct VerifyWindowPipelineStats {
    depth: usize,
    direct_prediction_return: bool,
    opened_windows: usize,
    max_in_flight: usize,
    stale_discarded: usize,
    stale_drain_ms: f64,
    policy_observed_windows: usize,
    policy_continuation_windows: usize,
    policy_permit_checks: usize,
    policy_permits: usize,
    policy_suppressed: usize,
}

impl VerifyWindowPipelineStats {
    pub(super) fn insert_response_timings(self, timings: &mut BTreeMap<String, serde_json::Value>) {
        timings.insert(
            "verify_window_depth".to_string(),
            serde_json::json!(self.depth),
        );
        timings.insert(
            "verify_window_direct_prediction_return".to_string(),
            serde_json::json!(self.direct_prediction_return),
        );
        timings.insert(
            "verify_window_opened".to_string(),
            serde_json::json!(self.opened_windows),
        );
        timings.insert(
            "verify_window_max_in_flight".to_string(),
            serde_json::json!(self.max_in_flight),
        );
        timings.insert(
            "verify_window_stale_discarded".to_string(),
            serde_json::json!(self.stale_discarded),
        );
        timings.insert(
            "verify_window_stale_drain_ms".to_string(),
            serde_json::json!(self.stale_drain_ms),
        );
        timings.insert(
            "verify_window_policy_observed_windows".to_string(),
            serde_json::json!(self.policy_observed_windows),
        );
        timings.insert(
            "verify_window_policy_continuation_windows".to_string(),
            serde_json::json!(self.policy_continuation_windows),
        );
        timings.insert(
            "verify_window_policy_permit_checks".to_string(),
            serde_json::json!(self.policy_permit_checks),
        );
        timings.insert(
            "verify_window_policy_permits".to_string(),
            serde_json::json!(self.policy_permits),
        );
        timings.insert(
            "verify_window_policy_suppressed".to_string(),
            serde_json::json!(self.policy_suppressed),
        );
    }
}

#[derive(Debug, Default)]
struct VerifyWindowWidthProfile {
    observations: VecDeque<VerifyWindowProfileObservation>,
    continuation_windows: usize,
    stage0_compute_ms: f64,
    downstream_wait_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct VerifyWindowProfileObservation {
    continues: bool,
    stage0_compute_ms: f64,
    downstream_wait_ms: f64,
}

impl VerifyWindowWidthProfile {
    fn observe(&mut self, continues: bool, stage0_compute_ms: f64, downstream_wait_ms: f64) {
        let observation = VerifyWindowProfileObservation {
            continues,
            stage0_compute_ms: stage0_compute_ms.max(0.0),
            downstream_wait_ms: downstream_wait_ms.max(0.0),
        };
        self.observations.push_back(observation);
        self.continuation_windows = self
            .continuation_windows
            .saturating_add(usize::from(continues));
        self.stage0_compute_ms += observation.stage0_compute_ms;
        self.downstream_wait_ms += observation.downstream_wait_ms;
        if self.observations.len() > PIPELINE_PROFILE_MAX_OBSERVATIONS {
            let expired = self
                .observations
                .pop_front()
                .expect("profile exceeded its non-empty bound");
            self.continuation_windows = self
                .continuation_windows
                .saturating_sub(usize::from(expired.continues));
            self.stage0_compute_ms -= expired.stage0_compute_ms;
            self.downstream_wait_ms -= expired.downstream_wait_ms;
        }
    }

    fn is_profitable(&self) -> bool {
        if self.observations.len() < PIPELINE_PROFILE_MIN_OBSERVATIONS {
            return false;
        }
        let observations = self.observations.len() as f64;
        let continuation_rate = self.continuation_windows as f64 / observations;
        let average_stage0_ms = self.stage0_compute_ms / observations;
        let average_downstream_ms = self.downstream_wait_ms / observations;
        let expected_overlap_ms = continuation_rate * average_downstream_ms;
        let expected_stale_ms =
            (1.0 - continuation_rate) * average_stage0_ms.max(average_downstream_ms);
        expected_overlap_ms > expected_stale_ms * PIPELINE_PROFIT_MARGIN
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerifyWindow {
    pub(super) id: i32,
    pub(super) base_position: usize,
    pub(super) decode_step: usize,
}

#[derive(Debug)]
pub(super) struct VerifyWindowScheduler {
    config: VerifyWindowPipelineConfig,
    next_id: i32,
    in_flight: VecDeque<VerifyWindow>,
    stats: VerifyWindowPipelineStats,
    width_profiles: BTreeMap<usize, VerifyWindowWidthProfile>,
}

impl VerifyWindowScheduler {
    pub(super) fn new(config: VerifyWindowPipelineConfig) -> Self {
        Self {
            config,
            next_id: 1,
            in_flight: VecDeque::new(),
            stats: VerifyWindowPipelineStats {
                depth: config.depth(),
                ..VerifyWindowPipelineStats::default()
            },
            width_profiles: BTreeMap::new(),
        }
    }

    pub(super) fn has_capacity(&self) -> bool {
        self.in_flight.len() < self.config.depth
    }

    pub(super) fn depth(&self) -> usize {
        self.config.depth()
    }

    pub(super) fn mark_direct_prediction_return(&mut self) {
        self.stats.direct_prediction_return = true;
    }

    pub(super) fn observe_pipeline_profile(
        &mut self,
        width: usize,
        continues: bool,
        stage0_compute_ms: f64,
        downstream_wait_ms: f64,
    ) {
        if width == 0 {
            return;
        }
        self.stats.policy_observed_windows = self.stats.policy_observed_windows.saturating_add(1);
        self.stats.policy_continuation_windows = self
            .stats
            .policy_continuation_windows
            .saturating_add(usize::from(continues));
        self.width_profiles.entry(width).or_default().observe(
            continues,
            stage0_compute_ms,
            downstream_wait_ms,
        );
    }

    pub(super) fn has_profitable_pipeline_width(&self) -> bool {
        self.config.depth() > 1
            && self
                .width_profiles
                .values()
                .any(VerifyWindowWidthProfile::is_profitable)
    }

    pub(super) fn permit_pipeline_width(&mut self, width: usize) -> bool {
        self.stats.policy_permit_checks = self.stats.policy_permit_checks.saturating_add(1);
        let permitted = self.config.depth() > 1
            && self
                .width_profiles
                .get(&width)
                .is_some_and(VerifyWindowWidthProfile::is_profitable);
        if permitted {
            self.stats.policy_permits = self.stats.policy_permits.saturating_add(1);
        } else {
            self.stats.policy_suppressed = self.stats.policy_suppressed.saturating_add(1);
        }
        permitted
    }

    pub(super) fn insert_policy_telemetry_attrs(
        &self,
        attrs: &mut BTreeMap<String, serde_json::Value>,
    ) {
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_observed_windows".to_string(),
            serde_json::json!(self.stats.policy_observed_windows),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_continuation_windows".to_string(),
            serde_json::json!(self.stats.policy_continuation_windows),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_permit_checks".to_string(),
            serde_json::json!(self.stats.policy_permit_checks),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_permits".to_string(),
            serde_json::json!(self.stats.policy_permits),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_suppressed".to_string(),
            serde_json::json!(self.stats.policy_suppressed),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_profitable_widths".to_string(),
            serde_json::json!(
                self.width_profiles
                    .values()
                    .filter(|profile| profile.is_profitable())
                    .count()
            ),
        );
    }

    pub(super) fn open(
        &mut self,
        base_position: usize,
        decode_step: usize,
    ) -> OpenAiResult<VerifyWindow> {
        if !self.has_capacity() {
            return Err(OpenAiError::backend(
                "verify window pipeline depth exceeded",
            ));
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| OpenAiError::backend("verify window id overflow"))?;
        let window = VerifyWindow {
            id,
            base_position,
            decode_step,
        };
        self.in_flight.push_back(window.clone());
        self.stats.opened_windows = self.stats.opened_windows.saturating_add(1);
        self.stats.max_in_flight = self.stats.max_in_flight.max(self.in_flight.len());
        Ok(window)
    }

    pub(super) fn complete_next(&mut self, reply_window_id: i32) -> OpenAiResult<VerifyWindow> {
        let Some(window) = self.in_flight.front() else {
            return Err(OpenAiError::backend(
                "verify window reply arrived with no in-flight window",
            ));
        };
        if window.id != reply_window_id {
            return Err(OpenAiError::backend(format!(
                "verify window reply out of order: got {reply_window_id}, expected {}",
                window.id
            )));
        }
        Ok(self.in_flight.pop_front().expect("checked non-empty queue"))
    }

    #[cfg(test)]
    pub(super) fn discard_stale(&mut self) -> usize {
        let discarded = self.in_flight.len();
        self.in_flight.clear();
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(discarded);
        discarded
    }

    pub(super) fn record_stale_discarded(&mut self, count: usize, drain_ms: f64) {
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(count);
        self.stats.stale_drain_ms += drain_ms;
    }

    pub(super) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn stale_discard_count(&self) -> usize {
        self.stats.stale_discarded
    }

    pub(super) fn stats(&self) -> VerifyWindowPipelineStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_depth_and_requires_fifo_reply_ids() {
        let config = VerifyWindowPipelineConfig { depth: 2 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        let first = scheduler.open(10, 0).unwrap();
        let second = scheduler.open(11, 1).unwrap();

        assert!(scheduler.open(12, 2).is_err());
        assert!(scheduler.complete_next(second.id).is_err());
        assert_eq!(scheduler.in_flight_len(), 2);
        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
        assert_eq!(first.id, 1);
        assert_eq!(scheduler.stats().depth, 2);
        assert_eq!(scheduler.stats().opened_windows, 2);
        assert_eq!(scheduler.stats().max_in_flight, 2);
        assert!(!scheduler.stats().direct_prediction_return);
    }

    #[test]
    fn discards_stale_windows_after_divergence() {
        let config = VerifyWindowPipelineConfig { depth: 3 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        scheduler.open(10, 0).unwrap();
        scheduler.open(11, 1).unwrap();
        scheduler.open(12, 2).unwrap();

        assert_eq!(scheduler.discard_stale(), 3);
        assert_eq!(scheduler.stale_discard_count(), 3);
        assert_eq!(scheduler.in_flight_len(), 0);
        assert_eq!(scheduler.stats().stale_discarded, 3);
    }

    #[test]
    fn pipeline_policy_waits_for_enough_width_specific_evidence() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for _ in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS - 1 {
            scheduler.observe_pipeline_profile(2, true, 20.0, 80.0);
        }
        assert!(!scheduler.has_profitable_pipeline_width());
        assert!(!scheduler.permit_pipeline_width(2));

        scheduler.observe_pipeline_profile(2, true, 20.0, 80.0);
        assert!(scheduler.has_profitable_pipeline_width());
        assert!(scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_policy_suppresses_low_acceptance_local_work() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for index in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, index < 2, 31.0, 24.0);
        }

        assert!(!scheduler.has_profitable_pipeline_width());
        assert!(!scheduler.permit_pipeline_width(2));
        assert_eq!(scheduler.stats().policy_permit_checks, 1);
        assert_eq!(scheduler.stats().policy_permits, 0);
        assert_eq!(scheduler.stats().policy_suppressed, 1);
    }

    #[test]
    fn pipeline_policy_profiles_each_verify_width_independently() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for index in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(1, index < 7, 20.0, 80.0);
            scheduler.observe_pipeline_profile(2, index < 2, 31.0, 24.0);
        }

        assert!(scheduler.has_profitable_pipeline_width());
        assert!(scheduler.permit_pipeline_width(1));
        assert!(!scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_depth_one_never_admits_dependent_work() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 1 });
        for _ in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, true, 20.0, 80.0);
        }

        assert!(!scheduler.has_profitable_pipeline_width());
        assert!(!scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_policy_adapts_when_recent_acceptance_changes() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for _ in 0..PIPELINE_PROFILE_MAX_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, true, 20.0, 80.0);
        }
        assert!(scheduler.permit_pipeline_width(2));

        for _ in 0..PIPELINE_PROFILE_MAX_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, false, 20.0, 80.0);
        }
        assert!(!scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_policy_counters_are_exposed_in_response_timings() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for _ in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, true, 20.0, 80.0);
        }
        assert!(scheduler.permit_pipeline_width(2));
        let mut timings = BTreeMap::new();
        scheduler.stats().insert_response_timings(&mut timings);

        assert_eq!(
            timings["verify_window_policy_observed_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(
            timings["verify_window_policy_continuation_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(timings["verify_window_policy_permit_checks"], 1);
        assert_eq!(timings["verify_window_policy_permits"], 1);
        assert_eq!(timings["verify_window_policy_suppressed"], 0);

        let mut attrs = BTreeMap::new();
        scheduler.insert_policy_telemetry_attrs(&mut attrs);
        assert_eq!(
            attrs["llama_stage.verify_window.pipeline_policy_observed_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(
            attrs["llama_stage.verify_window.pipeline_policy_continuation_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(
            attrs["llama_stage.verify_window.pipeline_policy_profitable_widths"],
            1
        );
    }
}
