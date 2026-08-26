use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use crate::{
    CacheAwareCandidate, IterationPhase, IterationPlan, IterationPrediction, IterationTelemetry,
    IterationWork, SchedulerConfig, SchedulerMetrics, Sequence, SequenceStatus,
    order_cache_aware_candidates,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("scheduler waiting queue is full ({capacity} requests)")]
    QueueFull { capacity: usize },
    #[error("sequence id is already queued or active: {0}")]
    DuplicateSequence(String),
    #[error("sequence prompt must contain at least one token")]
    EmptyPrompt,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    pub active_ids: Vec<String>,
    pub waiting_ids: Vec<String>,
    pub component_used_bytes: Vec<(String, u64)>,
}

pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<Sequence>,
    active: BTreeMap<String, Sequence>,
    component_used_bytes: Vec<u64>,
    metrics: SchedulerMetrics,
    consecutive_prefill_iterations: usize,
    waiting_turn: u64,
    next_waiting_order: u64,
    waiting_order_dirty: bool,
    decode_us_per_row_ewma: Option<f64>,
    mixed_prefill_tokens: usize,
}

const DURATION_EWMA_ALPHA: f64 = 0.25;
const MIXED_DECODE_SLOWDOWN_FRACTION: f64 = 0.20;
const MIXED_PREFILL_INITIAL_TOKENS: usize = 64;
const MIXED_PREFILL_MIN_EXTRA_US: f64 = 8_000.0;
const MIXED_PREFILL_MAX_EXTRA_US: f64 = 16_000.0;

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let config = config.normalized();
        let mixed_prefill_tokens = config
            .max_tokens_per_iteration
            .min(MIXED_PREFILL_INITIAL_TOKENS);
        Self {
            component_used_bytes: vec![0; config.memory_components.len()],
            config,
            waiting: VecDeque::new(),
            active: BTreeMap::new(),
            metrics: SchedulerMetrics::default(),
            consecutive_prefill_iterations: 0,
            waiting_turn: 0,
            next_waiting_order: 0,
            waiting_order_dirty: false,
            decode_us_per_row_ewma: None,
            mixed_prefill_tokens,
        }
    }

    pub fn submit(&mut self, mut sequence: Sequence) -> Result<(), AdmissionError> {
        if sequence.prompt_tokens.is_empty() {
            return Err(AdmissionError::EmptyPrompt);
        }
        if self.active.contains_key(&sequence.id)
            || self.waiting.iter().any(|queued| queued.id == sequence.id)
        {
            return Err(AdmissionError::DuplicateSequence(sequence.id));
        }
        if self.waiting.len() >= self.config.max_waiting_sequences {
            self.metrics.rejected_overload = self.metrics.rejected_overload.saturating_add(1);
            return Err(AdmissionError::QueueFull {
                capacity: self.config.max_waiting_sequences,
            });
        }
        if sequence.prefix_restore.is_some() {
            self.metrics.prefix_hits = self.metrics.prefix_hits.saturating_add(1);
        } else {
            self.metrics.prefix_misses = self.metrics.prefix_misses.saturating_add(1);
        }
        sequence.enqueued_turn = self.waiting_turn;
        sequence.enqueue_order = self.next_waiting_order;
        self.next_waiting_order = self.next_waiting_order.saturating_add(1);
        self.waiting.push_back(sequence);
        self.waiting_order_dirty = true;
        self.refresh_counts();
        Ok(())
    }

    pub fn plan_iteration(&mut self) -> IterationPlan {
        self.waiting_turn = self.waiting_turn.saturating_add(1);
        let admitted = self.admit_waiting();
        let mut plan = IterationPlan {
            admitted,
            ..IterationPlan::default()
        };
        if self.config.mixed_prefill_decode {
            self.plan_mixed_iteration(&mut plan);
            return plan;
        }
        let mut budget = self.config.max_tokens_per_iteration;
        let mut prefill_sequences = 0usize;
        let ids = self.active.keys().cloned().collect::<Vec<_>>();
        // llama's mixed-token ABI can batch many prefills or many decode rows,
        // but combining a long prefill with live decode rows changes dense
        // model outputs. Give chunked prefill/recompute one iteration, then
        // resume decode for all active sequences on the next iteration.
        let has_prefill = ids.iter().any(|id| {
            self.active
                .get(id)
                .is_some_and(|sequence| sequence.prefill_cursor < sequence.recompute_token_count())
        });
        let has_live_decode = ids.iter().any(|id| {
            self.active.get(id).is_some_and(|sequence| {
                sequence.prefill_cursor >= sequence.recompute_token_count()
                    && sequence.pending_decode_token().is_some()
            })
        });
        let run_prefill_phase = has_prefill
            && (!has_live_decode
                || self.consecutive_prefill_iterations
                    < self.config.max_consecutive_prefill_iterations);

        for id in ids {
            if budget == 0 {
                break;
            }
            let Some(sequence) = self.active.get_mut(&id) else {
                continue;
            };
            let replay = sequence.recompute_tokens();
            if sequence.prefill_cursor < replay.len() {
                if !run_prefill_phase {
                    continue;
                }
                if prefill_sequences >= self.config.max_prefill_sequences_per_iteration {
                    continue;
                }
                let count = self
                    .config
                    .prefill_chunk_tokens
                    .min(replay.len() - sequence.prefill_cursor)
                    .min(budget);
                let start = sequence.prefill_cursor;
                let end = start + count;
                let phase = if sequence.generated_tokens.is_empty() {
                    IterationPhase::Prefill
                } else {
                    IterationPhase::Recompute
                };
                let sample_last = end == replay.len() && sequence.generated_tokens.is_empty();
                plan.work.push(IterationWork {
                    sequence_id: id,
                    tokens: replay[start..end].to_vec(),
                    positions: contiguous_positions(start, count),
                    sample_last,
                    phase,
                    sampling: sequence.sampling.clone(),
                });
                sequence.prefill_cursor = end;
                prefill_sequences += 1;
                plan.token_count += count;
                budget -= count;
                continue;
            }

            if run_prefill_phase {
                continue;
            }

            if let Some(token) = sequence.pending_decode_token() {
                plan.work.push(IterationWork {
                    sequence_id: id,
                    tokens: vec![token],
                    positions: contiguous_positions(replay.len(), 1),
                    sample_last: true,
                    phase: IterationPhase::Decode,
                    sampling: sequence.sampling.clone(),
                });
                // The token scheduled for decode occupies the next position.
                // Keep the replay cursor aligned so uninterrupted sequences do
                // not replay that token as recompute work on the next step.
                sequence.prefill_cursor = replay.len().saturating_add(1);
                plan.token_count += 1;
                budget -= 1;
            }
        }

        if plan
            .work
            .iter()
            .any(|work| work.phase != IterationPhase::Decode)
        {
            self.consecutive_prefill_iterations = if has_live_decode {
                self.consecutive_prefill_iterations.saturating_add(1)
            } else {
                0
            };
        } else if plan
            .work
            .iter()
            .any(|work| work.phase == IterationPhase::Decode)
        {
            self.consecutive_prefill_iterations = 0;
        }

        plan
    }

    fn plan_mixed_iteration(&mut self, plan: &mut IterationPlan) {
        let mut budget = self.config.max_tokens_per_iteration;
        let ids = self.active.keys().cloned().collect::<Vec<_>>();

        // Decode is latency-sensitive and consumes exactly one token per live
        // sequence. Reserve those rows first so prompt traffic cannot starve
        // active generation.
        for id in &ids {
            if budget == 0 {
                break;
            }
            let Some(sequence) = self.active.get_mut(id) else {
                continue;
            };
            let replay_len = sequence.recompute_token_count();
            if sequence.prefill_cursor < replay_len {
                continue;
            }
            let Some(token) = sequence.pending_decode_token() else {
                continue;
            };
            plan.work.push(IterationWork {
                sequence_id: id.clone(),
                tokens: vec![token],
                positions: contiguous_positions(replay_len, 1),
                sample_last: true,
                phase: IterationPhase::Decode,
                sampling: sequence.sampling.clone(),
            });
            sequence.prefill_cursor = replay_len.saturating_add(1);
            plan.token_count += 1;
            budget -= 1;
        }

        let decode_rows = plan
            .work
            .iter()
            .filter(|work| work.phase == IterationPhase::Decode)
            .count();
        let mut mixed_prefill_budget = self.mixed_prefill_budget(budget, decode_rows);
        let mut prefill_sequences = 0usize;
        for id in ids {
            if budget == 0
                || mixed_prefill_budget == 0
                || prefill_sequences >= self.config.max_prefill_sequences_per_iteration
            {
                break;
            }
            let Some(sequence) = self.active.get_mut(&id) else {
                continue;
            };
            let replay = sequence.recompute_tokens();
            if sequence.prefill_cursor >= replay.len() {
                continue;
            }
            let count = self
                .config
                .prefill_chunk_tokens
                .min(replay.len() - sequence.prefill_cursor)
                .min(budget)
                .min(mixed_prefill_budget);
            let start = sequence.prefill_cursor;
            let end = start + count;
            let phase = if sequence.generated_tokens.is_empty() {
                IterationPhase::Prefill
            } else {
                IterationPhase::Recompute
            };
            plan.work.push(IterationWork {
                sequence_id: id,
                tokens: replay[start..end].to_vec(),
                positions: contiguous_positions(start, count),
                sample_last: end == replay.len() && sequence.generated_tokens.is_empty(),
                phase,
                sampling: sequence.sampling.clone(),
            });
            sequence.prefill_cursor = end;
            prefill_sequences += 1;
            plan.token_count += count;
            budget -= count;
            mixed_prefill_budget -= count;
        }

        self.consecutive_prefill_iterations = 0;
    }

    fn mixed_prefill_budget(&self, available_tokens: usize, decode_rows: usize) -> usize {
        if decode_rows == 0 {
            return available_tokens;
        }
        available_tokens.min(self.mixed_prefill_tokens.max(1))
    }

    /// Feed a completed native step back into duration-aware mixed admission.
    /// Decode-only steps calibrate latency-sensitive row cost. Mixed steps use
    /// that baseline to shrink promptly when prompt work exceeds its duration
    /// allowance and grow conservatively when spare latency remains.
    pub fn observe_iteration_duration(&mut self, plan: &IterationPlan, elapsed: Duration) {
        let elapsed_us = elapsed.as_secs_f64() * 1_000_000.0;
        if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
            return;
        }
        let prefill_tokens = plan
            .work
            .iter()
            .filter(|work| work.phase != IterationPhase::Decode)
            .map(|work| work.tokens.len())
            .sum::<usize>();
        let decode_rows = plan
            .work
            .iter()
            .filter(|work| work.phase == IterationPhase::Decode)
            .count();
        let samples_prefill = plan
            .work
            .iter()
            .any(|work| work.phase != IterationPhase::Decode && work.sample_last);

        match (prefill_tokens, decode_rows) {
            (0, 0) => {}
            (0, decode_rows) => {
                let sample = elapsed_us / decode_rows as f64;
                update_duration_ewma(&mut self.decode_us_per_row_ewma, sample);
            }
            (_, 0) => {}
            (prefill_tokens, decode_rows) => {
                // Sampling the first output token adds work that does not
                // describe the cost of another non-final prompt slice.
                if samples_prefill {
                    return;
                }
                let Some(decode_us_per_row) = self.decode_us_per_row_ewma else {
                    return;
                };
                let predicted_decode_us = decode_us_per_row * decode_rows as f64;
                let observed_extra_us = (elapsed_us - predicted_decode_us).max(1.0);
                let allowed_extra_us = (predicted_decode_us * MIXED_DECODE_SLOWDOWN_FRACTION)
                    .clamp(MIXED_PREFILL_MIN_EXTRA_US, MIXED_PREFILL_MAX_EXTRA_US);
                let ratio = (allowed_extra_us / observed_extra_us).clamp(0.125, 2.0);
                let observed_tokens = prefill_tokens as f64;
                let adjusted_tokens = observed_tokens * ratio;
                let next_tokens = if ratio < 1.0 {
                    adjusted_tokens.floor()
                } else {
                    observed_tokens + DURATION_EWMA_ALPHA * (adjusted_tokens - observed_tokens)
                };
                self.mixed_prefill_tokens =
                    (next_tokens.round() as usize).clamp(1, self.config.max_tokens_per_iteration);
            }
        }
    }

    pub fn complete_iteration(
        &mut self,
        plan: &IterationPlan,
        predictions: &[IterationPrediction],
    ) -> IterationTelemetry {
        let mut terminal = Vec::new();
        let mut prefill_tokens = 0usize;
        let mut recompute_tokens = 0usize;
        let mut decode_tokens = 0usize;
        let mut predicted_tokens = vec![None; plan.work.len()];
        let mut duplicate_predictions = vec![false; plan.work.len()];
        for prediction in predictions {
            let Some(slot) = predicted_tokens.get_mut(prediction.work_index) else {
                continue;
            };
            if slot.is_some() {
                duplicate_predictions[prediction.work_index] = true;
                *slot = None;
            } else if !duplicate_predictions[prediction.work_index] {
                *slot = Some(prediction.token);
            }
        }

        for (index, work) in plan.work.iter().enumerate() {
            match work.phase {
                IterationPhase::Prefill => prefill_tokens += work.tokens.len(),
                IterationPhase::Recompute => recompute_tokens += work.tokens.len(),
                IterationPhase::Decode => decode_tokens += work.tokens.len(),
            }
            if !work.sample_last {
                continue;
            }
            let Some(sequence) = self.active.get_mut(&work.sequence_id) else {
                continue;
            };
            let Some(predicted) = predicted_tokens.get(index).copied().flatten() else {
                sequence.status = SequenceStatus::Failed;
                terminal.push((work.sequence_id.clone(), true));
                continue;
            };
            if predicted < 0 {
                sequence.status = SequenceStatus::Finished;
                terminal.push((work.sequence_id.clone(), false));
                continue;
            }
            sequence.generated_tokens.push(predicted);
            if sequence.generated_tokens.len() >= sequence.max_tokens as usize {
                sequence.status = SequenceStatus::Finished;
                terminal.push((work.sequence_id.clone(), false));
            }
        }

        for (id, failed) in terminal {
            self.remove_active(&id);
            if failed {
                self.metrics.failed = self.metrics.failed.saturating_add(1);
            } else {
                self.metrics.finished = self.metrics.finished.saturating_add(1);
            }
        }
        self.metrics.iterations = self.metrics.iterations.saturating_add(1);
        self.metrics.admitted = self.metrics.admitted.saturating_add(plan.admitted as u64);
        self.metrics.prefill_tokens = self
            .metrics
            .prefill_tokens
            .saturating_add(prefill_tokens as u64);
        self.metrics.recompute_tokens = self
            .metrics
            .recompute_tokens
            .saturating_add(recompute_tokens as u64);
        self.metrics.decode_tokens = self
            .metrics
            .decode_tokens
            .saturating_add(decode_tokens as u64);
        self.refresh_counts();

        IterationTelemetry {
            iteration: self.metrics.iterations,
            active_sequences: self.active.len(),
            waiting_sequences: self.waiting.len(),
            admitted: plan.admitted,
            preempted: plan.preempted,
            prefill_tokens,
            recompute_tokens,
            decode_tokens,
            component_used_bytes: self.component_usage(),
            component_available_bytes: self
                .config
                .memory_components
                .iter()
                .map(|component| (component.name.clone(), component.available_bytes()))
                .collect(),
            prefix_hits: self.metrics.prefix_hits,
            prefix_misses: self.metrics.prefix_misses,
            finished: self.metrics.finished,
            failed: self.metrics.failed,
            cancelled: self.metrics.cancelled,
            rejected_overload: self.metrics.rejected_overload,
        }
    }

    pub fn preempt_for_component_pressure(&mut self, component_used_bytes: &[u64]) -> Vec<String> {
        for (target, used) in self
            .component_used_bytes
            .iter_mut()
            .zip(component_used_bytes.iter().copied())
        {
            *target = used;
        }
        let mut preempted = Vec::new();
        while self.component_over_capacity() {
            let Some(victim) = self.preemption_victim() else {
                break;
            };
            let Some(mut sequence) = self.active.remove(&victim) else {
                break;
            };
            self.release_memory(&sequence);
            sequence.reset_for_recompute();
            sequence.enqueue_order = self.next_waiting_order;
            self.next_waiting_order = self.next_waiting_order.saturating_add(1);
            self.waiting.push_front(sequence);
            self.waiting_order_dirty = true;
            preempted.push(victim);
            self.metrics.preempted = self.metrics.preempted.saturating_add(1);
        }
        self.refresh_counts();
        preempted
    }

    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    pub fn snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            active_ids: self.active.keys().cloned().collect(),
            waiting_ids: self
                .waiting
                .iter()
                .map(|sequence| sequence.id.clone())
                .collect(),
            component_used_bytes: self.component_usage(),
        }
    }

    pub fn sequence(&self, id: &str) -> Option<&Sequence> {
        self.active
            .get(id)
            .or_else(|| self.waiting.iter().find(|sequence| sequence.id == id))
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        if self.active.contains_key(id) {
            self.remove_active(id);
            self.metrics.cancelled = self.metrics.cancelled.saturating_add(1);
            self.refresh_counts();
            return true;
        }
        if let Some(index) = self.waiting.iter().position(|sequence| sequence.id == id) {
            self.waiting.remove(index);
            self.waiting_order_dirty = true;
            self.metrics.cancelled = self.metrics.cancelled.saturating_add(1);
            self.refresh_counts();
            return true;
        }
        false
    }

    fn admit_waiting(&mut self) -> usize {
        let mut admitted = 0;
        let mut deferred = VecDeque::new();
        if self.waiting_order_dirty {
            let order = order_cache_aware_candidates(
                self.waiting
                    .iter()
                    .enumerate()
                    .map(|(index, sequence)| CacheAwareCandidate {
                        index,
                        priority: sequence.priority,
                        affinity: &sequence.cache_affinity,
                        prompt_tokens: &sequence.prompt_tokens,
                        enqueued_turn: sequence.enqueued_turn,
                        order: sequence.enqueue_order,
                    }),
                self.waiting_turn,
                self.config.cache_aging_cost_per_iteration,
                self.config.group_waiting_prefixes,
            );
            if is_complete_permutation(&order, self.waiting.len()) {
                let mut waiting = self.waiting.drain(..).map(Some).collect::<Vec<_>>();
                self.waiting = order
                    .into_iter()
                    .filter_map(|index| waiting[index].take())
                    .collect();
            }
            self.waiting_order_dirty = false;
        }
        while let Some(mut sequence) = self.waiting.pop_front() {
            if self.active.len() >= self.config.max_active_sequences
                || !self.can_reserve_memory(&sequence)
            {
                deferred.push_back(sequence);
                continue;
            }
            self.reserve_memory(&sequence);
            sequence.status = SequenceStatus::Running;
            sequence.admitted_at = Some(Instant::now());
            self.active.insert(sequence.id.clone(), sequence);
            admitted += 1;
        }
        self.waiting = deferred;
        self.refresh_counts();
        admitted
    }

    fn can_reserve_memory(&self, sequence: &Sequence) -> bool {
        self.config
            .memory_components
            .iter()
            .zip(self.component_used_bytes.iter().copied())
            .all(|(component, used)| {
                used.saturating_add(component.reservation_bytes(sequence.admission_tokens))
                    <= component.available_bytes()
            })
    }

    fn reserve_memory(&mut self, sequence: &Sequence) {
        for (index, component) in self.config.memory_components.iter().enumerate() {
            self.component_used_bytes[index] = self.component_used_bytes[index]
                .saturating_add(component.reservation_bytes(sequence.admission_tokens));
        }
    }

    fn release_memory(&mut self, sequence: &Sequence) {
        for (index, component) in self.config.memory_components.iter().enumerate() {
            self.component_used_bytes[index] = self.component_used_bytes[index]
                .saturating_sub(component.reservation_bytes(sequence.admission_tokens));
        }
    }

    fn remove_active(&mut self, id: &str) {
        if let Some(sequence) = self.active.remove(id) {
            self.release_memory(&sequence);
        }
    }

    fn component_over_capacity(&self) -> bool {
        self.config
            .memory_components
            .iter()
            .zip(self.component_used_bytes.iter().copied())
            .any(|(component, used)| used > component.available_bytes())
    }

    fn preemption_victim(&self) -> Option<String> {
        self.active
            .values()
            .min_by(|left, right| {
                left.priority.cmp(&right.priority).then_with(|| {
                    right
                        .admitted_at
                        .unwrap_or_else(Instant::now)
                        .cmp(&left.admitted_at.unwrap_or_else(Instant::now))
                })
            })
            .map(|sequence| sequence.id.clone())
    }

    fn component_usage(&self) -> Vec<(String, u64)> {
        self.config
            .memory_components
            .iter()
            .zip(self.component_used_bytes.iter().copied())
            .map(|(component, used)| (component.name.clone(), used))
            .collect()
    }

    fn refresh_counts(&mut self) {
        self.metrics.active_sequences = self.active.len();
        self.metrics.waiting_sequences = self.waiting.len();
    }
}

fn is_complete_permutation(order: &[usize], len: usize) -> bool {
    if order.len() != len {
        return false;
    }
    let mut seen = vec![false; len];
    order
        .iter()
        .copied()
        .all(|index| index < len && !std::mem::replace(&mut seen[index], true))
}

fn update_duration_ewma(estimate: &mut Option<f64>, sample: f64) {
    if !sample.is_finite() || sample <= 0.0 {
        return;
    }
    *estimate = Some(estimate.map_or(sample, |current| {
        current + DURATION_EWMA_ALPHA * (sample - current)
    }));
}

fn contiguous_positions(start: usize, count: usize) -> Vec<i32> {
    (start..start.saturating_add(count))
        .map(|position| i32::try_from(position).unwrap_or(i32::MAX))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryComponent, PrefixRestore, PrefixRestoreKind};

    fn sequence(id: &str, prompt_len: usize, max_tokens: u32) -> Sequence {
        Sequence::new(
            id.into(),
            (0..prompt_len).map(|token| token as i32).collect(),
            max_tokens,
            None,
            1,
        )
    }

    #[test]
    fn iteration_separates_prefill_and_decode_without_replaying_final_prompt_token() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_tokens_per_iteration: 8,
            prefill_chunk_tokens: 4,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("decode", 2, 4)).unwrap();
        let first = scheduler.plan_iteration();
        assert!(first.work[0].sample_last);
        scheduler.complete_iteration(
            &first,
            &[IterationPrediction {
                work_index: 0,
                token: 42,
            }],
        );
        scheduler.submit(sequence("prefill", 8, 4)).unwrap();

        let prefill_chunk = scheduler.plan_iteration();
        assert_eq!(prefill_chunk.work.len(), 1);
        assert_eq!(prefill_chunk.work[0].sequence_id, "prefill");
        assert_eq!(prefill_chunk.work[0].phase, IterationPhase::Prefill);
        assert!(!prefill_chunk.work[0].sample_last);
        scheduler.complete_iteration(&prefill_chunk, &[]);

        let final_prefill = scheduler.plan_iteration();
        assert_eq!(final_prefill.work.len(), 1);
        assert_eq!(final_prefill.work[0].sequence_id, "prefill");
        assert_eq!(final_prefill.work[0].phase, IterationPhase::Prefill);
        assert!(final_prefill.work[0].sample_last);
        scheduler.complete_iteration(
            &final_prefill,
            &[IterationPrediction {
                work_index: 0,
                token: 100,
            }],
        );

        let next = scheduler.plan_iteration();
        let decode = next
            .work
            .iter()
            .find(|work| work.sequence_id == "decode")
            .unwrap();
        assert_eq!(decode.phase, IterationPhase::Decode);
        assert_eq!(decode.tokens, vec![42]);
        assert_eq!(decode.positions, vec![2]);
        assert!(
            next.work
                .iter()
                .all(|work| work.phase == IterationPhase::Decode)
        );
        assert!(
            next.work
                .iter()
                .all(|work| work.sequence_id != "decode" || work.phase != IterationPhase::Recompute)
        );
    }

    #[test]
    fn bounded_prefill_iterations_allow_live_decode_progress() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_tokens_per_iteration: 8,
            prefill_chunk_tokens: 4,
            max_consecutive_prefill_iterations: 1,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("decode", 2, 4)).unwrap();
        let initial = scheduler.plan_iteration();
        scheduler.complete_iteration(
            &initial,
            &[IterationPrediction {
                work_index: 0,
                token: 42,
            }],
        );
        scheduler.submit(sequence("prefill", 8, 4)).unwrap();

        let prefill = scheduler.plan_iteration();
        assert_eq!(prefill.work.len(), 1);
        assert_eq!(prefill.work[0].sequence_id, "prefill");
        assert_eq!(prefill.work[0].phase, IterationPhase::Prefill);
        scheduler.complete_iteration(&prefill, &[]);

        let decode = scheduler.plan_iteration();
        assert_eq!(decode.work.len(), 1);
        assert_eq!(decode.work[0].sequence_id, "decode");
        assert_eq!(decode.work[0].phase, IterationPhase::Decode);
        scheduler.complete_iteration(
            &decode,
            &[IterationPrediction {
                work_index: 0,
                token: 43,
            }],
        );

        let resumed_prefill = scheduler.plan_iteration();
        assert_eq!(resumed_prefill.work.len(), 1);
        assert_eq!(resumed_prefill.work[0].sequence_id, "prefill");
        assert_eq!(resumed_prefill.work[0].phase, IterationPhase::Prefill);
    }

    #[test]
    fn mixed_iteration_schedules_decode_first_and_fills_remaining_budget() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_tokens_per_iteration: 5,
            prefill_chunk_tokens: 5,
            mixed_prefill_decode: true,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("decode", 2, 4)).unwrap();
        let initial = scheduler.plan_iteration();
        scheduler.observe_iteration_duration(&initial, Duration::from_micros(200));
        scheduler.complete_iteration(
            &initial,
            &[IterationPrediction {
                work_index: 0,
                token: 42,
            }],
        );
        let decode_only = scheduler.plan_iteration();
        scheduler.observe_iteration_duration(&decode_only, Duration::from_millis(10));
        scheduler.complete_iteration(
            &decode_only,
            &[IterationPrediction {
                work_index: 0,
                token: 43,
            }],
        );
        scheduler.submit(sequence("prefill", 8, 4)).unwrap();

        let mixed = scheduler.plan_iteration();

        assert_eq!(mixed.token_count, 5);
        assert_eq!(mixed.work.len(), 2);
        assert_eq!(mixed.work[0].sequence_id, "decode");
        assert_eq!(mixed.work[0].phase, IterationPhase::Decode);
        assert_eq!(mixed.work[0].tokens, [43]);
        assert_eq!(mixed.work[1].sequence_id, "prefill");
        assert_eq!(mixed.work[1].phase, IterationPhase::Prefill);
        assert_eq!(mixed.work[1].tokens.len(), 4);
        assert!(!mixed.work[1].sample_last);
    }

    #[test]
    fn duration_aware_mixed_admission_caps_slow_prompt_work() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_tokens_per_iteration: 128,
            prefill_chunk_tokens: 128,
            mixed_prefill_decode: true,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("decode", 2, 8)).unwrap();
        let initial = scheduler.plan_iteration();
        scheduler.complete_iteration(
            &initial,
            &[IterationPrediction {
                work_index: 0,
                token: 42,
            }],
        );
        let decode_only = scheduler.plan_iteration();
        scheduler.observe_iteration_duration(&decode_only, Duration::from_millis(10));
        scheduler.complete_iteration(
            &decode_only,
            &[IterationPrediction {
                work_index: 0,
                token: 43,
            }],
        );
        scheduler.submit(sequence("prefill", 256, 4)).unwrap();

        let slow_mixed = scheduler.plan_iteration();
        assert_eq!(slow_mixed.work[1].tokens.len(), 64);
        scheduler.observe_iteration_duration(&slow_mixed, Duration::from_millis(170));
        scheduler.complete_iteration(
            &slow_mixed,
            &[IterationPrediction {
                work_index: 0,
                token: 44,
            }],
        );

        let bounded_mixed = scheduler.plan_iteration();

        assert_eq!(bounded_mixed.work.len(), 2);
        assert_eq!(bounded_mixed.work[0].phase, IterationPhase::Decode);
        assert_eq!(bounded_mixed.work[1].phase, IterationPhase::Prefill);
        assert_eq!(bounded_mixed.work[1].tokens.len(), 8);
        assert_eq!(bounded_mixed.token_count, 9);
    }

    #[test]
    fn sparse_iteration_predictions_map_to_their_explicit_work_rows() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_active_sequences: 3,
            max_tokens_per_iteration: 8,
            prefill_chunk_tokens: 4,
            mixed_prefill_decode: true,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("a-decode", 2, 4)).unwrap();
        let initial = scheduler.plan_iteration();
        scheduler.observe_iteration_duration(&initial, Duration::from_micros(200));
        scheduler.complete_iteration(
            &initial,
            &[IterationPrediction {
                work_index: 0,
                token: 42,
            }],
        );
        let decode_only = scheduler.plan_iteration();
        scheduler.observe_iteration_duration(&decode_only, Duration::from_millis(10));
        scheduler.complete_iteration(
            &decode_only,
            &[IterationPrediction {
                work_index: 0,
                token: 43,
            }],
        );
        scheduler.submit(sequence("b-prefill", 8, 4)).unwrap();
        scheduler.submit(sequence("c-final", 2, 4)).unwrap();

        let mixed = scheduler.plan_iteration();
        assert_eq!(mixed.work.len(), 3);
        assert!(mixed.work[0].sample_last);
        assert!(!mixed.work[1].sample_last);
        assert!(mixed.work[2].sample_last);

        scheduler.complete_iteration(
            &mixed,
            &[
                IterationPrediction {
                    work_index: 0,
                    token: 44,
                },
                IterationPrediction {
                    work_index: 2,
                    token: 99,
                },
            ],
        );

        assert_eq!(
            scheduler.sequence("a-decode").unwrap().generated_tokens,
            [42, 43, 44]
        );
        assert_eq!(
            scheduler.sequence("c-final").unwrap().generated_tokens,
            [99]
        );
        assert!(scheduler.sequence("b-prefill").is_some());
    }

    #[test]
    fn admission_uses_effective_minimum_across_components() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_active_sequences: 8,
            memory_components: vec![
                MemoryComponent {
                    name: "full-attention".into(),
                    capacity_bytes: 10_000,
                    resident_bytes: 0,
                    bytes_per_token: 1,
                    bytes_per_sequence: 0,
                },
                MemoryComponent {
                    name: "recurrent".into(),
                    capacity_bytes: 1024,
                    resident_bytes: 0,
                    bytes_per_token: 0,
                    bytes_per_sequence: 1024,
                },
            ],
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("a", 10, 10)).unwrap();
        scheduler.submit(sequence("b", 10, 10)).unwrap();
        let plan = scheduler.plan_iteration();
        assert_eq!(plan.admitted, 1);
        assert_eq!(scheduler.snapshot().waiting_ids, vec!["b"]);
    }

    #[test]
    fn admission_groups_the_heaviest_waiting_prefix_subtree() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_active_sequences: 1,
            ..SchedulerConfig::default()
        });
        scheduler
            .submit(Sequence::new("unique".into(), vec![9, 9, 9], 1, None, 0))
            .unwrap();
        scheduler
            .submit(Sequence::new("shared-a".into(), vec![1, 2, 3], 1, None, 0))
            .unwrap();
        scheduler
            .submit(Sequence::new("shared-b".into(), vec![1, 2, 4], 1, None, 0))
            .unwrap();

        let plan = scheduler.plan_iteration();

        assert_eq!(plan.admitted, 1);
        assert_eq!(scheduler.snapshot().active_ids, ["shared-a"]);
    }

    #[test]
    fn prefill_sequence_limit_serializes_recurrent_prefills() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_active_sequences: 2,
            max_prefill_sequences_per_iteration: 1,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("a", 2, 1)).unwrap();
        scheduler.submit(sequence("b", 2, 1)).unwrap();

        let first = scheduler.plan_iteration();
        assert_eq!(first.work.len(), 1);
        assert_eq!(first.work[0].sequence_id, "a");
        scheduler.complete_iteration(
            &first,
            &[IterationPrediction {
                work_index: 0,
                token: 10,
            }],
        );

        let second = scheduler.plan_iteration();
        assert_eq!(second.work.len(), 1);
        assert_eq!(second.work[0].sequence_id, "b");
    }

    #[test]
    fn prefix_restore_skips_only_the_restored_token_span() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let restored = sequence("a", 8, 4).with_prefix_restore(PrefixRestore {
            page_id: "page".into(),
            token_count: 6,
            kind: PrefixRestoreKind::RecurrentWholeState,
        });
        scheduler.submit(restored).unwrap();
        let plan = scheduler.plan_iteration();
        assert_eq!(plan.work[0].tokens, vec![6, 7]);
        assert_eq!(plan.work[0].positions, vec![6, 7]);
    }

    #[test]
    fn full_prefix_restore_replays_the_final_token_for_logits() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let restored = sequence("a", 8, 4).with_prefix_restore(PrefixRestore {
            page_id: "page".into(),
            token_count: 8,
            kind: PrefixRestoreKind::ResidentKv,
        });
        scheduler.submit(restored).unwrap();

        let plan = scheduler.plan_iteration();

        assert_eq!(plan.work.len(), 1);
        assert_eq!(plan.work[0].tokens, vec![7]);
        assert_eq!(plan.work[0].positions, vec![7]);
        assert!(plan.work[0].sample_last);
    }

    #[test]
    fn overload_is_rejected_before_native_decode() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_waiting_sequences: 1,
            ..SchedulerConfig::default()
        });
        scheduler.submit(sequence("a", 2, 1)).unwrap();
        assert_eq!(
            scheduler.submit(sequence("b", 2, 1)),
            Err(AdmissionError::QueueFull { capacity: 1 })
        );
        assert_eq!(scheduler.metrics().rejected_overload, 1);
    }

    #[test]
    fn missing_sample_prediction_fails_instead_of_finishing_sequence() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        scheduler.submit(sequence("a", 2, 4)).unwrap();
        let plan = scheduler.plan_iteration();

        scheduler.complete_iteration(&plan, &[]);

        assert!(scheduler.sequence("a").is_none());
        assert_eq!(scheduler.metrics().failed, 1);
        assert_eq!(scheduler.metrics().finished, 0);
    }

    #[test]
    fn cancellation_releases_memory_and_removes_waiting_or_active_sequence() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            memory_components: vec![MemoryComponent {
                name: "kv".into(),
                capacity_bytes: 16,
                resident_bytes: 0,
                bytes_per_token: 1,
                bytes_per_sequence: 0,
            }],
            ..SchedulerConfig::default()
        });
        scheduler
            .submit(sequence("active", 2, 4).with_admission_tokens(8))
            .unwrap();
        scheduler
            .submit(sequence("waiting", 2, 4).with_admission_tokens(16))
            .unwrap();
        scheduler.plan_iteration();

        assert!(scheduler.cancel("active"));
        assert!(scheduler.cancel("waiting"));
        assert!(scheduler.snapshot().active_ids.is_empty());
        assert!(scheduler.snapshot().waiting_ids.is_empty());
        assert_eq!(scheduler.metrics().cancelled, 2);
        assert_eq!(
            scheduler.snapshot().component_used_bytes,
            vec![("kv".into(), 0)]
        );
    }

    #[test]
    fn repeated_component_preemption_preserves_aging_credit() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_active_sequences: 1,
            cache_aging_cost_per_iteration: 10,
            memory_components: vec![MemoryComponent {
                name: "kv".into(),
                capacity_bytes: 2,
                resident_bytes: 0,
                bytes_per_token: 1,
                bytes_per_sequence: 0,
            }],
            ..SchedulerConfig::default()
        });
        let hot_affinity = crate::CacheAffinity::from_stage(crate::StageCacheAffinity {
            stage_index: 0,
            matched_tokens: 1,
            prefill_cost_per_token: 80,
            restore_cost: 0,
            cache_epoch: 0,
        });
        scheduler
            .submit(sequence("cold", 2, 4).with_admission_tokens(1))
            .unwrap();
        scheduler.plan_iteration();
        scheduler.waiting_turn = 10;

        for hot_id in ["hot-a", "hot-b"] {
            assert_eq!(scheduler.preempt_for_component_pressure(&[3]), ["cold"]);
            scheduler.preempt_for_component_pressure(&[0]);
            scheduler
                .submit(
                    sequence(hot_id, 2, 4)
                        .with_admission_tokens(1)
                        .with_cache_affinity(hot_affinity.clone()),
                )
                .unwrap();

            scheduler.plan_iteration();

            assert_eq!(scheduler.snapshot().active_ids, ["cold"]);
            assert_eq!(scheduler.sequence("cold").unwrap().enqueued_turn, 0);
        }
    }
}
