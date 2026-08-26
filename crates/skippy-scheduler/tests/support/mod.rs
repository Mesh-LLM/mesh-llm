use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use skippy_cache::UnifiedRadixCache;
use skippy_scheduler::{
    CacheAffinity, IterationPhase, IterationPlan, IterationPrediction, PrefixRestore,
    PrefixRestoreKind, Scheduler, SchedulerConfig, Sequence, StageCacheAffinity,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimRequest {
    pub id: String,
    pub arrival_us: u64,
    pub prompt_tokens: usize,
    pub output_tokens: u32,
    pub prefix_tokens: usize,
    pub priority: u64,
    pub token_offset: i32,
    pub cache_affinity: CacheAffinity,
}

impl SimRequest {
    pub fn new(
        id: impl Into<String>,
        arrival_us: u64,
        prompt_tokens: usize,
        output_tokens: u32,
    ) -> Self {
        Self {
            id: id.into(),
            arrival_us,
            prompt_tokens,
            output_tokens,
            prefix_tokens: 0,
            priority: 0,
            token_offset: 0,
            cache_affinity: CacheAffinity::default(),
        }
    }

    pub fn with_prefix_tokens(mut self, prefix_tokens: usize) -> Self {
        self.prefix_tokens = prefix_tokens.min(self.prompt_tokens);
        self
    }

    pub fn with_token_offset(mut self, token_offset: i32) -> Self {
        self.token_offset = token_offset;
        self
    }

    fn prompt_token_ids(&self) -> Vec<i32> {
        (0..self.prompt_tokens)
            .map(|token| {
                self.token_offset
                    .saturating_add(i32::try_from(token).unwrap_or(i32::MAX))
            })
            .collect()
    }
}

/// Probe the real unified radix and attach stage-local affinity plus the
/// corresponding modeled restore cursor to each request.
pub fn apply_resident_radix_affinity<R: Clone, E>(
    cache: &UnifiedRadixCache<R, E>,
    namespace: &str,
    stage_index: u32,
    prefill_cost_per_token: u64,
    requests: &mut [SimRequest],
) {
    for request in requests {
        let tokens = request.prompt_token_ids();
        let Some(hit) = cache.peek_resident(namespace, &tokens) else {
            continue;
        };
        request.prefix_tokens = hit.matched_tokens.min(request.prompt_tokens);
        request.cache_affinity = CacheAffinity::from_stage(StageCacheAffinity {
            stage_index,
            matched_tokens: request.prefix_tokens,
            prefill_cost_per_token,
            restore_cost: 0,
            cache_epoch: cache.epoch(),
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCostModel {
    pub iteration_overhead_us: u64,
    pub prefill_token_us: u64,
    pub decode_batch_us: u64,
    pub decode_sequence_us: u64,
}

impl Default for RuntimeCostModel {
    fn default() -> Self {
        Self {
            iteration_overhead_us: 50,
            prefill_token_us: 2,
            decode_batch_us: 100,
            decode_sequence_us: 20,
        }
    }
}

impl RuntimeCostModel {
    fn iteration_duration_us(self, plan: &IterationPlan) -> u64 {
        let prefill_tokens = plan
            .work
            .iter()
            .filter(|work| {
                matches!(
                    work.phase,
                    IterationPhase::Prefill | IterationPhase::Recompute
                )
            })
            .map(|work| work.tokens.len())
            .sum::<usize>();
        let decode_sequences = plan
            .work
            .iter()
            .filter(|work| work.phase == IterationPhase::Decode)
            .count();
        let mut duration = self.iteration_overhead_us.saturating_add(
            self.prefill_token_us
                .saturating_mul(u64::try_from(prefill_tokens).unwrap_or(u64::MAX)),
        );
        if decode_sequences > 0 {
            duration = duration
                .saturating_add(self.decode_batch_us)
                .saturating_add(
                    self.decode_sequence_us
                        .saturating_mul(u64::try_from(decode_sequences).unwrap_or(u64::MAX)),
                );
        }
        duration.max(1)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestMetrics {
    pub arrival_us: u64,
    pub first_scheduled_us: Option<u64>,
    pub first_token_us: Option<u64>,
    pub completed_us: Option<u64>,
    pub generated_tokens: u32,
    pub max_inter_token_gap_us: u64,
    last_token_us: Option<u64>,
}

impl RequestMetrics {
    pub fn queue_wait_us(&self) -> Option<u64> {
        self.first_scheduled_us
            .map(|scheduled| scheduled.saturating_sub(self.arrival_us))
    }

    pub fn ttft_us(&self) -> Option<u64> {
        self.first_token_us
            .map(|first| first.saturating_sub(self.arrival_us))
    }

    pub fn latency_us(&self) -> Option<u64> {
        self.completed_us
            .map(|completed| completed.saturating_sub(self.arrival_us))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimulationReport {
    pub requests: BTreeMap<String, RequestMetrics>,
    pub makespan_us: u64,
    pub iterations: u64,
    pub prefill_iterations: u64,
    pub decode_iterations: u64,
    pub mixed_iterations: u64,
    pub mean_batch_size: f64,
    pub mean_token_occupancy: f64,
}

impl SimulationReport {
    pub fn request(&self, id: &str) -> &RequestMetrics {
        self.requests
            .get(id)
            .unwrap_or_else(|| panic!("missing simulated request: {id}"))
    }

    pub fn throughput_requests_per_second(&self) -> f64 {
        if self.makespan_us == 0 {
            return 0.0;
        }
        self.requests.len() as f64 * 1_000_000.0 / self.makespan_us as f64
    }
}

pub fn simulate(
    config: SchedulerConfig,
    cost: RuntimeCostModel,
    mut requests: Vec<SimRequest>,
) -> Result<SimulationReport, String> {
    validate_requests(&requests)?;
    requests.sort_by(|left, right| {
        left.arrival_us
            .cmp(&right.arrival_us)
            .then_with(|| left.id.cmp(&right.id))
    });
    let normalized = config.normalized();
    let iteration_capacity = normalized.max_tokens_per_iteration;
    let mut scheduler = Scheduler::new(normalized);
    let mut pending = VecDeque::from(requests);
    let total_requests = pending.len();
    let mut metrics = pending
        .iter()
        .map(|request| {
            (
                request.id.clone(),
                RequestMetrics {
                    arrival_us: request.arrival_us,
                    ..RequestMetrics::default()
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut now_us = 0u64;
    let mut completed = 0usize;
    let mut iterations = 0u64;
    let mut prefill_iterations = 0u64;
    let mut decode_iterations = 0u64;
    let mut mixed_iterations = 0u64;
    let mut batch_size_sum = 0usize;
    let mut occupancy_sum = 0.0f64;

    while completed < total_requests {
        submit_arrivals(&mut scheduler, &mut pending, now_us)?;
        let snapshot = scheduler.snapshot();
        if snapshot.active_ids.is_empty() && snapshot.waiting_ids.is_empty() {
            let Some(next) = pending.front() else {
                return Err("simulation ended before all requests completed".to_string());
            };
            now_us = now_us.max(next.arrival_us);
            continue;
        }

        let plan = scheduler.plan_iteration();
        if plan.work.is_empty() {
            return Err(format!(
                "scheduler stalled with active={:?}, waiting={:?}",
                snapshot.active_ids, snapshot.waiting_ids
            ));
        }
        let iteration_start_us = now_us;
        record_scheduled(&plan, iteration_start_us, &mut metrics);
        let duration_us = cost.iteration_duration_us(&plan);
        now_us = now_us.saturating_add(duration_us);
        record_predictions(&plan, now_us, &mut metrics);
        let predictions = plan
            .work
            .iter()
            .enumerate()
            .filter(|(_, work)| work.sample_last)
            .map(|(work_index, _)| IterationPrediction {
                work_index,
                token: 1,
            })
            .collect::<Vec<_>>();
        scheduler.observe_iteration_duration(&plan, Duration::from_micros(duration_us));
        scheduler.complete_iteration(&plan, &predictions);
        completed += record_completions(&scheduler, now_us, &mut metrics);

        let has_prefill = plan.work.iter().any(|work| {
            matches!(
                work.phase,
                IterationPhase::Prefill | IterationPhase::Recompute
            )
        });
        let has_decode = plan
            .work
            .iter()
            .any(|work| work.phase == IterationPhase::Decode);
        match (has_prefill, has_decode) {
            (true, true) => mixed_iterations = mixed_iterations.saturating_add(1),
            (true, false) => prefill_iterations = prefill_iterations.saturating_add(1),
            (false, true) => decode_iterations = decode_iterations.saturating_add(1),
            (false, false) => {}
        }
        iterations = iterations.saturating_add(1);
        batch_size_sum = batch_size_sum.saturating_add(plan.work.len());
        occupancy_sum += plan.token_count as f64 / iteration_capacity as f64;
    }

    Ok(SimulationReport {
        requests: metrics,
        makespan_us: now_us,
        iterations,
        prefill_iterations,
        decode_iterations,
        mixed_iterations,
        mean_batch_size: batch_size_sum as f64 / iterations as f64,
        mean_token_occupancy: occupancy_sum / iterations as f64,
    })
}

fn validate_requests(requests: &[SimRequest]) -> Result<(), String> {
    if requests.is_empty() {
        return Err("simulation requires at least one request".to_string());
    }
    let mut ids = requests
        .iter()
        .map(|request| request.id.as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("simulation request ids must be unique".to_string());
    }
    if let Some(request) = requests.iter().find(|request| {
        request.prompt_tokens == 0
            || request.output_tokens == 0
            || request.prefix_tokens > request.prompt_tokens
    }) {
        return Err(format!("invalid simulated request: {}", request.id));
    }
    Ok(())
}

fn submit_arrivals(
    scheduler: &mut Scheduler,
    pending: &mut VecDeque<SimRequest>,
    now_us: u64,
) -> Result<(), String> {
    while pending
        .front()
        .is_some_and(|request| request.arrival_us <= now_us)
    {
        let request = pending.pop_front().expect("front checked above");
        let prompt_tokens = request.prompt_token_ids();
        let mut sequence = Sequence::new(
            request.id.clone(),
            prompt_tokens,
            request.output_tokens,
            None,
            request.priority,
        );
        if request.prefix_tokens > 0 {
            sequence = sequence.with_prefix_restore(PrefixRestore {
                page_id: format!("sim-prefix-{}", request.id),
                token_count: request.prefix_tokens,
                kind: PrefixRestoreKind::ResidentKv,
            });
        }
        sequence = sequence.with_cache_affinity(request.cache_affinity);
        scheduler
            .submit(sequence)
            .map_err(|error| format!("submit {}: {error}", request.id))?;
    }
    Ok(())
}

fn record_scheduled(
    plan: &IterationPlan,
    now_us: u64,
    metrics: &mut BTreeMap<String, RequestMetrics>,
) {
    for work in &plan.work {
        let request = metrics
            .get_mut(&work.sequence_id)
            .expect("scheduler returned unknown request");
        request.first_scheduled_us.get_or_insert(now_us);
    }
}

fn record_predictions(
    plan: &IterationPlan,
    now_us: u64,
    metrics: &mut BTreeMap<String, RequestMetrics>,
) {
    for work in plan.work.iter().filter(|work| work.sample_last) {
        let request = metrics
            .get_mut(&work.sequence_id)
            .expect("scheduler returned unknown request");
        request.first_token_us.get_or_insert(now_us);
        if let Some(previous) = request.last_token_us.replace(now_us) {
            request.max_inter_token_gap_us = request
                .max_inter_token_gap_us
                .max(now_us.saturating_sub(previous));
        }
        request.generated_tokens = request.generated_tokens.saturating_add(1);
    }
}

fn record_completions(
    scheduler: &Scheduler,
    now_us: u64,
    metrics: &mut BTreeMap<String, RequestMetrics>,
) -> usize {
    let mut completed = 0;
    for (id, request) in metrics {
        if request.completed_us.is_none()
            && request.first_scheduled_us.is_some()
            && scheduler.sequence(id).is_none()
        {
            request.completed_us = Some(now_us);
            completed += 1;
        }
    }
    completed
}

pub fn burst_requests(
    concurrency: usize,
    prompt_tokens: usize,
    output_tokens: u32,
    prefix_tokens: usize,
) -> Vec<SimRequest> {
    (0..concurrency)
        .map(|index| {
            SimRequest::new(format!("request-{index}"), 0, prompt_tokens, output_tokens)
                .with_prefix_tokens(prefix_tokens)
        })
        .collect()
}

pub fn staggered_prefill_requests() -> Vec<SimRequest> {
    let mut requests = vec![SimRequest::new("decoder", 0, 32, 64)];
    requests.extend((0..4).map(|index| {
        SimRequest::new(
            format!("prefill-{index}"),
            1_000 + u64::try_from(index).unwrap_or(u64::MAX) * 1_000,
            1_024,
            4,
        )
    }));
    requests
}
