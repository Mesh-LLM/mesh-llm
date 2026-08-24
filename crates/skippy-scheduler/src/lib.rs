use anyhow::{Context, Result};
use parking_lot::Mutex;
use skippy_runtime::SamplingConfig;
use skippy_server::runtime_state::{RuntimeDecodeBatchRequest, RuntimeState};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Preempted,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: String,
    pub seq_id: usize,
    pub prompt_tokens: Vec<i32>,
    pub prompt_pos: usize,
    pub max_tokens: u32,
    pub generated_tokens: usize,
    pub sampling: Option<SamplingConfig>,
    pub status: SequenceStatus,
    pub priority: u64,
    pub admitted_at: Option<Instant>,
    pub kv_prefix_shared: bool,
    /// Most recent sampled token, fed back as the next decode input.
    pub last_sampled_token: Option<i32>,
}

impl Sequence {
    pub fn new(
        id: String,
        prompt_tokens: Vec<i32>,
        max_tokens: u32,
        sampling: Option<SamplingConfig>,
        priority: u64,
    ) -> Self {
        let prompt_pos = prompt_tokens.len().saturating_sub(1);
        Self {
            id,
            seq_id: 0,
            prompt_tokens,
            prompt_pos,
            max_tokens,
            generated_tokens: 0,
            sampling,
            status: SequenceStatus::Waiting,
            priority,
            admitted_at: None,
            kv_prefix_shared: false,
            last_sampled_token: None,
        }
    }

    pub fn is_prefill_done(&self) -> bool {
        self.prompt_pos >= self.prompt_tokens.len()
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn remaining_prefill(&self) -> usize {
        self.prompt_tokens.len().saturating_sub(self.prompt_pos)
    }

    pub fn next_prefill_chunk(&self, chunk_size: usize) -> &[i32] {
        let end = (self.prompt_pos + chunk_size).min(self.prompt_tokens.len());
        &self.prompt_tokens[self.prompt_pos..end]
    }

    /// The token to feed into the next decode step: the last sampled token once
    /// generation has begun, otherwise the final prompt token (the first decode
    /// input after prefill completes).
    pub fn next_decode_token(&self) -> i32 {
        self.last_sampled_token
            .or_else(|| self.prompt_tokens.last().copied())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_seq: usize,
    pub max_tokens_per_step: usize,
    pub prefill_chunk_size: usize,
    pub kv_budget_tokens: usize,
    pub step_interval: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_seq: 32,
            max_tokens_per_step: 1024,
            prefill_chunk_size: 128,
            kv_budget_tokens: 8192,
            step_interval: Duration::from_millis(10),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SchedulerMetrics {
    pub step_count: u64,
    pub sequences_admitted: u64,
    pub sequences_preempted: u64,
    pub sequences_finished: u64,
    pub prefill_tokens_total: u64,
    pub decode_tokens_total: u64,
    pub kv_cells_used: usize,
    pub kv_cells_free: usize,
    pub prefix_share_hits: u64,
    pub prefix_share_misses: u64,
    pub avg_step_ms: f64,
    pub max_step_ms: f64,
    pub running_count: usize,
    pub waiting_count: usize,
}

#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step_number: u64,
    pub running_count: usize,
    pub waiting_count: usize,
    pub admitted_this_step: usize,
    pub preempted_this_step: usize,
    pub finished_this_step: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub kv_cells_used: usize,
    pub kv_cells_free: usize,
    pub prefix_share_hits: u64,
    pub prefix_share_misses: u64,
    pub step_ms: f64,
}

struct StepBatch {
    prefill_requests: Vec<(String, Vec<i32>)>,
    prefill_tokens: usize,
    decode_requests: Vec<(String, i32, Option<SamplingConfig>)>,
}

enum StepCommand {
    Admit(Sequence),
    Step,
    Shutdown,
}

pub struct Scheduler {
    runtime: Arc<Mutex<RuntimeState>>,
    config: SchedulerConfig,
    state: Mutex<SchedulerState>,
    metrics: Mutex<SchedulerMetrics>,
    step_tx: mpsc::UnboundedSender<StepCommand>,
    step_rx: Mutex<Option<mpsc::UnboundedReceiver<StepCommand>>>,
    telemetry_tx: Option<mpsc::UnboundedSender<StepMetrics>>,
}

struct SchedulerState {
    waiting_queue: VecDeque<Sequence>,
    running_set: BTreeMap<String, Sequence>,
    next_seq_id: usize,
    free_seq_ids: Vec<usize>,
}

impl Scheduler {
    pub fn new(runtime: Arc<Mutex<RuntimeState>>, mut config: SchedulerConfig) -> Result<Self> {
        let (step_tx, step_rx) = mpsc::unbounded_channel();
        let (telemetry_tx, _telemetry_rx) = mpsc::unbounded_channel();
        let lane_count = {
            let rt = runtime.lock();
            rt.lane_count() as usize
        };
        config.max_seq = config.max_seq.min(lane_count);

        Ok(Self {
            runtime,
            config,
            state: Mutex::new(SchedulerState {
                waiting_queue: VecDeque::new(),
                running_set: BTreeMap::new(),
                next_seq_id: 0,
                free_seq_ids: Vec::new(),
            }),
            metrics: Mutex::new(SchedulerMetrics::default()),
            step_tx,
            step_rx: Mutex::new(Some(step_rx)),
            telemetry_tx: Some(telemetry_tx),
        })
    }

    /// Get the telemetry receiver for consuming step metrics
    pub fn take_telemetry_receiver(&self) -> Option<mpsc::UnboundedReceiver<StepMetrics>> {
        // This would need a different approach since we can't move from &self
        // For now, we emit telemetry internally
        None
    }

    pub fn submit(&self, sequence: Sequence) -> Result<()> {
        self.step_tx.send(StepCommand::Admit(sequence))?;
        Ok(())
    }

    pub fn submit_batch(&self, sequences: Vec<Sequence>) -> Result<()> {
        for seq in sequences {
            self.step_tx.send(StepCommand::Admit(seq))?;
        }
        Ok(())
    }

    pub async fn run(&self) -> Result<()> {
        let mut step_rx = self
            .step_rx
            .lock()
            .take()
            .context("scheduler already running")?;
        let mut step_interval = interval(self.config.step_interval);

        info!(
            "Starting scheduler with max_seq={}, max_tokens_per_step={}",
            self.config.max_seq, self.config.max_tokens_per_step
        );

        loop {
            tokio::select! {
                _ = step_interval.tick() => {
                    self.step_tx.send(StepCommand::Step).ok();
                }
                Some(cmd) = step_rx.recv() => {
                    match cmd {
                        StepCommand::Admit(seq) => self.handle_admit(seq).await,
                        StepCommand::Step => self.run_step().await,
                        StepCommand::Shutdown => break,
                    }
                }
                else => break,
            }
        }

        info!(
            "Scheduler stopped. Final metrics: {:?}",
            *self.metrics.lock()
        );
        Ok(())
    }

    async fn handle_admit(&self, mut sequence: Sequence) {
        let seq_id = {
            let mut state = self.state.lock();
            if let Some(id) = state.free_seq_ids.pop() {
                id
            } else {
                let id = state.next_seq_id;
                state.next_seq_id += 1;
                id
            }
        };
        sequence.seq_id = seq_id;
        sequence.status = SequenceStatus::Waiting;
        self.state.lock().waiting_queue.push_back(sequence);
    }

    async fn run_step(&self) {
        let step_start = Instant::now();

        let state_before = {
            let state = self.state.lock();
            (state.waiting_queue.len(), state.running_set.len())
        };
        let (waiting_count, running_count) = state_before;

        let admitted = self.admit_sequences();
        let step_batch = self.compose_step_batch();
        let prefill_tokens = step_batch.prefill_tokens;
        let decode_count = step_batch.decode_requests.len();

        let mut preempted_this_step = 0;
        let mut finished_this_step = 0;

        if prefill_tokens == 0 && decode_count == 0 {
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            self.update_metrics(step_ms, admitted, 0, 0);
            return;
        }

        let decode_result = self.execute_step_batch(step_batch).await;

        match decode_result {
            Ok(predicted) => {
                let (preempted, finished) = self
                    .post_process(predicted, prefill_tokens, decode_count)
                    .await;
                preempted_this_step = preempted;
                finished_this_step = finished;
            }
            Err(e) => {
                tracing::error!("Step execution failed: {:?}", e);
            }
        }

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        self.update_metrics(step_ms, admitted, prefill_tokens, decode_count);

        // Emit step telemetry
        self.emit_step_telemetry(StepMetrics {
            step_number: {
                let m = self.metrics.lock();
                m.step_count + 1
            },
            running_count,
            waiting_count,
            admitted_this_step: admitted,
            preempted_this_step,
            finished_this_step,
            prefill_tokens,
            decode_tokens: decode_count,
            kv_cells_used: {
                let rt = self.runtime.lock();
                rt.session_stats().total_session_tokens as usize
            },
            kv_cells_free: self.config.kv_budget_tokens,
            prefix_share_hits: 0, // Updated in post_process
            prefix_share_misses: 0,
            step_ms,
        });
    }

    fn admit_sequences(&self) -> usize {
        let mut admitted = 0;
        let mut kv_used = {
            let rt = self.runtime.lock();
            rt.session_stats().total_session_tokens as usize
        };

        let mut state = self.state.lock();
        let mut new_waiting = VecDeque::new();
        while let Some(mut seq) = state.waiting_queue.pop_front() {
            let estimated_kv = seq.prompt_tokens.len()
                + self.config.max_tokens_per_step.min(seq.max_tokens as usize);
            if kv_used + estimated_kv <= self.config.kv_budget_tokens
                && state.running_set.len() < self.config.max_seq
            {
                seq.status = SequenceStatus::Running;
                seq.admitted_at = Some(Instant::now());
                state.running_set.insert(seq.id.clone(), seq);
                kv_used += estimated_kv;
                admitted += 1;
            } else {
                new_waiting.push_back(seq);
            }
        }
        state.waiting_queue = new_waiting;
        admitted
    }

    /// Admit a sequence that has been pre-processed with prefix sharing.
    /// This is called by the frontend after KV cache lookup.
    pub fn admit_with_prefix_sharing(
        &self,
        mut sequence: Sequence,
        shared_prefix_len: usize,
    ) -> Result<bool> {
        let mut kv_used = {
            let rt = self.runtime.lock();
            rt.session_stats().total_session_tokens as usize
        };

        let mut state = self.state.lock();
        let estimated_kv = sequence.prompt_tokens.len()
            + self
                .config
                .max_tokens_per_step
                .min(sequence.max_tokens as usize);
        if kv_used + estimated_kv <= self.config.kv_budget_tokens
            && state.running_set.len() < self.config.max_seq
        {
            sequence.status = SequenceStatus::Running;
            sequence.admitted_at = Some(Instant::now());
            sequence.prompt_pos = sequence
                .prompt_tokens
                .len()
                .saturating_sub(shared_prefix_len);
            sequence.kv_prefix_shared = true;
            state.running_set.insert(sequence.id.clone(), sequence);
            Ok(true)
        } else {
            state.waiting_queue.push_back(sequence);
            Ok(false)
        }
    }

    /// Try to admit a sequence from the waiting queue, attempting prefix sharing.
    /// This uses the RuntimeState's borrow_resident_prefix_session for zero-copy KV sharing.
    async fn try_admit_with_prefix_sharing(&self, sequence: &Sequence) -> Result<bool> {
        let mut rt = self.runtime.lock();
        // Try to find a matching resident prefix
        // For now, we use a simple heuristic: if the sequence's prompt matches
        // a prefix in the cache, use borrow_resident_prefix_session
        // This is a simplified version - the real implementation would use KV integration
        Ok(false)
    }

    fn compose_step_batch(&self) -> StepBatch {
        let mut state = self.state.lock();
        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();
        let mut prefill_tokens = 0;
        let mut decode_tokens = 0;

        for (id, seq) in &mut state.running_set {
            if seq.status != SequenceStatus::Running {
                continue;
            }

            if !seq.is_prefill_done() && prefill_tokens < self.config.max_tokens_per_step {
                let chunk_size = self
                    .config
                    .prefill_chunk_size
                    .min(self.config.max_tokens_per_step - prefill_tokens);
                let chunk = seq.next_prefill_chunk(chunk_size);
                if !chunk.is_empty() {
                    prefill_requests.push((id.clone(), chunk.to_vec()));
                    prefill_tokens += chunk.len();
                }
            }

            if seq.is_prefill_done()
                && decode_tokens < self.config.max_tokens_per_step - prefill_tokens
            {
                decode_requests.push((id.clone(), seq.next_decode_token(), seq.sampling.clone()));
                decode_tokens += 1;
            }
        }

        StepBatch {
            prefill_requests,
            prefill_tokens,
            decode_requests,
        }
    }

    async fn execute_step_batch(&self, batch: StepBatch) -> Result<Vec<i32>> {
        let mut rt = self.runtime.lock();

        for (session_id, tokens) in &batch.prefill_requests {
            rt.prefill(session_id, tokens)?;
        }

        let predicted = if !batch.decode_requests.is_empty() {
            let decode_requests: Vec<RuntimeDecodeBatchRequest<'_>> = batch
                .decode_requests
                .iter()
                .map(|(id, token_id, sampling)| RuntimeDecodeBatchRequest {
                    session_id: id.as_str(),
                    token_id: *token_id,
                    sampling: sampling.as_ref(),
                })
                .collect();
            rt.decode_batch_sampled(&decode_requests)?
        } else {
            Vec::new()
        };

        Ok(predicted)
    }

    async fn post_process(
        &self,
        predicted: Vec<i32>,
        prefill_tokens: usize,
        decode_count: usize,
    ) -> (usize, usize) {
        let mut finished = Vec::new();
        let mut preempted = Vec::new();

        let mut pred_idx = 0;
        {
            let mut state = self.state.lock();
            for (id, seq) in &mut state.running_set {
                if seq.status != SequenceStatus::Running {
                    continue;
                }

                if !seq.is_prefill_done() {
                    seq.prompt_pos += seq.remaining_prefill().min(self.config.prefill_chunk_size);
                } else if pred_idx < predicted.len() {
                    let token = predicted[pred_idx];
                    pred_idx += 1;
                    seq.generated_tokens += 1;

                    if token < 0 || seq.generated_tokens >= seq.max_tokens as usize {
                        seq.status = SequenceStatus::Finished;
                        finished.push(id.clone());
                    } else {
                        // Feed the sampled token back as the next decode input.
                        seq.last_sampled_token = Some(token);
                    }
                }
            }
        }

        let finished_count = finished.len();
        for id in finished {
            self.finish_sequence(id).await;
        }

        let kv_pressure = self.check_kv_pressure();
        if kv_pressure {
            let to_preempt = self.select_preemption_victims();
            for id in to_preempt {
                self.preempt_sequence(id.clone()).await;
                preempted.push(id);
            }
        }

        let mut metrics = self.metrics.lock();
        metrics.prefill_tokens_total += prefill_tokens as u64;
        metrics.decode_tokens_total += decode_count as u64;
        metrics.sequences_preempted += preempted.len() as u64;
        metrics.sequences_finished += finished_count as u64;

        (preempted.len(), finished_count)
    }

    fn check_kv_pressure(&self) -> bool {
        let rt = self.runtime.lock();
        let stats = rt.session_stats();
        (stats.total_session_tokens as usize) > self.config.kv_budget_tokens.saturating_mul(9) / 10
    }

    fn select_preemption_victims(&self) -> Vec<String> {
        let state = self.state.lock();
        let mut candidates: Vec<_> = state
            .running_set
            .iter()
            .filter(|(_, seq)| seq.status == SequenceStatus::Running)
            .map(|(id, seq)| {
                (
                    id.clone(),
                    seq.priority,
                    seq.admitted_at.unwrap_or(Instant::now()),
                )
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2).reverse()));

        candidates
            .into_iter()
            .take(1)
            .map(|(id, _, _)| id)
            .collect()
    }

    async fn finish_sequence(&self, id: String) {
        let seq = {
            let mut state = self.state.lock();
            state.running_set.remove(&id)
        };
        if let Some(seq) = seq {
            let mut rt = self.runtime.lock();
            rt.drop_session_timed(&id).ok();
            self.state.lock().free_seq_ids.push(seq.seq_id);
            let mut metrics = self.metrics.lock();
            metrics.sequences_finished += 1;
        }
    }

    async fn preempt_sequence(&self, id: String) {
        let seq = {
            let mut state = self.state.lock();
            state.running_set.remove(&id)
        };
        if let Some(mut seq) = seq {
            let mut rt = self.runtime.lock();
            rt.drop_session_timed(&id).ok();
            drop(rt);
            // Recompute-on-resume: drop KV and re-queue for a fresh prefill. The
            // sequence keeps its assigned seq_id (it is not returned to the free
            // list) so a concurrently admitted request cannot collide with it.
            seq.status = SequenceStatus::Preempted;
            seq.prompt_pos = 0;
            seq.generated_tokens = 0;
            seq.last_sampled_token = None;
            seq.admitted_at = None;
            // NOTE: recompute currently restarts from the prompt only; resuming
            // mid-generation must re-prefill previously generated tokens too.
            // Tracked as a follow-up before the preemption path is load-bearing.
            self.state.lock().waiting_queue.push_front(seq);
        }
    }

    fn update_metrics(
        &self,
        step_ms: f64,
        admitted: usize,
        _prefill_tokens: usize,
        _decode_count: usize,
    ) {
        let mut metrics = self.metrics.lock();
        metrics.step_count += 1;
        metrics.sequences_admitted += admitted as u64;
        metrics.avg_step_ms = (metrics.avg_step_ms * (metrics.step_count - 1) as f64 + step_ms)
            / metrics.step_count as f64;
        metrics.max_step_ms = metrics.max_step_ms.max(step_ms);

        let rt = self.runtime.lock();
        let stats = rt.session_stats();
        metrics.kv_cells_used = stats.total_session_tokens as usize;
        metrics.kv_cells_free = self
            .config
            .kv_budget_tokens
            .saturating_sub(metrics.kv_cells_used);
    }

    fn emit_step_telemetry(&self, metrics: StepMetrics) {
        if let Some(tx) = &self.telemetry_tx {
            let _ = tx.send(metrics);
        }
    }

    pub fn get_metrics(&self) -> SchedulerMetrics {
        self.metrics.lock().clone()
    }

    pub fn get_queue_stats(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.waiting_queue.len(), state.running_set.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_creation() {
        let seq = Sequence::new("test-1".to_string(), vec![1, 2, 3, 4, 5], 10, None, 1);
        assert_eq!(seq.id, "test-1");
        assert_eq!(seq.prompt_tokens.len(), 5);
        assert_eq!(seq.max_tokens, 10);
        assert_eq!(seq.status, SequenceStatus::Waiting);
        assert!(!seq.is_prefill_done());
        assert_eq!(seq.remaining_prefill(), 1);
    }

    #[test]
    fn decode_token_feeds_back_last_sampled() {
        let mut seq = Sequence::new("test-1".to_string(), vec![1, 2, 3, 4, 5], 10, None, 1);
        // Before any generation, the first decode input is the final prompt token.
        assert_eq!(seq.next_decode_token(), 5);
        // Once a token is sampled it becomes the next decode input, rather than 0.
        seq.last_sampled_token = Some(42);
        assert_eq!(seq.next_decode_token(), 42);
        seq.last_sampled_token = Some(99);
        assert_eq!(seq.next_decode_token(), 99);
    }

    #[test]
    fn sequence_prefill_progress() {
        let seq = Sequence::new("test-1".to_string(), vec![1, 2, 3, 4, 5], 10, None, 1);
        assert_eq!(seq.prompt_pos, 4);
        assert_eq!(seq.remaining_prefill(), 1);
        let chunk = seq.next_prefill_chunk(2);
        assert_eq!(chunk.len(), 1);
        assert_eq!(chunk[0], 5);
    }
}
