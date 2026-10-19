//! Execution cost and outstanding work, independent of request wall time.
//!
//! A load estimate is advisory: unknown batch shapes, EOS and cache hits make
//! it unsuitable for rejecting accepted work. The admission queue still enforces
//! its real deadline, capacity, cancellation and fairness guarantees.
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) struct GenerationAdmissionWork {
    pub prompt_tokens: u64,
    pub decode_tokens: u64,
}

impl GenerationAdmissionWork {
    pub(in crate::frontend) fn new(prompt_tokens: usize, decode_tokens: u32) -> Self {
        Self {
            prompt_tokens: prompt_tokens as u64,
            decode_tokens: u64::from(decode_tokens),
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            decode_tokens: self.decode_tokens.saturating_add(other.decode_tokens),
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens.saturating_sub(other.prompt_tokens),
            decode_tokens: self.decode_tokens.saturating_sub(other.decode_tokens),
        }
    }
}

#[derive(Default)]
struct ExecutionRate {
    ewma: Option<f64>,
    samples: VecDeque<f64>,
}

impl ExecutionRate {
    fn observe(&mut self, tokens: u64, elapsed: Duration) {
        if tokens == 0 || elapsed.is_zero() {
            return;
        }
        let sample = elapsed.as_secs_f64() * 1000.0 / tokens as f64;
        self.ewma = Some(
            self.ewma
                .map_or(sample, |previous| previous * 0.75 + sample * 0.25),
        );
        if self.samples.len() == 64 {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    fn conservative(&self) -> Option<f64> {
        let mut sorted: Vec<_> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let p95 = sorted.get((sorted.len() * 95).div_ceil(100).saturating_sub(1))?;
        Some(self.ewma.unwrap_or(*p95).max(*p95))
    }
}

#[derive(Default)]
struct ServiceState {
    active: BTreeMap<String, GenerationAdmissionWork>,
    queued: GenerationAdmissionWork,
    prefill: ExecutionRate,
    decode: ExecutionRate,
}

#[derive(Default)]
pub(in crate::frontend) struct GenerationServiceEstimator {
    state: Mutex<ServiceState>,
}

impl GenerationServiceEstimator {
    pub(in crate::frontend) fn new(_concurrency: usize) -> Self {
        Self::default()
    }

    /// Remaining execution load, not a promise about an individual queue slot.
    pub(in crate::frontend) fn predicted_wait_ms(&self) -> Option<f64> {
        let state = self.state.lock().ok()?;
        let work = state
            .active
            .values()
            .fold(state.queued, |sum, work| sum.add(*work));
        let cost = |tokens, rate: &ExecutionRate| {
            if tokens == 0 {
                Some(0.0)
            } else {
                Some(tokens as f64 * rate.conservative()?)
            }
        };
        Some(cost(work.prompt_tokens, &state.prefill)? + cost(work.decode_tokens, &state.decode)?)
    }

    pub(in crate::frontend) fn estimated_prefill_ms(&self, tokens: usize) -> Option<f64> {
        self.state
            .lock()
            .ok()?
            .prefill
            .ewma
            .map(|rate| rate * tokens as f64)
    }

    pub(in crate::frontend) fn reserve_queued(
        self: &Arc<Self>,
        work: GenerationAdmissionWork,
    ) -> GenerationQueuedWorkReservation {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.queued = state.queued.add(work);
        GenerationQueuedWorkReservation {
            estimator: self.clone(),
            work,
            queued: true,
        }
    }

    pub(in crate::frontend) fn start_active(
        self: &Arc<Self>,
        work: GenerationAdmissionWork,
        session: &str,
    ) -> GenerationActiveWorkReservation {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .insert(session.to_owned(), work);
        GenerationActiveWorkReservation {
            estimator: self.clone(),
            session: session.to_owned(),
        }
    }

    /// Called only for completed native execution epochs, excluding lock/queue waits.
    /// An epoch may span several asynchronous batches. Mixed epochs never train
    /// a pure-phase rate, even when the final batch itself contains only decode.
    pub(in crate::frontend) fn observe_execution(
        &self,
        work: GenerationAdmissionWork,
        elapsed: Duration,
    ) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match (work.prompt_tokens, work.decode_tokens) {
            (tokens, 0) => state.prefill.observe(tokens, elapsed),
            (0, tokens) => state.decode.observe(tokens, elapsed),
            _ => {}
        }
    }

    pub(in crate::frontend) fn advance(&self, session: &str, work: GenerationAdmissionWork) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(remaining) = state.active.get_mut(session) {
            *remaining = remaining.subtract(work);
        }
    }
}

pub(in crate::frontend) struct GenerationQueuedWorkReservation {
    estimator: Arc<GenerationServiceEstimator>,
    work: GenerationAdmissionWork,
    queued: bool,
}

impl GenerationQueuedWorkReservation {
    pub(in crate::frontend) fn promote(mut self, session: &str) -> GenerationActiveWorkReservation {
        {
            let mut state = self
                .estimator
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.queued = state.queued.subtract(self.work);
            state.active.insert(session.to_owned(), self.work);
        }
        self.queued = false;
        GenerationActiveWorkReservation {
            estimator: self.estimator.clone(),
            session: session.to_owned(),
        }
    }
}

impl Drop for GenerationQueuedWorkReservation {
    fn drop(&mut self) {
        if self.queued {
            let mut state = self
                .estimator
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.queued = state.queued.subtract(self.work);
        }
    }
}

pub(in crate::frontend) struct GenerationActiveWorkReservation {
    estimator: Arc<GenerationServiceEstimator>,
    session: String,
}

impl Drop for GenerationActiveWorkReservation {
    fn drop(&mut self) {
        self.estimator
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .remove(&self.session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_batch_is_counted_once_without_dividing_throughput_by_lanes_again() {
        let estimator = Arc::new(GenerationServiceEstimator::new(4));
        estimator.observe_execution(
            GenerationAdmissionWork::new(0, 4),
            Duration::from_millis(20),
        );
        let active = estimator.start_active(GenerationAdmissionWork::new(0, 8), "request");
        assert_eq!(estimator.predicted_wait_ms(), Some(40.0));
        estimator.advance("request", GenerationAdmissionWork::new(0, 4));
        assert_eq!(estimator.predicted_wait_ms(), Some(20.0));
        drop(active);
        assert_eq!(estimator.predicted_wait_ms(), Some(0.0));
    }

    #[test]
    fn cached_prefill_progress_does_not_train_a_false_execution_rate() {
        let estimator = Arc::new(GenerationServiceEstimator::new(4));
        estimator.observe_execution(
            GenerationAdmissionWork::new(100, 0),
            Duration::from_millis(25),
        );
        let queued = estimator.reserve_queued(GenerationAdmissionWork::new(1000, 0));
        let active = queued.promote("cached");
        estimator.advance("cached", GenerationAdmissionWork::new(900, 0));
        assert_eq!(estimator.predicted_wait_ms(), Some(25.0));
        assert_eq!(estimator.estimated_prefill_ms(40), Some(10.0));
        drop(active);
        let queued = estimator.reserve_queued(GenerationAdmissionWork::new(1000, 0));
        drop(queued);
        assert_eq!(estimator.predicted_wait_ms(), Some(0.0));
    }

    #[test]
    fn long_prefill_cannot_become_a_one_token_decode_sample() {
        let estimator = GenerationServiceEstimator::new(4);
        estimator.observe_execution(
            GenerationAdmissionWork::new(88_389, 0),
            Duration::from_secs(57),
        );
        estimator.observe_execution(
            GenerationAdmissionWork::new(0, 1),
            Duration::from_millis(17),
        );
        let state = estimator.state.lock().unwrap();
        assert_eq!(state.decode.conservative(), Some(17.0));
    }

    #[test]
    fn mixed_execution_does_not_double_charge_phases_and_unknown_cost_stays_unknown() {
        let estimator = Arc::new(GenerationServiceEstimator::new(4));
        let _active = estimator.start_active(GenerationAdmissionWork::new(512, 1), "mixed");
        estimator.observe_execution(GenerationAdmissionWork::new(512, 1), Duration::from_secs(1));
        assert_eq!(estimator.predicted_wait_ms(), None);
        estimator.advance("mixed", GenerationAdmissionWork::new(512, 1));
        assert_eq!(estimator.predicted_wait_ms(), Some(0.0));
    }
}
