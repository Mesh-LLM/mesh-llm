//! How long queued generation work is predicted to wait, from the work
//! admitted and queued and the service rate observed so far. Predictions
//! are advisory: the queue owns capacity and deadlines.

use skippy_inference_api::InferenceRequestTier;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

const SERVICE_RATE_EWMA_ALPHA: f64 = 0.25;
const SERVICE_RATE_WINDOW: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) struct GenerationAdmissionWork {
    pub(in crate::frontend) prompt_tokens: u64,
    pub(in crate::frontend) decode_tokens: u64,
}

impl GenerationAdmissionWork {
    pub(in crate::frontend) fn new(prompt_tokens: usize, decode_tokens: u32) -> Self {
        Self {
            prompt_tokens: u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
            decode_tokens: u64::from(decode_tokens),
        }
    }
}

#[derive(Default)]
struct GenerationServiceState {
    active: GenerationAdmissionWork,
    /// Queued work by request tier: a request waits behind active work and
    /// queued work at its own tier or higher, which is admitted first.
    queued: BTreeMap<InferenceRequestTier, GenerationAdmissionWork>,
    prefill_ms_per_token_ewma: Option<f64>,
    decode_ms_per_token_ewma: Option<f64>,
    prefill_ms_per_token_samples: VecDeque<f64>,
    decode_ms_per_token_samples: VecDeque<f64>,
}

pub(in crate::frontend) struct GenerationServiceEstimator {
    concurrency: AtomicUsize,
    state: Mutex<GenerationServiceState>,
}

impl GenerationServiceEstimator {
    pub(in crate::frontend) fn new(concurrency: usize) -> Self {
        Self {
            concurrency: AtomicUsize::new(concurrency.max(1)),
            state: Mutex::new(GenerationServiceState::default()),
        }
    }

    /// The predicted wait for a request at `tier`.
    pub(in crate::frontend) fn predicted_wait_ms(&self, tier: InferenceRequestTier) -> Option<f64> {
        let state = self.state.lock().ok()?;
        predicted_wait_ms_for_state(&state, self.concurrency.load(Ordering::Acquire), tier)
    }

    pub(in crate::frontend) fn estimated_prefill_ms(&self, tokens: usize) -> Option<f64> {
        let state = self.state.lock().ok()?;
        state
            .prefill_ms_per_token_ewma
            .map(|per_token| per_token * tokens as f64)
            .filter(|cost| cost.is_finite() && *cost >= 0.0)
    }

    pub(in crate::frontend) fn set_concurrency(&self, concurrency: usize) {
        self.concurrency
            .store(concurrency.max(1), Ordering::Release);
    }

    /// Account `work` as queued for a request at `tier`.
    pub(in crate::frontend) fn reserve_queued(
        self: &Arc<Self>,
        work: GenerationAdmissionWork,
        tier: InferenceRequestTier,
    ) -> GenerationQueuedWorkReservation {
        let Ok(mut state) = self.state.lock() else {
            return GenerationQueuedWorkReservation {
                estimator: self.clone(),
                work,
                tier,
                queued: false,
            };
        };
        // Predictions include interference and unfinished requests' original
        // work, so they are advisory. The queue owns capacity and deadlines.
        let queued = state.queued.entry(tier).or_default();
        *queued = add_work(*queued, work);
        GenerationQueuedWorkReservation {
            estimator: self.clone(),
            work,
            tier,
            queued: true,
        }
    }

    pub(in crate::frontend) fn start_active(
        self: &Arc<Self>,
        work: GenerationAdmissionWork,
    ) -> GenerationActiveWorkReservation {
        if let Ok(mut state) = self.state.lock() {
            state.active = add_work(state.active, work);
        }
        GenerationActiveWorkReservation {
            estimator: self.clone(),
            work,
        }
    }

    pub(in crate::frontend) fn observe_completed(
        &self,
        work: GenerationAdmissionWork,
        prompt_ms: f64,
        decode_ms: f64,
    ) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if work.prompt_tokens > 0 && prompt_ms.is_finite() && prompt_ms > 0.0 {
            let sample = prompt_ms / work.prompt_tokens as f64;
            state.prefill_ms_per_token_ewma =
                Some(update_ewma(state.prefill_ms_per_token_ewma, sample));
            push_sample(&mut state.prefill_ms_per_token_samples, sample);
        }
        if work.decode_tokens > 0 && decode_ms.is_finite() && decode_ms > 0.0 {
            let sample = decode_ms / work.decode_tokens as f64;
            state.decode_ms_per_token_ewma =
                Some(update_ewma(state.decode_ms_per_token_ewma, sample));
            push_sample(&mut state.decode_ms_per_token_samples, sample);
        }
    }
}

fn predicted_wait_ms_for_state(
    state: &GenerationServiceState,
    concurrency: usize,
    tier: InferenceRequestTier,
) -> Option<f64> {
    let prompt_ms_per_token = conservative_ms_per_token(
        state.prefill_ms_per_token_ewma,
        &state.prefill_ms_per_token_samples,
    );
    let decode_ms_per_token = conservative_ms_per_token(
        state.decode_ms_per_token_ewma,
        &state.decode_ms_per_token_samples,
    );
    if prompt_ms_per_token.is_none() && decode_ms_per_token.is_none() {
        return None;
    }
    let work = state
        .queued
        .range(tier..)
        .fold(state.active, |work, (_, queued)| add_work(work, *queued));
    let prompt_ms = prompt_ms_per_token.unwrap_or(0.0) * work.prompt_tokens as f64;
    let decode_ms = decode_ms_per_token.unwrap_or(0.0) * work.decode_tokens as f64;
    Some((prompt_ms + decode_ms) / concurrency.max(1) as f64)
}

pub(in crate::frontend) struct GenerationQueuedWorkReservation {
    estimator: Arc<GenerationServiceEstimator>,
    work: GenerationAdmissionWork,
    tier: InferenceRequestTier,
    queued: bool,
}

impl GenerationQueuedWorkReservation {
    pub(in crate::frontend) fn promote(mut self) -> GenerationActiveWorkReservation {
        if let Ok(mut state) = self.estimator.state.lock() {
            unqueue(&mut state, self.tier, self.work);
            state.active = add_work(state.active, self.work);
        }
        self.queued = false;
        GenerationActiveWorkReservation {
            estimator: self.estimator.clone(),
            work: self.work,
        }
    }
}

impl Drop for GenerationQueuedWorkReservation {
    fn drop(&mut self) {
        if self.queued
            && let Ok(mut state) = self.estimator.state.lock()
        {
            unqueue(&mut state, self.tier, self.work);
        }
    }
}

pub(in crate::frontend) struct GenerationActiveWorkReservation {
    estimator: Arc<GenerationServiceEstimator>,
    work: GenerationAdmissionWork,
}

impl Drop for GenerationActiveWorkReservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.estimator.state.lock() {
            state.active = subtract_work(state.active, self.work);
        }
    }
}

fn unqueue(
    state: &mut GenerationServiceState,
    tier: InferenceRequestTier,
    work: GenerationAdmissionWork,
) {
    if let Some(queued) = state.queued.get_mut(&tier) {
        *queued = subtract_work(*queued, work);
        if *queued == GenerationAdmissionWork::default() {
            state.queued.remove(&tier);
        }
    }
}

fn add_work(
    left: GenerationAdmissionWork,
    right: GenerationAdmissionWork,
) -> GenerationAdmissionWork {
    GenerationAdmissionWork {
        prompt_tokens: left.prompt_tokens.saturating_add(right.prompt_tokens),
        decode_tokens: left.decode_tokens.saturating_add(right.decode_tokens),
    }
}

fn subtract_work(
    left: GenerationAdmissionWork,
    right: GenerationAdmissionWork,
) -> GenerationAdmissionWork {
    GenerationAdmissionWork {
        prompt_tokens: left.prompt_tokens.saturating_sub(right.prompt_tokens),
        decode_tokens: left.decode_tokens.saturating_sub(right.decode_tokens),
    }
}

fn update_ewma(previous: Option<f64>, sample: f64) -> f64 {
    previous.map_or(sample, |previous| {
        previous * (1.0 - SERVICE_RATE_EWMA_ALPHA) + sample * SERVICE_RATE_EWMA_ALPHA
    })
}

fn push_sample(samples: &mut VecDeque<f64>, sample: f64) {
    if samples.len() == SERVICE_RATE_WINDOW {
        samples.pop_front();
    }
    samples.push_back(sample);
}

fn conservative_ms_per_token(ewma: Option<f64>, samples: &VecDeque<f64>) -> Option<f64> {
    let mut sorted = samples.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let p95 = if sorted.is_empty() {
        None
    } else {
        let index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
        sorted.get(index).copied()
    };
    match (ewma, p95) {
        (Some(ewma), Some(p95)) => Some(ewma.max(p95)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
mod service_estimator_tests {
    use super::*;

    #[test]
    fn high_prediction_does_not_refuse_work_accounting() {
        let estimator = Arc::new(GenerationServiceEstimator::new(1));
        let work = GenerationAdmissionWork::new(100, 100);
        estimator.observe_completed(work, 100.0, 100.0);
        let active = estimator.start_active(work);
        let local = InferenceRequestTier::Local;

        assert_eq!(estimator.predicted_wait_ms(local), Some(200.0));
        let queued = estimator.reserve_queued(work, local);
        assert_eq!(estimator.predicted_wait_ms(local), Some(400.0));
        drop(queued);
        assert_eq!(estimator.predicted_wait_ms(local), Some(200.0));
        drop(active);
        assert_eq!(estimator.predicted_wait_ms(local), Some(0.0));
    }

    #[test]
    fn queued_and_active_work_are_released_by_raii() {
        let estimator = Arc::new(GenerationServiceEstimator::new(2));
        let work = GenerationAdmissionWork::new(80, 20);
        estimator.observe_completed(work, 80.0, 20.0);
        let queued = estimator.reserve_queued(work, InferenceRequestTier::Local);
        assert_eq!(
            estimator.predicted_wait_ms(InferenceRequestTier::Local),
            Some(50.0)
        );

        let active = queued.promote();
        assert_eq!(
            estimator.predicted_wait_ms(InferenceRequestTier::Local),
            Some(50.0)
        );
        drop(active);
        assert_eq!(
            estimator.predicted_wait_ms(InferenceRequestTier::Local),
            Some(0.0)
        );
    }

    /// Queued mesh work is admitted after a local request, so it does not
    /// count toward the local request's predicted wait.
    #[test]
    fn a_request_waits_only_behind_work_at_its_tier_or_higher() {
        let estimator = Arc::new(GenerationServiceEstimator::new(1));
        let work = GenerationAdmissionWork::new(100, 100);
        estimator.observe_completed(work, 100.0, 100.0);
        let _mesh = estimator.reserve_queued(work, InferenceRequestTier::Mesh);
        let _trusted = estimator.reserve_queued(work, InferenceRequestTier::Trusted);

        assert_eq!(
            estimator.predicted_wait_ms(InferenceRequestTier::Local),
            Some(0.0)
        );
        assert_eq!(
            estimator.predicted_wait_ms(InferenceRequestTier::Trusted),
            Some(200.0)
        );
        assert_eq!(
            estimator.predicted_wait_ms(InferenceRequestTier::Mesh),
            Some(400.0)
        );
    }

    #[test]
    fn conservative_rate_uses_the_slower_p95_sample() {
        let samples = VecDeque::from([1.0, 1.0, 1.0, 10.0]);
        assert_eq!(conservative_ms_per_token(Some(1.5), &samples), Some(10.0));
    }

    #[test]
    fn prefill_estimate_uses_observed_stage_rate() {
        let estimator = GenerationServiceEstimator::new(1);
        assert_eq!(estimator.estimated_prefill_ms(128), None);
        estimator.observe_completed(GenerationAdmissionWork::new(100, 0), 25.0, 0.0);
        assert_eq!(estimator.estimated_prefill_ms(40), Some(10.0));
    }
}
