//! Energy a node spends serving, measured from its own GPU power sensor (#2320).
//!
//! A sampler reads board power about once a second ([`super::energy_sampler`])
//! and the inference lifecycle reports when requests run and how many tokens
//! they generate. Each interval between two power samples is counted as busy
//! when a request ran during any part of it, and as idle otherwise. That gives
//! two numbers an operator can price with: the idle draw, which keeping the
//! node on costs whether or not it serves, and the energy per 1,000 output
//! tokens on top of that idle draw, which serving adds.
//!
//! The per-token figures only use requests whose whole run was measured: each
//! busy interval's energy is shared among the requests that ran in it, and a
//! request's tokens count together with its share, or not at all when part of
//! its run fell in a sampling gap or after the sampler went quiet. A request
//! that ends without output tokens, failed or cancelled, is left out too.
//! Tokens and energy therefore always describe the same work.
//!
//! Board power covers the GPU only, not the CPU, memory or power supply, so the
//! figures are estimates and are labelled with their source.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde::Serialize;
use skippy_inference_api::{
    InferenceBackendOperation, InferenceLifecycleEvent, InferenceLifecycleObserver,
    InferenceTerminalResult, RequestId,
};

/// Samples further apart than this leave a gap rather than an interval: the
/// sampler stalled or restarted, and nothing is known about the time between.
const MAX_SAMPLE_GAP_SECONDS: f64 = 10.0;
/// Idle draw is reported once this much idle time has been measured.
const MIN_IDLE_SECONDS: f64 = 5.0;

/// One request's share of the busy energy, kept until it can be rated.
#[derive(Debug, Default)]
struct RequestEnergy {
    joules: f64,
    seconds: f64,
    /// Output tokens, from whichever event reports them first.
    tokens: Option<u64>,
    /// Part of its run was not measured: a sampling gap, or no recent sample
    /// when it started or ended.
    unmeasured: bool,
    /// The backend is still computing for it.
    computing: bool,
    /// Its terminal event arrived.
    closed: bool,
}

/// Accumulates power samples and request activity into busy and idle energy.
#[derive(Debug, Default)]
pub(crate) struct EnergyMeter {
    last_sample: Option<(f64, f64)>,
    requests: HashMap<RequestId, RequestEnergy>,
    /// Requests that ran during the interval since the last sample.
    active: HashSet<RequestId>,
    busy_seconds: f64,
    idle_joules: f64,
    idle_seconds: f64,
    /// Energy, busy time and tokens of the fully measured requests only.
    rated_joules: f64,
    rated_seconds: f64,
    output_tokens: u64,
}

/// What [`EnergyMeter`] has measured so far.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EnergyStatusPayload {
    /// Where power readings come from, such as `nvidia-smi`.
    pub source: &'static str,
    /// Mean board power while no request ran, once enough idle time is measured.
    pub idle_watts: Option<f64>,
    /// Energy per 1,000 output tokens on top of the idle draw.
    pub wh_per_1k_output_tokens: Option<f64>,
    /// Energy per 1,000 output tokens including the idle draw during requests.
    pub wh_per_1k_output_tokens_total: Option<f64>,
    /// Output tokens of the requests whose whole run was measured. Both
    /// per-token rates divide those requests' share of the busy energy by these.
    pub output_tokens: u64,
    /// All measured busy time, including requests left out of the rates.
    pub busy_seconds: f64,
    pub idle_seconds: f64,
}

impl EnergyMeter {
    /// One power reading, `watts` summed over the GPUs, at `seconds` on a
    /// monotonic clock.
    pub(crate) fn record_power(&mut self, seconds: f64, watts: f64) {
        if !watts.is_finite() || watts < 0.0 {
            return;
        }
        if let Some((previous_seconds, previous_watts)) = self.last_sample {
            let elapsed = seconds - previous_seconds;
            if elapsed <= 0.0 {
                return;
            }
            if elapsed <= MAX_SAMPLE_GAP_SECONDS {
                let joules = (previous_watts + watts) / 2.0 * elapsed;
                if self.active.is_empty() {
                    self.idle_joules += joules;
                    self.idle_seconds += elapsed;
                } else {
                    self.busy_seconds += elapsed;
                    let share = 1.0 / self.active.len() as f64;
                    for request in &self.active {
                        if let Some(energy) = self.requests.get_mut(request) {
                            energy.joules += joules * share;
                            energy.seconds += elapsed * share;
                        }
                    }
                }
            } else {
                // Nothing is known about the gap, so neither is what the
                // requests running through it cost.
                for request in &self.active {
                    if let Some(energy) = self.requests.get_mut(request) {
                        energy.unmeasured = true;
                    }
                }
            }
        }
        self.last_sample = Some((seconds, watts));
        // A request that stopped computing has had its last interval counted.
        let settled: Vec<RequestId> = self
            .active
            .iter()
            .filter(|request| {
                self.requests
                    .get(request)
                    .is_none_or(|energy| !energy.computing)
            })
            .copied()
            .collect();
        for request in settled {
            self.active.remove(&request);
            if self
                .requests
                .get(&request)
                .is_some_and(|energy| energy.closed)
            {
                self.rate(&request);
            }
        }
    }

    pub(crate) fn request_started(&mut self, request: RequestId, seconds: f64) {
        let measured = self.sampled_recently(seconds);
        let energy = self.requests.entry(request).or_default();
        energy.computing = true;
        energy.unmeasured |= !measured;
        self.active.insert(request);
    }

    /// The backend stopped computing for `request`. The interval this happens
    /// in stays busy: part of it was.
    pub(crate) fn compute_finished(&mut self, request: &RequestId) {
        if let Some(energy) = self.requests.get_mut(request) {
            energy.computing = false;
        }
    }

    /// `request` is over: no later event can report its tokens again. It is
    /// rated once the interval it ended in is counted, which the next sample
    /// does; with no recent sample, none may come, and it is left out.
    pub(crate) fn request_closed(&mut self, request: &RequestId, seconds: f64) {
        let measured = self.sampled_recently(seconds);
        let Some(energy) = self.requests.get_mut(request) else {
            return;
        };
        energy.computing = false;
        energy.closed = true;
        energy.unmeasured |= !measured;
        if !measured || !self.active.contains(request) {
            self.active.remove(request);
            self.rate(request);
        }
    }

    /// Keeps a request's output tokens, whichever event reports them first.
    pub(crate) fn tokens_completed(&mut self, request: RequestId, completion_tokens: u64) {
        if let Some(energy) = self.requests.get_mut(&request) {
            energy.tokens.get_or_insert(completion_tokens);
        }
    }

    /// Adds a finished request to the rates if its whole run was measured.
    fn rate(&mut self, request: &RequestId) {
        let Some(energy) = self.requests.remove(request) else {
            return;
        };
        if let Some(tokens) = energy.tokens
            && !energy.unmeasured
        {
            self.rated_joules += energy.joules;
            self.rated_seconds += energy.seconds;
            self.output_tokens += tokens;
        }
    }

    fn sampled_recently(&self, seconds: f64) -> bool {
        self.last_sample
            .is_some_and(|(sampled, _)| seconds - sampled <= MAX_SAMPLE_GAP_SECONDS)
    }

    pub(crate) fn status(&self, source: &'static str) -> EnergyStatusPayload {
        let idle_watts =
            (self.idle_seconds >= MIN_IDLE_SECONDS).then(|| self.idle_joules / self.idle_seconds);
        let per_1k_tokens = |joules: f64| {
            (self.output_tokens > 0).then(|| joules / 3600.0 / (self.output_tokens as f64 / 1000.0))
        };
        EnergyStatusPayload {
            source,
            idle_watts,
            wh_per_1k_output_tokens: idle_watts.and_then(|idle| {
                per_1k_tokens((self.rated_joules - idle * self.rated_seconds).max(0.0))
            }),
            wh_per_1k_output_tokens_total: per_1k_tokens(self.rated_joules),
            output_tokens: self.output_tokens,
            busy_seconds: self.busy_seconds,
            idle_seconds: self.idle_seconds,
        }
    }
}

struct Energy {
    meter: Mutex<EnergyMeter>,
    started: Instant,
    source: OnceLock<&'static str>,
}

fn energy() -> &'static Energy {
    static ENERGY: OnceLock<Energy> = OnceLock::new();
    ENERGY.get_or_init(|| Energy {
        meter: Mutex::new(EnergyMeter::default()),
        started: Instant::now(),
        source: OnceLock::new(),
    })
}

fn with_meter<T>(f: impl FnOnce(&mut EnergyMeter) -> T) -> T {
    let mut meter = energy()
        .meter
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut meter)
}

/// Feeds the process-wide meter from the power sampler.
pub(crate) fn record_power(source: &'static str, watts: f64) {
    let energy = energy();
    let _ = energy.source.set(source);
    let seconds = energy.started.elapsed().as_secs_f64();
    with_meter(|meter| meter.record_power(seconds, watts));
}

/// `None` until a power sampler has produced a reading.
pub fn energy_status() -> Option<EnergyStatusPayload> {
    let source = *energy().source.get()?;
    Some(with_meter(|meter| meter.status(source)))
}

/// Lifecycle observer that tells the meter when requests run and what they
/// generate. It only takes a lock and never blocks on I/O.
pub(crate) struct EnergyLifecycleObserver;

impl InferenceLifecycleObserver for EnergyLifecycleObserver {
    fn observe(&self, event: &InferenceLifecycleEvent) {
        // Tracked before the first power reading too: a request already
        // running then makes the first interval busy. Only a request's own
        // terminal event retires it: no event marks progress within a stream,
        // so its age can't tell a long generation from a lost one. Closing
        // without a recent sample drops it, so nothing piles up on a node with
        // no sampler.
        let energy = energy();
        let seconds = energy.started.elapsed().as_secs_f64();
        with_meter(|meter| apply_event(meter, event, seconds));
    }
}

pub(crate) fn energy_lifecycle_observer() -> Arc<dyn InferenceLifecycleObserver> {
    Arc::new(EnergyLifecycleObserver)
}

/// Listing models or counting tokens doesn't run the model, so it never makes
/// an interval busy; a client polling `/v1/models` would otherwise do so.
fn runs_model(operation: InferenceBackendOperation) -> bool {
    !matches!(
        operation,
        InferenceBackendOperation::Models | InferenceBackendOperation::MessagesCountTokens
    )
}

/// For a streamed operation the backend call only opens the stream: its
/// terminal event comes before the first token, and generation goes on until
/// the stream's own terminal event.
fn opens_stream(operation: InferenceBackendOperation) -> bool {
    matches!(
        operation,
        InferenceBackendOperation::ChatCompletionStream
            | InferenceBackendOperation::CompletionStream
            | InferenceBackendOperation::ResponsesStream
            | InferenceBackendOperation::MessagesStream
    )
}

fn apply_event(meter: &mut EnergyMeter, event: &InferenceLifecycleEvent, seconds: f64) {
    match event {
        InferenceLifecycleEvent::BackendDispatched { context, operation } => {
            if runs_model(*operation) {
                meter.request_started(context.request_id, seconds);
            }
        }
        InferenceLifecycleEvent::ResponseCompleted { context, usage, .. } => {
            meter.tokens_completed(context.request_id, u64::from(usage.completion_tokens));
        }
        InferenceLifecycleEvent::StreamTerminal { context, result } => {
            if let InferenceTerminalResult::CompletedWithUsage { usage, .. } = result
                && let Some(tokens) = usage.completion_tokens
            {
                meter.tokens_completed(context.request_id, tokens);
            }
            meter.request_closed(&context.request_id, seconds);
        }
        InferenceLifecycleEvent::BackendTerminal {
            context,
            operation,
            result,
        } => {
            let failed = matches!(result, InferenceTerminalResult::Failed { .. });
            if failed || !opens_stream(*operation) {
                meter.compute_finished(&context.request_id);
            }
        }
        // Every way a request can end; closing one twice is harmless.
        InferenceLifecycleEvent::NonStreamTerminal { context, .. }
        | InferenceLifecycleEvent::StreamDropped { context, .. }
        | InferenceLifecycleEvent::StreamCancelled { context, .. }
        | InferenceLifecycleEvent::RequestCancelled { context, .. }
        | InferenceLifecycleEvent::Rejected { context, .. } => {
            meter.request_closed(&context.request_id, seconds);
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "energy_tests.rs"]
mod tests;
