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
/// A request still open after this long lost its end event; forgetting it
/// keeps one lost event from counting the node as busy for good.
const MAX_REQUEST_SECONDS: f64 = 30.0 * 60.0;

/// Accumulates power samples and request activity into busy and idle energy.
#[derive(Debug, Default)]
pub(crate) struct EnergyMeter {
    last_sample: Option<(f64, f64)>,
    in_flight: HashMap<RequestId, f64>,
    counted: HashSet<RequestId>,
    busy_since_last_sample: bool,
    busy_joules: f64,
    busy_seconds: f64,
    idle_joules: f64,
    idle_seconds: f64,
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
    pub output_tokens: u64,
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
            if elapsed > 0.0 && elapsed <= MAX_SAMPLE_GAP_SECONDS {
                let joules = (previous_watts + watts) / 2.0 * elapsed;
                if self.busy_since_last_sample {
                    self.busy_joules += joules;
                    self.busy_seconds += elapsed;
                } else {
                    self.idle_joules += joules;
                    self.idle_seconds += elapsed;
                }
            }
        }
        self.last_sample = Some((seconds, watts));
        let stale: Vec<RequestId> = self
            .in_flight
            .iter()
            .filter(|(_, started)| seconds - **started > MAX_REQUEST_SECONDS)
            .map(|(request, _)| *request)
            .collect();
        for request in &stale {
            self.request_closed(request);
        }
        self.busy_since_last_sample = !self.in_flight.is_empty();
    }

    pub(crate) fn request_started(&mut self, request: RequestId, seconds: f64) {
        self.in_flight.insert(request, seconds);
        self.busy_since_last_sample = true;
    }

    /// The backend stopped computing for `request`. The interval this happens
    /// in stays busy: part of it was.
    pub(crate) fn compute_finished(&mut self, request: &RequestId) {
        self.in_flight.remove(request);
    }

    /// `request` is over: no later event can report its tokens again.
    pub(crate) fn request_closed(&mut self, request: &RequestId) {
        self.in_flight.remove(request);
        self.counted.remove(request);
    }

    /// Counts a request's output tokens once, whichever event reports them first.
    pub(crate) fn tokens_completed(&mut self, request: RequestId, completion_tokens: u64) {
        if self.counted.insert(request) {
            self.output_tokens += completion_tokens;
        }
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
                per_1k_tokens((self.busy_joules - idle * self.busy_seconds).max(0.0))
            }),
            wh_per_1k_output_tokens_total: per_1k_tokens(self.busy_joules),
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
        // running then makes the first interval busy. Every request ends with
        // a terminal event, so nothing piles up on a node with no sampler.
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
            meter.request_closed(&context.request_id);
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
            meter.request_closed(&context.request_id);
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "energy_tests.rs"]
mod tests;
