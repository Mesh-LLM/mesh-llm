//! Where local inference observations meet their subscribers.
//!
//! The frontends' observers call in synchronously on the request path, so
//! every method here takes one short lock and never waits on a plugin. An
//! event gets its sequence number and its time under that lock, and a drain
//! reads its watermark under it too: every event stamped at or before a
//! watermark is already queued when the watermark is read.

use super::proto;
use rand::RngExt;
use skippy_inference_api::RequestId;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::Notify;

/// Events a subscriber's queue holds before the host drops them and resyncs
/// that subscriber from a snapshot.
pub(crate) const SUBSCRIBER_QUEUE: usize = 4096;

pub(crate) type SubscriberId = u64;

struct Execution {
    id: u64,
    model: String,
    operation: proto::InferenceOperation,
    streaming: bool,
    started_at_unix_ns: u64,
    first_output: bool,
    compute_finished: bool,
    usage_reported: bool,
}

struct Subscriber {
    queue: VecDeque<proto::InferenceObservationEvent>,
    /// Events dropped since the last drain: (first sequence, count).
    lost: Option<(u64, u64)>,
    notify: Arc<Notify>,
}

#[derive(Default)]
struct HubState {
    next_sequence: u64,
    next_execution: u64,
    next_subscriber: SubscriberId,
    executions: HashMap<RequestId, Execution>,
    /// Frontends serving each model.
    ready_models: BTreeMap<String, usize>,
    subscribers: HashMap<SubscriberId, Subscriber>,
}

/// What a subscriber's delivery task takes in one go.
pub(crate) struct Drained {
    pub(crate) events: Vec<proto::InferenceObservationEvent>,
    pub(crate) watermark_unix_ns: u64,
    pub(crate) loss: Option<proto::InferenceObservationLoss>,
    pub(crate) resync: Option<proto::InferenceObservationSnapshot>,
}

pub(crate) struct ObservationHub {
    host_generation: u64,
    epoch: Instant,
    epoch_unix_ns: u64,
    state: Mutex<HubState>,
}

impl Default for ObservationHub {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservationHub {
    pub(crate) fn new() -> Self {
        let epoch_unix_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or_default();
        Self {
            // Never 0, so a plugin can tell "unset" from a real generation.
            host_generation: rand::rng().random_range(1..u64::MAX),
            epoch: Instant::now(),
            epoch_unix_ns,
            state: Mutex::new(HubState {
                next_sequence: 1,
                next_execution: 1,
                next_subscriber: 1,
                ..HubState::default()
            }),
        }
    }

    pub(crate) fn host_generation(&self) -> u64 {
        self.host_generation
    }

    /// Host time, Unix nanoseconds, monotonic within this hub.
    fn now_unix_ns(&self) -> u64 {
        let elapsed = u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.epoch_unix_ns.saturating_add(elapsed)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HubState> {
        self.state
            .lock()
            .expect("inference observation hub poisoned")
    }

    pub(crate) fn subscribe(
        &self,
    ) -> (
        SubscriberId,
        Arc<Notify>,
        proto::InferenceObservationSnapshot,
    ) {
        let mut state = self.lock();
        let id = state.next_subscriber;
        state.next_subscriber += 1;
        let notify = Arc::new(Notify::new());
        state.subscribers.insert(
            id,
            Subscriber {
                queue: VecDeque::new(),
                lost: None,
                notify: notify.clone(),
            },
        );
        let snapshot = self.snapshot(&state);
        (id, notify, snapshot)
    }

    pub(crate) fn unsubscribe(&self, id: SubscriberId) {
        self.lock().subscribers.remove(&id);
    }

    #[cfg(test)]
    pub(crate) fn subscriber_count(&self) -> usize {
        self.lock().subscribers.len()
    }

    /// Up to `max` queued events, the watermark, and after a loss the loss and
    /// a fresh snapshot instead of events. `None` once unsubscribed.
    pub(crate) fn drain(&self, id: SubscriberId, max: usize) -> Option<Drained> {
        let mut state = self.lock();
        let watermark_unix_ns = self.now_unix_ns();
        let subscriber = state.subscribers.get_mut(&id)?;
        if let Some((first, count)) = subscriber.lost.take() {
            subscriber.queue.clear();
            let resync = self.snapshot(&state);
            return Some(Drained {
                events: Vec::new(),
                watermark_unix_ns,
                loss: Some(proto::InferenceObservationLoss {
                    first_lost_sequence: first,
                    lost_events: count,
                }),
                resync: Some(resync),
            });
        }
        let take = subscriber.queue.len().min(max);
        let events = subscriber.queue.drain(..take).collect();
        // Events still queued are older than now: the watermark stops just
        // before the first of them, which the next batch carries.
        let watermark_unix_ns = subscriber
            .queue
            .front()
            .map_or(watermark_unix_ns, |next| next.at_unix_ns.saturating_sub(1));
        Some(Drained {
            events,
            watermark_unix_ns,
            loss: None,
            resync: None,
        })
    }

    fn snapshot(&self, state: &HubState) -> proto::InferenceObservationSnapshot {
        let mut active = state
            .executions
            .values()
            .map(|execution| proto::ActiveInferenceExecution {
                execution_id: execution.id,
                model: execution.model.clone(),
                operation: execution.operation as i32,
                streaming: execution.streaming,
                started_at_unix_ns: execution.started_at_unix_ns,
                compute_finished: execution.compute_finished,
            })
            .collect::<Vec<_>>();
        active.sort_by_key(|execution| execution.execution_id);
        proto::InferenceObservationSnapshot {
            host_generation: self.host_generation,
            next_sequence: state.next_sequence,
            taken_at_unix_ns: self.now_unix_ns(),
            ready_models: state.ready_models.keys().cloned().collect(),
            active,
        }
    }

    fn emit(
        &self,
        state: &mut HubState,
        execution_id: u64,
        kind: proto::inference_observation_event::Kind,
    ) {
        let event = proto::InferenceObservationEvent {
            sequence: state.next_sequence,
            at_unix_ns: self.now_unix_ns(),
            execution_id,
            kind: Some(kind),
        };
        state.next_sequence += 1;
        for subscriber in state.subscribers.values_mut() {
            match &mut subscriber.lost {
                Some((_, count)) => *count += 1,
                None if subscriber.queue.len() >= SUBSCRIBER_QUEUE => {
                    let first = subscriber
                        .queue
                        .front()
                        .map_or(event.sequence, |queued| queued.sequence);
                    let count = subscriber.queue.len() as u64 + 1;
                    subscriber.queue.clear();
                    subscriber.lost = Some((first, count));
                }
                None => subscriber.queue.push_back(event.clone()),
            }
            subscriber.notify.notify_one();
        }
    }

    pub(crate) fn model_ready(&self, model: &str) {
        let mut state = self.lock();
        let frontends = state.ready_models.entry(model.to_string()).or_default();
        *frontends += 1;
        if *frontends == 1 {
            self.emit(
                &mut state,
                0,
                proto::inference_observation_event::Kind::ModelReady(model.to_string()),
            );
        }
    }

    pub(crate) fn model_unloaded(&self, model: &str) {
        let mut state = self.lock();
        let Some(frontends) = state.ready_models.get_mut(model) else {
            return;
        };
        *frontends -= 1;
        if *frontends == 0 {
            state.ready_models.remove(model);
            self.emit(
                &mut state,
                0,
                proto::inference_observation_event::Kind::ModelUnloaded(model.to_string()),
            );
        }
    }

    /// A request started running `model`. A second start for a request id
    /// already running is ignored: the frontend reports each request once.
    pub(crate) fn started(
        &self,
        request: RequestId,
        model: &str,
        operation: proto::InferenceOperation,
        streaming: bool,
    ) {
        let mut state = self.lock();
        if state.executions.contains_key(&request) {
            return;
        }
        let id = state.next_execution;
        state.next_execution += 1;
        let started_at_unix_ns = self.now_unix_ns();
        state.executions.insert(
            request,
            Execution {
                id,
                model: model.to_string(),
                operation,
                streaming,
                started_at_unix_ns,
                first_output: false,
                compute_finished: false,
                usage_reported: false,
            },
        );
        self.emit(
            &mut state,
            id,
            proto::inference_observation_event::Kind::Started(proto::InferenceExecutionStarted {
                model: model.to_string(),
                operation: operation as i32,
                streaming,
            }),
        );
    }

    pub(crate) fn first_output(&self, request: &RequestId) {
        let mut state = self.lock();
        let Some(execution) = state.executions.get_mut(request) else {
            return;
        };
        if execution.first_output || !execution.streaming {
            return;
        }
        execution.first_output = true;
        let id = execution.id;
        self.emit(
            &mut state,
            id,
            proto::inference_observation_event::Kind::FirstOutput(proto::Empty {}),
        );
    }

    pub(crate) fn compute_finished(&self, request: &RequestId) {
        let mut state = self.lock();
        let Some(execution) = state.executions.get_mut(request) else {
            return;
        };
        if execution.compute_finished {
            return;
        }
        execution.compute_finished = true;
        let id = execution.id;
        self.emit(
            &mut state,
            id,
            proto::inference_observation_event::Kind::ComputeFinished(proto::Empty {}),
        );
    }

    /// The first usage report for an execution; later ones are ignored, so
    /// no work is counted twice.
    pub(crate) fn usage(&self, request: &RequestId, usage: proto::InferenceOutputUsage) {
        let mut state = self.lock();
        let Some(execution) = state.executions.get_mut(request) else {
            return;
        };
        if execution.usage_reported {
            return;
        }
        execution.usage_reported = true;
        let id = execution.id;
        self.emit(
            &mut state,
            id,
            proto::inference_observation_event::Kind::Usage(usage),
        );
    }

    /// The execution ended; only the first end counts.
    pub(crate) fn ended(&self, request: &RequestId, outcome: proto::InferenceExecutionOutcome) {
        let mut state = self.lock();
        let Some(execution) = state.executions.remove(request) else {
            return;
        };
        self.emit(
            &mut state,
            execution.id,
            proto::inference_observation_event::Kind::Ended(proto::InferenceExecutionEnded {
                outcome: outcome as i32,
            }),
        );
    }
}
