use super::adapter::ObservationAdapter;
use super::hub::{ObservationHub, SUBSCRIBER_QUEUE};
use super::proto;
use proto::inference_observation_event::Kind;
use skippy_events::usage::TokenUsage;
use skippy_inference_api::{
    InferenceBackendOperation as Op, InferenceFrontendRoute, InferenceLifecycleContext,
    InferenceLifecycleEvent as Event, InferenceLifecycleObserver, InferenceRequestMethod,
    InferenceTerminalResult, InferenceUsage, RequestId, generate_request_id,
};
use std::sync::Arc;

fn context(request: RequestId) -> InferenceLifecycleContext {
    InferenceLifecycleContext::new(
        request,
        InferenceRequestMethod::Post,
        InferenceFrontendRoute::ChatCompletions,
    )
}

struct Frontend {
    hub: Arc<ObservationHub>,
    adapter: ObservationAdapter,
}

impl Frontend {
    fn new(model: &str) -> Self {
        let hub = Arc::new(ObservationHub::new());
        let adapter = ObservationAdapter::new(hub.clone(), model);
        Self { hub, adapter }
    }

    fn see(&self, event: Event) {
        self.adapter.observe(&event);
    }

    fn dispatched(&self, request: RequestId, operation: Op) {
        self.see(Event::BackendDispatched {
            context: context(request),
            operation,
        });
    }

    fn kinds(&self, subscriber: u64) -> Vec<(u64, &'static str)> {
        let drained = self.hub.drain(subscriber, usize::MAX).unwrap();
        drained
            .events
            .iter()
            .map(|event| (event.execution_id, kind_name(event.kind.as_ref().unwrap())))
            .collect()
    }
}

fn kind_name(kind: &Kind) -> &'static str {
    match kind {
        Kind::Started(_) => "started",
        Kind::FirstOutput(_) => "first_output",
        Kind::ComputeFinished(_) => "compute_finished",
        Kind::Usage(_) => "usage",
        Kind::Ended(_) => "ended",
        Kind::ModelReady(_) => "model_ready",
        Kind::ModelUnloaded(_) => "model_unloaded",
    }
}

fn completed() -> InferenceTerminalResult {
    InferenceTerminalResult::Completed { status_code: 200 }
}

#[test]
fn a_streamed_request_reports_its_stages_once_each_and_compute_ends_with_the_stream() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, snapshot) = frontend.hub.subscribe();
    assert_eq!(snapshot.ready_models, ["qwen"]);
    let request = generate_request_id();

    frontend.dispatched(request, Op::ChatCompletionStream);
    // The backend call returns as the stream opens: not the end of compute.
    frontend.see(Event::BackendTerminal {
        context: context(request),
        operation: Op::ChatCompletionStream,
        result: completed(),
    });
    frontend.see(Event::StreamFirstItem {
        context: context(request),
        operation: Op::ChatCompletionStream,
    });
    frontend.see(Event::StreamFirstItem {
        context: context(request),
        operation: Op::ChatCompletionStream,
    });
    frontend.see(Event::ResponseCompleted {
        context: context(request),
        operation: Op::ChatCompletionStream,
        usage: InferenceUsage {
            prompt_tokens: 10,
            cached_tokens: None,
            completion_tokens: 42,
            total_tokens: 52,
        },
    });
    // The terminal's usage repeats the same work: counted once.
    frontend.see(Event::StreamTerminal {
        context: context(request),
        result: InferenceTerminalResult::CompletedWithUsage {
            status_code: 200,
            usage: TokenUsage {
                completion_tokens: Some(42),
                ..TokenUsage::default()
            },
        },
    });
    frontend.see(Event::StreamDropped {
        context: context(request),
    });

    assert_eq!(
        frontend.kinds(subscriber),
        [
            (1, "started"),
            (1, "first_output"),
            (1, "usage"),
            (1, "compute_finished"),
            (1, "ended"),
        ]
    );
}

#[test]
fn a_non_streamed_request_finishes_compute_when_the_backend_returns() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, _) = frontend.hub.subscribe();
    let request = generate_request_id();
    frontend.dispatched(request, Op::ChatCompletion);
    frontend.see(Event::BackendTerminal {
        context: context(request),
        operation: Op::ChatCompletion,
        result: completed(),
    });
    frontend.see(Event::NonStreamTerminal {
        context: context(request),
        result: completed(),
        exchange_id: None,
    });
    frontend.see(Event::NonStreamTerminal {
        context: context(request),
        result: completed(),
        exchange_id: None,
    });
    assert_eq!(
        frontend.kinds(subscriber),
        [(1, "started"), (1, "compute_finished"), (1, "ended")]
    );
}

#[test]
fn listing_models_and_counting_tokens_are_not_executions() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, _) = frontend.hub.subscribe();
    for operation in [Op::Models, Op::MessagesCountTokens] {
        let request = generate_request_id();
        frontend.dispatched(request, operation);
        frontend.see(Event::NonStreamTerminal {
            context: context(request),
            result: completed(),
            exchange_id: None,
        });
    }
    // A request rejected before dispatch never ran either.
    frontend.see(Event::Rejected {
        context: context(generate_request_id()),
        status_code: 400,
        rejection: skippy_inference_api::InferenceRejection::InvalidRequest,
    });
    assert!(frontend.kinds(subscriber).is_empty());
}

#[test]
fn concurrent_requests_keep_their_own_executions_and_outcomes() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, _) = frontend.hub.subscribe();
    let (a, b) = (generate_request_id(), generate_request_id());
    frontend.dispatched(a, Op::ChatCompletionStream);
    frontend.dispatched(b, Op::Completion);
    frontend.see(Event::StreamCancelled {
        context: context(a),
    });
    frontend.see(Event::NonStreamTerminal {
        context: context(b),
        result: InferenceTerminalResult::Failed {
            status_code: 500,
            failure: skippy_inference_api::InferenceFailure::Backend,
        },
        exchange_id: None,
    });
    let drained = frontend.hub.drain(subscriber, usize::MAX).unwrap();
    let outcomes = drained
        .events
        .iter()
        .filter_map(|event| match event.kind.as_ref()? {
            Kind::Ended(ended) => Some((event.execution_id, ended.outcome())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes,
        [
            (1, proto::InferenceExecutionOutcome::Cancelled),
            (2, proto::InferenceExecutionOutcome::Failed),
        ]
    );
    let sequences = drained
        .events
        .iter()
        .map(|event| event.sequence)
        .collect::<Vec<_>>();
    assert!(sequences.windows(2).all(|pair| pair[1] == pair[0] + 1));
    let times = drained.events.iter().map(|event| event.at_unix_ns);
    assert!(times.clone().zip(times.skip(1)).all(|(a, b)| a <= b));
}

#[test]
fn a_long_stream_stays_in_every_snapshot_until_it_ends() {
    let frontend = Frontend::new("qwen");
    let request = generate_request_id();
    frontend.dispatched(request, Op::ResponsesStream);
    for _ in 0..3 {
        let (subscriber, _, snapshot) = frontend.hub.subscribe();
        assert_eq!(snapshot.active.len(), 1);
        assert!(snapshot.active[0].streaming);
        assert_eq!(snapshot.active[0].model, "qwen");
        frontend.hub.unsubscribe(subscriber);
    }
    frontend.see(Event::StreamTerminal {
        context: context(request),
        result: completed(),
    });
    let (_, _, snapshot) = frontend.hub.subscribe();
    assert!(snapshot.active.is_empty());
}

#[test]
fn a_slow_subscriber_loses_events_visibly_and_resyncs_with_what_still_runs() {
    let frontend = Frontend::new("qwen");
    let (slow, _, _) = frontend.hub.subscribe();
    let (fast, _, _) = frontend.hub.subscribe();
    let running = generate_request_id();
    frontend.dispatched(running, Op::ChatCompletionStream);
    for _ in 0..SUBSCRIBER_QUEUE {
        let request = generate_request_id();
        frontend.dispatched(request, Op::ChatCompletion);
        frontend.see(Event::NonStreamTerminal {
            context: context(request),
            result: completed(),
            exchange_id: None,
        });
        // The fast one keeps up.
        frontend.hub.drain(fast, usize::MAX).unwrap();
    }

    let drained = frontend.hub.drain(slow, usize::MAX).unwrap();
    assert!(drained.events.is_empty(), "the hole is not papered over");
    let loss = drained.loss.expect("the loss is reported");
    // Sequence 1 was the model becoming ready, before anyone subscribed.
    assert_eq!(loss.first_lost_sequence, 2);
    // The running stream's start, then three events per finished request.
    assert_eq!(loss.lost_events, 1 + 3 * SUBSCRIBER_QUEUE as u64);
    let resync = drained.resync.expect("a fresh snapshot follows");
    assert_eq!(resync.active.len(), 1, "the running stream is still there");
    assert_eq!(resync.next_sequence, 3 + 3 * SUBSCRIBER_QUEUE as u64);

    // After the resync the slow one gets events again.
    frontend.see(Event::StreamTerminal {
        context: context(running),
        result: completed(),
    });
    assert_eq!(
        frontend.kinds(slow),
        [(1, "compute_finished"), (1, "ended")]
    );
}

#[test]
fn a_watermark_covers_every_event_already_stamped() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, snapshot) = frontend.hub.subscribe();
    let request = generate_request_id();
    frontend.dispatched(request, Op::ChatCompletion);
    let drained = frontend.hub.drain(subscriber, usize::MAX).unwrap();
    let event = &drained.events[0];
    assert!(event.at_unix_ns >= snapshot.taken_at_unix_ns);
    assert!(drained.watermark_unix_ns >= event.at_unix_ns);
    // Nothing later can be stamped at or before that watermark.
    frontend.see(Event::NonStreamTerminal {
        context: context(request),
        result: completed(),
        exchange_id: None,
    });
    let later = frontend.hub.drain(subscriber, usize::MAX).unwrap();
    assert!(later.events[0].at_unix_ns >= drained.watermark_unix_ns);
}

#[test]
fn a_batch_cut_short_does_not_claim_the_events_still_queued() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, _) = frontend.hub.subscribe();
    for _ in 0..3 {
        frontend.dispatched(generate_request_id(), Op::ChatCompletion);
    }
    let first = frontend.hub.drain(subscriber, 2).unwrap();
    assert_eq!(first.events.len(), 2);
    let rest = frontend.hub.drain(subscriber, 2).unwrap();
    let left_behind = &rest.events[0];
    assert!(
        first.watermark_unix_ns < left_behind.at_unix_ns,
        "the first batch's watermark must not cover an event it did not carry"
    );
    assert!(first.watermark_unix_ns >= first.events[1].at_unix_ns);
    assert!(rest.watermark_unix_ns >= left_behind.at_unix_ns);
}

#[test]
fn a_model_is_ready_while_a_frontend_serves_it() {
    let hub = Arc::new(ObservationHub::new());
    let (subscriber, _, _) = hub.subscribe();
    let first = ObservationAdapter::new(hub.clone(), "qwen");
    let second = ObservationAdapter::new(hub.clone(), "qwen");
    drop(first);
    assert_eq!(hub.subscribe().2.ready_models, ["qwen"]);
    drop(second);
    assert!(hub.subscribe().2.ready_models.is_empty());
    let kinds = hub
        .drain(subscriber, usize::MAX)
        .unwrap()
        .events
        .into_iter()
        .map(|event| match event.kind.unwrap() {
            Kind::ModelReady(model) => format!("ready {model}"),
            Kind::ModelUnloaded(model) => format!("unloaded {model}"),
            other => panic!("unexpected {other:?}"),
        })
        .collect::<Vec<_>>();
    // Once per model, not per frontend.
    assert_eq!(kinds, ["ready qwen", "unloaded qwen"]);
}

#[test]
fn an_unsubscribed_subscriber_gets_nothing_more() {
    let frontend = Frontend::new("qwen");
    let (subscriber, _, _) = frontend.hub.subscribe();
    frontend.hub.unsubscribe(subscriber);
    frontend.dispatched(generate_request_id(), Op::ChatCompletion);
    assert!(frontend.hub.drain(subscriber, usize::MAX).is_none());
    assert_eq!(frontend.hub.subscriber_count(), 0);
}
