use super::*;
use skippy_events::usage::TokenUsage;
use skippy_inference_api::{
    InferenceBackendOperation, InferenceFailure, InferenceFrontendRoute, InferenceLifecycleContext,
    InferenceRequestMethod, InferenceUsage,
};

const SOURCE: &str = "test";

fn context(request: RequestId) -> InferenceLifecycleContext {
    InferenceLifecycleContext::new(
        request,
        InferenceRequestMethod::Post,
        InferenceFrontendRoute::ChatCompletions,
    )
}

fn usage(completion_tokens: u32) -> InferenceUsage {
    InferenceUsage {
        prompt_tokens: 10,
        cached_tokens: None,
        completion_tokens,
        total_tokens: 10 + completion_tokens,
    }
}

/// One sample a second from `from` to `to` inclusive, all at `watts`.
fn samples(meter: &mut EnergyMeter, from: u32, to: u32, watts: f64) {
    for second in from..=to {
        meter.record_power(f64::from(second), watts);
    }
}

#[test]
fn idle_draw_needs_enough_idle_time_and_no_tokens_means_no_rate() {
    let mut meter = EnergyMeter::default();
    samples(&mut meter, 0, 3, 10.0);
    assert_eq!(meter.status(SOURCE).idle_watts, None);
    samples(&mut meter, 4, 10, 10.0);
    let status = meter.status(SOURCE);
    assert_eq!(status.idle_watts, Some(10.0));
    assert_eq!(status.idle_seconds, 10.0);
    assert_eq!(status.wh_per_1k_output_tokens, None);
    assert_eq!(status.wh_per_1k_output_tokens_total, None);
}

#[test]
fn serving_energy_is_split_into_idle_draw_and_what_tokens_add() {
    let mut meter = EnergyMeter::default();
    let request = RequestId::new();
    samples(&mut meter, 0, 10, 10.0);
    meter.request_started(request, 10.0);
    // [10, 11] averages 10 W and 110 W; then nine seconds at 110 W.
    samples(&mut meter, 11, 19, 110.0);
    meter.tokens_completed(request, 1_000);
    meter.request_closed(&request, 19.5);
    // The interval it ended in is counted, then it is rated.
    meter.record_power(20.0, 110.0);

    let status = meter.status(SOURCE);
    assert_eq!(status.idle_watts, Some(10.0));
    assert_eq!(status.busy_seconds, 10.0);
    assert_eq!(status.output_tokens, 1_000);
    let total = status.wh_per_1k_output_tokens_total.unwrap();
    let above_idle = status.wh_per_1k_output_tokens.unwrap();
    assert!((total - 1_050.0 / 3_600.0).abs() < 1e-9, "{total}");
    assert!((above_idle - 950.0 / 3_600.0).abs() < 1e-9, "{above_idle}");
}

#[test]
fn an_interval_a_request_touched_counts_as_busy() {
    let mut meter = EnergyMeter::default();
    let request = RequestId::new();
    meter.record_power(0.0, 50.0);
    meter.request_started(request, 0.2);
    meter.request_closed(&request, 0.3);
    meter.record_power(1.0, 50.0);
    meter.record_power(2.0, 50.0);
    let status = meter.status(SOURCE);
    assert_eq!(status.busy_seconds, 1.0);
    assert_eq!(status.idle_seconds, 1.0);
}

#[test]
fn gaps_and_impossible_readings_are_not_integrated() {
    let mut meter = EnergyMeter::default();
    meter.record_power(0.0, 10.0);
    meter.record_power(60.0, 10.0);
    meter.record_power(61.0, f64::NAN);
    meter.record_power(61.0, -5.0);
    meter.record_power(62.0, 10.0);
    assert_eq!(meter.status(SOURCE).idle_seconds, 2.0);
}

/// A rate from one request: 1,000 tokens over ten seconds at 110 W, after ten
/// idle seconds at 10 W.
fn meter_with_a_rate() -> EnergyMeter {
    let mut meter = EnergyMeter::default();
    let request = RequestId::new();
    samples(&mut meter, 0, 10, 10.0);
    meter.request_started(request, 10.0);
    samples(&mut meter, 11, 19, 110.0);
    meter.tokens_completed(request, 1_000);
    meter.request_closed(&request, 19.5);
    meter.record_power(20.0, 110.0);
    meter
}

#[test]
fn a_long_stream_stays_busy_until_its_terminal_event() {
    // 4,096 tokens at 2 tokens/s run well past half an hour.
    let mut meter = EnergyMeter::default();
    let stream = RequestId::new();
    samples(&mut meter, 0, 10, 10.0);
    meter.request_started(stream, 10.0);
    samples(&mut meter, 11, 2_058, 110.0);
    meter.tokens_completed(stream, 4_096);
    meter.request_closed(&stream, 2_058.5);
    meter.record_power(2_059.0, 110.0);
    let status = meter.status(SOURCE);
    assert_eq!(status.idle_seconds, 10.0);
    assert_eq!(status.busy_seconds, 2_049.0);
    assert_eq!(status.output_tokens, 4_096);
}

#[test]
fn tokens_produced_after_the_sampler_stops_leave_the_rate_alone() {
    let mut meter = meter_with_a_rate();
    let before = meter.status(SOURCE);
    // The sampler's last reading was at 20 s; these requests run without any.
    for start in [30.0, 60.0, 90.0] {
        let request = RequestId::new();
        meter.request_started(request, start);
        meter.tokens_completed(request, 5_000);
        meter.request_closed(&request, start + 20.0);
    }
    assert_eq!(meter.status(SOURCE), before);
    assert!(meter.requests.is_empty() && meter.active.is_empty());
}

#[test]
fn a_request_running_through_a_sampling_gap_is_left_out_of_the_rate() {
    let mut meter = meter_with_a_rate();
    let before = meter.status(SOURCE);
    let request = RequestId::new();
    meter.request_started(request, 20.5);
    meter.record_power(21.0, 110.0);
    // The sampler stalls past the gap limit, then resumes.
    meter.record_power(40.0, 110.0);
    samples(&mut meter, 41, 45, 110.0);
    meter.tokens_completed(request, 2_000);
    meter.request_closed(&request, 45.5);
    meter.record_power(46.0, 110.0);
    let after = meter.status(SOURCE);
    assert_eq!(after.output_tokens, before.output_tokens);
    assert_eq!(
        after.wh_per_1k_output_tokens,
        before.wh_per_1k_output_tokens
    );
    assert_eq!(
        after.wh_per_1k_output_tokens_total,
        before.wh_per_1k_output_tokens_total
    );
    // Its measured intervals still count as busy time.
    assert_eq!(after.busy_seconds, before.busy_seconds + 7.0);
}

#[test]
fn concurrent_requests_share_the_energy_of_the_intervals_they_ran_in() {
    let mut meter = EnergyMeter::default();
    samples(&mut meter, 0, 10, 10.0);
    let first = RequestId::new();
    let second = RequestId::new();
    meter.request_started(first, 10.0);
    meter.request_started(second, 10.0);
    samples(&mut meter, 11, 20, 210.0);
    meter.tokens_completed(first, 300);
    meter.tokens_completed(second, 700);
    meter.request_closed(&first, 20.5);
    meter.request_closed(&second, 20.5);
    meter.record_power(21.0, 210.0);
    let status = meter.status(SOURCE);
    assert_eq!(status.output_tokens, 1_000);
    // [10, 11] averages 10 W and 210 W, then ten seconds at 210 W: 2,210 J for
    // the 1,000 tokens of both requests together.
    let total = status.wh_per_1k_output_tokens_total.unwrap();
    assert!((total - 2_210.0 / 3_600.0).abs() < 1e-9, "{total}");
}

#[test]
fn lifecycle_events_drive_the_meter_and_count_tokens_once() {
    let mut meter = EnergyMeter::default();
    meter.record_power(0.0, 50.0);
    let streamed = RequestId::new();
    let operation = InferenceBackendOperation::ChatCompletionStream;
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::BackendDispatched {
            context: context(streamed),
            operation,
        },
        0.0,
    );
    assert!(meter.active.contains(&streamed));
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::ResponseCompleted {
            context: context(streamed),
            operation,
            usage: usage(40),
        },
        1.0,
    );
    // The backend call of a stream ends before generation does; the stream's
    // own terminal event, which reports the same usage again, closes it.
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::BackendTerminal {
            context: context(streamed),
            operation,
            result: InferenceTerminalResult::Completed { status_code: 200 },
        },
        1.5,
    );
    assert!(meter.requests[&streamed].computing);
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::StreamTerminal {
            context: context(streamed),
            result: InferenceTerminalResult::CompletedWithUsage {
                status_code: 200,
                usage: TokenUsage::from_counts(Some(10), Some(40), Some(50)).unwrap(),
            },
        },
        2.0,
    );
    // Rated once the interval it ended in is counted.
    meter.record_power(2.5, 50.0);
    assert_eq!(meter.output_tokens, 40);
    assert!(meter.requests.is_empty());

    let plain = RequestId::new();
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::BackendDispatched {
            context: context(plain),
            operation: InferenceBackendOperation::ChatCompletion,
        },
        3.0,
    );
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::ResponseCompleted {
            context: context(plain),
            operation: InferenceBackendOperation::ChatCompletion,
            usage: usage(7),
        },
        4.0,
    );
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::NonStreamTerminal {
            context: context(plain),
            result: InferenceTerminalResult::Completed { status_code: 200 },
            exchange_id: None,
        },
        4.0,
    );
    meter.record_power(5.0, 50.0);
    assert_eq!(meter.output_tokens, 47);
    assert!(meter.active.is_empty() && meter.requests.is_empty());
}

fn dispatched(request: RequestId, operation: InferenceBackendOperation) -> InferenceLifecycleEvent {
    InferenceLifecycleEvent::BackendDispatched {
        context: context(request),
        operation,
    }
}

fn backend_done(
    request: RequestId,
    operation: InferenceBackendOperation,
    result: InferenceTerminalResult,
) -> InferenceLifecycleEvent {
    InferenceLifecycleEvent::BackendTerminal {
        context: context(request),
        operation,
        result,
    }
}

#[test]
fn a_stream_keeps_the_node_busy_until_the_stream_ends() {
    // Measured on an RTX 4070 Ti: with BackendTerminal taken as the end of a
    // stream, four concurrent streams at 200 W were counted mostly as idle.
    let mut meter = EnergyMeter::default();
    let stream = RequestId::new();
    let operation = InferenceBackendOperation::ChatCompletionStream;
    meter.record_power(0.0, 30.0);
    apply_event(&mut meter, &dispatched(stream, operation), 0.1);
    let opened = InferenceTerminalResult::Completed { status_code: 200 };
    apply_event(&mut meter, &backend_done(stream, operation, opened), 0.2);
    samples(&mut meter, 1, 5, 200.0);
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::StreamTerminal {
            context: context(stream),
            result: InferenceTerminalResult::Completed { status_code: 200 },
        },
        5.5,
    );
    samples(&mut meter, 6, 7, 30.0);
    let status = meter.status(SOURCE);
    assert_eq!(status.busy_seconds, 6.0);
    assert_eq!(status.idle_seconds, 1.0);
}

#[test]
fn listing_models_or_counting_tokens_does_not_make_the_node_busy() {
    let mut meter = EnergyMeter::default();
    for operation in [
        InferenceBackendOperation::Models,
        InferenceBackendOperation::MessagesCountTokens,
    ] {
        apply_event(&mut meter, &dispatched(RequestId::new(), operation), 0.5);
    }
    samples(&mut meter, 0, 2, 25.0);
    assert!(meter.active.is_empty() && meter.requests.is_empty());
    assert_eq!(meter.status(SOURCE).busy_seconds, 0.0);
}

#[test]
fn a_stream_that_fails_to_open_frees_the_node() {
    let mut meter = EnergyMeter::default();
    let stream = RequestId::new();
    let operation = InferenceBackendOperation::ChatCompletionStream;
    apply_event(&mut meter, &dispatched(stream, operation), 0.0);
    let failed = InferenceTerminalResult::Failed {
        status_code: 503,
        failure: InferenceFailure::Backend,
    };
    apply_event(&mut meter, &backend_done(stream, operation, failed), 0.1);
    assert!(!meter.requests[&stream].computing);
}

#[test]
fn a_request_running_at_the_first_reading_makes_the_first_interval_busy() {
    let mut meter = EnergyMeter::default();
    let request = RequestId::new();
    let operation = InferenceBackendOperation::ChatCompletion;
    apply_event(&mut meter, &dispatched(request, operation), 0.0);
    meter.record_power(0.5, 180.0);
    meter.record_power(1.5, 180.0);
    assert_eq!(meter.status(SOURCE).busy_seconds, 1.0);
}
