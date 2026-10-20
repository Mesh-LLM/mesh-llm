use super::*;
use skippy_events::usage::TokenUsage;
use skippy_inference_api::{
    InferenceBackendOperation, InferenceFrontendRoute, InferenceLifecycleContext,
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
    samples(&mut meter, 11, 20, 110.0);
    meter.tokens_completed(request, 1_000);
    meter.request_closed(&request);

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
    meter.request_closed(&request);
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

#[test]
fn a_request_that_never_ends_stops_counting_as_busy() {
    let mut meter = EnergyMeter::default();
    meter.request_started(RequestId::new(), 0.0);
    meter.record_power(1.0, 10.0);
    meter.record_power(2.0, 10.0);
    assert_eq!(meter.status(SOURCE).busy_seconds, 1.0);
    meter.record_power(MAX_REQUEST_SECONDS + 1.0, 10.0);
    meter.record_power(MAX_REQUEST_SECONDS + 2.0, 10.0);
    meter.record_power(MAX_REQUEST_SECONDS + 3.0, 10.0);
    assert_eq!(meter.status(SOURCE).idle_seconds, 2.0);
}

#[test]
fn lifecycle_events_drive_the_meter_and_count_tokens_once() {
    let mut meter = EnergyMeter::default();
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
    assert!(meter.in_flight.contains_key(&streamed));
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::ResponseCompleted {
            context: context(streamed),
            operation,
            usage: usage(40),
        },
        1.0,
    );
    // The backend may finish before the stream's own terminal event, which
    // reports the same usage again.
    apply_event(
        &mut meter,
        &InferenceLifecycleEvent::BackendTerminal {
            context: context(streamed),
            operation,
            result: InferenceTerminalResult::Completed { status_code: 200 },
        },
        1.5,
    );
    assert!(!meter.in_flight.contains_key(&streamed));
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
    assert_eq!(meter.output_tokens, 40);
    assert!(meter.counted.is_empty());

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
    assert_eq!(meter.output_tokens, 47);
    assert!(meter.in_flight.is_empty() && meter.counted.is_empty());
}
