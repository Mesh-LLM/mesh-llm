use std::sync::{Arc, Mutex};

use skippy_api::serving::{InferenceOptions, ModelLoadRequest};
use skippy_protocol::StageConfig;
use skippy_runtime::{
    RuntimeEvent, RuntimeEventCategory, RuntimeEventEmitterKind, RuntimeEventFailureCode,
    RuntimeEventKind, RuntimeEventProgressUnit,
};
use skippy_serving::{EmbeddedRuntimeOptions, binary_transport::WireCondition};

use super::observation::ModelOpenContradiction;
use super::*;

fn progress(sequence: u64) -> RuntimeEvent {
    RuntimeEvent {
        abi_version: 1,
        struct_size: 144,
        category: RuntimeEventCategory::ModelOpen,
        kind: RuntimeEventKind::ModelOpenProgress,
        emitter: RuntimeEventEmitterKind::OpenThread,
        reserved0: 0,
        sequence,
        timestamp_mono_ns: sequence,
        model_id: 1,
        stage_id: 0,
        session_id: 0,
        progress_current: sequence,
        progress_total: 10,
        progress_unit: RuntimeEventProgressUnit::Steps,
        failure_code: RuntimeEventFailureCode::None,
        status: skippy_ffi::Status::Ok,
        reserved1: 0,
        detail_bytes: Vec::new(),
        numeric_summary_0: None,
        numeric_summary_1: None,
        numeric_summary_2: None,
        numeric_summary_3: None,
    }
}

type Reconciled = Arc<Mutex<Vec<(Option<ModelOpenObservation>, ModelOpenReturn)>>>;

fn recording_events(reporter: NativeModelOpenEventReporter) -> (NativeModelOpenEvents, Reconciled) {
    let reconciled: Reconciled = Arc::default();
    let slot = Arc::clone(&reconciled);
    let events = NativeModelOpenEvents {
        reporter,
        reconcile: Box::new(move |observation, returned| {
            slot.lock()
                .expect("reconciled")
                .push((observation.copied(), returned));
        }),
    };
    (events, reconciled)
}

fn observer(events: NativeModelOpenEvents) -> ModelOpenObserver {
    ModelOpenObserver {
        reporter: events.reporter,
        reconcile: Arc::new(Mutex::new(Some(events.reconcile))),
    }
}

fn reconciled(reconciled: &Reconciled) -> Vec<(Option<ModelOpenObservation>, ModelOpenReturn)> {
    reconciled.lock().expect("reconciled").clone()
}

/// A load Skippy rejects before any native work: a worker stage cannot be
/// served directly.
fn rejected_load(open_events: ModelOpenEvents) -> ModelLoadRequest {
    let config = StageConfig {
        stage_index: 1,
        ..StageConfig::default()
    };
    ModelLoadRequest {
        openai: InferenceOptions::direct_single_stage_defaults(
            config.model_id.clone(),
            32,
            1,
            false,
        ),
        runtime: EmbeddedRuntimeOptions {
            config,
            topology: None,
            n_threads: None,
            n_threads_batch: None,
            mtp_source: Default::default(),
            recurrent_rollback_tokens: 0,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 0,
            telemetry_level: skippy_serving::telemetry::TelemetryLevel::Off,
            session_lifecycle_observer: None,
            receipts: None,
        },
        open_events,
        hooks_factory: None,
        generation_observer: None,
        kv_observer: None,
        hook_policy: None,
        guardrails: None,
        guardrail_telemetry: None,
        downstream_wire_condition: WireCondition::new(0.0, None).expect("wire condition"),
        serving_telemetry: None,
        l3_manager: None,
    }
}

#[test]
fn every_event_reaches_the_reporter_in_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let (events, _) = recording_events(Box::new(move |event| {
        sink.lock().expect("sink").push(event.sequence)
    }));
    let mut observer = observer(events);

    (1..=5).for_each(|sequence| observer.event(progress(sequence)));

    assert_eq!(*seen.lock().expect("sink"), vec![1, 2, 3, 4, 5]);
}

#[test]
fn the_reporter_gets_the_native_record_unchanged() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let (events, _) = recording_events(Box::new(move |event| {
        sink.lock().expect("sink").push(event)
    }));
    let mut observer = observer(events);
    let record = RuntimeEvent {
        kind: RuntimeEventKind::ModelOpenFailedHandled,
        stage_id: 23,
        session_id: 29,
        failure_code: RuntimeEventFailureCode::ModelError,
        status: skippy_ffi::Status::ModelError,
        detail_bytes: b"bad header".to_vec(),
        numeric_summary_0: Some(101),
        numeric_summary_1: Some(102),
        numeric_summary_2: Some(0),
        numeric_summary_3: None,
        ..progress(7)
    };

    observer.event(record.clone());

    assert_eq!(*seen.lock().expect("sink"), vec![record]);
}

#[test]
fn a_panicking_reporter_loses_only_its_own_event() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let (events, reconciled_slot) = recording_events(Box::new(move |event| {
        assert_ne!(event.sequence, 2, "reporter fails on the second record");
        sink.lock().expect("sink").push(event.sequence);
    }));
    let mut observer = observer(events);

    (1..=3).for_each(|sequence| observer.event(progress(sequence)));
    observer.summary(
        &ModelOpenObservation {
            drained: 3,
            ..Default::default()
        },
        ModelOpenReturn::Succeeded,
    );

    assert_eq!(*seen.lock().expect("sink"), vec![1, 3]);
    assert_eq!(reconciled(&reconciled_slot).len(), 1);
}

#[test]
fn the_summary_is_reconciled_once_with_its_losses_and_return() {
    let (events, reconciled_slot) = recording_events(Box::new(|_| {}));
    let mut observer = observer(events);
    let observation = ModelOpenObservation {
        drained: 3,
        dropped: 2,
        rejected: 1,
        last_sequence: Some(6),
        ..Default::default()
    };

    observer.summary(&observation, ModelOpenReturn::Failed);
    observer.summary(&ModelOpenObservation::default(), ModelOpenReturn::Succeeded);

    assert_eq!(
        reconciled(&reconciled_slot),
        vec![(Some(observation), ModelOpenReturn::Failed)]
    );
    assert_eq!(observation.lost(), 3);
}

#[test]
fn no_events_means_no_observer() {
    let (open_events, settlement) = model_open_events(None);

    assert!(matches!(open_events, ModelOpenEvents::Enabled(None)));
    settlement.finish(ModelOpenReturn::Succeeded);
}

#[test]
fn a_failed_load_keeps_its_error_and_reconciles_once_before_it_returns() {
    let (events, reconciled_slot) = recording_events(Box::new(|_| {}));
    let (open_events, settlement) = model_open_events(Some(events));

    let error = rejected_load(open_events)
        .load()
        .err()
        .expect("worker stage must be rejected");
    let before_finish = reconciled(&reconciled_slot);
    settlement.finish(ModelOpenReturn::Failed);

    assert!(error.to_string().contains("requires stage 0"), "{error:#}");
    assert_eq!(
        before_finish,
        vec![(
            Some(ModelOpenObservation::default()),
            ModelOpenReturn::Failed
        )]
    );
    assert_eq!(reconciled(&reconciled_slot), before_finish);
}

/// Skippy drops the observer unused when it cannot start its drain thread:
/// the load goes on without events and no summary is delivered.
#[test]
fn a_load_whose_observer_never_ran_is_reconciled_without_an_observation() {
    let (events, reconciled_slot) = recording_events(Box::new(|_| {}));
    let (open_events, settlement) = model_open_events(Some(events));

    drop(open_events);
    settlement.finish(ModelOpenReturn::Succeeded);

    assert_eq!(
        reconciled(&reconciled_slot),
        vec![(None, ModelOpenReturn::Succeeded)]
    );
}

#[test]
fn finished_callback_with_failed_return_is_a_contradiction() {
    let observation = ModelOpenObservation {
        saw_finished: true,
        ..Default::default()
    };

    assert_eq!(
        observation.contradiction(ModelOpenReturn::Failed),
        Some(ModelOpenContradiction::FinishedButReturnedFailure)
    );
    assert_eq!(observation.contradiction(ModelOpenReturn::Succeeded), None);
}

#[test]
fn failed_handled_callback_with_successful_return_is_a_contradiction() {
    let observation = ModelOpenObservation {
        saw_failed_handled: true,
        ..Default::default()
    };

    assert_eq!(
        observation.contradiction(ModelOpenReturn::Succeeded),
        Some(ModelOpenContradiction::FailedHandledButReturnedSuccess)
    );
    assert_eq!(observation.contradiction(ModelOpenReturn::Failed), None);
}

#[test]
fn missing_terminal_callback_is_not_a_contradiction() {
    let observation = ModelOpenObservation {
        drained: 1,
        ..Default::default()
    };

    assert_eq!(observation.contradiction(ModelOpenReturn::Succeeded), None);
    assert_eq!(observation.contradiction(ModelOpenReturn::Failed), None);
}
