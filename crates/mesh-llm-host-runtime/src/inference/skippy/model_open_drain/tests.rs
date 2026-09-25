use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::Duration;

use skippy_ffi::{
    SkippyRuntimeEventCategory, SkippyRuntimeEventEmitterKind, SkippyRuntimeEventFailureCode,
    SkippyRuntimeEventKind, SkippyRuntimeEventProgressUnit, SkippyRuntimeEventV1, Status,
};

use super::observation::ModelOpenContradiction;
use super::*;

const WAIT: Duration = Duration::from_secs(5);

fn raw_event(kind: SkippyRuntimeEventKind, sequence: u64) -> SkippyRuntimeEventV1 {
    SkippyRuntimeEventV1 {
        abi_version: 1,
        struct_size: std::mem::size_of::<SkippyRuntimeEventV1>() as u32,
        category: SkippyRuntimeEventCategory::MODEL_OPEN,
        kind,
        emitter: SkippyRuntimeEventEmitterKind::OPEN_THREAD,
        reserved0: 0,
        sequence,
        timestamp_mono_ns: sequence,
        model_id: 1,
        stage_id: 0,
        session_id: 0,
        progress_current: sequence,
        progress_total: 10,
        progress_unit: SkippyRuntimeEventProgressUnit::STEPS,
        failure_code: SkippyRuntimeEventFailureCode::NONE,
        status: Status::Ok,
        reserved1: 0,
        detail_ptr: std::ptr::null(),
        detail_len: 0,
        numeric_summary_0: 0,
        numeric_summary_1: 0,
        numeric_summary_2: 0,
        numeric_summary_3: 0,
    }
}

fn push(queue: &ModelOpenEventQueue, kind: SkippyRuntimeEventKind, sequence: u64) {
    // SAFETY: the event is a valid stack value for the whole call.
    unsafe { queue.deliver_for_test(&raw_event(kind, sequence)) };
}

fn push_progress(queue: &ModelOpenEventQueue, sequence: u64) {
    push(queue, SkippyRuntimeEventKind::MODEL_OPEN_PROGRESS, sequence);
}

fn sequence_reporter(seen: &Arc<Mutex<Vec<u64>>>) -> NativeModelOpenEventReporter {
    let sink = Arc::clone(seen);
    Box::new(move |event| sink.lock().expect("sink").push(event.sequence))
}

type Reconciled = Arc<Mutex<Option<(ModelOpenObservation, ModelOpenReturn)>>>;

fn recording_events(reporter: NativeModelOpenEventReporter) -> (NativeModelOpenEvents, Reconciled) {
    let reconciled: Reconciled = Arc::default();
    let slot = Arc::clone(&reconciled);
    let events = NativeModelOpenEvents {
        reporter,
        reconcile: Box::new(move |observation, returned| {
            *slot.lock().expect("reconciled") = Some((*observation, returned));
        }),
    };
    (events, reconciled)
}

fn take_reconciled(reconciled: &Reconciled) -> (ModelOpenObservation, ModelOpenReturn) {
    reconciled
        .lock()
        .expect("reconciled")
        .take()
        .expect("the reconciler runs exactly once after the open returns")
}

#[test]
fn every_record_pushed_during_load_reaches_the_reporter_in_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (events, _) = recording_events(sequence_reporter(&seen));

    let returned = observe_model_open(OperationId(4), Some(events), |queue| {
        let queue = queue.expect("events yield a queue");
        assert_eq!(queue.operation_id(), OperationId(4));
        (1..=5).for_each(|sequence| push_progress(&queue, sequence));
        Ok::<_, ()>("loaded")
    });

    assert_eq!(returned, Ok("loaded"));
    assert_eq!(*seen.lock().expect("sink"), vec![1, 2, 3, 4, 5]);
}

#[test]
fn no_events_means_no_queue() {
    let had_queue = observe_model_open(OperationId(1), None, |queue| Ok::<_, ()>(queue.is_some()));
    assert_eq!(had_queue, Ok(false));
}

#[test]
fn sink_runs_on_drainer_thread_not_caller_or_native() {
    let sink_threads = Arc::new(Mutex::new(Vec::<ThreadId>::new()));
    let sink = Arc::clone(&sink_threads);
    let reporter: NativeModelOpenEventReporter =
        Box::new(move |_| sink.lock().expect("sink").push(thread::current().id()));
    let (events, _) = recording_events(reporter);
    let caller = thread::current().id();

    let native = observe_model_open(OperationId(2), Some(events), |queue| {
        let queue = queue.expect("queue");
        push_progress(&queue, 1);
        let native = thread::spawn(move || {
            push_progress(&queue, 2);
            thread::current().id()
        });
        Ok::<_, ()>(native.join().expect("native thread"))
    })
    .expect("load");

    let sink_threads = sink_threads.lock().expect("sink");
    assert_eq!(sink_threads.len(), 2);
    assert!(
        sink_threads.iter().all(|id| *id != caller && *id != native),
        "the reporter must only run on the host drainer thread"
    );
}

#[test]
fn progress_is_delivered_while_open_is_still_blocked() {
    let (tx, rx) = mpsc::channel();
    let reporter: NativeModelOpenEventReporter = Box::new(move |event| {
        let _ = tx.send(event.sequence);
    });
    let (events, _) = recording_events(reporter);

    let delivered_before_return = observe_model_open(OperationId(3), Some(events), |queue| {
        let queue = queue.expect("queue");
        (1..=3).for_each(|sequence| push_progress(&queue, sequence));
        let delivered: Vec<u64> = (0..3).map_while(|_| rx.recv_timeout(WAIT).ok()).collect();
        Ok::<_, ()>(delivered)
    })
    .expect("load");

    assert_eq!(delivered_before_return, vec![1, 2, 3]);
}

#[test]
fn final_drain_after_return_loses_nothing() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (events, reconciled) = recording_events(sequence_reporter(&seen));

    let _ = observe_model_open(OperationId(5), Some(events), |queue| {
        let queue = queue.expect("queue");
        (1..=40).for_each(|sequence| push_progress(&queue, sequence));
        push(&queue, SkippyRuntimeEventKind::MODEL_OPEN_FINISHED, 41);
        Ok::<_, ()>(())
    });

    let (observation, returned) = take_reconciled(&reconciled);
    assert_eq!(*seen.lock().expect("sink"), (1..=41).collect::<Vec<_>>());
    assert_eq!(observation.drained, 41);
    assert_eq!(observation.last_sequence, Some(41));
    assert!(observation.saw_finished);
    assert_eq!(observation.lost(), 0);
    assert_eq!(returned, ModelOpenReturn::Succeeded);
}

#[test]
fn full_queue_reports_dropped_count() {
    let (got_tx, got_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = Mutex::new(release_rx);
    let reporter: NativeModelOpenEventReporter = Box::new(move |event| {
        if event.sequence == 1 {
            let _ = got_tx.send(());
            let _ = release_rx.lock().expect("release").recv_timeout(WAIT);
        }
    });
    let (NativeModelOpenEvents { reporter, .. }, _) = recording_events(reporter);
    let queue = ModelOpenEventQueue::with_capacity(OperationId(6), 2);

    let ((), observation) = with_model_open_events(queue, reporter, |queue| {
        let queue = queue.expect("queue");
        push_progress(&queue, 1);
        got_rx
            .recv_timeout(WAIT)
            .expect("drainer took the first record");
        (2..=6).for_each(|sequence| push_progress(&queue, sequence));
        release_tx.send(()).expect("release drainer");
    });

    assert_eq!(observation.drained, 3);
    assert_eq!(observation.dropped, 3);
    assert_eq!(observation.rejected, 0);
}

#[test]
fn rejected_events_are_counted_as_lost() {
    let (events, reconciled) = recording_events(Box::new(|_| {}));

    let _ = observe_model_open(OperationId(7), Some(events), |queue| {
        let queue = queue.expect("queue");
        // SAFETY: a null event is the documented rejection path.
        unsafe { queue.deliver_for_test(std::ptr::null()) };
        Ok::<_, ()>(())
    });

    let (observation, _) = take_reconciled(&reconciled);
    assert_eq!(observation.rejected, 1);
    assert_eq!(observation.lost(), 1);
}

#[test]
fn finished_callback_with_failed_return_keeps_the_failure_and_reports_a_contradiction() {
    let (events, reconciled) = recording_events(Box::new(|_| {}));

    let result = observe_model_open(OperationId(8), Some(events), |queue| {
        push(
            &queue.expect("queue"),
            SkippyRuntimeEventKind::MODEL_OPEN_FINISHED,
            1,
        );
        Err::<(), _>("native open failed")
    });

    let (observation, returned) = take_reconciled(&reconciled);
    assert_eq!(result, Err("native open failed"));
    assert_eq!(returned, ModelOpenReturn::Failed);
    assert_eq!(
        observation.contradiction(returned),
        Some(ModelOpenContradiction::FinishedButReturnedFailure)
    );
}

#[test]
fn failed_handled_callback_with_successful_return_keeps_the_success_and_reports_a_contradiction() {
    let (events, reconciled) = recording_events(Box::new(|_| {}));

    let result = observe_model_open(OperationId(9), Some(events), |queue| {
        push(
            &queue.expect("queue"),
            SkippyRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED,
            1,
        );
        Ok::<_, ()>("runtime")
    });

    let (observation, returned) = take_reconciled(&reconciled);
    assert_eq!(result, Ok("runtime"));
    assert_eq!(
        observation.contradiction(returned),
        Some(ModelOpenContradiction::FailedHandledButReturnedSuccess)
    );
}

#[test]
fn missing_terminal_callback_with_successful_return_is_not_a_contradiction() {
    let (events, reconciled) = recording_events(Box::new(|_| {}));

    let _ = observe_model_open(OperationId(10), Some(events), |queue| {
        push_progress(&queue.expect("queue"), 1);
        Ok::<_, ()>(())
    });

    let (observation, returned) = take_reconciled(&reconciled);
    assert_eq!(observation.contradiction(returned), None);
}

#[test]
fn panicking_load_stops_the_drainer_and_propagates() {
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_model_open_events(
                ModelOpenEventQueue::new(OperationId(7)),
                sequence_reporter(&seen),
                |_queue| -> () { panic!("load panicked") },
            )
        }));
        done_tx.send(outcome.is_err()).expect("send");
    });
    let propagated = done_rx
        .recv_timeout(WAIT)
        .expect("a panicking load must not hang the drainer scope");
    assert!(propagated);
}
