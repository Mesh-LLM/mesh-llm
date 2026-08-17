use std::ptr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use skippy_ffi::{
    SkippyRuntimeEventCategory as RawRuntimeEventCategory,
    SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
    SkippyRuntimeEventKind as RawRuntimeEventKind,
    SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use super::{
    RuntimeEvent, RuntimeEventCategory, RuntimeEventEmitterKind, RuntimeEventFailureCode,
    RuntimeEventKind, RuntimeEventProgressUnit, Status, collect_model_open_events, run_model_open,
};

fn make_raw_runtime_event(
    kind: RawRuntimeEventKind,
    sequence: u64,
    status: Status,
    detail: &[u8],
) -> RawRuntimeEvent {
    RawRuntimeEvent {
        abi_version: 1,
        struct_size: std::mem::size_of::<RawRuntimeEvent>() as u32,
        category: RawRuntimeEventCategory::MODEL_OPEN,
        kind,
        emitter: RawRuntimeEventEmitterKind::OPEN_THREAD,
        reserved0: 0,
        sequence,
        timestamp_mono_ns: sequence * 10,
        model_id: 11,
        stage_id: 3,
        session_id: 0,
        progress_current: if kind == RawRuntimeEventKind::MODEL_OPEN_PROGRESS {
            500
        } else {
            0
        },
        progress_total: if kind == RawRuntimeEventKind::MODEL_OPEN_PROGRESS {
            1000
        } else {
            0
        },
        progress_unit: if kind == RawRuntimeEventKind::MODEL_OPEN_PROGRESS {
            RawRuntimeEventProgressUnit::STEPS
        } else {
            RawRuntimeEventProgressUnit::NONE
        },
        failure_code: if kind == RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED {
            RawRuntimeEventFailureCode::MODEL_ERROR
        } else {
            RawRuntimeEventFailureCode::NONE
        },
        status,
        reserved1: 0,
        detail_ptr: detail.as_ptr().cast(),
        detail_len: detail.len() as u64,
    }
}

fn collect_runtime_events_for_test<OpenFn>(
    open_fn: OpenFn,
) -> (
    *mut skippy_ffi::Model,
    Status,
    *mut skippy_ffi::Error,
    Vec<RuntimeEvent>,
)
where
    OpenFn: FnOnce(
        *const skippy_ffi::SkippyRuntimeEventReporterV1,
        *mut *mut skippy_ffi::Model,
        *mut *mut skippy_ffi::Error,
    ) -> Status,
{
    let mut events = Vec::new();
    let (raw, status, error) = collect_model_open_events(open_fn, |event| events.push(event));
    (raw, status, error, events)
}

#[test]
fn runtime_event_from_raw_ptr_converts_unknown_values_and_copies_detail() {
    let mut detail = b"backend-selected".to_vec();
    let raw = RawRuntimeEvent {
        abi_version: 7,
        struct_size: std::mem::size_of::<RawRuntimeEvent>() as u32,
        category: RawRuntimeEventCategory(999),
        kind: RawRuntimeEventKind::BACKEND_DEVICE_SELECTED,
        emitter: RawRuntimeEventEmitterKind(77),
        reserved0: 0,
        sequence: 42,
        timestamp_mono_ns: 4242,
        model_id: 12,
        stage_id: 5,
        session_id: 9,
        progress_current: 12,
        progress_total: 34,
        progress_unit: RawRuntimeEventProgressUnit(88),
        failure_code: RawRuntimeEventFailureCode(66),
        status: Status::Unsupported,
        reserved1: 0,
        detail_ptr: detail.as_ptr().cast(),
        detail_len: detail.len() as u64,
    };

    let event = RuntimeEvent::from_raw_ptr(&raw).expect("raw event should convert");
    detail.fill(b'x');

    assert_eq!(event.abi_version, 7);
    assert_eq!(event.category, RuntimeEventCategory::Unknown(999));
    assert_eq!(event.kind, RuntimeEventKind::BackendDeviceSelected);
    assert_eq!(event.emitter, RuntimeEventEmitterKind::Other(77));
    assert_eq!(event.sequence, 42);
    assert_eq!(event.timestamp_mono_ns, 4242);
    assert_eq!(event.model_id, 12);
    assert_eq!(event.stage_id, 5);
    assert_eq!(event.session_id, 9);
    assert_eq!(event.progress_current, 12);
    assert_eq!(event.progress_total, 34);
    assert_eq!(event.progress_unit, RuntimeEventProgressUnit::Unknown(88));
    assert_eq!(event.failure_code, RuntimeEventFailureCode::Unknown(66));
    assert_eq!(event.status, Status::Unsupported);
    assert_eq!(event.detail_bytes, b"backend-selected");
}

pub(crate) fn assert_model_open_events_success() {
    let mut started_detail = b"started".to_vec();
    let backend_detail = b"Metal-0".to_vec();
    let progress_detail = b"progress".to_vec();
    let finished_detail = b"finished".to_vec();
    let (raw, status, error, events) =
        collect_runtime_events_for_test(|reporter, out_model, _out_error| {
            let callback = unsafe { (*reporter).callback.expect("callback") };
            unsafe {
                callback(ptr::null(), (*reporter).user_data);
            }

            let mut too_small = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_STARTED,
                0,
                Status::Ok,
                b"ignored",
            );
            too_small.struct_size -= 1;
            unsafe {
                callback(&too_small, (*reporter).user_data);
            }

            let started = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_STARTED,
                1,
                Status::Ok,
                &started_detail,
            );
            unsafe {
                callback(&started, (*reporter).user_data);
            }
            started_detail.fill(b'x');

            let backend_selected = make_raw_runtime_event(
                RawRuntimeEventKind::BACKEND_DEVICE_SELECTED,
                2,
                Status::Ok,
                &backend_detail,
            );
            unsafe {
                callback(&backend_selected, (*reporter).user_data);
            }

            let progress = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_PROGRESS,
                3,
                Status::Ok,
                &progress_detail,
            );
            unsafe {
                callback(&progress, (*reporter).user_data);
            }

            let finished = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_FINISHED,
                4,
                Status::Ok,
                &finished_detail,
            );
            unsafe {
                callback(&finished, (*reporter).user_data);
            }

            unsafe {
                *out_model = ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr();
            }
            Status::Ok
        });

    assert_eq!(status, Status::Ok);
    assert!(error.is_null());
    assert_eq!(raw, ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr());
    assert_eq!(events.len(), 4);
    assert_eq!(
        events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        events.iter().map(|event| event.kind).collect::<Vec<_>>(),
        vec![
            RuntimeEventKind::ModelOpenStarted,
            RuntimeEventKind::BackendDeviceSelected,
            RuntimeEventKind::ModelOpenProgress,
            RuntimeEventKind::ModelOpenFinished,
        ]
    );
    assert_eq!(events[0].category, RuntimeEventCategory::ModelOpen);
    assert_eq!(events[0].kind, RuntimeEventKind::ModelOpenStarted);
    assert_eq!(events[0].emitter, RuntimeEventEmitterKind::OpenThread);
    assert_eq!(events[0].detail_bytes, b"started");
    assert_eq!(events[1].kind, RuntimeEventKind::BackendDeviceSelected);
    assert_eq!(events[1].detail_bytes, b"Metal-0");
    assert_eq!(events[2].kind, RuntimeEventKind::ModelOpenProgress);
    assert_eq!(events[2].progress_current, 500);
    assert_eq!(events[2].progress_total, 1000);
    assert_eq!(events[2].progress_unit, RuntimeEventProgressUnit::Steps);
    assert_eq!(events[3].kind, RuntimeEventKind::ModelOpenFinished);
    assert_eq!(events[3].detail_bytes, b"finished");
}

pub(crate) fn assert_model_open_events_handled_failure() {
    let failure_detail = b"failed to load llama model from GGUF parts".to_vec();
    let (_raw, status, error, events) =
        collect_runtime_events_for_test(|reporter, _out_model, _out_error| {
            let callback = unsafe { (*reporter).callback.expect("callback") };
            let started = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_STARTED,
                1,
                Status::Ok,
                b"started",
            );
            unsafe {
                callback(&started, (*reporter).user_data);
            }
            let progress = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_PROGRESS,
                2,
                Status::Ok,
                b"progress",
            );
            unsafe {
                callback(&progress, (*reporter).user_data);
            }
            let failure = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED,
                3,
                Status::ModelError,
                &failure_detail,
            );
            unsafe {
                callback(&failure, (*reporter).user_data);
            }
            Status::ModelError
        });

    assert_eq!(status, Status::ModelError);
    assert!(error.is_null());
    assert_eq!(events.len(), 3);
    assert_eq!(
        events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        events.iter().map(|event| event.kind).collect::<Vec<_>>(),
        vec![
            RuntimeEventKind::ModelOpenStarted,
            RuntimeEventKind::ModelOpenProgress,
            RuntimeEventKind::ModelOpenFailedHandled,
        ]
    );
    assert_eq!(events[1].progress_current, 500);
    assert_eq!(events[1].progress_total, 1000);
    assert_eq!(events[2].kind, RuntimeEventKind::ModelOpenFailedHandled);
    assert_eq!(events[2].failure_code, RuntimeEventFailureCode::ModelError);
    assert_eq!(events[2].status, Status::ModelError);
    assert_eq!(
        events[2].detail_bytes,
        b"failed to load llama model from GGUF parts"
    );
}

pub(crate) fn assert_model_open_events_missing_terminal_callback_uses_return() {
    let (raw, status, error, events) =
        collect_runtime_events_for_test(|reporter, _out_model, _out_error| {
            let callback = unsafe { (*reporter).callback.expect("callback") };
            let started = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_STARTED,
                1,
                Status::Ok,
                b"started",
            );
            unsafe {
                callback(&started, (*reporter).user_data);
            }
            let progress = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_PROGRESS,
                2,
                Status::Ok,
                b"progress",
            );
            unsafe {
                callback(&progress, (*reporter).user_data);
            }
            Status::RuntimeError
        });

    assert!(raw.is_null());
    assert!(error.is_null());
    assert_eq!(status, Status::RuntimeError);
    assert_eq!(events.len(), 2);
    assert_eq!(
        events.iter().map(|event| event.kind).collect::<Vec<_>>(),
        vec![
            RuntimeEventKind::ModelOpenStarted,
            RuntimeEventKind::ModelOpenProgress
        ]
    );
}

pub(crate) fn assert_model_open_events_forwarded_before_open_returns() {
    let forwarded_sequences = Arc::new(Mutex::new(Vec::new()));
    let sink_sequences = Arc::clone(&forwarded_sequences);
    let open_sequences = Arc::clone(&forwarded_sequences);
    let (raw, status, error) = collect_model_open_events(
        |reporter, out_model, _out_error| {
            let callback = unsafe { (*reporter).callback.expect("callback") };
            let started = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_STARTED,
                1,
                Status::Ok,
                b"started",
            );
            unsafe {
                callback(&started, (*reporter).user_data);
            }
            assert_eq!(open_sequences.lock().expect("event lock").as_slice(), &[1]);

            let finished = make_raw_runtime_event(
                RawRuntimeEventKind::MODEL_OPEN_FINISHED,
                2,
                Status::Ok,
                b"finished",
            );
            unsafe {
                callback(&finished, (*reporter).user_data);
            }
            assert_eq!(
                open_sequences.lock().expect("event lock").as_slice(),
                &[1, 2]
            );

            unsafe {
                *out_model = ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr();
            }
            Status::Ok
        },
        |event| {
            sink_sequences
                .lock()
                .expect("event lock")
                .push(event.sequence);
        },
    );

    assert_eq!(status, Status::Ok);
    assert!(error.is_null());
    assert_eq!(raw, ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr());
    assert_eq!(
        forwarded_sequences.lock().expect("event lock").as_slice(),
        &[1, 2]
    );
}

pub(crate) fn assert_model_open_events_feature_missing_falls_back() {
    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let event_path_calls = Arc::new(AtomicUsize::new(0));

    let mut bridged_events = Vec::new();
    let mut bridged_event_sink = |event| bridged_events.push(event);
    let (raw, status, error) = run_model_open(
        {
            let legacy_calls = Arc::clone(&legacy_calls);
            move |out_model, _out_error| {
                legacy_calls.fetch_add(1, Ordering::SeqCst);
                unsafe {
                    *out_model = ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr();
                }
                Status::Ok
            }
        },
        {
            let event_path_calls = Arc::clone(&event_path_calls);
            move |_reporter, _out_model, _out_error| {
                event_path_calls.fetch_add(1, Ordering::SeqCst);
                Status::Ok
            }
        },
        Some(&mut bridged_event_sink),
        false,
    );

    assert_eq!(status, Status::Ok);
    assert!(error.is_null());
    assert_eq!(raw, ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr());
    assert_eq!(legacy_calls.load(Ordering::SeqCst), 1);
    assert_eq!(event_path_calls.load(Ordering::SeqCst), 0);
    assert!(bridged_events.is_empty());

    let (raw, status, error) = run_model_open(
        {
            let legacy_calls = Arc::clone(&legacy_calls);
            move |out_model, _out_error| {
                legacy_calls.fetch_add(1, Ordering::SeqCst);
                unsafe {
                    *out_model = ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr();
                }
                Status::Ok
            }
        },
        {
            let event_path_calls = Arc::clone(&event_path_calls);
            move |_reporter, _out_model, _out_error| {
                event_path_calls.fetch_add(1, Ordering::SeqCst);
                Status::Ok
            }
        },
        None,
        false,
    );

    assert_eq!(status, Status::Ok);
    assert!(error.is_null());
    assert_eq!(raw, ptr::NonNull::<skippy_ffi::Model>::dangling().as_ptr());
    assert_eq!(legacy_calls.load(Ordering::SeqCst), 2);
    assert_eq!(event_path_calls.load(Ordering::SeqCst), 0);
}
