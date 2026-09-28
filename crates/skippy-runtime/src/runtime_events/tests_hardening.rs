use skippy_ffi::{
    SkippyRuntimeEventCategory as RawRuntimeEventCategory,
    SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
    SkippyRuntimeEventKind as RawRuntimeEventKind,
    SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use super::{NativeEventRecord, RecordRejection, RuntimeEvent, RuntimeEventKind, Status};

fn raw_event(kind: RawRuntimeEventKind, sequence: u64) -> RawRuntimeEvent {
    RawRuntimeEvent {
        abi_version: 1,
        struct_size: std::mem::size_of::<RawRuntimeEvent>() as u32,
        category: RawRuntimeEventCategory::MODEL_OPEN,
        kind,
        emitter: RawRuntimeEventEmitterKind::WORKER_THREAD,
        reserved0: 0,
        sequence,
        timestamp_mono_ns: sequence,
        model_id: 1,
        stage_id: 0,
        session_id: 0,
        progress_current: 0,
        progress_total: 0,
        progress_unit: RawRuntimeEventProgressUnit::NONE,
        failure_code: RawRuntimeEventFailureCode::NONE,
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

#[test]
fn from_raw_ptr_rejects_short_struct_before_reading_detail_fields() {
    let mut event = raw_event(RawRuntimeEventKind::MODEL_OPEN_STARTED, 1);
    // Below the BASE (pre-extension) layout, not just the full one -- see
    // the equivalent comment in runtime_events/tests.rs.
    event.struct_size = std::mem::offset_of!(RawRuntimeEvent, numeric_summary_0) as u32 - 1;
    assert!(RuntimeEvent::from_raw_ptr(&event).is_none());
}

#[test]
fn from_raw_ptr_rejects_oversized_detail_len() {
    let detail = b"x".repeat(16);
    let mut event = raw_event(RawRuntimeEventKind::MODEL_OPEN_PROGRESS, 2);
    event.detail_ptr = detail.as_ptr().cast();
    event.detail_len = u64::MAX;
    assert!(RuntimeEvent::from_raw_ptr(&event).is_none());
}

#[test]
fn from_raw_ptr_rejects_null_event() {
    assert!(RuntimeEvent::from_raw_ptr(std::ptr::null()).is_none());
}

#[test]
fn from_raw_ptr_enumerates_all_five_known_kinds() {
    let cases = [
        (
            RawRuntimeEventKind::MODEL_OPEN_STARTED,
            RuntimeEventKind::ModelOpenStarted,
        ),
        (
            RawRuntimeEventKind::MODEL_OPEN_PROGRESS,
            RuntimeEventKind::ModelOpenProgress,
        ),
        (
            RawRuntimeEventKind::BACKEND_DEVICE_SELECTED,
            RuntimeEventKind::BackendDeviceSelected,
        ),
        (
            RawRuntimeEventKind::MODEL_OPEN_FINISHED,
            RuntimeEventKind::ModelOpenFinished,
        ),
        (
            RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED,
            RuntimeEventKind::ModelOpenFailedHandled,
        ),
    ];
    for (raw_kind, expected) in cases {
        let event = raw_event(raw_kind, 1);
        let decoded = RuntimeEvent::from_raw_ptr(&event).expect("known kind decodes");
        assert_eq!(decoded.kind, expected);
    }
}

#[test]
fn from_raw_ptr_preserves_unknown_kind_rather_than_dropping_it() {
    let event = raw_event(RawRuntimeEventKind(9999), 1);
    let decoded =
        RuntimeEvent::from_raw_ptr(&event).expect("unknown-but-well-formed event decodes");
    assert_eq!(decoded.kind, RuntimeEventKind::Unknown(9999));
}

#[test]
fn rejects_wrong_abi_version() {
    let mut event = raw_event(RawRuntimeEventKind::MODEL_OPEN_STARTED, 1);
    event.abi_version = 2;
    let rejection = unsafe { NativeEventRecord::from_raw_ptr(&event) };
    assert_eq!(rejection, Err(RecordRejection::AbiVersion(2)));
    assert!(RuntimeEvent::from_raw_ptr(&event).is_none());
}

/// A newer runtime may append fields; the record reads only the fields this
/// build knows and ignores the tail.
#[test]
fn accepts_struct_size_larger_than_known_layout() {
    #[repr(C)]
    struct ExtendedEvent {
        known: RawRuntimeEvent,
        appended: [u8; 64],
    }
    let mut extended = ExtendedEvent {
        known: raw_event(RawRuntimeEventKind::MODEL_OPEN_PROGRESS, 9),
        appended: [0xAB; 64],
    };
    extended.known.struct_size = std::mem::size_of::<ExtendedEvent>() as u32;
    extended.known.numeric_summary_2 = 42;

    let record = unsafe { NativeEventRecord::from_raw_ptr(&extended.known) }
        .expect("a larger struct_size is forward-compatible");

    assert_eq!(record.sequence, 9);
    assert_eq!(
        record.struct_size as usize,
        std::mem::size_of::<ExtendedEvent>()
    );
    assert_eq!(record.to_event().numeric_summary_2, Some(42));
}

#[test]
fn null_and_short_struct_are_distinct_rejections() {
    let mut short = raw_event(RawRuntimeEventKind::MODEL_OPEN_STARTED, 1);
    short.struct_size = std::mem::offset_of!(RawRuntimeEvent, numeric_summary_0) as u32 - 1;

    let null = unsafe { NativeEventRecord::from_raw_ptr(std::ptr::null()) };
    let short = unsafe { NativeEventRecord::from_raw_ptr(&short) };

    assert_eq!(null, Err(RecordRejection::Null));
    assert_eq!(short, Err(RecordRejection::ShortStruct));
}

#[test]
fn oversized_detail_is_its_own_rejection() {
    let mut event = raw_event(RawRuntimeEventKind::MODEL_OPEN_PROGRESS, 2);
    event.detail_len = u64::MAX;
    let rejection = unsafe { NativeEventRecord::from_raw_ptr(&event) };
    assert_eq!(rejection, Err(RecordRejection::OversizedDetail));
}
