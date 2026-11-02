use std::mem;

use skippy_ffi::{
    RUNTIME_EVENT_CATEGORY_BACKEND, RUNTIME_EVENT_CATEGORY_DEVICE,
    RUNTIME_EVENT_CATEGORY_DIAGNOSTIC, RUNTIME_EVENT_CATEGORY_KV,
    RUNTIME_EVENT_CATEGORY_MODEL_OPEN, RUNTIME_EVENT_CATEGORY_SESSION,
    RUNTIME_EVENT_CATEGORY_UNLOAD, RUNTIME_EVENT_CATEGORY_WARNING,
    RUNTIME_EVENT_EMITTER_OPEN_THREAD, RUNTIME_EVENT_EMITTER_UNKNOWN,
    RUNTIME_EVENT_EMITTER_WORKER_THREAD, RUNTIME_EVENT_FAILURE_BACKEND_ERROR,
    RUNTIME_EVENT_FAILURE_CANCELLED, RUNTIME_EVENT_FAILURE_INTERNAL_ERROR,
    RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT, RUNTIME_EVENT_FAILURE_IO_ERROR,
    RUNTIME_EVENT_FAILURE_MODEL_ERROR, RUNTIME_EVENT_FAILURE_NONE,
    RUNTIME_EVENT_FAILURE_RUNTIME_ERROR, RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED,
    RUNTIME_EVENT_KIND_DEVICE_BACKEND_INITIALIZED, RUNTIME_EVENT_KIND_DEVICE_DEGRADED,
    RUNTIME_EVENT_KIND_DEVICE_FALLBACK_ACTIVATED, RUNTIME_EVENT_KIND_DEVICE_LOST,
    RUNTIME_EVENT_KIND_DEVICE_OUT_OF_MEMORY, RUNTIME_EVENT_KIND_DEVICE_READY,
    RUNTIME_EVENT_KIND_DEVICE_RECOVERED, RUNTIME_EVENT_KIND_DEVICE_RESOURCE_ALLOCATED,
    RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE, RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE,
    RUNTIME_EVENT_KIND_DIAGNOSTIC_INVARIANT_VIOLATION,
    RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE,
    RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_CLEARED, RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED,
    RUNTIME_EVENT_KIND_KV_CONTEXT_APPROACHING_CAPACITY,
    RUNTIME_EVENT_KIND_KV_CONTEXT_CAPACITY_EXHAUSTED, RUNTIME_EVENT_KIND_KV_INITIALIZED,
    RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED, RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED,
    RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY,
    RUNTIME_EVENT_KIND_MODEL_LOAD_MEMORY_ALLOCATED, RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED,
    RUNTIME_EVENT_KIND_MODEL_LOAD_TENSORS_OFFLOADED, RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY,
    RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED, RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED,
    RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS, RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED,
    RUNTIME_EVENT_KIND_UNLOAD_COMPLETED, RUNTIME_EVENT_KIND_UNLOAD_FAILED,
    RUNTIME_EVENT_KIND_UNLOAD_FORCED, RUNTIME_EVENT_KIND_UNLOAD_SESSION_DRAINING,
    RUNTIME_EVENT_KIND_UNLOAD_STARTED, RUNTIME_EVENT_PROGRESS_UNIT_BYTES,
    RUNTIME_EVENT_PROGRESS_UNIT_ITEMS, RUNTIME_EVENT_PROGRESS_UNIT_NONE,
    RUNTIME_EVENT_PROGRESS_UNIT_STEPS, RUNTIME_EVENT_PROGRESS_UNIT_TENSORS,
};
use skippy_ffi::{SkippyRuntimeEventV1 as RawRuntimeEvent, Status};

#[cfg(test)]
use super::NativeEventRecord;

/// Safety bound on a native event's `detail_len`. A malformed or hostile
/// value here must never drive an unbounded copy; anything past this is
/// rejected rather than trusted.
pub(super) const MAX_DETAIL_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventCategory {
    ModelOpen,
    Backend,
    Session,
    Kv,
    Warning,
    Device,
    Diagnostic,
    Unload,
    Unknown(u32),
}

impl From<skippy_ffi::SkippyRuntimeEventCategory> for RuntimeEventCategory {
    fn from(value: skippy_ffi::SkippyRuntimeEventCategory) -> Self {
        match value {
            RUNTIME_EVENT_CATEGORY_MODEL_OPEN => Self::ModelOpen,
            RUNTIME_EVENT_CATEGORY_BACKEND => Self::Backend,
            RUNTIME_EVENT_CATEGORY_SESSION => Self::Session,
            RUNTIME_EVENT_CATEGORY_KV => Self::Kv,
            RUNTIME_EVENT_CATEGORY_WARNING => Self::Warning,
            RUNTIME_EVENT_CATEGORY_DEVICE => Self::Device,
            RUNTIME_EVENT_CATEGORY_DIAGNOSTIC => Self::Diagnostic,
            RUNTIME_EVENT_CATEGORY_UNLOAD => Self::Unload,
            raw => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventCategory {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::ModelOpen => RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
            Self::Backend => RUNTIME_EVENT_CATEGORY_BACKEND,
            Self::Session => RUNTIME_EVENT_CATEGORY_SESSION,
            Self::Kv => RUNTIME_EVENT_CATEGORY_KV,
            Self::Warning => RUNTIME_EVENT_CATEGORY_WARNING,
            Self::Device => RUNTIME_EVENT_CATEGORY_DEVICE,
            Self::Diagnostic => RUNTIME_EVENT_CATEGORY_DIAGNOSTIC,
            Self::Unload => RUNTIME_EVENT_CATEGORY_UNLOAD,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventKind {
    ModelOpenStarted,
    ModelOpenProgress,
    BackendDeviceSelected,
    ModelOpenFinished,
    ModelOpenFailedHandled,
    ModelLoadPhaseChanged,
    ModelLoadMemoryAllocated,
    ModelLoadTensorsOffloaded,
    ModelLoadTokenizerReady,
    ModelLoadAuxComponentReady,
    KvInitialized,
    KvPressureCrossed,
    KvPressureCleared,
    KvContextApproachingCapacity,
    KvContextCapacityExhausted,
    DeviceBackendInitialized,
    DeviceReady,
    DeviceDegraded,
    DeviceUnavailable,
    DeviceRecovered,
    DeviceLost,
    DeviceResourceAllocated,
    DeviceOutOfMemory,
    DeviceFallbackActivated,
    DiagnosticWarningRaised,
    DiagnosticWarningCleared,
    DiagnosticRecoverableFailure,
    DiagnosticFatalFailure,
    DiagnosticInvariantViolation,
    UnloadStarted,
    UnloadCompleted,
    UnloadFailed,
    UnloadForced,
    UnloadSessionDraining,
    Unknown(u32),
}

impl From<skippy_ffi::SkippyRuntimeEventKind> for RuntimeEventKind {
    fn from(value: skippy_ffi::SkippyRuntimeEventKind) -> Self {
        match value {
            RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED => Self::ModelOpenStarted,
            RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS => Self::ModelOpenProgress,
            RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED => Self::BackendDeviceSelected,
            RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED => Self::ModelOpenFinished,
            RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED => Self::ModelOpenFailedHandled,
            RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED => Self::ModelLoadPhaseChanged,
            RUNTIME_EVENT_KIND_MODEL_LOAD_MEMORY_ALLOCATED => Self::ModelLoadMemoryAllocated,
            RUNTIME_EVENT_KIND_MODEL_LOAD_TENSORS_OFFLOADED => Self::ModelLoadTensorsOffloaded,
            RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY => Self::ModelLoadTokenizerReady,
            RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY => Self::ModelLoadAuxComponentReady,
            RUNTIME_EVENT_KIND_KV_INITIALIZED => Self::KvInitialized,
            RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED => Self::KvPressureCrossed,
            RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED => Self::KvPressureCleared,
            RUNTIME_EVENT_KIND_KV_CONTEXT_APPROACHING_CAPACITY => {
                Self::KvContextApproachingCapacity
            }
            RUNTIME_EVENT_KIND_KV_CONTEXT_CAPACITY_EXHAUSTED => Self::KvContextCapacityExhausted,
            RUNTIME_EVENT_KIND_DEVICE_BACKEND_INITIALIZED => Self::DeviceBackendInitialized,
            RUNTIME_EVENT_KIND_DEVICE_READY => Self::DeviceReady,
            RUNTIME_EVENT_KIND_DEVICE_DEGRADED => Self::DeviceDegraded,
            RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE => Self::DeviceUnavailable,
            RUNTIME_EVENT_KIND_DEVICE_RECOVERED => Self::DeviceRecovered,
            RUNTIME_EVENT_KIND_DEVICE_LOST => Self::DeviceLost,
            RUNTIME_EVENT_KIND_DEVICE_RESOURCE_ALLOCATED => Self::DeviceResourceAllocated,
            RUNTIME_EVENT_KIND_DEVICE_OUT_OF_MEMORY => Self::DeviceOutOfMemory,
            RUNTIME_EVENT_KIND_DEVICE_FALLBACK_ACTIVATED => Self::DeviceFallbackActivated,
            RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED => Self::DiagnosticWarningRaised,
            RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_CLEARED => Self::DiagnosticWarningCleared,
            RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE => Self::DiagnosticRecoverableFailure,
            RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE => Self::DiagnosticFatalFailure,
            RUNTIME_EVENT_KIND_DIAGNOSTIC_INVARIANT_VIOLATION => Self::DiagnosticInvariantViolation,
            RUNTIME_EVENT_KIND_UNLOAD_STARTED => Self::UnloadStarted,
            RUNTIME_EVENT_KIND_UNLOAD_COMPLETED => Self::UnloadCompleted,
            RUNTIME_EVENT_KIND_UNLOAD_FAILED => Self::UnloadFailed,
            RUNTIME_EVENT_KIND_UNLOAD_FORCED => Self::UnloadForced,
            RUNTIME_EVENT_KIND_UNLOAD_SESSION_DRAINING => Self::UnloadSessionDraining,
            raw => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventKind {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::ModelOpenStarted => RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED,
            Self::ModelOpenProgress => RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS,
            Self::BackendDeviceSelected => RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED,
            Self::ModelOpenFinished => RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED,
            Self::ModelOpenFailedHandled => RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED,
            Self::ModelLoadPhaseChanged => RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED,
            Self::ModelLoadMemoryAllocated => RUNTIME_EVENT_KIND_MODEL_LOAD_MEMORY_ALLOCATED,
            Self::ModelLoadTensorsOffloaded => RUNTIME_EVENT_KIND_MODEL_LOAD_TENSORS_OFFLOADED,
            Self::ModelLoadTokenizerReady => RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY,
            Self::ModelLoadAuxComponentReady => RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY,
            Self::KvInitialized => RUNTIME_EVENT_KIND_KV_INITIALIZED,
            Self::KvPressureCrossed => RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED,
            Self::KvPressureCleared => RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED,
            Self::KvContextApproachingCapacity => {
                RUNTIME_EVENT_KIND_KV_CONTEXT_APPROACHING_CAPACITY
            }
            Self::KvContextCapacityExhausted => RUNTIME_EVENT_KIND_KV_CONTEXT_CAPACITY_EXHAUSTED,
            Self::DeviceBackendInitialized => RUNTIME_EVENT_KIND_DEVICE_BACKEND_INITIALIZED,
            Self::DeviceReady => RUNTIME_EVENT_KIND_DEVICE_READY,
            Self::DeviceDegraded => RUNTIME_EVENT_KIND_DEVICE_DEGRADED,
            Self::DeviceUnavailable => RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE,
            Self::DeviceRecovered => RUNTIME_EVENT_KIND_DEVICE_RECOVERED,
            Self::DeviceLost => RUNTIME_EVENT_KIND_DEVICE_LOST,
            Self::DeviceResourceAllocated => RUNTIME_EVENT_KIND_DEVICE_RESOURCE_ALLOCATED,
            Self::DeviceOutOfMemory => RUNTIME_EVENT_KIND_DEVICE_OUT_OF_MEMORY,
            Self::DeviceFallbackActivated => RUNTIME_EVENT_KIND_DEVICE_FALLBACK_ACTIVATED,
            Self::DiagnosticWarningRaised => RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED,
            Self::DiagnosticWarningCleared => RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_CLEARED,
            Self::DiagnosticRecoverableFailure => RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE,
            Self::DiagnosticFatalFailure => RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE,
            Self::DiagnosticInvariantViolation => RUNTIME_EVENT_KIND_DIAGNOSTIC_INVARIANT_VIOLATION,
            Self::UnloadStarted => RUNTIME_EVENT_KIND_UNLOAD_STARTED,
            Self::UnloadCompleted => RUNTIME_EVENT_KIND_UNLOAD_COMPLETED,
            Self::UnloadFailed => RUNTIME_EVENT_KIND_UNLOAD_FAILED,
            Self::UnloadForced => RUNTIME_EVENT_KIND_UNLOAD_FORCED,
            Self::UnloadSessionDraining => RUNTIME_EVENT_KIND_UNLOAD_SESSION_DRAINING,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventEmitterKind {
    Unknown,
    OpenThread,
    WorkerThread,
    Other(u32),
}

impl From<skippy_ffi::SkippyRuntimeEventEmitterKind> for RuntimeEventEmitterKind {
    fn from(value: skippy_ffi::SkippyRuntimeEventEmitterKind) -> Self {
        match value {
            RUNTIME_EVENT_EMITTER_UNKNOWN => Self::Unknown,
            RUNTIME_EVENT_EMITTER_OPEN_THREAD => Self::OpenThread,
            RUNTIME_EVENT_EMITTER_WORKER_THREAD => Self::WorkerThread,
            raw => Self::Other(raw),
        }
    }
}

impl RuntimeEventEmitterKind {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Unknown => RUNTIME_EVENT_EMITTER_UNKNOWN,
            Self::OpenThread => RUNTIME_EVENT_EMITTER_OPEN_THREAD,
            Self::WorkerThread => RUNTIME_EVENT_EMITTER_WORKER_THREAD,
            Self::Other(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventProgressUnit {
    None,
    Bytes,
    Items,
    Tensors,
    Steps,
    Unknown(u32),
}

impl From<skippy_ffi::SkippyRuntimeEventProgressUnit> for RuntimeEventProgressUnit {
    fn from(value: skippy_ffi::SkippyRuntimeEventProgressUnit) -> Self {
        match value {
            RUNTIME_EVENT_PROGRESS_UNIT_NONE => Self::None,
            RUNTIME_EVENT_PROGRESS_UNIT_BYTES => Self::Bytes,
            RUNTIME_EVENT_PROGRESS_UNIT_ITEMS => Self::Items,
            RUNTIME_EVENT_PROGRESS_UNIT_TENSORS => Self::Tensors,
            RUNTIME_EVENT_PROGRESS_UNIT_STEPS => Self::Steps,
            raw => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventProgressUnit {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::None => RUNTIME_EVENT_PROGRESS_UNIT_NONE,
            Self::Bytes => RUNTIME_EVENT_PROGRESS_UNIT_BYTES,
            Self::Items => RUNTIME_EVENT_PROGRESS_UNIT_ITEMS,
            Self::Tensors => RUNTIME_EVENT_PROGRESS_UNIT_TENSORS,
            Self::Steps => RUNTIME_EVENT_PROGRESS_UNIT_STEPS,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventFailureCode {
    None,
    InvalidArgument,
    IoError,
    ModelError,
    RuntimeError,
    BackendError,
    Cancelled,
    InternalError,
    Unknown(u32),
}

impl From<skippy_ffi::SkippyRuntimeEventFailureCode> for RuntimeEventFailureCode {
    fn from(value: skippy_ffi::SkippyRuntimeEventFailureCode) -> Self {
        match value {
            RUNTIME_EVENT_FAILURE_NONE => Self::None,
            RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT => Self::InvalidArgument,
            RUNTIME_EVENT_FAILURE_IO_ERROR => Self::IoError,
            RUNTIME_EVENT_FAILURE_MODEL_ERROR => Self::ModelError,
            RUNTIME_EVENT_FAILURE_RUNTIME_ERROR => Self::RuntimeError,
            RUNTIME_EVENT_FAILURE_BACKEND_ERROR => Self::BackendError,
            RUNTIME_EVENT_FAILURE_CANCELLED => Self::Cancelled,
            RUNTIME_EVENT_FAILURE_INTERNAL_ERROR => Self::InternalError,
            raw => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventFailureCode {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::None => RUNTIME_EVENT_FAILURE_NONE,
            Self::InvalidArgument => RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT,
            Self::IoError => RUNTIME_EVENT_FAILURE_IO_ERROR,
            Self::ModelError => RUNTIME_EVENT_FAILURE_MODEL_ERROR,
            Self::RuntimeError => RUNTIME_EVENT_FAILURE_RUNTIME_ERROR,
            Self::BackendError => RUNTIME_EVENT_FAILURE_BACKEND_ERROR,
            Self::Cancelled => RUNTIME_EVENT_FAILURE_CANCELLED,
            Self::InternalError => RUNTIME_EVENT_FAILURE_INTERNAL_ERROR,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEvent {
    pub abi_version: u32,
    pub struct_size: u32,
    pub category: RuntimeEventCategory,
    pub kind: RuntimeEventKind,
    pub emitter: RuntimeEventEmitterKind,
    pub reserved0: u32,
    pub sequence: u64,
    pub timestamp_mono_ns: u64,
    pub model_id: u64,
    pub stage_id: u64,
    pub session_id: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub progress_unit: RuntimeEventProgressUnit,
    pub failure_code: RuntimeEventFailureCode,
    pub status: Status,
    pub reserved1: u32,
    pub detail_bytes: Vec<u8>,
    // `None` when the emitting native runtime's `struct_size` did not cover
    // these append-only fields (native ABI 0.1.44 or earlier, or a build
    // without SKIPPY_FEATURE_RUNTIME_EVENT_REPORTER) -- never read as zero.
    pub numeric_summary_0: Option<u64>,
    pub numeric_summary_1: Option<u64>,
    pub numeric_summary_2: Option<u64>,
    pub numeric_summary_3: Option<u64>,
}

/// Mirrors `RawRuntimeEvent`'s original (pre-extension) prefix exactly, field
/// for field. Used only to bound how much of the allocation is safe to read
/// as a full struct reference when `struct_size` covers the base layout but
/// not the four appended `numeric_summary_*` fields: casting to this type's
/// reference reads only the common initial sequence, never past what
/// `struct_size` proved is allocated.
#[repr(C)]
pub(super) struct BaseRawRuntimeEvent {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) category: skippy_ffi::SkippyRuntimeEventCategory,
    pub(super) kind: skippy_ffi::SkippyRuntimeEventKind,
    pub(super) emitter: skippy_ffi::SkippyRuntimeEventEmitterKind,
    pub(super) reserved0: u32,
    pub(super) sequence: u64,
    pub(super) timestamp_mono_ns: u64,
    pub(super) model_id: u64,
    pub(super) stage_id: u64,
    pub(super) session_id: u64,
    pub(super) progress_current: u64,
    pub(super) progress_total: u64,
    pub(super) progress_unit: skippy_ffi::SkippyRuntimeEventProgressUnit,
    pub(super) failure_code: skippy_ffi::SkippyRuntimeEventFailureCode,
    /// The C ABI stores status as a plain `int32_t`; decode with
    /// `skippy_ffi::status_from_raw` before use so an out-of-range value
    /// never materializes as an invalid `Status`.
    pub(super) status: i32,
    pub(super) reserved1: u32,
    pub(super) detail_ptr: *const std::ffi::c_char,
    pub(super) detail_len: u64,
}

macro_rules! const_assert_eq {
    ($left:expr, $right:expr $(,)?) => {
        const _: () = assert!($left == $right);
    };
}

const_assert_eq!(
    mem::align_of::<BaseRawRuntimeEvent>(),
    mem::align_of::<RawRuntimeEvent>(),
);
const_assert_eq!(
    mem::size_of::<BaseRawRuntimeEvent>(),
    mem::offset_of!(RawRuntimeEvent, numeric_summary_0),
);

macro_rules! assert_shared_field_offsets {
    ($($field:ident),+ $(,)?) => {
        $(
            const_assert_eq!(
                mem::offset_of!(BaseRawRuntimeEvent, $field),
                mem::offset_of!(RawRuntimeEvent, $field),
            );
        )+
    };
}

assert_shared_field_offsets!(
    abi_version,
    struct_size,
    category,
    kind,
    emitter,
    reserved0,
    sequence,
    timestamp_mono_ns,
    model_id,
    stage_id,
    session_id,
    progress_current,
    progress_total,
    progress_unit,
    failure_code,
    status,
    reserved1,
    detail_ptr,
    detail_len,
);

impl RuntimeEvent {
    /// Single validation path: the same checks the callback boundary applies
    /// (see [`NativeEventRecord::from_raw_ptr`]), then expansion on the
    /// caller's thread. Production code consumes records from a queue; this
    /// one-shot form exists for tests.
    #[cfg(test)]
    pub(crate) fn from_raw_ptr(event: *const RawRuntimeEvent) -> Option<Self> {
        // SAFETY: callers pass null or a pointer honoring the reporter ABI
        // contract, which is exactly `NativeEventRecord::from_raw_ptr`'s.
        unsafe { NativeEventRecord::from_raw_ptr(event) }
            .ok()
            .map(|record| record.to_event())
    }
}
