use std::ffi::{c_char, c_void};
use std::mem;
use std::ptr;
use std::sync::OnceLock;

use skippy_ffi::{
    Error as RawError, Model as RawModel, SkippyRuntimeEventCategory as RawRuntimeEventCategory,
    SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
    SkippyRuntimeEventKind as RawRuntimeEventKind,
    SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
    SkippyRuntimeEventReporterV1 as RawRuntimeEventReporter,
    SkippyRuntimeEventV1 as RawRuntimeEvent, Status,
};

const RUNTIME_EVENT_V1_ABI_VERSION: u32 = 1;

pub(crate) type RawModelOpenWithEventsFn = unsafe extern "C" fn(
    path: *const c_char,
    config: *const skippy_ffi::RuntimeConfig,
    reporter: *const RawRuntimeEventReporter,
    out_model: *mut *mut RawModel,
    out_error: *mut *mut RawError,
) -> Status;

pub(crate) type RawModelOpenFromPartsWithEventsFn = unsafe extern "C" fn(
    paths: *const *const c_char,
    path_count: usize,
    config: *const skippy_ffi::RuntimeConfig,
    reporter: *const RawRuntimeEventReporter,
    out_model: *mut *mut RawModel,
    out_error: *mut *mut RawError,
) -> Status;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventCategory {
    ModelOpen,
    Backend,
    Session,
    Kv,
    Warning,
    Unknown(u32),
}

impl From<RawRuntimeEventCategory> for RuntimeEventCategory {
    fn from(value: RawRuntimeEventCategory) -> Self {
        match value {
            RawRuntimeEventCategory::MODEL_OPEN => Self::ModelOpen,
            RawRuntimeEventCategory::BACKEND => Self::Backend,
            RawRuntimeEventCategory::SESSION => Self::Session,
            RawRuntimeEventCategory::KV => Self::Kv,
            RawRuntimeEventCategory::WARNING => Self::Warning,
            RawRuntimeEventCategory(raw) => Self::Unknown(raw),
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
    Unknown(u32),
}

impl From<RawRuntimeEventKind> for RuntimeEventKind {
    fn from(value: RawRuntimeEventKind) -> Self {
        match value {
            RawRuntimeEventKind::MODEL_OPEN_STARTED => Self::ModelOpenStarted,
            RawRuntimeEventKind::MODEL_OPEN_PROGRESS => Self::ModelOpenProgress,
            RawRuntimeEventKind::BACKEND_DEVICE_SELECTED => Self::BackendDeviceSelected,
            RawRuntimeEventKind::MODEL_OPEN_FINISHED => Self::ModelOpenFinished,
            RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED => Self::ModelOpenFailedHandled,
            RawRuntimeEventKind(raw) => Self::Unknown(raw),
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

impl From<RawRuntimeEventEmitterKind> for RuntimeEventEmitterKind {
    fn from(value: RawRuntimeEventEmitterKind) -> Self {
        match value {
            RawRuntimeEventEmitterKind::UNKNOWN => Self::Unknown,
            RawRuntimeEventEmitterKind::OPEN_THREAD => Self::OpenThread,
            RawRuntimeEventEmitterKind::WORKER_THREAD => Self::WorkerThread,
            RawRuntimeEventEmitterKind(raw) => Self::Other(raw),
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

impl From<RawRuntimeEventProgressUnit> for RuntimeEventProgressUnit {
    fn from(value: RawRuntimeEventProgressUnit) -> Self {
        match value {
            RawRuntimeEventProgressUnit::NONE => Self::None,
            RawRuntimeEventProgressUnit::BYTES => Self::Bytes,
            RawRuntimeEventProgressUnit::ITEMS => Self::Items,
            RawRuntimeEventProgressUnit::TENSORS => Self::Tensors,
            RawRuntimeEventProgressUnit::STEPS => Self::Steps,
            RawRuntimeEventProgressUnit(raw) => Self::Unknown(raw),
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

impl From<RawRuntimeEventFailureCode> for RuntimeEventFailureCode {
    fn from(value: RawRuntimeEventFailureCode) -> Self {
        match value {
            RawRuntimeEventFailureCode::NONE => Self::None,
            RawRuntimeEventFailureCode::INVALID_ARGUMENT => Self::InvalidArgument,
            RawRuntimeEventFailureCode::IO_ERROR => Self::IoError,
            RawRuntimeEventFailureCode::MODEL_ERROR => Self::ModelError,
            RawRuntimeEventFailureCode::RUNTIME_ERROR => Self::RuntimeError,
            RawRuntimeEventFailureCode::BACKEND_ERROR => Self::BackendError,
            RawRuntimeEventFailureCode::CANCELLED => Self::Cancelled,
            RawRuntimeEventFailureCode::INTERNAL_ERROR => Self::InternalError,
            RawRuntimeEventFailureCode(raw) => Self::Unknown(raw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEvent {
    pub abi_version: u32,
    pub category: RuntimeEventCategory,
    pub kind: RuntimeEventKind,
    pub emitter: RuntimeEventEmitterKind,
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
    pub detail_bytes: Vec<u8>,
}

impl RuntimeEvent {
    pub(crate) fn from_raw_ptr(event: *const RawRuntimeEvent) -> Option<Self> {
        if event.is_null() {
            return None;
        }
        let event = unsafe { &*event };
        if event.struct_size < mem::size_of::<RawRuntimeEvent>() as u32 {
            return None;
        }
        let detail_len = usize::try_from(event.detail_len).ok()?;
        let detail_bytes = if detail_len == 0 || event.detail_ptr.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(event.detail_ptr.cast::<u8>(), detail_len) }
                .to_vec()
        };
        Some(Self {
            abi_version: event.abi_version,
            category: event.category.into(),
            kind: event.kind.into(),
            emitter: event.emitter.into(),
            sequence: event.sequence,
            timestamp_mono_ns: event.timestamp_mono_ns,
            model_id: event.model_id,
            stage_id: event.stage_id,
            session_id: event.session_id,
            progress_current: event.progress_current,
            progress_total: event.progress_total,
            progress_unit: event.progress_unit.into(),
            failure_code: event.failure_code.into(),
            status: event.status,
            detail_bytes,
        })
    }
}

struct ModelOpenEventBridge<'a> {
    event_reporter: &'a mut dyn FnMut(RuntimeEvent),
}

struct ModelOpenEventReporterRegistration<'a> {
    _bridge: Box<ModelOpenEventBridge<'a>>,
    reporter: RawRuntimeEventReporter,
}

impl<'a> ModelOpenEventReporterRegistration<'a> {
    fn new(event_reporter: &'a mut dyn FnMut(RuntimeEvent)) -> Self {
        let mut bridge = Box::new(ModelOpenEventBridge { event_reporter });
        let reporter = RawRuntimeEventReporter {
            abi_version: RUNTIME_EVENT_V1_ABI_VERSION,
            struct_size: mem::size_of::<RawRuntimeEventReporter>() as u32,
            callback: Some(model_open_event_trampoline),
            user_data: bridge.as_mut() as *mut ModelOpenEventBridge<'a> as *mut c_void,
        };
        Self {
            _bridge: bridge,
            reporter,
        }
    }

    fn reporter_ptr(&self) -> *const RawRuntimeEventReporter {
        &self.reporter
    }
}

unsafe extern "C" fn model_open_event_trampoline(
    event: *const RawRuntimeEvent,
    user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            return;
        }
        let Some(event) = RuntimeEvent::from_raw_ptr(event) else {
            return;
        };
        let bridge = unsafe { &mut *(user_data as *mut ModelOpenEventBridge<'_>) };
        (bridge.event_reporter)(event);
    }));
}

fn collect_model_open_events<OpenFn, EventFn>(
    open_fn: OpenFn,
    mut event_reporter: EventFn,
) -> (*mut RawModel, Status, *mut RawError)
where
    OpenFn:
        FnOnce(*const RawRuntimeEventReporter, *mut *mut RawModel, *mut *mut RawError) -> Status,
    EventFn: FnMut(RuntimeEvent),
{
    let registration = ModelOpenEventReporterRegistration::new(&mut event_reporter);
    let mut raw = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = open_fn(registration.reporter_ptr(), &mut raw, &mut error);
    (raw, status, error)
}

pub(crate) fn run_model_open<OpenFn, OpenWithEventsFn>(
    open_fn: OpenFn,
    open_with_events_fn: OpenWithEventsFn,
    event_reporter: Option<&mut dyn FnMut(RuntimeEvent)>,
    use_event_reporter: bool,
) -> (*mut RawModel, Status, *mut RawError)
where
    OpenFn: FnOnce(*mut *mut RawModel, *mut *mut RawError) -> Status,
    OpenWithEventsFn:
        FnOnce(*const RawRuntimeEventReporter, *mut *mut RawModel, *mut *mut RawError) -> Status,
{
    match (event_reporter, use_event_reporter) {
        (Some(event_reporter), true) => {
            collect_model_open_events(open_with_events_fn, event_reporter)
        }
        _ => {
            let mut raw = ptr::null_mut();
            let mut error = ptr::null_mut();
            let status = open_fn(&mut raw, &mut error);
            (raw, status, error)
        }
    }
}

#[cfg(all(unix, not(feature = "dynamic-native-runtime")))]
fn lookup_model_open_with_events_symbol(name: &[u8]) -> Option<*mut c_void> {
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr().cast()) };
    (!symbol.is_null()).then_some(symbol)
}

#[cfg(all(not(unix), not(feature = "dynamic-native-runtime")))]
fn lookup_model_open_with_events_symbol(_name: &[u8]) -> Option<*mut c_void> {
    None
}

pub(crate) fn model_open_with_events_symbol() -> Option<RawModelOpenWithEventsFn> {
    static SYMBOL: OnceLock<Option<RawModelOpenWithEventsFn>> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        #[cfg(feature = "dynamic-native-runtime")]
        {
            skippy_ffi::skippy_model_open_with_events_fn()
        }
        #[cfg(not(feature = "dynamic-native-runtime"))]
        {
            lookup_model_open_with_events_symbol(b"skippy_model_open_with_events\0").map(
                |symbol| unsafe {
                    std::mem::transmute::<*mut c_void, RawModelOpenWithEventsFn>(symbol)
                },
            )
        }
    })
}

pub(crate) fn model_open_from_parts_with_events_symbol() -> Option<RawModelOpenFromPartsWithEventsFn>
{
    static SYMBOL: OnceLock<Option<RawModelOpenFromPartsWithEventsFn>> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        #[cfg(feature = "dynamic-native-runtime")]
        {
            skippy_ffi::skippy_model_open_from_parts_with_events_fn()
        }
        #[cfg(not(feature = "dynamic-native-runtime"))]
        {
            lookup_model_open_with_events_symbol(b"skippy_model_open_from_parts_with_events\0").map(
                |symbol| unsafe {
                    std::mem::transmute::<*mut c_void, RawModelOpenFromPartsWithEventsFn>(symbol)
                },
            )
        }
    })
}

pub(crate) fn model_open_events_supported() -> bool {
    skippy_ffi::ABI_VERSION_MAJOR == 0
        && skippy_ffi::ABI_VERSION_MINOR == 1
        && skippy_ffi::ABI_VERSION_PATCH >= 26
        && skippy_ffi::native_runtime_loaded()
        && abi_features_bitmask()
            .is_some_and(|features| (features & skippy_ffi::FEATURE_RUNTIME_EVENTS) != 0)
        && model_open_with_events_symbol().is_some()
        && model_open_from_parts_with_events_symbol().is_some()
}

fn abi_features_bitmask() -> Option<u64> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_abi_features_optional().map(|features| unsafe { features() })
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        Some(unsafe { skippy_ffi::skippy_abi_features() })
    }
}

#[cfg(test)]
pub(crate) mod tests;
