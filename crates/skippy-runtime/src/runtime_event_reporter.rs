use std::ffi::c_void;
use std::mem;
use std::sync::{Mutex, OnceLock};

use skippy_ffi::{
    FEATURE_RUNTIME_EVENT_REPORTER, SkippyRuntimeEventReporterV1 as RawReporter,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use crate::capability_probe::probe_capabilities;
use crate::runtime_events::{RUNTIME_EVENT_V1_ABI_VERSION, RuntimeEvent};

type ReporterSink = Box<dyn FnMut(RuntimeEvent) + Send>;

static REPORTER_SINK: OnceLock<Mutex<Option<ReporterSink>>> = OnceLock::new();

fn sink_slot() -> &'static Mutex<Option<ReporterSink>> {
    REPORTER_SINK.get_or_init(|| Mutex::new(None))
}

/// Correlate-and-submit only: no formatting, logging, I/O, or blocking runs
/// on this native callback thread, matching the model-open trampoline's
/// contract in `runtime_events.rs`.
unsafe extern "C" fn runtime_reporter_trampoline(
    event: *const RawRuntimeEvent,
    _user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(event) = RuntimeEvent::from_raw_ptr(event) else {
            return;
        };
        if let Ok(mut guard) = sink_slot().lock()
            && let Some(sink) = guard.as_mut()
        {
            sink(event);
        }
    }));
}

type SetReporterFn = unsafe extern "C" fn(*const RawReporter) -> skippy_ffi::Status;
type ClearReporterFn = unsafe extern "C" fn();

fn set_reporter_fn() -> Option<SetReporterFn> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_set_runtime_event_reporter_fn()
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        static CACHE: OnceLock<Option<SetReporterFn>> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(unix)]
            {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"skippy_set_runtime_event_reporter".as_ptr(),
                    )
                };
                (!symbol.is_null())
                    .then(|| unsafe { std::mem::transmute::<*mut c_void, SetReporterFn>(symbol) })
            }
            #[cfg(not(unix))]
            {
                None
            }
        })
    }
}

fn clear_reporter_fn() -> Option<ClearReporterFn> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_clear_runtime_event_reporter_fn()
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        static CACHE: OnceLock<Option<ClearReporterFn>> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(unix)]
            {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"skippy_clear_runtime_event_reporter".as_ptr(),
                    )
                };
                (!symbol.is_null())
                    .then(|| unsafe { std::mem::transmute::<*mut c_void, ClearReporterFn>(symbol) })
            }
            #[cfg(not(unix))]
            {
                None
            }
        })
    }
}

/// Installs the runtime-scoped (process-global) event reporter, gated on the
/// probed `runtime_event_reporter` family. Returns `false` without touching
/// native state when the family is unavailable or a symbol failed to
/// resolve, so a caller can fall back cleanly on an older runtime.
pub fn install_runtime_event_reporter<F>(sink: F) -> bool
where
    F: FnMut(RuntimeEvent) + Send + 'static,
{
    if !probe_capabilities().family_confirmed(FEATURE_RUNTIME_EVENT_REPORTER) {
        return false;
    }
    let Some(set_fn) = set_reporter_fn() else {
        return false;
    };
    let Ok(mut guard) = sink_slot().lock() else {
        return false;
    };
    *guard = Some(Box::new(sink));
    drop(guard);

    let reporter = RawReporter {
        abi_version: RUNTIME_EVENT_V1_ABI_VERSION,
        struct_size: mem::size_of::<RawReporter>() as u32,
        callback: Some(runtime_reporter_trampoline),
        user_data: std::ptr::null_mut(),
    };
    let status = unsafe { set_fn(&reporter) };
    status == skippy_ffi::Status::Ok
}

/// Clears the runtime-scoped event reporter. Blocks (via the native
/// `skippy_clear_runtime_event_reporter` quiescence contract) until every
/// in-flight callback has returned before this function returns, so no
/// callback fires into a dropped sink. A no-op when nothing was installed.
pub fn clear_runtime_event_reporter() {
    if let Some(clear_fn) = clear_reporter_fn() {
        unsafe { clear_fn() };
    }
    if let Ok(mut guard) = sink_slot().lock() {
        *guard = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_returns_false_without_a_confirmed_family() {
        // No native runtime is loaded in unit tests, so the family probe
        // always reports unconfirmed; install must refuse cleanly rather
        // than dereference an absent native symbol.
        assert!(!install_runtime_event_reporter(|_event| {}));
    }

    #[test]
    fn clear_is_a_safe_no_op_when_nothing_was_installed() {
        clear_runtime_event_reporter();
    }
}
