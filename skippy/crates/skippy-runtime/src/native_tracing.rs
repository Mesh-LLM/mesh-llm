//! Where time goes inside native calls (`skippy/tracing.h`).
//!
//! Two native callbacks report what a timed call did: [`spans`] names the
//! phases of host work (llama graph build, encode, synchronize, sampling,
//! Metal library and pipeline compilation) on the thread that did it, and
//! [`gpu_commands`] reports each command buffer once the GPU has finished it.
//! Both carry the trace tag the calling thread set, and timed native calls
//! set the id their observer hands out (see [`crate::native_timing`]), so a
//! consumer can place everything a call caused under it.
//!
//! The callbacks only copy records into bounded lock-free rings; consumers
//! drain them on their own thread. Installation is deferred to the first
//! timed native call after [`request_native_tracing`], when the runtime
//! library is certainly loaded, and is itself observed as a native call:
//! installing reaches every backend, and initialising Metal compiles its
//! kernel libraries there.

use std::cell::Cell;
use std::ffi::{CStr, c_char};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use skippy_ffi::{
    SkippySetGpuCommandTimingCallbackFn, SkippySetNativeSpanCallbackFn, SkippySetTraceTagFn, Status,
};

pub mod gpu_commands;
pub mod spans;

const UNREQUESTED: u8 = 0;
const REQUESTED: u8 = 1;
const INSTALLING: u8 = 2;
const ACTIVE: u8 = 3;
const UNAVAILABLE: u8 = 4;

static STATE: AtomicU8 = AtomicU8::new(UNREQUESTED);
static TAG_FN: OnceLock<SkippySetTraceTagFn> = OnceLock::new();

thread_local! {
    /// The tag this thread last set, so a call nested in another (a native
    /// callback into Rust that calls back down) can restore its caller's.
    static TAG: Cell<u64> = const { Cell::new(0) };
}

/// Ask for native spans and GPU command timing. Nothing is installed until
/// the next timed native call; with no call observer, never.
pub fn request_native_tracing() {
    let _ = STATE.compare_exchange(UNREQUESTED, REQUESTED, Ordering::AcqRel, Ordering::Acquire);
}

/// Whether the runtime is reporting native spans.
pub fn native_tracing_active() -> bool {
    STATE.load(Ordering::Acquire) == ACTIVE
}

/// Install the native callbacks if tracing was requested and nobody has yet.
/// One caller installs; calls racing it on other threads run untagged.
pub(crate) fn ensure_installed() {
    if STATE
        .compare_exchange(REQUESTED, INSTALLING, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        let state = if install() { ACTIVE } else { UNAVAILABLE };
        STATE.store(state, Ordering::Release);
    }
}

/// Tag native spans and GPU work this thread causes from now on, returning
/// the tag it replaces; 0 stops tagging. The native tag is untouched unless
/// tracing is active.
pub(crate) fn swap_tag(tag: u64) -> u64 {
    let previous = TAG.with(|current| current.replace(tag));
    if STATE.load(Ordering::Acquire) == ACTIVE
        && let Some(set_tag) = TAG_FN.get()
    {
        // SAFETY: a thread-local store in the native runtime; takes no pointers.
        unsafe { set_tag(tag) };
    }
    previous
}

struct InstallFns {
    set_span_callback: SkippySetNativeSpanCallbackFn,
    set_gpu_callback: SkippySetGpuCommandTimingCallbackFn,
    set_tag: SkippySetTraceTagFn,
}

fn install() -> bool {
    let Some(fns) = install_fns() else {
        return false;
    };
    // Allocate before the first callback can arrive.
    spans::ring();
    gpu_commands::ring();
    let _ = TAG_FN.set(fns.set_tag);
    crate::native_timing::timed_with_id("skippy_set_native_span_callback", |id| {
        // Kernel libraries compiled while installing land under this call.
        // SAFETY: a thread-local store in the native runtime.
        unsafe { (fns.set_tag)(id) };
        // SAFETY: the trampolines match the ABI callback types, never
        // unwind, and use no user data.
        let spans =
            unsafe { (fns.set_span_callback)(Some(spans::trampoline), std::ptr::null_mut()) };
        // A runtime without a GPU backend has no command timing to report;
        // its native spans are still worth having.
        // SAFETY: as above.
        let _ =
            unsafe { (fns.set_gpu_callback)(Some(gpu_commands::trampoline), std::ptr::null_mut()) };
        // SAFETY: as above.
        unsafe { (fns.set_tag)(0) };
        spans == Status::Ok
    })
}

fn install_fns() -> Option<InstallFns> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        Some(InstallFns {
            set_span_callback: skippy_ffi::skippy_set_native_span_callback_fn()?,
            set_gpu_callback: skippy_ffi::skippy_set_gpu_command_timing_callback_fn()?,
            set_tag: skippy_ffi::skippy_set_trace_tag_fn()?,
        })
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        let set_span_callback = linked_symbol(c"skippy_set_native_span_callback")?;
        let set_gpu_callback = linked_symbol(c"skippy_set_gpu_command_timing_callback")?;
        let set_tag = linked_symbol(c"skippy_set_trace_tag")?;
        // SAFETY: each symbol, when linked, has exactly the transmuted signature.
        unsafe {
            Some(InstallFns {
                set_span_callback: std::mem::transmute::<
                    *mut std::ffi::c_void,
                    SkippySetNativeSpanCallbackFn,
                >(set_span_callback),
                set_gpu_callback: std::mem::transmute::<
                    *mut std::ffi::c_void,
                    SkippySetGpuCommandTimingCallbackFn,
                >(set_gpu_callback),
                set_tag: std::mem::transmute::<*mut std::ffi::c_void, SkippySetTraceTagFn>(set_tag),
            })
        }
    }
}

#[cfg(not(feature = "dynamic-native-runtime"))]
fn linked_symbol(name: &CStr) -> Option<*mut std::ffi::c_void> {
    #[cfg(unix)]
    {
        // SAFETY: looks up a symbol in the already-loaded process image.
        let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        None
    }
}

/// A NUL-terminated string the ABI gives static storage; `fallback` when null
/// or not UTF-8.
fn static_str(name: *const c_char, fallback: &'static str) -> &'static str {
    if name.is_null() {
        return fallback;
    }
    // SAFETY: the ABI gives these strings static storage.
    unsafe { CStr::from_ptr(name) }.to_str().unwrap_or(fallback)
}

fn unix_nanos(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagging_without_a_request_never_touches_the_runtime() {
        // No library is loaded in unit tests; reaching the FFI would panic.
        if STATE.load(Ordering::Acquire) == UNREQUESTED {
            assert_eq!(swap_tag(7), 0);
            ensure_installed();
        }
        assert!(!native_tracing_active());
    }

    #[test]
    fn a_nested_call_restores_its_callers_tag() {
        let outer = swap_tag(11);
        assert_eq!(swap_tag(12), 11);
        assert_eq!(swap_tag(11), 12);
        assert_eq!(swap_tag(outer), 11);
    }

    #[test]
    fn static_strings_fall_back_when_missing() {
        assert_eq!(static_str(std::ptr::null(), "gpu"), "gpu");
        assert_eq!(static_str(c"MTL".as_ptr(), "gpu"), "MTL");
    }
}
