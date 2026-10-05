//! Metal capacity probe shared by both model CLIs.

struct RetainedObjcObject(*mut std::ffi::c_void);

impl Drop for RetainedObjcObject {
    /// Releases the retained object, balancing the "create rule" retain.
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        #[link(name = "objc")]
        unsafe extern "C" {
            fn objc_release(obj: *mut std::ffi::c_void);
        }
        // SAFETY: the guard owns one retain returned by a Metal create/copy function.
        unsafe { objc_release(self.0) };
    }
}

/// Sends an argument-less Objective-C message and returns the raw result.
/// # Safety
/// The live receiver must implement the argument-less selector with a pointer-sized return.
unsafe fn msg_send(receiver: *mut std::ffi::c_void, selector: &std::ffi::CStr) -> usize {
    use std::ffi::{c_char, c_void};

    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend(receiver: *mut c_void, selector: *mut c_void, ...) -> usize;
    }

    // SAFETY: the caller guarantees the receiver and selector ABI; CStr is terminated.
    unsafe { objc_msgSend(receiver, sel_registerName(selector.as_ptr())) }
}

/// Runs `query` against the Metal device the survey describes.
///
/// That is the system default device. When `MTLCreateSystemDefaultDevice`
/// returns nil -- as it does on macOS 14 in a command-line process that has
/// not loaded CoreGraphics -- it is the first device `MTLCopyAllDevices`
/// lists instead. That list is unordered, so on a multi-GPU Mac it may not be
/// the device the system would pick; Apple Silicon has one GPU, so there the
/// two agree. Without the fallback the survey reported no GPU at all.
fn with_metal_device<T>(query: impl FnOnce(*mut std::ffi::c_void) -> Option<T>) -> Option<T> {
    use std::ffi::c_void;

    type CreateFn = unsafe extern "C" fn() -> *mut c_void;

    // SAFETY: Metal exports the declared create/copy signatures. RAII guards balance
    // retained objects before the library unloads; NSArray keeps borrowed devices alive.
    unsafe {
        let metal =
            libloading::Library::new("/System/Library/Frameworks/Metal.framework/Versions/A/Metal")
                .ok()?;
        let default_device = RetainedObjcObject(metal
            .get::<CreateFn>(b"MTLCreateSystemDefaultDevice")
            .ok()?());
        if !default_device.0.is_null() {
            return query(default_device.0);
        }
        // The array owns its devices, so the first one stays valid while `devices` lives.
        let devices = RetainedObjcObject(metal.get::<CreateFn>(b"MTLCopyAllDevices").ok()?());
        if devices.0.is_null() {
            return None;
        }
        let first = msg_send(devices.0, c"firstObject") as *mut c_void;
        if first.is_null() { None } else { query(first) }
    }
}

/// Queries the Metal-recommended working-set size in bytes for the survey's
/// device (see `with_metal_device`) — best-effort, OS-reported, not a
/// verified measurement.
pub(super) fn recommended_working_set_bytes() -> Option<u64> {
    with_metal_device(|device| {
        // SAFETY: the retained Metal device implements this NSUInteger-returning selector.
        let bytes = unsafe { msg_send(device, c"recommendedMaxWorkingSetSize") } as u64;
        (bytes > 0).then_some(bytes)
    })
}
