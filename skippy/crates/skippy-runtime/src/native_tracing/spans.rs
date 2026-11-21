//! Native spans: the phases of host work inside a native call.
//!
//! The runtime reports each span when it ends, on the thread that did the
//! work, usually the one inside the timed call, so the callback only copies
//! the span into the ring. A span's detail is the one thing it allocates for,
//! and only spans that name what they worked on (a pipeline, a kernel
//! library) carry one. A push into a full ring drops the span and counts it.

use std::ffi::{CStr, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::ArrayQueue;
use skippy_ffi::{NATIVE_SPAN_ID_BIT, NATIVE_SPAN_V1_ABI_VERSION, SkippyNativeSpanV1 as RawSpan};

use super::{static_str, unix_nanos};

/// Spans the ring holds before it starts dropping: a few hundred decode
/// steps between drains.
pub const NATIVE_SPAN_RING_CAPACITY: usize = 8192;

/// One finished phase of native work. Times are Unix nanoseconds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeSpan {
    /// Process-unique; has [`NATIVE_SPAN_ID_BIT`] set.
    pub id: u64,
    /// The enclosing native span ([`is_native_span_id`]), else the trace tag
    /// of the thread that did the work, else 0.
    pub parent: u64,
    /// Dotted name such as `"llama.graph_compute"`.
    pub name: &'static str,
    /// What the span worked on, such as a Metal pipeline name.
    pub detail: Option<Box<str>>,
    pub start_unix_nanos: u64,
    pub end_unix_nanos: u64,
    /// One measured quantity, such as `("tokens", 4)`.
    pub value: Option<(&'static str, i64)>,
}

/// Whether `id` names a native span rather than a caller's trace tag.
pub fn is_native_span_id(id: u64) -> bool {
    id & NATIVE_SPAN_ID_BIT != 0
}

static RING: OnceLock<ArrayQueue<NativeSpan>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);

pub(super) fn ring() -> &'static ArrayQueue<NativeSpan> {
    RING.get_or_init(|| ArrayQueue::new(NATIVE_SPAN_RING_CAPACITY))
}

/// Spans lost to a full ring since the process started.
pub fn dropped_native_spans() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Hand every pending span to `consume`, in the order they ended.
pub fn drain_native_spans(mut consume: impl FnMut(NativeSpan)) {
    let Some(ring) = RING.get() else {
        return;
    };
    while let Some(span) = ring.pop() {
        consume(span);
    }
}

/// Copy one span into the ring. Runs inside native work: no locks and no
/// waiting.
pub(super) unsafe extern "C" fn trampoline(span: *const RawSpan, _user_data: *mut c_void) {
    // SAFETY: the native caller passes null or a span valid for this call.
    let Some(raw) = (unsafe { span.as_ref() }) else {
        return;
    };
    if raw.abi_version != NATIVE_SPAN_V1_ABI_VERSION
        || (raw.struct_size as usize) < size_of::<RawSpan>()
        || !is_native_span_id(raw.id)
    {
        return;
    }
    let detail = (!raw.detail.is_null()).then(|| {
        // SAFETY: a non-null detail is NUL-terminated and valid for this call.
        unsafe { CStr::from_ptr(raw.detail) }
            .to_string_lossy()
            .into()
    });
    let value =
        (!raw.value_name.is_null()).then(|| (static_str(raw.value_name, "value"), raw.value));
    let span = NativeSpan {
        id: raw.id,
        parent: raw.parent,
        name: static_str(raw.name, "native"),
        detail,
        start_unix_nanos: unix_nanos(raw.start_unix_ns),
        end_unix_nanos: unix_nanos(raw.end_unix_ns),
        value,
    };
    if ring().push(span).is_err() {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Deliver a raw span as the native callback would.
///
/// # Safety
///
/// `span` must be null or point to a span valid for the call, whose `name`
/// and `value_name` are null or NUL-terminated strings with static storage
/// and whose `detail` is null or NUL-terminated.
#[doc(hidden)]
pub unsafe fn deliver_native_span_for_test(span: *const RawSpan) {
    unsafe { trampoline(span, std::ptr::null_mut()) };
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use super::*;

    /// Draining takes every queued record, so tests that drain run one at a time.
    static RING_TESTS: Mutex<()> = Mutex::new(());

    fn raw(id: u64, detail: &CStr) -> RawSpan {
        RawSpan {
            abi_version: NATIVE_SPAN_V1_ABI_VERSION,
            struct_size: u32::try_from(size_of::<RawSpan>()).unwrap(),
            id: id | NATIVE_SPAN_ID_BIT,
            parent: 41,
            name: c"llama.graph_build".as_ptr(),
            detail: detail.as_ptr(),
            start_unix_ns: 1_000,
            end_unix_ns: 4_000,
            value_name: c"nodes".as_ptr(),
            value: 1_234,
        }
    }

    fn drained_with_id(id: u64) -> Vec<NativeSpan> {
        let mut found = Vec::new();
        drain_native_spans(|span| {
            if span.id == id | NATIVE_SPAN_ID_BIT {
                found.push(span);
            }
        });
        found
    }

    #[test]
    fn delivered_spans_keep_their_identity_detail_and_value() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let detail = std::ffi::CString::new("kernel_mul_mv_q4_K_f32").unwrap();
        let span = raw(0x5a, &detail);
        // SAFETY: a valid span with static names and a live detail.
        unsafe { deliver_native_span_for_test(&raw const span) };
        // The detail was copied; the native buffer may go away.
        drop(detail);

        let drained = drained_with_id(0x5a);
        assert_eq!(drained.len(), 1);
        let span = &drained[0];
        assert_eq!(span.name, "llama.graph_build");
        assert_eq!(span.parent, 41);
        assert_eq!(span.detail.as_deref(), Some("kernel_mul_mv_q4_K_f32"));
        assert_eq!(span.value, Some(("nodes", 1_234)));
        assert_eq!((span.start_unix_nanos, span.end_unix_nanos), (1_000, 4_000));
    }

    #[test]
    fn malformed_spans_are_ignored() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let mut span = raw(0x6b, c"x");
        span.abi_version += 1;
        // SAFETY: a valid span with static strings.
        unsafe { deliver_native_span_for_test(&raw const span) };
        let mut span = raw(0x6b, c"x");
        span.id &= !NATIVE_SPAN_ID_BIT;
        // SAFETY: as above.
        unsafe { deliver_native_span_for_test(&raw const span) };
        // SAFETY: null is accepted.
        unsafe { deliver_native_span_for_test(std::ptr::null()) };
        assert!(drained_with_id(0x6b).is_empty());
    }

    #[test]
    fn span_ids_are_told_apart_from_caller_tags() {
        assert!(is_native_span_id(NATIVE_SPAN_ID_BIT | 7));
        assert!(!is_native_span_id(7));
    }
}
