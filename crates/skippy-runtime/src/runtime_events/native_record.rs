//! A `Copy` snapshot of one native runtime event.
//!
//! The reporter callback runs on whichever native worker thread raised the
//! event -- a decode thread, a loader thread, a device monitor. Whatever it
//! does there, that thread is not doing its real work. So it does exactly
//! two things: copy the event onto the stack, and push it into a ring.
//!
//! That rules out [`super::RuntimeEvent`] as the callback-thread type. It
//! owns its detail bytes in a `Vec`, so building one allocates, and it is
//! not `Copy`, so moving it around is not free either. [`NativeEventRecord`]
//! is plain data with the detail inlined: constructing it is a fixed-size
//! memcpy with no branch that can allocate.
//!
//! ## Detail truncation
//!
//! Native detail is bounded at a megabyte by the ABI, which is far too
//! large to inline. A record carries the first [`INLINE_DETAIL_BYTES`] and
//! records that it truncated. That is a deliberate trade: detail is
//! diagnostic prose, the consumer already bounds what it puts on the wire,
//! and a truncation flag is information the old path did not have at all --
//! it copied the whole thing and then let the consumer silently bound it.
//!
//! ## Scalars stay raw
//!
//! Every enum-like field is kept as the raw integer the ABI delivered and
//! converted by [`NativeEventRecord::to_event`] on the consumer side. A
//! conversion is a branch, and a branch on the callback thread is work that
//! does not have to happen there.

use std::mem;
use std::ptr;

use skippy_ffi::{SkippyRuntimeEventV1 as RawRuntimeEvent, Status};

use super::wire_types::{BaseRawRuntimeEvent, MAX_DETAIL_BYTES, RuntimeEvent};

/// Detail bytes carried inline. Longer detail is truncated to this and
/// flagged.
pub const INLINE_DETAIL_BYTES: usize = 256;

/// Bit positions in [`NativeEventRecord::numeric_summary_present`].
const SUMMARY_PRESENT_ALL: u8 = 0b1111;

/// One native event, copied whole onto the callback thread's stack.
///
/// `repr(C)` so the layout is stated rather than inferred; `Copy` so moving
/// it into and out of the ring is a memcpy with no destructor to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct NativeEventRecord {
    pub abi_version: u32,
    pub struct_size: u32,
    pub category: u32,
    pub kind: u32,
    pub emitter: u32,
    pub reserved0: u32,
    pub sequence: u64,
    pub timestamp_mono_ns: u64,
    pub model_id: u64,
    pub stage_id: u64,
    pub session_id: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub progress_unit: u32,
    pub failure_code: u32,
    /// Kept as the ABI's own `Status` rather than a raw `i32`: it is
    /// already `Copy` and `repr(i32)`, and the existing wire type stores it
    /// the same way, so there is no new conversion and no new hazard here.
    pub status: Status,
    pub reserved1: u32,
    /// The four append-only extension values, meaningful only for the bits
    /// set in `numeric_summary_present`. A runtime whose `struct_size` did
    /// not cover them leaves the bits clear, so they are never read as
    /// zero.
    pub numeric_summary: [u64; 4],
    pub numeric_summary_present: u8,
    /// Whether the native detail was longer than [`INLINE_DETAIL_BYTES`].
    pub detail_truncated: bool,
    /// Bytes actually copied into `detail`, at most [`INLINE_DETAIL_BYTES`].
    pub detail_len: u16,
    pub detail: [u8; INLINE_DETAIL_BYTES],
}

impl NativeEventRecord {
    /// Copy `event` onto the stack, or `None` when the pointer is null or
    /// its `struct_size` does not cover the base layout.
    ///
    /// Allocation-free and branch-light by construction: the only variable
    /// work is one bounded `copy_from_slice`.
    ///
    /// # Safety
    ///
    /// `event` must be null or point to a `SkippyRuntimeEventV1` whose
    /// `struct_size` field is readable, per the reporter ABI contract. When
    /// `struct_size` covers a larger layout, those bytes must be
    /// initialized and in bounds for the duration of this call, and
    /// `detail_ptr`'s `detail_len` bytes must be valid and immutable for
    /// the same duration.
    pub unsafe fn from_raw_ptr(event: *const RawRuntimeEvent) -> Option<Self> {
        if event.is_null() {
            return None;
        }
        // Prefix-validate before any other field read, exactly as
        // `RuntimeEvent::from_raw_ptr` does: read only `struct_size` via
        // `read_unaligned`, and refuse to form a struct reference until it
        // proves the allocation covers the base layout.
        let struct_size = unsafe { ptr::read_unaligned(ptr::addr_of!((*event).struct_size)) };
        if (struct_size as usize) < mem::size_of::<BaseRawRuntimeEvent>() {
            return None;
        }
        let covers_extension = (struct_size as usize) >= mem::size_of::<RawRuntimeEvent>();

        // SAFETY: `struct_size` was just validated to cover at least the
        // base layout, and `BaseRawRuntimeEvent` is `repr(C)` with the same
        // field prefix as `RawRuntimeEvent` (the C common-initial-sequence
        // pattern), so reading through it never touches bytes past what
        // `struct_size` proved is allocated.
        let base = unsafe { &*event.cast::<BaseRawRuntimeEvent>() };

        let declared = usize::try_from(base.detail_len).ok()?;
        if declared > MAX_DETAIL_BYTES {
            return None;
        }
        let mut detail = [0u8; INLINE_DETAIL_BYTES];
        let copied = if declared == 0 || base.detail_ptr.is_null() {
            0
        } else {
            let copied = declared.min(INLINE_DETAIL_BYTES);
            // SAFETY: `copied` is bounded by the declared length, which was
            // range-checked above, and `detail_ptr` is non-null; the
            // reporter contract guarantees this byte range is valid and
            // immutable for the callback's duration.
            let source =
                unsafe { std::slice::from_raw_parts(base.detail_ptr.cast::<u8>(), copied) };
            detail[..copied].copy_from_slice(source);
            copied
        };

        let (numeric_summary, numeric_summary_present) = if covers_extension {
            // SAFETY: `struct_size` covers the full extended layout, so the
            // ABI contract guarantees this range is initialized and in
            // bounds.
            let full = unsafe { &*event };
            (
                [
                    full.numeric_summary_0,
                    full.numeric_summary_1,
                    full.numeric_summary_2,
                    full.numeric_summary_3,
                ],
                SUMMARY_PRESENT_ALL,
            )
        } else {
            ([0; 4], 0)
        };

        Some(Self {
            abi_version: base.abi_version,
            struct_size,
            category: base.category.0,
            kind: base.kind.0,
            emitter: base.emitter.0,
            reserved0: base.reserved0,
            sequence: base.sequence,
            timestamp_mono_ns: base.timestamp_mono_ns,
            model_id: base.model_id,
            stage_id: base.stage_id,
            session_id: base.session_id,
            progress_current: base.progress_current,
            progress_total: base.progress_total,
            progress_unit: base.progress_unit.0,
            failure_code: base.failure_code.0,
            status: base.status,
            reserved1: base.reserved1,
            numeric_summary,
            numeric_summary_present,
            // Truncation is reported against what the native side
            // declared, not against what was copied.
            detail_truncated: declared > copied,
            detail_len: u16::try_from(copied).unwrap_or(u16::MAX),
            detail,
        })
    }

    /// The detail bytes this record carries, already bounded.
    #[must_use]
    pub fn detail(&self) -> &[u8] {
        &self.detail[..usize::from(self.detail_len)]
    }

    fn summary(&self, index: usize) -> Option<u64> {
        (self.numeric_summary_present & (1 << index) != 0).then(|| self.numeric_summary[index])
    }

    /// Expand into the owned [`RuntimeEvent`] consumers work with.
    ///
    /// This is where the allocation the callback thread refused to do
    /// happens -- on the consumer's thread, where waiting is what it is
    /// for.
    #[must_use]
    pub fn to_event(&self) -> RuntimeEvent {
        RuntimeEvent {
            abi_version: self.abi_version,
            struct_size: self.struct_size,
            category: skippy_ffi::SkippyRuntimeEventCategory(self.category).into(),
            kind: skippy_ffi::SkippyRuntimeEventKind(self.kind).into(),
            emitter: skippy_ffi::SkippyRuntimeEventEmitterKind(self.emitter).into(),
            reserved0: self.reserved0,
            sequence: self.sequence,
            timestamp_mono_ns: self.timestamp_mono_ns,
            model_id: self.model_id,
            stage_id: self.stage_id,
            session_id: self.session_id,
            progress_current: self.progress_current,
            progress_total: self.progress_total,
            progress_unit: skippy_ffi::SkippyRuntimeEventProgressUnit(self.progress_unit).into(),
            failure_code: skippy_ffi::SkippyRuntimeEventFailureCode(self.failure_code).into(),
            status: self.status,
            reserved1: self.reserved1,
            detail_bytes: self.detail().to_vec(),
            numeric_summary_0: self.summary(0),
            numeric_summary_1: self.summary(1),
            numeric_summary_2: self.summary(2),
            numeric_summary_3: self.summary(3),
        }
    }
}
