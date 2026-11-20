//! Device-side timing of the work native calls submit.
//!
//! A supporting backend (Metal) reports each command buffer after the GPU has
//! finished it, on the backend's completion thread, carrying the trace tag
//! the submitting thread had set. The callback only copies the record into
//! the ring: it runs on a thread that completes every command on the device,
//! and anything slower there would delay the very work being measured. A push
//! into a full ring drops the record and counts it.

use std::ffi::{CStr, c_void};
use std::mem::offset_of;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::ArrayQueue;
use skippy_ffi::{GPU_COMMAND_TIMING_V1_ABI_VERSION, SkippyGpuCommandTimingV1 as RawTiming};

use super::{static_str, unix_nanos};

/// Records the ring holds before it starts dropping: a few seconds of decode
/// steps between drains.
pub const GPU_TIMING_RING_CAPACITY: usize = 4096;

/// One unit of device work, such as a Metal command buffer. Times are Unix
/// nanoseconds; 0 when the backend did not observe the instant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuCommandTiming {
    /// The submitting native call's id; 0 when the work was submitted untagged.
    pub tag: u64,
    /// The backend registry name, such as `"MTL"`.
    pub backend: &'static str,
    pub context_id: u32,
    /// 1-based graph compute count of the backend instance; for Metal it
    /// counts like `GGML_METAL_CAPTURE_COMPUTE`.
    pub compute_index: u32,
    pub command_index: u32,
    pub command_count: u32,
    pub node_count: u32,
    /// 0 when the device completed the command.
    pub status: i32,
    pub encode_start_unix_nanos: u64,
    pub encode_end_unix_nanos: u64,
    pub execute_start_unix_nanos: u64,
    pub execute_end_unix_nanos: u64,
    /// What the command's graph nodes do; `None` from a runtime that does
    /// not classify them.
    pub op_classes: Option<GpuOpClasses>,
}

/// What a command's graph nodes do. Views and reshapes have no class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GpuOpClass {
    /// Flash attention, or the masked soft max and K/Q, K/Q/V products of
    /// unfused attention.
    Attention,
    /// Matrix products with weights.
    Matmul,
    /// SSM, RWKV, gated linear attention and delta-net state, and copies to
    /// or from its cache.
    Recurrent,
    /// Writes into the attention KV cache.
    KvWrite,
    Other,
}

impl GpuOpClass {
    pub const ALL: [Self; 5] = [
        Self::Attention,
        Self::Matmul,
        Self::Recurrent,
        Self::KvWrite,
        Self::Other,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::Matmul => "matmul",
            Self::Recurrent => "recurrent",
            Self::KvWrite => "kv_write",
            Self::Other => "other",
        }
    }

    /// The class a native label names; `None` for `"mixed"` and any label
    /// this build does not know.
    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.name() == name)
    }
}

/// One value per [`GpuOpClass`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PerOpClass<T> {
    pub attention: T,
    pub matmul: T,
    pub recurrent: T,
    pub kv_write: T,
    pub other: T,
}

impl<T: Copy> PerOpClass<T> {
    pub fn get(&self, class: GpuOpClass) -> T {
        match class {
            GpuOpClass::Attention => self.attention,
            GpuOpClass::Matmul => self.matmul,
            GpuOpClass::Recurrent => self.recurrent,
            GpuOpClass::KvWrite => self.kv_write,
            GpuOpClass::Other => self.other,
        }
    }
}

/// How a command's graph nodes divide among op classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuOpClasses {
    /// The class of every classified node, or `None` when the command mixes
    /// classes.
    pub single: Option<GpuOpClass>,
    pub layer: Option<u32>,
    /// The backend split the graph at class boundaries (op profiling), so
    /// this command's GPU time belongs to `single` alone.
    pub split_by_class: bool,
    /// The most KV cells one query addresses: the row list length when
    /// attention reads per-sequence row lists, else the K/V span.
    pub attention_kv_cells: u32,
    /// The most query rows one attention node processes.
    pub attention_queries: u32,
    /// Attention nodes that read per-sequence row lists.
    pub attention_row_list_nodes: u32,
    /// Attention nodes that are flash attention.
    pub attention_flash_nodes: u32,
    pub nodes: PerOpClass<u32>,
    /// Bytes the nodes read: a gather from or row write into a cache counts
    /// only its rows, and attention only the K/V rows its nodes address.
    pub bytes: PerOpClass<u64>,
    /// The recurrent states the command moves; `None` from a runtime that
    /// does not count them.
    pub recurrent_states: Option<RecurrentStates>,
}

/// The recurrent states one step moves through its state caches. A graph
/// gathers the states from the batch's first state cell to its last; those no
/// sequence of the batch owns are copied back unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecurrentStates {
    /// The most states one gather reads.
    pub gathered: u32,
    /// The most gathered states copied back unchanged.
    pub carried: u32,
}

/// How attention read the KV cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AttentionPath {
    /// Flash attention over each sequence's own row list.
    RowLists,
    /// Flash attention over the whole K/V span, masked.
    Flash,
    /// Masked soft max between K/Q and K/Q/V products, over the whole span.
    Unfused,
}

impl AttentionPath {
    pub fn name(self) -> &'static str {
        match self {
            Self::RowLists => "row_lists",
            Self::Flash => "flash",
            Self::Unfused => "unfused",
        }
    }
}

impl GpuOpClasses {
    /// The path the command's attention took; `None` when it has none.
    pub fn attention_path(&self) -> Option<AttentionPath> {
        if self.nodes.attention == 0 {
            None
        } else if self.attention_row_list_nodes > 0 {
            Some(AttentionPath::RowLists)
        } else if self.attention_flash_nodes > 0 {
            Some(AttentionPath::Flash)
        } else {
            Some(AttentionPath::Unfused)
        }
    }

    /// Reads the fields a runtime with op classes appends; `None` for an
    /// older, shorter record.
    fn from_raw(raw: &RawTiming) -> Option<Self> {
        let size = raw.struct_size as usize;
        if size < OP_CLASS_TIMING_SIZE {
            return None;
        }
        let recurrent_states = (size >= size_of::<RawTiming>()).then_some(RecurrentStates {
            gathered: raw.recurrent_states,
            carried: raw.recurrent_states_carried,
        });
        let label = if raw.op_class.is_null() {
            "mixed"
        } else {
            // SAFETY: the runtime passes null or a static NUL-terminated label.
            unsafe { CStr::from_ptr(raw.op_class) }
                .to_str()
                .unwrap_or("mixed")
        };
        Some(Self {
            single: GpuOpClass::from_name(label),
            layer: u32::try_from(raw.layer).ok(),
            split_by_class: raw.split_by_class != 0,
            attention_kv_cells: raw.attention_kv_cells,
            attention_queries: raw.attention_queries,
            attention_row_list_nodes: raw.attention_row_list_nodes,
            attention_flash_nodes: raw.attention_flash_nodes,
            nodes: PerOpClass {
                attention: raw.attention_nodes,
                matmul: raw.matmul_nodes,
                recurrent: raw.recurrent_nodes,
                kv_write: raw.kv_write_nodes,
                other: raw.other_nodes,
            },
            bytes: PerOpClass {
                attention: raw.attention_bytes,
                matmul: raw.matmul_bytes,
                recurrent: raw.recurrent_bytes,
                kv_write: raw.kv_write_bytes,
                other: raw.other_bytes,
            },
            recurrent_states,
        })
    }
}

/// The fields every runtime with command timing reports, up to op classes.
const BASE_TIMING_SIZE: usize = offset_of!(RawTiming, op_class);
/// The fields a runtime with op classes reports, up to recurrent state counts.
const OP_CLASS_TIMING_SIZE: usize = offset_of!(RawTiming, recurrent_states);

/// Copy the `struct_size` bytes a runtime reported into a whole record, the
/// fields an older runtime does not have left zero; `None` for a null or
/// unrecognised record.
///
/// # Safety
///
/// `timing` must be null or point to a record of at least `struct_size`
/// bytes whose header fields are valid.
unsafe fn read_record(timing: *const RawTiming) -> Option<RawTiming> {
    if timing.is_null() {
        return None;
    }
    // SAFETY: the header fields precede op classes in every record version;
    // reading them through the place does not assume a whole record.
    let (abi_version, struct_size) = unsafe {
        (
            std::ptr::addr_of!((*timing).abi_version).read(),
            std::ptr::addr_of!((*timing).struct_size).read() as usize,
        )
    };
    if abi_version != GPU_COMMAND_TIMING_V1_ABI_VERSION || struct_size < BASE_TIMING_SIZE {
        return None;
    }
    let mut record = std::mem::MaybeUninit::<RawTiming>::zeroed();
    // SAFETY: the source holds `struct_size` bytes and the copy is capped at
    // the destination's size; an all-zero RawTiming is valid (null labels).
    unsafe {
        std::ptr::copy_nonoverlapping(
            timing.cast::<u8>(),
            record.as_mut_ptr().cast::<u8>(),
            struct_size.min(size_of::<RawTiming>()),
        );
        Some(record.assume_init())
    }
}

static RING: OnceLock<ArrayQueue<GpuCommandTiming>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);

pub(super) fn ring() -> &'static ArrayQueue<GpuCommandTiming> {
    RING.get_or_init(|| ArrayQueue::new(GPU_TIMING_RING_CAPACITY))
}

/// Records lost to a full ring since the process started.
pub fn dropped_gpu_command_timings() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Hand every pending record to `consume`, oldest first.
pub fn drain_gpu_command_timings(mut consume: impl FnMut(GpuCommandTiming)) {
    let Some(ring) = RING.get() else {
        return;
    };
    while let Some(timing) = ring.pop() {
        consume(timing);
    }
}

/// Copy one record into the ring. Runs on the backend's completion thread:
/// no locks, no waiting, and no allocation once the ring exists.
pub(super) unsafe extern "C" fn trampoline(timing: *const RawTiming, _user_data: *mut c_void) {
    // SAFETY: the native caller passes null or a record valid for this call.
    let Some(raw) = (unsafe { read_record(timing) }) else {
        return;
    };
    let raw = &raw;
    let record = GpuCommandTiming {
        tag: raw.tag,
        backend: static_str(raw.backend_name, "gpu"),
        context_id: raw.context_id,
        compute_index: raw.compute_index,
        command_index: raw.command_index,
        command_count: raw.command_count,
        node_count: raw.node_count,
        status: raw.status,
        encode_start_unix_nanos: unix_nanos(raw.encode_start_unix_ns),
        encode_end_unix_nanos: unix_nanos(raw.encode_end_unix_ns),
        execute_start_unix_nanos: unix_nanos(raw.execute_start_unix_ns),
        execute_end_unix_nanos: unix_nanos(raw.execute_end_unix_ns),
        op_classes: GpuOpClasses::from_raw(raw),
    };
    if ring().push(record).is_err() {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Deliver a raw record as the native callback would.
///
/// # Safety
///
/// `timing` must be null or point to a record valid for the call, whose
/// `backend_name` is null or a NUL-terminated string with static storage.
#[doc(hidden)]
pub unsafe fn deliver_gpu_command_timing_for_test(timing: *const RawTiming) {
    unsafe { trampoline(timing, std::ptr::null_mut()) };
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use super::*;

    /// Draining takes every queued record, so tests that drain run one at a time.
    static RING_TESTS: Mutex<()> = Mutex::new(());

    fn raw(tag: u64) -> RawTiming {
        RawTiming {
            abi_version: GPU_COMMAND_TIMING_V1_ABI_VERSION,
            struct_size: u32::try_from(size_of::<RawTiming>()).unwrap(),
            tag,
            backend_name: c"MTL".as_ptr(),
            context_id: 2,
            compute_index: 9,
            command_index: 1,
            command_count: 3,
            node_count: 120,
            status: 0,
            encode_start_unix_ns: 1_000,
            encode_end_unix_ns: 2_000,
            execute_start_unix_ns: 2_500,
            execute_end_unix_ns: -1,
            op_class: c"attention".as_ptr(),
            layer: 7,
            split_by_class: 1,
            attention_kv_cells: 4096,
            attention_queries: 4,
            attention_row_list_nodes: 1,
            attention_flash_nodes: 1,
            attention_nodes: 1,
            matmul_nodes: 0,
            recurrent_nodes: 0,
            kv_write_nodes: 0,
            other_nodes: 2,
            attention_bytes: 1 << 20,
            matmul_bytes: 0,
            recurrent_bytes: 0,
            kv_write_bytes: 0,
            other_bytes: 64,
            recurrent_states: 6,
            recurrent_states_carried: 2,
        }
    }

    fn drained_with_tag(tag: u64) -> Vec<GpuCommandTiming> {
        let mut found = Vec::new();
        drain_gpu_command_timings(|timing| {
            if timing.tag == tag {
                found.push(timing);
            }
        });
        found
    }

    #[test]
    fn delivered_records_drain_with_their_tag_and_backend() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let record = raw(0xa11ce);
        // SAFETY: a valid record with a static backend name.
        unsafe { deliver_gpu_command_timing_for_test(&raw const record) };

        let drained = drained_with_tag(0xa11ce);
        assert_eq!(drained.len(), 1);
        let timing = drained[0];
        assert_eq!(timing.backend, "MTL");
        assert_eq!((timing.context_id, timing.compute_index), (2, 9));
        assert_eq!((timing.command_index, timing.command_count), (1, 3));
        assert_eq!(timing.node_count, 120);
        assert_eq!(timing.encode_start_unix_nanos, 1_000);
        assert_eq!(timing.execute_start_unix_nanos, 2_500);
        assert_eq!(
            timing.execute_end_unix_nanos, 0,
            "negative means unobserved"
        );
    }

    #[test]
    fn records_from_another_abi_version_are_ignored() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let mut record = raw(0xb0b);
        record.abi_version += 1;
        // SAFETY: a valid record with a static backend name.
        unsafe { deliver_gpu_command_timing_for_test(&raw const record) };
        // SAFETY: null is accepted.
        unsafe { deliver_gpu_command_timing_for_test(std::ptr::null()) };
        assert!(drained_with_tag(0xb0b).is_empty());
    }

    #[test]
    fn op_classes_are_parsed_at_the_boundary() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let record = raw(0xc1a55);
        // SAFETY: a valid record with static labels.
        unsafe { deliver_gpu_command_timing_for_test(&raw const record) };
        let classes = drained_with_tag(0xc1a55)[0]
            .op_classes
            .expect("a runtime with op classes");
        assert_eq!(classes.single, Some(GpuOpClass::Attention));
        assert_eq!(classes.layer, Some(7));
        assert!(classes.split_by_class);
        assert_eq!(
            (classes.attention_kv_cells, classes.attention_queries),
            (4096, 4)
        );
        assert_eq!(classes.nodes.get(GpuOpClass::Other), 2);
        assert_eq!(classes.bytes.get(GpuOpClass::Attention), 1 << 20);
        assert_eq!(
            classes.recurrent_states,
            Some(RecurrentStates {
                gathered: 6,
                carried: 2
            })
        );
        assert_eq!(classes.attention_path(), Some(AttentionPath::RowLists));
        let flash = GpuOpClasses {
            attention_row_list_nodes: 0,
            ..classes
        };
        assert_eq!(flash.attention_path(), Some(AttentionPath::Flash));
        let unfused = GpuOpClasses {
            attention_flash_nodes: 0,
            ..flash
        };
        assert_eq!(unfused.attention_path(), Some(AttentionPath::Unfused));

        let mixed = RawTiming {
            op_class: c"mixed".as_ptr(),
            layer: -1,
            ..raw(0xc1a56)
        };
        // SAFETY: as above.
        unsafe { deliver_gpu_command_timing_for_test(&raw const mixed) };
        let classes = drained_with_tag(0xc1a56)[0].op_classes.expect("classes");
        assert_eq!((classes.single, classes.layer), (None, None));
    }

    #[test]
    fn a_record_from_before_op_classes_still_times_the_command() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        // Only the older prefix is valid: what an older runtime hands over.
        let record = RawTiming {
            struct_size: u32::try_from(BASE_TIMING_SIZE).unwrap(),
            ..raw(0x01d)
        };
        // SAFETY: the record is whole; the size marks the rest as absent.
        unsafe { deliver_gpu_command_timing_for_test(&raw const record) };
        let drained = drained_with_tag(0x01d);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].node_count, 120);
        assert_eq!(drained[0].op_classes, None);
    }

    #[test]
    fn a_record_from_before_recurrent_counts_keeps_its_op_classes() {
        let _serial = RING_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        let record = RawTiming {
            struct_size: u32::try_from(OP_CLASS_TIMING_SIZE).unwrap(),
            ..raw(0x01e)
        };
        // SAFETY: the record is whole; the size marks the counts as absent.
        unsafe { deliver_gpu_command_timing_for_test(&raw const record) };
        let classes = drained_with_tag(0x01e)[0].op_classes.expect("op classes");
        assert_eq!(classes.single, Some(GpuOpClass::Attention));
        assert_eq!(classes.recurrent_states, None);
    }
}
