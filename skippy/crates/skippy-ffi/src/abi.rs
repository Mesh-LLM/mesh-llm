//! Ergonomic Rust aliases over the generated Skippy native ABI bindings.
//!
//! Every struct, enum, and constant here is a thin re-export or value alias
//! of [`crate::bindings`]; the generated declarations are the single source
//! of truth for the native layout. Only initialization ergonomics (`Default`
//! impls that stamp `struct_size`/`abi_version`), the ABI compatibility
//! predicate, and callback aliases remain hand-written.

use std::ffi::{c_char, c_void};

use crate::{NativeMtpDraft, SamplingConfig, bindings};

// Feature bits. Enum members come from `enum skippy_feature`; bits 31+ are
// `uint64_t` constants in the header. Keep the historical short names: the
// capability probe and family gating consume them by name.
pub const FEATURE_STAGE_PLAN: u64 = bindings::skippy::skippy_feature::StagePlan as u64;
pub const FEATURE_BACKEND_DEVICES: u64 = bindings::skippy::skippy_feature::BackendDevices as u64;
pub const FEATURE_RUNTIME_EVENTS: u64 = bindings::skippy::skippy_feature::RuntimeEvents as u64;
pub const FEATURE_NATIVE_MTP_N1: u64 = bindings::skippy::skippy_feature::NativeMtpN1 as u64;
pub const FEATURE_NGRAM_CACHE_DRAFT: u64 = bindings::skippy::skippy_feature::NgramCacheDraft as u64;
pub const FEATURE_MTP_MULTIMODAL: u64 = bindings::skippy::skippy_feature::MtpMultimodal as u64;
pub const FEATURE_ITERATION_BATCH: u64 = bindings::skippy::skippy_feature::IterationBatch as u64;
pub const FEATURE_ACTIVATION_BOUNDARY: u64 =
    bindings::skippy::skippy_feature::ActivationBoundary as u64;
pub const FEATURE_MODEL_SOURCE: u64 = bindings::skippy::skippy_feature::ModelSource as u64;
pub const FEATURE_RUNTIME_EVENT_REPORTER: u64 =
    bindings::skippy::skippy_ffi_bridge_feature_runtime_event_reporter;
pub const FEATURE_MODEL_LOAD_EVENTS_V2: u64 =
    bindings::skippy::skippy_ffi_bridge_feature_model_load_events_v2;
pub const FEATURE_KV_EVENTS: u64 = bindings::skippy::skippy_ffi_bridge_feature_kv_events;
pub const FEATURE_DEVICE_EVENTS: u64 = bindings::skippy::skippy_ffi_bridge_feature_device_events;
pub const FEATURE_DIAGNOSTIC_EVENTS: u64 =
    bindings::skippy::skippy_ffi_bridge_feature_diagnostic_events;
pub const FEATURE_UNLOAD_EVENTS: u64 = bindings::skippy::skippy_ffi_bridge_feature_unload_events;
pub const FEATURE_NON_CHAT_WORKLOADS: u64 =
    bindings::skippy::skippy_ffi_bridge_feature_non_chat_workloads;
pub const FEATURE_SYSTEM_ONE: u64 = bindings::skippy::skippy_ffi_bridge_feature_system_one;
pub const FEATURE_CACHEGEN_KV_PAGE: u64 =
    bindings::skippy::skippy_ffi_bridge_feature_cachegen_kv_page;
pub const FEATURE_LAYA_DECISIONS: u64 = bindings::skippy::skippy_ffi_bridge_feature_laya_decisions;
pub const FEATURE_SPEECH_RECOGNITION: u64 =
    bindings::skippy::skippy_ffi_bridge_feature_speech_recognition;

pub const MODEL_TENSOR_SOURCE_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_MODEL_TENSOR_SOURCE_V1_ABI_VERSION;
pub const WORKLOAD_INFO_V1_ABI_VERSION: u32 = bindings::skippy::SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION;
pub const LAYA_INFO_V1_ABI_VERSION: u32 = bindings::skippy::SKIPPY_LAYA_INFO_V1_ABI_VERSION;
pub const LAYA_MEMORY_V1_ABI_VERSION: u32 = bindings::skippy::SKIPPY_LAYA_MEMORY_V1_ABI_VERSION;
pub const ASR_INFO_V1_ABI_VERSION: u32 = bindings::skippy::SKIPPY_ASR_INFO_V1_ABI_VERSION;
pub const ASR_TRANSCRIBE_PARAMS_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_ASR_TRANSCRIBE_PARAMS_V1_ABI_VERSION;
pub const ASR_SEGMENT_V1_ABI_VERSION: u32 = bindings::skippy::SKIPPY_ASR_SEGMENT_V1_ABI_VERSION;
pub const ASR_SAMPLE_RATE: u32 = bindings::skippy::SKIPPY_ASR_SAMPLE_RATE;
pub const ASR_MAX_SAMPLES: usize = bindings::skippy::SKIPPY_ASR_MAX_SAMPLES as usize;
pub const LAYA_QTYPE_COUNT: usize = bindings::skippy::SKIPPY_LAYA_QTYPE_COUNT as usize;

pub type WorkloadKind = bindings::skippy::skippy_workload_kind;
pub type WorkloadPooling = bindings::skippy::skippy_workload_pooling;
pub type WorkloadInfoV1 = bindings::skippy::skippy_workload_info_v1;

impl Default for WorkloadInfoV1 {
    /// Initialize the workload descriptor with its ABI size and version.
    fn default() -> Self {
        Self {
            abi_version: WORKLOAD_INFO_V1_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            kind: WorkloadKind::CausalGeneration,
            pooling: WorkloadPooling::None,
            output_dimensions: 0,
            classifier_outputs: 0,
            has_encoder: false,
            has_decoder: false,
            full_model_only: true,
            reserved0: 0,
        }
    }
}

pub type ModelReadTensorF32Callback = bindings::skippy::skippy_model_read_tensor_f32_callback;
pub type ModelImatrixEntryV1 = bindings::skippy::skippy_model_imatrix_entry_v1;
pub type ModelTensorSourceV1 = bindings::skippy::skippy_model_tensor_source_v1;
pub type IterationRequest = bindings::skippy::skippy_iteration_request;
pub type TokenScoreRequest = bindings::skippy::skippy_token_score_request;
pub type SystemOneSlot = bindings::skippy::skippy_system_one_slot;
pub type LayaInfoV1 = bindings::skippy::skippy_laya_info_v1;

impl Default for LayaInfoV1 {
    /// Initialize the Laya descriptor with its ABI size and version.
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            abi_version: LAYA_INFO_V1_ABI_VERSION,
            max_len: 0,
            head_max_len: 0,
            max_markers: 0,
            n_act: 0,
            cls_token_id: 0,
            sep_token_id: 0,
            mask_token_id: 0,
            n_embd: 0,
            n_layer: 0,
            parameter_count: 0,
            temperature: [0.0; LAYA_QTYPE_COUNT],
        }
    }
}

/// Measured memory of an opened Laya model.
pub type LayaMemoryV1 = bindings::skippy::skippy_laya_memory_v1;

impl Default for LayaMemoryV1 {
    /// Initialize the memory descriptor with its ABI size and version.
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            abi_version: LAYA_MEMORY_V1_ABI_VERSION,
            weights_bytes: 0,
            compute_bytes: 0,
            host_scratch_bytes: 0,
            on_accelerator: false,
        }
    }
}

/// One question sequence inside a packed Laya read.
pub type LayaSequence = bindings::skippy::skippy_laya_sequence;
/// Shape and memory of an opened speech-recognition model.
pub type AsrInfoV1 = bindings::skippy::skippy_asr_info_v1;

impl Default for AsrInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            abi_version: ASR_INFO_V1_ABI_VERSION,
            multilingual: false,
            n_vocab: 0,
            n_audio_layer: 0,
            n_text_layer: 0,
            n_mels: 0,
            weights_bytes: 0,
            on_accelerator: false,
        }
    }
}
/// Options for one transcription. Zero values select the defaults.
pub type AsrTranscribeParamsV1 = bindings::skippy::skippy_asr_transcribe_params_v1;

impl Default for AsrTranscribeParamsV1 {
    /// Initialize the transcription options with their ABI size and version.
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            abi_version: ASR_TRANSCRIBE_PARAMS_V1_ABI_VERSION,
            language: std::ptr::null(),
            translate: false,
            prompt: std::ptr::null(),
            temperature: 0.0,
            n_threads: 0,
        }
    }
}

/// One timed segment; `text` is owned by the native result.
pub type AsrSegmentV1 = bindings::skippy::skippy_asr_segment_v1;

impl Default for AsrSegmentV1 {
    /// Initialize the segment descriptor with its ABI size and version.
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            abi_version: ASR_SEGMENT_V1_ABI_VERSION,
            start_ms: 0,
            end_ms: 0,
            text: std::ptr::null(),
            no_speech_prob: 0.0,
        }
    }
}

/// Polled between decoder steps; returning true cancels the transcription.
pub type AsrAbortCallback = bindings::skippy::skippy_asr_abort_callback;
pub type AbiVersion = bindings::skippy::skippy_abi_version;

/// Whether a native runtime reporting `version` can back this binary's ABI
/// bindings. Required symbol signatures and by-value struct layouts may change
/// between patches, so the loader requires an exact ABI match.
pub const fn runtime_abi_supported(version: AbiVersion) -> bool {
    version.major == crate::ABI_VERSION_MAJOR
        && version.minor == crate::ABI_VERSION_MINOR
        && version.patch == crate::ABI_VERSION_PATCH
}

/// Log callback shared by llama.cpp, ggml, and mtmd logging entrypoints.
pub type LlamaLogCallback = bindings::llama::ggml_log_callback;
pub type MtmdProgressCallback = bindings::mtmd::mtmd_progress_callback;
pub type SkippyRuntimeEventCallback = bindings::skippy::skippy_runtime_event_callback;

// Runtime-event family discriminants. The native headers model these as
// anonymous `uint32_t` enumerations; the generated constants are re-exported
// under their C names so event construction reads exactly like the header.
pub use bindings::skippy::{
    SKIPPY_RUNTIME_EVENT_CATEGORY_BACKEND as RUNTIME_EVENT_CATEGORY_BACKEND,
    SKIPPY_RUNTIME_EVENT_CATEGORY_DEVICE as RUNTIME_EVENT_CATEGORY_DEVICE,
    SKIPPY_RUNTIME_EVENT_CATEGORY_DIAGNOSTIC as RUNTIME_EVENT_CATEGORY_DIAGNOSTIC,
    SKIPPY_RUNTIME_EVENT_CATEGORY_KV as RUNTIME_EVENT_CATEGORY_KV,
    SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN as RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
    SKIPPY_RUNTIME_EVENT_CATEGORY_SESSION as RUNTIME_EVENT_CATEGORY_SESSION,
    SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD as RUNTIME_EVENT_CATEGORY_UNLOAD,
    SKIPPY_RUNTIME_EVENT_CATEGORY_WARNING as RUNTIME_EVENT_CATEGORY_WARNING,
    SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD as RUNTIME_EVENT_EMITTER_OPEN_THREAD,
    SKIPPY_RUNTIME_EVENT_EMITTER_UNKNOWN as RUNTIME_EVENT_EMITTER_UNKNOWN,
    SKIPPY_RUNTIME_EVENT_EMITTER_WORKER_THREAD as RUNTIME_EVENT_EMITTER_WORKER_THREAD,
    SKIPPY_RUNTIME_EVENT_FAILURE_BACKEND_ERROR as RUNTIME_EVENT_FAILURE_BACKEND_ERROR,
    SKIPPY_RUNTIME_EVENT_FAILURE_CANCELLED as RUNTIME_EVENT_FAILURE_CANCELLED,
    SKIPPY_RUNTIME_EVENT_FAILURE_INTERNAL_ERROR as RUNTIME_EVENT_FAILURE_INTERNAL_ERROR,
    SKIPPY_RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT as RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT,
    SKIPPY_RUNTIME_EVENT_FAILURE_IO_ERROR as RUNTIME_EVENT_FAILURE_IO_ERROR,
    SKIPPY_RUNTIME_EVENT_FAILURE_MODEL_ERROR as RUNTIME_EVENT_FAILURE_MODEL_ERROR,
    SKIPPY_RUNTIME_EVENT_FAILURE_NONE as RUNTIME_EVENT_FAILURE_NONE,
    SKIPPY_RUNTIME_EVENT_FAILURE_RUNTIME_ERROR as RUNTIME_EVENT_FAILURE_RUNTIME_ERROR,
    SKIPPY_RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED as RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_BACKEND_INITIALIZED as RUNTIME_EVENT_KIND_DEVICE_BACKEND_INITIALIZED,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_DEGRADED as RUNTIME_EVENT_KIND_DEVICE_DEGRADED,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_FALLBACK_ACTIVATED as RUNTIME_EVENT_KIND_DEVICE_FALLBACK_ACTIVATED,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_LOST as RUNTIME_EVENT_KIND_DEVICE_LOST,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_OUT_OF_MEMORY as RUNTIME_EVENT_KIND_DEVICE_OUT_OF_MEMORY,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY as RUNTIME_EVENT_KIND_DEVICE_READY,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_RECOVERED as RUNTIME_EVENT_KIND_DEVICE_RECOVERED,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_RESOURCE_ALLOCATED as RUNTIME_EVENT_KIND_DEVICE_RESOURCE_ALLOCATED,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE as RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE as RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_INVARIANT_VIOLATION as RUNTIME_EVENT_KIND_DIAGNOSTIC_INVARIANT_VIOLATION,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE as RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_CLEARED as RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_CLEARED,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED as RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED,
    SKIPPY_RUNTIME_EVENT_KIND_KV_CONTEXT_APPROACHING_CAPACITY as RUNTIME_EVENT_KIND_KV_CONTEXT_APPROACHING_CAPACITY,
    SKIPPY_RUNTIME_EVENT_KIND_KV_CONTEXT_CAPACITY_EXHAUSTED as RUNTIME_EVENT_KIND_KV_CONTEXT_CAPACITY_EXHAUSTED,
    SKIPPY_RUNTIME_EVENT_KIND_KV_INITIALIZED as RUNTIME_EVENT_KIND_KV_INITIALIZED,
    SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED as RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED,
    SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED as RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY as RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_MEMORY_ALLOCATED as RUNTIME_EVENT_KIND_MODEL_LOAD_MEMORY_ALLOCATED,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED as RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_TENSORS_OFFLOADED as RUNTIME_EVENT_KIND_MODEL_LOAD_TENSORS_OFFLOADED,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY as RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED as RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED as RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS as RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED as RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED as RUNTIME_EVENT_KIND_UNLOAD_COMPLETED,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_FAILED as RUNTIME_EVENT_KIND_UNLOAD_FAILED,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_FORCED as RUNTIME_EVENT_KIND_UNLOAD_FORCED,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_SESSION_DRAINING as RUNTIME_EVENT_KIND_UNLOAD_SESSION_DRAINING,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_STARTED as RUNTIME_EVENT_KIND_UNLOAD_STARTED,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_BYTES as RUNTIME_EVENT_PROGRESS_UNIT_BYTES,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_ITEMS as RUNTIME_EVENT_PROGRESS_UNIT_ITEMS,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE as RUNTIME_EVENT_PROGRESS_UNIT_NONE,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_STEPS as RUNTIME_EVENT_PROGRESS_UNIT_STEPS,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_TENSORS as RUNTIME_EVENT_PROGRESS_UNIT_TENSORS,
};

/// Numeric aliases for the runtime-event family discriminants above.
pub type SkippyRuntimeEventCategory = bindings::skippy::skippy_runtime_event_category;
pub type SkippyRuntimeEventKind = bindings::skippy::skippy_runtime_event_kind;
pub type SkippyRuntimeEventEmitterKind = bindings::skippy::skippy_runtime_event_emitter_kind;
pub type SkippyRuntimeEventProgressUnit = bindings::skippy::skippy_runtime_event_progress_unit;
pub type SkippyRuntimeEventFailureCode = bindings::skippy::skippy_runtime_event_failure_code;

/// Versioned event payload delivered to a runtime event callback. The
/// append-only `numeric_summary_*` extension is advertised by `struct_size`;
/// callers read it only when `struct_size` covers it.
pub type SkippyRuntimeEventV1 = bindings::skippy::skippy_runtime_event_v1;
pub type SkippyRuntimeEventReporterV1 = bindings::skippy::skippy_runtime_event_reporter_v1;

pub type Status = bindings::skippy::skippy_status;
pub type LoadMode = bindings::skippy::skippy_load_mode;

#[allow(clippy::derivable_impls)]
impl Default for LoadMode {
    fn default() -> Self {
        Self::RuntimeSlice
    }
}

pub type MtpSource = bindings::skippy::skippy_mtp_source;

#[allow(clippy::derivable_impls)]
impl Default for MtpSource {
    fn default() -> Self {
        Self::Disabled
    }
}

pub type TensorRole = bindings::skippy::skippy_tensor_role;
pub type BackendDeviceType = bindings::skippy::skippy_backend_device_type;

pub const BACKEND_DEVICE_CAP_ASYNC: u64 = bindings::skippy::skippy_backend_device_cap::Async as u64;
pub const BACKEND_DEVICE_CAP_HOST_BUFFER: u64 =
    bindings::skippy::skippy_backend_device_cap::HostBuffer as u64;
pub const BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR: u64 =
    bindings::skippy::skippy_backend_device_cap::BufferFromHostPtr as u64;
pub const BACKEND_DEVICE_CAP_EVENTS: u64 =
    bindings::skippy::skippy_backend_device_cap::Events as u64;

pub type Error = bindings::skippy::skippy_error;
pub type RuntimeConfig = bindings::skippy::skippy_runtime_config;
pub type BackendDevice = bindings::skippy::skippy_backend_device;

pub type Model = bindings::skippy::skippy_model;
pub type LayaModel = bindings::skippy::skippy_laya_model;
pub type AsrModel = bindings::skippy::skippy_asr_model;
pub type AsrResult = bindings::skippy::skippy_asr_result;
pub type NgramCache = bindings::skippy::skippy_ngram_cache;
pub type Session = bindings::skippy::skippy_session;
pub type ModelInfo = bindings::skippy::skippy_model_info;

pub type SkippyModelAttachMtpDraftModelFn = unsafe extern "C" fn(
    target_model: *mut Model,
    path: *const c_char,
    config: *const RuntimeConfig,
    out_error: *mut *mut Error,
) -> Status;

pub type SkippyDecodeStepSampledMtpFn = unsafe extern "C" fn(
    session: *mut Session,
    token_id: i32,
    sampling: *const SamplingConfig,
    out_predicted_token: *mut i32,
    max_draft_tokens: usize,
    out_mtp_draft: *mut NativeMtpDraft,
    out_error: *mut *mut Error,
) -> Status;

/// Untyped native pointers borrowed across the ABI boundary.
/// Sentinel for `RuntimeConfig` tri-state fields: keep the native derived default.
pub const TRISTATE_AUTO: i32 = bindings::skippy::SKIPPY_TRISTATE_AUTO;
/// Sentinel for `RuntimeConfig` tri-state fields: force false.
pub const TRISTATE_FALSE: i32 = bindings::skippy::SKIPPY_TRISTATE_FALSE as i32;
/// Sentinel for `RuntimeConfig` tri-state fields: force true.
pub const TRISTATE_TRUE: i32 = bindings::skippy::SKIPPY_TRISTATE_TRUE as i32;

/// Untyped native pointers borrowed across the ABI boundary.
pub type Opaque = c_void;

/// `llama_perf_context`, resolved dynamically when the runtime exports it and
/// called directly when the runtime is linked statically.
pub type LlamaPerfContextFn = unsafe extern "C" fn(ctx: *mut Opaque) -> LlamaPerfContextData;

/// Graph-reuse and realloc counters returned by value from
/// `llama_perf_context`.
///
/// `n_reused` is the number of times a compute graph was reused instead of
/// rebuilt. It is the only direct read on whether graph reuse is firing under
/// real load, and nothing on the host could see it before: the counter stopped
/// at the C++ boundary, so the reuse hit rate had to be inferred from
/// throughput deltas.
///
/// This type is generated from the pinned upstream header because the layout
/// must track upstream exactly: `llama_perf_context` returns the struct by
/// value, so a mirror that is *smaller* than the upstream struct does not fail
/// to link or return obviously wrong numbers — native code writes past the
/// destination. That is exactly what happened with the v0.7.0 pin: upstream
/// added `n_reallocs_same`/`n_reallocs_diff` and the resulting eight-byte
/// stack overwrite surfaced as a SIGSEGV in an unrelated status reader, not as
/// a wrong counter value. Bindgen generation removes the handwritten-mirror
/// failure mode entirely.
pub type LlamaPerfContextData = bindings::llama::llama_perf_context_data;

impl Default for LlamaPerfContextData {
    fn default() -> Self {
        Self {
            t_start_ms: 0.0,
            t_load_ms: 0.0,
            t_p_eval_ms: 0.0,
            t_eval_ms: 0.0,
            n_p_eval: 0,
            n_eval: 0,
            n_reused: 0,
            n_reallocs_same: 0,
            n_reallocs_diff: 0,
        }
    }
}
