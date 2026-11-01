//! Behavioral tests for the FFI facade.
//!
//! Layout correctness is owned by the generated bindings: `bindgen` derives
//! every struct field layout from the pinned headers at generation time and
//! CI fails when the committed bindings drift from those headers
//! (`SKIPPY_FFI_BINDGEN=check`). The tests here therefore exercise behavior
//! and cross-boundary agreement, not hard-coded field offsets.

use crate::{
    ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH, AbiVersion, WorkloadInfoV1,
    runtime_abi_supported,
};

#[cfg(not(feature = "dynamic-runtime"))]
use crate::mtmd_context_params_default;

const fn version(major: u32, minor: u32, patch: u32) -> AbiVersion {
    AbiVersion {
        major,
        minor,
        patch,
    }
}

#[test]
/// Runtime-event families must never masquerade as full-model workload support.
fn workload_feature_is_distinct_from_runtime_event_features() {
    let events = crate::FEATURE_RUNTIME_EVENT_REPORTER
        | crate::FEATURE_MODEL_LOAD_EVENTS_V2
        | crate::FEATURE_KV_EVENTS
        | crate::FEATURE_DEVICE_EVENTS
        | crate::FEATURE_DIAGNOSTIC_EVENTS
        | crate::FEATURE_UNLOAD_EVENTS;
    assert_eq!(crate::FEATURE_NON_CHAT_WORKLOADS, 1_u64 << 37);
    assert_eq!(events & crate::FEATURE_NON_CHAT_WORKLOADS, 0);
}

/// The re-exported feature bits must equal the generated header values: this
/// is the unit-level drift guard for the value aliases in `abi.rs`.
#[test]
fn feature_bits_match_the_generated_header_values() {
    assert_eq!(
        crate::FEATURE_STAGE_PLAN,
        crate::bindings::skippy::skippy_feature::StagePlan as u64
    );
    assert_eq!(
        crate::FEATURE_RUNTIME_EVENTS,
        crate::bindings::skippy::skippy_feature::RuntimeEvents as u64
    );
    assert_eq!(
        crate::FEATURE_SPEECH_RECOGNITION,
        crate::FEATURE_SPEECH_RECOGNITION
    );
    assert_eq!(crate::FEATURE_STAGE_PLAN, 1 << 17);
    assert_eq!(crate::FEATURE_SPEECH_RECOGNITION, 1 << 41);
}

#[test]
fn versioned_descriptors_advertise_their_native_abi_size() {
    let info = WorkloadInfoV1::default();
    assert_eq!(info.abi_version, crate::WORKLOAD_INFO_V1_ABI_VERSION);
    assert_eq!(
        info.struct_size,
        u32::try_from(std::mem::size_of::<WorkloadInfoV1>()).unwrap()
    );
    assert_eq!(
        crate::LAYA_INFO_V1_ABI_VERSION,
        crate::bindings::skippy::SKIPPY_LAYA_INFO_V1_ABI_VERSION
    );
    assert_eq!(
        crate::LayaInfoV1::default().struct_size as usize,
        std::mem::size_of::<crate::LayaInfoV1>()
    );
}

/// The linked native library must advertise the same independent feature bits.
#[test]
#[cfg(not(feature = "dynamic-runtime"))]
fn native_workload_feature_preserves_runtime_event_features() {
    // SAFETY: This process-global ABI query takes no pointers or mutable state.
    let features = unsafe { crate::skippy_abi_features() };
    for expected in [
        crate::FEATURE_NON_CHAT_WORKLOADS,
        crate::FEATURE_RUNTIME_EVENT_REPORTER,
        crate::FEATURE_MODEL_LOAD_EVENTS_V2,
        crate::FEATURE_KV_EVENTS,
        crate::FEATURE_DEVICE_EVENTS,
        crate::FEATURE_DIAGNOSTIC_EVENTS,
        crate::FEATURE_UNLOAD_EVENTS,
    ] {
        assert_ne!(
            features & expected,
            0,
            "native feature {expected:#x} is absent"
        );
    }
}

#[test]
fn accepts_current_patch_runtime() {
    assert!(runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
    )));
}

#[test]
fn rejects_pre_workload_abi_even_with_the_same_major_and_minor() {
    // 0.1.60 predates the mandatory workload exports in the dynamic facade.
    assert!(!runtime_abi_supported(version(0, 1, 60)));
}

#[test]
#[cfg(not(feature = "dynamic-runtime"))]
fn linked_native_abi_matches_the_rust_facade() {
    // SAFETY: The statically linked symbol takes no arguments and owns no
    // mutable state.
    let native = unsafe { crate::bindings::skippy::skippy_abi_version() };
    assert_eq!(native.major, ABI_VERSION_MAJOR);
    assert_eq!(native.minor, ABI_VERSION_MINOR);
    assert_eq!(native.patch, ABI_VERSION_PATCH);
}

#[test]
fn rejects_other_patch_runtimes() {
    assert!(!runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH + 1,
    )));
    if let Some(lower_patch) = ABI_VERSION_PATCH.checked_sub(1) {
        assert!(!runtime_abi_supported(version(
            ABI_VERSION_MAJOR,
            ABI_VERSION_MINOR,
            lower_patch,
        )));
    }
}

#[test]
fn rejects_major_and_minor_mismatches() {
    assert!(!runtime_abi_supported(version(
        ABI_VERSION_MAJOR + 1,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
    )));
    assert!(!runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR + 1,
        ABI_VERSION_PATCH,
    )));
}

#[test]
#[cfg(not(feature = "dynamic-runtime"))]
fn native_mtmd_defaults_cross_the_ffi_boundary() {
    // SAFETY: the native default-params query allocates nothing and borrows
    // nothing from the caller.
    let params = unsafe { mtmd_context_params_default() };

    assert_eq!(params.batch_max_tokens, 1024);
    assert!(params.progress_callback.is_none());
    assert!(params.progress_callback_user_data.is_null());
}
