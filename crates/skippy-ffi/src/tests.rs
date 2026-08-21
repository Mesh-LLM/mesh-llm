use std::mem::{offset_of, size_of};

use crate::{
    ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH, AbiVersion, runtime_abi_supported,
};

#[cfg(target_pointer_width = "64")]
use crate::MtmdContextParams;

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
fn accepts_current_patch_runtime() {
    assert!(runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
    )));
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
#[cfg(target_pointer_width = "64")]
fn mtmd_context_params_matches_native_layout() {
    assert_eq!(size_of::<MtmdContextParams>(), 80);
    assert_eq!(offset_of!(MtmdContextParams, batch_max_tokens), 56);
    assert_eq!(offset_of!(MtmdContextParams, progress_callback), 64);
    assert_eq!(
        offset_of!(MtmdContextParams, progress_callback_user_data),
        72
    );
}

#[test]
#[cfg(not(feature = "dynamic-runtime"))]
fn native_mtmd_defaults_cross_the_ffi_boundary() {
    let params = unsafe { mtmd_context_params_default() };

    assert_eq!(params.batch_max_tokens, 1024);
    assert!(params.progress_callback.is_none());
    assert!(params.progress_callback_user_data.is_null());
}
