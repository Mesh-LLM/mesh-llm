use std::mem::{offset_of, size_of};

use crate::{
    ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH, AbiVersion, runtime_abi_supported,
};

#[cfg(target_pointer_width = "64")]
use crate::{
    MtmdContextParams, MtmdHelperBitmapWrapper, MtmdHelperInitOpt, MtmdHelperVideoInitParams,
    MtmdInputText,
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
    // Mirrors `struct mtmd_context_params` in tools/mtmd/mtmd.h. `device` sits
    // second, right after `use_gpu`; leaving it out shifts everything below it
    // by 8 bytes and makes the struct 16 bytes short, so the C side reads
    // `progress_callback` from past the end of what Rust allocated.
    assert_eq!(size_of::<MtmdContextParams>(), 96);
    assert_eq!(offset_of!(MtmdContextParams, device), 8);
    assert_eq!(offset_of!(MtmdContextParams, batch_max_tokens), 72);
    assert_eq!(offset_of!(MtmdContextParams, progress_callback), 80);
    assert_eq!(
        offset_of!(MtmdContextParams, progress_callback_user_data),
        88
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn mtmd_input_text_matches_native_layout() {
    // Mirrors `struct mtmd_input_text` in tools/mtmd/mtmd.h. `mtmd_tokenize`
    // reads `text_len` to size the prompt, so a missing field here hands the
    // native side a length built from uninitialised padding.
    assert_eq!(size_of::<MtmdInputText>(), 24);
    assert_eq!(offset_of!(MtmdInputText, text_len), 8);
    assert_eq!(offset_of!(MtmdInputText, add_special), 16);
    assert_eq!(offset_of!(MtmdInputText, parse_special), 17);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn mtmd_helper_bitmap_types_match_native_layout() {
    // Mirrors `mtmd_helper_bitmap_wrapper`, `mtmd_helper_init_opt` and
    // `mtmd_helper_video_init_params` in tools/mtmd/mtmd-helper.h.
    // `mtmd_helper_bitmap_init_from_buf` returns the wrapper by value and takes
    // the opt by value, so both layouts are part of the calling convention.
    assert_eq!(size_of::<MtmdHelperBitmapWrapper>(), 16);
    assert_eq!(offset_of!(MtmdHelperBitmapWrapper, bitmap), 0);
    assert_eq!(offset_of!(MtmdHelperBitmapWrapper, video_ctx), 8);

    assert_eq!(size_of::<MtmdHelperVideoInitParams>(), 24);
    assert_eq!(offset_of!(MtmdHelperVideoInitParams, fps_target), 0);
    assert_eq!(offset_of!(MtmdHelperVideoInitParams, ffmpeg_bin_dir), 8);
    assert_eq!(
        offset_of!(MtmdHelperVideoInitParams, timestamp_interval_ms),
        16
    );

    assert_eq!(size_of::<MtmdHelperInitOpt>(), 24);
    assert_eq!(offset_of!(MtmdHelperInitOpt, video_params), 0);
}

#[test]
#[cfg(not(feature = "dynamic-runtime"))]
fn native_mtmd_defaults_cross_the_ffi_boundary() {
    let params = unsafe { mtmd_context_params_default() };

    assert_eq!(params.batch_max_tokens, 1024);
    assert!(params.progress_callback.is_none());
    assert!(params.progress_callback_user_data.is_null());
}
