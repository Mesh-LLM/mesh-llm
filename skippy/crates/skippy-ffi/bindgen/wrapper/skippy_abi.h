#ifndef SKIPPY_FFI_BINDGEN_SKIPPY_ABI_WRAPPER_H
#define SKIPPY_FFI_BINDGEN_SKIPPY_ABI_WRAPPER_H

/*
 * Narrow bindgen entry point for Skippy's own public native-runtime C ABI.
 *
 * bindgen consumes this wrapper instead of wandering the llama.cpp include
 * tree directly, so the supported ABI surface is exactly the umbrella header
 * installed by the patched llama.cpp build (include/skippy.h aggregates every
 * capability header under include/skippy/).
 *
 * Feature bits 31+ are declared in common.h as uint64_t constant-expression
 * macros with casts (`((uint64_t)1 << 31)`), which bindgen does not translate.
 * Bridge them through `static const` definitions initialized FROM the macros:
 * the header stays the single source of truth, and any header-side value
 * change flows into the generated constants and fails the CI freshness check.
 *
 * This file is an input to deterministic binding regeneration; keep it free
 * of platform conditionals so committed bindings stay identical across
 * macOS, Linux, and Windows toolchains.
 */

#include "skippy.h"

#include <stdint.h>

static const uint64_t skippy_ffi_bridge_feature_runtime_event_reporter =
    SKIPPY_FEATURE_RUNTIME_EVENT_REPORTER;
static const uint64_t skippy_ffi_bridge_feature_model_load_events_v2 =
    SKIPPY_FEATURE_MODEL_LOAD_EVENTS_V2;
static const uint64_t skippy_ffi_bridge_feature_kv_events = SKIPPY_FEATURE_KV_EVENTS;
static const uint64_t skippy_ffi_bridge_feature_device_events = SKIPPY_FEATURE_DEVICE_EVENTS;
static const uint64_t skippy_ffi_bridge_feature_diagnostic_events =
    SKIPPY_FEATURE_DIAGNOSTIC_EVENTS;
static const uint64_t skippy_ffi_bridge_feature_unload_events = SKIPPY_FEATURE_UNLOAD_EVENTS;
static const uint64_t skippy_ffi_bridge_feature_non_chat_workloads =
    SKIPPY_FEATURE_NON_CHAT_WORKLOADS;
static const uint64_t skippy_ffi_bridge_feature_system_one = SKIPPY_FEATURE_SYSTEM_ONE;
static const uint64_t skippy_ffi_bridge_feature_cachegen_kv_page = SKIPPY_FEATURE_CACHEGEN_KV_PAGE;
static const uint64_t skippy_ffi_bridge_feature_laya_decisions = SKIPPY_FEATURE_LAYA_DECISIONS;
static const uint64_t skippy_ffi_bridge_feature_speech_recognition =
    SKIPPY_FEATURE_SPEECH_RECOGNITION;

/* Same cast-form macro bridge as the feature bits above. */
static const uint32_t skippy_ffi_bridge_activation_part_optional = SKIPPY_ACTIVATION_PART_OPTIONAL;

#endif /* SKIPPY_FFI_BINDGEN_SKIPPY_ABI_WRAPPER_H */
