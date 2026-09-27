use super::super::parser_policy::{BACKEND_CATEGORY, MODEL_CATEGORY};
use super::super::*;
use skippy_ffi::{
    FEATURE_DEVICE_EVENTS, FEATURE_KV_EVENTS, FEATURE_MODEL_LOAD_EVENTS_V2,
    FEATURE_RUNTIME_EVENT_REPORTER, FEATURE_RUNTIME_EVENTS,
};
use std::ffi::CString;
use std::ptr;

const CATEGORIES: [&str; 5] = ["backend", "model", "memory", "kv_cache", "tokenizer"];
const FULL_STRUCTURED_COVERAGE: u64 = FEATURE_RUNTIME_EVENT_REPORTER
    | FEATURE_RUNTIME_EVENTS
    | FEATURE_MODEL_LOAD_EVENTS_V2
    | FEATURE_KV_EVENTS
    | FEATURE_DEVICE_EVENTS;

fn forwarded_categories(mode: NativeLogParserMode, confirmed: u64) -> Vec<&'static str> {
    let report = crate::CapabilityReport {
        confirmed,
        health_messages: Vec::new(),
    };
    let policy = NativeLogParserPolicy::new(mode, &report);
    CATEGORIES
        .into_iter()
        .filter(|category| policy.forwards(category))
        .collect()
}

#[test]
fn auto_disables_backend_when_device_events_confirmed() {
    assert_eq!(
        forwarded_categories(
            NativeLogParserMode::Auto,
            FEATURE_RUNTIME_EVENT_REPORTER | FEATURE_DEVICE_EVENTS
        ),
        vec!["model", "memory", "kv_cache", "tokenizer"]
    );
}

#[test]
fn auto_disables_model_when_structured_model_events_are_confirmed() {
    assert_eq!(
        forwarded_categories(
            NativeLogParserMode::Auto,
            FEATURE_RUNTIME_EVENT_REPORTER | FEATURE_MODEL_LOAD_EVENTS_V2 | FEATURE_RUNTIME_EVENTS
        ),
        vec!["backend", "kv_cache"]
    );
}

#[test]
fn auto_keeps_model_fallback_when_model_open_events_are_missing() {
    assert_eq!(
        forwarded_categories(
            NativeLogParserMode::Auto,
            FEATURE_RUNTIME_EVENT_REPORTER | FEATURE_MODEL_LOAD_EVENTS_V2
        ),
        vec!["backend", "model", "kv_cache"]
    );
}

#[test]
fn auto_disables_kv_cache_when_kv_events_confirmed() {
    assert_eq!(
        forwarded_categories(
            NativeLogParserMode::Auto,
            FEATURE_RUNTIME_EVENT_REPORTER | FEATURE_KV_EVENTS
        ),
        vec!["backend", "model", "memory", "tokenizer"]
    );
}

#[test]
fn auto_disables_every_parsed_category_with_full_structured_coverage() {
    assert_eq!(
        forwarded_categories(NativeLogParserMode::Auto, FULL_STRUCTURED_COVERAGE),
        Vec::<&str>::new()
    );
}

#[test]
fn auto_keeps_only_the_dedicated_model_fallback_note() {
    let report = crate::CapabilityReport {
        confirmed: FULL_STRUCTURED_COVERAGE,
        health_messages: Vec::new(),
    };
    let policy = NativeLogParserPolicy::new(NativeLogParserMode::Auto, &report);
    assert!(policy.forwards_model_fallback_note());
    assert!(!policy.forwards("model"));
}

#[test]
fn auto_forwards_everything_without_reporter_family() {
    assert_eq!(
        forwarded_categories(
            NativeLogParserMode::Auto,
            FULL_STRUCTURED_COVERAGE & !FEATURE_RUNTIME_EVENT_REPORTER
        ),
        CATEGORIES.to_vec()
    );
}

#[test]
fn auto_forwards_everything_on_legacy_runtime() {
    assert_eq!(
        forwarded_categories(NativeLogParserMode::Auto, 0),
        CATEGORIES.to_vec()
    );
}

#[test]
fn enabled_and_disabled_ignore_capabilities() {
    for confirmed in [0, FULL_STRUCTURED_COVERAGE, u64::MAX] {
        let report = crate::CapabilityReport {
            confirmed,
            health_messages: Vec::new(),
        };
        assert_eq!(
            forwarded_categories(NativeLogParserMode::Enabled, confirmed),
            CATEGORIES.to_vec()
        );
        assert!(
            NativeLogParserPolicy::new(NativeLogParserMode::Enabled, &report)
                .forwards_model_fallback_note()
        );
        assert!(forwarded_categories(NativeLogParserMode::Disabled, confirmed).is_empty());
        assert!(
            !NativeLogParserPolicy::new(NativeLogParserMode::Disabled, &report)
                .forwards_model_fallback_note()
        );
    }
}

#[test]
fn auto_keeps_only_safetensors_fallback_note_with_full_coverage() {
    let _native_log_guard = native_log_test_guard();
    struct ResetForwarding;
    impl Drop for ResetForwarding {
        fn drop(&mut self) {
            unregister_filtered_native_logs();
            set_filtered_native_logs_enabled(false);
        }
    }
    let _reset = ResetForwarding;
    let mut receiver = register_filtered_native_logs();
    let line =
        CString::new("init_tokenizer: initializing tokenizer for type 2\n").expect("cstring");

    configure_native_log_parser(NativeLogParserPolicy::new(
        NativeLogParserMode::Auto,
        &crate::CapabilityReport {
            confirmed: FULL_STRUCTURED_COVERAGE,
            health_messages: Vec::new(),
        },
    ));
    unsafe { write_native_log(0, line.as_ptr(), ptr::null_mut()) };
    write_native_log_note("ordinary GGUF model note");
    write_native_log_fallback_note(
        "SafeTensors source loading does not yet emit native model-open events",
    );
    let note = receiver
        .try_recv()
        .expect("SafeTensors model-open fallback note should be forwarded");
    assert_eq!(note.category, "model");
    assert!(note.message.contains("SafeTensors source loading"));
    assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

    configure_native_log_parser(NativeLogParserPolicy::new(
        NativeLogParserMode::Auto,
        &crate::CapabilityReport::default(),
    ));
    unsafe { write_native_log(0, line.as_ptr(), ptr::null_mut()) };
    assert_eq!(
        receiver.try_recv().map(|event| event.category),
        Ok("tokenizer")
    );
}

#[test]
fn native_log_note_obeys_the_model_category_mask() {
    let _native_log_guard = native_log_test_guard();
    let mut receiver = register_filtered_native_logs();
    struct RestoreForwardingMask(u8);

    impl Drop for RestoreForwardingMask {
        fn drop(&mut self) {
            NATIVE_LOG_FORWARDING_MASK.store(self.0, Ordering::Relaxed);
        }
    }

    let _mask_guard =
        RestoreForwardingMask(NATIVE_LOG_FORWARDING_MASK.swap(BACKEND_CATEGORY, Ordering::Relaxed));
    write_native_log_note("hidden model note despite backend forwarding");
    assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

    NATIVE_LOG_FORWARDING_MASK.store(MODEL_CATEGORY, Ordering::Relaxed);
    write_native_log_note("visible model note");
    let event = receiver
        .try_recv()
        .expect("enabled model note should forward");
    assert_eq!(event.category, "model");
    assert!(event.message.contains("visible model note"));

    unregister_filtered_native_logs();
}

use tokio::sync::mpsc::error::TryRecvError;

mod native_log {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tests/native_log.rs"
    ));
}
