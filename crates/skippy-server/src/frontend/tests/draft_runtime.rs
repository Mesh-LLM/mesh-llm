use super::*;

#[test]
fn draft_loader_controls_reach_the_native_runtime_config() {
    let stage = support::prefix_cache_test_config();
    let speculative = SpeculativeDecodeConfig {
        draft_device: Some("CUDA1".to_string()),
        draft_threads: Some(6),
        draft_cache_type_k: "q8_0".to_string(),
        draft_cache_type_v: "q4_0".to_string(),
        ..SpeculativeDecodeConfig::default()
    };

    let runtime = draft_runtime_config(
        &stage,
        Some(12),
        &speculative,
        skippy_runtime::MtpSource::External,
        24,
    )
    .expect("map draft runtime config");

    assert_eq!(runtime.selected_backend_device.as_deref(), Some("CUDA1"));
    assert_eq!(runtime.n_threads, Some(6));
    assert_eq!(runtime.n_threads_batch, Some(6));
    assert_eq!(runtime.cache_type_k, skippy_runtime::GGML_TYPE_Q8_0);
    assert_eq!(runtime.cache_type_v, skippy_runtime::GGML_TYPE_Q4_0);
    assert_eq!(runtime.n_gpu_layers, 12);
    assert_eq!(runtime.mtp_source, skippy_runtime::MtpSource::External);
}
