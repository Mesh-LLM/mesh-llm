use super::test_support::*;
use super::*;
use crate::inference::skippy::SkippyTelemetryOptions;
use skippy_protocol::LoadMode;
use skippy_runtime::package::{
    PackageGenerationInfo, PackageSpeculativeDecodingInfo, PackageSpeculativeStrategyInfo,
    PackageWindowPolicyInfo,
};
use std::collections::BTreeMap;

fn native_mtp_generation() -> PackageGenerationInfo {
    let mut strategies = BTreeMap::new();
    strategies.insert(
        "native-mtp-n1".to_string(),
        PackageSpeculativeStrategyInfo {
            strategy_type: "native-mtp".to_string(),
            prediction_depth: Some(1),
            layer_indices: vec![46],
            window_policy: Some(PackageWindowPolicyInfo {
                default: "fixed".to_string(),
                initial_window: 1,
                min_window: 1,
                max_window: 1,
            }),
        },
    );

    PackageGenerationInfo {
        speculative_decoding: Some(PackageSpeculativeDecodingInfo {
            default: "native-mtp-n1".to_string(),
            strategies,
        }),
    }
}

#[test]
fn speculative_strategy_auto_without_package_generation_disables_native_mtp() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("default speculative strategy should resolve");

    assert_eq!(resolved.speculative.strategy, "auto");
    assert!(!resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(!load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert!(!stage.native_mtp_enabled);
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(!openai.native_mtp_enabled);
}

#[test]
fn speculative_strategy_auto_detects_direct_gguf_native_mtp_tensors() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file_with_tensor_names(&["blk.23.nextn.eh_proj.weight"], None);

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "unsloth/Qwen3.6-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("direct GGUF native MTP tensors should enable auto native MTP");

    assert_eq!(resolved.speculative.strategy, "auto");
    assert!(resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert!(stage.native_mtp_enabled);
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(openai.native_mtp_enabled);
}

#[test]
fn speculative_strategy_auto_detects_direct_gguf_native_mtp_metadata() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file_with_tensor_names(&[], Some(1));

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "unsloth/Qwen3.6-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("direct GGUF native MTP metadata should enable auto native MTP");

    assert!(resolved.speculative.native_mtp_enabled);
}

#[test]
fn speculative_strategy_auto_uses_package_native_mtp_default() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file();
    let generation = native_mtp_generation();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .expect("package native MTP default should resolve");

    assert_eq!(resolved.speculative.strategy, "auto");
    assert!(resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert!(stage.native_mtp_enabled);
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(openai.native_mtp_enabled);
}

#[test]
fn speculative_strategy_native_mtp_rejects_direct_gguf_without_proven_support() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "native-mtp-n1"
"#,
    );
    let model_file = temp_model_file();

    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("requires proven native-mtp-n1 support"));
}

#[test]
fn speculative_default_false_disables_auto_native_mtp_for_direct_gguf() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
spec_default = false
"#,
    );
    let model_file = temp_model_file_with_tensor_names(&["blk.23.nextn.eh_proj.weight"], None);

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "unsloth/Qwen3.6-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("spec_default=false should resolve");

    assert_eq!(resolved.speculative.strategy, "auto");
    assert!(!resolved.speculative.native_mtp_enabled);
}

#[test]
fn speculative_strategy_native_mtp_rejects_package_without_native_mtp_metadata() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "native-mtp-n1"
"#,
    );
    let model_file = temp_model_file();
    let generation = PackageGenerationInfo {
        speculative_decoding: None,
    };

    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/package-without-mtp",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("requires proven native-mtp-n1 support"));
}

#[test]
fn speculative_strategy_disabled_reaches_stage_and_openai_args() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("disabled speculative strategy should resolve");

    assert_eq!(resolved.speculative.strategy, "disabled");
    assert!(!resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(!load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert!(!stage.native_mtp_enabled);
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(!openai.native_mtp_enabled);
}
