use super::test_support::*;
use super::*;
use crate::inference::skippy::SkippyTelemetryOptions;
use skippy_protocol::LoadMode;
use skippy_runtime::package::{
    PackageExtensionPolicyInfo, PackageGenerationInfo, PackageSpeculativeDecodingInfo,
    PackageSpeculativeProposerInfo, PackageSpeculativeStrategyInfo, PackageWindowPolicyInfo,
};
use std::collections::BTreeMap;

fn native_mtp_generation() -> PackageGenerationInfo {
    let mut proposers = BTreeMap::new();
    proposers.insert(
        "mtp".to_string(),
        PackageSpeculativeProposerInfo {
            proposer_type: "native-mtp".to_string(),
            prediction_depth: Some(1),
            layer_indices: vec![46],
            ngram_min: None,
            ngram_max: None,
            max_proposal_tokens: None,
            history_scope: None,
        },
    );
    let mut strategies = BTreeMap::new();
    strategies.insert(
        "mtp".to_string(),
        PackageSpeculativeStrategyInfo {
            strategy_type: "native-mtp".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            window_policy: Some(PackageWindowPolicyInfo {
                default: "fixed".to_string(),
                initial_window: 1,
                min_window: 1,
                max_window: 1,
            }),
            proposer: Some("mtp".to_string()),
            primary: None,
            extender: None,
            extension_policy: None,
        },
    );

    PackageGenerationInfo {
        speculative_decoding: Some(PackageSpeculativeDecodingInfo {
            default: "mtp".to_string(),
            proposers,
            strategies,
        }),
    }
}

fn native_mtp_cache_generation() -> PackageGenerationInfo {
    let mut proposers = BTreeMap::new();
    proposers.insert(
        "mtp".to_string(),
        PackageSpeculativeProposerInfo {
            proposer_type: "native-mtp".to_string(),
            prediction_depth: Some(1),
            layer_indices: vec![46],
            ngram_min: None,
            ngram_max: None,
            max_proposal_tokens: None,
            history_scope: None,
        },
    );
    proposers.insert(
        "cache".to_string(),
        PackageSpeculativeProposerInfo {
            proposer_type: "ngram-cache".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            ngram_min: Some(2),
            ngram_max: Some(4),
            max_proposal_tokens: Some(10),
            history_scope: Some("request".to_string()),
        },
    );
    let mut strategies = BTreeMap::new();
    strategies.insert(
        "mtp-cache".to_string(),
        PackageSpeculativeStrategyInfo {
            strategy_type: "composite".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            window_policy: Some(PackageWindowPolicyInfo {
                default: "adaptive".to_string(),
                initial_window: 2,
                min_window: 1,
                max_window: 6,
            }),
            proposer: None,
            primary: Some("mtp".to_string()),
            extender: Some("cache".to_string()),
            extension_policy: Some(PackageExtensionPolicyInfo { max_tokens: 8 }),
        },
    );
    PackageGenerationInfo {
        speculative_decoding: Some(PackageSpeculativeDecodingInfo {
            default: "mtp-cache".to_string(),
            proposers,
            strategies,
        }),
    }
}

fn ngram_cache_generation() -> PackageGenerationInfo {
    let mut proposers = BTreeMap::new();
    proposers.insert(
        "cache".to_string(),
        PackageSpeculativeProposerInfo {
            proposer_type: "ngram-cache".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            ngram_min: Some(2),
            ngram_max: Some(4),
            max_proposal_tokens: Some(6),
            history_scope: Some("request".to_string()),
        },
    );
    let mut strategies = BTreeMap::new();
    strategies.insert(
        "ngram-cache".to_string(),
        PackageSpeculativeStrategyInfo {
            strategy_type: "ngram-cache".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            window_policy: Some(PackageWindowPolicyInfo {
                default: "fixed".to_string(),
                initial_window: 6,
                min_window: 1,
                max_window: 6,
            }),
            proposer: Some("cache".to_string()),
            primary: None,
            extender: None,
            extension_policy: None,
        },
    );
    PackageGenerationInfo {
        speculative_decoding: Some(PackageSpeculativeDecodingInfo {
            default: "ngram-cache".to_string(),
            proposers,
            strategies,
        }),
    }
}

fn ngram_suffix_generation() -> PackageGenerationInfo {
    let mut proposers = BTreeMap::new();
    proposers.insert(
        "suffix".to_string(),
        PackageSpeculativeProposerInfo {
            proposer_type: "ngram-suffix".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            ngram_min: Some(5),
            ngram_max: Some(32),
            max_proposal_tokens: Some(48),
            history_scope: Some("request".to_string()),
        },
    );
    let mut strategies = BTreeMap::new();
    strategies.insert(
        "ngram-suffix".to_string(),
        PackageSpeculativeStrategyInfo {
            strategy_type: "ngram-suffix".to_string(),
            prediction_depth: None,
            layer_indices: Vec::new(),
            window_policy: Some(PackageWindowPolicyInfo {
                default: "fixed".to_string(),
                initial_window: 32,
                min_window: 1,
                max_window: 32,
            }),
            proposer: Some("suffix".to_string()),
            primary: None,
            extender: None,
            extension_policy: None,
        },
    );
    PackageGenerationInfo {
        speculative_decoding: Some(PackageSpeculativeDecodingInfo {
            default: "ngram-suffix".to_string(),
            proposers,
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
fn speculative_strategy_auto_uses_hardware_model_path_for_direct_gguf_detection() {
    let requested_model_file = temp_model_file();
    let resolved_model_file =
        temp_model_file_with_tensor_names(&["blk.40.nextn.eh_proj.weight"], None);
    let mesh_config = parse_config(&format!(
        r#"
[[models]]
model = "unsloth/Qwen3.6-MTP-GGUF"

[models.hardware]
model_path = "{}"
"#,
        resolved_model_file.path().display()
    ));

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "unsloth/Qwen3.6-MTP-GGUF",
        model_path: requested_model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("hardware model_path native MTP tensors should enable auto native MTP");

    assert_eq!(
        resolved.hardware.resolved_model_path,
        resolved_model_file.path()
    );
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
    assert_eq!(openai.native_mtp_max_tokens, 3);
    assert_eq!(openai.native_mtp_min_tokens, 0);
}

#[test]
fn package_composite_strategy_resolves_native_mtp_with_cache_extension() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "mtp-cache"
ngram_max_proposal_tokens = 4
extension_max_tokens = 8
verify_window_pipeline_depth = 1

[[models]]
model = "meshllm/GLM-4.7-Flash-MTP-GGUF"

[models.speculative]
ngram_max_proposal_tokens = 9
extension_max_tokens = 7
verify_window_pipeline_depth = 2
"#,
    );
    assert_eq!(
        mesh_config
            .defaults
            .as_ref()
            .and_then(|defaults| defaults.speculative.as_ref())
            .and_then(|speculative| speculative.ngram_max_proposal_tokens),
        Some(4)
    );
    assert_eq!(
        mesh_config
            .models
            .first()
            .and_then(|model| model.speculative.as_ref())
            .and_then(|speculative| speculative.ngram_max_proposal_tokens),
        Some(9)
    );
    let model_file = temp_model_file();
    let generation = native_mtp_cache_generation();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .expect("package composite strategy should resolve");

    assert!(resolved.speculative.native_mtp_enabled);
    assert_eq!(
        resolved.speculative.decode.effective_strategy,
        "native-mtp+ngram-cache"
    );
    let ngram = resolved
        .speculative
        .decode
        .ngram
        .as_ref()
        .expect("cache proposer should resolve");
    assert_eq!(ngram.min_ngram, 2);
    assert_eq!(ngram.max_ngram, 4);
    assert_eq!(ngram.max_proposal_tokens, 9);
    let extension = resolved
        .speculative
        .decode
        .extension
        .as_ref()
        .expect("extension policy should resolve");
    assert_eq!(extension.max_tokens, 7);
    assert_eq!(resolved.speculative.decode.verify_window.min_tokens, 1);
    assert_eq!(resolved.speculative.decode.verify_window.max_tokens, 6);
    assert_eq!(resolved.speculative.decode.verify_window.pipeline_depth, 2);
}

#[test]
fn package_cache_strategy_uses_the_declared_verify_window() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-cache"
"#,
    );
    let model_file = temp_model_file();
    let generation = ngram_cache_generation();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .expect("package cache strategy should resolve");

    assert!(!resolved.speculative.native_mtp_enabled);
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("package cache strategy should build OpenAI args");
    assert_eq!(openai.speculative_window, 6);
    assert_eq!(
        openai.speculative.ngram.as_ref().map(|ngram| ngram.kind),
        Some(skippy_server::NgramProposerKind::Cache)
    );
}

#[test]
fn package_suffix_strategy_resolves_as_a_standalone_proposer() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-suffix"
"#,
    );
    let model_file = temp_model_file();
    let generation = ngram_suffix_generation();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/coding-model",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .expect("package suffix strategy should resolve without native MTP");

    assert!(!resolved.speculative.native_mtp_enabled);
    assert_eq!(
        resolved.speculative.decode.effective_strategy,
        "ngram-suffix"
    );
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("package suffix strategy should build OpenAI args");
    assert_eq!(openai.speculative_window, 48);
    assert_eq!(
        openai.speculative.ngram.as_ref().map(|ngram| ngram.kind),
        Some(skippy_server::NgramProposerKind::Suffix)
    );
}

#[test]
fn direct_native_mtp_can_use_a_request_local_cache_extension() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "mtp"
ngram_min = 2
ngram_max = 4
ngram_max_proposal_tokens = 6
"#,
    );
    let model_file = temp_model_file_with_tensor_names(&["blk.23.nextn.eh_proj.weight"], None);

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("direct native MTP with cache extension should resolve");

    assert!(resolved.speculative.native_mtp_enabled);
    assert_eq!(
        resolved.speculative.decode.effective_strategy,
        "native-mtp+ngram-cache"
    );
    let extension = resolved
        .speculative
        .decode
        .extension
        .as_ref()
        .expect("direct cache strategy should synthesize an extension plan");
    assert_eq!(extension.max_tokens, 6);
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("direct cache strategy should build OpenAI args");
    assert!(openai.native_mtp_enabled);
    assert_eq!(
        openai.speculative.ngram.as_ref().map(|ngram| ngram.kind),
        Some(skippy_server::NgramProposerKind::Cache)
    );
}

#[test]
fn direct_native_mtp_can_use_a_request_local_suffix_extension() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "mtp"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
extension_max_tokens = 48
verify_window_min_tokens = 1
verify_window_max_tokens = 32
verify_window_pipeline_depth = 2
"#,
    );
    let model_file = temp_model_file_with_tensor_names(&["blk.23.nextn.eh_proj.weight"], None);

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("direct native MTP with suffix extension should resolve");

    assert!(resolved.speculative.native_mtp_enabled);
    assert_eq!(
        resolved.speculative.decode.effective_strategy,
        "native-mtp+ngram-suffix"
    );
    let ngram = resolved
        .speculative
        .decode
        .ngram
        .as_ref()
        .expect("suffix proposer should resolve");
    assert_eq!(ngram.kind, skippy_server::NgramProposerKind::Suffix);
    assert_eq!(ngram.min_ngram, 5);
    assert_eq!(ngram.max_ngram, 32);
    assert_eq!(ngram.max_proposal_tokens, 48);
    let extension = resolved
        .speculative
        .decode
        .extension
        .as_ref()
        .expect("suffix strategy should synthesize an extension plan");
    assert_eq!(extension.max_tokens, 48);
    assert_eq!(resolved.speculative.decode.verify_window.max_tokens, 32);
    assert_eq!(resolved.speculative.decode.verify_window.pipeline_depth, 2);
}

#[test]
fn direct_cache_strategy_rejects_an_unsupported_cache_window() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-cache"
ngram_proposer = "cache"
ngram_min = 2
ngram_max = 5
ngram_max_proposal_tokens = 6
"#,
    );
    let model_file = temp_model_file();

    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect_err("cache windows above the llama.cpp limit must be rejected");

    assert!(
        error
            .to_string()
            .contains("must not exceed llama.cpp limit 4")
    );
}

#[test]
fn direct_cache_strategy_resolves_a_request_local_cache_proposer() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-cache"
ngram_min = 2
ngram_max = 4
ngram_max_proposal_tokens = 6
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("direct cache strategy should resolve");

    assert!(!resolved.speculative.native_mtp_enabled);
    assert_eq!(
        resolved.speculative.decode.effective_strategy,
        "ngram-cache"
    );
    let ngram = resolved
        .speculative
        .decode
        .ngram
        .as_ref()
        .expect("direct cache strategy should select an N-gram proposer");
    assert_eq!(ngram.kind, skippy_server::NgramProposerKind::Cache);
    assert_eq!(ngram.min_ngram, 2);
    assert_eq!(ngram.max_ngram, 4);
    assert_eq!(ngram.max_proposal_tokens, 6);
}

#[test]
fn direct_suffix_strategy_resolves_without_native_mtp() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/coding-model",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("direct suffix strategy should resolve without native MTP");

    assert!(!resolved.speculative.native_mtp_enabled);
    assert_eq!(resolved.speculative.mode, "ngram");
    assert_eq!(
        resolved.speculative.decode.effective_strategy,
        "ngram-suffix"
    );
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("direct suffix strategy should build OpenAI args");
    assert_eq!(openai.speculative_window, 48);
    assert_eq!(
        openai.speculative.ngram.as_ref().map(|ngram| ngram.kind),
        Some(skippy_server::NgramProposerKind::Suffix)
    );
}

#[test]
fn strategy_disabled_ignores_inherited_ngram_bounds() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/coding-model",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("disabled strategy should resolve");

    assert!(!resolved.speculative.native_mtp_enabled);
    assert_eq!(resolved.speculative.mode, "disabled");
    assert!(resolved.speculative.decode.ngram.is_none());
    assert_eq!(resolved.speculative.decode.effective_strategy, "disabled");
}

#[test]
fn explicit_standalone_strategy_without_bounds_fails() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-suffix"
"#,
    );
    let model_file = temp_model_file();

    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/coding-model",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect_err("an explicit N-gram strategy without bounds must fail");

    assert!(
        error.to_string().contains("both ngram_min and ngram_max"),
        "{error}"
    );
}

#[test]
fn standalone_ngram_rejected_for_single_stage_serving() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "ngram-suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/coding-model",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("standalone suffix strategy should resolve");

    // Multi-stage (staged) serving has a verify path and builds fine.
    resolved
        .to_embedded_openai_args(4096, true)
        .expect("staged serving should build OpenAI args");

    // Single-stage/direct serving has no N-gram verify path, so it must reject
    // rather than silently run target-only while reporting a proposer.
    let error = resolved
        .to_embedded_openai_args(0, false)
        .expect_err("single-stage standalone N-gram must be rejected");
    assert!(
        error
            .to_string()
            .contains("requires multi-stage split serving"),
        "{error}"
    );
}

#[test]
fn speculative_strategy_native_mtp_rejects_direct_gguf_without_proven_support() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "mtp"
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

    assert!(error.contains("requires proven native MTP support"));
}

#[test]
fn speculative_strategy_native_mtp_accepts_external_mtp_sidecar() {
    let draft_file = temp_model_file_with_tensor_names(&["blk.10.nextn.eh_proj.weight"], None);
    let draft_path = draft_file.path().display().to_string();
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
strategy = "mtp"
draft_model_path = "{draft_path}"
draft_max_tokens = 3
draft_min_tokens = 0
"#
    ));
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "google/gemma-4-31b-it:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("external MTP sidecar should prove native MTP support");

    assert!(resolved.speculative.native_mtp_enabled);
    assert_eq!(resolved.speculative.mode, "disabled");
    let openai = resolved
        .to_embedded_openai_args(4096, false)
        .expect("openai args should build");
    assert_eq!(
        openai.native_mtp_draft_model_path.as_deref(),
        Some(draft_file.path())
    );
    assert!(openai.draft_model_path.is_none());
    assert_eq!(openai.native_mtp_max_tokens, 3);
    assert_eq!(openai.native_mtp_min_tokens, 0);
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
strategy = "mtp"
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

    assert!(error.contains("requires proven native MTP support"));
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
