use super::test_support::*;
use super::*;
use crate::inference::skippy::SkippyTelemetryOptions;
use crate::plugin::{MeshConfig, RequestDefaultsConfig};
use skippy_protocol::LoadMode;
use std::path::Path;
use tempfile::NamedTempFile;

const FULL_SURFACE_VALID_FIXTURE: &str =
    include_str!("../../../../tests/fixtures/skippy_full_surface_valid.toml");
const FULL_SURFACE_INVALID_FIXTURE: &str =
    include_str!("../../../../tests/fixtures/skippy_full_surface_invalid.toml");

struct FullSurfaceFixture {
    mesh_config: MeshConfig,
    explicit_model: NamedTempFile,
    defaults_model: NamedTempFile,
    _projector_file: NamedTempFile,
}
fn full_surface_fixture_with_model_paths() -> FullSurfaceFixture {
    let mut mesh_config = parse_config(FULL_SURFACE_VALID_FIXTURE);
    let explicit_model = temp_model_file();
    let defaults_model = temp_model_file();
    let projector_file = NamedTempFile::new().expect("temp projector");

    mesh_config.models[0]
        .hardware
        .as_mut()
        .expect("explicit hardware")
        .model_path = Some(explicit_model.path().display().to_string());
    mesh_config.models[0]
        .hardware
        .as_mut()
        .expect("explicit hardware")
        .mmproj = Some(projector_file.path().display().to_string());
    mesh_config.models[0]
        .multimodal
        .as_mut()
        .expect("explicit multimodal")
        .mmproj = Some(projector_file.path().display().to_string());
    mesh_config.models[1]
        .hardware
        .as_mut()
        .expect("defaults hardware")
        .model_path = Some(defaults_model.path().display().to_string());

    FullSurfaceFixture {
        mesh_config,
        explicit_model,
        defaults_model,
        _projector_file: projector_file,
    }
}

fn resolve_explicit_full_surface_config(
    fixture: &FullSurfaceFixture,
    request_defaults: &RequestDefaultsConfig,
) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &fixture.mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: fixture.explicit_model.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: Some(12 * 1024 * 1024 * 1024),
        request_defaults: Some(request_defaults),
        package_generation: None,
        compact_meta: None,
    })
    .expect("explicit model should resolve")
}

fn resolve_defaults_full_surface_config(fixture: &FullSurfaceFixture) -> ResolvedSkippyConfig {
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &fixture.mesh_config,
        model_id: "ggml-org/gemma-3-270m-it-GGUF:Q8_0",
        model_path: fixture.defaults_model.path(),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: Some(12 * 1024 * 1024 * 1024),
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("defaults-only model should resolve")
}

fn assert_explicit_full_surface_resolution(explicit: &ResolvedSkippyConfig) {
    assert_eq!(explicit.model_fit.ctx_size, 16384);
    assert_eq!(explicit.model_fit.batch, 1024);
    assert_eq!(explicit.model_fit.ubatch, 128);
    assert_eq!(explicit.hardware.device.as_deref(), Some("CUDA1"));
    assert_eq!(explicit.hardware.stage_layer_start, Some(12));
    assert_eq!(explicit.hardware.stage_layer_end, Some(24));
    assert_eq!(explicit.throughput.parallel, 3);
    assert_eq!(explicit.throughput.threads, Some(10));
    assert_eq!(explicit.throughput.threads_batch, Some(6));
    assert_eq!(explicit.request_defaults.temperature, Some(0.7));
    assert_eq!(explicit.request_defaults.max_tokens, 256);
}

fn assert_explicit_full_surface_stage_config(explicit: &ResolvedSkippyConfig) {
    let stage = explicit
        .to_stage_config(Some(fake_package_identity(32)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    assert_eq!((stage.layer_start, stage.layer_end), (12, 24));
    assert_eq!(stage.n_batch, Some(1024));
    assert_eq!(stage.n_ubatch, Some(128));
    assert_eq!(stage.n_gpu_layers, 99);
}

fn assert_explicit_full_surface_runtime_options(explicit: &ResolvedSkippyConfig) {
    let runtime = explicit
        .to_embedded_runtime_options(
            &SkippyTelemetryOptions::off(),
            Some(fake_package_identity(32)),
            LoadMode::RuntimeSlice,
        )
        .expect("embedded runtime options should build");
    assert_eq!(runtime.n_threads, Some(10));
    assert_eq!(runtime.n_threads_batch, Some(6));
    assert_eq!(runtime.config.layer_start, 12);
    assert_eq!(runtime.config.layer_end, 24);
}

fn assert_explicit_full_surface_openai_args(explicit: &ResolvedSkippyConfig) {
    let openai = explicit
        .to_embedded_openai_args(4096, true)
        .expect("embedded openai args should build");
    assert_eq!(openai.prefill_chunk_policy, "schedule");
    assert_eq!(openai.prefill_chunk_size, 128);
    assert_eq!(
        openai.prefill_chunk_schedule.as_deref(),
        Some("128,256,384")
    );
    assert_eq!(openai.speculative_window, 8);
    assert_eq!(openai.draft_n_gpu_layers, Some(12));
    assert_eq!(openai.default_max_tokens, 256);
}

fn assert_defaults_full_surface_resolution(omitted: &ResolvedSkippyConfig) {
    assert_eq!(omitted.model_fit.ctx_size, 8192);
    assert_eq!(omitted.model_fit.batch, 512);
    assert_eq!(omitted.model_fit.ubatch, 128);
    assert_eq!(omitted.hardware.device.as_deref(), Some("CUDA2"));
    assert_eq!(omitted.throughput.parallel, 2);
    assert_eq!(omitted.request_defaults.temperature, Some(0.2));
    assert_eq!(omitted.request_defaults.max_tokens, 128);

    let single_stage = omitted
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("defaults-only model should remain single-stage safe");
    assert_eq!(single_stage.ctx_size, 8192);
    assert_eq!(single_stage.n_batch, Some(512));
    assert_eq!(single_stage.n_ubatch, Some(128));
    assert!(
        single_stage.package_identity.is_some(),
        "single-stage load should preserve precomputed package identity"
    );
}

#[test]
fn speculative_auto_selection_policy_without_draft_source_resolves_disabled() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
mode = "auto"
draft_selection_policy = "auto"
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
        compact_meta: None,
    })
    .expect("auto draft selection policy should not force draft resolution");

    assert_eq!(resolved.speculative.mode, "disabled");
    assert!(resolved.speculative.draft_model_path.is_none());
    assert!(!resolved.speculative.explicit);
}

#[test]
fn speculative_draft_translates_for_direct_embedded_openai() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = "/models/qwen3-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail_open"
draft_max_tokens = 8
draft_min_tokens = 2
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
        compact_meta: None,
    })
    .expect("draft speculative config should resolve");

    resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("direct model load options should allow draft speculation");
    let openai = resolved
        .to_embedded_openai_args(0, false)
        .expect("direct embedded OpenAI args should allow draft");
    assert_eq!(openai.speculative_window, 8);
    assert_eq!(
        openai.draft_model_path.as_deref(),
        Some(Path::new("/models/qwen3-draft.gguf"))
    );
    assert_eq!(openai.draft_n_gpu_layers, None);
}

#[test]
fn benchmark_shaped_model_entry_draft_translates_for_direct_embedded_openai() {
    let mesh_config = parse_config(
        r#"
[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.hardware]
model_path = "/models/qwen3.gguf"

[models.speculative]
strategy = "disabled"
mode = "draft"
draft_model_path = "/models/qwen3-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 4
draft_min_tokens = 0
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
        compact_meta: None,
    })
    .expect("benchmark-shaped draft speculative config should resolve");

    assert_eq!(resolved.speculative.mode, "draft");
    assert_eq!(resolved.speculative.pairing_fault, "fail_closed");
    let openai = resolved
        .to_embedded_openai_args(0, false)
        .expect("direct embedded OpenAI args should allow benchmark draft");
    assert_eq!(openai.speculative_window, 4);
    assert_eq!(
        openai.draft_model_path.as_deref(),
        Some(Path::new("/models/qwen3-draft.gguf"))
    );
}

#[test]
fn benchmark_shaped_hf_identity_row_uses_canonical_selector_for_served_alias() {
    let model_file = temp_model_file();
    let toml = format!(
        r#"
[[models]]
model = "Qwen/Qwen3-GGUF@sha/qwen3-q4_k_m.gguf"

[models.hardware]
model_path = "{}"

[models.speculative]
strategy = "disabled"
mode = "draft"
draft_model_path = "/models/qwen3-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 4
"#,
        model_file.path().display()
    );
    let mesh_config = parse_config(&toml);
    let config_model_id = mesh_config
        .models
        .first()
        .expect("benchmark model config should be present")
        .model
        .as_str();

    // Runtime startup keeps the served alias in `model_id` and passes the
    // exact configured entry selector separately. The resolver must use that
    // canonical selector rather than guessing from the served identity/path.
    let resolved = resolve_skippy_config_for_selector(
        SkippyConfigResolveRequest {
            mesh_config: &mesh_config,
            model_id: "Qwen/Qwen3-GGUF:Q4_K_M",
            model_path: model_file.path(),
            model_bytes: 4 * 1024 * 1024 * 1024,
            allocatable_memory_bytes: None,
            request_defaults: None,
            package_generation: None,
            compact_meta: None,
        },
        Some(config_model_id),
    )
    .expect("benchmark HF identity row should match by canonical selector");

    let openai = resolved
        .to_embedded_openai_args(0, false)
        .expect("direct embedded OpenAI args should include draft");
    assert_eq!(openai.speculative_window, 4);
    assert_eq!(
        openai.draft_model_path.as_deref(),
        Some(Path::new("/models/qwen3-draft.gguf"))
    );
}

#[test]
fn staged_only_controls_fail_closed_for_single_stage_loads() {
    let mesh_config = parse_config(
        r#"
[defaults.skippy]
prefill_chunk_size = 128
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
        compact_meta: None,
    })
    .expect("config should resolve");

    let err = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .unwrap_err()
        .to_string();
    assert!(err.contains("prefill chunk controls require staged serving"));
}

#[test]
fn incompatible_draft_pairing_warn_disable_turns_speculation_off() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = "/models/llama-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "warn_disable"
draft_max_tokens = 8
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
        compact_meta: None,
    })
    .expect("warn_disable should resolve");

    assert_eq!(resolved.speculative.mode, "disabled");
    assert!(resolved.speculative.draft_model_path.is_none());
}

#[test]
fn incompatible_draft_pairing_fail_closed_rejects_before_launch() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = "/models/llama-draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 8
"#,
    );
    let model_file = temp_model_file();

    let err = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("incompatible speculative draft pairing"));
}

#[test]
fn manual_stage_layer_range_is_staged_only_and_reaches_stage_config() {
    let mesh_config = parse_config(
        r#"
[defaults.hardware]
stage_layer_start = 12
stage_layer_end = 24
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
        compact_meta: None,
    })
    .expect("config should resolve");

    let err = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .unwrap_err()
        .to_string();
    assert!(err.contains("staged-only controls"));

    let stage_config = resolved
        .to_stage_config(Some(fake_package_identity(32)), LoadMode::RuntimeSlice)
        .expect("stage config should preserve explicit layer range");
    assert_eq!((stage_config.layer_start, stage_config.layer_end), (12, 24));
}

#[test]
fn benchmark_speculative_thresholds_are_now_accepted() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
draft_acceptance_threshold = 0.5
draft_split_probability = 0.3
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
        compact_meta: None,
    })
    .expect("draft_acceptance_threshold and draft_split_probability should be accepted");

    // They are schema-level only for now; the resolver accepts and stores them
    // on the resolved config, which the benchmark tune path writes into trial configs.
    assert_eq!(resolved.speculative.mode, "disabled");
}

#[test]
fn integrated_full_surface_fixture_resolves_defaults_overrides_staged_and_runtime_paths() {
    let fixture = full_surface_fixture_with_model_paths();
    let request_defaults = RequestDefaultsConfig {
        temperature: Some(0.7),
        max_tokens: Some(256),
        ..Default::default()
    };

    let explicit = resolve_explicit_full_surface_config(&fixture, &request_defaults);
    assert_explicit_full_surface_resolution(&explicit);
    assert_explicit_full_surface_stage_config(&explicit);
    assert_explicit_full_surface_runtime_options(&explicit);
    assert_explicit_full_surface_openai_args(&explicit);

    let omitted = resolve_defaults_full_surface_config(&fixture);
    assert_defaults_full_surface_resolution(&omitted);
}

#[test]
fn integrated_invalid_fixture_accepts_request_defaults_and_rejects_single_stage_staged_knobs() {
    let repaired_batch = FULL_SURFACE_INVALID_FIXTURE.replace("batch = 0", "batch = 64");
    let repaired_device = format!(
        "{repaired_batch}\ndevice = \"CUDA0\"\n",
        repaired_batch = repaired_batch.trim_end()
    );

    let config = parse_config(&repaired_device);
    let model_file = temp_model_file();
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 2 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("request defaults should resolve before staged-only translation gating");
    assert_eq!(
        resolved.request_defaults.chat_template.as_deref(),
        Some("unsafe-template")
    );
    let staged_only_error = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .unwrap_err()
        .to_string();
    assert!(staged_only_error.contains("prefill chunk controls require staged serving"));
}

#[test]
fn explicit_ngram_suffix_strategy_resolves_a_suffix_proposer() {
    use crate::plugin::SpeculativeConfig;
    let model_config: SpeculativeConfig = toml::from_str(
        r#"
strategy = "ngram-suffix"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
verify_window_min_tokens = 1
verify_window_max_tokens = 32
verify_window_pipeline_depth = 2
"#,
    )
    .expect("parse speculative config");
    let resolved = super::speculative::resolve_speculative_config(
        Some(&model_config),
        None,
        "meshllm/test-model",
        std::path::Path::new("/nonexistent/test-model.gguf"),
        None,
    )
    .expect("explicit ngram-suffix must resolve");
    let decode = resolved.decode;
    assert_eq!(
        decode.effective_strategy, "ngram-suffix",
        "effective strategy"
    );
    let ngram = decode.ngram.expect("suffix proposer must be present");
    assert_eq!(ngram.kind, skippy_server::NgramProposerKind::Suffix);
    assert_eq!(ngram.min_ngram, 5);
    assert_eq!(ngram.max_ngram, 32);
    assert_eq!(ngram.max_proposal_tokens, 48);
    assert_eq!(decode.verify_window.pipeline_depth, 2);
    assert_eq!(
        resolved.mode, "ngram",
        "resolved mode drives the embedded frontend"
    );
}

#[test]
fn explicit_ngram_suffix_survives_staged_openai_translation() {
    use crate::plugin::SpeculativeConfig;
    let model_config: SpeculativeConfig = toml::from_str(
        r#"
strategy = "ngram-suffix"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
verify_window_min_tokens = 1
verify_window_max_tokens = 32
verify_window_pipeline_depth = 2
"#,
    )
    .expect("parse speculative config");
    let mesh_config = crate::plugin::MeshConfig {
        defaults: Some(crate::plugin::ModelConfigDefaults {
            speculative: Some(model_config),
            ..Default::default()
        }),
        ..Default::default()
    };
    let resolved = super::resolve_skippy_config(super::SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/Qwen3-8B-Q4_K_M-layers",
        model_path: std::path::Path::new("/nonexistent/model.gguf"),
        model_bytes: 8_000_000_000,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("resolve skippy config");
    let ngram = resolved
        .speculative
        .decode
        .ngram
        .as_ref()
        .expect("resolved decode plan must keep the suffix proposer");
    assert_eq!(ngram.kind, skippy_server::NgramProposerKind::Suffix);
    assert_eq!(ngram.min_ngram, 5);
    assert_eq!(ngram.max_proposal_tokens, 48);
    let args = resolved
        .to_embedded_openai_args(4096, true)
        .expect("staged translation");
    let translated = args
        .speculative
        .ngram
        .as_ref()
        .expect("staged embedded args must keep the suffix proposer");
    assert_eq!(translated.kind, skippy_server::NgramProposerKind::Suffix);
    assert_eq!(translated.min_ngram, 5);
    assert_eq!(translated.max_ngram, 32);
    assert_eq!(translated.max_proposal_tokens, 48);
    assert_eq!(args.speculative.verify_window.pipeline_depth, 2);
}

#[test]
fn speculative_runtime_controls_reach_embedded_openai_translation() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
mode = "draft"
draft_model = "/tmp/default-draft.gguf"
draft_selection_policy = "manual"
draft_acceptance_threshold = 0.2
draft_split_probability = 0.3
draft_device = "CPU"
draft_threads = 2
draft_cache_type_k = "f16"
draft_cache_type_v = "f16"

[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.speculative]
draft_model = "/tmp/model-draft.gguf"
draft_acceptance_threshold = 0.7
draft_split_probability = 0.8
draft_device = "CUDA0"
draft_threads = 6
draft_cache_type_k = "q8_0"
draft_cache_type_v = "q4_0"
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
        compact_meta: None,
    })
    .expect("draft runtime controls must resolve");
    let args = resolved
        .to_embedded_openai_args(4096, true)
        .expect("draft runtime controls must translate");

    assert_eq!(
        args.draft_model_path.as_deref(),
        Some(Path::new("/tmp/model-draft.gguf"))
    );
    assert_eq!(args.speculative.draft_acceptance_threshold, 0.7);
    assert_eq!(args.speculative.draft_split_probability, 0.8);
    assert_eq!(args.speculative.draft_device.as_deref(), Some("CUDA0"));
    assert_eq!(args.speculative.draft_threads, Some(6));
    assert_eq!(args.speculative.draft_cache_type_k, "q8_0");
    assert_eq!(args.speculative.draft_cache_type_v, "q4_0");
}

#[test]
fn speculative_default_true_enables_automatic_defaults_without_failing_resolution() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
spec_default = true
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
        compact_meta: None,
    })
    .expect("spec_default = true must use automatic speculative defaults");

    assert_eq!(resolved.speculative.strategy, "auto");
}

#[test]
fn automatic_draft_selection_chooses_a_sibling_draft_gguf() {
    let directory = tempfile::tempdir().expect("create model directory");
    let target = directory.path().join("qwen3-target.gguf");
    let draft = directory.path().join("qwen3-draft.gguf");
    std::fs::write(&target, []).expect("write target fixture");
    std::fs::write(&draft, []).expect("write draft fixture");
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
draft_selection_policy = "auto"
pairing_fault = "fail_open"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "qwen3-target",
        model_path: &target,
        model_bytes: 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("auto policy should select sibling draft");

    assert_eq!(
        resolved.speculative.draft_model_path.as_deref(),
        Some(draft.as_path())
    );
    assert_eq!(resolved.speculative.mode, "draft");
    assert_eq!(resolved.speculative.draft_max_tokens, 3);
}

#[test]
fn automatic_draft_selection_preserves_explicit_draft_max_tokens() {
    let directory = tempfile::tempdir().expect("create model directory");
    let target = directory.path().join("qwen3-target.gguf");
    let draft = directory.path().join("qwen3-draft.gguf");
    std::fs::write(&target, []).expect("write target fixture");
    std::fs::write(&draft, []).expect("write draft fixture");
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
draft_selection_policy = "auto"
pairing_fault = "fail_open"
draft_max_tokens = 7
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "qwen3-target",
        model_path: &target,
        model_bytes: 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("auto policy should select sibling draft");

    assert_eq!(
        resolved.speculative.draft_model_path.as_deref(),
        Some(draft.as_path())
    );
    assert_eq!(resolved.speculative.draft_max_tokens, 7);
}

#[test]
fn speculative_default_false_suppresses_automatic_sibling_draft_selection() {
    let directory = tempfile::tempdir().expect("create model directory");
    let target = directory.path().join("qwen3-target.gguf");
    let draft = directory.path().join("qwen3-draft.gguf");
    std::fs::write(&target, []).expect("write target fixture");
    std::fs::write(&draft, []).expect("write draft fixture");
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
spec_default = false
draft_selection_policy = "auto"
"#,
    );

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "qwen3-target",
        model_path: &target,
        model_bytes: 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("disabled automatic defaults should resolve");

    assert_eq!(resolved.speculative.mode, "disabled");
    assert_eq!(resolved.speculative.draft_model_path, None);
}

#[test]
fn model_hf_draft_source_overrides_the_global_pair_as_a_unit() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
draft_hf_repo = "global/draft"
draft_hf_file = "global.gguf"
draft_max_tokens = 4
pairing_fault = "fail_open"

[[models]]
model = "mesh/test-target"

[models.speculative]
draft_hf_repo = "model/draft"
draft_hf_file = "model.gguf"
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "mesh/test-target",
        model_path: model_file.path(),
        model_bytes: 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("model HF pair should resolve");

    let draft_path = resolved
        .speculative
        .draft_model_path
        .expect("model draft source");
    assert!(
        draft_path
            .to_string_lossy()
            .contains("model/draft:model.gguf")
    );
    assert!(!draft_path.to_string_lossy().contains("global"));
}

#[test]
fn draft_hf_pair_becomes_the_runtime_draft_reference() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
mode = "draft"
draft_hf_repo = "mesh/test-draft"
draft_hf_file = "draft.gguf"
draft_selection_policy = "manual"
pairing_fault = "fail_open"
draft_max_tokens = 4
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "mesh/test-target",
        model_path: model_file.path(),
        model_bytes: 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("HF draft pair should resolve");

    assert!(
        resolved
            .speculative
            .draft_model_path
            .as_ref()
            .is_some_and(|path| path
                .to_string_lossy()
                .contains("mesh/test-draft:draft.gguf"))
    );
}
