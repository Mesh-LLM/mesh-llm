use super::test_support::*;
use super::*;
use crate::SkippyTelemetryOptions;
use mesh_llm_config::{MeshConfig, RequestDefaultsConfig};
use skippy_protocol::LoadMode;
use std::path::Path;
use tempfile::NamedTempFile;

const FULL_SURFACE_VALID_FIXTURE: &str =
    include_str!("../../tests/fixtures/skippy_full_surface_valid.toml");
const FULL_SURFACE_INVALID_FIXTURE: &str =
    include_str!("../../tests/fixtures/skippy_full_surface_invalid.toml");

struct FullSurfaceFixture {
    mesh_config: MeshConfig,
    explicit_model: NamedTempFile,
    defaults_model: NamedTempFile,
    _projector_file: NamedTempFile,
}

#[test]
fn toml_path_escapes_windows_path_separators() {
    let path = Path::new(r"C:\Users\runner\model.gguf");
    let parsed: toml::Value = toml::from_str(&format!("path = {}", toml_path(path)))
        .expect("serialized path should remain valid TOML");

    assert_eq!(parsed["path"].as_str(), Some(r"C:\Users\runner\model.gguf"));
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
    assert_eq!(explicit.request_defaults.max_tokens, Some(256));
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
    assert_eq!(omitted.request_defaults.max_tokens, Some(128));

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
    let model_file = temp_model_file();
    let draft_file = temp_model_file();
    let mesh_config = parse_config(&format!(
        r#"
[[models]]
model = "Qwen/Qwen3-0.6B:Q4_K_M"

[models.hardware]
model_path = {}

[models.speculative]
strategy = "disabled"
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 4
draft_min_tokens = 0
"#,
        toml_path(model_file.path()),
        toml_path(draft_file.path())
    ));

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
    assert_eq!(openai.draft_model_path.as_deref(), Some(draft_file.path()));
}

#[test]
fn benchmark_shaped_hf_identity_row_uses_canonical_selector_for_served_alias() {
    let model_file = temp_model_file();
    let draft_file = temp_model_file();
    let toml = format!(
        r#"
[[models]]
model = "Qwen/Qwen3-GGUF@sha/qwen3-q4_k_m.gguf"

[models.hardware]
model_path = {}

[models.speculative]
strategy = "disabled"
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 4
"#,
        toml_path(model_file.path()),
        toml_path(draft_file.path())
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
    assert_eq!(openai.draft_model_path.as_deref(), Some(draft_file.path()));
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
    let draft_file = temp_model_file_with_architecture("qwen2");
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "warn_disable"
draft_max_tokens = 8
"#,
        toml_path(draft_file.path())
    ));
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
    let draft_file = temp_model_file_with_architecture("qwen2");
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 8
"#,
        toml_path(draft_file.path())
    ));
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
fn missing_draft_architecture_warn_disable_turns_speculation_off() {
    let draft_file = NamedTempFile::new().expect("temp draft model");
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "warn_disable"
draft_max_tokens = 8
"#,
        toml_path(draft_file.path())
    ));
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
fn missing_target_architecture_fail_closed_rejects_before_launch() {
    let draft_file = temp_model_file();
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "fail_closed"
draft_max_tokens = 8
"#,
        toml_path(draft_file.path())
    ));
    let model_file = NamedTempFile::new().expect("temp target model");

    let err = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "unknown",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("target model architecture metadata is unavailable"));
}

#[test]
fn missing_architecture_fail_open_preserves_explicit_draft() {
    let draft_file = NamedTempFile::new().expect("temp draft model");
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
mode = "draft"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "fail_open"
draft_max_tokens = 8
"#,
        toml_path(draft_file.path())
    ));
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
    .expect("fail_open should preserve explicit draft mode");

    assert_eq!(resolved.speculative.mode, "draft");
    assert_eq!(
        resolved.speculative.draft_model_path.as_deref(),
        Some(draft_file.path())
    );
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
    use mesh_llm_config::SpeculativeConfig;
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
        None,
    )
    .expect("explicit ngram-suffix must resolve");
    let decode = resolved.decode;
    assert_eq!(
        decode.effective_strategy, "ngram-suffix",
        "effective strategy"
    );
    let ngram = decode.ngram.expect("suffix proposer must be present");
    assert_eq!(ngram.kind, skippy_serving::NgramProposerKind::Suffix);
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
    use mesh_llm_config::SpeculativeConfig;
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
    let mesh_config = mesh_llm_config::MeshConfig {
        defaults: Some(mesh_llm_config::ModelConfigDefaults {
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
    assert_eq!(ngram.kind, skippy_serving::NgramProposerKind::Suffix);
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
    assert_eq!(translated.kind, skippy_serving::NgramProposerKind::Suffix);
    assert_eq!(translated.min_ngram, 5);
    assert_eq!(translated.max_ngram, 32);
    assert_eq!(translated.max_proposal_tokens, 48);
    assert_eq!(args.speculative.verify_window.pipeline_depth, 2);
}

#[test]
fn speculative_runtime_controls_reach_embedded_openai_translation() {
    let draft_file = temp_model_file();
    let mesh_config = parse_config(&format!(
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
draft_model = {}
draft_acceptance_threshold = 0.7
draft_split_probability = 0.8
draft_device = "CUDA0"
draft_threads = 6
draft_cache_type_k = "q8_0"
draft_cache_type_v = "q4_0"
"#,
        toml_path(draft_file.path())
    ));
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

    assert_eq!(args.draft_model_path.as_deref(), Some(draft_file.path()));
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

fn resolve_with_dflash_draft(strategy: &str, pairing_fault: &str) -> ResolvedSkippyConfig {
    resolve_with_dflash_draft_and(strategy, pairing_fault, "")
}

fn resolve_with_dflash_draft_and(
    strategy: &str,
    pairing_fault: &str,
    extra: &str,
) -> ResolvedSkippyConfig {
    let draft_file = temp_model_file_with_architecture("dflash");
    let mesh_config = parse_config(&format!(
        r#"
[defaults.speculative]
strategy = "{strategy}"
draft_model_path = {}
draft_selection_policy = "manual"
pairing_fault = "{pairing_fault}"
draft_max_tokens = 8
draft_gpu_layers = 12
{extra}
"#,
        toml_path(draft_file.path())
    ));
    let model_file = temp_model_file_with_architecture("qwen3");
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-4B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect("a DFlash draft should resolve")
}

#[test]
fn dflash_draft_resolves_to_a_dflash_plan_without_a_draft_runner() {
    let resolved = resolve_with_dflash_draft("dflash", "warn_disable");
    let decode = &resolved.speculative.decode;
    assert_eq!(decode.effective_strategy, "dflash");
    assert_eq!(
        decode
            .dflash
            .as_ref()
            .and_then(|dflash| dflash.max_draft_tokens),
        Some(8)
    );
    assert!(!resolved.speculative.native_mtp_enabled);
    assert_eq!(resolved.speculative.mode, "disabled");
    assert!(resolved.speculative.draft_model_path.is_none());

    let args = resolved
        .to_embedded_openai_args(0, false)
        .expect("single-stage translation");
    assert!(args.draft_model_path.is_none());
    assert!(args.speculative.dflash.is_some());
    assert_eq!(args.draft_n_gpu_layers, Some(12));
}

#[test]
fn dflash_plan_keeps_the_configured_gate() {
    let resolved = resolve_with_dflash_draft_and(
        "dflash",
        "warn_disable",
        "gate = false\ngate_min_requests = 3\ngate_cooldown_s = 7",
    );
    let decode = &resolved.speculative.decode;

    assert!(decode.dflash.is_some());
    assert!(!decode.gate.enabled);
    assert_eq!(decode.gate.min_requests, 3);
    assert_eq!(decode.gate.cooldown_s, 7);
}

#[test]
fn auto_does_not_select_a_dflash_draft() {
    let resolved = resolve_with_dflash_draft("auto", "warn_disable");

    assert!(resolved.speculative.decode.dflash.is_none());
    assert_eq!(resolved.speculative.mode, "disabled");
    assert!(resolved.speculative.draft_model_path.is_none());
}

#[test]
fn split_targets_keep_an_explicit_dflash_plan() {
    for pairing_fault in ["warn_disable", "fail_closed"] {
        let resolved = resolve_with_dflash_draft("dflash", pairing_fault);
        let staged = resolved
            .to_embedded_openai_args(4096, true)
            .expect("staged translation keeps DFlash");
        let dflash = staged.speculative.dflash.expect("split DFlash plan");
        assert_eq!(dflash.max_draft_tokens, Some(8));
        assert_eq!(staged.speculative.effective_strategy, "dflash");
        assert!(!staged.native_mtp_enabled);
    }
}

/// A Qwen3.5-9B target alone in its directory, so no sibling draft is found.
fn default_dflash_target() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("target directory");
    let path = directory.path().join("Qwen3.5-9B-Q4_K_M.gguf");
    std::fs::copy(temp_model_file_with_architecture("qwen35").path(), &path).expect("copy target");
    (directory, path)
}

fn resolve_default_dflash(
    config: &str,
    model_path: &Path,
    meta: &skippy_model_artifact::gguf::GgufCompactMeta,
) -> ResolvedSkippyConfig {
    let mesh_config = parse_config(config);
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        model_path,
        model_bytes: 5 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: Some(meta),
    })
    .expect("a known DFlash target should resolve")
}

#[test]
fn a_known_target_defaults_to_its_dflash_draft() {
    let (_directory, model_path) = default_dflash_target();
    let resolved = resolve_default_dflash("", &model_path, &target_meta("qwen35", 32, 4096, 0));
    let speculative = &resolved.speculative;
    let pairing = speculative.default_dflash.expect("default pairing");
    let dflash = speculative.decode.dflash.as_ref().expect("DFlash plan");

    assert_eq!(pairing.draft.repo, "AtomicChat/Qwen3.5-9B-DFlash-GGUF");
    assert_eq!(dflash.draft_model_path, pairing.draft.cache_path());
    let cap = if cfg!(target_os = "macos") { 6 } else { 15 };
    assert_eq!(dflash.max_draft_tokens, Some(cap));
    assert_eq!(speculative.draft_max_tokens, cap as u32);
    assert_eq!(speculative.decode.requested_strategy, "auto");
    assert_eq!(speculative.decode.effective_strategy, "dflash");
    assert_eq!(speculative.mode, "disabled");
    assert!(!speculative.explicit);
}

#[test]
fn a_configured_draft_cap_overrides_the_backend_default() {
    let (_directory, model_path) = default_dflash_target();
    let resolved = resolve_default_dflash(
        "[defaults.speculative]\ndraft_max_tokens = 10",
        &model_path,
        &target_meta("qwen35", 32, 4096, 0),
    );

    let dflash = resolved.speculative.decode.dflash.expect("DFlash plan");
    assert_eq!(dflash.max_draft_tokens, Some(10));
}

#[test]
fn unknown_targets_do_not_default_to_dflash() {
    let (_directory, model_path) = default_dflash_target();
    let resolved = resolve_default_dflash("", &model_path, &target_meta("qwen35", 48, 5120, 0));

    assert!(resolved.speculative.default_dflash.is_none());
    assert!(resolved.speculative.decode.dflash.is_none());
}

#[test]
fn explicit_speculation_settings_win_over_the_default_dflash_draft() {
    let meta = target_meta("qwen35", 32, 4096, 0);
    for config in [
        "[defaults.speculative]\nstrategy = \"disabled\"",
        "[defaults.speculative]\nmode = \"disabled\"",
        "[defaults.speculative]\nspec_default = false",
        "[defaults.speculative]\nngram_min = 2\nngram_max = 4",
        "[defaults.speculative]\nstrategy = \"ngram-cache\"\nngram_min = 2\nngram_max = 4",
    ] {
        let (_directory, model_path) = default_dflash_target();
        let resolved = resolve_default_dflash(config, &model_path, &meta);
        assert!(resolved.speculative.default_dflash.is_none(), "{config}");
        assert!(resolved.speculative.decode.dflash.is_none(), "{config}");
    }

    // A draft next to the target, or one named in configuration, is an
    // ordinary draft and keeps the DFlash default out.
    let (directory, model_path) = default_dflash_target();
    let sibling = directory.path().join("Qwen3.5-0.8B-draft.gguf");
    std::fs::copy(&model_path, &sibling).expect("copy sibling draft");
    let resolved = resolve_default_dflash("", &model_path, &meta);
    assert!(resolved.speculative.decode.dflash.is_none());
    assert_eq!(resolved.speculative.draft_model_path, Some(sibling.clone()));

    let named = format!(
        "[defaults.speculative]\ndraft_model_path = {}\ndraft_selection_policy = \"manual\"",
        toml_path(&sibling)
    );
    let resolved = resolve_default_dflash(&named, &model_path, &meta);
    assert!(resolved.speculative.decode.dflash.is_none());
    assert_eq!(resolved.speculative.mode, "draft");
}

#[test]
fn dropping_the_default_keeps_an_explicit_dflash_draft() {
    let (_directory, model_path) = default_dflash_target();
    let draft = temp_model_file_with_architecture("dflash");
    let config = format!(
        "[defaults.speculative]\nstrategy = \"dflash\"\ndraft_model_path = {}\ndraft_max_tokens = 5",
        toml_path(draft.path())
    );
    let mut resolved =
        resolve_default_dflash(&config, &model_path, &target_meta("qwen35", 32, 4096, 0));
    assert!(resolved.speculative.default_dflash.is_none());

    // The host drops the default whenever its memory plan holds no default draft.
    resolved.speculative.drop_default_dflash();
    let args = resolved.to_embedded_openai_args(0, false).unwrap();
    let dflash = args.speculative.dflash.expect("explicit DFlash plan");
    assert_eq!(dflash.draft_model_path, draft.path());
    assert_eq!(dflash.max_draft_tokens, Some(5));
}

#[test]
fn a_default_dflash_draft_is_served_only_once_fetched() {
    let (_directory, model_path) = default_dflash_target();
    let mut resolved = resolve_default_dflash(
        "[defaults.speculative]\npairing_fault = \"fail_closed\"",
        &model_path,
        &target_meta("qwen35", 32, 4096, 0),
    );

    resolved
        .speculative
        .set_default_dflash_draft(Path::new("/nonexistent/draft-dflash.gguf"));
    let unfetched = resolved.to_embedded_openai_args(0, false).unwrap();
    assert!(unfetched.speculative.dflash.is_none());
    assert_eq!(unfetched.speculative.effective_strategy, "disabled");

    let draft = temp_model_file_with_architecture("dflash");
    resolved.speculative.set_default_dflash_draft(draft.path());
    let fetched = resolved.to_embedded_openai_args(0, false).unwrap();
    assert_eq!(
        fetched
            .speculative
            .dflash
            .map(|dflash| dflash.draft_model_path),
        Some(draft.path().to_path_buf())
    );
    // A split target keeps a fetched default draft too.
    let staged = resolved.to_embedded_openai_args(4096, true).unwrap();
    assert_eq!(
        staged
            .speculative
            .dflash
            .map(|dflash| dflash.draft_model_path),
        Some(draft.path().to_path_buf())
    );

    resolved.speculative.drop_default_dflash();
    assert!(resolved.speculative.default_dflash.is_none());
    let dropped = resolved.to_embedded_openai_args(0, false).unwrap();
    assert!(dropped.speculative.dflash.is_none());
    assert_eq!(dropped.speculative.effective_strategy, "disabled");
}

/// A target carrying a NextN layer, alone in its directory so no sibling
/// draft is found; native MTP is on for it by default.
fn native_mtp_target() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("target directory");
    let path = directory.path().join("Qwen3.8-27B-UD-Q4_K_M.gguf");
    let source = temp_model_file_with_architecture_and_tensor_names(
        "qwen35",
        &["blk.64.nextn.eh_proj.weight"],
        None,
    );
    std::fs::copy(source.path(), &path).expect("copy target");
    (directory, path)
}

fn qwen38_27b_meta() -> skippy_model_artifact::gguf::GgufCompactMeta {
    named(target_meta("qwen35", 64, 5120, 0), "Qwen3.8")
}

#[test]
fn a_known_pairing_replaces_automatic_native_mtp() {
    let (_directory, model_path) = native_mtp_target();
    let mut resolved = resolve_default_dflash("", &model_path, &qwen38_27b_meta());
    let speculative = &resolved.speculative;
    let pairing = speculative.default_dflash.expect("default pairing");
    let dflash = speculative.decode.dflash.as_ref().expect("DFlash plan");

    assert_eq!(pairing.draft.repo, "z-lab/Qwen3.8-27B-DFlash2-GGUF");
    assert!(!speculative.native_mtp_enabled);
    assert!(!speculative.decode.native_mtp.enabled);
    assert_eq!(speculative.decode.effective_strategy, "dflash");
    // The pairing's block caps the backend default, and rollback follows it.
    let cap = if cfg!(target_os = "macos") { 6 } else { 7 };
    assert_eq!(dflash.max_draft_tokens, Some(cap));
    assert_eq!(dflash.recurrent_rollback_tokens(), cap as u32 + 1);
    let draft = temp_model_file_with_architecture("dflash");
    resolved.speculative.set_default_dflash_draft(draft.path());
    let args = resolved.to_embedded_openai_args(0, false).unwrap();
    assert!(!args.native_mtp_enabled);
    assert!(args.speculative.dflash.is_some());

    // Without the draft (not fetched, no room, or a split target), the load
    // keeps the native MTP it would have served.
    resolved.speculative.drop_default_dflash();
    let speculative = &resolved.speculative;
    assert!(speculative.default_dflash.is_none());
    assert!(speculative.decode.dflash.is_none());
    assert!(speculative.native_mtp_enabled);
    assert_eq!(speculative.decode.effective_strategy, "native-mtp");
    let staged = resolved.to_embedded_openai_args(4096, true).unwrap();
    assert!(staged.native_mtp_enabled);
}

#[test]
fn native_mtp_targets_without_a_pairing_keep_native_mtp() {
    let (_directory, model_path) = native_mtp_target();
    let resolved = resolve_default_dflash("", &model_path, &target_meta("qwen35", 48, 5120, 0));

    assert!(resolved.speculative.default_dflash.is_none());
    assert!(resolved.speculative.native_mtp_fallback.is_none());
    assert!(resolved.speculative.native_mtp_enabled);
    assert_eq!(resolved.speculative.decode.effective_strategy, "native-mtp");
}

#[test]
fn configured_speculation_keeps_its_plan_on_a_native_mtp_target_with_a_pairing() {
    let unpaired = target_meta("qwen35", 48, 5120, 0);
    for config in [
        "[defaults.speculative]\nstrategy = \"mtp\"",
        "[defaults.speculative]\nnative_mtp_reject_cooldown_tokens = 4",
        "[defaults.speculative]\nstrategy = \"disabled\"",
        "[defaults.speculative]\nmode = \"disabled\"",
        "[defaults.speculative]\nspec_default = false",
        "[defaults.speculative]\nngram_min = 2\nngram_max = 4",
        "[defaults.speculative]\nstrategy = \"ngram-cache\"\nngram_min = 2\nngram_max = 4",
    ] {
        let (_directory, model_path) = native_mtp_target();
        let paired = resolve_default_dflash(config, &model_path, &qwen38_27b_meta());
        let without_pairing = resolve_default_dflash(config, &model_path, &unpaired);
        assert!(paired.speculative.default_dflash.is_none(), "{config}");
        assert_eq!(paired.speculative, without_pairing.speculative, "{config}");
    }
    let (_directory, model_path) = native_mtp_target();
    let explicit_mtp = resolve_default_dflash(
        "[defaults.speculative]\nstrategy = \"mtp\"",
        &model_path,
        &qwen38_27b_meta(),
    );
    assert!(explicit_mtp.speculative.native_mtp_enabled);
}
