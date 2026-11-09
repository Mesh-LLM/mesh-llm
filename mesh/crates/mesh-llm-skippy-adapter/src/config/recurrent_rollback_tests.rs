use super::test_support::{parse_config, temp_model_file};
use super::*;
use crate::SkippyTelemetryOptions;
use skippy_protocol::LoadMode;

#[test]
fn embedded_runtime_sizes_only_active_speculative_plans() {
    let config = parse_config("[defaults.speculative]\nmode = \"disabled\"\n");
    let model = temp_model_file();
    let mut resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "test-model",
        model_path: model.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();
    let depth = |resolved: &ResolvedSkippyConfig| {
        resolved
            .to_embedded_runtime_options(
                &SkippyTelemetryOptions::off(),
                None,
                LoadMode::RuntimeSlice,
            )
            .unwrap()
            .recurrent_rollback_tokens
    };
    assert_eq!(depth(&resolved), 0);
    resolved.speculative.native_mtp_enabled = true;
    resolved.speculative.draft_max_tokens = 6;
    assert_eq!(depth(&resolved), 7);
    resolved.speculative.native_mtp_enabled = false;
    resolved.speculative.draft_model_path = Some("draft.gguf".into());
    resolved.speculative.draft_max_tokens = 8;
    assert_eq!(depth(&resolved), 8);
    resolved.speculative.draft_model_path = None;
    assert_eq!(depth(&resolved), 0);
}
