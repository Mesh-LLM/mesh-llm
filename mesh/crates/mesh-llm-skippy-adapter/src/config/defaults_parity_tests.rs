//! Prevent Mesh and standalone Skippy from drifting when a serving default changes.
use super::{test_support::*, *};
use mesh_llm_config::MeshConfig;
use skippy_api::{SingleStageOptions, serving::InferenceOptions};
use skippy_protocol::LoadMode;

#[test]
fn mesh_and_skippy_resolve_identical_default_stages_and_frontends() {
    for native_mtp in [false, true] {
        let model = if native_mtp {
            temp_model_file_with_tensor_names(&["blk.23.nextn.eh_proj.weight"], None)
        } else {
            temp_model_file()
        };
        let config = MeshConfig::default();
        let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
            mesh_config: &config,
            model_id: "model",
            model_path: model.path(),
            model_bytes: 1234,
            allocatable_memory_bytes: None,
            request_defaults: None,
            package_generation: None,
            compact_meta: None,
        })
        .unwrap();
        let mesh_frontend = resolved.to_embedded_openai_args(0, true).unwrap();
        let mut skippy_frontend = InferenceOptions::direct_single_stage_defaults(
            "model".into(),
            skippy_config::local_serving::MAX_OUTPUT_TOKENS,
            skippy_config::local_serving::PARALLEL,
            native_mtp,
        );
        skippy_api::speculative::apply_auto_speculation(&mut skippy_frontend, model.path());
        assert_eq!(mesh_frontend, skippy_frontend);

        let mut options = SingleStageOptions::new("model", model.path());
        options.native_mtp_enabled = native_mtp;
        let identity = fake_package_identity(24);
        let mesh_stage = resolved
            .to_stage_config(Some(identity.clone()), LoadMode::RuntimeSlice)
            .unwrap();
        let skippy_stage =
            skippy_api::single_stage_config(&options, identity.into(), mesh_stage.run_id.clone())
                .unwrap();
        assert_eq!(
            serde_json::to_value(mesh_stage).unwrap(),
            serde_json::to_value(skippy_stage).unwrap()
        );
    }
}

#[test]
fn partial_mesh_sampling_extensions_inherit_runtime_defaults() {
    let model = temp_model_file();
    let config = parse_config(
        "[defaults.request_defaults.dry]\nmultiplier = 0.5\n[defaults.request_defaults.xtc]\nprobability = 0.2\n",
    );
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "model",
        model_path: model.path(),
        model_bytes: 1234,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap();
    let frontend = resolved.to_embedded_openai_args(0, true).unwrap();
    let mut defaults = skippy_runtime::SamplingConfig::default();
    defaults.dry.multiplier = 0.5;
    defaults.xtc.probability = 0.2;
    assert_eq!(frontend.request_defaults.dry, Some(defaults.dry));
    assert_eq!(frontend.request_defaults.xtc, Some(defaults.xtc));
}

#[test]
fn automatic_draft_discovery_and_pairing_match_mesh_policy() {
    for native_mtp in [false, true] {
        for compatible in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let target = if native_mtp {
                temp_model_file_with_tensor_names(&["blk.23.nextn.eh_proj.weight"], None)
            } else {
                temp_model_file()
            };
            let draft =
                temp_model_file_with_architecture(if compatible { "llama" } else { "qwen2" });
            let target_path = directory.path().join("model.gguf");
            let draft_path = directory.path().join("draft.gguf");
            std::fs::copy(target.path(), &target_path).unwrap();
            std::fs::copy(draft.path(), &draft_path).unwrap();
            let config = MeshConfig::default();
            let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
                mesh_config: &config,
                model_id: "model",
                model_path: &target_path,
                model_bytes: 1234,
                allocatable_memory_bytes: None,
                request_defaults: None,
                package_generation: None,
                compact_meta: None,
            })
            .unwrap();
            let mesh = resolved.to_embedded_openai_args(0, true).unwrap();
            let mut standalone = InferenceOptions::direct_single_stage_defaults(
                "model".into(),
                skippy_config::local_serving::MAX_OUTPUT_TOKENS,
                skippy_config::local_serving::PARALLEL,
                native_mtp,
            );
            skippy_api::speculative::apply_auto_speculation(&mut standalone, &target_path);
            assert_eq!(
                mesh, standalone,
                "native={native_mtp}, compatible={compatible}"
            );
        }
    }
}
