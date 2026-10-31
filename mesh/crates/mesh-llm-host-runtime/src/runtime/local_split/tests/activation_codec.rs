use super::*;
use skippy_protocol::{StageActivationCodec as Codec, StageActivationCodecPolicy as Policy};
use std::path::Path;

fn codec_config(
    model_path: &Path,
    defaults: Option<&str>,
    model: Option<&str>,
) -> plugin::MeshConfig {
    let mut text = String::new();
    if let Some(codec) = defaults {
        text.push_str(&format!(
            "[defaults.skippy]\nactivation_codec = \"{codec}\"\n\n"
        ));
    }
    text.push_str(&format!(
        "[[models]]\nmodel = \"codec-model\"\n\n[models.hardware]\nmodel_path = {}\n",
        toml::Value::String(model_path.to_string_lossy().into_owned())
    ));
    if let Some(codec) = model {
        text.push_str(&format!(
            "\n[models.skippy]\nactivation_codec = \"{codec}\"\n"
        ));
    }
    toml::from_str(&text).expect("activation codec config")
}

#[test]
fn activation_codec_setting_round_trips_through_config_toml() {
    let config = codec_config(Path::new("/models/m.gguf"), Some("bf16"), Some("s8-row"));
    let text = toml::to_string(&config).expect("serialize config");
    let reparsed: plugin::MeshConfig = toml::from_str(&text).expect("reparse config");

    assert_eq!(reparsed.models[0].skippy, config.models[0].skippy);
    assert_eq!(
        reparsed
            .defaults
            .as_ref()
            .unwrap()
            .skippy
            .as_ref()
            .unwrap()
            .activation_codec
            .as_deref(),
        Some("bf16")
    );
    assert_eq!(
        reparsed.models[0]
            .skippy
            .as_ref()
            .unwrap()
            .activation_codec
            .as_deref(),
        Some("s8-row")
    );
}

#[test]
fn split_activation_codec_resolves_model_then_defaults_then_f16() {
    let path = Path::new("/models/m.gguf");
    let resolve = |config: &plugin::MeshConfig| {
        loading::configured_split_activation_codec(config, Some("codec-model"))
    };
    let fixed = |codec| SplitActivationCodec {
        codec,
        policy: Policy::Fixed,
    };

    // No setting anywhere: f16.
    assert_eq!(
        resolve(&codec_config(path, None, None)),
        fixed(Codec::F16RneV1)
    );
    assert_eq!(
        resolve(&codec_config(path, None, None)),
        SplitActivationCodec::DEFAULT
    );
    assert_eq!(
        resolve(&codec_config(path, Some("f16"), None)),
        fixed(Codec::F16RneV1)
    );
    assert_eq!(
        resolve(&codec_config(path, Some("f16"), Some("s8-row"))),
        fixed(Codec::S8RowF32RneV1)
    );
    assert_eq!(
        resolve(&codec_config(path, None, Some("bf16"))),
        fixed(Codec::Bf16RneV1)
    );
    assert_eq!(
        resolve(&codec_config(path, Some("f32"), None)),
        fixed(Codec::RawF32V1)
    );
    assert_eq!(
        resolve(&codec_config(path, Some("lossless"), None)),
        SplitActivationCodec::LOSSLESS
    );
    // Another model's override does not leak.
    assert_eq!(
        loading::configured_split_activation_codec(
            &codec_config(path, None, Some("s8-row")),
            Some("other-model")
        ),
        SplitActivationCodec::DEFAULT
    );
    assert!(SplitActivationCodec::from_setting("int4").is_none());
    // Validation accepts any case, so parsing does too.
    assert_eq!(
        SplitActivationCodec::from_setting("LOSSLESS"),
        Some(SplitActivationCodec::LOSSLESS)
    );
    assert_eq!(
        SplitActivationCodec::from_setting("F16"),
        Some(fixed(Codec::F16RneV1))
    );
    for (setting, codec) in [
        ("fp8-e4m3-row", Codec::Fp8E4m3RowF32RneV1),
        ("fp8-e4m3-block32", Codec::Fp8E4m3Block32E8m0RneV1),
        ("fp8-e5m2-row", Codec::Fp8E5m2RowF32RneV1),
    ] {
        assert_eq!(
            resolve(&codec_config(path, Some(setting), None)),
            fixed(codec)
        );
    }
}

fn codec_generation(
    local_id: iroh::EndpointId,
    remote_id: iroh::EndpointId,
    choice: SplitActivationCodec,
) -> SplitTopologyGeneration {
    let stages = vec![
        local_stage(local_id, 0, 0, 20),
        local_stage(remote_id, 1, 20, 40),
    ];
    let admissions = stages
        .iter()
        .map(|stage| skippy::test_stage_admission(stage.layer_start, stage.layer_end))
        .collect();
    SplitTopologyGeneration::new(
        "codec-topology".into(),
        "codec-run".into(),
        1,
        vec![
            SplitParticipant::new(local_id, 24_000_000_000, None),
            SplitParticipant::new(remote_id, 24_000_000_000, None),
        ],
        stages,
    )
    .with_activation_codec(choice)
    .with_admissions(admissions)
    .unwrap()
}

#[test]
fn an_fp8_choice_reaches_the_generation_unchanged() {
    let fp8 = SplitActivationCodec::from_setting("fp8-e4m3-block32").unwrap();
    let local = make_id(0x61);
    let generation = SplitTopologyGeneration::new(
        "codec-topology".into(),
        "codec-run".into(),
        1,
        vec![SplitParticipant::new(local, 24_000_000_000, None)],
        vec![local_stage(local, 0, 0, 20), local_stage(local, 1, 20, 40)],
    )
    .with_activation_codec(fp8);
    assert_eq!(
        (
            generation.activation_codec,
            generation.activation_codec_policy
        ),
        (Codec::Fp8E4m3Block32E8m0RneV1, Policy::Fixed)
    );
}

/// Every stage load of a two-stage split under `[defaults.skippy]` and model
/// `activation_codec` settings, with the topology hash of a lossless plan.
async fn split_stage_loads(
    defaults: Option<&str>,
    model: Option<&str>,
) -> ([skippy::StageLoadRequest; 2], String) {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("codec-model.gguf");
    write_fake_gguf_model(&model_path);
    let config = codec_config(&model_path, defaults, model);
    let package = package(40);
    let compact_meta =
        crate::models::gguf::scan_gguf_compact_meta(&model_path).expect("synthetic GGUF metadata");
    let choice = loading::configured_split_activation_codec(&config, Some("codec-model"));
    let generation = codec_generation(node.id(), make_id(0x71), choice);
    let lossless = codec_generation(node.id(), make_id(0x71), SplitActivationCodec::LOSSLESS);
    let spec = SplitGenerationLoadSpec {
        node: &node,
        mesh_config: &config,
        model_ref: "codec-model",
        config_model_id: Some("codec-model"),
        runtime_profile: "codec-profile",
        model_path: &model_path,
        package: &package,
        generation: &generation,
        projector_path: None,
        ctx_size: 4096,
        compact_meta: &compact_meta,
        split_certification: skippy::SplitCertificationAdmission::Certified,
        capacity_budget_bytes: None,
        pinned_gpu: None,
        device_override: None,
        slots: 1,
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            skippy_inference_api::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
        serving_hooks_factory: None,
        local_source_required: false,
    };
    let settings = loading::split_generation_load_settings(&spec)
        .await
        .expect("split settings");
    let downstream_stage = &generation.stages[1];
    let downstream = skippy::StagePeerDescriptor {
        stage_id: downstream_stage.stage_id.clone(),
        stage_index: downstream_stage.stage_index,
        endpoint: "127.0.0.1:41001".to_string(),
        node_id: Some(downstream_stage.node_id),
        persistent_lane_marker: false,
    };
    let loads = [
        split_runtime_stage_load_request(
            &spec,
            &settings,
            &generation.stages[0],
            Some(downstream),
            "127.0.0.1:41000",
            &std::collections::HashMap::new(),
        )
        .unwrap(),
        split_runtime_stage_load_request(
            &spec,
            &settings,
            downstream_stage,
            None,
            "127.0.0.1:41000",
            &std::collections::HashMap::new(),
        )
        .unwrap(),
    ];

    let lossless_hash = split_topology_hash(
        &lossless.stages,
        &lossless.admissions,
        lossless.activation_codec,
        lossless.activation_codec_policy,
    );
    (loads, lossless_hash)
}

#[tokio::test]
async fn configured_activation_codec_reaches_every_stage_load() {
    let (loads, lossless_hash) = split_stage_loads(Some("f16"), Some("s8-row")).await;

    for load in &loads {
        assert_eq!(load.activation_codec, Codec::S8RowF32RneV1);
        assert_eq!(load.activation_codec_policy, Policy::Fixed);
        // Stages built for the lossless plan would not attest to this one.
        assert_ne!(load.topology_hash, lossless_hash);
    }
    assert_eq!(loads[0].topology_hash, loads[1].topology_hash);
}

#[tokio::test]
async fn a_split_load_with_no_codec_setting_sends_f16() {
    let (loads, lossless_hash) = split_stage_loads(None, None).await;

    for load in &loads {
        assert_eq!(load.activation_codec, Codec::F16RneV1);
        assert_eq!(load.activation_codec_policy, Policy::Fixed);
        assert_ne!(load.topology_hash, lossless_hash);
    }
    assert_eq!(loads[0].topology_hash, loads[1].topology_hash);
}
