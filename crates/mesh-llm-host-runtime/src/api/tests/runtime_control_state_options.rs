use crate::api::tests::runtime_control_state_builder::{
    RuntimeOptionSpec, runtime_option, runtime_source_setting,
};
use mesh_llm_config::{ConfigConditionValue, ConfigOptionsSource};

#[test]
fn runtime_config_control_state_builder_populates_local_model_choices() {
    let setting = runtime_source_setting(
        "plugin.demo.settings.model_ref",
        ConfigOptionsSource::RuntimeLocalModels,
    );
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources {
            local_models: crate::api::routes::runtime_control_state::RuntimeOptionsState::Options(
                vec![runtime_option(
                    RuntimeOptionSpec::enabled(
                        ConfigOptionsSource::RuntimeLocalModels,
                        "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
                        "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
                    )
                    .with_note("0.6 GiB"),
                )],
            ),
            ..Default::default()
        },
    );

    let options = payload
        .settings
        .get("plugin.demo.settings.model_ref")
        .and_then(|entry| entry.options.as_ref())
        .expect("local model options should exist");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].source, ConfigOptionsSource::RuntimeLocalModels);
}

#[test]
fn runtime_config_control_state_builder_populates_installed_plugin_choices() {
    let setting = runtime_source_setting(
        "plugin.demo.settings.projector_path",
        ConfigOptionsSource::RuntimeInstalledPlugins,
    );
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources {
            installed_plugins:
                crate::api::routes::runtime_control_state::RuntimeOptionsState::Options(vec![
                    runtime_option(
                        RuntimeOptionSpec::enabled(
                            ConfigOptionsSource::RuntimeInstalledPlugins,
                            "blobstore",
                            "blobstore",
                        )
                        .with_note("v1.0.0"),
                    ),
                    runtime_option(
                        RuntimeOptionSpec::enabled(
                            ConfigOptionsSource::RuntimeInstalledPlugins,
                            "flash-moe",
                            "flash-moe",
                        )
                        .with_note("v0.9.0")
                        .disabled_with_reason("Installed plugin is disabled."),
                    ),
                ]),
            ..Default::default()
        },
    );

    let options = payload
        .settings
        .get("plugin.demo.settings.projector_path")
        .and_then(|entry| entry.options.as_ref())
        .expect("installed plugin options should exist");
    assert_eq!(options.len(), 2);
    assert_eq!(
        options[0].source,
        ConfigOptionsSource::RuntimeInstalledPlugins
    );
    assert!(options[1].disabled);
    assert_eq!(
        options[1].reason.as_deref(),
        Some("Installed plugin is disabled.")
    );
}

#[test]
fn runtime_config_control_state_builder_supports_synthetic_mesh_peer_choices() {
    let setting = runtime_source_setting(
        "plugin.demo.settings.target_peer",
        ConfigOptionsSource::RuntimeMeshPeers,
    );
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources {
            mesh_peers: crate::api::routes::runtime_control_state::RuntimeOptionsState::Options(
                vec![runtime_option(RuntimeOptionSpec::enabled(
                    ConfigOptionsSource::RuntimeMeshPeers,
                    "peer-1234",
                    "node.local",
                ))],
            ),
            ..Default::default()
        },
    );

    let options = payload
        .settings
        .get("plugin.demo.settings.target_peer")
        .and_then(|entry| entry.options.as_ref())
        .expect("mesh peer options should exist");
    assert_eq!(options[0].source, ConfigOptionsSource::RuntimeMeshPeers);
    assert_eq!(options[0].label.as_deref(), Some("node.local"));
    assert_eq!(
        options[0].value,
        ConfigConditionValue::String("peer-1234".to_string())
    );
}
