use super::*;
use mesh_llm_config::{
    ConfigConditionValue, ConfigControlAvailabilitySource, ConfigDisabledWritePolicy,
    ConfigOptionsSource,
};
use serial_test::serial;
use std::collections::BTreeMap;

struct RuntimeControlStateTestOverrideGuard;

impl RuntimeControlStateTestOverrideGuard {
    fn install(
        sources: crate::api::routes::runtime_control_state::RuntimeControlStateSources,
    ) -> Self {
        crate::api::routes::runtime_control_state::set_test_runtime_control_state_sources(Some(
            sources,
        ));
        Self
    }
}

impl Drop for RuntimeControlStateTestOverrideGuard {
    fn drop(&mut self) {
        crate::api::routes::runtime_control_state::set_test_runtime_control_state_sources(None);
    }
}

#[tokio::test]
#[serial]
async fn runtime_config_control_state_api_returns_empty_overlay_by_default() {
    let _override_guard = RuntimeControlStateTestOverrideGuard::install(Default::default());
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/runtime/config-control-state HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    let body = json_body(&response);

    assert!(response.starts_with("HTTP/1.1 200"), "response: {response}");
    assert_eq!(body, serde_json::json!({ "settings": {} }));

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn runtime_config_control_state_serializes_runtime_options_without_null_noise() {
    let mut settings = BTreeMap::new();
    settings.insert(
        "defaults.hardware.device".to_string(),
        crate::api::routes::runtime::ConfigControlStateEntry {
            enabled: true,
            reason: None,
            note: Some("Loopback inventory populated runtime GPU choices".to_string()),
            source: ConfigControlAvailabilitySource::Runtime,
            write_policy: ConfigDisabledWritePolicy::PreserveExisting,
            options: Some(vec![
                crate::api::routes::runtime::ConfigControlOption {
                    value: ConfigConditionValue::String("cuda:0".to_string()),
                    label: Some("NVIDIA GPU 0".to_string()),
                    note: Some("24 GiB VRAM".to_string()),
                    disabled: false,
                    reason: None,
                    source: ConfigOptionsSource::RuntimeGpus,
                },
                crate::api::routes::runtime::ConfigControlOption {
                    value: ConfigConditionValue::String("metal:0".to_string()),
                    label: Some("Metal GPU 0".to_string()),
                    note: None,
                    disabled: true,
                    reason: Some("Backend unavailable for current runtime".to_string()),
                    source: ConfigOptionsSource::RuntimeNativeBackends,
                },
            ]),
        },
    );
    let payload = crate::api::routes::runtime::ConfigControlStatePayload { settings };

    let value = serde_json::to_value(payload).expect("payload should serialize");

    assert_eq!(
        value.pointer("/settings/defaults.hardware.device/source"),
        Some(&serde_json::json!("runtime"))
    );
    assert_eq!(
        value.pointer("/settings/defaults.hardware.device/write_policy"),
        Some(&serde_json::json!("preserve_existing"))
    );
    assert_eq!(
        value.pointer("/settings/defaults.hardware.device/options/0/source"),
        Some(&serde_json::json!("runtime_gpus"))
    );
    assert_eq!(
        value.pointer("/settings/defaults.hardware.device/options/0/value"),
        Some(&serde_json::json!({ "kind": "string", "value": "cuda:0" }))
    );
    assert_eq!(
        value.pointer("/settings/defaults.hardware.device/options/1/source"),
        Some(&serde_json::json!("runtime_native_backends"))
    );
    assert_eq!(
        value.pointer("/settings/defaults.hardware.device/options/1/reason"),
        Some(&serde_json::json!(
            "Backend unavailable for current runtime"
        ))
    );
    assert!(
        value
            .pointer("/settings/defaults.hardware.device/reason")
            .is_none(),
        "optional reason should be omitted when absent"
    );
    assert!(
        value
            .pointer("/settings/defaults.hardware.device/options/0/reason")
            .is_none(),
        "option reason should be omitted when absent"
    );
}

#[tokio::test]
async fn runtime_config_control_state_non_loopback_calls_are_forbidden() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    });

    let (mut server_stream, _) = listener.accept().await.unwrap();
    let allowed = crate::api::routes::runtime::ensure_loopback_control_caller_for_peer_addr(
        &mut server_stream,
        Ok(std::net::SocketAddr::from(([192, 0, 2, 10], 40123))),
    )
    .await
    .unwrap();
    assert!(!allowed);
    drop(server_stream);

    let response = client.await.unwrap();
    let body = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 403"), "response: {response}");
    assert_eq!(
        body,
        serde_json::json!({
            "error": "runtime control endpoints only accept localhost connections"
        })
    );
}

#[test]
fn runtime_config_control_state_builder_omits_unknown_runtime_sources() {
    let setting = super::runtime_control_state_builder::runtime_source_setting(
        "defaults.hardware.device",
        ConfigOptionsSource::RuntimeGpus,
    );
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources::default(),
    );
    assert!(payload.settings.is_empty());
}

#[test]
fn runtime_config_control_state_builder_uses_disabled_or_omitted_policy_for_missing_sources() {
    let backend_setting = super::runtime_control_state_builder::runtime_source_setting(
        "plugin.demo.settings.runtime_kind",
        ConfigOptionsSource::RuntimeNativeBackends,
    );
    let plugin_setting = super::runtime_control_state_builder::runtime_source_setting(
        "plugin.demo.settings.plugin_name",
        ConfigOptionsSource::RuntimeInstalledPlugins,
    );
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&backend_setting, &plugin_setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources {
            native_backends:
                crate::api::routes::runtime_control_state::RuntimeOptionsState::Unavailable {
                    reason: "No native runtime backends are available on this host.".to_string(),
                    note: Some("The current value will be preserved.".to_string()),
                },
            installed_plugins:
                crate::api::routes::runtime_control_state::RuntimeOptionsState::Unknown,
            ..Default::default()
        },
    );
    let backend_entry = payload
        .settings
        .get("plugin.demo.settings.runtime_kind")
        .expect("backend entry should be disabled with a reason");
    assert!(!backend_entry.enabled);
    assert_eq!(
        backend_entry.reason.as_deref(),
        Some("No native runtime backends are available on this host.")
    );
    assert!(
        !payload
            .settings
            .contains_key("plugin.demo.settings.plugin_name")
    );
}
