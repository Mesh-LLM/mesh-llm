use super::*;

fn endpoint_config() -> plugin::MeshConfig {
    toml::from_str(
        r#"[[plugin]]
name = "openai-endpoint"
url = "http://127.0.0.1:18080/v1"
"#,
    )
    .unwrap()
}

#[test]
fn enabled_endpoint_without_local_models_permits_external_startup() {
    let options = RuntimeOptions::default();
    let mut config = endpoint_config();
    assert!(permits_external_startup(&options, &config));
    config.plugins[0].enabled = Some(true);
    assert!(permits_external_startup(&options, &config));
}

#[test]
fn missing_disabled_and_blank_endpoints_do_not_permit_external_startup() {
    let options = RuntimeOptions::default();
    assert!(!permits_external_startup(
        &options,
        &plugin::MeshConfig::default()
    ));
    for url in [None, Some(String::new()), Some(" \t\n".into())] {
        let mut config = endpoint_config();
        config.plugins[0].url = url;
        assert!(!permits_external_startup(&options, &config));
    }
    let mut config = endpoint_config();
    config.plugins[0].enabled = Some(false);
    assert!(!permits_external_startup(&options, &config));
}

#[test]
fn every_explicit_local_source_retains_native_runtime_requirement() {
    let config = endpoint_config();
    let sources = [
        RuntimeOptions {
            model: vec!["model.gguf".into()],
            ..Default::default()
        },
        RuntimeOptions {
            model: vec!["org/layer-package".into()],
            ..Default::default()
        },
        RuntimeOptions {
            gguf: vec!["model.gguf".into()],
            ..Default::default()
        },
        RuntimeOptions {
            mmproj: Some("mmproj.gguf".into()),
            ..Default::default()
        },
        RuntimeOptions {
            draft: Some("draft.gguf".into()),
            ..Default::default()
        },
        RuntimeOptions {
            split: true,
            ..Default::default()
        },
        RuntimeOptions {
            local_model_only: true,
            ..Default::default()
        },
        RuntimeOptions {
            native_serving_plugin: Some("plugin.so".into()),
            ..Default::default()
        },
    ];
    for options in sources {
        assert!(!permits_external_startup(&options, &config));
    }
    for model in ["model.gguf", "org/layer-package"] {
        let mut config = endpoint_config();
        config.models.push(plugin::ModelConfigEntry {
            model: model.into(),
            ..Default::default()
        });
        assert!(!permits_external_startup(
            &RuntimeOptions::default(),
            &config
        ));
    }
}

#[test]
fn explicit_backend_version_and_abi_pins_retain_native_requirement() {
    let options = RuntimeOptions::default();
    for native_runtime in [
        mesh_llm_config::NativeRuntimeConfig {
            selection: Some("cpu".into()),
            ..Default::default()
        },
        mesh_llm_config::NativeRuntimeConfig {
            mesh_version: Some("0.59.0".into()),
            ..Default::default()
        },
        mesh_llm_config::NativeRuntimeConfig {
            skippy_abi: Some("0.1.52".into()),
            ..Default::default()
        },
    ] {
        let mut config = endpoint_config();
        config.runtime.native_runtime = native_runtime;
        assert!(!permits_external_startup(&options, &config));
    }
    for flavor in mesh_llm_system::backend::BinaryFlavor::ALL {
        let options = RuntimeOptions {
            llama_flavor: Some(flavor),
            ..Default::default()
        };
        assert!(!permits_external_startup(&options, &endpoint_config()));
    }
}

#[test]
fn malformed_config_is_rejected_before_provider_eligibility() {
    assert!(plugin::parse_config_toml("[[plugin]\nurl = 5").is_err());
}

#[test]
fn non_inference_plugin_handshake_does_not_confirm_external_serving() {
    let error = require_external_inference_provider("native runtime unavailable", 0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no enabled plugin completed a compatible inference endpoint handshake")
    );
    assert!(error.to_string().contains("native runtime unavailable"));
    require_external_inference_provider("native runtime unavailable", 1).unwrap();
}

#[cfg(feature = "dynamic-native-runtime")]
fn resolution_error(enumeration_failed: bool) -> anyhow::Error {
    anyhow::Error::new(mesh_llm_runtime_install::NativeRuntimeResolutionError {
        summary: "fixture native resolution failed".into(),
        selection: mesh_llm_native_runtime::RuntimeSelection::Recommended,
        catalogs: Default::default(),
        candidates: Vec::new(),
        set_aside: 0,
        enumeration_failed,
    })
}

#[cfg(feature = "dynamic-native-runtime")]
#[test]
fn only_typed_availability_errors_qualify_for_external_fallback() {
    assert!(is_runtime_availability_error(&resolution_error(false)));
    assert!(!is_runtime_availability_error(&resolution_error(true)));
    for message in [
        "runtime library has incompatible ABI",
        "invalid runtime selection",
        "checksum mismatch",
        "native loader failed",
        "no runtime available",
    ] {
        assert!(!is_runtime_availability_error(&anyhow::anyhow!(message)));
    }
    assert!(is_runtime_availability_error(
        &resolution_error(false).context("startup context")
    ));
}

#[cfg(feature = "dynamic-native-runtime")]
#[test]
fn installed_corrupt_or_bundled_artifacts_prevent_external_fallback() {
    assert!(runtime_artifacts_absent(0, 0, 0));
    assert!(!runtime_artifacts_absent(1, 0, 0));
    assert!(!runtime_artifacts_absent(0, 1, 0));
    assert!(!runtime_artifacts_absent(0, 0, 1));
}

#[cfg(feature = "dynamic-native-runtime")]
#[test]
fn unknown_backend_selection_remains_explicit_and_cannot_fallback() {
    let native_runtime = mesh_llm_config::NativeRuntimeConfig {
        selection: Some("unsupported-backend".into()),
        ..Default::default()
    };
    let selection = crate::system::native_runtime::NativeRuntimeStartupSelection::from_config(
        native_runtime.clone(),
        None,
    )
    .unwrap();
    assert_eq!(
        selection.runtime_selection,
        mesh_llm_native_runtime::RuntimeSelection::Backend {
            kind: mesh_llm_native_runtime::NativeRuntimeBackendKind::Other(
                "unsupported-backend".into()
            ),
            cuda_toolkit_major: None,
        }
    );
    let mut config = endpoint_config();
    config.runtime.native_runtime = native_runtime;
    assert!(!permits_external_startup(
        &RuntimeOptions::default(),
        &config
    ));
}

#[cfg(feature = "dynamic-native-runtime")]
#[tokio::test]
async fn unavailable_manifest_connection_is_recoverable_but_invalid_url_is_not() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(1))
        .build()
        .unwrap();
    let connection = client
        .get(format!("http://{address}/manifest.json"))
        .send()
        .await
        .unwrap_err();
    assert!(connection.is_connect());
    assert!(is_runtime_availability_error(&connection.into()));
    let invalid_url = client.get("http://[").build().unwrap_err();
    assert!(!is_runtime_availability_error(&invalid_url.into()));
}
