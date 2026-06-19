use super::*;
use mesh_llm_config::{
    ConfigDiagnosticCode, ConfigDiagnosticSeverity, ConfigPath, MeshConfig,
    validate_config_diagnostics,
};
use mesh_llm_plugin_manager::{
    InstalledPluginApplyMode, InstalledPluginConfigSchema, InstalledPluginConstraint,
    InstalledPluginManifestMetadata, InstalledPluginMetadata, InstalledPluginRestartScope,
    InstalledPluginSettingSchema, InstalledPluginValueKind, InstalledPluginValueSchema,
    InstalledPluginVisibility, PluginStore, SUPPORTED_PLUGIN_SCHEMA_VERSION,
};
use std::collections::BTreeSet;

const VALID_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/schema_driven_controls_valid.toml"
));
const INVALID_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/schema_driven_controls_invalid.toml"
));

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DiagnosticSignature {
    path: String,
    canonical_path: String,
    severity: String,
    code: String,
}

impl DiagnosticSignature {
    fn new(
        path: String,
        canonical_path: String,
        severity: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            path,
            canonical_path,
            severity: severity.into(),
            code: code.into(),
        }
    }
}

fn severity_label(severity: ConfigDiagnosticSeverity) -> &'static str {
    match severity {
        ConfigDiagnosticSeverity::Error => "error",
        ConfigDiagnosticSeverity::Warning => "warning",
        ConfigDiagnosticSeverity::Info => "info",
    }
}

fn code_label(code: ConfigDiagnosticCode) -> &'static str {
    match code {
        ConfigDiagnosticCode::InvalidValue => "invalid_value",
        ConfigDiagnosticCode::MissingRequiredValue => "missing_required_value",
        ConfigDiagnosticCode::UnknownField => "unknown_field",
        ConfigDiagnosticCode::UnsupportedField => "unsupported_field",
        ConfigDiagnosticCode::RejectedField => "rejected_field",
        ConfigDiagnosticCode::AliasApplied => "alias_applied",
        ConfigDiagnosticCode::MisplacedField => "misplaced_field",
        ConfigDiagnosticCode::SchemaUnavailable => "schema_unavailable",
        ConfigDiagnosticCode::LegacyUnvalidatedConfig => "legacy_unvalidated_config",
        ConfigDiagnosticCode::UnsupportedSchemaVersion => "unsupported_schema_version",
    }
}

fn expected_signatures(raw: &str) -> BTreeSet<DiagnosticSignature> {
    let config: MeshConfig = toml::from_str(raw).expect("fixture should deserialize");
    validate_config_diagnostics(&config)
        .into_iter()
        .map(|diagnostic| {
            DiagnosticSignature::new(
                diagnostic
                    .path
                    .as_ref()
                    .map(ConfigPath::render)
                    .expect("validator diagnostics should include path"),
                diagnostic
                    .canonical_path
                    .as_ref()
                    .map(ConfigPath::render)
                    .expect("validator diagnostics should include canonical path"),
                severity_label(diagnostic.severity),
                code_label(diagnostic.code),
            )
        })
        .collect()
}

fn payload_signatures(payload: &serde_json::Value) -> BTreeSet<DiagnosticSignature> {
    payload["diagnostics"]
        .as_array()
        .expect("diagnostics should be an array")
        .iter()
        .map(|diagnostic| {
            DiagnosticSignature::new(
                diagnostic["path"]
                    .as_str()
                    .expect("path should be a string")
                    .to_string(),
                diagnostic["canonical_path"]
                    .as_str()
                    .expect("canonical path should be a string")
                    .to_string(),
                diagnostic["severity"]
                    .as_str()
                    .expect("severity should be a string"),
                diagnostic["code"]
                    .as_str()
                    .expect("code should be a string"),
            )
        })
        .collect()
}

struct PluginDirGuard {
    previous: Option<std::ffi::OsString>,
}

impl PluginDirGuard {
    fn set(path: &std::path::Path) -> Self {
        let previous = std::env::var_os("MESH_LLM_PLUGIN_DIR");
        // SAFETY: This helper is used only by `#[serial]` tests in this module, so no
        // concurrent test in this process can observe a partially updated plugin dir env var.
        unsafe { std::env::set_var("MESH_LLM_PLUGIN_DIR", path) };
        Self { previous }
    }
}

impl Drop for PluginDirGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            // SAFETY: This runs as part of the same `#[serial]` test-scoped guard that set the
            // variable, so restoring the process env cannot race with other tests in this module.
            Some(previous) => unsafe { std::env::set_var("MESH_LLM_PLUGIN_DIR", previous) },
            // SAFETY: This is the paired restoration for the serialized test-scoped env override.
            None => unsafe { std::env::remove_var("MESH_LLM_PLUGIN_DIR") },
        }
    }
}

fn blackboard_schema() -> InstalledPluginConfigSchema {
    InstalledPluginConfigSchema {
        plugin_name: "blackboard".to_string(),
        schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
        allow_unvalidated_config: false,
        settings: vec![
            InstalledPluginSettingSchema {
                key: "retention_days".to_string(),
                value_schema: InstalledPluginValueSchema {
                    kind: InstalledPluginValueKind::Integer,
                    enum_values: Vec::new(),
                    items: None,
                    object_properties: Vec::new(),
                    allow_additional_properties: false,
                },
                required: true,
                default_json: Some("14".to_string()),
                constraints: vec![InstalledPluginConstraint::Range {
                    min: Some("1".to_string()),
                    max: Some("365".to_string()),
                }],
                apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                restart_scope: InstalledPluginRestartScope::PluginProcess,
                visibility: InstalledPluginVisibility::User,
                description: Some("Retention window".to_string()),
                presentation: None,
                control_behavior: None,
            },
            InstalledPluginSettingSchema {
                key: "mode".to_string(),
                value_schema: InstalledPluginValueSchema {
                    kind: InstalledPluginValueKind::Enum,
                    enum_values: vec!["strict".to_string(), "relaxed".to_string()],
                    items: None,
                    object_properties: Vec::new(),
                    allow_additional_properties: false,
                },
                required: false,
                default_json: Some("\"strict\"".to_string()),
                constraints: Vec::new(),
                apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                restart_scope: InstalledPluginRestartScope::PluginProcess,
                visibility: InstalledPluginVisibility::User,
                description: Some("Conflict mode".to_string()),
                presentation: None,
                control_behavior: None,
            },
        ],
    }
}

fn install_blackboard_schema(plugin_dir: &std::path::Path) {
    let store = PluginStore::new(plugin_dir);
    store
        .save(&InstalledPluginMetadata {
            name: "blackboard".to_string(),
            source_repository: "https://github.com/mesh-llm/blackboard".to_string(),
            installed_version: "v1.0.0".to_string(),
            target_triple: std::env::consts::ARCH.to_string(),
            downloaded_asset_name: "blackboard.tar.gz".to_string(),
            install_path: std::env::temp_dir().join("mesh-llm-plugin-blackboard-api-tests"),
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: Some(blackboard_schema()),
            }),
            last_protocol_version: Some(1),
            last_status: Some("installed".to_string()),
            last_error: None,
        })
        .expect("save plugin metadata");
}

fn validate_request_body(toml: &str) -> String {
    json!({ "toml": toml, "path": "manual.toml" }).to_string()
}

#[tokio::test]
async fn runtime_config_validate_api_rejects_ubatch_above_batch() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = validate_request_body(
        r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.model_fit]
batch = 32
ubatch = 64
"#,
    );

    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/config/validate", &body),
    )
    .await;
    let payload = json_body(&response);
    let diagnostics = payload["diagnostics"]
        .as_array()
        .expect("diagnostics should be an array");

    assert_eq!(payload["ok"], false, "response: {response}");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic["path"] == "models[0].model_fit.ubatch"
            && diagnostic["canonical_path"] == "models.<model-ref>.model_fit.ubatch"
            && diagnostic["message"]
                .as_str()
                .expect("message should be a string")
                .contains("must be less than or equal to models[0].model_fit.batch")
    }));

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn runtime_config_validate_api_reports_hf_pair_and_rejected_control_diagnostics() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = validate_request_body(
        r#"version = 1

[defaults.speculative]
draft_hf_repo = "mesh/test"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.hardware]
rpc_backend = "rpc"
"#,
    );

    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/config/validate", &body),
    )
    .await;
    let payload = json_body(&response);
    let diagnostics = payload["diagnostics"]
        .as_array()
        .expect("diagnostics should be an array");

    assert_eq!(payload["ok"], false, "response: {response}");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic["path"] == "defaults.speculative.draft_hf_file"
            && diagnostic["canonical_path"] == "defaults.speculative.draft_hf_file"
            && diagnostic["message"]
                .as_str()
                .expect("message should be a string")
                .contains("must be set when defaults.speculative.draft_hf_repo is set")
    }));
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic["path"] == "models[0].hardware.rpc_backend"
            && diagnostic["canonical_path"] == "models.<model-ref>.hardware.rpc_backend"
            && diagnostic["code"] == "rejected_field"
            && diagnostic["schema_source"] == "built_in"
    }));

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn runtime_config_validate_api_uses_installed_plugin_schema_for_required_and_unknown_settings()
 {
    let plugin_dir = tempfile::tempdir().expect("plugin dir tempdir");
    install_blackboard_schema(plugin_dir.path());
    let _guard = PluginDirGuard::set(plugin_dir.path());

    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = validate_request_body(
        r#"version = 1

[[plugin]]
name = "blackboard"

[plugin.settings]
mode = "strict"
unknown = true
"#,
    );

    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/config/validate", &body),
    )
    .await;
    let payload = json_body(&response);
    let diagnostics = payload["diagnostics"]
        .as_array()
        .expect("diagnostics should be an array");

    assert_eq!(payload["ok"], false, "response: {response}");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic["path"] == "plugin.blackboard.settings.retention_days"
            && diagnostic["canonical_path"] == "plugin.blackboard.settings.retention_days"
            && diagnostic["code"] == "missing_required_value"
            && diagnostic["schema_source"] == "plugin"
    }));
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic["path"] == "plugin.blackboard.settings.unknown"
            && diagnostic["canonical_path"] == "plugin.blackboard.settings.unknown"
            && diagnostic["code"] == "unknown_field"
            && diagnostic["schema_source"] == "plugin"
    }));

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn runtime_config_validate_api_accepts_schema_driven_valid_fixture() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = validate_request_body(VALID_FIXTURE);

    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/config/validate", &body),
    )
    .await;
    let payload = json_body(&response);

    assert_eq!(payload["ok"], true, "response: {response}");
    assert!(payload["diagnostics"].as_array().unwrap().is_empty());

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn runtime_config_validate_api_matches_validator_signatures_for_schema_driven_invalid_fixture()
 {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = validate_request_body(INVALID_FIXTURE);

    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/config/validate", &body),
    )
    .await;
    let payload = json_body(&response);

    assert_eq!(payload["ok"], false, "response: {response}");
    assert_eq!(
        payload_signatures(&payload),
        expected_signatures(INVALID_FIXTURE)
    );

    handle.await.unwrap().unwrap();
}
