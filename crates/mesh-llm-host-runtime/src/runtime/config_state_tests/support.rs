use super::*;
pub(super) const FULL_SURFACE_VALID_FIXTURE: &str =
    include_str!("../../../tests/fixtures/skippy_full_surface_valid.toml");
pub(super) const CONTROL_FIXTURE_VALID: &str =
    include_str!("../../../tests/fixtures/schema_driven_controls_valid.toml");
pub(super) const CONTROL_FIXTURE_INVALID: &str =
    include_str!("../../../tests/fixtures/schema_driven_controls_invalid.toml");

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct DiagnosticSignature {
    path: String,
    canonical_path: String,
    severity: &'static str,
    code: &'static str,
}

pub(super) type LoggingConfigChange = (&'static str, fn(&mut LoggingConfig));
pub(super) type MeshConfigChange = (&'static str, fn(&mut MeshConfig));

impl DiagnosticSignature {
    pub(super) fn new(
        path: String,
        canonical_path: String,
        severity: &'static str,
        code: &'static str,
    ) -> Self {
        Self {
            path,
            canonical_path,
            severity,
            code,
        }
    }
}

pub(super) fn severity_label(severity: ConfigDiagnosticSeverity) -> &'static str {
    match severity {
        ConfigDiagnosticSeverity::Error => "error",
        ConfigDiagnosticSeverity::Warning => "warning",
        ConfigDiagnosticSeverity::Info => "info",
    }
}

pub(super) fn code_label(code: ConfigDiagnosticCode) -> &'static str {
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

pub(super) fn diagnostic_signatures(
    diagnostics: &[mesh_llm_config::ConfigDiagnostic],
) -> BTreeSet<DiagnosticSignature> {
    diagnostics
        .iter()
        .map(|diagnostic| {
            DiagnosticSignature::new(
                diagnostic
                    .path
                    .as_ref()
                    .map(|path| path.render())
                    .expect("diagnostic should include path"),
                diagnostic
                    .canonical_path
                    .as_ref()
                    .map(|path| path.render())
                    .expect("diagnostic should include canonical path"),
                severity_label(diagnostic.severity),
                code_label(diagnostic.code),
            )
        })
        .collect()
}

pub(super) fn test_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mesh-llm-config-state-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir
}

pub(super) fn minimal_valid_config() -> MeshConfig {
    MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    }
}

pub(super) fn installed_plugin_metadata(
    name: &str,
    schema: Option<InstalledPluginConfigSchema>,
) -> InstalledPluginMetadata {
    InstalledPluginMetadata {
        name: name.to_string(),
        source_repository: format!("https://github.com/mesh-llm/{name}"),
        installed_version: "v1.0.0".to_string(),
        target_triple: std::env::consts::ARCH.to_string(),
        downloaded_asset_name: format!("{name}.tar.gz"),
        install_path: std::env::temp_dir().join(format!("mesh-llm-plugin-{name}")),
        enabled: true,
        manifest: Some(InstalledPluginManifestMetadata {
            config_schema: schema,
            web_ui: None,
        }),
        last_protocol_version: Some(1),
        last_status: Some("installed".to_string()),
        last_error: None,
    }
}

pub(super) fn legacy_unvalidated_schema(plugin_name: &str) -> InstalledPluginConfigSchema {
    InstalledPluginConfigSchema {
        plugin_name: plugin_name.to_string(),
        schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
        allow_unvalidated_config: true,
        settings: Vec::new(),
    }
}

pub(super) fn strict_blackboard_schema(
    plugin_name: &str,
    allow_unvalidated_config: bool,
) -> InstalledPluginConfigSchema {
    InstalledPluginConfigSchema {
        plugin_name: plugin_name.to_string(),
        schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
        allow_unvalidated_config,
        settings: vec![
            mesh_llm_plugin_manager::InstalledPluginSettingSchema {
                key: "retention_days".to_string(),
                value_schema: mesh_llm_plugin_manager::InstalledPluginValueSchema {
                    kind: mesh_llm_plugin_manager::InstalledPluginValueKind::Integer,
                    enum_values: Vec::new(),
                    items: None,
                    object_properties: Vec::new(),
                    allow_additional_properties: false,
                },
                required: true,
                default_json: Some("14".to_string()),
                constraints: vec![mesh_llm_plugin_manager::InstalledPluginConstraint::Range {
                    min: Some("1".to_string()),
                    max: Some("365".to_string()),
                }],
                apply_mode:
                    mesh_llm_plugin_manager::InstalledPluginApplyMode::DynamicValidationOnly,
                restart_scope: mesh_llm_plugin_manager::InstalledPluginRestartScope::PluginProcess,
                visibility: mesh_llm_plugin_manager::InstalledPluginVisibility::User,
                description: Some("Retention window".to_string()),
                presentation: None,
                control_behavior: None,
            },
            mesh_llm_plugin_manager::InstalledPluginSettingSchema {
                key: "mode".to_string(),
                value_schema: mesh_llm_plugin_manager::InstalledPluginValueSchema {
                    kind: mesh_llm_plugin_manager::InstalledPluginValueKind::Enum,
                    enum_values: vec!["strict".to_string(), "relaxed".to_string()],
                    items: None,
                    object_properties: Vec::new(),
                    allow_additional_properties: false,
                },
                required: false,
                default_json: Some("\"strict\"".to_string()),
                constraints: Vec::new(),
                apply_mode:
                    mesh_llm_plugin_manager::InstalledPluginApplyMode::DynamicValidationOnly,
                restart_scope: mesh_llm_plugin_manager::InstalledPluginRestartScope::PluginProcess,
                visibility: mesh_llm_plugin_manager::InstalledPluginVisibility::User,
                description: Some("Conflict mode".to_string()),
                presentation: None,
                control_behavior: None,
            },
        ],
    }
}

pub(super) fn with_plugin_store(metadata: &[InstalledPluginMetadata], test: impl FnOnce()) {
    struct PluginDirRestoreGuard {
        previous: Option<std::ffi::OsString>,
    }

    impl Drop for PluginDirRestoreGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                // SAFETY: `with_plugin_store` is only called from `#[serial_test::serial]`
                // tests in this module, so restoring the process env here cannot race with
                // other tests that read or write `MESH_LLM_PLUGIN_DIR`.
                unsafe { std::env::set_var("MESH_LLM_PLUGIN_DIR", previous) };
            } else {
                // SAFETY: This is the paired env cleanup for the same serialized test scope.
                unsafe { std::env::remove_var("MESH_LLM_PLUGIN_DIR") };
            }
        }
    }

    let temp = tempfile::TempDir::new().expect("plugin store temp dir");
    let store = PluginStore::new(temp.path());
    for entry in metadata {
        store.save(entry).expect("save plugin metadata");
    }

    let previous = std::env::var_os("MESH_LLM_PLUGIN_DIR");
    let _restore_plugin_dir = PluginDirRestoreGuard { previous };
    // SAFETY: `with_plugin_store` is only used by `#[serial_test::serial]` tests in this
    // module, so this temporary process-wide override cannot race with concurrent tests.
    unsafe { std::env::set_var("MESH_LLM_PLUGIN_DIR", temp.path()) };
    test();
}

pub(super) fn representative_nested_config() -> MeshConfig {
    toml::from_str(
        r#"version = 1

[gpu]
assignment = "auto"
parallel = 2

[defaults.model_fit]
ctx_size = 8192
kv_unified = "auto"

[defaults.hardware]
gpu_layers = "auto"
tensor_split = []

[defaults.throughput]
parallel = 3

[defaults.skippy]

[defaults.speculative]
mode = "auto"
pairing_fault = "warn_disable"

[defaults.request_defaults]
reasoning_budget = "auto"
reasoning_format = "auto"

[defaults.multimodal]
mmproj = "defaults-projector.gguf"

[defaults.advanced.server]
alias = "defaults-alias"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.model_fit]
ctx_size = 16384
cache_type_k = "q8_0"

[models.hardware]
gpu_layers = 99
tensor_split = []

[models.throughput]
parallel = 4

[models.skippy]
binary_stage_transport = "auto"

[models.speculative]
mode = "auto"
draft_selection_policy = "auto"

[models.request_defaults]
top_p = 0.95
reasoning_budget = "auto"

[models.multimodal]
mmproj = "model-projector.gguf"

[models.advanced.server]
alias = "model-alias"
"#,
    )
    .expect("representative nested config should parse")
}

pub(super) fn assert_representative_nested_fields(config: &MeshConfig) {
    let json = serde_json::to_value(config).expect("config should serialize");
    assert_eq!(json["defaults"]["model_fit"]["kv_unified"], "auto");
    assert_eq!(json["defaults"]["hardware"]["gpu_layers"], "auto");
    assert_eq!(json["defaults"]["throughput"]["parallel"], 3);
    assert_eq!(json["defaults"]["speculative"]["mode"], "auto");
    assert_eq!(
        json["defaults"]["request_defaults"]["reasoning_budget"],
        "auto"
    );
    assert_eq!(
        json["defaults"]["multimodal"]["mmproj"],
        "defaults-projector.gguf"
    );
    assert_eq!(
        json["defaults"]["advanced"]["server"]["alias"],
        "defaults-alias"
    );

    assert_eq!(json["models"][0]["model_fit"]["ctx_size"], 16384);
    assert_eq!(json["models"][0]["hardware"]["gpu_layers"], 99);
    assert_eq!(json["models"][0]["throughput"]["parallel"], 4);
    assert_eq!(
        json["models"][0]["skippy"]["binary_stage_transport"],
        "auto"
    );
    assert_eq!(
        json["models"][0]["speculative"]["draft_selection_policy"],
        "auto"
    );
    assert_eq!(json["models"][0]["request_defaults"]["top_p"], 0.95);
    assert_eq!(
        json["models"][0]["multimodal"]["mmproj"],
        "model-projector.gguf"
    );
    assert_eq!(
        json["models"][0]["advanced"]["server"]["alias"],
        "model-alias"
    );
}
