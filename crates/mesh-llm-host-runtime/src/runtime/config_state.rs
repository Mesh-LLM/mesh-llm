use anyhow::Result;
use mesh_llm_config::{
    ConfigDiagnostic, ConfigDiagnosticSeverity, ConfigPath, LoggingConfig,
    built_in_config_schema_descriptor, legacy_validation_error_text,
};

// Disambiguate from the local proto-mirror ConfigApplyMode (Staged/Noop).
use mesh_llm_config::ConfigApplyMode as SchemaApplyMode;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::plugin::{
    ConfigStore, MeshConfig, config_to_toml, load_config,
    validate_config_diagnostics_with_installed_plugin_schemas,
};
use crate::protocol::convert::{canonical_config_hash, mesh_config_to_proto};

/// Mirrors the `ConfigApplyMode` proto enum; kept in the domain layer so
/// `config_state` does not depend on the generated proto crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigApplyMode {
    /// Config written to disk and revision counter advanced.
    Staged,
    /// Config written to disk and the installed runtime accepted the complete
    /// dynamic change before this result was reported.
    Live,
    /// No-op: the incoming config was identical to the current one.
    Noop,
}

#[derive(Debug)]
pub(crate) enum ApplyResult {
    Applied {
        revision: u64,
        hash: [u8; 32],
        apply_mode: ConfigApplyMode,
        diagnostics: Vec<ConfigDiagnostic>,
    },
    AppliedWithRestartRequired {
        revision: u64,
        hash: [u8; 32],
        diagnostics: Vec<ConfigDiagnostic>,
    },
    RevisionConflict {
        current_revision: u64,
    },
    PersistedWithRevisionTrackingError {
        revision: u64,
        hash: [u8; 32],
        error: String,
        diagnostics: Vec<ConfigDiagnostic>,
    },
    ValidationError {
        error: String,
        diagnostics: Vec<ConfigDiagnostic>,
    },
    PersistError(String),
}

#[derive(Debug)]
pub(crate) enum ConfigApplyPreparation {
    Immediate(ApplyResult),
    Pending(Box<PendingConfigApply>),
}

#[derive(Debug)]
pub(crate) enum ConfigPersistence {
    Persisted,
    RevisionTrackingError(String),
    PersistError(String),
}

#[derive(Debug)]
pub(crate) struct PendingConfigApply {
    config: MeshConfig,
    config_path: PathBuf,
    revision: u64,
    hash: [u8; 32],
    write_hash: [u8; 32],
    diagnostics: Vec<ConfigDiagnostic>,
    logging_requires_restart: bool,
    dynamic_logging_only: bool,
    old_logging: LoggingConfig,
}

impl PendingConfigApply {
    pub(crate) fn persist(&self) -> ConfigPersistence {
        if let Err(error) = ConfigStore::open(self.config_path.clone()).save(&self.config) {
            return ConfigPersistence::PersistError(format!("failed to write config: {error}"));
        }

        let sidecar = revision_sidecar_path(&self.config_path);
        match atomic_write(&sidecar, self.revision.to_string().as_bytes()) {
            Ok(()) => ConfigPersistence::Persisted,
            Err(error) => ConfigPersistence::RevisionTrackingError(format!(
                "failed to write revision sidecar: {error}; config persisted and in-memory revision advanced, but on-disk revision tracking may be stale"
            )),
        }
    }

    pub(crate) fn apply_live_logging(&self) -> bool {
        if !self.dynamic_logging_only {
            return false;
        }

        if crate::apply_live_logging_limits(&self.config.logging).is_err() {
            tracing::warn!(
                "Logging runtime unavailable; retaining dynamic logging settings as staged configuration"
            );
            return false;
        }
        true
    }

    pub(crate) fn restore_live_logging(&self) {
        let _ = crate::apply_live_logging_limits(&self.old_logging);
    }

    fn applied_result(&self, apply_mode: ConfigApplyMode) -> ApplyResult {
        if self.logging_requires_restart {
            ApplyResult::AppliedWithRestartRequired {
                revision: self.revision,
                hash: self.hash,
                diagnostics: self.diagnostics.clone(),
            }
        } else {
            ApplyResult::Applied {
                revision: self.revision,
                hash: self.hash,
                apply_mode,
                diagnostics: self.diagnostics.clone(),
            }
        }
    }
}

pub(crate) struct ConfigState {
    revision: u64,
    config_hash: [u8; 32],
    config: MeshConfig,
    config_path: PathBuf,
    last_write_config_hash: [u8; 32],
    apply_serialization_lock: std::sync::Arc<std::sync::Mutex<()>>,
}

fn revision_sidecar_path(config_path: &Path) -> PathBuf {
    let parent = config_path.parent().unwrap_or(Path::new("."));
    if let Some(file_name) = config_path.file_name() {
        let mut sidecar_name = std::ffi::OsString::from(file_name);
        sidecar_name.push(".revision");
        parent.join(sidecar_name)
    } else {
        parent.join("config-revision")
    }
}

fn read_revision(sidecar: &Path) -> u64 {
    let rev = std::fs::read_to_string(sidecar)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    if let Some(rev) = rev {
        return rev;
    }
    let legacy = sidecar
        .parent()
        .unwrap_or(Path::new("."))
        .join("config-revision");
    std::fs::read_to_string(&legacy)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn atomic_write(target: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file_name = target
        .file_name()
        .unwrap_or(target.as_os_str())
        .to_string_lossy();
    let parent = target.parent().unwrap_or(Path::new("."));
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let tmp = parent.join(format!(".{}.{}.{}.tmp", file_name, pid, nanos));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    file.write_all(contents)?;
    file.sync_all()?;
    drop(file);
    // TODO(windows): this remove+rename sequence is not truly atomic on Windows.
    // Replace with MoveFileExW(MOVEFILE_REPLACE_EXISTING) or tempfile::persist_noclobber-like behavior.
    #[cfg(windows)]
    if target.exists() {
        std::fs::remove_file(target)?;
    }
    if let Err(e) = std::fs::rename(&tmp, target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

fn local_config_write_hash(config: &MeshConfig) -> [u8; 32] {
    let bytes = serde_json::to_vec(config)
        .or_else(|_| crate::plugin::config_to_toml(config).map(String::into_bytes))
        .unwrap_or_default();
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn field_requires_restart(field_name: &str) -> bool {
    let rendered_path = format!("logging.{field_name}");
    let descriptor = ConfigPath::parse_rendered(&rendered_path)
        .ok()
        .and_then(|path| built_in_config_schema_descriptor(&path));

    // Unknown/unparseable paths cannot acquire a live-apply contract by
    // accident; validation will reject them before a config is persisted.
    descriptor.is_none_or(|schema| schema.apply_mode != SchemaApplyMode::DynamicApply)
}

fn logging_changes_require_restart(old: &LoggingConfig, new: &LoggingConfig) -> bool {
    let changed = [
        ("enabled", old.enabled != new.enabled),
        (
            "application_state_root",
            old.application_state_root != new.application_state_root,
        ),
        (
            "summary_line_limit",
            old.summary_line_limit != new.summary_line_limit,
        ),
        (
            "event_buffer_size",
            old.event_buffer_size != new.event_buffer_size,
        ),
        (
            "retention_ttl_secs",
            old.retention_ttl_secs != new.retention_ttl_secs,
        ),
        (
            "replay_capacity",
            old.replay_capacity != new.replay_capacity,
        ),
        ("queue_capacity", old.queue_capacity != new.queue_capacity),
        (
            "artifact.capture_mode",
            old.artifact.capture_mode != new.artifact.capture_mode,
        ),
        (
            "artifact.byte_limit_bytes",
            old.artifact.byte_limit_bytes != new.artifact.byte_limit_bytes,
        ),
        (
            "artifact.aggregate_limit_bytes",
            old.artifact.aggregate_limit_bytes != new.artifact.aggregate_limit_bytes,
        ),
        (
            "export_limit_bytes",
            old.export_limit_bytes != new.export_limit_bytes,
        ),
        (
            "cleanup_cadence_secs",
            old.cleanup_cadence_secs != new.cleanup_cadence_secs,
        ),
        (
            "webhook.enabled",
            old.webhook.enabled != new.webhook.enabled,
        ),
        ("webhook.url", old.webhook.url != new.webhook.url),
        (
            "webhook.max_attempts",
            old.webhook.max_attempts != new.webhook.max_attempts,
        ),
        (
            "webhook.timeout_secs",
            old.webhook.timeout_secs != new.webhook.timeout_secs,
        ),
        (
            "webhook.dead_letter_retention_secs",
            old.webhook.dead_letter_retention_secs != new.webhook.dead_letter_retention_secs,
        ),
    ];

    changed
        .into_iter()
        .any(|(field, differs)| differs && field_requires_restart(field))
}

fn logging_dynamic_limits_changed(old: &LoggingConfig, new: &LoggingConfig) -> bool {
    old.retention_ttl_secs != new.retention_ttl_secs || old.replay_capacity != new.replay_capacity
}

impl Default for ConfigState {
    fn default() -> Self {
        let config = crate::plugin::MeshConfig::default();
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        Self {
            revision: 0,
            config_hash,
            config,
            config_path: std::path::PathBuf::from("config.toml"),
            last_write_config_hash: [0xFF; 32],
            apply_serialization_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
        }
    }
}

impl ConfigState {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let config = load_config(Some(path))?;
        let revision = read_revision(&revision_sidecar_path(path));
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        let last_write_config_hash = if path.exists() {
            local_config_write_hash(&config)
        } else {
            [0xFF; 32]
        };
        Ok(Self {
            revision,
            config_hash,
            config,
            config_path: path.to_path_buf(),
            last_write_config_hash,
            apply_serialization_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
        })
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn config_hash(&self) -> &[u8; 32] {
        &self.config_hash
    }

    pub(crate) fn config(&self) -> &MeshConfig {
        &self.config
    }

    pub(crate) fn apply_serialization_lock(&self) -> std::sync::Arc<std::sync::Mutex<()>> {
        std::sync::Arc::clone(&self.apply_serialization_lock)
    }

    pub(crate) fn apply(&mut self, new_config: MeshConfig, expected_revision: u64) -> ApplyResult {
        match self.prepare_apply(new_config, expected_revision) {
            ConfigApplyPreparation::Immediate(result) => result,
            ConfigApplyPreparation::Pending(pending) => {
                let persistence = pending.persist();
                self.finish_apply(*pending, persistence)
            }
        }
    }

    pub(crate) fn prepare_apply(
        &self,
        new_config: MeshConfig,
        expected_revision: u64,
    ) -> ConfigApplyPreparation {
        let raw_toml = config_to_toml(&new_config).ok();
        let diagnostics = validate_config_diagnostics_with_installed_plugin_schemas(
            &new_config,
            raw_toml.as_deref(),
        );
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        {
            return ConfigApplyPreparation::Immediate(ApplyResult::ValidationError {
                error: legacy_validation_error_text(&diagnostics),
                diagnostics,
            });
        }

        if expected_revision != self.revision {
            return ConfigApplyPreparation::Immediate(ApplyResult::RevisionConflict {
                current_revision: self.revision,
            });
        }

        let proto = mesh_config_to_proto(&new_config);
        let new_hash = canonical_config_hash(&proto);
        let new_write_hash = local_config_write_hash(&new_config);

        if new_write_hash == self.last_write_config_hash {
            return ConfigApplyPreparation::Immediate(ApplyResult::Applied {
                revision: self.revision,
                hash: self.config_hash,
                apply_mode: ConfigApplyMode::Noop,
                diagnostics,
            });
        }

        let old_logging = self.config.logging.clone();
        let logging_requires_restart =
            logging_changes_require_restart(&old_logging, &new_config.logging);
        let dynamic_logging_only =
            logging_dynamic_limits_changed(&old_logging, &new_config.logging)
                && !logging_requires_restart;

        ConfigApplyPreparation::Pending(Box::new(PendingConfigApply {
            config: new_config,
            config_path: self.config_path.clone(),
            revision: self.revision + 1,
            hash: new_hash,
            write_hash: new_write_hash,
            diagnostics,
            logging_requires_restart,
            dynamic_logging_only,
            old_logging,
        }))
    }

    pub(crate) fn finish_apply(
        &mut self,
        pending: PendingConfigApply,
        persistence: ConfigPersistence,
    ) -> ApplyResult {
        match persistence {
            ConfigPersistence::PersistError(error) => ApplyResult::PersistError(error),
            ConfigPersistence::Persisted | ConfigPersistence::RevisionTrackingError(_) => {
                if self.revision.checked_add(1) != Some(pending.revision) {
                    return ApplyResult::RevisionConflict {
                        current_revision: self.revision,
                    };
                }
                self.config = pending.config.clone();
                self.config_hash = pending.hash;
                self.last_write_config_hash = pending.write_hash;
                self.revision = pending.revision;

                match persistence {
                    ConfigPersistence::Persisted => pending.applied_result(ConfigApplyMode::Staged),
                    ConfigPersistence::RevisionTrackingError(error) => {
                        ApplyResult::PersistedWithRevisionTrackingError {
                            revision: self.revision,
                            hash: self.config_hash,
                            error,
                            diagnostics: pending.diagnostics.clone(),
                        }
                    }
                    ConfigPersistence::PersistError(_) => unreachable!("handled above"),
                }
            }
        }
    }

    /// Apply configuration and, only for a dynamic-only logging limits change,
    /// update the installed local logging runtime before advertising `Live`.
    ///
    /// Static logging changes always flow through [`Self::apply`] untouched,
    /// even when they are submitted with new dynamic values. An unavailable
    /// runtime likewise leaves the valid configuration staged: this preserves
    /// an operator's desired settings without claiming a live mutation.
    #[cfg(test)]
    pub(crate) fn apply_with_live_logging(
        &mut self,
        new_config: MeshConfig,
        expected_revision: u64,
    ) -> ApplyResult {
        let preparation = self.prepare_apply(new_config, expected_revision);
        apply_prepared_config_with_live_logging(
            preparation,
            PendingConfigApply::persist,
            |pending, persistence| self.finish_apply(pending, persistence),
        )
    }
}

/// Persist a prepared configuration apply while keeping dynamic logging limits
/// synchronized with the persisted state.
pub(crate) fn apply_prepared_config_with_live_logging<Persist, Finish>(
    preparation: ConfigApplyPreparation,
    persist: Persist,
    finish: Finish,
) -> ApplyResult
where
    Persist: FnOnce(&PendingConfigApply) -> ConfigPersistence,
    Finish: FnOnce(PendingConfigApply, ConfigPersistence) -> ApplyResult,
{
    let pending = match preparation {
        ConfigApplyPreparation::Immediate(result) => return result,
        ConfigApplyPreparation::Pending(pending) => *pending,
    };
    let live_logging_applied = pending.apply_live_logging();
    let persistence = persist(&pending);
    if live_logging_applied && matches!(&persistence, ConfigPersistence::PersistError(_)) {
        // A persistence failure after a live apply is rare, but restore the
        // prior runtime settings before exposing the failure so the
        // configuration and service do not diverge.
        pending.restore_live_logging();
    }
    let result = finish(pending, persistence);

    match result {
        ApplyResult::Applied {
            revision,
            hash,
            diagnostics,
            ..
        } if live_logging_applied => ApplyResult::Applied {
            revision,
            hash,
            apply_mode: ConfigApplyMode::Live,
            diagnostics,
        },
        result @ ApplyResult::PersistedWithRevisionTrackingError { .. } => {
            // The primary config write succeeded and `ConfigState` has
            // already adopted the new revision. Keep the live pair in place;
            // only revision-sidecar recovery needs attention.
            result
        }
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{GpuAssignment, GpuConfig, MeshConfig};
    use mesh_llm_config::{
        ConfigDiagnosticCode, ConfigDiagnosticSeverity, validate_config_diagnostics,
    };
    use mesh_llm_plugin_manager::{
        InstalledPluginConfigSchema, InstalledPluginManifestMetadata, InstalledPluginMetadata,
        PluginStore, SUPPORTED_PLUGIN_SCHEMA_VERSION,
    };
    use std::collections::BTreeSet;

    const FULL_SURFACE_VALID_FIXTURE: &str =
        include_str!("../../tests/fixtures/skippy_full_surface_valid.toml");
    const CONTROL_FIXTURE_VALID: &str =
        include_str!("../../tests/fixtures/schema_driven_controls_valid.toml");
    const CONTROL_FIXTURE_INVALID: &str =
        include_str!("../../tests/fixtures/schema_driven_controls_invalid.toml");

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct DiagnosticSignature {
        path: String,
        canonical_path: String,
        severity: &'static str,
        code: &'static str,
    }

    type LoggingConfigChange = (&'static str, fn(&mut LoggingConfig));
    type MeshConfigChange = (&'static str, fn(&mut MeshConfig));

    impl DiagnosticSignature {
        fn new(
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

    fn diagnostic_signatures(
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

    fn test_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mesh-llm-config-state-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    fn minimal_valid_config() -> MeshConfig {
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

    fn installed_plugin_metadata(
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

    fn legacy_unvalidated_schema(plugin_name: &str) -> InstalledPluginConfigSchema {
        InstalledPluginConfigSchema {
            plugin_name: plugin_name.to_string(),
            schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
            allow_unvalidated_config: true,
            settings: Vec::new(),
        }
    }

    fn strict_blackboard_schema(
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
                    restart_scope:
                        mesh_llm_plugin_manager::InstalledPluginRestartScope::PluginProcess,
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
                    restart_scope:
                        mesh_llm_plugin_manager::InstalledPluginRestartScope::PluginProcess,
                    visibility: mesh_llm_plugin_manager::InstalledPluginVisibility::User,
                    description: Some("Conflict mode".to_string()),
                    presentation: None,
                    control_behavior: None,
                },
            ],
        }
    }

    fn with_plugin_store(metadata: &[InstalledPluginMetadata], test: impl FnOnce()) {
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

    fn representative_nested_config() -> MeshConfig {
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
activation_wire_dtype = "auto"

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
tensor_split = [0.7, 0.3]

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

    fn assert_representative_nested_fields(config: &MeshConfig) {
        let json = serde_json::to_value(config).expect("config should serialize");
        assert_eq!(json["defaults"]["model_fit"]["kv_unified"], "auto");
        assert_eq!(json["defaults"]["hardware"]["gpu_layers"], "auto");
        assert_eq!(json["defaults"]["throughput"]["parallel"], 3);
        assert_eq!(json["defaults"]["skippy"]["activation_wire_dtype"], "auto");
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

    #[test]
    fn config_sync_state_load() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        std::fs::write(
            &config_path,
            "version = 1\n\n[gpu]\nassignment = \"auto\"\n",
        )
        .expect("write config");

        let state = ConfigState::load(&config_path).expect("load");
        assert_eq!(state.revision(), 0);
        assert_eq!(state.config().version, Some(1));
        assert_eq!(state.config().gpu.assignment, GpuAssignment::Auto);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_apply_success() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        let mut state = ConfigState::load(&config_path).expect("load");
        assert_eq!(state.revision(), 0);

        let result = state.apply(minimal_valid_config(), 0);
        match result {
            ApplyResult::Applied {
                revision,
                hash: _,
                apply_mode,
                diagnostics,
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                assert!(diagnostics.is_empty());
            }
            other => panic!("expected Applied, got {other:?}"),
        }

        assert!(config_path.exists(), "config file not written");

        let sidecar = revision_sidecar_path(&config_path);
        let sidecar_contents = std::fs::read_to_string(&sidecar).expect("read sidecar");
        assert_eq!(sidecar_contents.trim(), "1");

        assert_eq!(state.revision(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_apply_preserves_additive_defaults_sections() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(
            &config_path,
            r#"version = 1

[defaults.throughput]
parallel = 2

[defaults.model_fit]
flash_attention = "auto"

[defaults.request_defaults]
reasoning_format = "deepseek"
"#,
        )
        .expect("write baseline config");

        let mut state = ConfigState::load(&config_path).expect("load baseline config");
        let mut config = minimal_valid_config();
        config.extra = toml::from_str(
            r#"[defaults.throughput]
parallel = 6

[defaults.model_fit]
flash_attention = "disabled"

[defaults.request_defaults]
reasoning_format = "qwen"
"#,
        )
        .expect("parse additive defaults table");

        let result = state.apply(config, 0);
        match result {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
            }
            other => panic!("expected additive defaults to be written, got {other:?}"),
        }

        let written = std::fs::read_to_string(&config_path).expect("read written config");
        let written: toml::Value = toml::from_str(&written).expect("written TOML parses");
        assert_eq!(
            written
                .get("defaults")
                .and_then(|defaults| defaults.get("throughput"))
                .and_then(|throughput| throughput.get("parallel"))
                .and_then(toml::Value::as_integer),
            Some(6)
        );
        assert_eq!(
            written
                .get("defaults")
                .and_then(|defaults| defaults.get("model_fit"))
                .and_then(|model_fit| model_fit.get("flash_attention"))
                .and_then(toml::Value::as_str),
            Some("disabled")
        );
        assert_eq!(
            written
                .get("defaults")
                .and_then(|defaults| defaults.get("request_defaults"))
                .and_then(|request_defaults| request_defaults.get("reasoning_format"))
                .and_then(toml::Value::as_str),
            Some("qwen")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_conflict() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        let mut state = ConfigState::load(&config_path).expect("load");

        let result = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(result, ApplyResult::Applied { revision: 1, .. }),
            "first apply failed: {result:?}"
        );

        let result2 = state.apply(minimal_valid_config(), 0);
        match result2 {
            ApplyResult::RevisionConflict { current_revision } => {
                assert_eq!(current_revision, 1);
            }
            other => panic!("expected RevisionConflict, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prepared_apply_keeps_config_and_hash_atomic_until_persistence_completes() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let initial_hash = *state.config_hash();

        let pending = match state.prepare_apply(minimal_valid_config(), 0) {
            ConfigApplyPreparation::Pending(pending) => pending,
            other => panic!("expected pending apply, got {other:?}"),
        };

        assert_eq!(state.revision(), 0);
        assert_eq!(*state.config_hash(), initial_hash);
        assert!(
            !config_path.exists(),
            "preparing an apply must not persist while state is locked"
        );

        let persistence = pending.persist();
        assert!(
            matches!(&persistence, ConfigPersistence::Persisted),
            "persistence should succeed: {persistence:?}"
        );
        assert!(config_path.exists(), "persistence should write config.toml");
        assert_eq!(state.revision(), 0);
        assert_eq!(
            *state.config_hash(),
            initial_hash,
            "in-memory config and hash remain unchanged until commit"
        );

        let result = state.finish_apply(*pending, persistence);
        assert!(
            matches!(result, ApplyResult::Applied { revision: 1, .. }),
            "commit should publish the persisted config atomically: {result:?}"
        );
        assert_ne!(*state.config_hash(), initial_hash);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn finish_apply_rejects_a_pending_revision_after_another_apply_wins() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let initial_hash = *state.config_hash();

        let pending = match state.prepare_apply(minimal_valid_config(), 0) {
            ConfigApplyPreparation::Pending(pending) => pending,
            other => panic!("expected pending apply, got {other:?}"),
        };
        let persistence = pending.persist();

        // Simulate a concurrent apply committing between preparation and this
        // pending apply's commit.
        state.revision = 1;

        let result = state.finish_apply(*pending, persistence);
        assert!(matches!(
            result,
            ApplyResult::RevisionConflict {
                current_revision: 1
            }
        ));
        assert_eq!(*state.config_hash(), initial_hash);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_concurrent_applies() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();

        let r1 = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(r1, ApplyResult::Applied { revision: 1, .. }),
            "first apply must succeed: {r1:?}"
        );

        let r2 = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(
                r2,
                ApplyResult::RevisionConflict {
                    current_revision: 1
                }
            ),
            "second apply with stale revision must conflict: {r2:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_revision_monotonic() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();

        let make_config = |model: &str| MeshConfig {
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
            models: vec![crate::plugin::ModelConfigEntry {
                model: model.to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            }],
            plugins: vec![],
            logging: Default::default(),
            extra: Default::default(),
        };

        assert_eq!(state.revision(), 0);
        state.apply(make_config("model-a.gguf"), 0);
        assert_eq!(state.revision(), 1);
        state.apply(make_config("model-b.gguf"), 1);
        assert_eq!(state.revision(), 2);
        state.apply(make_config("model-c.gguf"), 2);
        assert_eq!(state.revision(), 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_hash_changes_on_different_config() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();
        let initial_hash = *state.config_hash();

        let config_with_model = MeshConfig {
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
            models: vec![crate::plugin::ModelConfigEntry {
                model: "test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            }],
            plugins: vec![],
            logging: Default::default(),
            extra: Default::default(),
        };
        state.apply(config_with_model, 0);
        let new_hash = *state.config_hash();
        assert_ne!(
            initial_hash, new_hash,
            "hash must change when config changes"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_apply_preserves_nested_sections_and_updates_hash() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let first = representative_nested_config();
        let first_result = state.apply(first.clone(), 0);
        let first_hash = match first_result {
            ApplyResult::Applied {
                revision,
                hash,
                apply_mode,
                diagnostics,
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                assert!(diagnostics.is_empty());
                hash
            }
            other => panic!("expected Applied, got {other:?}"),
        };
        assert_representative_nested_fields(state.config());

        let persisted = ConfigState::load(&config_path).expect("reload persisted config");
        assert_representative_nested_fields(persisted.config());

        let mut changed = first;
        changed
            .models
            .first_mut()
            .expect("model")
            .advanced
            .get_or_insert_with(Default::default)
            .server
            .get_or_insert_with(Default::default)
            .alias = Some("model-alias-updated".to_string());

        let second_result = state.apply(changed, 1);
        match second_result {
            ApplyResult::Applied { revision, hash, .. } => {
                assert_eq!(revision, 2);
                assert_ne!(first_hash, hash, "nested field change must change hash");
            }
            other => panic!("expected Applied, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_load_propagates_invalid_toml_error() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, "this is [not valid toml !!!\n").expect("write bad toml");
        let result = ConfigState::load(&config_path);
        assert!(result.is_err(), "load must return Err on malformed TOML");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_load_nested_validation_error_is_stable() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(
            &config_path,
            r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
        )
        .expect("write invalid config");

        let error = match ConfigState::load(&config_path) {
            Ok(_) => panic!("load must fail"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(
            message.contains(
                "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
            ),
            "unexpected error: {message}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn runtime_config_diagnostics_transport() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let invalid: MeshConfig = toml::from_str(
            r#"version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
        )
        .expect("invalid fixture should still deserialize");

        match state.apply(invalid, 0) {
            ApplyResult::ValidationError { error, diagnostics } => {
                assert!(
                    error.contains(
                        "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
                    ),
                    "unexpected legacy error: {error}"
                );
                assert!(diagnostics.iter().any(|diagnostic| {
                    diagnostic.severity == ConfigDiagnosticSeverity::Error
                        && diagnostic
                            .message
                            .contains("reasoning_format must be one of")
                        && diagnostic
                            .path
                            .as_ref()
                            .map(|path| path.render())
                            .as_deref()
                            == Some("models[0].request_defaults.reasoning_format")
                        && diagnostic.help.is_none()
                }));
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[serial_test::serial]
    fn runtime_config_success_preserves_warning_diagnostics() {
        with_plugin_store(
            &[installed_plugin_metadata(
                "blackboard",
                Some(legacy_unvalidated_schema("blackboard")),
            )],
            || {
                let dir = test_dir();
                let config_path = dir.join("config.toml");
                let mut state = ConfigState::load(&config_path).expect("load");
                let config: MeshConfig = toml::from_str(
                    r#"
version = 1

[[plugin]]
name = "blackboard"

[plugin.settings]
arbitrary = "kept"
"#,
                )
                .expect("legacy plugin config should deserialize");

                match state.apply(config, 0) {
                    ApplyResult::Applied {
                        revision,
                        apply_mode,
                        diagnostics,
                        ..
                    } => {
                        assert_eq!(revision, 1);
                        assert_eq!(apply_mode, ConfigApplyMode::Staged);
                        assert!(diagnostics.iter().any(|diagnostic| {
                            diagnostic.code == ConfigDiagnosticCode::LegacyUnvalidatedConfig
                                && diagnostic.severity == ConfigDiagnosticSeverity::Warning
                                && diagnostic
                                    .canonical_path
                                    .as_ref()
                                    .map(|path| path.render())
                                    .as_deref()
                                    == Some("plugin.blackboard.settings")
                        }));
                    }
                    other => panic!("expected Applied with warning diagnostics, got {other:?}"),
                }

                std::fs::remove_dir_all(&dir).ok();
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn runtime_config_apply_legacy_plugin_schema_keeps_unknown_settings_but_rejects_bad_known_values()
     {
        with_plugin_store(
            &[installed_plugin_metadata(
                "blackboard",
                Some(strict_blackboard_schema("blackboard", true)),
            )],
            || {
                let dir = test_dir();
                let config_path = dir.join("config.toml");
                let mut state = ConfigState::load(&config_path).expect("load");
                let config: MeshConfig = toml::from_str(
                    r#"
version = 1

[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 0
mode = "mystery"
unknown = true
"#,
                )
                .expect("legacy plugin config should deserialize");

                match state.apply(config, 0) {
                    ApplyResult::ValidationError { error, diagnostics } => {
                        assert!(
                            !error.is_empty(),
                            "legacy error summary should not be empty"
                        );
                        assert!(diagnostics.iter().any(|diagnostic| {
                            diagnostic.code == ConfigDiagnosticCode::LegacyUnvalidatedConfig
                                && diagnostic.severity == ConfigDiagnosticSeverity::Warning
                                && diagnostic
                                    .canonical_path
                                    .as_ref()
                                    .map(|path| path.render())
                                    .as_deref()
                                    == Some("plugin.blackboard.settings")
                        }));
                        assert!(diagnostics.iter().any(|diagnostic| {
                            diagnostic.code == ConfigDiagnosticCode::InvalidValue
                                && diagnostic
                                    .canonical_path
                                    .as_ref()
                                    .map(|path| path.render())
                                    .as_deref()
                                    == Some("plugin.blackboard.settings.retention_days")
                        }));
                        assert!(diagnostics.iter().any(|diagnostic| {
                            diagnostic.code == ConfigDiagnosticCode::InvalidValue
                                && diagnostic
                                    .canonical_path
                                    .as_ref()
                                    .map(|path| path.render())
                                    .as_deref()
                                    == Some("plugin.blackboard.settings.mode")
                        }));
                        assert!(!diagnostics.iter().any(|diagnostic| {
                            diagnostic.code == ConfigDiagnosticCode::UnknownField
                                && diagnostic
                                    .canonical_path
                                    .as_ref()
                                    .map(|path| path.render())
                                    .as_deref()
                                    == Some("plugin.blackboard.settings.unknown")
                        }));
                    }
                    other => panic!("expected ValidationError, got {other:?}"),
                }

                std::fs::remove_dir_all(&dir).ok();
            },
        );
    }

    #[test]
    fn runtime_config_apply_accepts_schema_driven_valid_fixture() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let valid: MeshConfig =
            toml::from_str(CONTROL_FIXTURE_VALID).expect("valid fixture should deserialize");

        match state.apply(valid, 0) {
            ApplyResult::Applied { diagnostics, .. } => assert!(diagnostics.is_empty()),
            other => panic!("expected Applied, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn runtime_config_apply_matches_validator_signatures_for_schema_driven_invalid_fixture() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let invalid: MeshConfig =
            toml::from_str(CONTROL_FIXTURE_INVALID).expect("invalid fixture should deserialize");
        let expected = diagnostic_signatures(&validate_config_diagnostics(&invalid));

        match state.apply(invalid, 0) {
            ApplyResult::ValidationError { diagnostics, .. } => {
                assert_eq!(diagnostic_signatures(&diagnostics), expected);
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_load_malformed_nested_toml_still_errors() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        std::fs::write(
            &config_path,
            r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults
temperature = 0.2
"#,
        )
        .expect("write malformed config");

        let result = ConfigState::load(&config_path);
        assert!(
            result.is_err(),
            "load must return Err on malformed nested TOML"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_noop_apply_skips_disk_write() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let config_with_model = MeshConfig {
            version: Some(1),
            gpu: crate::plugin::GpuConfig {
                assignment: GpuAssignment::Auto,
                parallel: None,
            },
            mesh_requirements: Default::default(),
            owner_control: Default::default(),
            telemetry: Default::default(),
            defaults: None,
            runtime: Default::default(),
            models: vec![crate::plugin::ModelConfigEntry {
                model: "noop-test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
                gpu_id: None,
                parallel: None,
                cache_type_k: None,
                cache_type_v: None,
                batch: None,
                ubatch: None,
                flash_attention: None,
                ..Default::default()
            }],
            plugins: vec![],
            logging: Default::default(),
            extra: Default::default(),
        };

        let r1 = state.apply(config_with_model.clone(), 0);
        let rev_after_first = match r1 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(
                    apply_mode,
                    ConfigApplyMode::Staged,
                    "first apply must save to disk"
                );
                revision
            }
            other => panic!("expected Applied, got {other:?}"),
        };

        let r2 = state.apply(config_with_model.clone(), rev_after_first);
        match r2 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(
                    apply_mode,
                    ConfigApplyMode::Noop,
                    "no-op apply must not save to disk"
                );
                assert_eq!(
                    revision, rev_after_first,
                    "revision must not change on no-op"
                );
            }
            other => panic!("expected Applied with Noop apply_mode, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_telemetry_only_change_is_persisted_locally() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let base = minimal_valid_config();
        let r1 = state.apply(base.clone(), 0);
        let rev_after_first = match r1 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                revision
            }
            other => panic!("expected Applied, got {other:?}"),
        };

        let mut telemetry_only = base;
        telemetry_only.telemetry.enabled = Some(true);
        telemetry_only.telemetry.endpoint = Some("https://otel.example.com".to_string());

        let r2 = state.apply(telemetry_only, rev_after_first);
        match r2 {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(
                    apply_mode,
                    ConfigApplyMode::Staged,
                    "local-only telemetry changes must still be written to config.toml"
                );
                assert_eq!(revision, rev_after_first + 1);
            }
            other => panic!("expected Applied with Staged apply_mode, got {other:?}"),
        }

        let persisted = std::fs::read_to_string(&config_path).expect("persisted config");
        assert!(persisted.contains("https://otel.example.com"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_sidecar_path_derived_from_filename() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let sidecar = revision_sidecar_path(&config_path);
        let expected = dir.join("config.toml.revision");
        assert_eq!(
            sidecar, expected,
            "sidecar path must be config filename + .revision suffix"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_sidecar_migration_fallback() {
        let dir = test_dir();
        let legacy_path = dir.join("config-revision");
        std::fs::write(&legacy_path, "42\n").expect("write legacy revision");

        let config_path = dir.join("config.toml");
        let new_sidecar = revision_sidecar_path(&config_path);
        assert_ne!(
            new_sidecar, legacy_path,
            "new sidecar must differ from legacy"
        );

        let revision = read_revision(&new_sidecar);
        assert_eq!(
            revision, 42,
            "must fall back to legacy config-revision file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_apply_persists_integrated_fixture_sections_and_hashes_changes() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");
        let config: MeshConfig =
            toml::from_str(FULL_SURFACE_VALID_FIXTURE).expect("fixture parses");

        let first = state.apply(config.clone(), 0);
        let first_hash = match first {
            ApplyResult::Applied {
                revision,
                hash,
                apply_mode,
                diagnostics,
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
                let has_errors = diagnostics
                    .iter()
                    .any(|d| d.severity == mesh_llm_config::ConfigDiagnosticSeverity::Error);
                assert!(!has_errors, "unexpected error diagnostics: {diagnostics:?}");
                hash
            }
            other => panic!("expected Applied, got {other:?}"),
        };

        let persisted = std::fs::read_to_string(&config_path).expect("persisted config");
        assert!(persisted.contains("[models.skippy]"));
        assert!(persisted.contains("prefill_chunk_schedule = \"128,256,384\""));
        assert!(persisted.contains("reasoning_budget = 256"));

        let reloaded = ConfigState::load(&config_path).expect("reload config");
        assert_eq!(reloaded.config().models.len(), 2);
        assert_eq!(
            reloaded.config().models[0]
                .advanced
                .as_ref()
                .and_then(|advanced| advanced.server.as_ref())
                .and_then(|server| server.alias.as_deref()),
            Some("model-alias")
        );

        let mut changed = config;
        changed
            .defaults
            .as_mut()
            .and_then(|defaults| defaults.request_defaults.as_mut())
            .expect("request defaults")
            .temperature = Some(0.6);
        let second = state.apply(changed, 1);
        match second {
            ApplyResult::Applied { revision, hash, .. } => {
                assert_eq!(revision, 2);
                assert_ne!(first_hash, hash, "request-default change must update hash");
            }
            other => panic!("expected Applied, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn retention_and_replay_logging_changes_remain_staged_without_restart() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        // Apply baseline config first.
        let base_config = minimal_valid_config();
        match state.apply(base_config.clone(), 0) {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
            }
            other => panic!("expected Applied for baseline, got {other:?}"),
        }

        // When: change only DynamicApply fields. This config state does not
        // apply them to a running service; it stages the persisted revision.
        let mut changed = base_config;
        changed.logging.retention_ttl_secs = 72 * 3600;
        changed.logging.replay_capacity = 256;

        match state.apply(changed, 1) {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(revision, 2);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
            }
            other => panic!("expected staged apply for dynamic change, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn logging_change_classifier_requires_restart_for_every_static_setting() {
        let base = minimal_valid_config().logging;
        let changes: [LoggingConfigChange; 15] = [
            ("enabled", |config| config.enabled = !config.enabled),
            ("application_state_root", |config| {
                config.application_state_root = Some(PathBuf::from("logging-state"));
            }),
            ("summary_line_limit", |config| {
                config.summary_line_limit += 1
            }),
            ("event_buffer_size", |config| config.event_buffer_size += 1),
            ("queue_capacity", |config| config.queue_capacity += 1),
            ("artifact.capture_mode", |config| {
                config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
            }),
            ("artifact.byte_limit_bytes", |config| {
                config.artifact.byte_limit_bytes += 1;
            }),
            ("artifact.aggregate_limit_bytes", |config| {
                config.artifact.aggregate_limit_bytes += 1;
            }),
            ("export_limit_bytes", |config| {
                config.export_limit_bytes += 1
            }),
            ("cleanup_cadence_secs", |config| {
                config.cleanup_cadence_secs += 1
            }),
            ("webhook.enabled", |config| {
                config.webhook.enabled = !config.webhook.enabled
            }),
            ("webhook.url", |config| {
                config.webhook.url = Some("https://example.test/logs".into());
            }),
            ("webhook.max_attempts", |config| {
                config.webhook.max_attempts += 1
            }),
            ("webhook.timeout_secs", |config| {
                config.webhook.timeout_secs += 1
            }),
            ("webhook.dead_letter_retention_secs", |config| {
                config.webhook.dead_letter_retention_secs += 1;
            }),
        ];

        for (name, change) in changes {
            let mut changed = base.clone();
            change(&mut changed);
            assert!(
                logging_changes_require_restart(&base, &changed),
                "logging.{name} must be restart-required"
            );
        }

        let dynamic_changes: [LoggingConfigChange; 2] = [
            (
                "retention_ttl_secs",
                (|config: &mut LoggingConfig| config.retention_ttl_secs += 1)
                    as fn(&mut LoggingConfig),
            ),
            (
                "replay_capacity",
                (|config: &mut LoggingConfig| config.replay_capacity += 1)
                    as fn(&mut LoggingConfig),
            ),
        ];
        for (name, change) in dynamic_changes {
            let mut changed = base.clone();
            change(&mut changed);
            assert!(
                !logging_changes_require_restart(&base, &changed),
                "logging.{name} must remain dynamically applicable"
            );
        }

        assert!(
            field_requires_restart("unsupported_future_setting"),
            "unknown logging settings must not accidentally receive live-apply semantics"
        );
    }

    #[test]
    fn dynamic_limit_change_combined_with_static_change_stays_restart_required() {
        let old = minimal_valid_config().logging;
        let mut new = old.clone();
        new.retention_ttl_secs += 1;
        new.replay_capacity += 1;
        new.queue_capacity += 1;

        assert!(
            logging_changes_require_restart(&old, &new),
            "a static logging change must not be hidden by an earlier dynamic change"
        );
    }

    #[test]
    fn dynamic_logging_change_does_not_mask_later_nested_static_change() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        let base_config = minimal_valid_config();
        match state.apply(base_config.clone(), 0) {
            ApplyResult::Applied { revision, .. } => assert_eq!(revision, 1),
            other => panic!("expected Applied for baseline, got {other:?}"),
        }

        let mut changed = base_config;
        changed.logging.retention_ttl_secs += 3600;
        changed.logging.artifact.byte_limit_bytes += 1024;

        match state.apply(changed, 1) {
            ApplyResult::AppliedWithRestartRequired { revision, .. } => {
                assert_eq!(revision, 2);
            }
            other => panic!(
                "expected AppliedWithRestartRequired when dynamic and nested static fields change, got {other:?}"
            ),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn static_logging_changes_return_restart_required_from_config_state() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).expect("load");

        // Apply baseline config first.
        let base_config = minimal_valid_config();
        match state.apply(base_config.clone(), 0) {
            ApplyResult::Applied {
                revision,
                apply_mode,
                ..
            } => {
                assert_eq!(revision, 1);
                assert_eq!(apply_mode, ConfigApplyMode::Staged);
            }
            other => panic!("expected Applied for baseline, got {other:?}"),
        }

        let changes: [MeshConfigChange; 4] = [
            ("enabled", |config: &mut MeshConfig| {
                config.logging.enabled = !config.logging.enabled;
            }),
            ("queue_capacity", |config: &mut MeshConfig| {
                config.logging.queue_capacity += 1;
            }),
            ("cleanup_cadence_secs", |config: &mut MeshConfig| {
                config.logging.cleanup_cadence_secs += 1;
            }),
            ("artifact.capture_mode", |config: &mut MeshConfig| {
                config.logging.artifact.capture_mode =
                    mesh_llm_config::CaptureMode::RedactedArtifacts;
            }),
        ];
        for (index, (change_name, change)) in changes.into_iter().enumerate() {
            let mut changed = state.config().clone();
            change(&mut changed);
            match state.apply(changed, index as u64 + 1) {
                ApplyResult::AppliedWithRestartRequired { revision, .. } => {
                    assert_eq!(revision, index as u64 + 2, "{change_name}");
                }
                other => {
                    panic!("expected AppliedWithRestartRequired for {change_name}, got {other:?}")
                }
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
