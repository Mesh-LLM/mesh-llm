use anyhow::Result;
use mesh_llm_config::{
    ConfigDiagnostic, ConfigDiagnosticSeverity, ConfigPath, KvDiskTierConfig, LoggingConfig,
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

use super::live_request_defaults::{
    self, AppliedRequestDefaults, LiveRequestDefaults, RequestDefaultsBackends,
    ScopedServingBackends,
};
use super::operational_logging::{
    ConfigDiagnosticsOutcome, ConfigOperationalEvent, record_config_operational_event,
};

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
    kv_disk_requires_restart: bool,
    dynamic_kv_disk_changed: bool,
    /// This apply changes request defaults, so models are swapped (or at least
    /// tried) and a failed persist has to put them back.
    request_defaults_changed: bool,
    /// Something outside the settings this apply can take live changed, so the
    /// saved config stays staged even when every live part took effect.
    changes_staged_settings: bool,
    live_request_defaults: std::sync::Arc<LiveRequestDefaults>,
    /// What the live request-defaults swap replaced, for a failed persist.
    request_defaults_undo: std::sync::Mutex<
        Option<
            AppliedRequestDefaults<<ScopedServingBackends as RequestDefaultsBackends>::Previous>,
        >,
    >,
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

    pub(crate) fn apply_live_kv_disk(&self) -> Option<super::kv_disk_config::KvDiskLiveRollback> {
        if !self.dynamic_kv_disk_changed {
            return None;
        }
        match super::kv_disk_config::apply_live_kv_disk_limits(&self.config) {
            Ok(rollback) => Some(rollback),
            Err(error) => {
                tracing::warn!(%error, "disk prompt-cache runtime unavailable; retaining limits as staged configuration");
                None
            }
        }
    }

    /// Whether this apply changes request defaults, so models were swapped (or
    /// at least tried) and a failed persist has to put them back.
    fn changes_request_defaults(&self) -> bool {
        self.request_defaults_changed
    }

    pub(crate) fn apply_live_request_defaults(&self) -> bool {
        if !self.changes_request_defaults() {
            return false;
        }
        let backends = self.live_request_defaults.serving_backends();
        let applied = self.live_request_defaults.apply(&self.config, &backends);
        let all_applied = applied.all_applied;
        *self
            .request_defaults_undo
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(applied);
        if !all_applied {
            tracing::warn!(
                "request defaults could not be applied to every loaded model; retaining them as staged configuration"
            );
        }
        all_applied
    }

    pub(crate) fn restore_live_request_defaults(&self) {
        let undo = self
            .request_defaults_undo
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(applied) = undo {
            self.live_request_defaults
                .undo(applied, &self.live_request_defaults.serving_backends());
        }
    }

    /// Whether every setting this apply changes took effect live: each live
    /// part that was attempted succeeded and nothing else changed.
    fn fully_live(
        &self,
        logging_applied: bool,
        kv_disk_applied: bool,
        defaults_applied: bool,
    ) -> bool {
        let attempted = self.dynamic_logging_only
            || self.dynamic_kv_disk_changed
            || self.request_defaults_changed;
        attempted
            && (!self.dynamic_logging_only || logging_applied)
            && (!self.dynamic_kv_disk_changed || kv_disk_applied)
            && (!self.request_defaults_changed || defaults_applied)
            && !self.changes_staged_settings
    }

    fn applied_result(&self, apply_mode: ConfigApplyMode) -> ApplyResult {
        if self.logging_requires_restart || self.kv_disk_requires_restart {
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
    /// The request-defaults tracker of the node this state belongs to.
    live_request_defaults: std::sync::Arc<LiveRequestDefaults>,
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
            "retention_max_rows",
            old.retention_max_rows != new.retention_max_rows,
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

/// Whether the two configs differ in anything beyond the settings an apply can
/// take live (`attempted_*`): those compare equal here because their outcome is
/// judged separately, so what is left over takes effect only after a restart or
/// reload and keeps the apply staged.
fn changes_outside_live_settings(
    old: &MeshConfig,
    new: &MeshConfig,
    logging_limits: bool,
    kv_disk_limits: bool,
    request_defaults: bool,
) -> bool {
    let without_live_settings = |config: &MeshConfig| {
        let mut value = serde_json::to_value(config).unwrap_or_default();
        if request_defaults {
            if let Some(defaults) = value.get_mut("defaults").and_then(|v| v.as_object_mut()) {
                defaults.remove("request_defaults");
            }
            for model in value
                .get_mut("models")
                .and_then(|v| v.as_array_mut())
                .into_iter()
                .flatten()
            {
                if let Some(model) = model.as_object_mut() {
                    model.remove("request_defaults");
                }
            }
        }
        if logging_limits {
            remove_keys(
                &mut value,
                &["logging"],
                &["retention_ttl_secs", "replay_capacity"],
            );
        }
        if kv_disk_limits {
            remove_keys(
                &mut value,
                &["runtime", "kv_cache", "disk"],
                &["budget_mib", "minimum_free_mib"],
            );
        }
        // A section that only holds unset values is the same as an absent one,
        // and the file-format `version` marker configures nothing at runtime.
        if let Some(root) = value.as_object_mut() {
            root.remove("version");
        }
        prune_unset(&mut value);
        value
    };
    without_live_settings(old) != without_live_settings(new)
}

/// Drop `null` values and the objects left empty without them, so a section
/// that sets nothing compares equal to a missing one.
fn prune_unset(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.values_mut().for_each(prune_unset);
            map.retain(|_, child| match child {
                serde_json::Value::Null => false,
                serde_json::Value::Object(inner) => !inner.is_empty(),
                _ => true,
            });
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(prune_unset),
        _ => {}
    }
}

/// Remove `keys` from the object at `path` in a JSON value, if it is there.
fn remove_keys(value: &mut serde_json::Value, path: &[&str], keys: &[&str]) {
    let mut current = value;
    for segment in path {
        let Some(next) = current.get_mut(*segment) else {
            return;
        };
        current = next;
    }
    if let Some(object) = current.as_object_mut() {
        for key in keys {
            object.remove(*key);
        }
    }
}

fn kv_disk_changes_require_restart(old: &KvDiskTierConfig, new: &KvDiskTierConfig) -> bool {
    old.mode != new.mode || old.directory != new.directory || old.codec != new.codec
}

fn kv_disk_dynamic_limits_changed(old: &KvDiskTierConfig, new: &KvDiskTierConfig) -> bool {
    old.budget_mib != new.budget_mib || old.minimum_free_mib != new.minimum_free_mib
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
            live_request_defaults: std::sync::Arc::new(LiveRequestDefaults::new()),
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
            live_request_defaults: std::sync::Arc::new(LiveRequestDefaults::new()),
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

    pub(crate) fn live_request_defaults(&self) -> std::sync::Arc<LiveRequestDefaults> {
        std::sync::Arc::clone(&self.live_request_defaults)
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
        mut new_config: MeshConfig,
        expected_revision: u64,
    ) -> ConfigApplyPreparation {
        record_config_operational_event(ConfigOperationalEvent::ApplyStarted);
        new_config.source_directory = self.config.source_directory.clone().or_else(|| {
            std::path::absolute(&self.config_path)
                .ok()
                .and_then(|path| path.parent().map(Path::to_path_buf))
        });
        let raw_toml = config_to_toml(&new_config).ok();
        let diagnostics = validate_config_diagnostics_with_installed_plugin_schemas(
            &new_config,
            raw_toml.as_deref(),
        );
        record_config_operational_event(ConfigOperationalEvent::Diagnostics(
            ConfigDiagnosticsOutcome::from_diagnostics(&diagnostics),
        ));
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        {
            record_config_operational_event(ConfigOperationalEvent::ApplyRejected);
            return ConfigApplyPreparation::Immediate(ApplyResult::ValidationError {
                error: legacy_validation_error_text(&diagnostics),
                diagnostics,
            });
        }

        if expected_revision != self.revision {
            record_config_operational_event(ConfigOperationalEvent::ApplyRejected);
            return ConfigApplyPreparation::Immediate(ApplyResult::RevisionConflict {
                current_revision: self.revision,
            });
        }

        let proto = mesh_config_to_proto(&new_config);
        let new_hash = canonical_config_hash(&proto);
        let new_write_hash = local_config_write_hash(&new_config);

        if new_write_hash == self.last_write_config_hash {
            record_config_operational_event(ConfigOperationalEvent::ApplyAccepted);
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
        let old_kv_disk = &self.config.runtime.kv_cache.disk;
        let new_kv_disk = &new_config.runtime.kv_cache.disk;
        let kv_disk_requires_restart = kv_disk_changes_require_restart(old_kv_disk, new_kv_disk);
        let dynamic_kv_disk_changed = kv_disk_dynamic_limits_changed(old_kv_disk, new_kv_disk);

        let request_defaults_changed =
            live_request_defaults::request_defaults_changed(&self.config, &new_config);
        let changes_staged_settings = changes_outside_live_settings(
            &self.config,
            &new_config,
            dynamic_logging_only,
            dynamic_kv_disk_changed,
            request_defaults_changed,
        );

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
            kv_disk_requires_restart,
            dynamic_kv_disk_changed,
            request_defaults_changed,
            changes_staged_settings,
            live_request_defaults: std::sync::Arc::clone(&self.live_request_defaults),
            request_defaults_undo: std::sync::Mutex::new(None),
        }))
    }

    pub(crate) fn finish_apply(
        &mut self,
        pending: PendingConfigApply,
        persistence: ConfigPersistence,
    ) -> ApplyResult {
        match persistence {
            ConfigPersistence::PersistError(error) => {
                record_config_operational_event(ConfigOperationalEvent::ApplyRejected);
                ApplyResult::PersistError(error)
            }
            ConfigPersistence::Persisted | ConfigPersistence::RevisionTrackingError(_) => {
                if self.revision.checked_add(1) != Some(pending.revision) {
                    record_config_operational_event(ConfigOperationalEvent::ApplyRejected);
                    return ApplyResult::RevisionConflict {
                        current_revision: self.revision,
                    };
                }
                self.config = pending.config.clone();
                self.config_hash = pending.hash;
                self.last_write_config_hash = pending.write_hash;
                self.revision = pending.revision;

                record_config_operational_event(ConfigOperationalEvent::ApplyAccepted);
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
    let live_kv_disk_rollback = pending.apply_live_kv_disk();
    let live_kv_disk_applied = live_kv_disk_rollback.is_some();
    let request_defaults_attempted = pending.changes_request_defaults();
    let live_request_defaults_applied = pending.apply_live_request_defaults();
    let persistence = persist(&pending);
    if live_logging_applied && matches!(&persistence, ConfigPersistence::PersistError(_)) {
        // A persistence failure after a live apply is rare, but restore the
        // prior runtime settings before exposing the failure so the
        // configuration and service do not diverge.
        pending.restore_live_logging();
    }
    if matches!(&persistence, ConfigPersistence::PersistError(_))
        && let Some(rollback) = live_kv_disk_rollback
    {
        super::kv_disk_config::restore_live_kv_disk_limits(rollback);
    }
    if request_defaults_attempted && matches!(&persistence, ConfigPersistence::PersistError(_)) {
        pending.restore_live_request_defaults();
    }
    let fully_live = pending.fully_live(
        live_logging_applied,
        live_kv_disk_applied,
        live_request_defaults_applied,
    );
    let result = finish(pending, persistence);

    match result {
        ApplyResult::Applied {
            revision,
            hash,
            diagnostics,
            ..
        } if fully_live => ApplyResult::Applied {
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
#[path = "config_state_tests.rs"]
mod tests;
