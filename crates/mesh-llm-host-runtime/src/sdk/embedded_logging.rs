//! Logging initialization shared by the embedded SDK startup boundary.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Validate an embedded configuration and retain the exact validated image.
///
/// Embedded startup crosses a worker-thread boundary and several subsystems
/// load configuration independently. Snapshotting the bytes after validation
/// ensures every startup load observes the image accepted by the SDK caller,
/// even if the original pathname changes while the worker starts.
pub(crate) fn snapshot_validated_config(config_path: Option<&Path>) -> Result<NamedTempFile> {
    let resolved_path = crate::plugin::config_path(config_path)?;
    let raw = if resolved_path.exists() {
        std::fs::read(&resolved_path)
            .with_context(|| format!("Failed to read config {}", resolved_path.display()))?
    } else {
        Vec::new()
    };
    let raw_text = std::str::from_utf8(&raw)
        .with_context(|| format!("Invalid config {}", resolved_path.display()))?;
    crate::plugin::parse_config_toml(raw_text)
        .with_context(|| format!("Invalid config {}", resolved_path.display()))?;

    let mut snapshot = NamedTempFile::new().context("create embedded config snapshot")?;
    snapshot
        .write_all(&raw)
        .context("write embedded config snapshot")?;
    snapshot.flush().context("flush embedded config snapshot")?;
    Ok(snapshot)
}

pub(crate) fn snapshot_path(snapshot: &NamedTempFile) -> PathBuf {
    snapshot.path().to_path_buf()
}

/// Install the host-owned logging foundation for an embedded runtime.
///
/// The config loader performs the same schema and plugin validation as the
/// non-embedded path. `initialize_logging_foundation` replaces the
/// process-local handles, so a later embedded start cannot retain logging
/// resources resolved from an earlier configuration.
pub(crate) async fn initialize_embedded_logging(config_path: Option<&Path>) -> Result<()> {
    let config = crate::plugin::load_config(config_path)?;
    crate::initialize_logging_foundation(&config.logging).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    fn write_logging_config(path: &Path, enabled: bool, root: &Path) {
        let root = root
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        std::fs::write(
            path,
            format!("[logging]\nenabled = {enabled}\napplication_state_root = \"{root}\"\n"),
        )
        .expect("write embedded logging config");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn validated_config_snapshot_is_stable_when_source_changes() {
        let temporary_directory = tempfile::tempdir().expect("temporary config directory");
        let config_path = temporary_directory.path().join("mesh-llm.toml");
        let validated_root = temporary_directory.path().join("validated-logging");
        let swapped_root = temporary_directory.path().join("swapped-logging");
        write_logging_config(&config_path, true, &validated_root);

        let snapshot = snapshot_validated_config(Some(&config_path))
            .expect("snapshot validated embedded config");
        write_logging_config(&config_path, true, &swapped_root);
        initialize_embedded_logging(Some(snapshot.path()))
            .await
            .expect("initialize logging from validated snapshot");

        assert_eq!(
            crate::logging_foundation()
                .expect("installed foundation")
                .app_state_root(),
            validated_root
        );
        assert!(!swapped_root.exists());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn embedded_entrypoint_migrates_configured_logging_store() {
        let temporary_directory = tempfile::tempdir().expect("temporary config directory");
        let config_path = temporary_directory.path().join("mesh-llm.toml");
        let configured_root = temporary_directory.path().join("embedded-logging");
        write_logging_config(&config_path, true, &configured_root);

        initialize_embedded_logging(Some(&config_path))
            .await
            .expect("initialize embedded logging");

        let foundation = crate::logging_foundation().expect("installed foundation");
        assert!(foundation.is_healthy());
        assert_eq!(foundation.app_state_root(), configured_root);
        assert!(foundation.store_dir().join("log_store.db").is_file());
        assert!(
            crate::logging_runtime_state()
                .expect("installed logging runtime state")
                .health()
                .metadata_available
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn embedded_entrypoint_honors_disabled_logging_without_creating_files() {
        let temporary_directory = tempfile::tempdir().expect("temporary config directory");
        let config_path = temporary_directory.path().join("mesh-llm.toml");
        let configured_root: PathBuf = temporary_directory.path().join("disabled-logging");
        write_logging_config(&config_path, false, &configured_root);

        initialize_embedded_logging(Some(&config_path))
            .await
            .expect("initialize disabled logging");

        assert!(
            !configured_root.exists(),
            "disabled embedded logging must not create its configured root"
        );
        assert!(
            !crate::logging_foundation()
                .expect("installed disabled foundation")
                .is_healthy()
        );
        assert!(
            !crate::logging_runtime_state()
                .expect("installed disabled logging state")
                .health()
                .metadata_available
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn embedded_replacement_retires_a_captured_runtime_state_before_reinstalling() {
        let temporary_directory = tempfile::tempdir().expect("temporary config directory");
        let first_path = temporary_directory.path().join("first.toml");
        let second_path = temporary_directory.path().join("second.toml");
        let first_root = temporary_directory.path().join("first-logging");
        let second_root = temporary_directory.path().join("second-logging");
        write_logging_config(&first_path, true, &first_root);
        write_logging_config(&second_path, true, &second_root);

        initialize_embedded_logging(Some(&first_path))
            .await
            .expect("initialize first embedded logging");
        let retired_state = crate::logging_runtime_state().expect("first runtime state");
        let retired_service = retired_state
            .start_persistence_worker()
            .await
            .expect("start first worker");
        assert!(retired_service.is_spawned());

        initialize_embedded_logging(Some(&second_path))
            .await
            .expect("replace embedded logging");

        assert!(retired_state.is_retired());
        assert!(retired_state.start_persistence_worker().await.is_none());
        assert!(!retired_service.is_startable());
        assert!(!retired_service.spawn());
        assert!(!retired_service.is_spawned());
        assert_eq!(
            crate::logging_foundation()
                .expect("replacement foundation")
                .app_state_root(),
            second_root
        );

        crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
            enabled: false,
            ..Default::default()
        })
        .await;
    }
}
