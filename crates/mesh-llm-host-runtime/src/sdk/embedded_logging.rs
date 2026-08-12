//! Logging initialization shared by the embedded SDK startup boundary.

use anyhow::Result;
use std::path::Path;

/// Validate an embedded configuration before starting its worker thread.
///
/// This keeps configuration errors attributable to the SDK caller rather than
/// reducing them to an early worker-thread exit during readiness polling.
pub(crate) fn validate_embedded_config(config_path: Option<&Path>) -> Result<()> {
    crate::plugin::load_config(config_path).map(|_| ())
}

/// Install the host-owned logging foundation for an embedded runtime.
///
/// The config loader performs the same schema and plugin validation as the
/// non-embedded path. `initialize_logging_foundation` replaces the
/// process-local handles, so a later embedded start cannot retain logging
/// resources resolved from an earlier configuration.
pub(crate) fn initialize_embedded_logging(config_path: Option<&Path>) -> Result<()> {
    let config = crate::plugin::load_config(config_path)?;
    crate::initialize_logging_foundation(&config.logging);
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

    #[test]
    #[serial_test::serial]
    fn embedded_entrypoint_migrates_configured_logging_store() {
        let temporary_directory = tempfile::tempdir().expect("temporary config directory");
        let config_path = temporary_directory.path().join("mesh-llm.toml");
        let configured_root = temporary_directory.path().join("embedded-logging");
        write_logging_config(&config_path, true, &configured_root);

        initialize_embedded_logging(Some(&config_path)).expect("initialize embedded logging");

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

    #[test]
    #[serial_test::serial]
    fn embedded_entrypoint_honors_disabled_logging_without_creating_files() {
        let temporary_directory = tempfile::tempdir().expect("temporary config directory");
        let config_path = temporary_directory.path().join("mesh-llm.toml");
        let configured_root: PathBuf = temporary_directory.path().join("disabled-logging");
        write_logging_config(&config_path, false, &configured_root);

        initialize_embedded_logging(Some(&config_path)).expect("initialize disabled logging");

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
}
