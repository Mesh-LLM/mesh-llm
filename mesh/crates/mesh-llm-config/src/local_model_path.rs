//! The checks `mesh-llm serve` runs on a configured `hardware.model_path`
//! before loading the model, shared with `mesh-llm config validate` so that a
//! config which validates on a host also starts on it.

use crate::MeshConfig;
use crate::diagnostic::{ConfigDiagnostic, ConfigDiagnosticSeverity};
use crate::validation_support::validation_diagnostic;
use std::path::Path;

/// Why a configured model path cannot be served from this host.
#[derive(Debug)]
pub enum LocalModelPathError {
    /// A relative path would depend on the directory `serve` starts in.
    NotAbsolute,
    /// The path cannot be read, usually because nothing is there.
    Unavailable(std::io::Error),
    /// A symlink, a directory, or anything else that is not a regular file.
    NotRegularFile,
}

/// Check a configured `hardware.model_path` the way `serve` does at startup.
pub fn check_local_model_path(path: &Path) -> Result<(), LocalModelPathError> {
    if !path.is_absolute() {
        return Err(LocalModelPathError::NotAbsolute);
    }
    let metadata = std::fs::symlink_metadata(path).map_err(LocalModelPathError::Unavailable)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(LocalModelPathError::NotRegularFile);
    }
    Ok(())
}

/// One warning for each `hardware.model_path` that `serve` would refuse on
/// this host. Warnings, not errors: a config can be validated here and applied
/// to another node through owner-control, where the path may be right. Kept
/// out of `validate_config_diagnostics`, which checks the document alone and
/// never reads the filesystem.
pub fn local_model_path_diagnostics(config: &MeshConfig) -> Vec<ConfigDiagnostic> {
    let mut configured = Vec::new();
    let default_path = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref())
        .and_then(|hardware| hardware.model_path.as_deref());
    let model_path = |index: usize| {
        config.models[index]
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.model_path.as_deref())
    };
    // `serve` only reads the default for models that don't set their own.
    if let Some(path) = default_path
        && (0..config.models.len()).any(|index| model_path(index).is_none())
    {
        configured.push(("defaults.hardware.model_path".to_string(), path));
    }
    for index in 0..config.models.len() {
        if let Some(path) = model_path(index) {
            configured.push((format!("models[{index}].hardware.model_path"), path));
        }
    }

    configured
        .into_iter()
        .filter_map(|(raw_path, value)| {
            let path = Path::new(value);
            let message = match check_local_model_path(path) {
                Ok(()) => return None,
                Err(LocalModelPathError::NotAbsolute) => {
                    format!("{raw_path} must be absolute: {}", path.display())
                }
                Err(LocalModelPathError::Unavailable(error)) => {
                    format!("{raw_path} is unavailable: {} ({error})", path.display())
                }
                Err(LocalModelPathError::NotRegularFile) => {
                    format!("{raw_path} must be a non-symlink file: {}", path.display())
                }
            };
            let mut diagnostic = validation_diagnostic(&raw_path, message).with_help(
                "mesh-llm serve refuses to start with this path on the machine that ran this check",
            );
            diagnostic.severity = ConfigDiagnosticSeverity::Warning;
            Some(diagnostic)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_config_toml_structural;

    fn config(raw: &str) -> MeshConfig {
        parse_config_toml_structural(raw).expect("test config parses")
    }

    fn model_with_path(path: &str) -> String {
        format!("[[models]]\nmodel = \"named-model\"\n[models.hardware]\nmodel_path = '{path}'\n")
    }

    fn paths_and_messages(config: &MeshConfig) -> Vec<(String, String)> {
        local_model_path_diagnostics(config)
            .into_iter()
            .map(|diagnostic| {
                assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Warning);
                (
                    diagnostic.path.expect("diagnostic has a path").render(),
                    diagnostic.message,
                )
            })
            .collect()
    }

    #[test]
    fn regular_absolute_file_passes() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"GGUF").unwrap();

        let config = config(&model_with_path(&model.display().to_string()));

        assert!(local_model_path_diagnostics(&config).is_empty());
    }

    #[test]
    fn relative_missing_and_directory_paths_are_errors_on_their_own_rows() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("model.gguf");
        std::fs::write(&good, b"GGUF").unwrap();
        let missing = dir.path().join("does-not-exist.gguf");
        let raw = [
            model_with_path(&good.display().to_string()),
            model_with_path("models/relative.gguf"),
            model_with_path(&missing.display().to_string()),
            model_with_path(&dir.path().display().to_string()),
        ]
        .concat();

        let found = paths_and_messages(&config(&raw));

        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(found[0].0, "models[1].hardware.model_path");
        assert!(found[0].1.contains("must be absolute"), "{found:?}");
        assert_eq!(found[1].0, "models[2].hardware.model_path");
        assert!(found[1].1.contains("is unavailable"), "{found:?}");
        assert_eq!(found[2].0, "models[3].hardware.model_path");
        assert!(
            found[2].1.contains("must be a non-symlink file"),
            "{found:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_a_valid_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model.gguf");
        std::fs::write(&target, b"GGUF").unwrap();
        let link = dir.path().join("link.gguf");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let found = paths_and_messages(&config(&model_with_path(&link.display().to_string())));

        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].1.contains("must be a non-symlink file"),
            "{found:?}"
        );
    }

    #[test]
    fn default_path_is_checked_only_when_a_model_inherits_it() {
        let inherited = "[defaults.hardware]\nmodel_path = 'models/default.gguf'\n\n[[models]]\nmodel = \"named-model\"\n";
        let found = paths_and_messages(&config(inherited));
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].0, "defaults.hardware.model_path");

        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let overridden = format!(
            "[defaults.hardware]\nmodel_path = 'models/default.gguf'\n\n{}",
            model_with_path(&model.display().to_string())
        );
        assert!(local_model_path_diagnostics(&config(&overridden)).is_empty());
    }
}
