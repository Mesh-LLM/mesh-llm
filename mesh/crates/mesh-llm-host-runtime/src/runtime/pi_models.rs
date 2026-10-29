//! The `mesh` provider `serve` keeps in Pi's `~/.pi/agent/models.json`.

use mesh_llm_events::{OutputEvent, emit_event};
use std::path::{Path, PathBuf};

/// Update ~/.pi/agent/models.json to include a "mesh" provider.
pub(super) fn update_pi_models_json(model_id: &str, port: u16) {
    let Some(home) = dirs::home_dir() else { return };
    update_models_json(&home.join(".pi/agent/models.json"), model_id, port);
}

/// Sets the `mesh` provider in the file at `path`. A file that can't be read,
/// doesn't parse or isn't a JSON object belongs to the user and stays as it
/// is.
fn update_models_json(path: &Path, model_id: &str, port: u16) {
    let Some(mut root) = load(path) else { return };
    if let Err(reason) = set_mesh_provider(&mut root, mesh_provider(model_id, port)) {
        warn_not_updated(path, reason);
        return;
    }
    let written = serde_json::to_vec_pretty(&root)
        .map_err(std::io::Error::other)
        .and_then(|json| replace(path, &json));
    if let Err(e) = written {
        warn_not_updated(path, e);
    }
}

fn warn_not_updated(path: &Path, reason: impl std::fmt::Display) {
    let _ = emit_event(OutputEvent::Warning {
        message: format!("Pi's {} was not updated: {reason}", path.display()),
        context: None,
    });
}

fn set_mesh_provider(
    root: &mut serde_json::Value,
    provider: serde_json::Value,
) -> Result<(), &'static str> {
    let providers = root
        .as_object_mut()
        .ok_or("it doesn't hold a JSON object")?
        .entry("providers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .ok_or("its providers aren't a JSON object")?;
    providers.insert("mesh".to_string(), provider);
    Ok(())
}

/// The file's JSON, `{}` when there is no file, `None` when it exists but
/// can't be used.
fn load(path: &Path) -> Option<serde_json::Value> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(serde_json::json!({})),
        Err(e) => {
            warn_not_updated(path, e);
            return None;
        }
    };
    serde_json::from_str(&content)
        .inspect_err(|e| warn_not_updated(path, e))
        .ok()
}

/// Replaces the file at `path`, or the file a symlink there points to, with
/// `contents`. The new file gets the old one's permissions, or on Windows its
/// DACL, before anything is written, and the old file stays in place if
/// anything fails.
fn replace(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    mesh_llm_config::replace_file_keeping_access(&write_target(path)?, contents)
}

/// `path`, or the file it links to.
fn write_target(path: &Path) -> std::io::Result<PathBuf> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => std::fs::canonicalize(path),
        _ => Ok(path.to_path_buf()),
    }
}

fn mesh_provider(model_id: &str, port: u16) -> serde_json::Value {
    serde_json::json!({
        "baseUrl": format!("http://localhost:{port}/v1"),
        "api": "openai-completions",
        "apiKey": "mesh",
        "models": [{
            "id": model_id,
            "name": model_id,
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "compat": {
                "supportsUsageInStreaming": false,
                "maxTokensField": "max_tokens",
                "supportsDeveloperRole": false
            }
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(path: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn adds_the_mesh_provider_and_keeps_the_others() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(
            &path,
            r#"{"providers": {"my-ollama": {"baseUrl": "http://localhost:11434/v1"}}, "other": 1}"#,
        )
        .unwrap();

        update_models_json(&path, "Qwen3-8B", 9337);

        let root = read(&path);
        assert_eq!(
            root["providers"]["my-ollama"]["baseUrl"],
            "http://localhost:11434/v1"
        );
        assert_eq!(root["other"], 1);
        assert_eq!(
            root["providers"]["mesh"]["baseUrl"],
            "http://localhost:9337/v1"
        );
        assert_eq!(root["providers"]["mesh"]["models"][0]["id"], "Qwen3-8B");
    }

    #[test]
    fn creates_the_file_when_there_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agent/models.json");

        update_models_json(&path, "Qwen3-8B", 9337);

        assert_eq!(read(&path)["providers"]["mesh"]["apiKey"], "mesh");
    }

    #[test]
    fn leaves_a_file_that_does_not_parse_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        let content = r#"{"providers": {"my-ollama": {},}}"#;
        std::fs::write(&path, content).unwrap();

        update_models_json(&path, "Qwen3-8B", 9337);

        assert_eq!(std::fs::read_to_string(&path).unwrap(), content);
    }

    #[cfg(unix)]
    #[test]
    fn keeps_the_file_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{}").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        update_models_json(&path, "Qwen3-8B", 9337);

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        assert_eq!(read(&path)["providers"]["mesh"]["apiKey"], "mesh");
    }

    #[cfg(unix)]
    #[test]
    fn updates_the_file_a_symlink_points_to() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("dotfiles-models.json");
        std::fs::write(&target, r#"{"providers": {"my-ollama": {}}}"#).unwrap();
        let link = dir.path().join("models.json");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        update_models_json(&link, "Qwen3-8B", 9337);

        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let root = read(&target);
        assert!(root["providers"]["my-ollama"].is_object());
        assert_eq!(root["providers"]["mesh"]["apiKey"], "mesh");
    }

    #[test]
    fn leaves_a_file_without_a_json_object_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        for content in [r#"["not", "an", "object"]"#, r#"{"providers": []}"#] {
            std::fs::write(&path, content).unwrap();

            update_models_json(&path, "Qwen3-8B", 9337);

            assert_eq!(std::fs::read_to_string(&path).unwrap(), content);
        }
    }
}
