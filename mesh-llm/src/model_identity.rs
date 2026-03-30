use std::path::Path;

pub fn resolved_model_name(path: &Path) -> String {
    if path.is_dir() {
        if let Some(config_name) = model_name_from_config(path) {
            return config_name;
        }
        return path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
    }

    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    crate::router::strip_split_suffix_owned(&stem)
}

fn model_name_from_config(path: &Path) -> Option<String> {
    let config_path = path.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    let raw_name = config.get("_name_or_path")?.as_str()?.trim();
    if raw_name.is_empty() {
        return None;
    }

    raw_name
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .map(str::trim)
        .filter(|name| !name.is_empty() && *name != ".")
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::resolved_model_name;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mesh-llm-{prefix}-{unique}"))
    }

    #[test]
    fn resolved_model_name_strips_split_suffix_for_gguf() {
        let path = Path::new("/tmp/MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf");
        assert_eq!(resolved_model_name(path), "MiniMax-M2.5-Q4_K_M");
    }

    #[test]
    fn resolved_model_name_uses_config_name_for_hf_dirs() {
        let dir = temp_test_dir("model-name-config");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"_name_or_path":"Qwen/Qwen3-0.6B"}"#,
        )
        .unwrap();

        assert_eq!(resolved_model_name(&dir), "Qwen3-0.6B");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolved_model_name_falls_back_to_directory_name_for_hf_dirs() {
        let root = temp_test_dir("model-name-dir");
        let dir = root.join("Qwen3-0.6B-bf16");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"model_type":"qwen3"}"#).unwrap();

        assert_eq!(resolved_model_name(&dir), "Qwen3-0.6B-bf16");

        let _ = std::fs::remove_dir_all(root);
    }
}
