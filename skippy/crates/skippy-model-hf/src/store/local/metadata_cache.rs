use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use super::huggingface_identity_for_path;

pub fn model_metadata_cache_dir() -> PathBuf {
    model_metadata_cache_dir_in(&crate::application_cache_dir())
}

pub fn model_metadata_cache_dir_in(cache_root: &Path) -> PathBuf {
    cache_root.join("model-meta")
}

pub fn gguf_metadata_cache_path(path: &Path) -> Option<PathBuf> {
    gguf_metadata_cache_path_in(path, &crate::application_cache_dir())
}

pub fn gguf_metadata_cache_path_in(path: &Path, cache_root: &Path) -> Option<PathBuf> {
    let key = if let Some(identity) = huggingface_identity_for_path(path) {
        format!("hf:{}", identity.canonical_ref)
    } else {
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        format!(
            "local:{}:{}:{}",
            path.to_string_lossy(),
            metadata.len(),
            modified
        )
    };
    let digest = Sha256::digest(key.as_bytes());
    Some(model_metadata_cache_dir_in(cache_root).join(format!("{}.json", hex::encode(digest))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_paths_are_isolated_by_cache_root() {
        let temp = tempfile::tempdir().expect("temp directory");
        let model = temp.path().join("model.gguf");
        std::fs::write(&model, b"fixture").expect("model fixture");
        let skippy_root = temp.path().join("skippy-cache");
        let mesh_root = temp.path().join("mesh-cache");
        let skippy_path = gguf_metadata_cache_path_in(&model, &skippy_root).expect("cache path");
        let mesh_path = gguf_metadata_cache_path_in(&model, &mesh_root).expect("cache path");
        assert_eq!(
            skippy_path.parent(),
            Some(model_metadata_cache_dir_in(&skippy_root).as_path())
        );
        assert_eq!(
            mesh_path.parent(),
            Some(model_metadata_cache_dir_in(&mesh_root).as_path())
        );
        assert_eq!(skippy_path.file_name(), mesh_path.file_name());
        assert_ne!(skippy_path, mesh_path);
    }
}
