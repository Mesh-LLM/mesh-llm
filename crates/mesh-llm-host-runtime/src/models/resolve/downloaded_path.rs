use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::huggingface_identity_for_path;

pub(super) fn canonicalize_hf_download_path(path: PathBuf) -> Result<PathBuf> {
    if !is_monolithic_gguf(&path)
        || !path
            .symlink_metadata()
            .with_context(|| format!("Inspect downloaded Hugging Face model {}", path.display()))?
            .file_type()
            .is_symlink()
    {
        return Ok(path);
    }
    path.canonicalize().with_context(|| {
        format!(
            "Canonicalize downloaded Hugging Face model {}",
            path.display()
        )
    })
}

pub(super) fn canonicalize_cached_hf_path(path: &Path) -> Result<PathBuf> {
    if !is_monolithic_gguf(path) || huggingface_identity_for_path(path).is_none() {
        return Ok(path.to_path_buf());
    }
    canonicalize_hf_download_path(path.to_path_buf())
}

fn is_monolithic_gguf(path: &Path) -> bool {
    let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some((stem, extension)) = filename.rsplit_once('.') else {
        return false;
    };
    extension.eq_ignore_ascii_case("gguf")
        && model_ref::split_gguf_shard_info(&format!("{stem}.gguf")).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_gguf_and_split_gguf_downloads_keep_their_snapshot_names() {
        for filename in [
            "model.safetensors",
            "model-00001-of-00002.gguf",
            "model-00001-of-00002.GGUF",
            "model-00001-of-00002.GgUf",
            "README.md",
        ] {
            let path = PathBuf::from("/cache/snapshots/revision").join(filename);
            assert_eq!(canonicalize_hf_download_path(path.clone()).unwrap(), path);
        }
    }

    #[test]
    fn regular_gguf_download_keeps_its_original_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"gguf").unwrap();

        assert_eq!(canonicalize_hf_download_path(path.clone()).unwrap(), path);
    }
}
