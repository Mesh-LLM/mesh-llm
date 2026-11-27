use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

pub(crate) fn config_overlay_path(data_dir: &Path) -> PathBuf {
    data_dir.join("config_overlay.json")
}

pub(crate) async fn persist_config_overlay(
    data_dir: &Path,
    patch: &serde_json::Value,
) -> anyhow::Result<()> {
    let path = config_overlay_path(data_dir);
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec(patch)?;
    let mut file = tokio::fs::File::create(&tmp).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    tokio::fs::rename(&tmp, &path).await?;
    Ok(())
}

pub(crate) async fn load_config_overlay(
    data_dir: &Path,
) -> anyhow::Result<Option<serde_json::Value>> {
    let path = config_overlay_path(data_dir);
    match tokio::fs::read(&path).await {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
