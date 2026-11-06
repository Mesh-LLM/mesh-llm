use super::Model;
use crate::command::DynResult;
use std::path::Path;
pub(super) const DATA_REPO: &str = "semianalysisai/cc-traces-weka-062126-256k";
pub(super) const DATA_REVISION: &str = "8fecd2fc56694469f758f0afbbb6335ad3043740";
pub(super) const DATA_SHA: &str =
    "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef";

pub(super) fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

pub(super) fn select_model(root: &Path, key: &str) -> DynResult<Model> {
    let models: Vec<Model> = serde_json::from_slice(&std::fs::read(
        root.join("skippy/evals/kv-validation/models.json"),
    )?)?;
    models
        .into_iter()
        .find(|model| model.key == key || model.registry_ref() == key)
        .ok_or_else(|| "unknown AgentX model".into())
}
pub(super) fn verify(path: &Path, expected: &str) -> DynResult<()> {
    let actual = crate::product::digest::file_sha256(path).map_err(|failure| failure.error)?;
    if actual != expected {
        return Err(format!("SHA-256 mismatch for {}", path.display()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupted_model_fails_before_use() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        assert!(verify(temp.path(), DATA_SHA).is_err());
    }
}
