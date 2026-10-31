//! Host-managed model selection and asset acquisition contracts.
//!
//! Model inference and session residency stay in the provider; downloads use
//! the host's credentials, source policy and shared Hugging Face cache.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub use mesh_llm_config::managed_models::{
    ManagedModelConfig, ModelLifecycle, ModelLoadPolicy, ModelSource,
};

/// The host-provided model selections for this authenticated plugin process.
pub const CONFIG_ENV: &str = "MESH_LLM_PLUGIN_MODELS";
/// Optional host feature used for provider-planned asset acquisition.
pub const ASSET_CAPABILITY: &str = "model_assets.v1";

/// Read only this process's explicitly selected models. Missing means none.
pub fn configured_models_from_env() -> Result<Vec<ManagedModelConfig>> {
    match std::env::var(CONFIG_ENV) {
        Ok(json) => serde_json::from_str(&json).context("invalid managed model configuration"),
        Err(std::env::VarError::NotPresent) => Ok(Vec::new()),
        Err(error) => Err(error.into()),
    }
}

/// A provider's explicit asset selection for one configured model.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AssetPlan {
    pub model: String,
    pub repo: String,
    pub revision: String,
    pub files: Vec<String>,
}

impl AssetPlan {
    /// Reject ambiguous sources, empty plans and filesystem traversal.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() || self.revision.trim().is_empty() {
            bail!("model and revision must not be empty");
        }
        let parts = self.repo.split('/').collect::<Vec<_>>();
        if parts.len() != 2 || parts.iter().any(|part| !safe_component(part)) {
            bail!("asset repository must be owner/name");
        }
        if self.files.is_empty() || self.files.len() > 256 {
            bail!("asset plan must contain between 1 and 256 files");
        }
        let mut unique = std::collections::BTreeSet::new();
        for file in &self.files {
            if file.len() > 1024 || file.split('/').any(|part| !safe_component(part)) {
                bail!("invalid relative asset path {file:?}");
            }
            if !unique.insert(file) {
                bail!("duplicate asset path {file:?}");
            }
        }
        Ok(())
    }
}

fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// A complete bundle. `files` maps recipe-relative names to local cache paths.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct PreparedModelAssets {
    pub directory: PathBuf,
    pub files: BTreeMap<String, PathBuf>,
    pub revision: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(files: &[&str]) -> AssetPlan {
        AssetPlan {
            model: "owner/repo@pin".into(),
            repo: "owner/repo".into(),
            revision: "pin".into(),
            files: files.iter().map(|v| (*v).into()).collect(),
        }
    }

    #[test]
    fn external_data_paths_remain_relative_and_unique() {
        assert!(
            plan(&["onnx/model.onnx", "onnx/model.onnx_data"])
                .validate()
                .is_ok()
        );
        for paths in [
            vec!["../secret"],
            vec!["/absolute"],
            vec!["a\\b"],
            vec!["a", "a"],
            vec![],
        ] {
            assert!(plan(&paths).validate().is_err());
        }
    }
}
