//! Explicit plugin-owned configuration. Loading never opens a wallet or ledger.
use anyhow::{Context, Result, ensure};
use mesh_llm_payments::ledger::Policy;
use mesh_llm_payments_types::pricing::Pricing;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub data_dir: PathBuf,
    #[serde(default)]
    pub prices: BTreeMap<String, Pricing>,
    #[serde(default)]
    pub buy_prices: BTreeMap<String, Pricing>,
    #[serde(default)]
    pub buy_peers: BTreeSet<String>,
    #[serde(default)]
    pub cap_msat: u64,
    #[serde(default)]
    pub policy: Policy,
    #[serde(default)]
    pub error_log: Option<PathBuf>,
    #[serde(default)]
    pub timing_log: Option<PathBuf>,
}
impl Config {
    /// Read a JSON file at an explicitly supplied path. No legacy environment
    /// policy or implicit adoption of an old host payment directory is supported.
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("read payment config {}", path.display()))?,
        )?;
        config.validate()?;
        Ok(config)
    }
    /// Hosts can pass a config path using the generic plugin environment map.
    /// Otherwise use config.json inside an explicitly selected plugin data dir.
    pub fn from_launch_env() -> Result<Self> {
        let path = if let Some(path) = std::env::var_os("MESH_LLM_PLUGIN_CONFIG") {
            PathBuf::from(path)
        } else {
            PathBuf::from(
                std::env::var_os("MESH_LLM_PLUGIN_DATA_DIR")
                    .context("set MESH_LLM_PLUGIN_CONFIG or MESH_LLM_PLUGIN_DATA_DIR")?,
            )
            .join("config.json")
        };
        Self::load(&path)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.data_dir.is_absolute(),
            "data_dir must be absolute isolated plugin state"
        );
        for path in [&self.error_log, &self.timing_log].into_iter().flatten() {
            ensure!(path.is_absolute(), "diagnostic paths must be absolute");
        }
        // Deserialize via the validated pricing parser as well: serde alone does
        // not enforce the domain's discount/rate bounds.
        self.policy.validate()?;
        crate::parse_prices(&serde_json::to_string(&self.prices)?)?;
        crate::parse_prices(&serde_json::to_string(&self.buy_prices)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_is_side_effect_free_and_defaults_to_no_spending() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("not-created");
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({"data_dir": data})).unwrap(),
        )
        .unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.cap_msat, 0);
        assert!(config.buy_peers.is_empty());
        assert!(!data.exists());
    }
    #[test]
    fn refuses_unknown_fields_and_relative_state() {
        assert!(
            serde_json::from_str::<Config>(r#"{"data_dir":"/tmp/unused","wallet":"legacy"}"#)
                .is_err()
        );
        let config: Config = serde_json::from_str(r#"{"data_dir":"relative"}"#).unwrap();
        assert!(config.validate().is_err());
    }
}
