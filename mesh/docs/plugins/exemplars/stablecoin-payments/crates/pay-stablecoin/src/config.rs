use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
use wallet_contract::decimal;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Price {
    #[serde(with = "decimal")]
    pub input_per_million: u64,
    #[serde(with = "decimal")]
    pub output_per_million: u64,
}
impl Price {
    pub fn cost(&self, input: u64, output: u64) -> Result<u64> {
        let numerator = (u128::from(input) * u128::from(self.input_per_million))
            .checked_add(u128::from(output) * u128::from(self.output_per_million))
            .context("price multiplication overflow")?;
        Ok(u64::try_from(numerator.div_ceil(1_000_000))?)
    }
    pub fn within(&self, maximum: &Self) -> bool {
        self.input_per_million <= maximum.input_per_million
            && self.output_per_million <= maximum.output_per_million
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Buying {
    pub enabled: bool,
    pub peers: Vec<String>,
    pub maximum_price: Price,
    #[serde(with = "decimal")]
    pub max_payment: u64,
    #[serde(with = "decimal")]
    pub max_fee: u64,
    #[serde(with = "decimal")]
    pub daily_budget: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub data_dir: PathBuf,
    #[serde(default = "binding")]
    pub wallet_binding: String,
    #[serde(default)]
    pub prices: BTreeMap<String, Price>,
    #[serde(default, with = "decimal")]
    pub admission_fee: u64,
    #[serde(default)]
    pub buying: Buying,
}
fn binding() -> String {
    "wallet".into()
}
impl Config {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        ensure!(path.is_absolute(), "config_path must be absolute");
        let config: Self =
            serde_json::from_slice(&std::fs::read(path).context("read payment config")?)?;
        ensure!(config.data_dir.is_absolute(), "data_dir must be absolute");
        ensure!(!config.wallet_binding.is_empty(), "wallet_binding is empty");
        for peer in &config.buying.peers {
            validate_peer(peer)?;
        }
        if config.buying.enabled {
            ensure!(
                !config.buying.peers.is_empty(),
                "spending requires an explicit seller allowlist"
            );
            ensure!(
                config.buying.max_payment > 0 && config.buying.daily_budget > 0,
                "spending requires nonzero payment and daily limits"
            );
        }
        Ok(config)
    }
}
pub fn validate_peer(peer: &str) -> Result<()> {
    ensure!(
        peer.len() == 64 && hex::decode(peer)?.len() == 32,
        "invalid Mesh peer ID"
    );
    Ok(())
}
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cost_rounds_once_and_first_credit_never_carries_forward() {
        let price = Price {
            input_per_million: 1,
            output_per_million: 1,
        };
        assert_eq!(price.cost(1, 1).unwrap(), 1);
        assert_eq!(price.cost(1_000_000, 1_000_000).unwrap(), 2);
        assert_eq!(price.cost(1, 1).unwrap().saturating_sub(10), 0);
        assert_eq!(price.cost(1, 1).unwrap(), 1);
    }
    #[test]
    fn maximum_counts_cannot_wrap_into_cheap_payment() {
        let price = Price {
            input_per_million: u64::MAX,
            output_per_million: u64::MAX,
        };
        assert!(price.cost(u64::MAX, u64::MAX).is_err());
    }
}
