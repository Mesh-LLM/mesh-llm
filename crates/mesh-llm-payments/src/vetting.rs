//! Durable observations, not payment consent or proof of model identity.
use crate::ledger::Ledger;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const CHALLENGE_VERSION: u32 = 1;
pub const DEFAULT_TTL_MS: u64 = 86_400_000;
const MAX_RECORDS: usize = 1024;

#[derive(Clone, Serialize, Deserialize)]
pub struct VettingRecord {
    pub provider_id: String,
    pub model: String,
    pub checked_at_ms: u64,
    pub challenge_version: u32,
}

impl VettingRecord {
    pub fn is_fresh(&self, now_ms: u64, ttl_ms: u64) -> bool {
        self.challenge_version == CHALLENGE_VERSION
            && ttl_ms > 0
            && now_ms
                .checked_sub(self.checked_at_ms)
                .is_some_and(|age| age < ttl_ms)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VettingPolicy {
    pub required: bool,
    #[serde(default)]
    pub serve_probes: bool,
    pub ttl_ms: u64,
}
impl Default for VettingPolicy {
    fn default() -> Self {
        Self {
            required: false,
            serve_probes: false,
            ttl_ms: DEFAULT_TTL_MS,
        }
    }
}
impl Ledger {
    pub fn vetting_policy(&self) -> Result<VettingPolicy> {
        self.get_setting("vetting_policy")?.map_or_else(
            || Ok(VettingPolicy::default()),
            |value| Ok(serde_json::from_str(&value)?),
        )
    }
    pub fn set_vetting_policy(&self, policy: &VettingPolicy) -> Result<()> {
        ensure!(
            policy.ttl_ms > 0 && policy.ttl_ms <= 604_800_000,
            "vetting TTL must be positive and at most seven days"
        );
        self.set_setting("vetting_policy", &serde_json::to_string(policy)?)
    }

    pub fn provider_vetted(&self, provider_id: &str, now_ms: u64, ttl_ms: u64) -> Result<bool> {
        Ok(self
            .vetting_records()?
            .iter()
            .any(|record| record.provider_id == provider_id && record.is_fresh(now_ms, ttl_ms)))
    }

    /// Explicit operator reset; never changes payment records or obligations.
    pub fn reset_provider_vetting(&self) -> Result<()> {
        self.set_setting("provider_vetting", "[]")
    }

    fn vetting_records(&self) -> Result<Vec<VettingRecord>> {
        self.get_setting("provider_vetting")?
            .map_or_else(|| Ok(Vec::new()), |value| Ok(serde_json::from_str(&value)?))
    }

    /// Caller must supply the authenticated endpoint ID after verifying a fresh
    /// challenge. The tested model is evidence metadata, not catalogue certification.
    pub fn record_provider_vetted(&self, record: VettingRecord) -> Result<()> {
        ensure!(
            record.provider_id.len() == 64
                && record.provider_id.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid provider ID"
        );
        ensure!(
            !record.model.is_empty() && record.model.len() <= 1024,
            "invalid model"
        );
        ensure!(
            record.challenge_version == CHALLENGE_VERSION,
            "unsupported challenge version"
        );
        use rusqlite::OptionalExtension;
        let mut connection = self.lock()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let stored: Option<String> = transaction
            .query_row(
                "SELECT value FROM settings WHERE key='provider_vetting'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let mut records: Vec<VettingRecord> =
            stored.map_or_else(|| Ok(Vec::new()), |value| serde_json::from_str(&value))?;
        records.retain(|old| old.provider_id != record.provider_id);
        records.push(record);
        records.sort_by_key(|record| record.checked_at_ms);
        if records.len() > MAX_RECORDS {
            records.drain(..records.len() - MAX_RECORDS);
        }
        transaction.execute(
            "INSERT INTO settings VALUES ('provider_vetting',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [serde_json::to_string(&records)?],
        )?;
        transaction.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_survives_restart_expires_lazily_and_rejects_clock_rollback() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let provider = "ab".repeat(32);
        {
            let ledger = Ledger::open(dir.path())?;
            assert!(!ledger.provider_vetted(&provider, 1000, 100)?);
            ledger.record_provider_vetted(VettingRecord {
                provider_id: provider.clone(),
                model: "tested".into(),
                checked_at_ms: 1000,
                challenge_version: CHALLENGE_VERSION,
            })?;
        }
        let ledger = Ledger::open(dir.path())?;
        assert!(ledger.provider_vetted(&provider, 1099, 100)?);
        assert!(!ledger.provider_vetted(&provider, 1100, 100)?);
        assert!(!ledger.provider_vetted(&provider, 999, 100)?);
        assert!(!ledger.provider_vetted(&"cd".repeat(32), 1000, 100)?);
        assert!(matches!(
            ledger.payment_intent()?,
            crate::intent::PaymentIntent::FreeOnly
        ));
        assert!(ledger.requests()?.is_empty());
        Ok(())
    }
}
