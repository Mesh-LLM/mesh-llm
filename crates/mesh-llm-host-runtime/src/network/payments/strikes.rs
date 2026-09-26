//! Local payee blocklist for "paid but undelivered" exchanges.
//!
//! A strike is recorded only when this node's input payment settled and the
//! provider then ended the exchange without delivering any output, and the
//! payer did not cancel. Repeated strikes against one provider endpoint block
//! it from paid routing for a while.
//!
//! This is deliberately local, like `target_health`: it is never gossiped,
//! because a shared blocklist would let any peer get an honest provider
//! excluded by reporting fabricated strikes. It also does not stop a provider
//! that returns under a fresh endpoint id; the per-strike loss is bounded by
//! the input charge because output is paid for only after delivery.
//!
//! State lives next to the payments ledger (`payments/payee_strikes.json`) so a
//! restart does not clear it. An operator clears a false positive by removing
//! that payee's entry or the file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const FILE_NAME: &str = "payee_strikes.json";
/// Strikes needed inside [`STRIKE_WINDOW_MS`] to block a payee.
pub(crate) const STRIKE_THRESHOLD: usize = 3;
pub(crate) const STRIKE_WINDOW_MS: u64 = 24 * 60 * 60 * 1000;
pub(crate) const BLOCK_DURATION_MS: u64 = 7 * 24 * 60 * 60 * 1000;

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    /// Strike times (unix ms) still inside the window.
    strikes: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blocked_until_ms: Option<u64>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PayeeStrikes {
    payees: BTreeMap<String, Entry>,
}

impl PayeeStrikes {
    /// Record a strike. Returns true when this strike blocks the payee.
    pub(crate) fn record(&mut self, payee: &str, now_ms: u64) -> bool {
        self.prune(now_ms);
        let entry = self.payees.entry(payee.to_owned()).or_default();
        entry.strikes.push(now_ms);
        if entry.blocked_until_ms.is_none() && entry.strikes.len() >= STRIKE_THRESHOLD {
            entry.blocked_until_ms = Some(now_ms.saturating_add(BLOCK_DURATION_MS));
            entry.strikes.clear();
            return true;
        }
        false
    }

    pub(crate) fn is_blocked(&self, payee: &str, now_ms: u64) -> bool {
        self.payees
            .get(payee)
            .and_then(|entry| entry.blocked_until_ms)
            .is_some_and(|until| now_ms < until)
    }

    fn prune(&mut self, now_ms: u64) {
        self.payees.retain(|_, entry| {
            entry
                .strikes
                .retain(|at| now_ms.saturating_sub(*at) < STRIKE_WINDOW_MS);
            if entry.blocked_until_ms.is_some_and(|until| now_ms >= until) {
                entry.blocked_until_ms = None;
            }
            !entry.strikes.is_empty() || entry.blocked_until_ms.is_some()
        });
    }

    fn path(directory: &Path) -> PathBuf {
        directory.join(FILE_NAME)
    }

    /// Missing or unreadable state is treated as empty: the blocklist is an
    /// optimisation, and must never make paid routing fail.
    pub(crate) fn load(directory: &Path) -> Self {
        std::fs::read(Self::path(directory))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    pub(crate) fn save(&self, directory: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(directory)?;
        let path = Self::path(directory);
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(temp, path)
    }
}

/// Load, record one strike, persist. Returns true if the payee is now blocked.
pub(crate) fn record_strike(directory: &Path, payee: &str, now_ms: u64) -> std::io::Result<bool> {
    let mut strikes = PayeeStrikes::load(directory);
    let blocked = strikes.record(payee, now_ms);
    strikes.save(directory)?;
    Ok(blocked)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: u64 = 60 * 60 * 1000;

    #[test]
    fn blocks_after_threshold_inside_window() {
        let mut strikes = PayeeStrikes::default();
        assert!(!strikes.record("a", 0));
        assert!(!strikes.record("a", HOUR));
        assert!(!strikes.is_blocked("a", HOUR));
        assert!(strikes.record("a", 2 * HOUR));
        assert!(strikes.is_blocked("a", 2 * HOUR));
        assert!(!strikes.is_blocked("b", 2 * HOUR));
    }

    #[test]
    fn strikes_outside_window_do_not_accumulate() {
        let mut strikes = PayeeStrikes::default();
        strikes.record("a", 0);
        strikes.record("a", 1);
        assert!(!strikes.record("a", STRIKE_WINDOW_MS + 1));
        assert!(!strikes.is_blocked("a", STRIKE_WINDOW_MS + 1));
    }

    #[test]
    fn block_expires() {
        let mut strikes = PayeeStrikes::default();
        for at in 0..STRIKE_THRESHOLD as u64 {
            strikes.record("a", at);
        }
        let blocked_at = STRIKE_THRESHOLD as u64 - 1;
        assert!(strikes.is_blocked("a", blocked_at + BLOCK_DURATION_MS - 1));
        assert!(!strikes.is_blocked("a", blocked_at + BLOCK_DURATION_MS));
    }

    #[test]
    fn survives_reload_and_tolerates_corrupt_state() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        for at in 0..STRIKE_THRESHOLD as u64 {
            record_strike(directory.path(), "a", at)?;
        }
        assert!(PayeeStrikes::load(directory.path()).is_blocked("a", STRIKE_THRESHOLD as u64));
        std::fs::write(directory.path().join(FILE_NAME), b"not json")?;
        assert_eq!(
            PayeeStrikes::load(directory.path()),
            PayeeStrikes::default()
        );
        Ok(())
    }
}
