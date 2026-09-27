//! Local payee blocklist for "paid but undelivered" exchanges.
//!
//! A strike is recorded only when this node's input payment settled, the
//! provider then ended the exchange without delivering any output and without
//! completing it, and the payer did not cancel. Repeated strikes against one
//! provider endpoint block it from paid routing for a while.
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
//!
//! Paid routing consults the blocklist on every request, so reads go through
//! the process-local [`READ_CACHE`] instead of touching the file each time.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

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

/// What a cached read was loaded from. Size and modification time identify a
/// rewrite by anything else on the machine; this process refreshes the cache
/// from what it just wrote, so its own writes cannot be missed even when the
/// filesystem timestamp has not moved.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Stamp {
    bytes: u64,
    modified: Option<std::time::SystemTime>,
}

impl Stamp {
    fn read(path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        Some(Self {
            bytes: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

/// One remembered read: what the file looked like, and the state read from it.
struct CachedRead {
    stamp: Option<Stamp>,
    strikes: PayeeStrikes,
}

/// The last read of each payments directory. A host has one payments
/// directory, so this normally holds a single entry, and a request that finds
/// no blocklist costs one `stat` rather than a read and a JSON parse.
static READ_CACHE: LazyLock<Mutex<HashMap<PathBuf, CachedRead>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Serialises strike read-modify-write cycles so two simultaneous failures
/// cannot each read the same state and lose one another's strike.
static WRITE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn cached(path: &Path, stamp: &Option<Stamp>) -> Option<PayeeStrikes> {
    let cache = READ_CACHE.lock().ok()?;
    cache
        .get(path)
        .filter(|entry| &entry.stamp == stamp)
        .map(|entry| entry.strikes.clone())
}

fn remember(path: PathBuf, stamp: Option<Stamp>, strikes: &PayeeStrikes) {
    if let Ok(mut cache) = READ_CACHE.lock() {
        cache.insert(
            path,
            CachedRead {
                stamp,
                strikes: strikes.clone(),
            },
        );
    }
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
        let path = Self::path(directory);
        let stamp = Stamp::read(&path);
        if let Some(strikes) = cached(&path, &stamp) {
            return strikes;
        }
        let strikes = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        remember(path, stamp, &strikes);
        strikes
    }

    pub(crate) fn save(&self, directory: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(directory)?;
        let path = Self::path(directory);
        let temp = path.with_extension("json.tmp");
        // Strikes are rare, so the syncs are cheap insurance: without them a
        // crash can leave the rename durable while the new content is not, and
        // the payee silently stays unblocked.
        let mut file = File::create(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(self)?)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, &path)?;
        // Syncing the file does not make the rename itself durable: the
        // directory entry needs a sync too. Opening a directory is not
        // portable (Windows refuses it), so this is best-effort — where it
        // works the newest strike survives a power loss, and where it does not
        // the worst case is losing that one strike, which the next repeats.
        #[cfg(unix)]
        let _ = File::open(directory).and_then(|directory| directory.sync_all());
        let stamp = Stamp::read(&path);
        remember(path, stamp, self);
        Ok(())
    }
}

/// The blocklist file, for logs and for the operator who has to clear it.
pub(crate) fn state_path(directory: &Path) -> PathBuf {
    PayeeStrikes::path(directory)
}

/// Load, record one strike, persist. Returns true if the payee is now blocked.
pub(crate) fn record_strike(directory: &Path, payee: &str, now_ms: u64) -> std::io::Result<bool> {
    let _writing = WRITE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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

    #[test]
    fn cached_reads_still_follow_a_file_that_changes_underneath() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join(FILE_NAME);
        // A missing file is cached as empty, and the first write is visible.
        assert!(!PayeeStrikes::load(directory.path()).is_blocked("a", 0));
        for at in 0..STRIKE_THRESHOLD as u64 {
            record_strike(directory.path(), "a", at)?;
        }
        assert!(PayeeStrikes::load(directory.path()).is_blocked("a", STRIKE_THRESHOLD as u64));
        // An operator deleting the file clears the block, even though this
        // process cached the state it wrote.
        std::fs::remove_file(&path)?;
        assert!(!PayeeStrikes::load(directory.path()).is_blocked("a", 0));
        // A replacement written by something else is re-read: the cached stamp
        // no longer matches its size and modification time.
        let edited = r#"{"payees":{"c":{"strikes":[1,2],"blocked_until_ms":9999999999999}}}"#;
        std::fs::write(&path, edited)?;
        assert!(PayeeStrikes::load(directory.path()).is_blocked("c", 0));
        Ok(())
    }

    #[test]
    fn simultaneous_strikes_are_all_recorded() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let recorded: Vec<_> = (0..STRIKE_THRESHOLD)
            .map(|at| {
                let directory = directory.path().to_path_buf();
                std::thread::spawn(move || record_strike(&directory, "a", at as u64).unwrap())
            })
            .collect();
        let mut blocking = 0;
        for thread in recorded {
            if thread.join().expect("strike thread panicked") {
                blocking += 1;
            }
        }
        assert_eq!(blocking, 1, "exactly one strike reaches the threshold");
        assert!(
            PayeeStrikes::load(directory.path()).is_blocked("a", STRIKE_THRESHOLD as u64),
            "no strike was lost to a concurrent read-modify-write"
        );
        Ok(())
    }
}
