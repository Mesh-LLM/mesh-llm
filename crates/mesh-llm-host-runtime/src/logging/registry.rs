//! Active/recent request registry for logging service.
//!
//! Tracks in-flight requests (active) and recently completed ones (recent). Both sets are bounded:
//! when capacity is exceeded, the oldest entry by `created_at` is evicted FIFO-style.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Summary record for a single request in the registry. Carries enough metadata to reconstruct
/// what happened without persisting full payloads by default.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RequestSummaryEntry {
    /// UUID string identifying this request (from `RequestId::as_uuid()`).
    pub request_id: String,

    /// Current lifecycle state: "active", "completed", "failed", etc. Updated on terminal transition.
    pub state: String,

    /// ISO 8601 timestamp when the entry was first registered in active. Never changes after creation.
    pub created_at: String,

    /// ISO 8601 timestamp of the terminal transition (completed/failed/etc.). `None` while active.
    pub terminal_at: Option<String>,
}

/// Configuration controlling registry capacity bounds.
#[derive(Clone, Debug)]
pub struct RegistryConfig {
    /// Maximum number of entries in the active set before FIFO eviction applies. Default: 1024.
    pub max_active: usize,

    /// Maximum number of entries in the recent set before FIFO eviction applies. Default: 8192.
    pub max_recent: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_active: 1024,
            max_recent: 8192,
        }
    }
}

/// Active/recent request registry with bounded capacity and FIFO eviction.
///
/// Thread-safe via internal Mutex guards on each set. Designed to be shared behind `Arc` across
/// the service facade (bus push path, terminal transition path, observability reads).
pub struct RequestRegistry {
    /// In-flight requests keyed by UUID string. Evicted oldest-first when exceeding max_active.
    active: Mutex<HashMap<String, RequestSummaryEntry>>,

    /// Recently completed/failed/dropped requests keyed by UUID string. Evicted oldest-first when exceeding max_recent.
    recent: Mutex<HashMap<String, RequestSummaryEntry>>,

    config: RegistryConfig,

    /// Total number of entries evicted from the active set (for observability).
    pub active_evictions: Arc<AtomicU64>,

    /// Total number of entries evicted from the recent set (for observability).
    #[allow(dead_code)]
    pub recent_evictions: Arc<AtomicU64>,
}

impl RequestRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            active: Mutex::new(HashMap::with_capacity(config.max_active)),
            recent: Mutex::new(HashMap::with_capacity(config.max_recent)),
            config,
            active_evictions: Arc::new(AtomicU64::new(0)),
            recent_evictions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Insert an entry into the active set. If the set is already at capacity, evict the oldest
    /// entry (by `created_at` lexicographic comparison) and increment `active_evictions`.
    pub fn register_active(&self, entry: RequestSummaryEntry) {
        let mut map = self.active.lock().expect("registry mutex poisoned");

        // Evict if at capacity.
        if map.len() >= self.config.max_active {
            evict_oldest(&mut map);
            self.active_evictions.fetch_add(1, Ordering::Relaxed);
        }

        map.insert(entry.request_id.clone(), entry);
    }

    /// Look up an active entry by request ID. Returns a clone (not a reference) since the caller
    /// may need to hold it across the Mutex unlock boundary.
    pub fn get_active(&self, request_id: &str) -> Option<RequestSummaryEntry> {
        self.active
            .lock()
            .expect("registry mutex poisoned")
            .get(request_id)
            .cloned()
    }

    /// Remove an entry from active and insert into recent. If recent exceeds max_recent, evict the
    /// oldest entry and increment `recent_evictions`. The caller is expected to have already updated
    /// the entry's state/terminal_at fields before calling this method.
    pub fn move_to_recent(&self, entry: RequestSummaryEntry) {
        let rid = entry.request_id.clone();

        // Remove from active first (may not exist if it was evicted; that's fine).
        {
            let mut map = self.active.lock().expect("registry mutex poisoned");
            map.remove(&rid);
        }

        // Insert into recent. Evict oldest if at capacity.
        {
            let mut map = self.recent.lock().expect("registry mutex poisoned");
            if map.len() >= self.config.max_recent {
                evict_oldest(&mut map);
                self.recent_evictions.fetch_add(1, Ordering::Relaxed);
            }

            map.insert(entry.request_id.clone(), entry);
        }
    }

    /// Current number of entries in the active set.
    pub fn active_count(&self) -> usize {
        self.active.lock().expect("registry mutex poisoned").len()
    }

    /// Current number of entries in the recent set.
    pub fn recent_count(&self) -> usize {
        self.recent.lock().expect("registry mutex poisoned").len()
    }

    /// Look up a recent entry by request ID. Returns a clone (not a reference).
    pub fn get_recent(&self, request_id: &str) -> Option<RequestSummaryEntry> {
        self.recent
            .lock()
            .expect("registry mutex poisoned")
            .get(request_id)
            .cloned()
    }

    /// Clear both active and recent sets. Used for shutdown or test isolation.
    pub fn clear(&self) {
        let mut map = self.active.lock().expect("registry mutex poisoned");
        map.clear();
        drop(map);

        let mut map = self.recent.lock().expect("registry mutex poisoned");
        map.clear();
    }

    /// Returns true if both active and recent sets are empty.
    pub fn is_empty(&self) -> bool {
        let a = self
            .active
            .lock()
            .expect("registry mutex poisoned")
            .is_empty();
        let r = self
            .recent
            .lock()
            .expect("registry mutex poisoned")
            .is_empty();
        a && r
    }
}

/// Remove the entry with the lexicographically smallest `created_at` from the map.
/// No-op if the map is empty. Uses ISO 8601 timestamp ordering (lexicographic = chronological).
fn evict_oldest(map: &mut HashMap<String, RequestSummaryEntry>) {
    let oldest_key = map
        .iter()
        .min_by_key(|(_, entry)| &entry.created_at)
        .map(|(key, _)| key.clone());

    if let Some(key) = oldest_key {
        map.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_entry(id: &str, ts: u8) -> RequestSummaryEntry {
        RequestSummaryEntry {
            request_id: id.to_string(),
            state: "active".into(),
            created_at: format!("2025-01-01T00:00:{:02}Z", ts),
            terminal_at: None,
        }
    }

    // ---------------------------------------------------------------------------
    // Basic active → recent movement
    // ---------------------------------------------------------------------------

    #[test]
    fn test_active_to_recent_movement() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 10,
            max_recent: 20,
        });

        let entry = make_entry("req-1", 0);
        reg.register_active(entry.clone());

        assert_eq!(reg.active_count(), 1);
        assert_eq!(reg.recent_count(), 0);

        // Move to recent.
        reg.move_to_recent(entry.clone());

        assert_eq!(reg.active_count(), 0);
        assert_eq!(reg.recent_count(), 1);
    }

    #[test]
    fn test_get_active_returns_clone() {
        let reg = RequestRegistry::new(RegistryConfig::default());

        let entry = make_entry("req-clone", 5);
        reg.register_active(entry.clone());

        let got = reg.get_active("req-clone").unwrap();
        assert_eq!(got.request_id, "req-clone");

        // Can still get it again — clones don't consume the map entry.
        let got2 = reg.get_active("req-clone");
        assert!(got2.is_some());
    }

    #[test]
    fn test_get_recent_returns_clone() {
        let reg = RequestRegistry::new(RegistryConfig::default());

        let entry = make_entry("req-recent", 5);
        reg.register_active(entry.clone());
        reg.move_to_recent(entry.clone());

        let got = reg.get_recent("req-recent").unwrap();
        assert_eq!(got.request_id, "req-recent");

        // Can still get it again.
        assert!(reg.get_recent("req-recent").is_some());
    }

    #[test]
    fn test_get_missing_returns_none() {
        let reg = RequestRegistry::new(RegistryConfig::default());
        assert!(reg.get_active("nonexistent").is_none());
        assert!(reg.get_recent("nonexistent").is_none());
    }

    // ---------------------------------------------------------------------------
    // Eviction behavior
    // ---------------------------------------------------------------------------

    #[test]
    fn test_active_eviction_when_over_capacity() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 3,
            max_recent: 10,
        });

        for i in 0..5u8 {
            reg.register_active(make_entry(&format!("req-{}", i), i));
        }

        assert_eq!(reg.active_count(), 3); // capped at max_active
        assert_eq!(reg.active_evictions.load(Ordering::Relaxed), 2); // 5 - 3 = 2 evicted

        // Oldest two (ts=0, ts=1) should be gone.
        assert!(reg.get_active("req-0").is_none());
        assert!(reg.get_active("req-1").is_none());

        // Newest three should remain.
        assert!(reg.get_active("req-2").is_some());
        assert!(reg.get_active("req-3").is_some());
        assert!(reg.get_active("req-4").is_some());
    }

    #[test]
    fn test_recent_eviction_when_over_capacity() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 20,
            max_recent: 3,
        });

        for i in 0..5u8 {
            let entry = make_entry(&format!("req-{}", i), i);
            reg.register_active(entry.clone());
            reg.move_to_recent(entry); // move immediately to recent.
        }

        assert_eq!(reg.recent_count(), 3); // capped at max_recent
        assert_eq!(reg.recent_evictions.load(Ordering::Relaxed), 2);

        // Oldest two should be evicted from recent.
        assert!(reg.get_recent("req-0").is_none());
        assert!(reg.get_recent("req-1").is_none());

        // Newest three remain.
        for i in 2..=4u8 {
            assert!(
                reg.get_recent(&format!("req-{}", i)).is_some(),
                "req-{} should be in recent",
                i
            );
        }
    }

    #[test]
    fn test_eviction_removes_oldest_by_created_at() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 2,
            max_recent: 10,
        });

        // Insert entries with non-monotonic IDs but monotonic timestamps.
        reg.register_active(make_entry("z-last", 3));
        reg.register_active(make_entry("a-first", 1));
        reg.register_active(make_entry("m-mid", 2));

        assert_eq!(reg.active_count(), 2);
        // "a-first" (ts=1) should be evicted — it's the oldest.
        assert!(reg.get_active("a-first").is_none());
    }

    // ---------------------------------------------------------------------------
    // Clear and is_empty
    // ---------------------------------------------------------------------------

    #[test]
    fn test_clear_empties_both_sets() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 10,
            max_recent: 20,
        });

        for i in 0..3u8 {
            let entry = make_entry(&format!("req-{}", i), i);
            reg.register_active(entry.clone());
            if i % 2 == 0 {
                reg.move_to_recent(entry);
            }
        }

        assert!(!reg.is_empty());

        reg.clear();

        assert_eq!(reg.active_count(), 0);
        assert_eq!(reg.recent_count(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn test_is_empty_on_fresh_registry() {
        let reg = RequestRegistry::new(RegistryConfig::default());
        assert!(reg.is_empty());
    }

    // ---------------------------------------------------------------------------
    // No registry leak: bounded sets don't grow unbounded
    // ---------------------------------------------------------------------------

    #[test]
    fn test_no_leak_under_pressure() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 5,
            max_recent: 10,
        });

        for i in 0..200u8 {
            let entry = make_entry(&format!("req-{}", i), i % 60);
            reg.register_active(entry.clone());
            if i % 3 == 0 {
                reg.move_to_recent(entry);
            }
        }

        assert!(
            reg.active_count() <= reg.config.max_active,
            "active should be bounded"
        );
        assert!(
            reg.recent_count() <= reg.config.max_recent,
            "recent should be bounded"
        );
    }

    // ---------------------------------------------------------------------------
    // Concurrent register from multiple threads (Send + Sync)
    // ---------------------------------------------------------------------------

    #[test]
    fn test_concurrent_register_active() {
        let reg = Arc::new(RequestRegistry::new(RegistryConfig {
            max_active: 100,
            max_recent: 200,
        }));

        let mut handles = Vec::new();

        for t in 0..4u8 {
            let r = Arc::clone(&reg);
            handles.push(thread::spawn(move || {
                for i in 0..50u8 {
                    let entry = RequestSummaryEntry {
                        request_id: format!("t{}-req-{}", t, i),
                        state: "active".into(),
                        created_at: format!(
                            "2025-01-01T00:{:02}:{:02}Z",
                            (t * 15) as u32,
                            i as u32 % 60
                        ),
                        terminal_at: None,
                    };
                    r.register_active(entry);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All 4 * 50 = 200 registrations attempted; max_active is 100. Some were evicted.
        assert!(reg.active_count() <= reg.config.max_active);
    }

    #[test]
    fn test_config_default_values() {
        let cfg = RegistryConfig::default();
        assert_eq!(cfg.max_active, 1024);
        assert_eq!(cfg.max_recent, 8192);
    }

    #[test]
    fn test_terminal_at_preserved_through_move_to_recent() {
        let reg = RequestRegistry::new(RegistryConfig::default());

        let mut entry = make_entry("req-term", 7);
        reg.register_active(entry.clone());

        // Simulate terminal transition: update state and set terminal_at.
        entry.state = "completed".into();
        entry.terminal_at = Some("2025-01-01T00:00:15Z".into());

        reg.move_to_recent(entry);

        let recent = reg.get_recent("req-term").unwrap();
        assert_eq!(recent.state, "completed");
        assert_eq!(recent.terminal_at.as_deref(), Some("2025-01-01T00:00:15Z"));
    }
}
