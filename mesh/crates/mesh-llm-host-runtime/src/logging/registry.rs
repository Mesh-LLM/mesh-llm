//! Active/recent request registry for logging service.
//!
//! Tracks in-flight requests (active) and recently completed ones (recent). Both sets are bounded:
//! when capacity is exceeded, the oldest entry by `created_at` is evicted FIFO-style.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::request_metadata::RequestSummaryMetadata;

/// Summary record for a single request in the registry. Carries enough metadata to reconstruct
/// what happened without persisting full payloads by default.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RequestSummaryEntry {
    /// UUID string identifying this request (from `RequestId::as_uuid()`).
    pub request_id: String,

    /// Current lifecycle state: "active", "completed", "failed", etc. Updated on terminal transition.
    pub state: String,

    /// ISO 8601 timestamp when the entry was first registered in active. Never changes after creation.
    pub created_at: String,

    /// ISO 8601 timestamp of the terminal transition (completed/failed/etc.). `None` while active.
    pub terminal_at: Option<String>,

    /// Bounded classifications captured by trusted lifecycle owners. Older
    /// queued summaries omit this field and deserialize as an empty projection.
    #[serde(default)]
    pub(crate) metadata: RequestSummaryMetadata,
}

impl RequestSummaryEntry {
    pub(crate) fn metadata(&self) -> &RequestSummaryMetadata {
        &self.metadata
    }
}

/// Bounded ledger state attached only to an internal replay-bus entry.
///
/// The SSE protocol projects the canonical lifecycle envelope separately, so
/// this snapshot never crosses the public stream boundary. It gives stream
/// filtering the same request-level fields that the REST ledger uses instead
/// of making filters depend on the particular lifecycle event being replayed.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct RequestSummarySnapshot {
    created_at: String,
    state: String,
    terminal_at: Option<String>,
    metadata: RequestSummaryMetadata,
}

impl RequestSummarySnapshot {
    fn from_entry(entry: &RequestSummaryEntry) -> Self {
        let created_at =
            mesh_llm_events::logging::timestamp::canonical_logging_timestamp(&entry.created_at)
                .unwrap_or_else(|_| entry.created_at.clone());
        Self {
            created_at,
            state: entry.state.clone(),
            terminal_at: entry.terminal_at.clone(),
            metadata: entry.metadata.clone(),
        }
    }

    pub(crate) fn created_at(&self) -> &str {
        &self.created_at
    }

    pub(crate) fn state(&self) -> &str {
        &self.state
    }

    pub(crate) fn terminal_at(&self) -> Option<&str> {
        self.terminal_at.as_deref()
    }

    pub(crate) fn metadata(&self) -> &RequestSummaryMetadata {
        &self.metadata
    }
}

/// Request-ledger membership before and after one replayed lifecycle event.
///
/// Non-terminal events carry only `after`. A terminal event carries both
/// snapshots so a live `outcome=active` subscription is notified to remove a
/// completed request while an `outcome=completed` subscription is notified to
/// add it.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct RequestSummaryEventSnapshots {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    before: Option<RequestSummarySnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    after: Option<RequestSummarySnapshot>,
}

impl RequestSummaryEventSnapshots {
    pub(crate) fn current(entry: &RequestSummaryEntry) -> Self {
        Self {
            before: None,
            after: Some(RequestSummarySnapshot::from_entry(entry)),
        }
    }

    pub(crate) fn terminal(before: &RequestSummaryEntry, after: &RequestSummaryEntry) -> Self {
        Self {
            before: Some(RequestSummarySnapshot::from_entry(before)),
            after: Some(RequestSummarySnapshot::from_entry(after)),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &RequestSummarySnapshot> {
        self.before.iter().chain(self.after.iter())
    }

    pub(crate) fn after(&self) -> Option<&RequestSummarySnapshot> {
        self.after.as_ref()
    }
}

/// A point-in-time, immutable-by-value view of active request summaries.
///
/// The registry copies the current active set while holding its mutex, releases
/// that mutex, and only then sorts and returns the copy.  Consumers therefore
/// never retain a registry lock while serializing, merging with durable rows,
/// or otherwise processing the result.  Entries deliberately contain only the
/// already-sanitized request summary fields; payloads, artifact paths, and
/// attempt payloads do not live in this read model.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ActiveRequestSnapshot {
    entries: Vec<RequestSummaryEntry>,
}

impl ActiveRequestSnapshot {
    /// Borrow the stable, oldest-first active summaries.
    pub fn entries(&self) -> &[RequestSummaryEntry] {
        &self.entries
    }

    /// Consume this snapshot and take ownership of its independently cloned
    /// summaries. Mutating those summaries cannot mutate the registry.
    pub fn into_entries(self) -> Vec<RequestSummaryEntry> {
        self.entries
    }

    /// Return whether the captured active set was empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the number of active summaries captured in this snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
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
    pub fn register_active(&self, mut entry: RequestSummaryEntry) {
        if let Ok(created_at) =
            mesh_llm_events::logging::timestamp::canonical_logging_timestamp(&entry.created_at)
        {
            entry.created_at = created_at;
        }
        let mut map = self.active.lock().expect("registry mutex poisoned");

        // A zero bound deliberately disables active retention. Do not admit a
        // single entry as an accidental off-by-one exception.
        if self.config.max_active == 0 {
            return;
        }

        // Re-registering an existing request updates its summary without
        // evicting an unrelated active request. New requests are bounded.
        if !map.contains_key(&entry.request_id)
            && map.len() >= self.config.max_active
            && evict_oldest(&mut map)
        {
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
            if self.config.max_recent == 0 {
                return;
            }
            if !map.contains_key(&entry.request_id)
                && map.len() >= self.config.max_recent
                && evict_oldest(&mut map)
            {
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

    /// Fill missing metadata for an active or recently terminal request.
    pub(crate) fn merge_metadata(
        &self,
        request_id: &str,
        metadata: RequestSummaryMetadata,
    ) -> Option<RequestSummaryEntry> {
        if metadata.is_empty() {
            return None;
        }

        let mut active = self.active.lock().expect("registry mutex poisoned");
        if let Some(entry) = active.get_mut(request_id) {
            return entry
                .metadata
                .merge_missing(metadata)
                .then(|| entry.clone());
        }
        drop(active);

        let mut recent = self.recent.lock().expect("registry mutex poisoned");
        recent.get_mut(request_id).and_then(|entry| {
            entry
                .metadata
                .merge_missing(metadata)
                .then(|| entry.clone())
        })
    }

    pub(crate) fn merge_authenticated_remote_caller(
        &self,
        request_id: &str,
        metadata: RequestSummaryMetadata,
    ) -> Option<RequestSummaryEntry> {
        let mut active = self.active.lock().expect("registry mutex poisoned");
        if let Some(entry) = active.get_mut(request_id) {
            return entry
                .metadata
                .merge_authenticated_remote_caller(metadata)
                .then(|| entry.clone());
        }
        drop(active);

        let mut recent = self.recent.lock().expect("registry mutex poisoned");
        recent.get_mut(request_id).and_then(|entry| {
            entry
                .metadata
                .merge_authenticated_remote_caller(metadata)
                .then(|| entry.clone())
        })
    }

    /// Atomically move an active request to recent while retaining metadata
    /// merged before its terminal transition.
    pub(crate) fn terminalize(
        &self,
        request_id: &str,
        state: &str,
        terminal_at: String,
    ) -> Option<(RequestSummaryEventSnapshots, RequestSummaryEntry)> {
        let mut active = self.active.lock().expect("registry mutex poisoned");
        let mut entry = active.remove(request_id)?;
        let before = entry.clone();
        entry.state = state.to_owned();
        entry.terminal_at = Some(terminal_at);

        let mut recent = self.recent.lock().expect("registry mutex poisoned");
        if self.config.max_recent != 0 {
            if !recent.contains_key(&entry.request_id)
                && recent.len() >= self.config.max_recent
                && evict_oldest(&mut recent)
            {
                self.recent_evictions.fetch_add(1, Ordering::Relaxed);
            }
            recent.insert(entry.request_id.clone(), entry.clone());
        }
        Some((
            RequestSummaryEventSnapshots::terminal(&before, &entry),
            entry,
        ))
    }

    /// Capture a stable, bounded, oldest-first view of the active set.
    ///
    /// The mutex is held only while cloning the current map. Sorting happens
    /// after it is released, so callers can safely hold the returned snapshot
    /// across durable-store queries without blocking request registration or
    /// terminal movement. Ties on creation timestamp use the request ID so
    /// pagination/active-to-durable merge callers receive deterministic order.
    pub fn snapshot_active(&self) -> ActiveRequestSnapshot {
        let mut entries: Vec<_> = {
            let map = self.active.lock().expect("registry mutex poisoned");
            map.values().cloned().collect()
        };
        entries.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.request_id.cmp(&right.request_id))
        });
        ActiveRequestSnapshot { entries }
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
fn evict_oldest(map: &mut HashMap<String, RequestSummaryEntry>) -> bool {
    let oldest_key = map
        .iter()
        .min_by_key(|(_, entry)| &entry.created_at)
        .map(|(key, _)| key.clone());

    if let Some(key) = oldest_key {
        map.remove(&key);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    fn make_entry(id: &str, ts: u8) -> RequestSummaryEntry {
        RequestSummaryEntry {
            request_id: id.to_string(),
            state: "active".into(),
            created_at: format!("2025-01-01T00:00:{:02}Z", ts),
            terminal_at: None,
            metadata: RequestSummaryMetadata::default(),
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
    // Active snapshots
    // ---------------------------------------------------------------------------

    #[test]
    fn snapshot_active_is_oldest_first_with_stable_tie_breaker() {
        let reg = RequestRegistry::new(RegistryConfig::default());
        reg.register_active(make_entry("request-c", 2));
        reg.register_active(make_entry("request-b", 1));
        reg.register_active(make_entry("request-a", 2));

        let snapshot = reg.snapshot_active();
        let ids: Vec<_> = snapshot
            .entries()
            .iter()
            .map(|entry| entry.request_id.as_str())
            .collect();

        assert_eq!(ids, ["request-b", "request-a", "request-c"]);
    }

    #[test]
    fn snapshot_active_returns_isolated_value_copies() {
        let reg = RequestRegistry::new(RegistryConfig::default());
        reg.register_active(make_entry("request-copy", 1));

        let mut entries = reg.snapshot_active().into_entries();
        entries[0].state = "mutated-by-reader".into();
        entries[0].terminal_at = Some("not-a-registry-write".into());

        let stored = reg.get_active("request-copy").expect("active entry kept");
        assert_eq!(stored.state, "active");
        assert_eq!(stored.terminal_at, None);
    }

    #[test]
    fn snapshot_active_excludes_terminal_entries_after_movement() {
        let reg = RequestRegistry::new(RegistryConfig::default());
        let mut terminal = make_entry("request-terminal", 1);
        let active = make_entry("request-active", 2);
        reg.register_active(terminal.clone());
        reg.register_active(active);

        terminal.state = "cancelled".into();
        terminal.terminal_at = Some("2025-01-01T00:00:03Z".into());
        reg.move_to_recent(terminal);

        let snapshot = reg.snapshot_active();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot.entries()[0].request_id, "request-active");
        assert!(snapshot.entries()[0].terminal_at.is_none());
    }

    #[test]
    fn concurrent_registration_and_terminal_movement_keep_snapshots_bounded() {
        let reg = Arc::new(RequestRegistry::new(RegistryConfig {
            max_active: 8,
            max_recent: 64,
        }));
        let start = Arc::new(Barrier::new(2));
        let writer_registry = Arc::clone(&reg);
        let writer_start = Arc::clone(&start);
        let writer = thread::spawn(move || {
            writer_start.wait();
            for index in 0..64u8 {
                let entry = make_entry(&format!("request-{index:02}"), index);
                writer_registry.register_active(entry.clone());
                if index % 2 == 0 {
                    let mut terminal = entry;
                    terminal.state = "completed".into();
                    terminal.terminal_at = Some(format!("2025-01-01T00:01:{index:02}Z"));
                    writer_registry.move_to_recent(terminal);
                }
            }
        });

        start.wait();
        for _ in 0..64 {
            let snapshot = reg.snapshot_active();
            assert!(snapshot.len() <= 8);
            assert!(
                snapshot
                    .entries()
                    .iter()
                    .all(|entry| { entry.state == "active" && entry.terminal_at.is_none() })
            );
            assert!(snapshot.entries().windows(2).all(|pair| {
                (&pair[0].created_at, &pair[0].request_id)
                    <= (&pair[1].created_at, &pair[1].request_id)
            }));
        }
        writer.join().expect("writer thread panicked");

        let final_snapshot = reg.snapshot_active();
        assert!(final_snapshot.len() <= 8);
        assert!(
            final_snapshot
                .entries()
                .iter()
                .all(|entry| { entry.state == "active" && entry.terminal_at.is_none() })
        );
        assert_eq!(reg.recent_count(), 32);
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
    fn test_zero_active_capacity_never_admits_an_entry() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 0,
            max_recent: 10,
        });

        reg.register_active(make_entry("request-zero", 1));

        assert!(reg.snapshot_active().is_empty());
        assert_eq!(reg.active_count(), 0);
        assert_eq!(reg.active_evictions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_reregistering_at_capacity_does_not_evict_another_request() {
        let reg = RequestRegistry::new(RegistryConfig {
            max_active: 2,
            max_recent: 10,
        });
        reg.register_active(make_entry("request-a", 1));
        reg.register_active(make_entry("request-b", 2));

        let mut replacement = make_entry("request-a", 1);
        replacement.state = "still-active".into();
        reg.register_active(replacement);

        assert_eq!(reg.active_count(), 2);
        assert_eq!(reg.active_evictions.load(Ordering::Relaxed), 0);
        assert_eq!(
            reg.get_active("request-a")
                .expect("replacement retained")
                .state,
            "still-active"
        );
        assert!(reg.get_active("request-b").is_some());
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
                        metadata: RequestSummaryMetadata::default(),
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
