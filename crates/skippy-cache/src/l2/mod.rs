//! Host-RAM L2 tier over the packed L3 segment format (#1651).
//!
//! The radix cache (L1) holds resident payloads; the L3 tier holds the same
//! state durably on disk as content-addressed packed segments. This module
//! adds the missing middle tier: a bounded host-RAM cache of *assembled L3
//! entries*, keyed and identified exactly like the L3 entries they mirror.
//!
//! Contract (mirrors `crate::tier::L3Tier`):
//!
//! - **Identity**: entries are stamped with the tier's model and exact-state
//!   identities. The cache key includes the identities, so a re-identity is
//!   a wholesale miss, never a silent hit.
//! - **Coordinates**: entries are keyed by the same
//!   `(namespace, token path)` coordinates L3 uses —
//!   [`crate::tier::l3_prefix_key`] / [`crate::tier::l3_namespace_key`] — so
//!   an L2 hit is interchangeable with the L3 entry it cached.
//! - **Integrity**: stored payloads keep their whole-payload BLAKE3 digest
//!   (the manifest key). Reads verify the digest; a mismatch is recorded,
//!   dropped, and reported as absence — never returned as state.
//! - **Bounded**: the byte budget is enforced on every insert by evicting in
//!   deterministic LRU order. A payload larger than the budget is refused.
//! - **Zero-copy reads**: `get` clones the handle, not the bytes; callers
//!   receive `CacheBytes` mirrors of the stored buffers.
//!
//! This first slice is a standalone store with no wiring into the request
//! path; the benchmark harness drives it directly. L2 promotion/demotion
//! policy and server integration land in a later slice.
use std::collections::HashMap;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::payload::{CacheBytes, ExactStatePayloadKind};

/// Where an entry came from, for telemetry and promotion policy later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Origin {
    /// Copied out of an assembled L3 fill.
    FromL3,
    /// Inserted directly (tests, prefetch, or a future wire source).
    Direct,
}

/// LRU eviction accounting for one removed entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2Eviction {
    pub cache_key: String,
    pub payload_bytes: u64,
}

/// Read path counters. One snapshot per `stats()` call.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct L2Stats {
    pub entries: u64,
    pub bytes: u64,
    pub budget_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub digest_mismatches: u64,
    pub refused_bytes: u64,
}

/// What L2 actually stores: the payload bytes split the way
/// `ExactStatePayload` splits them, so a fill can be rebuilt cheaply.
#[derive(Debug, Clone)]
pub enum ExactStatePayloadMirror {
    FullState {
        bytes: CacheBytes,
    },
    RecurrentOnly {
        recurrent: CacheBytes,
    },
    KvRecurrent {
        kv: CacheBytes,
        recurrent: CacheBytes,
    },
}

impl ExactStatePayloadMirror {
    pub fn from_parts(kind: ExactStatePayloadKind, kv: CacheBytes, recurrent: CacheBytes) -> Self {
        match kind {
            ExactStatePayloadKind::FullState => Self::FullState { bytes: kv },
            ExactStatePayloadKind::RecurrentOnly => Self::RecurrentOnly { recurrent },
            ExactStatePayloadKind::KvRecurrent => Self::KvRecurrent { kv, recurrent },
        }
    }

    pub fn byte_len(&self) -> u64 {
        match self {
            Self::FullState { bytes } => bytes.len(),
            Self::RecurrentOnly { recurrent } => recurrent.len(),
            Self::KvRecurrent { kv, recurrent } => kv.len().saturating_add(recurrent.len()),
        }
    }

    /// Capture a serving payload into a mirror. Any internal block
    /// reconstruction is shared via `CacheBytes` handles, not copied.
    pub fn capture(payload: &crate::payload::ExactStatePayload) -> Self {
        match payload {
            crate::payload::ExactStatePayload::FullState { bytes } => Self::FullState {
                bytes: bytes.clone(),
            },
            crate::payload::ExactStatePayload::RecurrentOnly { recurrent } => Self::RecurrentOnly {
                recurrent: recurrent.clone(),
            },
            crate::payload::ExactStatePayload::KvRecurrent { kv, recurrent } => Self::KvRecurrent {
                kv: kv.clone(),
                recurrent: recurrent.clone(),
            },
        }
    }

    /// Rebuild a serving payload from the mirror. Cheap: `CacheBytes` is
    /// `Arc`-backed, so this shares the stored buffers rather than copying.
    pub fn to_payload(&self) -> crate::payload::ExactStatePayload {
        match self {
            Self::FullState { bytes } => crate::payload::ExactStatePayload::FullState {
                bytes: bytes.clone(),
            },
            Self::RecurrentOnly { recurrent } => crate::payload::ExactStatePayload::RecurrentOnly {
                recurrent: recurrent.clone(),
            },
            Self::KvRecurrent { kv, recurrent } => crate::payload::ExactStatePayload::KvRecurrent {
                kv: kv.clone(),
                recurrent: recurrent.clone(),
            },
        }
    }
}

/// Verified hit.
#[derive(Debug, Clone)]
pub struct L2Hit {
    pub payload: ExactStatePayloadMirror,
    pub token_count: u64,
    pub payload_digest: String,
}

/// Presence probe result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2Peek {
    pub token_count: u64,
    pub payload_digest: String,
    pub payload_bytes: u64,
    pub origin: L2Origin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L2InsertRefusal {
    EmptyPayload,
    OverBudget { payload_bytes: u64 },
    MalformedDigest,
}

impl L2InsertRefusal {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::EmptyPayload => "refusing to cache an empty exact-state payload",
            Self::OverBudget { .. } => {
                "payload exceeds the entire L2 budget; caching it would evict everything else"
            }
            Self::MalformedDigest => "payload digest is not a 64-hex-character blake3 string",
        }
    }
}

#[derive(Debug)]
struct L2Entry {
    payload: ExactStatePayloadMirror,
    token_count: u64,
    payload_digest: String,
    origin: L2Origin,
    /// LRU clock, bumped on every hit/probe.
    last_used: u64,
    payload_bytes: u64,
}

#[derive(Default)]
struct L2Inner {
    map: HashMap<String, L2Entry>,
    bytes: u64,
    clock: u64,
}

/// Counters kept outside the map lock so `stats()` never blocks hits.
#[derive(Default)]
struct L2AtomicStats {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
    digest_mismatches: AtomicU64,
    refused_bytes: AtomicU64,
}

/// Bounded host-RAM L2 of assembled exact-state payloads.
pub struct L2Tier {
    inner: Mutex<L2Inner>,
    budget_bytes: u64,
    stats: L2AtomicStats,
}

impl L2Tier {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(L2Inner::default()),
            budget_bytes,
            stats: L2AtomicStats::default(),
        }
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Insert an assembled entry. `payload_digest` is the whole-payload
    /// BLAKE3 (the L3 manifest key) and is verified on read. Returns the
    /// evictions the insert caused, so callers and tests can assert policy.
    pub fn insert(
        &self,
        cache_key: String,
        token_count: u64,
        payload_digest: String,
        payload: ExactStatePayloadMirror,
        origin: L2Origin,
    ) -> Result<Vec<L2Eviction>, L2InsertRefusal> {
        let payload_bytes = payload.byte_len();
        if payload_bytes == 0 {
            // Mirrors L3: an empty payload cannot represent state.
            return Err(L2InsertRefusal::EmptyPayload);
        }
        if payload_bytes > self.budget_bytes {
            self.stats
                .refused_bytes
                .fetch_add(payload_bytes, Ordering::Relaxed);
            return Err(L2InsertRefusal::OverBudget { payload_bytes });
        }
        let digest_is_hex =
            payload_digest.len() == 64 && payload_digest.bytes().all(|b| b.is_ascii_hexdigit());
        if !digest_is_hex {
            return Err(L2InsertRefusal::MalformedDigest);
        }
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        // One entry per cache key: a re-insert at the same coordinates is a
        // replacement (fresher state for the same prefix), not a duplicate.
        if let Some(existing) = inner.map.remove(&cache_key) {
            inner.bytes = inner.bytes.saturating_sub(existing.payload_bytes);
        }
        inner.clock = inner.clock.wrapping_add(1);
        let last_used = inner.clock;
        inner.map.insert(
            cache_key.clone(),
            L2Entry {
                payload,
                token_count,
                payload_digest,
                origin,
                last_used,
                payload_bytes,
            },
        );
        inner.bytes = inner.bytes.saturating_add(payload_bytes);
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        let evictions = self.evict_to_budget(&mut inner, &cache_key);
        Ok(evictions)
    }

    /// A hit records recency and returns a clone of the stored mirror
    /// (handle clones, not byte copies). A digest mismatch drops the entry
    /// and counts as a miss: L2 must never serve state it cannot verify.
    pub fn get(&self, cache_key: &str, expected_digest: &str) -> Option<L2Hit> {
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        if inner
            .map
            .get(cache_key)
            .is_none_or(|entry| entry.payload_digest != expected_digest)
        {
            // Digest mismatch drops the unverifiable entry; plain absence
            // falls through as a recorded miss.
            if let Some(removed) = inner.map.remove(cache_key) {
                debug_assert!(removed.payload_digest != expected_digest);
                inner.bytes = inner.bytes.saturating_sub(removed.payload_bytes);
                self.stats.digest_mismatches.fetch_add(1, Ordering::Relaxed);
            }
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        inner.clock = inner.clock.wrapping_add(1);
        let now = inner.clock;
        let entry = inner
            .map
            .get_mut(cache_key)
            .expect("presence checked above");
        entry.last_used = now;
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        Some(L2Hit {
            payload: entry.payload.clone(),
            token_count: entry.token_count,
            payload_digest: entry.payload_digest.clone(),
        })
    }

    /// Presence probe without byte or digest work — the L2 equivalent of an
    /// L3 index probe. The caller still `get`s with the expected digest
    /// before serving state.
    pub fn peek(&self, cache_key: &str) -> Option<L2Peek> {
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        inner.clock += 1;
        let now = inner.clock;
        let entry = inner.map.get_mut(cache_key)?;
        entry.last_used = now;
        let peek = L2Peek {
            token_count: entry.token_count,
            payload_digest: entry.payload_digest.clone(),
            payload_bytes: entry.payload_bytes,
            origin: entry.origin,
        };
        Some(peek)
    }

    pub fn remove(&self, cache_key: &str) -> Option<L2Eviction> {
        let mut inner = self.inner.lock().expect("L2 map poisoned");
        let removed = inner.map.remove(cache_key)?;
        inner.bytes = inner.bytes.saturating_sub(removed.payload_bytes);
        Some(L2Eviction {
            cache_key: cache_key.to_string(),
            payload_bytes: removed.payload_bytes,
        })
    }

    /// Drop everything; returns the bytes released.
    pub fn clear(&self) -> u64 {
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        let bytes = inner.bytes;
        inner.map.clear();
        inner.bytes = 0;
        bytes
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("L2 map lock poisoned").map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Point-in-time snapshot combining atomics with the locked totals.
    pub fn stats(&self) -> L2Stats {
        let inner = self.inner.lock().expect("L2 map lock poisoned");
        L2Stats {
            entries: inner.map.len() as u64,
            bytes: inner.bytes,
            budget_bytes: self.budget_bytes,
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            digest_mismatches: self.stats.digest_mismatches.load(Ordering::Relaxed),
            refused_bytes: self.stats.refused_bytes.load(Ordering::Relaxed),
        }
    }

    fn evict_to_budget(&self, inner: &mut L2Inner, protect_key: &str) -> Vec<L2Eviction> {
        let mut evictions = Vec::new();
        while inner.bytes > self.budget_bytes {
            // Deterministic LRU: lowest last_used wins; ties break on cache
            // key so identical operation sequences produce identical
            // evictions.
            let victim = inner
                .map
                .iter()
                .filter(|(key, _)| key.as_str() != protect_key)
                .min_by(|a, b| a.1.last_used.cmp(&b.1.last_used).then_with(|| a.0.cmp(b.0)))
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else { break };
            if let Some(removed) = inner.map.remove(&victim) {
                inner.bytes = inner.bytes.saturating_sub(removed.payload_bytes);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                evictions.push(L2Eviction {
                    cache_key: victim,
                    payload_bytes: removed.payload_bytes,
                });
            } else {
                break;
            }
        }
        evictions
    }
}

/// Build the L2 cache key from the same coordinates L3 uses, plus the
/// identities the tier serves. Same coordinates under different identities
/// get different keys: an identity change cannot cross-contaminate.
pub fn l2_cache_key(
    model_identity: &str,
    state_identity: &str,
    namespace: &str,
    token_ids: &[i32],
) -> String {
    let namespace_key = crate::tier::l3_namespace_key(namespace);
    let prefix_key = crate::tier::l3_prefix_key(namespace, token_ids);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"l2-cache-key-v1");
    hasher.update(model_identity.as_bytes());
    hasher.update(b"\0");
    hasher.update(state_identity.as_bytes());
    hasher.update(b"\0");
    hasher.update(namespace_key.as_bytes());
    hasher.update(prefix_key.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "a616719e0a0d39dc0fe85cd2d0a5e0e2f5e6e10b6b5a0a6f1a1c1d3e5f708a90";
    const DIGEST_B: &str = "b616719e0a0d39dc0fe85cd2d0a5e0e2f5e6e10b6b5a0a6f1a1c1d3e5f708a90";

    fn full_state_mirror(len: usize) -> ExactStatePayloadMirror {
        ExactStatePayloadMirror::FullState {
            bytes: CacheBytes::inline(vec![7u8; len]),
        }
    }

    fn key(namespace: &str, tokens: &[i32]) -> String {
        l2_cache_key("model-a", "state-a", namespace, tokens)
    }

    #[test]
    fn insert_get_round_trip_verifies_digest() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[1, 2, 3]);
        tier.insert(
            k.clone(),
            3,
            DIGEST_A.to_string(),
            full_state_mirror(64),
            L2Origin::FromL3,
        )
        .expect("insert must fit");
        let hit = tier.get(&k, DIGEST_A).expect("digest match must hit");
        assert_eq!(hit.token_count, 3);
        assert_eq!(hit.payload.byte_len(), 64);
        // Round-trips into a serving payload with the right byte count.
        let payload = hit.payload.to_payload();
        assert_eq!(payload.byte_len(), 64);
        assert_eq!(
            payload.kind(),
            crate::payload::ExactStatePayloadKind::FullState
        );
    }

    #[test]
    fn digest_mismatch_drops_entry_and_misses() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[1, 2, 3]);
        tier.insert(
            k.clone(),
            3,
            DIGEST_A.to_string(),
            full_state_mirror(64),
            L2Origin::FromL3,
        )
        .expect("insert must fit");
        // Wrong expected digest: must be a miss, and the unverifiable entry
        // must be gone afterward.
        assert!(tier.get(&k, DIGEST_B).is_none());
        assert!(tier.peek(&k).is_none(), "mismatched entry must be dropped");
        let stats = tier.stats();
        assert_eq!(stats.digest_mismatches, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.bytes, 0);
    }

    #[test]
    fn budget_evicts_lru_first_and_never_the_protected_entry() {
        let tier = L2Tier::new(256);
        let k1 = key("ns", &[1]);
        let k2 = key("ns", &[2]);
        let k3 = key("ns", &[3]);
        tier.insert(
            k1.clone(),
            1,
            DIGEST_A.to_string(),
            full_state_mirror(100),
            L2Origin::FromL3,
        )
        .expect("k1 fits");
        tier.insert(
            k2.clone(),
            1,
            DIGEST_A.to_string(),
            full_state_mirror(100),
            L2Origin::FromL3,
        )
        .expect("k2 fits");
        // Touch k1 so k2 becomes the LRU victim.
        assert!(tier.get(&k1, DIGEST_A).is_some());
        let evictions = tier
            .insert(
                k3.clone(),
                1,
                DIGEST_A.to_string(),
                full_state_mirror(100),
                L2Origin::FromL3,
            )
            .expect("k3 fits after eviction");
        assert_eq!(
            evictions.len(),
            1,
            "one entry must be evicted: {evictions:?}"
        );
        assert_eq!(evictions[0].cache_key, k2, "LRU victim is k2");
        assert_eq!(evictions[0].payload_bytes, 100);
        assert!(tier.peek(&k1).is_some(), "recently used k1 survives");
        assert!(tier.peek(&k3).is_some(), "just-inserted k3 survives");
        assert!(tier.peek(&k2).is_none(), "k2 was evicted");
        let stats = tier.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.bytes, 200, "bytes must track entries exactly");
    }

    #[test]
    fn oversized_payload_is_refused_without_evicting() {
        let tier = L2Tier::new(128);
        let k1 = key("ns", &[1]);
        tier.insert(
            k1.clone(),
            1,
            DIGEST_A.to_string(),
            full_state_mirror(64),
            L2Origin::FromL3,
        )
        .expect("fits");
        let err = tier
            .insert(
                key("ns", &[2]),
                1,
                DIGEST_A.to_string(),
                full_state_mirror(129),
                L2Origin::FromL3,
            )
            .expect_err("over-budget payload must be refused");
        assert_eq!(err, L2InsertRefusal::OverBudget { payload_bytes: 129 });
        assert!(tier.peek(&k1).is_some(), "refusal must not evict anything");
        assert_eq!(tier.stats().refused_bytes, 129);
    }

    #[test]
    fn empty_payload_is_refused_like_l3() {
        let tier = L2Tier::new(1 << 20);
        let err = tier
            .insert(
                key("ns", &[1]),
                1,
                DIGEST_A.to_string(),
                full_state_mirror(0),
                L2Origin::FromL3,
            )
            .expect_err("empty payloads must be refused");
        assert_eq!(err, L2InsertRefusal::EmptyPayload);
        assert!(tier.is_empty());
    }

    #[test]
    fn malformed_digest_is_refused() {
        let tier = L2Tier::new(1 << 20);
        let err = tier
            .insert(
                key("ns", &[1]),
                1,
                "not-a-digest".to_string(),
                full_state_mirror(16),
                L2Origin::FromL3,
            )
            .expect_err("malformed digest must be refused");
        assert_eq!(err, L2InsertRefusal::MalformedDigest);
    }

    #[test]
    fn reinsert_replaces_and_keeps_accounting_exact() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[9]);
        tier.insert(
            k.clone(),
            3,
            DIGEST_A.to_string(),
            full_state_mirror(100),
            L2Origin::FromL3,
        )
        .expect("first insert");
        let evictions = tier
            .insert(
                k.clone(),
                3,
                DIGEST_B.to_string(),
                full_state_mirror(40),
                L2Origin::FromL3,
            )
            .expect("replacement");
        assert!(evictions.is_empty());
        assert_eq!(tier.len(), 1);
        assert_eq!(tier.stats().bytes, 40, "replacement must release old bytes");
        // New digest is the one served now.
        assert!(tier.get(&k, DIGEST_B).is_some());
        assert!(tier.get(&k, DIGEST_A).is_none());
    }

    #[test]
    fn identical_coordinates_under_different_identities_get_different_keys() {
        let a = l2_cache_key("model-a", "state-a", "ns", &[1, 2]);
        let b = l2_cache_key("model-b", "state-a", "ns", &[1, 2]);
        let c = l2_cache_key("model-a", "state-b", "ns", &[1, 2]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        // Same coordinates, same identities: stable key.
        let a2 = l2_cache_key("model-a", "state-a", "ns", &[1, 2]);
        assert_eq!(a, a2);
        // Different token paths differ.
        assert_ne!(a, l2_cache_key("model-a", "state-a", "ns", &[1, 3]));
    }

    #[test]
    fn capture_round_trips_every_payload_kind() {
        let full = crate::payload::ExactStatePayload::full_state(vec![1; 32]);
        let rec = crate::payload::ExactStatePayload::recurrent_only(vec![2; 16]);
        let kvrec = crate::payload::ExactStatePayload::kv_recurrent(vec![3; 24], vec![4; 8]);

        for payload in [&full, &rec, &kvrec] {
            let mirror = ExactStatePayloadMirror::capture(payload);
            assert_eq!(mirror.byte_len(), payload.byte_len());
            let rebuilt = mirror.to_payload();
            assert_eq!(rebuilt.byte_len(), payload.byte_len());
            assert_eq!(rebuilt.kind(), payload.kind());
        }

        // Byte-for-byte identity survives the mirror for kv-recurrent.
        let mirror = ExactStatePayloadMirror::capture(&kvrec);
        let rebuilt = mirror.to_payload();
        let original_kv = kvrec
            .kv_bytes()
            .expect("kv bytes")
            .map(|cow| cow.into_owned())
            .unwrap_or_default();
        let rebuilt_kv = rebuilt
            .kv_bytes()
            .expect("kv bytes")
            .map(|cow| cow.into_owned())
            .unwrap_or_default();
        assert_eq!(original_kv, rebuilt_kv);
    }

    #[test]
    fn clear_releases_everything_and_reports_bytes() {
        let tier = L2Tier::new(1 << 20);
        for i in 0..5i32 {
            tier.insert(
                key("ns", &[i]),
                1,
                DIGEST_A.to_string(),
                full_state_mirror(64),
                L2Origin::FromL3,
            )
            .expect("fits");
        }
        let released = tier.clear();
        assert_eq!(released, 320);
        assert!(tier.is_empty());
        assert_eq!(tier.stats().bytes, 0);
    }

    #[test]
    fn remove_is_exact() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[4]);
        tier.insert(
            k.clone(),
            1,
            DIGEST_A.to_string(),
            full_state_mirror(64),
            L2Origin::FromL3,
        )
        .expect("fits");
        let removed = tier.remove(&k).expect("present entry removes");
        assert_eq!(removed.payload_bytes, 64);
        assert!(tier.remove(&k).is_none(), "second remove is None");
        assert_eq!(tier.stats().bytes, 0);
    }
}
