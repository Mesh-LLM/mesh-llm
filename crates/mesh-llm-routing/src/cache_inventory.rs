//! Bounded positive cache evidence and privacy-preserving advertisements.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

pub const CACHE_AFFINITY_DIGEST_BYTES: usize = 16;
pub const CACHE_AFFINITY_SALT_BYTES: usize = 32;
pub const CACHE_AFFINITY_MAX_ENTRIES: usize = 128;
/// Keep evidence valid across the mesh's 60-second heartbeat/gossip cadence.
/// Two intervals tolerate one delayed or missed refresh while remaining far
/// shorter-lived than the removed 20-minute learned-affinity entries.
pub const CACHE_AFFINITY_TTL: Duration = Duration::from_secs(2 * 60);
pub const CACHE_AFFINITY_SALT_ROTATION_MS: u64 = 5 * 60 * 1_000;
pub const CACHE_AFFINITY_MAX_FUTURE_SKEW_MS: u64 = 30_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTier {
    #[default]
    L1,
    L3,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheAffinityEntry {
    pub model: String,
    pub prefix_digest: [u8; CACHE_AFFINITY_DIGEST_BYTES],
    pub matched_tokens: u32,
    pub suffix_prefill_tokens: u32,
    pub tier: CacheTier,
    pub restore_micros: u64,
    pub queue_delay_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheAffinityAdvertisement {
    pub salt: [u8; CACHE_AFFINITY_SALT_BYTES],
    pub epoch: u64,
    pub generated_at_unix_ms: u64,
    pub ttl_ms: u32,
    pub entries: Vec<CacheAffinityEntry>,
}

impl CacheAffinityAdvertisement {
    /// Compare the advertised cache state while ignoring its freshness clock.
    /// Receivers still retain the newer timestamp for expiry decisions.
    pub fn has_same_cache_state(&self, other: &Self) -> bool {
        self.salt == other.salt
            && self.epoch == other.epoch
            && self.ttl_ms == other.ttl_ms
            && self.entries == other.entries
    }

    pub fn probe(
        &self,
        model: &str,
        prefix_hash: u64,
        now_unix_ms: u64,
    ) -> Option<CacheAffinityEntry> {
        if self.generated_at_unix_ms > now_unix_ms.saturating_add(CACHE_AFFINITY_MAX_FUTURE_SKEW_MS)
            || now_unix_ms.saturating_sub(self.generated_at_unix_ms) > u64::from(self.ttl_ms)
        {
            return None;
        }
        let digest = prefix_digest(&self.salt, model, prefix_hash);
        self.entries
            .iter()
            .find(|entry| entry.model == model && entry.prefix_digest == digest)
            .cloned()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CacheKey {
    model: String,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
struct Observation {
    matched_tokens: u32,
    suffix_prefill_tokens: u32,
    tier: CacheTier,
    restore_micros: u64,
    queue_delay_micros: u64,
    observed_at: Instant,
    order: u64,
}

#[derive(Debug)]
pub struct CacheInventory {
    entries: HashMap<CacheKey, Observation>,
    lru: BTreeMap<u64, CacheKey>,
    epoch: u64,
    next_order: u64,
    max_entries: usize,
    ttl: Duration,
}

impl CacheInventory {
    /// Look up local residency directly without rebuilding the gossip digest
    /// set. The digest is not meaningful for a local-only exact-key probe.
    pub fn probe_local(&mut self, model: &str, prefix_hash: u64) -> Option<CacheAffinityEntry> {
        self.prune_expired();
        let key = CacheKey {
            model: model.to_string(),
            prefix_hash,
        };
        let observation = self.entries.get(&key)?;
        Some(CacheAffinityEntry {
            model: key.model,
            prefix_digest: [0; CACHE_AFFINITY_DIGEST_BYTES],
            matched_tokens: observation.matched_tokens,
            suffix_prefill_tokens: observation.suffix_prefill_tokens,
            tier: observation.tier,
            restore_micros: observation.restore_micros,
            queue_delay_micros: observation.queue_delay_micros,
        })
    }

    pub fn record_l1_hit(
        &mut self,
        model: &str,
        prefix_hash: u64,
        matched_tokens: u32,
        suffix_prefill_tokens: u32,
        queue_delay_micros: u64,
    ) {
        if matched_tokens == 0 {
            return;
        }
        self.prune_expired();
        let key = CacheKey {
            model: model.to_string(),
            prefix_hash,
        };
        if let Some(previous) = self.entries.remove(&key) {
            self.lru.remove(&previous.order);
        }
        let order = self.take_order();
        self.entries.insert(
            key.clone(),
            Observation {
                matched_tokens,
                suffix_prefill_tokens,
                tier: CacheTier::L1,
                restore_micros: 0,
                queue_delay_micros,
                observed_at: Instant::now(),
                order,
            },
        );
        self.lru.insert(order, key);
        self.epoch = self.epoch.saturating_add(1);
        while self.entries.len() > self.max_entries {
            let Some((_, oldest)) = self.lru.pop_first() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    pub fn invalidate(&mut self, model: &str, prefix_hash: u64) -> bool {
        let key = CacheKey {
            model: model.to_string(),
            prefix_hash,
        };
        let Some(previous) = self.entries.remove(&key) else {
            return false;
        };
        self.lru.remove(&previous.order);
        self.epoch = self.epoch.saturating_add(1);
        true
    }

    pub fn advertisement(
        &mut self,
        salt: [u8; CACHE_AFFINITY_SALT_BYTES],
        generated_at_unix_ms: u64,
    ) -> CacheAffinityAdvertisement {
        self.prune_expired();
        let entries = self
            .lru
            .iter()
            .rev()
            .filter_map(|(_, key)| {
                self.entries.get(key).map(|observation| CacheAffinityEntry {
                    model: key.model.clone(),
                    prefix_digest: prefix_digest(&salt, &key.model, key.prefix_hash),
                    matched_tokens: observation.matched_tokens,
                    suffix_prefill_tokens: observation.suffix_prefill_tokens,
                    tier: observation.tier,
                    restore_micros: observation.restore_micros,
                    queue_delay_micros: observation.queue_delay_micros,
                })
            })
            .collect();
        CacheAffinityAdvertisement {
            salt,
            epoch: self.epoch,
            generated_at_unix_ms,
            ttl_ms: u32::try_from(self.ttl.as_millis()).unwrap_or(u32::MAX),
            entries,
        }
    }

    fn prune_expired(&mut self) {
        let now = Instant::now();
        while let Some((_, key)) = self.lru.first_key_value() {
            if self
                .entries
                .get(key)
                .is_some_and(|entry| now.duration_since(entry.observed_at) <= self.ttl)
            {
                break;
            }
            let Some((_, key)) = self.lru.pop_first() else {
                break;
            };
            if self.entries.remove(&key).is_some() {
                self.epoch = self.epoch.saturating_add(1);
            }
        }
    }

    fn take_order(&mut self) -> u64 {
        if self.next_order == u64::MAX {
            let mut next = 0;
            let mut rebased = BTreeMap::new();
            for (_, key) in std::mem::take(&mut self.lru) {
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.order = next;
                    rebased.insert(next, key);
                    next += 1;
                }
            }
            self.lru = rebased;
            self.next_order = next;
        }
        let order = self.next_order;
        self.next_order += 1;
        order
    }
}

impl Default for CacheInventory {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            lru: BTreeMap::new(),
            epoch: 0,
            next_order: 0,
            max_entries: CACHE_AFFINITY_MAX_ENTRIES,
            ttl: CACHE_AFFINITY_TTL,
        }
    }
}

pub fn prefix_digest(
    salt: &[u8; CACHE_AFFINITY_SALT_BYTES],
    model: &str,
    prefix_hash: u64,
) -> [u8; CACHE_AFFINITY_DIGEST_BYTES] {
    let key = *blake3::hash(salt).as_bytes();
    let mut hasher = blake3::Hasher::new_keyed(&key);
    hasher.update(b"mesh-llm-cache-affinity-v1\0");
    hasher.update(model.as_bytes());
    hasher.update(&[0]);
    hasher.update(&prefix_hash.to_le_bytes());
    let mut digest = [0; CACHE_AFFINITY_DIGEST_BYTES];
    digest.copy_from_slice(&hasher.finalize().as_bytes()[..CACHE_AFFINITY_DIGEST_BYTES]);
    digest
}

/// Derive a public, node-scoped salt that rotates independently of inventory
/// epochs. Rotation limits how long an observed digest remains linkable while
/// still letting peers probe an advertisement without sharing prompt content.
pub fn rotating_salt(node_id: &[u8], now_unix_ms: u64) -> [u8; CACHE_AFFINITY_SALT_BYTES] {
    let bucket = now_unix_ms / CACHE_AFFINITY_SALT_ROTATION_MS;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mesh-llm-cache-affinity-salt-v1\0");
    hasher.update(node_id);
    hasher.update(&bucket.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertisement_contains_only_salted_digest_and_positive_hits() {
        let mut inventory = CacheInventory::default();
        inventory.record_l1_hit("model", 0xfeed_beef, 512, 32, 10);
        inventory.record_l1_hit("model", 17, 0, 0, 0);

        let advertisement = inventory.advertisement([3; CACHE_AFFINITY_SALT_BYTES], 1_000);

        assert_eq!(advertisement.entries.len(), 1);
        assert_eq!(advertisement.entries[0].matched_tokens, 512);
        assert_eq!(
            advertisement.probe("model", 0xfeed_beef, 1_001),
            Some(advertisement.entries[0].clone())
        );
        assert_eq!(
            advertisement.probe("model", 0xfeed_beef, 61_000),
            Some(advertisement.entries[0].clone()),
            "evidence must survive one 60-second heartbeat interval"
        );
        assert!(advertisement.probe("model", 0xfeed_beef, 121_001).is_none());
        let mut future = advertisement.clone();
        future.generated_at_unix_ms = CACHE_AFFINITY_MAX_FUTURE_SKEW_MS + 1_002;
        assert!(future.probe("model", 0xfeed_beef, 1_001).is_none());
    }

    #[test]
    fn local_probe_returns_exact_residency_without_a_gossip_digest() {
        let mut inventory = CacheInventory::default();
        inventory.record_l1_hit("model", 0xfeed_beef, 512, 32, 10);

        let entry = inventory
            .probe_local("model", 0xfeed_beef)
            .expect("local hit");
        assert_eq!(entry.prefix_digest, [0; CACHE_AFFINITY_DIGEST_BYTES]);
        assert_eq!(entry.matched_tokens, 512);
        assert!(inventory.probe_local("model", 17).is_none());
    }

    #[test]
    fn semantic_state_ignores_only_the_refresh_timestamp() {
        let mut inventory = CacheInventory::default();
        inventory.record_l1_hit("model", 7, 512, 32, 10);
        let first = inventory.advertisement([3; CACHE_AFFINITY_SALT_BYTES], 1_000);
        let mut refreshed = first.clone();
        refreshed.generated_at_unix_ms = 2_000;

        assert!(first.has_same_cache_state(&refreshed));
        refreshed.epoch = refreshed.epoch.saturating_add(1);
        assert!(!first.has_same_cache_state(&refreshed));
    }

    #[test]
    fn digest_rotates_with_salt_and_is_model_scoped() {
        let first = prefix_digest(&[1; CACHE_AFFINITY_SALT_BYTES], "a", 7);
        assert_ne!(
            first,
            prefix_digest(&[2; CACHE_AFFINITY_SALT_BYTES], "a", 7)
        );
        assert_ne!(
            first,
            prefix_digest(&[1; CACHE_AFFINITY_SALT_BYTES], "b", 7)
        );
    }

    #[test]
    fn public_salt_rotates_on_a_bounded_epoch() {
        let node_id = [9; 32];
        let first = rotating_salt(&node_id, 1);
        assert_eq!(
            first,
            rotating_salt(&node_id, CACHE_AFFINITY_SALT_ROTATION_MS - 1)
        );
        assert_ne!(
            first,
            rotating_salt(&node_id, CACHE_AFFINITY_SALT_ROTATION_MS)
        );
    }
}
