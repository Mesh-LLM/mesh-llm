//! Host-RAM L2 tier over the packed L3 segment format (#1651).
//!
//! The radix cache (L1) holds resident payloads; the L3 tier holds the same
//! state durably on disk as content-addressed packed segments. This module
//! adds the missing middle tier: a bounded host-RAM cache of *immutable
//! packed segments*, keyed and identified exactly like the L3 entries they
//! mirror.
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
//! - **Segment sharing**: an entry stores immutable `Arc<Vec<u8>>` segment
//!   handles keyed by their content digests plus a layout that maps the
//!   entry's L3 manifest segment list onto those handles. A longer prefix
//!   that extends a shorter one references the same segment handles, so
//!   turn growth shares prefix bytes instead of duplicating them — L2 RAM
//!   tracks *distinct segment* bytes, not per-prefix assembled bytes.
//! - **Integrity**: the whole concatenated L3 wire digest (the manifest key)
//!   is verified exactly once, at admission, against the payload being
//!   admitted. After admission the segment bytes are immutable, so every
//!   later read is a digest lookup plus handle assembly — no re-hash. An
//!   admission-time mismatch refuses the insert; L2 never holds bytes it
//!   did not verify.
//! - **Bounded**: the byte budget is charged with each entry's *distinct*
//!   segment bytes (bytes not already held by an in-flight insert) and
//!   enforced by evicting in deterministic LRU order. A payload whose
//!   distinct bytes exceed the whole budget is refused. A segment shared
//!   with an already-admitted entry is shared for accounting too: only the
//!   first admission pays for it.
//! - **Zero-copy reads**: `get` clones handles, not bytes; the returned
//!   `CacheBytes` is a block-backed view over the shared segment storages,
//!   contiguous in the single-segment case.
//!
//! This first slice is a standalone store with no wiring into the request
//! path; the benchmark harness drives it directly. L2 promotion/demotion
//! policy and server integration land in a later slice.
use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::payload::{CacheBytes, ExactStatePayloadKind};
use crate::{HandoffManifest, segment_digest};
#[cfg(test)]
use crate::{HandoffSegmentRef, MANIFEST_VERSION};

/// Where an entry came from, for telemetry and promotion policy later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Origin {
    /// Admitted from an assembled L3 fill (verified wire).
    FromL3,
    /// Admitted from another verified wire source (tests, prefetch).
    Direct,
}

/// LRU eviction accounting for one removed entry.
///
/// `freed_bytes` is what removal actually released: segments whose last
/// referencing entry left the tier. `retained_bytes` is shared-segment
/// bytes that stay because another entry still references them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2Eviction {
    pub cache_key: String,
    pub freed_bytes: u64,
    pub retained_bytes: u64,
}

impl L2Eviction {
    /// Bytes charged to the budget for this entry (what its removal freed).
    pub fn payload_bytes(&self) -> u64 {
        self.freed_bytes
    }
}

/// Read path counters. One snapshot per `stats()` call.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct L2Stats {
    pub entries: u64,
    /// Sum of entries' distinct segment charges — the live budget usage.
    pub bytes: u64,
    /// Sum of entry payload lengths including cross-entry sharing; larger
    /// than `bytes` exactly when entries share prefix segments.
    pub logical_bytes: u64,
    /// Distinct immutable segment handles currently held.
    pub segments: u64,
    /// Bytes held in the segment pool (== `bytes` when the pool is live).
    pub segment_bytes: u64,
    /// Distinct segment bytes a single admission did not have to copy
    /// because an earlier admission already held them.
    pub shared_bytes_admitted: u64,
    pub budget_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
    /// Admissions refused because the payload digest did not match the
    /// bytes (hash mismatch or malformed digest string).
    pub admission_rejects: u64,
    pub refused_bytes: u64,
}

/// One assembled L2 entry: which stored segment handles make up the wire,
/// in manifest order, plus the payload split so a fill can be rebuilt.
#[derive(Debug, Clone)]
pub struct L2Layout {
    pub payload_kind: ExactStatePayloadKind,
    pub total_bytes: u64,
    pub kv_bytes: u64,
    pub recurrent_bytes: u64,
    /// `(segment digest, byte range within the assembled wire)` per
    /// manifest segment, in manifest order. Ranges concatenate to
    /// `0..total_bytes` exactly as the L3 manifest tiles them.
    pub segments: Vec<(String, Range<u64>)>,
}

/// The L2 mirror of an assembled L3 entry: verified segment handles plus
/// the layout needed to rebuild a serving payload without disk I/O.
#[derive(Debug, Clone)]
pub enum ExactStatePayloadMirror {
    FullState { layout: L2Layout },
    RecurrentOnly { layout: L2Layout },
    KvRecurrent { layout: L2Layout },
}

impl ExactStatePayloadMirror {
    pub fn kind(&self) -> ExactStatePayloadKind {
        match self {
            Self::FullState { .. } => ExactStatePayloadKind::FullState,
            Self::RecurrentOnly { .. } => ExactStatePayloadKind::RecurrentOnly,
            Self::KvRecurrent { .. } => ExactStatePayloadKind::KvRecurrent,
        }
    }

    /// Total wire length of the entry (the assembled payload length).
    pub fn byte_len(&self) -> u64 {
        match self {
            Self::FullState { layout }
            | Self::RecurrentOnly { layout }
            | Self::KvRecurrent { layout } => layout.total_bytes,
        }
    }

    /// Build a mirror from a captured L3 manifest. Callers must verify the
    /// payload wire against `manifest.payload_digest` — `admit` does this —
    /// before the mirror is stored.
    pub fn from_manifest(manifest: &HandoffManifest) -> Result<Self, L2InsertRefusal> {
        let kind = match manifest.payload_kind.as_str() {
            "full-state" => ExactStatePayloadKind::FullState,
            "recurrent-only" => ExactStatePayloadKind::RecurrentOnly,
            "kv-recurrent" => ExactStatePayloadKind::KvRecurrent,
            other => {
                return Err(L2InsertRefusal::UnknownPayloadKind(other.to_string()));
            }
        };
        let mut offset = 0u64;
        let segments =
            manifest
                .segments
                .iter()
                .map(|segment| {
                    let start = offset;
                    offset = offset.checked_add(segment.bytes).ok_or(
                        L2InsertRefusal::MalformedManifest("segment tiling overflows".to_string()),
                    )?;
                    Ok((segment.digest.clone(), start..offset))
                })
                .collect::<Result<Vec<_>, _>>()?;
        if offset != manifest.total_bytes {
            return Err(L2InsertRefusal::MalformedManifest(format!(
                "segments tile {offset} bytes but the manifest records {}",
                manifest.total_bytes
            )));
        }
        let layout = L2Layout {
            payload_kind: kind,
            total_bytes: manifest.total_bytes,
            kv_bytes: manifest.kv_bytes,
            recurrent_bytes: manifest.recurrent_bytes,
            segments,
        };
        Ok(match kind {
            ExactStatePayloadKind::FullState => Self::FullState { layout },
            ExactStatePayloadKind::RecurrentOnly => Self::RecurrentOnly { layout },
            ExactStatePayloadKind::KvRecurrent => Self::KvRecurrent { layout },
        })
    }

    fn layout(&self) -> &L2Layout {
        match self {
            Self::FullState { layout }
            | Self::RecurrentOnly { layout }
            | Self::KvRecurrent { layout } => layout,
        }
    }

    /// Segment digests in wire order, deduplicated.
    fn segment_digests(&self) -> Vec<&str> {
        let mut seen = Vec::new();
        for (digest, _) in &self.layout().segments {
            if !seen.contains(&digest.as_str()) {
                seen.push(digest.as_str());
            }
        }
        seen
    }
}

/// A hit handed to the caller: the entry's layout plus `Arc` clones of the
/// segment handles the layout references, keyed by digest. The tier stores
/// this directly on `L2Hit` so payload assembly needs no tier lock.
#[derive(Debug, Clone)]
pub struct L2Hit {
    pub payload: ExactStatePayloadMirror,
    pub token_count: u64,
    pub payload_digest: String,
    /// Distinct segment handles referenced by the layout, keyed by digest.
    pub(crate) segments: HashMap<String, SegmentHandle>,
}

impl L2Hit {
    /// Rebuild a serving payload. Cheap in the common cases: the returned
    /// `CacheBytes` is a block-backed view sharing the stored segment
    /// storages (`Arc` clones, not byte copies); a single whole-storage
    /// segment borrows it contiguously. Only a multi-segment read of
    /// distinct storages materializes bytes, and only into the caller's
    /// `Cow` on `as_cow`.
    pub fn to_payload(&self) -> crate::payload::ExactStatePayload {
        let layout = self.payload.layout();
        let wire = self.wire_view(0..layout.total_bytes);
        match self.payload.kind() {
            crate::payload::ExactStatePayloadKind::FullState => {
                crate::payload::ExactStatePayload::FullState { bytes: wire }
            }
            crate::payload::ExactStatePayloadKind::RecurrentOnly => {
                crate::payload::ExactStatePayload::RecurrentOnly { recurrent: wire }
            }
            crate::payload::ExactStatePayloadKind::KvRecurrent => {
                // Split the wire at kv_bytes exactly like L3 load does: kv
                // is the leading block-backed view, recurrent the tail. Both
                // share the same storages; no bytes are copied here.
                let kv_len = layout.kv_bytes.min(layout.total_bytes);
                let kv = self.wire_view(0..kv_len);
                let recurrent = self.wire_view(kv_len..layout.total_bytes);
                crate::payload::ExactStatePayload::KvRecurrent { kv, recurrent }
            }
        }
    }

    /// Block-backed `CacheBytes` over `range` of the assembled wire, in
    /// wire order. Blocks outside `range` are skipped; edge blocks are
    /// narrowed to the overlap. Byte-identical views share the same
    /// segment storages; nothing is copied.
    pub(crate) fn wire_view(&self, range: Range<u64>) -> CacheBytes {
        let layout = self.payload.layout();
        let start = range.start.min(layout.total_bytes);
        let end = range.end.min(layout.total_bytes);
        let blocks = layout
            .segments
            .iter()
            .filter_map(|(digest, segment_range)| {
                let block_start = segment_range.start.max(start);
                let block_end = segment_range.end.min(end);
                if block_start >= block_end {
                    return None;
                }
                let storage = self
                    .segments
                    .get(digest)
                    .map(|handle| Arc::clone(&handle.bytes))
                    .unwrap_or_else(|| Arc::new(Vec::new()));
                let len = storage.len() as u64;
                let from = (block_start - segment_range.start).min(len);
                let to = (block_end - segment_range.start).min(len);
                Some((digest.clone(), storage, (from as usize)..(to as usize)))
            })
            .collect::<Vec<_>>();
        CacheBytes::from_shared_blocks(end.saturating_sub(start), blocks)
    }
}

/// Presence probe result. Probing never changes LRU recency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2Peek {
    pub token_count: u64,
    pub payload_digest: String,
    /// Total wire length including segments shared with other entries.
    pub payload_bytes: u64,
    /// Distinct segment bytes charged to the budget for this entry.
    pub distinct_bytes: u64,
    pub origin: L2Origin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L2InsertRefusal {
    EmptyPayload,
    OverBudget {
        payload_bytes: u64,
    },
    MalformedDigest,
    /// Admission hashing found the wire's BLAKE3 different from the digest
    /// the payload claims (the L3 manifest key).
    DigestMismatch {
        expected: String,
        actual: String,
    },
    UnknownPayloadKind(String),
    MalformedManifest(String),
}

impl L2InsertRefusal {
    pub fn reason(&self) -> String {
        match self {
            Self::EmptyPayload => "refusing to cache an empty exact-state payload".to_string(),
            Self::OverBudget { payload_bytes } => format!(
                "payload of {payload_bytes} distinct bytes exceeds the entire L2 budget; \
                 caching it would evict everything else"
            ),
            Self::MalformedDigest => {
                "payload digest is not a 64-hex-character blake3 string".to_string()
            }
            Self::DigestMismatch { expected, actual } => format!(
                "admission digest check failed: wire hashes to {actual} but the payload \
                 claims {expected}"
            ),
            Self::UnknownPayloadKind(kind) => {
                format!("manifest holds unknown payload kind {kind}")
            }
            Self::MalformedManifest(detail) => {
                format!("malformed L3 manifest: {detail}")
            }
        }
    }
}

/// An immutable segment: content-addressed bytes shared by `Arc`.
#[derive(Debug, Clone)]
pub(crate) struct SegmentHandle {
    pub bytes: Arc<Vec<u8>>,
}

#[derive(Debug)]
struct L2Entry {
    payload: ExactStatePayloadMirror,
    token_count: u64,
    payload_digest: String,
    origin: L2Origin,
    /// LRU clock, bumped on successful hits only (probes are side-effect
    /// free).
    last_used: u64,
    /// Distinct segment bytes charged against the budget. Shared segments
    /// already held by other entries are not charged here.
    charge_bytes: u64,
    /// Total wire length including shared segments (telemetry).
    payload_bytes: u64,
}

#[derive(Default)]
struct L2Inner {
    map: HashMap<String, L2Entry>,
    /// Content-addressed pool of immutable segments.
    segments: HashMap<String, SegmentHandle>,
    /// Distinct segment bytes in the pool — the real RAM footprint.
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
    admission_rejects: AtomicU64,
    refused_bytes: AtomicU64,
    shared_bytes_admitted: AtomicU64,
}

/// Bounded host-RAM L2 over immutable packed L3 segments.
pub struct L2Tier {
    inner: Mutex<L2Inner>,
    budget_bytes: u64,
    stats: L2AtomicStats,
}

/// A digest string must be a BLAKE3 hex digest: `blake3:`-prefixed (as L3
/// digests are) or bare 64 hex characters.
fn is_valid_digest(digest: &str) -> bool {
    let hex = digest.strip_prefix("blake3:").unwrap_or(digest);
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
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

    /// Admit an assembled entry.
    ///
    /// `wire` is the payload's concatenated L3 wire — the exact bytes whose
    /// BLAKE3 is the manifest key. Admission verifies
    /// `segment_digest(&wire) == payload_digest` exactly once and refuses
    /// the insert on mismatch: L2 never holds bytes it did not verify.
    /// After admission the segment bytes are immutable, so reads are a
    /// digest lookup plus handle assembly — no re-hash.
    ///
    /// Returns the evictions the admission caused, so callers and tests can
    /// assert policy. The budget is charged with the entry's *distinct*
    /// segment bytes: segments already held by another entry are shared,
    /// not duplicated, and only the first admission pays for them.
    pub fn admit(
        &self,
        cache_key: String,
        token_count: u64,
        payload_digest: String,
        wire: &[u8],
        mirror: ExactStatePayloadMirror,
        origin: L2Origin,
    ) -> Result<Vec<L2Eviction>, L2InsertRefusal> {
        if !is_valid_digest(&payload_digest) {
            self.stats.admission_rejects.fetch_add(1, Ordering::Relaxed);
            return Err(L2InsertRefusal::MalformedDigest);
        }
        // The one integrity check: the wire must hash to the claimed
        // manifest-key digest.
        let actual = segment_digest(wire);
        if actual != payload_digest {
            self.stats.admission_rejects.fetch_add(1, Ordering::Relaxed);
            return Err(L2InsertRefusal::DigestMismatch {
                expected: payload_digest,
                actual,
            });
        }
        let payload_bytes = mirror.byte_len();
        if payload_bytes == 0 {
            // Mirrors L3: an empty payload cannot represent state.
            return Err(L2InsertRefusal::EmptyPayload);
        }
        if payload_bytes != wire.len() as u64 {
            return Err(L2InsertRefusal::MalformedManifest(format!(
                "mirror claims {payload_bytes} payload bytes but the verified wire holds {}",
                wire.len()
            )));
        }
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        // Distinct-byte charge: segments the pool already holds (shared
        // prefix with another entry) cost nothing new, provided the pool's
        // copy still covers the segment's full length. Anything else is cut
        // out of the verified wire.
        let mut new_segments: Vec<(String, SegmentHandle)> = Vec::new();
        let mut new_bytes = 0u64;
        let mut shared_bytes = 0u64;
        for (digest, range) in &mirror.layout().segments {
            let expected_len = (range.end.saturating_sub(range.start)) as usize;
            if let Some(handle) = inner.segments.get(digest) {
                if handle.bytes.len() == expected_len {
                    shared_bytes += expected_len as u64;
                    continue;
                }
                // Pool copy disagrees with the verified wire: replace it.
                let stale = inner.segments.remove(digest);
                if let Some(handle) = stale {
                    inner.bytes = inner.bytes.saturating_sub(handle.bytes.len() as u64);
                }
            }
            if let Some(position) = new_segments.iter().position(|(d, _)| d == digest) {
                // Within-admission duplicate: keep the first copy.
                let existing = &new_segments[position].1;
                if existing.bytes.len() == expected_len {
                    shared_bytes += expected_len as u64;
                    continue;
                }
                new_bytes = new_bytes.saturating_sub(existing.bytes.len() as u64);
                new_segments.remove(position);
            }
            let start = (range.start as usize).min(wire.len());
            let end = (range.end as usize).min(wire.len());
            let bytes = Arc::new(wire[start..end].to_vec());
            new_bytes = new_bytes.saturating_add(bytes.len() as u64);
            new_segments.push((digest.to_string(), SegmentHandle { bytes }));
        }
        if new_bytes > self.budget_bytes {
            self.stats
                .refused_bytes
                .fetch_add(new_bytes, Ordering::Relaxed);
            return Err(L2InsertRefusal::OverBudget {
                payload_bytes: new_bytes,
            });
        }
        // One entry per cache key: a re-admit at the same coordinates is a
        // replacement (fresher state for the same prefix), not a duplicate.
        if let Some(existing) = inner.map.remove(&cache_key) {
            self.release_entry_segments(&mut inner, &existing);
        }
        // Evict to make room BEFORE the new segments land: the projected
        // footprint is the live pool plus this admission's distinct bytes.
        let headroom = self.budget_bytes.saturating_sub(new_bytes);
        let evictions = self.evict_to_limit(&mut inner, headroom, &cache_key);
        inner.clock = inner.clock.wrapping_add(1);
        let last_used = inner.clock;
        for (digest, handle) in new_segments {
            inner.bytes = inner.bytes.saturating_add(handle.bytes.len() as u64);
            inner.segments.insert(digest, handle);
        }
        self.stats
            .shared_bytes_admitted
            .fetch_add(shared_bytes, Ordering::Relaxed);
        inner.map.insert(
            cache_key.clone(),
            L2Entry {
                payload: mirror,
                token_count,
                payload_digest,
                origin,
                last_used,
                charge_bytes: new_bytes,
                payload_bytes,
            },
        );
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        Ok(evictions)
    }

    /// A verified hit records recency and returns the entry's layout with
    /// `Arc` clones of its segment handles (no byte copies). Digests are
    /// not re-hashed: admission verified the wire, and segments are
    /// immutable afterward.
    pub fn get(&self, cache_key: &str) -> Option<L2Hit> {
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        let now = {
            inner.clock = inner.clock.wrapping_add(1);
            inner.clock
        };
        let entry = inner.map.get_mut(cache_key)?;
        entry.last_used = now;
        let payload = entry.payload.clone();
        let token_count = entry.token_count;
        let payload_digest = entry.payload_digest.clone();
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        let mut segments = HashMap::with_capacity(payload.segment_digests().len());
        for digest in payload.segment_digests() {
            if let Some(handle) = inner.segments.get(digest) {
                segments.insert(digest.to_string(), handle.clone());
            }
        }
        Some(L2Hit {
            payload,
            token_count,
            payload_digest,
            segments,
        })
    }

    /// Presence probe: side-effect free. It does not touch LRU recency —
    /// prefix probing must not make entries hot — and returns only
    /// metadata. Recency is updated by `get` after a successful verified
    /// hit.
    pub fn peek(&self, cache_key: &str) -> Option<L2Peek> {
        let inner = self.inner.lock().expect("L2 map lock poisoned");
        let entry = inner.map.get(cache_key)?;
        Some(L2Peek {
            token_count: entry.token_count,
            payload_digest: entry.payload_digest.clone(),
            payload_bytes: entry.payload_bytes,
            distinct_bytes: entry.charge_bytes,
            origin: entry.origin,
        })
    }

    pub fn remove(&self, cache_key: &str) -> Option<L2Eviction> {
        let mut inner = self.inner.lock().expect("L2 map poisoned");
        let removed = inner.map.remove(cache_key)?;
        let before = inner.bytes;
        self.release_entry_segments(&mut inner, &removed);
        let freed = before.saturating_sub(inner.bytes);
        Some(L2Eviction {
            cache_key: cache_key.to_string(),
            freed_bytes: freed,
            retained_bytes: removed
                .payload
                .byte_len()
                .saturating_sub(freed)
                .min(removed.payload_bytes),
        })
    }

    /// Drop everything; returns the distinct bytes released.
    pub fn clear(&self) -> u64 {
        let mut inner = self.inner.lock().expect("L2 map lock poisoned");
        let bytes = inner.bytes;
        inner.map.clear();
        inner.segments.clear();
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
        let logical_bytes = inner
            .map
            .values()
            .map(|entry| entry.payload_bytes)
            .sum::<u64>();
        L2Stats {
            entries: inner.map.len() as u64,
            bytes: inner.bytes,
            logical_bytes,
            segments: inner.segments.len() as u64,
            segment_bytes: inner.bytes,
            shared_bytes_admitted: self.stats.shared_bytes_admitted.load(Ordering::Relaxed),
            budget_bytes: self.budget_bytes,
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            admission_rejects: self.stats.admission_rejects.load(Ordering::Relaxed),
            refused_bytes: self.stats.refused_bytes.load(Ordering::Relaxed),
        }
    }

    /// Drop an entry's exclusive segments from the pool, decrementing the
    /// pool byte total. Shared segments stay: another entry still
    /// references them. Zero-byte segments are dropped without accounting
    /// (a pool without payload bytes must never charge the budget).
    fn release_entry_segments(&self, inner: &mut L2Inner, entry: &L2Entry) {
        for digest in entry.payload.segment_digests() {
            let still_referenced = inner
                .map
                .values()
                .any(|other| other.payload.segment_digests().contains(&digest));
            if still_referenced {
                continue;
            }
            if let Some(handle) = inner.segments.remove(digest) {
                let released = handle.bytes.len() as u64;
                inner.bytes = inner.bytes.saturating_sub(released);
            }
        }
    }

    /// Evict in deterministic LRU order until the pool fits `limit` bytes.
    /// Shared segments are released only with their last referencing
    /// entry; a victim that frees nothing is still counted as an eviction.
    fn evict_to_limit(
        &self,
        inner: &mut L2Inner,
        limit: u64,
        protect_key: &str,
    ) -> Vec<L2Eviction> {
        let mut evictions = Vec::new();
        while inner.bytes > limit {
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
            let Some(removed) = inner.map.remove(&victim) else {
                break;
            };
            let before = inner.bytes;
            self.release_entry_segments(inner, &removed);
            let freed = before.saturating_sub(inner.bytes);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            evictions.push(L2Eviction {
                cache_key: victim,
                freed_bytes: freed,
                retained_bytes: removed.payload_bytes.saturating_sub(freed),
            });
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

    const DIGEST_B: &str = "b616719e0a0d39dc0fe85cd2d0a5e0e2f5e6e10b6b5a0a6f1a1c1d3e5f708a90";

    thread_local! {
        static FULL_WIRE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Build a synthetic wire of `len` bytes and its true BLAKE3 digest.
    fn wire(len: usize, fill: u8) -> (Vec<u8>, String) {
        // Position-dependent bytes so equal-length slices are never equal
        // content: segment digests stay distinct across the wire.
        let bytes: Vec<u8> = (0..len)
            .map(|i| (fill as usize + i) % 251)
            .map(|v| v as u8)
            .collect();
        let digest = segment_digest(&bytes);
        FULL_WIRE.with(|cell| *cell.borrow_mut() = bytes.clone());
        (bytes, digest)
    }

    /// Single-segment full-state mirror over `len` wire bytes. The segment
    /// key is the content digest of the whole wire, as L3 would produce.
    fn single_segment_mirror(w: &[u8]) -> ExactStatePayloadMirror {
        let len = w.len() as u64;
        ExactStatePayloadMirror::FullState {
            layout: L2Layout {
                payload_kind: ExactStatePayloadKind::FullState,
                total_bytes: len,
                kv_bytes: len,
                recurrent_bytes: 0,
                segments: vec![(segment_digest(w), 0..len)],
            },
        }
    }

    /// Manifest-shaped mirror: digest-keyed segments cut at every
    /// `segment_len` boundary, matching how `from_manifest` tiles.
    fn manifest_shaped_mirror(w: &[u8], segment_len: u64) -> ExactStatePayloadMirror {
        // `w` is a suffix of the test's full wire: segment digests are
        // keyed by offset in that full wire so entries sharing a prefix
        // also share segment identity.
        let full = FULL_WIRE.with(|cell| cell.borrow().clone());
        let len = w.len() as u64;
        let mut segments = Vec::new();
        let mut offset = 0u64;
        while offset < len {
            let end = (offset + segment_len).min(len);
            let digest = segment_digest(&full[offset as usize..end as usize]);
            segments.push((digest, offset..end));
            offset = end;
        }
        ExactStatePayloadMirror::FullState {
            layout: L2Layout {
                payload_kind: ExactStatePayloadKind::FullState,
                total_bytes: len,
                kv_bytes: len,
                recurrent_bytes: 0,
                segments,
            },
        }
    }

    fn key(namespace: &str, tokens: &[i32]) -> String {
        l2_cache_key("model-a", "state-a", namespace, tokens)
    }

    #[test]
    fn admit_get_round_trip_serves_wire_bytes() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[1, 2, 3]);
        let (w, digest) = wire(64, 7);
        tier.admit(
            k.clone(),
            3,
            digest.clone(),
            &w,
            single_segment_mirror(&w),
            L2Origin::FromL3,
        )
        .expect("admission must fit");
        let hit = tier.get(&k).expect("admitted key must hit");
        assert_eq!(hit.token_count, 3);
        assert_eq!(hit.payload.byte_len(), 64);
        assert_eq!(hit.payload_digest, digest);
        // Round-trips into a serving payload with the right byte count and
        // exactly the admitted wire bytes.
        let payload = hit.to_payload();
        assert_eq!(payload.byte_len(), 64);
        assert_eq!(
            payload.kind(),
            crate::payload::ExactStatePayloadKind::FullState
        );
        let (bytes, _) = payload.full_state_bytes_timed().expect("full state");
        assert_eq!(bytes.as_ref(), &w[..], "served bytes must equal the wire");
    }

    #[test]
    fn admission_digest_mismatch_refuses_and_stores_nothing() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[1, 2, 3]);
        let (w, _) = wire(64, 7);
        let err = tier
            .admit(
                k.clone(),
                3,
                DIGEST_B.to_string(),
                &w,
                single_segment_mirror(&w),
                L2Origin::Direct,
            )
            .expect_err("a wire that does not hash to the claimed digest must be refused");
        assert!(matches!(err, L2InsertRefusal::DigestMismatch { .. }));
        assert!(tier.peek(&k).is_none(), "refused bytes must not be stored");
        assert!(tier.get(&k).is_none());
        let stats = tier.stats();
        assert_eq!(stats.admission_rejects, 1);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.bytes, 0);
    }

    #[test]
    fn corrupted_wire_never_reaches_the_tier() {
        // L2Origin::Direct with arbitrary bytes under a valid-looking
        // digest is exactly the hole this closes: the digest check runs on
        // the actual bytes.
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[9]);
        let (mut w, digest) = wire(128, 3);
        w[42] ^= 0xff; // one flipped bit
        let err = tier
            .admit(
                k.clone(),
                1,
                digest,
                &w,
                single_segment_mirror(&w),
                L2Origin::Direct,
            )
            .expect_err("corrupted wire must be refused at admission");
        assert!(matches!(err, L2InsertRefusal::DigestMismatch { .. }));
        assert_eq!(tier.stats().admission_rejects, 1);
        assert!(tier.is_empty());
    }

    #[test]
    fn peek_is_side_effect_free_for_lru() {
        let tier = L2Tier::new(160);
        let k1 = key("ns", &[1]);
        let k2 = key("ns", &[2]);
        let (w1, d1) = wire(64, 1);
        let (w2, d2) = wire(64, 2);
        tier.admit(
            k1.clone(),
            1,
            d1,
            &w1,
            single_segment_mirror(&w1),
            L2Origin::FromL3,
        )
        .expect("k1 fits");
        tier.admit(
            k2.clone(),
            1,
            d2,
            &w2,
            single_segment_mirror(&w2),
            L2Origin::FromL3,
        )
        .expect("k2 fits");
        // Probe k1 many times: recency must not move.
        for _ in 0..10 {
            assert!(tier.peek(&k1).is_some());
        }
        // Insert a third entry: k1 (never truly used) must still be the
        // LRU victim, not k2.
        let k3 = key("ns", &[3]);
        let (w3, d3) = wire(64, 3);
        let evictions = tier
            .admit(
                k3.clone(),
                1,
                d3,
                &w3,
                single_segment_mirror(&w3),
                L2Origin::FromL3,
            )
            .expect("k3 fits after eviction");
        assert_eq!(evictions.len(), 1);
        assert_eq!(evictions[0].cache_key, k1, "probed-but-unused k1 is LRU");
        assert!(tier.peek(&k2).is_some(), "untouched k2 survives");
    }

    #[test]
    fn recency_moves_only_on_verified_hit() {
        let tier = L2Tier::new(160);
        let k1 = key("ns", &[1]);
        let k2 = key("ns", &[2]);
        let (w1, d1) = wire(64, 1);
        let (w2, d2) = wire(64, 2);
        tier.admit(
            k1.clone(),
            1,
            d1,
            &w1,
            single_segment_mirror(&w1),
            L2Origin::FromL3,
        )
        .expect("k1 fits");
        tier.admit(
            k2.clone(),
            1,
            d2,
            &w2,
            single_segment_mirror(&w2),
            L2Origin::FromL3,
        )
        .expect("k2 fits");
        // A real hit on k1 makes k2 the victim of the next insertion.
        assert!(tier.get(&k1).is_some());
        let k3 = key("ns", &[3]);
        let (w3, d3) = wire(64, 3);
        let evictions = tier
            .admit(k3, 1, d3, &w3, single_segment_mirror(&w3), L2Origin::FromL3)
            .expect("k3 fits after eviction");
        assert_eq!(evictions.len(), 1);
        assert_eq!(evictions[0].cache_key, k2, "k2 is now LRU");
        assert!(tier.peek(&k1).is_some(), "recently hit k1 survives");
    }

    #[test]
    fn prefix_growth_shares_segment_bytes_instead_of_duplicating() {
        // 16 KiB of four 4 KiB segments; the shorter prefix shares the
        // first three segments with the longer one.
        let segment_len = 4096u64;
        let total = segment_len * 4;
        let tier = L2Tier::new(total * 2);
        let short = key("ns", &[1, 2, 3]);
        let long = key("ns", &[1, 2, 3, 4, 5]);
        let (w, digest) = wire(total as usize, 5);
        let short_len = segment_len * 3;
        tier.admit(
            short.clone(),
            3,
            segment_digest(&w[..short_len as usize]),
            &w[..short_len as usize],
            manifest_shaped_mirror(&w[..short_len as usize], segment_len),
            L2Origin::FromL3,
        )
        .expect("short prefix admitted");

        tier.admit(
            long.clone(),
            5,
            digest,
            &w,
            manifest_shaped_mirror(&w, segment_len),
            L2Origin::FromL3,
        )
        .expect("long prefix admitted");

        let stats = tier.stats();
        // The long entry pays only for its one new (4th) segment.
        assert_eq!(
            stats.bytes, total,
            "pool must hold distinct segment bytes once: got {}",
            stats.bytes
        );
        assert_eq!(
            stats.logical_bytes,
            total + short_len,
            "logical bytes count both entries' full wires"
        );
        assert_eq!(stats.segments, 4, "four distinct segments, not seven");
        assert_eq!(stats.shared_bytes_admitted, short_len);
        // Both entries serve their own wire slices.
        let hit = tier.get(&long).expect("long hit");
        let payload = hit.to_payload();
        let (bytes, _) = payload.full_state_bytes_timed().expect("bytes");
        assert_eq!(bytes.as_ref(), &w[..]);
        let hit_short = tier.get(&short).expect("short hit");
        let payload_short = hit_short.to_payload();
        let (bytes_short, _) = payload_short.full_state_bytes_timed().expect("bytes");
        assert_eq!(bytes_short.as_ref(), &w[..short_len as usize]);
    }

    #[test]
    fn evicting_one_entry_keeps_shared_prefix_segments() {
        let segment_len = 4096u64;
        let total = segment_len * 4;
        let tier = L2Tier::new(total * 2);
        let short = key("ns", &[1]);
        let long = key("ns", &[2]);
        let (w, digest) = wire(total as usize, 6);
        let short_len = segment_len * 3;
        tier.admit(
            short,
            3,
            segment_digest(&w[..short_len as usize]),
            &w[..short_len as usize],
            manifest_shaped_mirror(&w[..short_len as usize], segment_len),
            L2Origin::FromL3,
        )
        .expect("short admitted");
        tier.admit(
            long.clone(),
            5,
            digest,
            &w,
            manifest_shaped_mirror(&w, segment_len),
            L2Origin::FromL3,
        )
        .expect("long admitted");
        // Removing the long entry frees only its exclusive tail segment.
        let removed = tier.remove(&long).expect("long entry present");
        assert_eq!(removed.freed_bytes, segment_len);
        assert_eq!(
            removed.retained_bytes, short_len,
            "shared prefix bytes are retained by the shorter entry"
        );
        let stats = tier.stats();
        assert_eq!(stats.bytes, short_len);
        assert_eq!(stats.segments, 3);
        assert!(tier.get(&key("ns", &[1])).is_some(), "short entry intact");
    }

    #[test]
    fn budget_evicts_lru_first_and_never_the_protected_entry() {
        let tier = L2Tier::new(256);
        let k1 = key("ns", &[1]);
        let k2 = key("ns", &[2]);
        let k3 = key("ns", &[3]);
        let (w1, d1) = wire(100, 1);
        let (w2, d2) = wire(100, 2);
        let (w3, d3) = wire(100, 3);
        tier.admit(
            k1.clone(),
            1,
            d1,
            &w1,
            single_segment_mirror(&w1),
            L2Origin::FromL3,
        )
        .expect("k1 fits");
        tier.admit(
            k2.clone(),
            1,
            d2,
            &w2,
            single_segment_mirror(&w2),
            L2Origin::FromL3,
        )
        .expect("k2 fits");
        // Touch k1 so k2 becomes the LRU victim.
        assert!(tier.get(&k1).is_some());
        let evictions = tier
            .admit(
                k3.clone(),
                1,
                d3,
                &w3,
                single_segment_mirror(&w3),
                L2Origin::FromL3,
            )
            .expect("k3 fits after eviction");
        assert_eq!(
            evictions.len(),
            1,
            "one entry must be evicted: {evictions:?}"
        );
        assert_eq!(evictions[0].cache_key, k2, "LRU victim is k2");
        assert_eq!(evictions[0].freed_bytes, 100);
        assert_eq!(evictions[0].retained_bytes, 0);
        assert!(tier.peek(&k1).is_some(), "recently used k1 survives");
        assert!(tier.peek(&k3).is_some(), "just-admitted k3 survives");
        assert!(tier.peek(&k2).is_none(), "k2 was evicted");
        let stats = tier.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.bytes, 200, "pool bytes must track survivors exactly");
    }

    #[test]
    fn oversized_distinct_bytes_are_refused_without_evicting() {
        let tier = L2Tier::new(128);
        let k1 = key("ns", &[1]);
        let (w1, d1) = wire(64, 1);
        tier.admit(
            k1.clone(),
            1,
            d1,
            &w1,
            single_segment_mirror(&w1),
            L2Origin::FromL3,
        )
        .expect("fits");
        let (w2, d2) = wire(129, 2);
        let err = tier
            .admit(
                key("ns", &[2]),
                1,
                d2,
                &w2,
                single_segment_mirror(&w2),
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
        let empty_digest = segment_digest(&[]);
        let err = tier
            .admit(
                key("ns", &[1]),
                1,
                empty_digest,
                &[],
                single_segment_mirror(&[]),
                L2Origin::FromL3,
            )
            .expect_err("empty payloads must be refused");
        assert_eq!(err, L2InsertRefusal::EmptyPayload);
        assert!(tier.is_empty());
    }

    #[test]
    fn malformed_digest_is_refused_before_any_hashing() {
        let tier = L2Tier::new(1 << 20);
        let (w, _) = wire(16, 4);
        let err = tier
            .admit(
                key("ns", &[1]),
                1,
                "not-a-digest".to_string(),
                &w,
                single_segment_mirror(&w),
                L2Origin::FromL3,
            )
            .expect_err("malformed digest must be refused");
        assert_eq!(err, L2InsertRefusal::MalformedDigest);
    }

    #[test]
    fn readmit_replaces_and_keeps_accounting_exact() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[9]);
        let (w1, d1) = wire(100, 1);
        let (w2, d2) = wire(40, 2);
        tier.admit(
            k.clone(),
            3,
            d1,
            &w1,
            single_segment_mirror(&w1),
            L2Origin::FromL3,
        )
        .expect("first admission");
        let evictions = tier
            .admit(
                k.clone(),
                3,
                d2.clone(),
                &w2,
                single_segment_mirror(&w2),
                L2Origin::FromL3,
            )
            .expect("replacement");
        assert!(evictions.is_empty());
        assert_eq!(tier.len(), 1);
        assert_eq!(
            tier.stats().bytes,
            40,
            "replacement must release the old bytes"
        );
        // New digest is the one served now.
        let hit = tier.get(&k).expect("replacement hit");
        assert_eq!(hit.payload_digest, d2);
        let payload = hit.to_payload();
        let (bytes, _) = payload.full_state_bytes_timed().expect("bytes");
        assert_eq!(bytes.as_ref(), &w2[..]);
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
    fn mirror_round_trips_every_payload_kind_from_manifest() {
        // kv-recurrent: kv 24 bytes then recurrent 8, cut into two segments.
        let (kv_wire, _) = wire(24, 3);
        let (rec_wire, _) = wire(8, 4);
        let wire_bytes: Vec<u8> = [kv_wire, rec_wire].concat();
        let digest = segment_digest(&wire_bytes);
        let manifest = HandoffManifest {
            version: MANIFEST_VERSION,
            model_identity: "blake3:model".to_string(),
            state_identity: "blake3:state".to_string(),
            payload_kind: "kv-recurrent".to_string(),
            total_bytes: wire_bytes.len() as u64,
            payload_digest: digest.clone(),
            segments: vec![
                HandoffSegmentRef {
                    index: 0,
                    offset: 0,
                    bytes: 16,
                    digest: "blake3:seg-a".to_string(),
                    meta_json: None,
                },
                HandoffSegmentRef {
                    index: 1,
                    offset: 16,
                    bytes: 16,
                    digest: "blake3:seg-b".to_string(),
                    meta_json: None,
                },
            ],
            kv_bytes: 24,
            recurrent_bytes: 8,
            kv_desc_json: None,
            token_count: 4,
            continuation_token: 0,
            expected_tokens: Vec::new(),
        };
        let mirror = ExactStatePayloadMirror::from_manifest(&manifest).expect("manifest parses");
        assert_eq!(mirror.byte_len(), 32);
        assert_eq!(mirror.kind(), ExactStatePayloadKind::KvRecurrent);

        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[7]);
        tier.admit(k.clone(), 4, digest, &wire_bytes, mirror, L2Origin::FromL3)
            .expect("fits");
        let hit = tier.get(&k).expect("hit");
        let payload = hit.to_payload();
        assert_eq!(payload.byte_len(), 32);
        let served_kv = payload
            .kv_bytes()
            .expect("kv bytes")
            .expect("kv-recurrent has kv")
            .into_owned();
        let served_rec = payload
            .recurrent_state_bytes()
            .expect("recurrent bytes")
            .into_owned();
        assert_eq!(served_kv, wire_bytes[..24], "kv slice must match the wire");
        assert_eq!(served_rec, wire_bytes[24..], "recurrent slice matches");
    }

    #[test]
    fn from_manifest_rejects_unknown_kind_and_bad_tiling() {
        let mut manifest = HandoffManifest {
            version: MANIFEST_VERSION,
            model_identity: "m".to_string(),
            state_identity: "s".to_string(),
            payload_kind: "blob".to_string(),
            total_bytes: 10,
            payload_digest: "blake3:aa".to_string(),
            segments: Vec::new(),
            kv_bytes: 10,
            recurrent_bytes: 0,
            kv_desc_json: None,
            token_count: 1,
            continuation_token: 0,
            expected_tokens: Vec::new(),
        };
        let err = ExactStatePayloadMirror::from_manifest(&manifest)
            .expect_err("unknown kind must be refused");
        assert!(matches!(err, L2InsertRefusal::UnknownPayloadKind(_)));

        manifest.payload_kind = "full-state".to_string();
        manifest.segments = vec![HandoffSegmentRef {
            index: 0,
            offset: 0,
            bytes: 7,
            digest: "blake3:seg".to_string(),
            meta_json: None,
        }];
        let err = ExactStatePayloadMirror::from_manifest(&manifest)
            .expect_err("tiling mismatch must be refused");
        assert!(matches!(err, L2InsertRefusal::MalformedManifest(_)));
    }

    #[test]
    fn multi_segment_reads_reassemble_exact_wire() {
        // 6 KiB in 1 KiB segments: every read path crosses many blocks.
        let segment_len = 1024u64;
        let total = segment_len * 6;
        let (w, digest) = wire(total as usize, 9);
        let tier = L2Tier::new(total * 2);
        let k = key("ns", &[1]);
        tier.admit(
            k.clone(),
            6,
            digest,
            &w,
            manifest_shaped_mirror(&w, segment_len),
            L2Origin::FromL3,
        )
        .expect("fits");
        let hit = tier.get(&k).expect("hit");
        let payload = hit.to_payload();
        let (bytes, reconstruct) = payload.full_state_bytes_timed().expect("bytes");
        assert_eq!(bytes.as_ref(), &w[..]);
        // Multiple distinct segment storages materialize on read; the
        // reconstruction length must equal the payload either way.
        assert_eq!(
            reconstruct.reconstruct_bytes, total,
            "multi-segment reads materialize the wire exactly once"
        );
    }

    #[test]
    fn clear_releases_everything_and_reports_bytes() {
        let tier = L2Tier::new(1 << 20);
        for i in 0..5i32 {
            let (w, d) = wire(64, i as u8 + 10);
            tier.admit(
                key("ns", &[i]),
                1,
                d,
                &w,
                single_segment_mirror(&w),
                L2Origin::FromL3,
            )
            .expect("fits");
        }
        let released = tier.clear();
        assert_eq!(released, 320);
        assert!(tier.is_empty());
        assert_eq!(tier.stats().bytes, 0);
        assert_eq!(tier.stats().segments, 0);
    }

    #[test]
    fn remove_is_exact() {
        let tier = L2Tier::new(1 << 20);
        let k = key("ns", &[4]);
        let (w, d) = wire(64, 8);
        tier.admit(
            k.clone(),
            1,
            d,
            &w,
            single_segment_mirror(&w),
            L2Origin::FromL3,
        )
        .expect("fits");
        let removed = tier.remove(&k).expect("present entry removes");
        assert_eq!(removed.freed_bytes, 64);
        assert_eq!(removed.retained_bytes, 0);
        assert!(tier.remove(&k).is_none(), "second remove is None");
        assert_eq!(tier.stats().bytes, 0);
        assert_eq!(tier.stats().segments, 0);
    }
}
