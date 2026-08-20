//! A memory-mapped disk tier for evicted KV prefixes.
//!
//! # Why a slow tier is the right shape here
//!
//! The workload is agentic traffic on a mesh: a stable system prompt plus tool
//! schemas, divergent tails, and the same logical conversation returning after
//! a gap. Requests are round-robined across peers, so a given peer sees a
//! familiar prefix again *infrequently* but the value of each hit is large.
//! That is precisely the shape a large, slow, cheap tier serves well and a
//! small fast tier does not.
//!
//! Restore cost is bounded by the page's bytes, and mmap means the kernel
//! faults in only the pages actually touched. A page still warm in the page
//! cache is nearly free; a cold one is a single sequential read. Prefill, by
//! contrast, is quadratic in prefix length. **The larger the reusable bulk,
//! the better the ratio** — the opposite of most caches, and the reason this
//! is worth building for big agentic prompts specifically.
//!
//! # What is deliberately not done
//!
//! Blocks are not written to disk. Reconstructing a block-based payload
//! allocates and concatenates the entire payload before the runtime copies it
//! again into device memory. For a multi-GB page that is gigabytes of pointless
//! traffic on the restore path. Each entry is one contiguous file, mapped and
//! borrowed.
//!
//! # Safety of reuse across restarts
//!
//! Page identity covers the model, topology, stage, layer range, KV dtypes,
//! backend, GPU split, and context size, so a page written under one
//! configuration cannot be read back under an incompatible one. On top of that
//! every entry carries its length and a content checksum, and the tier refuses
//! to serve an entry whose bytes do not match. A silent miss is acceptable; a
//! silently *wrong* restore is numerical corruption, so mismatches are hard
//! errors that quarantine the entry.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::CacheBytes;

/// Bumped whenever the on-disk layout or the identity contract changes, so a
/// stale directory from an older build is discarded rather than misread.
const DISK_TIER_FORMAT_VERSION: u32 = 1;

/// Checksum the caller metadata that describes how to interpret a page.
///
/// Uses `serde_json::to_vec` on the value; `serde_json::Value` maps are
/// `BTreeMap`-backed so key order is deterministic and the digest is stable
/// across processes.
fn extra_digest(extra: Option<&serde_json::Value>) -> Option<String> {
    let value = extra?;
    let encoded = serde_json::to_vec(value).ok()?;
    Some(blake3::hash(&encoded).to_hex().to_string())
}
const INDEX_FILE_NAME: &str = "index.json";
const ENTRY_DIR_NAME: &str = "pages";

/// One payload component stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiskComponent {
    /// Byte offset of this component inside the entry file.
    pub offset: u64,
    pub len: u64,
    /// BLAKE3 of the component bytes, verified before the entry is served.
    pub checksum: String,
}

/// Metadata for a single demoted prefix.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiskEntry {
    pub page_id: String,
    pub token_count: u64,
    /// File name inside the pages directory. Never an absolute path, so the
    /// cache directory stays relocatable and no host paths are persisted.
    pub file_name: String,
    pub total_bytes: u64,
    /// Payload components in order. KV-recurrent payloads have two.
    pub components: Vec<DiskComponent>,
    /// Payload discriminant, so a restore cannot reinterpret a full-state page
    /// as a KV page.
    pub payload_kind: String,
    /// Opaque caller metadata persisted alongside the bytes.
    ///
    /// KV pages cannot be imported without their `KvPageDesc` (row strides,
    /// element types, layer range). Keeping it only in RAM would make the
    /// disk tier silently useless across a restart: the bytes would survive
    /// and be unusable. The cache layer treats this as an opaque value so it
    /// does not need to know the runtime's descriptor type.
    #[serde(default)]
    pub extra: Option<serde_json::Value>,
    /// Checksum over the canonical encoding of `extra`.
    ///
    /// The payload checksum covers the bytes but not the metadata that says
    /// how to interpret them. For a KV page `extra` carries the descriptor --
    /// layer range, K/V ggml types, row strides -- so an index corrupted into
    /// still-valid JSON could hand correctly-checksummed bytes to the runtime
    /// under the wrong layout. That is silent numerical corruption on a path
    /// that looks verified, so the metadata is checksummed too.
    ///
    /// Optional for forward/backward tolerance: an entry written before this
    /// field existed has `None` and is rejected on load rather than trusted,
    /// which is why the format version is also bumped.
    #[serde(default)]
    pub extra_checksum: Option<String>,
    pub written_at_secs: u64,
    pub last_used_secs: u64,
    /// Monotonic use counter, used to break ties when several entries share a
    /// `last_used_secs`. Wall-clock seconds are far too coarse: a burst of
    /// stores within one second would otherwise make eviction order arbitrary,
    /// and could pick the entry that was just written.
    #[serde(default)]
    pub use_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskIndex {
    format_version: u32,
    entries: Vec<DiskEntry>,
}

/// Counters describing disk-tier behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiskTierStats {
    pub entries: usize,
    pub bytes: u64,
    pub max_bytes: u64,
    /// Prefixes written to disk on eviction.
    pub demotions: u64,
    /// Prefixes served back from disk.
    pub promotions: u64,
    /// Entries dropped to stay inside the size budget.
    pub evictions: u64,
    /// Pages rejected because one page exceeded the whole tier budget.
    pub pages_rejected_too_large: u64,
    /// Byte size of the most recently rejected oversized page.
    pub last_rejected_page_bytes: u64,
    /// Entries rejected because their bytes failed verification.
    pub corrupt_entries: u64,
    /// Loads that hashed the payload against its stored checksums.
    pub verifications: u64,
    /// Loads served from an already-verified live mapping.
    pub verifications_skipped: u64,
}

/// An exclusive advisory lock over a cache directory, held for the lifetime of
/// the tier and released when the file descriptor closes (including on crash).
#[derive(Debug)]
struct DirectoryLock {
    _file: File,
}

impl DirectoryLock {
    fn acquire(root: &Path) -> Result<Self> {
        let path = root.join("owner.lock");
        let file = File::create(&path)
            .with_context(|| format!("create KV disk tier lock {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // Non-blocking: a busy directory means another live instance owns
            // it, and the correct response is to decline, not to wait.
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc != 0 {
                return Err(anyhow!(
                    "KV disk tier at {} is owned by another instance",
                    root.display()
                ));
            }
        }
        #[cfg(not(unix))]
        {
            // No advisory-lock equivalent is wired up here yet. Failing open
            // would be the dangerous choice: two instances sharing a cache
            // directory delete each other's mapped page files during LRU
            // eviction and orphan reclaim, and on Windows unlinking a mapped
            // file can fault the reader. Refuse to enable the tier instead of
            // silently running unprotected.
            let _ = &file;
            return Err(anyhow!(
                "KV disk tier requires advisory directory locking, which is not \
                 implemented on this platform; refusing to open {}",
                root.display()
            ));
        }
        #[cfg(unix)]
        Ok(Self { _file: file })
    }
}

/// A size-bounded, mmap-backed store of evicted prefixes.
#[derive(Debug)]
pub struct PrefixDiskTier {
    _lock: DirectoryLock,
    root: PathBuf,
    max_bytes: u64,
    bytes: u64,
    entries: HashMap<String, DiskEntry>,
    /// Live mappings, kept so repeated restores of a hot page reuse one map.
    mappings: HashMap<String, Arc<Mmap>>,
    /// Pages whose bytes this process has already hashed against their stored
    /// checksums, and whose mapping has been live continuously since.
    ///
    /// Verification dominates restore cost -- a 2 GB page is ~0.9s of blake3
    /// against ~40ms of native import -- and re-hashing bytes this process
    /// already verified, while holding an exclusive lock on the directory and
    /// a live mapping of the file, adds no protection. Membership is dropped
    /// wherever the mapping is (quarantine, budget eviction, reset), so an
    /// entry can never be trusted across a remap.
    verified: HashSet<String>,
    use_clock: u64,
    temp_counter: u64,
    stats: DiskTierStats,
}

impl PrefixDiskTier {
    /// Open or create a disk tier rooted at `root`.
    ///
    /// A directory written by an incompatible build is discarded rather than
    /// migrated: the contents are a regenerable cache, so correctness is worth
    /// far more than preserving them.
    pub fn open(root: impl AsRef<Path>, max_bytes: u64) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join(ENTRY_DIR_NAME))
            .with_context(|| format!("create KV disk tier at {}", root.display()))?;

        // Take exclusive ownership of the directory before touching anything
        // in it. Two instances serving the same model on one machine hash to
        // the same root, and this tier is not concurrency-safe: `commit_index`
        // is last-writer-wins and `remove_orphan_files` deletes page files
        // that are absent from *this* process's index — including files
        // another instance just wrote and still has mapped. Sharing a
        // directory is therefore not a degraded mode, it is mutual
        // destruction, so a second instance declines the tier instead.
        let lock = DirectoryLock::acquire(&root)?;

        let mut tier = Self {
            _lock: lock,
            root,
            max_bytes,
            bytes: 0,
            entries: HashMap::new(),
            mappings: HashMap::new(),
            verified: HashSet::new(),
            use_clock: 0,
            temp_counter: 0,
            stats: DiskTierStats {
                max_bytes,
                ..DiskTierStats::default()
            },
        };
        tier.load_index()?;
        tier.enforce_budget()?;
        Ok(tier)
    }

    fn index_path(&self) -> PathBuf {
        self.root.join(INDEX_FILE_NAME)
    }

    fn entry_path(&self, file_name: &str) -> PathBuf {
        self.root.join(ENTRY_DIR_NAME).join(file_name)
    }

    /// Load a previously written index, dropping anything unusable.
    ///
    /// Every failure mode here is recoverable by discarding: a cache that
    /// starts empty is merely slow, whereas a cache that serves unverified
    /// bytes is wrong.
    fn load_index(&mut self) -> Result<()> {
        let path = self.index_path();
        let Ok(raw) = fs::read(&path) else {
            // No index: any page files present are orphans from a crash
            // between writing a file and committing the index.
            self.remove_orphan_files();
            return Ok(());
        };
        let Ok(index) = serde_json::from_slice::<DiskIndex>(&raw) else {
            return self.reset("unreadable index");
        };
        if index.format_version != DISK_TIER_FORMAT_VERSION {
            return self.reset("format version changed");
        }

        for entry in index.entries {
            // The index is plain JSON on disk. Never join an unvalidated name
            // onto the cache root: a garbled or crafted entry could otherwise
            // steer `remove_file` outside the cache directory.
            if !is_safe_file_name(&entry.file_name) {
                continue;
            }
            let path = self.entry_path(&entry.file_name);
            // Size is checked here; content is verified lazily on first use so
            // startup does not read the entire cache.
            let Ok(metadata) = fs::metadata(&path) else {
                continue;
            };
            if metadata.len() != entry.total_bytes {
                let _ = fs::remove_file(&path);
                continue;
            }
            self.bytes = self.bytes.saturating_add(entry.total_bytes);
            self.use_clock = self.use_clock.max(entry.use_sequence);
            self.entries.insert(entry.page_id.clone(), entry);
        }
        self.stats.entries = self.entries.len();
        self.stats.bytes = self.bytes;
        self.remove_orphan_files();
        Ok(())
    }

    /// Discard the whole tier and start clean.
    fn reset(&mut self, _reason: &str) -> Result<()> {
        let pages = self.root.join(ENTRY_DIR_NAME);
        let _ = fs::remove_dir_all(&pages);
        let _ = fs::remove_file(self.index_path());
        fs::create_dir_all(&pages).with_context(|| format!("reset {}", pages.display()))?;
        self.entries.clear();
        self.mappings.clear();
        self.verified.clear();
        self.bytes = 0;
        self.stats.entries = 0;
        self.stats.bytes = 0;
        Ok(())
    }

    /// Delete page files with no index entry, which is how a crash between
    /// writing a file and committing the index shows up.
    fn remove_orphan_files(&self) {
        let known: std::collections::HashSet<&str> = self
            .entries
            .values()
            .map(|entry| entry.file_name.as_str())
            .collect();
        let Ok(dir) = fs::read_dir(self.root.join(ENTRY_DIR_NAME)) else {
            return;
        };
        for file in dir.flatten() {
            let name = file.file_name();
            let Some(name) = name.to_str() else { continue };
            if !known.contains(name) {
                if name == "owner.lock" {
                    continue;
                }
                // Temp files are reclaimed too. This only runs at open, and
                // opening means we hold the exclusive directory lock, so no
                // other writer exists and every `.tmp` here is debris from a
                // crash between writing a page and renaming it into place.
                // Skipping them (the previous behaviour) leaked a full page's
                // bytes per crash, unbounded, with nothing to ever clean up.
                let _ = fs::remove_file(file.path());
            }
        }
    }

    /// Write a payload's components to disk as one contiguous file.
    ///
    /// The file is written to a temporary name and renamed into place, so a
    /// crash mid-write cannot leave a partial file that the index believes is
    /// complete.
    pub fn store(
        &mut self,
        page_id: &str,
        token_count: u64,
        payload_kind: &str,
        components: &[&[u8]],
        extra: Option<serde_json::Value>,
    ) -> Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        let total_bytes: u64 = components.iter().map(|bytes| bytes.len() as u64).sum();
        if total_bytes == 0 {
            return Ok(());
        }
        if total_bytes > self.max_bytes {
            // A single page larger than the whole budget would evict
            // everything else and then itself; make the rejection observable.
            self.stats.pages_rejected_too_large =
                self.stats.pages_rejected_too_large.saturating_add(1);
            self.stats.last_rejected_page_bytes = total_bytes;
            return Ok(());
        }
        if self.entries.contains_key(page_id) {
            return Ok(());
        }

        let file_name = format!("{}.kvp", blake3::hash(page_id.as_bytes()).to_hex());
        let final_path = self.entry_path(&file_name);
        // Unique per attempt: a deterministic temp name lets two writers
        // interleave into one file and rename a torn result into place.
        self.temp_counter = self.temp_counter.saturating_add(1);
        let temp_path = final_path.with_extension(format!(
            "kvp.{}.{}.tmp",
            std::process::id(),
            self.temp_counter
        ));

        let mut descriptors = Vec::with_capacity(components.len());
        {
            let mut file = File::create(&temp_path)
                .with_context(|| format!("create KV page file {}", temp_path.display()))?;
            let mut offset = 0u64;
            for bytes in components {
                file.write_all(bytes)
                    .with_context(|| format!("write KV page file {}", temp_path.display()))?;
                descriptors.push(DiskComponent {
                    offset,
                    len: bytes.len() as u64,
                    checksum: blake3::hash(bytes).to_hex().to_string(),
                });
                offset = offset.saturating_add(bytes.len() as u64);
            }
            file.sync_all()
                .with_context(|| format!("flush KV page file {}", temp_path.display()))?;
        }
        fs::rename(&temp_path, &final_path)
            .with_context(|| format!("publish KV page file {}", final_path.display()))?;

        let now = now_secs();
        self.use_clock = self.use_clock.saturating_add(1);
        self.entries.insert(
            page_id.to_string(),
            DiskEntry {
                page_id: page_id.to_string(),
                token_count,
                file_name,
                total_bytes,
                components: descriptors,
                payload_kind: payload_kind.to_string(),
                extra_checksum: extra_digest(extra.as_ref()),
                extra,
                written_at_secs: now,
                last_used_secs: now,
                use_sequence: self.use_clock,
            },
        );
        self.bytes = self.bytes.saturating_add(total_bytes);
        self.stats.demotions = self.stats.demotions.saturating_add(1);
        self.enforce_budget()?;
        self.commit_index()?;
        Ok(())
    }

    /// Map an entry and return borrowing [`CacheBytes`] for each component.
    ///
    /// Returns `Ok(None)` for a plain miss. Returns `Err` when an entry exists
    /// but cannot be trusted — that is a corruption signal, not a miss, and
    /// callers must not silently fall through to treating it as absent without
    /// recording it.
    pub fn load(&mut self, page_id: &str, expected_kind: &str) -> Result<Option<DiskLoad>> {
        let Some(entry) = self.entries.get(page_id).cloned() else {
            return Ok(None);
        };
        if entry.payload_kind != expected_kind {
            // Identity should make this impossible; if it happens, the entry
            // is not what we think it is and must not be imported.
            self.quarantine(page_id);
            return Err(anyhow!(
                "KV disk entry payload kind mismatch: stored {} expected {expected_kind}",
                entry.payload_kind
            ));
        }

        let mut freshly_mapped = false;
        let mmap = match self.mappings.get(page_id) {
            Some(mmap) => mmap.clone(),
            None => {
                freshly_mapped = true;
                let path = self.entry_path(&entry.file_name);
                let file = File::open(&path)
                    .with_context(|| format!("open KV page file {}", path.display()))?;
                // SAFETY: this process holds an exclusive lock on the cache
                // directory (see `DirectoryLock`), entries are published by
                // atomic rename and never modified in place, and the size is
                // checked below before any range is read. A concurrent
                // external truncation of the backing file would still be
                // undefined behaviour; the directory lock is what rules that
                // out for other mesh-llm instances.
                let mmap = unsafe { Mmap::map(&file) }
                    .with_context(|| format!("map KV page file {}", path.display()))?;
                if mmap.len() as u64 != entry.total_bytes {
                    self.quarantine(page_id);
                    return Err(anyhow!(
                        "KV disk entry size changed on disk: expected {} got {}",
                        entry.total_bytes,
                        mmap.len()
                    ));
                }
                let mmap = Arc::new(mmap);
                self.mappings.insert(page_id.to_string(), mmap.clone());
                mmap
            }
        };

        // Verify the metadata before the payload. `extra` tells the caller
        // how to interpret these bytes (for a KV page: layer range, ggml
        // types, row strides), so trusting it while only checksumming the
        // payload would let a corrupted index apply a valid page under the
        // wrong layout. An entry written by an older format has no digest;
        // the format version bump means we never see one, but reject rather
        // than trust if we somehow do.
        if extra_digest(entry.extra.as_ref()) != entry.extra_checksum {
            self.quarantine(page_id);
            self.stats.corrupt_entries = self.stats.corrupt_entries.saturating_add(1);
            return Err(anyhow!(
                "KV disk entry failed metadata checksum verification for page {page_id}"
            ));
        }

        // Hash the payload on the first load of a mapping, and not again
        // while that mapping stays live. The bytes cannot change underneath a
        // held mapping: entries are published by atomic rename and never
        // modified in place, and this process holds an exclusive lock on the
        // directory. Re-hashing them on every restore would be ~95% of the
        // cost of a restore and would detect nothing a first verification did
        // not already rule out. A remap re-verifies, because that is where a
        // different file could appear.
        let must_verify = freshly_mapped || !self.verified.contains(page_id);
        let mut parts = Vec::with_capacity(entry.components.len());
        for component in &entry.components {
            let bytes = CacheBytes::mapped(mmap.clone(), component.offset, component.len)?;
            if must_verify {
                // Verify before handing bytes to the runtime. This faults the
                // pages in, which a restore is about to do anyway.
                let actual = blake3::hash(bytes.as_cow()?.as_ref()).to_hex().to_string();
                if actual != component.checksum {
                    self.quarantine(page_id);
                    self.stats.corrupt_entries = self.stats.corrupt_entries.saturating_add(1);
                    return Err(anyhow!(
                        "KV disk entry failed checksum verification for page {page_id}"
                    ));
                }
            }
            parts.push(bytes);
        }
        if must_verify {
            self.verified.insert(page_id.to_string());
            self.stats.verifications = self.stats.verifications.saturating_add(1);
        } else {
            self.stats.verifications_skipped = self.stats.verifications_skipped.saturating_add(1);
        }

        self.use_clock = self.use_clock.saturating_add(1);
        let use_sequence = self.use_clock;
        if let Some(entry) = self.entries.get_mut(page_id) {
            entry.last_used_secs = now_secs();
            entry.use_sequence = use_sequence;
        }
        self.stats.promotions = self.stats.promotions.saturating_add(1);
        Ok(Some(DiskLoad {
            token_count: entry.token_count,
            components: parts,
            extra: entry.extra.clone(),
        }))
    }

    /// Mark an entry as recently used without mapping or verifying its payload.
    pub fn touch(&mut self, page_id: &str) -> bool {
        let Some(entry) = self.entries.get_mut(page_id) else {
            return false;
        };
        self.use_clock = self.use_clock.saturating_add(1);
        entry.last_used_secs = now_secs();
        entry.use_sequence = self.use_clock;
        true
    }

    pub fn contains(&self, page_id: &str) -> bool {
        self.entries.contains_key(page_id)
    }

    /// How many pages the tier currently holds.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Longest stored prefix lengths, for extending lookup probing to lengths
    /// that only exist on disk.
    pub fn token_counts_at_most(&self, max_token_count: u64) -> Vec<u64> {
        let mut counts: Vec<u64> = self
            .entries
            .values()
            .map(|entry| entry.token_count)
            .filter(|count| *count <= max_token_count)
            .collect();
        counts.sort_unstable_by(|left, right| right.cmp(left));
        counts.dedup();
        counts
    }

    /// Drop an entry that cannot be trusted.
    fn quarantine(&mut self, page_id: &str) {
        self.mappings.remove(page_id);
        self.verified.remove(page_id);
        if let Some(entry) = self.entries.remove(page_id) {
            self.bytes = self.bytes.saturating_sub(entry.total_bytes);
            let _ = fs::remove_file(self.entry_path(&entry.file_name));
        }
        self.stats.entries = self.entries.len();
        self.stats.bytes = self.bytes;
        let _ = self.commit_index();
    }

    /// Evict least-recently-used entries until inside the size budget.
    fn enforce_budget(&mut self) -> Result<()> {
        while self.max_bytes > 0 && self.bytes > self.max_bytes {
            let Some(victim) = self
                .entries
                .values()
                .min_by_key(|entry| (entry.last_used_secs, entry.use_sequence))
                .map(|entry| entry.page_id.clone())
            else {
                break;
            };
            self.mappings.remove(&victim);
            self.verified.remove(&victim);
            if let Some(entry) = self.entries.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(entry.total_bytes);
                let _ = fs::remove_file(self.entry_path(&entry.file_name));
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            }
        }
        self.stats.entries = self.entries.len();
        self.stats.bytes = self.bytes;
        Ok(())
    }

    /// Persist the index atomically so a crash cannot leave it torn.
    fn commit_index(&mut self) -> Result<()> {
        self.stats.entries = self.entries.len();
        self.stats.bytes = self.bytes;

        let index = DiskIndex {
            format_version: DISK_TIER_FORMAT_VERSION,
            entries: self.entries.values().cloned().collect(),
        };
        let encoded = serde_json::to_vec(&index).context("encode KV disk tier index")?;
        let temp = self.index_path().with_extension("json.tmp");
        fs::write(&temp, &encoded)
            .with_context(|| format!("write KV disk tier index {}", temp.display()))?;
        fs::rename(&temp, self.index_path()).context("publish KV disk tier index")?;
        Ok(())
    }

    pub fn stats(&self) -> DiskTierStats {
        DiskTierStats {
            entries: self.entries.len(),
            bytes: self.bytes,
            ..self.stats
        }
    }
}

/// Mapped components of a restored prefix.
#[derive(Debug, Clone)]
pub struct DiskLoad {
    pub token_count: u64,
    pub components: Vec<CacheBytes>,
    /// Caller metadata stored with the entry, if any.
    pub extra: Option<serde_json::Value>,
}

/// A cache file name must be a plain component of the pages directory.
fn is_safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(dir: &tempfile::TempDir, max_bytes: u64) -> PrefixDiskTier {
        PrefixDiskTier::open(dir.path(), max_bytes).expect("open tier")
    }

    #[test]
    fn stored_prefix_round_trips_through_the_disk_tier() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        let payload = vec![7u8; 4096];

        tier.store("page-a", 2048, "kv-recurrent", &[&payload], None)
            .unwrap();
        let loaded = tier
            .load("page-a", "kv-recurrent")
            .unwrap()
            .expect("stored page should load");

        assert_eq!(loaded.token_count, 2048);
        assert_eq!(loaded.components.len(), 1);
        assert_eq!(
            loaded.components[0].as_cow().unwrap().as_ref(),
            &payload[..]
        );
        // Restores must borrow the mapping rather than copy it.
        assert!(loaded.components[0].is_mapped());
    }

    #[test]
    fn multi_component_payloads_keep_their_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        let kv = vec![1u8; 1000];
        let recurrent = vec![2u8; 500];

        tier.store("page-a", 64, "kv-recurrent", &[&kv, &recurrent], None)
            .unwrap();
        let loaded = tier.load("page-a", "kv-recurrent").unwrap().unwrap();

        assert_eq!(loaded.components.len(), 2);
        assert_eq!(loaded.components[0].as_cow().unwrap().as_ref(), &kv[..]);
        assert_eq!(
            loaded.components[1].as_cow().unwrap().as_ref(),
            &recurrent[..]
        );
    }

    #[test]
    fn missing_page_is_a_plain_miss() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);

        assert!(tier.load("absent", "kv-recurrent").unwrap().is_none());
    }

    /// The headline capability: a prefix written by one process is reusable by
    /// the next, which is what survives a restart.
    #[test]
    fn entries_survive_reopening_the_tier() {
        let dir = tempfile::tempdir().unwrap();
        let payload = vec![9u8; 2048];
        {
            let mut tier = tier(&dir, 1 << 20);
            tier.store("page-a", 512, "full-state", &[&payload], None)
                .unwrap();
        }

        let mut reopened = tier(&dir, 1 << 20);
        let loaded = reopened
            .load("page-a", "full-state")
            .unwrap()
            .expect("entry should survive restart");

        assert_eq!(
            loaded.components[0].as_cow().unwrap().as_ref(),
            &payload[..]
        );
        assert_eq!(reopened.stats().entries, 1);
    }

    #[test]
    fn above_half_context_pages_survive_reopen_at_split_stage_densities() {
        const HALF_CONTEXT_TOKENS: u64 = 4_096;
        const PAGE_TOKENS: u64 = 4_864;
        // Scaled-down forms of the observed stage densities (76,160 and 2,176
        // bytes/token). Their ratio is preserved without writing 353 MiB.
        const SCALED_STAGE_BYTES_PER_TOKEN: [u64; 2] = [2_380, 68];

        for bytes_per_token in SCALED_STAGE_BYTES_PER_TOKEN {
            let dir = tempfile::tempdir().unwrap();
            let payload_len = PAGE_TOKENS * bytes_per_token;
            let old_resident_derived_budget = HALF_CONTEXT_TOKENS * bytes_per_token;
            let payload = vec![11u8; payload_len as usize];

            let mut old_tier = tier(&dir, old_resident_derived_budget);
            old_tier
                .store("above-half", PAGE_TOKENS, "resident-kv", &[&payload], None)
                .unwrap();
            assert_eq!(old_tier.stats().entries, 0);
            assert_eq!(old_tier.stats().pages_rejected_too_large, 1);
            drop(old_tier);

            let disk_derived_budget = payload_len + bytes_per_token;
            {
                let mut tier = tier(&dir, disk_derived_budget);
                tier.store("above-half", PAGE_TOKENS, "resident-kv", &[&payload], None)
                    .unwrap();
                assert_eq!(tier.stats().entries, 1);
            }

            let mut reopened = tier(&dir, disk_derived_budget);
            let loaded = reopened
                .load("above-half", "resident-kv")
                .unwrap()
                .expect("disk-derived budget should survive restart");
            assert_eq!(loaded.token_count, PAGE_TOKENS);
            assert_eq!(loaded.components[0].len(), payload_len);
        }
    }

    #[test]
    fn oversized_page_rejection_is_observable() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1024);

        tier.store("too-large", 512, "resident-kv", &[&[7u8; 1025]], None)
            .unwrap();

        let stats = tier.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.pages_rejected_too_large, 1);
        assert_eq!(stats.last_rejected_page_bytes, 1025);
    }

    /// Verification is ~95% of restore cost on a multi-gigabyte page, so a
    /// repeat load of a live mapping must not pay it again.
    #[test]
    fn a_repeat_load_of_a_live_mapping_skips_re_verification() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        tier.store("page-a", 512, "full-state", &[&[7u8; 4096]], None)
            .unwrap();

        tier.load("page-a", "full-state").unwrap().unwrap();
        tier.load("page-a", "full-state").unwrap().unwrap();
        tier.load("page-a", "full-state").unwrap().unwrap();

        let stats = tier.stats();
        assert_eq!(stats.verifications, 1, "first load must verify");
        assert_eq!(
            stats.verifications_skipped, 2,
            "later loads of the same mapping must not re-hash"
        );
    }

    /// The safety condition on skipping: trust is tied to a live mapping, so
    /// a fresh process -- which could be looking at a different file -- must
    /// verify again.
    #[test]
    fn reopening_the_tier_verifies_again() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut tier = tier(&dir, 1 << 20);
            tier.store("page-a", 512, "full-state", &[&[7u8; 4096]], None)
                .unwrap();
            tier.load("page-a", "full-state").unwrap().unwrap();
            tier.load("page-a", "full-state").unwrap().unwrap();
            assert_eq!(tier.stats().verifications_skipped, 1);
        }

        let mut reopened = tier(&dir, 1 << 20);
        reopened.load("page-a", "full-state").unwrap().unwrap();
        assert_eq!(
            reopened.stats().verifications,
            1,
            "a new process must not inherit trust from the previous one"
        );
    }

    /// Skipping must never let corruption through on a page this process has
    /// not actually verified.
    #[test]
    fn corruption_is_still_caught_on_the_first_load_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let file_name = {
            let mut tier = tier(&dir, 1 << 20);
            tier.store("page-a", 64, "full-state", &[&[3u8; 4096]], None)
                .unwrap();
            // Verify once, then drop the tier so the mapping and its trust go
            // with it.
            tier.load("page-a", "full-state").unwrap().unwrap();
            tier.entries["page-a"].file_name.clone()
        };

        let path = dir.path().join(ENTRY_DIR_NAME).join(&file_name);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        let mut reopened = tier(&dir, 1 << 20);
        let error = reopened
            .load("page-a", "full-state")
            .expect_err("corruption must be caught after a reopen");
        assert!(
            error.to_string().contains("checksum"),
            "expected a checksum failure, got: {error}"
        );
    }

    /// Corrupt bytes must never reach the runtime: that is silent numerical
    /// corruption, far worse than a miss.
    #[test]
    fn corrupted_entry_is_rejected_and_quarantined() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        tier.store("page-a", 64, "full-state", &[&[3u8; 4096]], None)
            .unwrap();

        // Corrupt the file in place, preserving its length so the size check
        // passes and only the checksum can catch it.
        let file_name = tier.entries["page-a"].file_name.clone();
        let path = tier.entry_path(&file_name);
        let mut corrupted = fs::read(&path).unwrap();
        corrupted[10] ^= 0xFF;
        fs::write(&path, &corrupted).unwrap();
        tier.mappings.clear();

        let error = tier.load("page-a", "full-state").unwrap_err();
        assert!(
            error.to_string().contains("checksum"),
            "unexpected error: {error}"
        );
        assert!(!tier.contains("page-a"), "corrupt entry must be dropped");
        assert_eq!(tier.stats().corrupt_entries, 1);
    }

    /// A page recorded under one payload kind must not be reinterpreted as
    /// another.
    #[test]
    fn payload_kind_mismatch_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        tier.store("page-a", 64, "kv-recurrent", &[&[1u8; 64]], None)
            .unwrap();

        assert!(tier.load("page-a", "full-state").is_err());
        assert!(!tier.contains("page-a"));
    }

    #[test]
    fn truncated_file_is_dropped_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let file_name = {
            let mut tier = tier(&dir, 1 << 20);
            tier.store("page-a", 64, "full-state", &[&[5u8; 4096]], None)
                .unwrap();
            tier.entries["page-a"].file_name.clone()
        };
        let path = dir.path().join(ENTRY_DIR_NAME).join(&file_name);
        fs::write(&path, vec![5u8; 100]).unwrap();

        let reopened = tier(&dir, 1 << 20);
        assert!(!reopened.contains("page-a"));
        assert_eq!(reopened.stats().bytes, 0);
    }

    #[test]
    fn size_budget_evicts_least_recently_used_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 10_000);
        tier.store("page-a", 64, "full-state", &[&[1u8; 4000]], None)
            .unwrap();
        tier.store("page-b", 64, "full-state", &[&[2u8; 4000]], None)
            .unwrap();
        // Touch page-a so page-b becomes the eviction victim.
        tier.load("page-a", "full-state").unwrap().unwrap();
        tier.store("page-c", 64, "full-state", &[&[3u8; 4000]], None)
            .unwrap();

        assert!(tier.stats().bytes <= 10_000);
        assert!(tier.contains("page-a"));
        assert!(!tier.contains("page-b"));
        assert!(tier.stats().evictions >= 1);
    }

    #[test]
    fn page_larger_than_the_budget_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1000);

        tier.store("huge", 64, "full-state", &[&[1u8; 5000]], None)
            .unwrap();

        assert!(!tier.contains("huge"));
        assert_eq!(tier.stats().bytes, 0);
    }

    #[test]
    fn zero_budget_disables_the_tier() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 0);

        tier.store("page-a", 64, "full-state", &[&[1u8; 64]], None)
            .unwrap();

        assert!(!tier.contains("page-a"));
    }

    #[test]
    fn stale_format_version_discards_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut tier = tier(&dir, 1 << 20);
            tier.store("page-a", 64, "full-state", &[&[1u8; 64]], None)
                .unwrap();
        }
        let index_path = dir.path().join(INDEX_FILE_NAME);
        let mut index: serde_json::Value =
            serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
        index["format_version"] = serde_json::json!(DISK_TIER_FORMAT_VERSION + 1);
        fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

        let reopened = tier(&dir, 1 << 20);
        assert_eq!(reopened.stats().entries, 0);
    }

    #[test]
    fn corrupt_index_discards_the_directory_without_failing() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut tier = tier(&dir, 1 << 20);
            tier.store("page-a", 64, "full-state", &[&[1u8; 64]], None)
                .unwrap();
        }
        fs::write(dir.path().join(INDEX_FILE_NAME), b"{not json").unwrap();

        let reopened = tier(&dir, 1 << 20);
        assert_eq!(reopened.stats().entries, 0);
    }

    /// A crash between writing a page file and committing the index leaves an
    /// orphan that must be reclaimed rather than leaked forever.
    #[test]
    fn orphan_page_files_are_reclaimed_on_open() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(ENTRY_DIR_NAME)).unwrap();
        let orphan = dir.path().join(ENTRY_DIR_NAME).join("orphan.kvp");
        fs::write(&orphan, vec![0u8; 1024]).unwrap();

        let _tier = tier(&dir, 1 << 20);

        assert!(!orphan.exists(), "orphan page file should be removed");
    }

    #[test]
    fn token_counts_report_lengths_available_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        tier.store("a", 2048, "full-state", &[&[1u8; 64]], None)
            .unwrap();
        tier.store("b", 512, "full-state", &[&[1u8; 64]], None)
            .unwrap();
        tier.store("c", 4096, "full-state", &[&[1u8; 64]], None)
            .unwrap();

        assert_eq!(tier.token_counts_at_most(2048), vec![2048, 512]);
    }

    #[test]
    fn repeated_store_of_the_same_page_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut tier = tier(&dir, 1 << 20);
        tier.store("page-a", 64, "full-state", &[&[1u8; 1024]], None)
            .unwrap();
        tier.store("page-a", 64, "full-state", &[&[1u8; 1024]], None)
            .unwrap();

        assert_eq!(tier.stats().entries, 1);
        assert_eq!(tier.stats().bytes, 1024);
    }
}

#[cfg(test)]
mod concurrency_and_safety_tests {
    use super::*;

    /// Two instances serving the same model hash to the same cache root. The
    /// tier is not concurrency-safe — `commit_index` is last-writer-wins and
    /// orphan reclaim deletes files it does not know about — so a second
    /// instance must decline rather than share and corrupt.
    #[test]
    fn a_second_instance_cannot_open_the_same_directory() {
        let dir = tempfile::tempdir().unwrap();
        let _first = PrefixDiskTier::open(dir.path(), 1 << 20).expect("first open succeeds");

        let second = PrefixDiskTier::open(dir.path(), 1 << 20);

        assert!(
            second.is_err(),
            "a concurrently held cache directory must not be opened twice"
        );
    }

    /// Releasing the first instance must hand ownership over cleanly.
    #[test]
    fn the_directory_can_be_reopened_after_release() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut tier = PrefixDiskTier::open(dir.path(), 1 << 20).unwrap();
            tier.store("page-a", 64, "full-state", &[&[1u8; 64]], None)
                .unwrap();
        }

        let reopened = PrefixDiskTier::open(dir.path(), 1 << 20);
        assert!(reopened.is_ok(), "lock must release when the tier drops");
        assert!(reopened.unwrap().contains("page-a"));
    }

    /// The lock file is not a cache entry and must survive orphan reclaim.
    #[test]
    fn orphan_reclaim_preserves_the_lock_file() {
        let dir = tempfile::tempdir().unwrap();
        let _tier = PrefixDiskTier::open(dir.path(), 1 << 20).unwrap();

        assert!(dir.path().join("owner.lock").exists());
    }

    /// File names come from a JSON index on disk. A traversal path must never
    /// be joined onto the cache root, or quarantine/eviction would delete
    /// files outside the cache.
    #[test]
    fn unsafe_file_names_are_rejected() {
        assert!(is_safe_file_name("abc123.kvp"));
        assert!(!is_safe_file_name("../../etc/passwd"));
        assert!(!is_safe_file_name("nested/name.kvp"));
        assert!(!is_safe_file_name(".."));
        assert!(!is_safe_file_name(""));
    }

    /// A crafted index entry pointing outside the cache must be dropped on
    /// load rather than trusted.
    #[test]
    fn index_entries_with_traversal_paths_are_dropped() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut tier = PrefixDiskTier::open(dir.path(), 1 << 20).unwrap();
            tier.store("page-a", 64, "full-state", &[&[1u8; 64]], None)
                .unwrap();
        }
        let index_path = dir.path().join(INDEX_FILE_NAME);
        let mut index: serde_json::Value =
            serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
        index["entries"][0]["file_name"] = serde_json::json!("../escaped.kvp");
        fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

        let tier = PrefixDiskTier::open(dir.path(), 1 << 20).unwrap();
        assert_eq!(tier.stats().entries, 0);
    }

    /// Metadata corruption must fail closed.
    ///
    /// The payload checksum does not cover `extra`, which for a KV page is
    /// the descriptor saying how to interpret the bytes. An index corrupted
    /// into still-valid JSON would otherwise hand correctly-checksummed bytes
    /// to the runtime under the wrong layout -- silent numerical corruption
    /// on a path that reports success.
    #[test]
    fn tampered_metadata_is_rejected_even_though_the_payload_is_intact() {
        let dir = tempfile::tempdir().expect("tempdir");
        let payload = vec![7u8; 256];
        {
            let mut tier = PrefixDiskTier::open(dir.path(), 1 << 20).expect("open");
            tier.store(
                "page-meta",
                128,
                "resident_kv_archive",
                &[&payload],
                Some(serde_json::json!({ "k_type": 1, "layer_start": 0 })),
            )
            .expect("store");
        }

        // Rewrite the descriptor the way a corrupted index would: still valid
        // JSON, payload untouched, so only the metadata digest can catch it.
        let index_path = dir.path().join("index.json");
        let raw = std::fs::read(&index_path).expect("read index");
        let mut index: serde_json::Value = serde_json::from_slice(&raw).expect("parse index");
        index["entries"][0]["extra"]["k_type"] = serde_json::json!(8);
        std::fs::write(&index_path, serde_json::to_vec(&index).expect("encode"))
            .expect("write index");

        let mut tier = PrefixDiskTier::open(dir.path(), 1 << 20).expect("reopen");
        let error = tier
            .load("page-meta", "resident_kv_archive")
            .expect_err("tampered metadata must be rejected");
        assert!(
            error.to_string().contains("metadata checksum"),
            "expected a metadata checksum failure, got: {error}"
        );
        assert!(
            tier.load("page-meta", "resident_kv_archive")
                .ok()
                .flatten()
                .is_none(),
            "a quarantined entry must not come back on a later load"
        );
    }

    /// Crash debris must not leak forever.
    ///
    /// A crash between writing a page file and committing the index leaves a
    /// `.tmp` file. Open holds the exclusive directory lock, so nothing can
    /// still be writing one, and skipping them leaked a full page's bytes per
    /// crash with nothing to ever reclaim it.
    #[test]
    fn crash_left_temp_files_are_reclaimed_on_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut tier = PrefixDiskTier::open(dir.path(), 1 << 20).expect("open");
            tier.store("page-live", 64, "kind", &[&[1u8; 64][..]], None)
                .expect("store");
        }
        let pages = dir.path().join(ENTRY_DIR_NAME);
        let debris = pages.join("kvp.999.1.tmp");
        std::fs::write(&debris, vec![0u8; 4096]).expect("write debris");
        assert!(debris.exists());

        let mut tier = PrefixDiskTier::open(dir.path(), 1 << 20).expect("reopen");
        assert!(!debris.exists(), "crash debris must be reclaimed at open");
        assert!(
            tier.load("page-live", "kind").expect("load").is_some(),
            "reclaiming debris must not disturb a live entry"
        );
    }
}
