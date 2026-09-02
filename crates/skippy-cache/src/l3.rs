//! L3 exact-state segment store.
//!
//! The durable tier under the radix cache: exported continuation state is cut
//! into content-addressed segments and committed under a manifest that records
//! ordering and completeness. Disk (this module) and the network handoff
//! stream are backends of the same contract:
//!
//! - **Segment identity**: every segment is addressed by the BLAKE3 digest of
//!   its bytes; reads verify the digest, so corruption is detected, never
//!   silently imported.
//! - **Ordering**: the manifest lists segments with explicit index/offset;
//!   assembly validates both.
//! - **Completeness**: a manifest only commits after every referenced segment
//!   is present and the assembled payload digest matches. Partial state can
//!   never be loaded — there is nothing to load until commit.
//! - **Idempotency**: putting a segment that already exists is a no-op;
//!   concurrent writers of the same bytes converge on one file via
//!   temp-file + atomic rename.
//! - **Capped budget**: `enforce_budget` evicts oldest manifests first (the
//!   newest is never evicted) and garbage-collects unreferenced segments.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::fsinfo;

const SEGMENT_DIR: &str = "segments";
const MANIFEST_DIR: &str = "manifests";
const PREFIX_INDEX_DIR: &str = "prefixes";
const QUARANTINE_DIR: &str = "quarantine";

/// Evict to this percentage of the budget rather than exactly to it.
///
/// An eviction pass is O(manifests x segments): it parses every manifest to
/// learn which segments would become unreferenced. Measured at 812 ms for 20
/// manifests of ~9.5k segments, which is what a 19K-token prefix costs at a
/// 64-row window. Evicting exactly to the cap means a full cache pays that on
/// every commit; leaving headroom amortises it over the writes that fill the
/// headroom back up.
const EVICTION_LOW_WATER_PERCENT: u64 = 85;
/// On-disk format version stamped into every manifest. A released change to
/// the layout bumps this and makes older entries misses, never migrations.
pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffSegmentRef {
    pub index: u32,
    pub offset: u64,
    pub bytes: u64,
    pub digest: String,
    /// Per-segment metadata for page-stream payloads (serialized
    /// `RuntimeKvPageDesc` plus token range), opaque to the store.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffManifest {
    pub version: u32,
    /// `exact_state_identity` of the producing runtime — the numerical
    /// identity a loader must match before importing this state.
    pub state_identity: String,
    pub payload_kind: String,
    pub total_bytes: u64,
    /// BLAKE3 of the assembled payload; also the manifest's key.
    pub payload_digest: String,
    pub segments: Vec<HandoffSegmentRef>,
    pub kv_bytes: u64,
    pub recurrent_bytes: u64,
    /// Serialized `RuntimeKvPageDesc` for kv-recurrent payloads; opaque to
    /// this crate so the store does not depend on the runtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_desc_json: Option<String>,
    pub token_count: u64,
    pub continuation_token: i32,
    /// Greedy continuation produced by the exporting session, when known —
    /// lets an offline restore self-verify determinism.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_tokens: Vec<i32>,
}

impl HandoffManifest {
    pub fn new(state_identity: String, payload_kind: String) -> Self {
        Self {
            version: MANIFEST_VERSION,
            state_identity,
            payload_kind,
            total_bytes: 0,
            payload_digest: String::new(),
            segments: Vec::new(),
            kv_bytes: 0,
            recurrent_bytes: 0,
            kv_desc_json: None,
            token_count: 0,
            continuation_token: 0,
            expected_tokens: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentPut {
    pub new: bool,
    pub bytes: u64,
}

/// How a payload is laid out, so segments can be cut where a growing prefix
/// keeps its bytes still.
///
/// The runtime exports exact state as a sequence of runs, each holding one
/// token-row per token in token order: every layer's K, then every layer's V,
/// then the indexer rows. Adding tokens extends every run, so a cut at a fixed
/// byte offset lands in a different place each turn and nothing dedupes — the
/// measured cost was 8x the newly committed bytes at 8 MiB segments.
///
/// Cutting each run into fixed windows of token-rows instead means turn N+1's
/// segments are byte-identical to turn N's up to the last partial window, and
/// only genuinely new state is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadGeometry {
    /// Runs in wire order.
    pub blocks: Vec<GeometryBlock>,
    /// Token-rows in every run.
    pub rows: u64,
    /// Rows per segment. Must depend only on the model's own shape, never on
    /// how many tokens this particular entry holds, or the boundaries move
    /// between turns and the dedupe is lost.
    pub window_rows: u64,
    /// Trailing bytes with no row structure (a recurrent snapshot), cut at the
    /// store's default segment size.
    pub tail_bytes: u64,
}

/// One run of token-rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryBlock {
    /// Bytes per token-row.
    pub stride: u64,
    /// What this run holds, for the segment metadata: `k`, `v` or `kidx`.
    pub kind: GeometryKind,
    /// Which layer, relative to the exported range.
    pub layer: u32,
    /// Sub-run within the layer. Always 0 except for transposed V, where each
    /// embedding column is its own token-contiguous run.
    pub column: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryKind {
    Key,
    Value,
    KeyIndex,
}

impl GeometryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Key => "k",
            Self::Value => "v",
            Self::KeyIndex => "kidx",
        }
    }
}

impl PayloadGeometry {
    /// Bytes the geometry claims, for checking it against the payload it is
    /// meant to describe.
    pub fn total_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .map(|block| block.stride.saturating_mul(self.rows))
            .fold(0u64, u64::saturating_add)
            .saturating_add(self.tail_bytes)
    }

    /// A geometry that does not describe this payload exactly is not usable:
    /// cutting to it would produce segments that reassemble to different
    /// bytes. Callers fall back to fixed-size cutting.
    pub fn matches(&self, payload_bytes: u64) -> bool {
        self.rows > 0
            && self.window_rows > 0
            && !self.blocks.is_empty()
            && self.blocks.iter().all(|block| block.stride > 0)
            && self.total_bytes() == payload_bytes
    }

    /// The cuts, as `(offset, len, label)` in wire order.
    pub fn plan(&self, tail_segment_bytes: u64) -> Vec<(u64, u64, String)> {
        let mut cuts = Vec::new();
        let mut offset = 0u64;
        for block in &self.blocks {
            let mut row = 0u64;
            while row < self.rows {
                let rows = self.window_rows.min(self.rows - row);
                let len = rows.saturating_mul(block.stride);
                cuts.push((
                    offset,
                    len,
                    format!(
                        "{}:{}:{}:{row}",
                        block.kind.as_str(),
                        block.layer,
                        block.column
                    ),
                ));
                offset = offset.saturating_add(len);
                row += rows;
            }
        }
        let tail_cut = tail_segment_bytes.max(1);
        let mut remaining = self.tail_bytes;
        while remaining > 0 {
            let len = tail_cut.min(remaining);
            cuts.push((offset, len, "tail".to_string()));
            offset = offset.saturating_add(len);
            remaining -= len;
        }
        cuts
    }
}

/// What the store is allowed to occupy. Both bounds are hard: the budget caps
/// what the cache manages, the reserve caps what it may take from everything
/// else on the filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLimits {
    /// Whole-root cap over segments, manifests, indexes and in-flight bytes.
    /// Zero disables the cap; the public configuration surface rejects zero
    /// rather than treating it as unlimited, so only internal callers and
    /// tests can reach that state.
    pub budget_bytes: u64,
    /// Free space the store preserves for everything else on the filesystem.
    pub minimum_free_bytes: u64,
}

impl StoreLimits {
    pub fn new(budget_bytes: u64, minimum_free_bytes: u64) -> Self {
        Self {
            budget_bytes,
            minimum_free_bytes,
        }
    }
}

/// Why a write was not admitted. Every refusal is a miss with a stable reason,
/// never a partial or best-effort write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteRefusal {
    /// The entry alone is larger than the whole budget. Storing it could only
    /// succeed by evicting everything else and still overflowing.
    SkippedOversize,
    /// The budget is full and eviction cannot free enough (everything else is
    /// pinned or the entry needs more than the whole cap).
    InsufficientSpace,
    /// The filesystem is at the minimum-free reserve. Reads still serve; the
    /// tier stops writing until space comes back.
    ReadOnlyLowSpace,
}

impl WriteRefusal {
    /// Stable reason code for the status surface.
    pub fn reason(self) -> &'static str {
        match self {
            Self::SkippedOversize => "skipped_oversize",
            Self::InsufficientSpace => "insufficient_space",
            Self::ReadOnlyLowSpace => "read_only_low_space",
        }
    }
}

/// Capacity held for bytes that are being written but not yet committed.
/// Counted against the budget for as long as it lives, so two concurrent
/// writers cannot each pass an admission check and together overflow the cap.
#[derive(Debug)]
pub struct Reservation<'store> {
    store: &'store HandoffSegmentStore,
    bytes: u64,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.store
            .reserved_inflight
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Holds one segment against collection while its manifest is being built.
///
/// A writer puts every segment before committing the manifest that binds them,
/// so for that window the bytes are unreferenced and eviction would collect
/// them out from under the commit about to name them. The hold lasts exactly
/// as long as the writer keeps the guard, so an abandoned write releases on
/// drop rather than leaking until restart.
#[derive(Debug)]
pub struct SegmentHold<'store> {
    store: &'store HandoffSegmentStore,
    digest: String,
}

impl Drop for SegmentHold<'_> {
    fn drop(&mut self) {
        self.store
            .inflight_segments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.digest);
    }
}

/// A stored segment and the hold that keeps it collectable-proof until the
/// caller's manifest commits.
#[derive(Debug)]
pub struct StoredSegment<'store> {
    pub digest: String,
    pub put: SegmentPut,
    _hold: SegmentHold<'store>,
}

/// Holds one manifest against eviction while it is being read or written.
#[derive(Debug)]
pub struct ManifestPin<'store> {
    store: &'store HandoffSegmentStore,
    key: String,
}

impl Drop for ManifestPin<'_> {
    fn drop(&mut self) {
        let mut pins = self
            .store
            .pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(count) = pins.get_mut(&self.key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                pins.remove(&self.key);
            }
        }
    }
}

/// What the store currently holds, for the status contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StoreUsage {
    pub budget_bytes: u64,
    pub used_bytes: u64,
    pub reserved_inflight_bytes: u64,
    pub filesystem_available_bytes: u64,
    pub minimum_free_bytes: u64,
    pub manifests: u64,
    pub unique_segments: u64,
    pub evicted_manifests: u64,
    pub quarantined_objects: u64,
}

#[derive(Debug)]
pub struct HandoffSegmentStore {
    root: PathBuf,
    limits: StoreLimits,
    /// Bytes reserved by in-flight writes. Part of the managed total, so the
    /// cap holds across concurrent writers rather than only at rest.
    reserved_inflight: AtomicU64,
    /// Manifests an active reader or writer is using. A pinned manifest is
    /// never evicted, pruned or cleared out from under the operation.
    pins: Mutex<std::collections::BTreeMap<String, usize>>,
    /// Manifests removed by budget enforcement or prune since open.
    evicted_manifests: AtomicU64,
    /// Objects moved to quarantine after failing verification since open.
    quarantined_objects: AtomicU64,
    /// Segments written but not yet referenced by a committed manifest.
    ///
    /// A writer puts every segment before committing the manifest that binds
    /// them, so for that window the bytes are unreferenced — and eviction
    /// triggered by another writer (or by this one needing room) would
    /// collect them out from under the commit that is about to reference
    /// them. Left unprotected this fails as "manifest references missing
    /// segment" under exactly the pressure the cache is for.
    inflight_segments: Mutex<std::collections::BTreeSet<String>>,
}

pub fn segment_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

impl HandoffSegmentStore {
    /// Open (creating if needed) a store rooted at `root`, capped by
    /// `budget_bytes` with no free-space reserve. Prefer
    /// [`Self::open_with_limits`]; this exists for callers that predate the
    /// reserve.
    pub fn open(root: impl Into<PathBuf>, budget_bytes: u64) -> Result<Self> {
        Self::open_with_limits(root, StoreLimits::new(budget_bytes, 0))
    }

    /// Open (creating if needed) a store rooted at `root` under `limits`.
    ///
    /// Refuses a root reached through a symlink and a root on a network
    /// filesystem: both break the atomic-rename and containment assumptions
    /// every later guarantee rests on, and neither is worth a partial mode.
    pub fn open_with_limits(root: impl Into<PathBuf>, limits: StoreLimits) -> Result<Self> {
        let root = root.into();
        fsinfo::refuse_symlink(&root)?;
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create cache root {}", root.display()))?;
        if fsinfo::is_network_filesystem(&root).unwrap_or(false) {
            let name = fsinfo::filesystem_type_name(&root).unwrap_or_default();
            bail!(
                "{} is on an unsupported network filesystem ({name}); the disk cache requires local storage",
                root.display()
            );
        }
        for directory in [SEGMENT_DIR, MANIFEST_DIR, PREFIX_INDEX_DIR] {
            let path = root.join(directory);
            fs::create_dir_all(&path).with_context(|| {
                format!("failed to create {directory} dir under {}", root.display())
            })?;
            fsinfo::restrict_to_owner(&path, 0o700)?;
        }
        fsinfo::restrict_to_owner(&root, 0o700)?;
        Ok(Self {
            root,
            limits,
            reserved_inflight: AtomicU64::new(0),
            pins: Mutex::new(std::collections::BTreeMap::new()),
            evicted_manifests: AtomicU64::new(0),
            quarantined_objects: AtomicU64::new(0),
            inflight_segments: Mutex::new(std::collections::BTreeSet::new()),
        })
    }

    fn segment_path(&self, digest: &str) -> PathBuf {
        self.root.join(SEGMENT_DIR).join(format!("{digest}.seg"))
    }

    fn manifest_path(&self, payload_digest: &str) -> PathBuf {
        self.root
            .join(MANIFEST_DIR)
            .join(format!("{payload_digest}.json"))
    }

    fn namespace_dir(&self, namespace_key: &str) -> PathBuf {
        let key = namespace_key
            .strip_prefix("blake3:")
            .unwrap_or(namespace_key);
        self.root.join(PREFIX_INDEX_DIR).join(key)
    }

    fn prefix_entry_path(&self, namespace_key: &str, token_len: u64, prefix_key: &str) -> PathBuf {
        // Zero-padded length keeps directory listings sorted and lets the
        // lookup filter by length without parsing every name.
        let key = prefix_key.strip_prefix("blake3:").unwrap_or(prefix_key);
        self.namespace_dir(namespace_key)
            .join(format!("{token_len:012}-{key}.key"))
    }

    /// Bind a (namespace, token-length, prefix-hash) coordinate to a
    /// committed manifest. Entries at many lengths coexist, which is what
    /// makes longest-recorded-prefix lookup work: each spill is a complete
    /// state for its own length, and later, longer prompts find the longest
    /// spilled length that is a prefix of theirs.
    pub fn link_prefix(
        &self,
        namespace_key: &str,
        token_len: u64,
        prefix_key: &str,
        payload_digest: &str,
    ) -> Result<()> {
        let path = self.prefix_entry_path(namespace_key, token_len, prefix_key);
        fs::create_dir_all(path.parent().context("prefix entry has no parent")?)?;
        write_atomically(&path, payload_digest.as_bytes())
    }

    /// Recorded token lengths for a namespace, longest first, deduplicated.
    pub fn recorded_prefix_lengths(&self, namespace_key: &str) -> Result<Vec<u64>> {
        let dir = self.namespace_dir(namespace_key);
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut lengths = Vec::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some((length, _)) = name.split_once('-') else {
                continue;
            };
            if let Ok(length) = length.parse::<u64>() {
                lengths.push(length);
            }
        }
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths.dedup();
        Ok(lengths)
    }

    /// The manifest recorded at exactly (namespace, token-length,
    /// prefix-hash), pruning links whose manifest was evicted. Transient
    /// I/O errors are surfaced, not treated as absence, so a briefly
    /// unreadable disk cannot delete healthy links.
    pub fn manifest_for_prefix(
        &self,
        namespace_key: &str,
        token_len: u64,
        prefix_key: &str,
    ) -> Result<Option<HandoffManifest>> {
        let path = self.prefix_entry_path(namespace_key, token_len, prefix_key);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let payload_digest = String::from_utf8(bytes).context("malformed prefix link")?;
        let manifest_path = self.manifest_path(&payload_digest);
        match fs::read(&manifest_path) {
            Ok(bytes) => {
                let manifest: HandoffManifest =
                    serde_json::from_slice(&bytes).context("malformed manifest")?;
                if manifest.version != MANIFEST_VERSION {
                    bail!(
                        "manifest {payload_digest} has version {} but this build reads {MANIFEST_VERSION}",
                        manifest.version
                    );
                }
                // A hit is a use. Recording it here is what makes eviction
                // least-recently-*used* rather than least-recently-written.
                self.touch_manifest(&payload_digest);
                Ok(Some(manifest))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // The manifest was evicted after the link was written; only
                // this definite absence prunes the link.
                let _ = fs::remove_file(&path);
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Content-addressed, idempotent put. Concurrent writers of the same
    /// bytes race benignly: both write temp files, both rename onto the same
    /// final path.
    ///
    /// Stores one segment, or reports why it was refused.
    ///
    /// A segment already present is a no-op that costs nothing and is always
    /// admitted: content addressing means the bytes on disk are the bytes
    /// being written, so re-putting cannot grow the store.
    pub fn put_segment(&self, bytes: &[u8]) -> Result<StoredSegment<'_>> {
        match self.try_put_segment(bytes)? {
            Ok(stored) => Ok(stored),
            Err(refusal) => bail!("cannot store segment: {}", refusal.reason()),
        }
    }

    /// As [`Self::put_segment`], returning the refusal rather than an error so
    /// callers can record a miss with its reason and carry on.
    pub fn try_put_segment(&self, bytes: &[u8]) -> Result<Result<StoredSegment<'_>, WriteRefusal>> {
        let digest = segment_digest(bytes);
        let path = self.segment_path(&digest);
        if path.exists() {
            // Already stored, but possibly unreferenced: hold it too, or a
            // concurrent eviction can collect it before this writer's manifest
            // names it.
            let hold = self.hold_segment(&digest);
            return Ok(Ok(StoredSegment {
                digest,
                put: SegmentPut {
                    new: false,
                    bytes: bytes.len() as u64,
                },
                _hold: hold,
            }));
        }
        // Reserve before any temporary bytes exist on disk, so two writers
        // cannot both pass the check and together exceed the cap.
        let reservation = match self.reserve(bytes.len() as u64)? {
            Ok(reservation) => reservation,
            Err(refusal) => return Ok(Err(refusal)),
        };
        write_atomically(&path, bytes)?;
        fsinfo::restrict_to_owner(&path, 0o600)?;
        let hold = self.hold_segment(&digest);
        drop(reservation);
        Ok(Ok(StoredSegment {
            digest,
            put: SegmentPut {
                new: true,
                bytes: bytes.len() as u64,
            },
            _hold: hold,
        }))
    }

    pub fn has_segment(&self, digest: &str) -> bool {
        self.segment_path(digest).exists()
    }

    /// Read one segment, verifying its content digest.
    pub fn read_segment(&self, digest: &str) -> Result<Vec<u8>> {
        let path = self.segment_path(digest);
        let bytes = fs::read(&path).with_context(|| format!("failed to read segment {digest}"))?;
        if segment_digest(&bytes) != digest {
            self.quarantine(&path)?;
            bail!("segment {digest} failed digest verification on read and was quarantined");
        }
        Ok(bytes)
    }

    /// Move a corrupt or truncated object out of the managed tree.
    ///
    /// A segment that fails verification can never become valid: every
    /// manifest referencing it is already a miss, and leaving it in place
    /// means paying to read and reject it again on the next hit. Quarantine
    /// keeps the bytes under `quarantine/` for inspection while taking them
    /// out of every lookup path.
    fn quarantine(&self, path: &Path) -> Result<()> {
        let directory = self.root.join(QUARANTINE_DIR);
        fs::create_dir_all(&directory).with_context(|| {
            format!(
                "failed to create quarantine dir under {}",
                self.root.display()
            )
        })?;
        fsinfo::restrict_to_owner(&directory, 0o700)?;
        let name = path
            .file_name()
            .context("quarantined path has no file name")?;
        self.quarantined_objects.fetch_add(1, Ordering::Relaxed);
        match fs::rename(path, directory.join(name)) {
            Ok(()) => Ok(()),
            // Losing the evidence beats serving corrupt state, so a failed
            // move falls back to removal.
            Err(_) => fs::remove_file(path)
                .with_context(|| format!("failed to quarantine {}", path.display())),
        }
    }

    /// Commit a manifest. Fails unless every referenced segment is present
    /// with the recorded size and offsets tile the payload exactly — the
    /// completeness gate that makes partial state unloadable.
    pub fn commit(&self, manifest: &HandoffManifest) -> Result<()> {
        if manifest.payload_digest.is_empty() {
            bail!("manifest has no payload digest");
        }
        let mut expected_offset = 0u64;
        for (position, segment) in manifest.segments.iter().enumerate() {
            if segment.index as usize != position {
                bail!(
                    "manifest segment order broken: index {} at position {position}",
                    segment.index
                );
            }
            if segment.offset != expected_offset {
                bail!(
                    "manifest segment {} offset {} does not tile payload (expected {expected_offset})",
                    segment.index,
                    segment.offset
                );
            }
            let path = self.segment_path(&segment.digest);
            let metadata = fs::metadata(&path).with_context(|| {
                format!(
                    "manifest references missing segment {} ({})",
                    segment.index, segment.digest
                )
            })?;
            if metadata.len() != segment.bytes {
                bail!(
                    "segment {} has {} bytes on disk but manifest records {}",
                    segment.digest,
                    metadata.len(),
                    segment.bytes
                );
            }
            expected_offset = expected_offset
                .checked_add(segment.bytes)
                .context("manifest offsets overflow")?;
        }
        if expected_offset != manifest.total_bytes {
            bail!(
                "manifest segments cover {expected_offset} bytes but total_bytes is {}",
                manifest.total_bytes
            );
        }
        if self.limits.budget_bytes > 0 && manifest.total_bytes > self.limits.budget_bytes {
            bail!(
                "manifest {} spans {} bytes, more than the {} byte budget ({})",
                manifest.payload_digest,
                manifest.total_bytes,
                self.limits.budget_bytes,
                WriteRefusal::SkippedOversize.reason()
            );
        }
        // Compact, not pretty: a 64-row window turns a long prefix into
        // thousands of segment refs, and eviction parses every manifest to
        // build its reference map. Indentation is pure cost on a file nobody
        // reads by hand.
        let serialized = serde_json::to_vec(manifest).context("failed to serialize manifest")?;
        // The manifest is what makes the segments loadable, so it is pinned
        // while it lands: eviction triggered by its own admission check must
        // not remove the entry being committed.
        let _pin = self.pin(&manifest.payload_digest);
        match self.reserve(serialized.len() as u64)? {
            Ok(_reservation) => {
                write_atomically(&self.manifest_path(&manifest.payload_digest), &serialized)?;
            }
            Err(refusal) => {
                bail!(
                    "cannot commit manifest {}: {}",
                    manifest.payload_digest,
                    refusal.reason()
                );
            }
        }
        self.enforce_budget()?;
        Ok(())
    }

    /// Record that an entry was used, so eviction can order by last use.
    /// Best-effort: a store on a read-only path still serves reads, and
    /// failing a restore because recency could not be recorded would trade a
    /// working cache for a bookkeeping detail.
    pub fn touch_manifest(&self, payload_digest: &str) {
        let _ = fsinfo::touch(&self.manifest_path(payload_digest));
    }

    pub fn load_manifest(&self, payload_digest: &str) -> Result<HandoffManifest> {
        let bytes = fs::read(self.manifest_path(payload_digest))
            .with_context(|| format!("failed to read manifest {payload_digest}"))?;
        let manifest: HandoffManifest =
            serde_json::from_slice(&bytes).context("malformed manifest")?;
        if manifest.version != MANIFEST_VERSION {
            bail!(
                "manifest {payload_digest} has version {} but this build reads {MANIFEST_VERSION}",
                manifest.version
            );
        }
        Ok(manifest)
    }

    /// Manifest keys, newest first by modification time.
    pub fn list_manifests(&self) -> Result<Vec<String>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(self.root.join(MANIFEST_DIR))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            let Some(stem) = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
            else {
                continue;
            };
            let modified = entry.metadata()?.modified()?;
            entries.push((modified, stem));
        }
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        Ok(entries.into_iter().map(|(_, stem)| stem).collect())
    }

    /// Assemble the full payload for a manifest, verifying every segment
    /// digest, the tiling, and the whole-payload digest.
    pub fn assemble(&self, manifest: &HandoffManifest) -> Result<Vec<u8>> {
        let total = usize::try_from(manifest.total_bytes).context("payload exceeds usize")?;
        let mut payload = Vec::with_capacity(total);
        for segment in &manifest.segments {
            if segment.offset != payload.len() as u64 {
                bail!(
                    "segment {} offset {} does not match assembled length {}",
                    segment.index,
                    segment.offset,
                    payload.len()
                );
            }
            payload.extend_from_slice(&self.read_segment(&segment.digest)?);
        }
        if payload.len() != total {
            bail!(
                "assembled {} bytes but manifest records {total}",
                payload.len()
            );
        }
        if segment_digest(&payload) != manifest.payload_digest {
            bail!("assembled payload failed manifest digest verification");
        }
        Ok(payload)
    }

    pub fn segment_footprint_bytes(&self) -> Result<u64> {
        directory_bytes(&self.root.join(SEGMENT_DIR))
    }

    /// Every byte the store manages: committed segments, the manifests that
    /// make them loadable, the prefix index that finds them, and capacity
    /// reserved by writes in flight.
    ///
    /// The budget is a physical cap on the cache root, so it has to count the
    /// bookkeeping too. Counting segments alone lets a store with many small
    /// entries sit well over its stated cap in manifests and index files.
    pub fn managed_usage_bytes(&self) -> Result<u64> {
        let mut total = directory_bytes(&self.root.join(SEGMENT_DIR))?;
        total = total.saturating_add(directory_bytes(&self.root.join(MANIFEST_DIR))?);
        total = total.saturating_add(directory_bytes_recursive(
            &self.root.join(PREFIX_INDEX_DIR),
        )?);
        Ok(total.saturating_add(self.reserved_inflight.load(Ordering::Acquire)))
    }

    /// Reserve capacity for `bytes` about to be written.
    ///
    /// Reservation happens before any temporary payload bytes exist on disk,
    /// so the cap and the free-space reserve hold even while several writers
    /// are mid-write. The returned guard releases the reservation on drop,
    /// including on the error paths.
    pub fn reserve(&self, bytes: u64) -> Result<Result<Reservation<'_>, WriteRefusal>> {
        if self.limits.budget_bytes > 0 && bytes > self.limits.budget_bytes {
            return Ok(Err(WriteRefusal::SkippedOversize));
        }
        let available = fsinfo::available_bytes(&self.root)?;
        if available.saturating_sub(bytes) < self.limits.minimum_free_bytes {
            return Ok(Err(WriteRefusal::ReadOnlyLowSpace));
        }
        if self.limits.budget_bytes > 0 {
            let used = self.managed_usage_bytes()?;
            if used.saturating_add(bytes) > self.limits.budget_bytes {
                // Make room from inactive entries before refusing: a full
                // cache is the normal steady state, not an error. Clear to the
                // low-water mark, or further when this write alone needs more.
                let target = self
                    .low_water_bytes()
                    .min(self.limits.budget_bytes.saturating_sub(bytes));
                self.enforce_budget_to(target)?;
                let used = self.managed_usage_bytes()?;
                if used.saturating_add(bytes) > self.limits.budget_bytes {
                    return Ok(Err(WriteRefusal::InsufficientSpace));
                }
            }
        }
        self.reserved_inflight.fetch_add(bytes, Ordering::AcqRel);
        Ok(Ok(Reservation { store: self, bytes }))
    }

    /// Pin a manifest for the duration of a read or write. Eviction, prune
    /// and clear all skip pinned entries, so an in-progress restore never has
    /// its segments removed underneath it.
    pub fn pin(&self, payload_digest: &str) -> ManifestPin<'_> {
        let mut pins = self
            .pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *pins.entry(payload_digest.to_string()).or_insert(0) += 1;
        ManifestPin {
            store: self,
            key: payload_digest.to_string(),
        }
    }

    fn is_pinned(&self, payload_digest: &str) -> bool {
        self.pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(payload_digest)
            .is_some_and(|count| *count > 0)
    }

    /// What the store holds right now, for the status contract.
    pub fn usage(&self) -> Result<StoreUsage> {
        let manifests = self.list_manifests()?;
        let mut segments = std::collections::HashSet::new();
        for key in &manifests {
            if let Ok(manifest) = self.load_manifest(key) {
                for segment in manifest.segments {
                    segments.insert(segment.digest);
                }
            }
        }
        Ok(StoreUsage {
            budget_bytes: self.limits.budget_bytes,
            used_bytes: self.managed_usage_bytes()?,
            reserved_inflight_bytes: self.reserved_inflight.load(Ordering::Acquire),
            filesystem_available_bytes: fsinfo::available_bytes(&self.root)?,
            minimum_free_bytes: self.limits.minimum_free_bytes,
            manifests: manifests.len() as u64,
            unique_segments: segments.len() as u64,
            evicted_manifests: self.evicted_manifests.load(Ordering::Relaxed),
            quarantined_objects: self.quarantined_objects.load(Ordering::Relaxed),
        })
    }

    /// Namespaces with at least one indexed prefix. Each namespace is one
    /// numerical identity (model, layout, layer range, load mode), so this is
    /// how many distinct configurations the root currently serves.
    pub fn namespace_count(&self) -> Result<u64> {
        let entries = match fs::read_dir(self.root.join(PREFIX_INDEX_DIR)) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error.into()),
        };
        let mut count = 0u64;
        for entry in entries {
            if entry?.file_type()?.is_dir() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Evict least-recently-used manifests and collect unreferenced segments
    /// until managed usage fits the budget. Returns bytes freed.
    ///
    /// Recency is last *use*, not last write: reads touch their manifest, so a
    /// prefix that is hit constantly but never re-recorded is the last thing
    /// evicted rather than the first.
    pub fn enforce_budget(&self) -> Result<u64> {
        if self.limits.budget_bytes == 0 {
            return Ok(0);
        }
        // Over the cap, evict below it: the pass is expensive enough that
        // paying it once per headroom refill beats paying it per commit.
        if self.managed_usage_bytes()? <= self.limits.budget_bytes {
            return Ok(0);
        }
        self.enforce_budget_to(self.low_water_bytes())
    }

    /// The level eviction drops to once it runs.
    fn low_water_bytes(&self) -> u64 {
        self.limits
            .budget_bytes
            .saturating_mul(EVICTION_LOW_WATER_PERCENT)
            / 100
    }

    /// Evict until managed usage is at or below `target_bytes`.
    ///
    /// Single pass regardless of how many manifests evict: usage, the manifest
    /// list and the reference map are each scanned once, eviction runs against
    /// the in-memory model, and one final GC removes what became unreferenced.
    /// Pinned manifests are never evicted, so an in-flight read or write keeps
    /// its state loadable even under pressure.
    pub fn enforce_budget_to(&self, target_bytes: u64) -> Result<u64> {
        let usage_before = self.managed_usage_bytes()?;
        if usage_before <= target_bytes {
            return Ok(0);
        }
        // Least-recently-used last, so eviction pops from the back.
        let keys = self.list_manifests()?;
        let mut manifests = Vec::with_capacity(keys.len());
        let mut reference_counts: std::collections::HashMap<String, (usize, u64)> =
            std::collections::HashMap::new();
        for key in &keys {
            let Ok(manifest) = self.load_manifest(key) else {
                continue;
            };
            for segment in &manifest.segments {
                let entry = reference_counts
                    .entry(segment.digest.clone())
                    .or_insert((0, segment.bytes));
                entry.0 += 1;
            }
            manifests.push(manifest);
        }
        let mut freeable = usage_before;
        let mut evicted_any = false;
        while freeable > target_bytes {
            let Some(position) = manifests
                .iter()
                .rposition(|manifest| !self.is_pinned(&manifest.payload_digest))
            else {
                // Everything left is in use. Refusing the write is correct;
                // tearing state out from under a live operation is not.
                break;
            };
            let evicted = manifests.remove(position);
            let manifest_path = self.manifest_path(&evicted.payload_digest);
            let manifest_bytes = fs::metadata(&manifest_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            fs::remove_file(&manifest_path)
                .with_context(|| format!("failed to evict manifest {}", evicted.payload_digest))?;
            self.evicted_manifests.fetch_add(1, Ordering::Relaxed);
            freeable = freeable.saturating_sub(manifest_bytes);
            for segment in &evicted.segments {
                if let Some(entry) = reference_counts.get_mut(&segment.digest) {
                    entry.0 = entry.0.saturating_sub(1);
                    if entry.0 == 0 {
                        freeable = freeable.saturating_sub(entry.1);
                    }
                }
            }
            evicted_any = true;
        }
        if !evicted_any {
            return Ok(0);
        }
        self.collect_unreferenced_segments()?;
        let usage_after = self.managed_usage_bytes()?;
        Ok(usage_before.saturating_sub(usage_after))
    }

    /// Evict least-recently-used inactive manifests until managed usage fits
    /// `target_bytes`. The user-facing prune: it never removes an active
    /// entry, and it reports what it actually freed rather than what it
    /// intended to.
    pub fn prune_to(&self, target_bytes: u64) -> Result<u64> {
        self.enforce_budget_to(target_bytes)
    }

    /// Remove every manifest that is not pinned, then collect the segments
    /// that became unreferenced. Returns bytes freed.
    ///
    /// Requests in flight keep serving: clearing removes stored state, and a
    /// miss falls back to cold prefill.
    pub fn clear(&self) -> Result<u64> {
        let before = self.managed_usage_bytes()?;
        for key in self.list_manifests()? {
            if self.is_pinned(&key) {
                continue;
            }
            let path = self.manifest_path(&key);
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| format!("failed to clear manifest {key}"));
                }
            }
        }
        self.collect_unreferenced_segments()?;
        let after = self.managed_usage_bytes()?;
        Ok(before.saturating_sub(after))
    }

    fn hold_segment(&self, digest: &str) -> SegmentHold<'_> {
        self.inflight_segments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(digest.to_string());
        SegmentHold {
            store: self,
            digest: digest.to_string(),
        }
    }

    /// Remove segments referenced by no manifest. Returns bytes freed.
    pub fn collect_unreferenced_segments(&self) -> Result<u64> {
        let mut referenced = std::collections::HashSet::new();
        for key in self.list_manifests()? {
            if let Ok(manifest) = self.load_manifest(&key) {
                for segment in manifest.segments {
                    referenced.insert(segment.digest);
                }
            }
        }
        let mut freed = 0u64;
        for entry in fs::read_dir(self.root.join(SEGMENT_DIR))? {
            let entry = entry?;
            let path = entry.path();
            let Some(stem) = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
            else {
                continue;
            };
            let held = self
                .inflight_segments
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(&stem);
            if !referenced.contains(&stem) && !held {
                freed = freed.saturating_add(entry.metadata()?.len());
                fs::remove_file(&path)
                    .with_context(|| format!("failed to collect segment {stem}"))?;
            }
        }
        Ok(freed)
    }
}

/// Bytes held by the files directly inside `directory`. Missing directories
/// count as empty so a partially built root reports rather than fails.
fn directory_bytes(directory: &Path) -> Result<u64> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", directory.display()));
        }
    };
    let mut total = 0u64;
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

/// As [`directory_bytes`], following one level of namespace subdirectories:
/// the prefix index is stored per namespace.
fn directory_bytes_recursive(directory: &Path) -> Result<u64> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", directory.display()));
        }
    };
    let mut total = 0u64;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        } else if file_type.is_dir() {
            total = total.saturating_add(directory_bytes_recursive(&entry.path())?);
        }
    }
    Ok(total)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let directory = path.parent().context("path has no parent directory")?;
    let mut temp = tempfile_in(directory)?;
    temp.1
        .write_all(bytes)
        .with_context(|| format!("failed to write {}", temp.0.display()))?;
    temp.1
        .sync_all()
        .with_context(|| format!("failed to sync {}", temp.0.display()))?;
    drop(temp.1);
    fs::rename(&temp.0, path).with_context(|| format!("failed to publish {}", path.display()))?;
    Ok(())
}

fn tempfile_in(directory: &Path) -> Result<(PathBuf, fs::File)> {
    // Distinct per-writer temp names without a clock or RNG dependency:
    // process id + a process-local counter.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = directory.join(format!(".tmp-{}-{unique}", std::process::id()));
    let file = fs::File::create(&path)
        .with_context(|| format!("failed to create temp file {}", path.display()))?;
    Ok((path, file))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(root: &Path, budget: u64) -> HandoffSegmentStore {
        HandoffSegmentStore::open(root, budget).expect("open store")
    }

    /// Builds a manifest and returns the write-side holds alongside it: a
    /// caller that puts segments and commits later must keep them alive, or an
    /// eviction in between collects the segments it is about to reference.
    /// This is the contract `L3Tier::spill` follows in production.
    fn manifest_for<'store>(
        store: &'store HandoffSegmentStore,
        payload: &[u8],
        segment_bytes: usize,
    ) -> (HandoffManifest, Vec<StoredSegment<'store>>) {
        let mut manifest = HandoffManifest::new("blake3:test".to_string(), "full-state".into());
        let mut held = Vec::new();
        for (index, chunk) in payload.chunks(segment_bytes).enumerate() {
            let stored = store.put_segment(chunk).expect("put segment");
            manifest.segments.push(HandoffSegmentRef {
                index: index as u32,
                offset: (index * segment_bytes) as u64,
                bytes: chunk.len() as u64,
                digest: stored.digest.clone(),
                meta_json: None,
            });
            held.push(stored);
        }
        manifest.total_bytes = payload.len() as u64;
        manifest.payload_digest = segment_digest(payload);
        (manifest, held)
    }

    /// Put a payload's segments and commit the manifest that binds them,
    /// releasing the write-side holds afterwards. The shape production uses:
    /// hold across the commit, then let eviction have them.
    fn commit_payload(
        store: &HandoffSegmentStore,
        payload: &[u8],
        segment_bytes: usize,
    ) -> HandoffManifest {
        let (manifest, held) = manifest_for(store, payload, segment_bytes);
        store.commit(&manifest).expect("commit");
        drop(held);
        manifest
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join("skippy-l3-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn roundtrip_assembles_identical_payload() {
        let root = temp_root("roundtrip");
        let store = store(&root, 0);
        let payload: Vec<u8> = (0..100_000u32).map(|value| value as u8).collect();
        let manifest = commit_payload(&store, &payload, 4096);
        let loaded = store
            .load_manifest(&manifest.payload_digest)
            .expect("load manifest");
        assert_eq!(store.assemble(&loaded).expect("assemble"), payload);
    }

    #[test]
    fn puts_are_idempotent_and_deduplicated() {
        let root = temp_root("idempotent");
        let store = store(&root, 0);
        let first = store.put_segment(b"same bytes").expect("first put");
        let first_digest = first.digest.clone();
        let second = store.put_segment(b"same bytes").expect("second put");
        let second_digest = second.digest.clone();
        assert_eq!(first_digest, second_digest);
        assert!(first.put.new);
        assert!(!second.put.new);
        assert_eq!(store.segment_footprint_bytes().expect("footprint"), 10);
    }

    #[test]
    fn commit_rejects_missing_segments_and_bad_tiling() {
        let root = temp_root("completeness");
        let store = store(&root, 0);
        let payload = vec![7u8; 10_000];
        let (mut manifest, _held) = manifest_for(&store, &payload, 4096);

        let mut missing = manifest.clone();
        missing.segments[1].digest = segment_digest(b"never stored");
        assert!(store.commit(&missing).is_err());

        manifest.segments[2].offset += 1;
        assert!(store.commit(&manifest).is_err());
    }

    #[test]
    fn corrupted_segment_fails_verification_on_read() {
        let root = temp_root("corruption");
        let store = store(&root, 0);
        let payload = vec![42u8; 8192];
        let manifest = commit_payload(&store, &payload, 4096);

        let victim = store.segment_path(&manifest.segments[0].digest);
        let mut bytes = fs::read(&victim).expect("read segment file");
        bytes[0] ^= 0xFF;
        fs::write(&victim, bytes).expect("corrupt segment file");

        assert!(store.assemble(&manifest).is_err());
    }

    #[test]
    fn budget_evicts_the_least_recently_used_manifest() {
        let root = temp_root("budget");
        // Budget fits one payload but not two.
        let store = store(&root, 12_000);
        let old_payload = vec![1u8; 8_000];
        let new_payload = vec![2u8; 8_000];
        let old_manifest = commit_payload(&store, &old_payload, 4096);
        // Ensure a later mtime for the second manifest.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let new_manifest = commit_payload(&store, &new_payload, 4096);

        let manifests = store.list_manifests().expect("list");
        assert!(
            !manifests.contains(&old_manifest.payload_digest),
            "the older entry survived eviction: {manifests:?}"
        );
        assert_eq!(manifests, vec![new_manifest.payload_digest.clone()]);
        assert!(store.assemble(&new_manifest).is_ok());
        assert!(store.segment_footprint_bytes().expect("footprint") <= 12_000);
    }

    #[test]
    fn eviction_follows_last_use_not_last_write() {
        let root = temp_root("lru-by-use");
        // Fits two payloads plus bookkeeping, not three.
        let store = store(&root, 20_000);
        let first = commit_payload(&store, &vec![1u8; 8_000], 4096);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second = commit_payload(&store, &vec![2u8; 8_000], 4096);

        // The older entry is the one being read, so it is the one that should
        // survive. Under least-recently-written it would be evicted first.
        std::thread::sleep(std::time::Duration::from_millis(20));
        store.touch_manifest(&first.payload_digest);

        std::thread::sleep(std::time::Duration::from_millis(20));
        // A third entry the budget cannot hold: something must go.
        commit_payload(&store, &vec![3u8; 8_000], 4096);

        let manifests = store.list_manifests().expect("list");
        assert!(
            manifests.contains(&first.payload_digest),
            "the recently used entry was evicted: {manifests:?}"
        );
        assert!(
            !manifests.contains(&second.payload_digest),
            "the least recently used entry survived: {manifests:?}"
        );
    }

    #[test]
    fn a_segment_larger_than_the_budget_is_refused() {
        let root = temp_root("oversize-segment");
        let store = store(&root, 4_000);
        let refusal = store
            .try_put_segment(&vec![7u8; 16_000])
            .expect("put")
            .expect_err("a segment larger than the whole budget was stored");
        assert_eq!(refusal, WriteRefusal::SkippedOversize);
        assert_eq!(refusal.reason(), "skipped_oversize");
        assert_eq!(store.segment_footprint_bytes().expect("footprint"), 0);
    }

    #[test]
    fn an_entry_larger_than_a_shrunken_budget_is_refused_at_commit() {
        // A budget shrunk between runs is the realistic way an entry ends up
        // bigger than the cap: it was admissible when its segments were
        // written and is not any more.
        let root = temp_root("oversize-commit");
        let uncapped = store(&root, 0);
        let manifest = {
            let (manifest, held) = manifest_for(&uncapped, &vec![7u8; 16_000], 4_000);
            drop(held);
            manifest
        };
        drop(uncapped);

        let capped = store(&root, 8_000);
        let error = capped
            .commit(&manifest)
            .expect_err("an entry larger than the budget was committed");
        assert!(
            format!("{error:#}").contains("skipped_oversize"),
            "refusal did not carry the reason code: {error:#}"
        );
        assert!(
            capped.list_manifests().expect("list").is_empty(),
            "the refused entry was left loadable"
        );
    }

    #[test]
    fn managed_usage_counts_more_than_segments() {
        let root = temp_root("usage");
        let store = store(&root, 0);
        let manifest = commit_payload(&store, &vec![5u8; 4096], 4096);
        store.commit(&manifest).expect("commit");
        store
            .link_prefix("namespace", 128, "prefix", &manifest.payload_digest)
            .expect("link prefix");

        let segments = store.segment_footprint_bytes().expect("footprint");
        let managed = store.managed_usage_bytes().expect("usage");
        assert!(
            managed > segments,
            "managed usage {managed} ignored manifests and index files (segments {segments})"
        );
    }

    #[test]
    fn a_pinned_manifest_outranks_a_new_write() {
        // Under pressure the store refuses the incoming write rather than
        // pulling state out from under an operation still using it. The new
        // entry is a miss; the pinned one stays loadable.
        let root = temp_root("pinned");
        let store = store(&root, 12_000);
        let pinned = commit_payload(&store, &vec![1u8; 8_000], 4096);
        store.commit(&pinned).expect("commit pinned");
        let guard = store.pin(&pinned.payload_digest);

        std::thread::sleep(std::time::Duration::from_millis(20));
        let refusal = store
            .try_put_segment(&vec![2u8; 8_000])
            .expect("put")
            .expect_err("the pinned entry was evicted to admit a new write");
        assert_eq!(refusal, WriteRefusal::InsufficientSpace);

        let manifests = store.list_manifests().expect("list");
        assert!(
            manifests.contains(&pinned.payload_digest),
            "a pinned manifest was evicted: {manifests:?}"
        );

        // Once nothing is using it, the same write is admitted.
        drop(guard);
        store
            .try_put_segment(&vec![2u8; 8_000])
            .expect("put")
            .expect("the write stayed refused after the pin was released");
    }

    #[test]
    fn the_free_space_reserve_refuses_writes() {
        let root = temp_root("reserve");
        // A reserve no filesystem can satisfy, rather than one derived from
        // live free space: another test freeing a few MiB mid-run must not
        // decide whether this one passes.
        let store = HandoffSegmentStore::open_with_limits(&root, StoreLimits::new(0, u64::MAX))
            .expect("open store");
        let refusal = store
            .try_put_segment(b"bytes that do not fit the reserve")
            .expect("put")
            .expect_err("write was admitted below the reserve");
        assert_eq!(refusal, WriteRefusal::ReadOnlyLowSpace);
        assert_eq!(refusal.reason(), "read_only_low_space");
    }

    #[test]
    fn reservations_are_released_after_the_write() {
        let root = temp_root("reservation");
        let store = store(&root, 1_000_000);
        store.put_segment(b"some bytes").expect("put");
        assert_eq!(
            store.usage().expect("usage").reserved_inflight_bytes,
            0,
            "a completed write left capacity reserved"
        );
    }

    #[test]
    fn clear_removes_every_unpinned_entry() {
        let root = temp_root("clear");
        let store = store(&root, 0);
        commit_payload(&store, &vec![1u8; 4096], 4096);
        commit_payload(&store, &vec![2u8; 4096], 4096);

        let freed = store.clear().expect("clear");
        assert!(freed > 0, "clear freed nothing");
        assert!(store.list_manifests().expect("list").is_empty());
        assert_eq!(store.segment_footprint_bytes().expect("footprint"), 0);
    }

    #[test]
    fn prune_frees_down_to_the_target() {
        let root = temp_root("prune");
        let store = store(&root, 0);
        for fill in 1u8..=3 {
            let manifest = commit_payload(&store, &vec![fill; 8_000], 4096);
            store.commit(&manifest).expect("commit");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let before = store.managed_usage_bytes().expect("usage");
        store.prune_to(before / 2).expect("prune");
        let after = store.managed_usage_bytes().expect("usage");
        assert!(after < before, "prune freed nothing ({before} -> {after})");
    }

    #[test]
    fn a_corrupt_segment_is_quarantined_not_left_in_place() {
        let root = temp_root("quarantine");
        let store = store(&root, 0);
        let digest = store.put_segment(b"segment bytes").expect("put").digest;
        fs::write(
            root.join("segments").join(format!("{digest}.seg")),
            b"tampered",
        )
        .expect("tamper with the segment");

        let error = store
            .read_segment(&digest)
            .expect_err("a tampered segment was served");
        assert!(
            format!("{error:#}").contains("quarantined"),
            "corrupt segment was not quarantined: {error:#}"
        );
        assert!(
            !store.has_segment(&digest),
            "the corrupt segment is still in the managed tree"
        );
        assert!(
            root.join("quarantine").exists(),
            "nothing was moved to quarantine"
        );
    }

    /// Not a pass/fail assertion: a stopwatch on the cost that smaller windows
    /// buy. Eviction parses every manifest to build its reference map, and a
    /// 64-row window turns a 19K-token entry into ~9.5k segment refs. Run with
    /// `cargo test -p skippy-cache --lib eviction_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check; takes tens of seconds"]
    fn eviction_cost_at_realistic_segment_counts() {
        const SEGMENTS_PER_MANIFEST: usize = 9_504; // 16 layers x 2 x ceil(19000/64)
        const MANIFESTS: usize = 20;
        let root = temp_root("eviction-cost");
        let store = store(&root, 0);

        // One physical segment shared by every ref: this measures manifest
        // parsing and reference mapping, not filesystem write throughput.
        let bytes = vec![7u8; 65_536];
        let digest = store.put_segment(&bytes).expect("put").digest;
        let build = std::time::Instant::now();
        for manifest_index in 0..MANIFESTS {
            let mut manifest = HandoffManifest::new("blake3:cost".to_string(), "full-state".into());
            for index in 0..SEGMENTS_PER_MANIFEST {
                manifest.segments.push(HandoffSegmentRef {
                    index: index as u32,
                    offset: (index * bytes.len()) as u64,
                    bytes: bytes.len() as u64,
                    digest: digest.clone(),
                    meta_json: Some(format!("k:{}:0:{}", index % 32, index / 32)),
                });
            }
            manifest.total_bytes = (SEGMENTS_PER_MANIFEST * bytes.len()) as u64;
            manifest.payload_digest = format!("blake3:manifest-{manifest_index}");
            store.commit(&manifest).expect("commit");
        }
        let build_ms = build.elapsed().as_millis();

        let manifest_bytes = directory_bytes(&root.join(MANIFEST_DIR)).expect("manifest bytes");
        let usage = store.managed_usage_bytes().expect("usage");
        let evict = std::time::Instant::now();
        store.enforce_budget_to(usage / 2).expect("enforce");
        let evict_ms = evict.elapsed().as_millis();

        println!(
            "eviction cost: {MANIFESTS} manifests x {SEGMENTS_PER_MANIFEST} refs, \
             manifest bytes {manifest_bytes} ({} KiB each), build {build_ms} ms, \
             enforce_budget {evict_ms} ms",
            manifest_bytes / MANIFESTS as u64 / 1024
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn eviction_leaves_headroom_so_a_full_cache_is_not_repriced_per_commit() {
        let root = temp_root("low-water");
        let store = store(&root, 40_000);
        for fill in 1u8..=6 {
            let manifest = commit_payload(&store, &vec![fill; 8_000], 4096);
            // A commit that overflows evicts; one that fits must not.
            store.commit(&manifest).expect("commit");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let usage = store.managed_usage_bytes().expect("usage");
        assert!(
            usage <= 40_000 * EVICTION_LOW_WATER_PERCENT / 100,
            "eviction stopped at the cap instead of below it: {usage}"
        );
        assert_eq!(
            store.enforce_budget().expect("second pass"),
            0,
            "a store already under the cap must not pay for another pass"
        );
    }

    #[test]
    fn unreferenced_segments_are_collected() {
        let root = temp_root("gc");
        let store = store(&root, 0);
        store.put_segment(b"orphan bytes").expect("orphan put");
        let payload = vec![9u8; 4096];
        let manifest = commit_payload(&store, &payload, 4096);

        let freed = store.collect_unreferenced_segments().expect("collect");
        assert_eq!(freed, 12);
        assert!(store.assemble(&manifest).is_ok());
    }
}
