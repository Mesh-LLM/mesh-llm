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
//!   concurrent writers of the same bytes converge on one object via
//!   temp-file + atomic rename.
//! - **Capped budget**: `enforce_budget` evicts oldest manifests first (the
//!   newest is never evicted) and garbage-collects unreferenced segments.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::fsinfo;

use footprint::ManifestFootprint;
pub use types::*;
use validation::{decode_manifest, validate_manifest_compatibility, verify_segments_at_offsets};

mod footprint;
mod geometry;
mod identity;
mod manifest_codec;
mod packed;
mod types;
mod validation;
use packed::{PACK_DIR, PACK_INDEX_DIR, PackedReadRequest, PackedSegmentStore};

const SEGMENT_DIR: &str = "segments";
const MANIFEST_DIR: &str = "manifests";
const PREFIX_INDEX_DIR: &str = "prefixes";
const QUARANTINE_DIR: &str = "quarantine";
const ROOT_LOCK_FILE: &str = ".owner.lock";

/// Evict to this percentage of the budget rather than exactly to it.
///
/// An eviction pass is O(manifests x segments): it parses every manifest to
/// learn which segments would become unreferenced. Measured at 812 ms for 20
/// manifests of ~9.5k segments, which is what a 19K-token prefix costs at a
/// 64-row window. Evicting exactly to the cap means a full cache pays that on
/// every commit; leaving headroom amortises it over the writes that fill the
/// headroom back up.
const EVICTION_LOW_WATER_PERCENT: u64 = 85;
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
        let mut holds = self
            .store
            .inflight_segments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(count) = holds.get_mut(&self.digest) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                holds.remove(&self.digest);
            }
        }
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

/// Repairs performed before a node starts serving a cache root.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct StoreReconciliation {
    pub removed_temporary_files: u64,
    pub quarantined_manifests: u64,
    pub removed_prefix_links: u64,
    pub removed_orphan_bytes: u64,
}

#[derive(Debug)]
pub struct HandoffSegmentStore {
    root: PathBuf,
    limits: RwLock<StoreLimits>,
    /// Exclusive process-level ownership of this cache root. The manager
    /// shares one store between all stages in the node; a second process must
    /// not build an independent reservation/pin universe over the same files.
    _root_lock: fs::File,
    /// Makes the admission check and reservation increment one transaction.
    /// Without this, two stages can both observe the same free budget before
    /// either publishes its reservation.
    admission: Mutex<()>,
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
    /// Last computed on-disk managed total, excluding in-flight reservations.
    ///
    /// `reserve` runs per segment put, and a full scan stats every file under
    /// the root, so recomputing it each time makes one spill of a long prefix
    /// O(segments^2) syscalls under the admission lock. The hot path keeps
    /// this total incrementally; every bulk mutation invalidates it and the
    /// next read pays for one authoritative scan.
    usage_bytes: AtomicU64,
    /// Whether `usage_bytes` can be trusted without rescanning.
    usage_valid: AtomicBool,
    /// Segments written but not yet referenced by a committed manifest.
    ///
    /// A writer puts every segment before committing the manifest that binds
    /// them, so for that window the bytes are unreferenced — and eviction
    /// triggered by another writer (or by this one needing room) would
    /// collect them out from under the commit that is about to reference
    /// them. Left unprotected this fails as "manifest references missing
    /// segment" under exactly the pressure the cache is for.
    inflight_segments: Mutex<std::collections::BTreeMap<String, usize>>,
    /// Per-manifest object footprint, recorded at commit.
    ///
    /// Budget enforcement needs to know which pack each segment lives in to
    /// learn what becomes free when a manifest is evicted. Rebuilding that
    /// from every manifest and packed index costs O(manifests x segments) per
    /// pass; recording it once at commit makes the same decision O(1) per
    /// evicted manifest.
    footprints: Mutex<std::collections::HashMap<String, ManifestFootprint>>,
    packed: PackedSegmentStore,
}

pub fn segment_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Encode a manifest in the on-disk container. Public so the peer protocol can
/// ship the same bytes it stores.
pub fn encode_manifest_container(manifest: &HandoffManifest) -> Result<Vec<u8>> {
    manifest_codec::encode_manifest(manifest)
}

/// Decode a manifest container produced by [`encode_manifest_container`].
pub fn decode_manifest_container(bytes: &[u8]) -> Result<HandoffManifest> {
    manifest_codec::decode_manifest(bytes)
}

/// Whether these bytes are a manifest container rather than legacy JSON.
pub fn is_manifest_container(bytes: &[u8]) -> bool {
    manifest_codec::looks_binary(bytes)
}

pub use geometry::{GeometryBlock, GeometryKind, PayloadGeometry};
pub use identity::{composition_root, fixed_segmentation_id, geometry_segmentation_id};

fn acquire_root_lock(root: &Path) -> Result<fs::File> {
    let path = root.join(ROOT_LOCK_FILE);
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("failed to open cache root lock {}", path.display()))?;
    fsinfo::restrict_to_owner(&path, 0o600)?;

    if let Err(error) = fs2::FileExt::try_lock_exclusive(&file) {
        if fsinfo::is_lock_contended(&error) {
            bail!(
                "cache root {} is already owned by another manager",
                root.display()
            );
        }
        return Err(error).with_context(|| format!("failed to lock cache root {}", root.display()));
    }

    Ok(file)
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
        let store = Self::open_unreconciled_with_limits(root, limits)?;
        store.reconcile_startup()?;
        Ok(store)
    }

    pub(crate) fn open_unreconciled_with_limits(
        root: impl Into<PathBuf>,
        limits: StoreLimits,
    ) -> Result<Self> {
        let root = root.into();
        if !root.is_absolute() {
            bail!("cache root must be absolute: {}", root.display());
        }
        fsinfo::create_dir_all_without_links(&root)?;
        // Anchor everything below to the resolved root, so a symlink crossed on
        // the way in cannot make containment checks disagree with where bytes
        // actually land.
        let root = fs::canonicalize(&root)
            .with_context(|| format!("failed to resolve cache root {}", root.display()))?;
        if fsinfo::is_network_filesystem(&root).unwrap_or(false) {
            let name = fsinfo::filesystem_type_name(&root).unwrap_or_default();
            bail!(
                "{} is on an unsupported network filesystem ({name}); the disk cache requires local storage",
                root.display()
            );
        }
        for directory in [
            SEGMENT_DIR,
            MANIFEST_DIR,
            PREFIX_INDEX_DIR,
            PACK_DIR,
            PACK_INDEX_DIR,
        ] {
            let path = root.join(directory);
            fsinfo::refuse_symlinked_descendant(&root, &path)?;
            fs::create_dir_all(&path).with_context(|| {
                format!("failed to create {directory} dir under {}", root.display())
            })?;
            fsinfo::restrict_to_owner(&path, 0o700)?;
        }
        fsinfo::restrict_to_owner(&root, 0o700)?;
        let root_lock = acquire_root_lock(&root)?;
        let packed = PackedSegmentStore::open(&root)?;
        Ok(Self {
            root,
            limits: RwLock::new(limits),
            _root_lock: root_lock,
            admission: Mutex::new(()),
            reserved_inflight: AtomicU64::new(0),
            // Left invalid so the first read pays for one authoritative scan
            // of whatever the previous process left behind.
            usage_bytes: AtomicU64::new(0),
            usage_valid: AtomicBool::new(false),
            pins: Mutex::new(std::collections::BTreeMap::new()),
            evicted_manifests: AtomicU64::new(0),
            quarantined_objects: AtomicU64::new(0),
            inflight_segments: Mutex::new(std::collections::BTreeMap::new()),
            footprints: Mutex::new(std::collections::HashMap::new()),
            packed,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn limits(&self) -> StoreLimits {
        *self
            .limits
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Replace the live hard cap and filesystem reserve. Admission is paused
    /// while the pair changes, so a writer can never observe half of the new
    /// policy. Shrinking evicts inactive entries immediately; pinned entries
    /// remain valid and subsequent writes stay refused until usage fits.
    pub fn update_limits(&self, limits: StoreLimits) -> Result<StoreLimits> {
        let _admission = self
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut current = self
            .limits
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = *current;
        *current = limits;
        drop(current);
        if limits.budget_bytes > 0 {
            self.enforce_budget_to(limits.budget_bytes)?;
        }
        Ok(previous)
    }

    /// Repair incomplete or invalid state left by an interrupted writer before
    /// the manager exposes this root to any stage.
    pub fn reconcile_startup(&self) -> Result<StoreReconciliation> {
        let mut report = StoreReconciliation::default();
        for directory in [
            SEGMENT_DIR,
            MANIFEST_DIR,
            PREFIX_INDEX_DIR,
            PACK_DIR,
            PACK_INDEX_DIR,
        ] {
            report.removed_temporary_files = report
                .removed_temporary_files
                .saturating_add(remove_temporary_files(&self.root.join(directory))?);
        }

        let mut quarantined = Vec::new();
        for key in self.list_manifests()? {
            if self.validate_committed_manifest(&key).is_err() {
                self.quarantine(&self.existing_manifest_path(&key))?;
                quarantined.push(key);
                report.quarantined_manifests += 1;
            }
        }
        self.forget_footprints(&quarantined);
        self.rebuild_packed_index()?;
        report.removed_prefix_links = self.remove_dangling_prefix_links()?;
        report.removed_orphan_bytes = self.collect_unreferenced_segments()?;
        Ok(report)
    }

    fn validate_committed_manifest(&self, key: &str) -> Result<()> {
        let manifest = self.load_manifest(key)?;
        if manifest.payload_digest != key {
            bail!(
                "manifest key {key} disagrees with payload digest {}",
                manifest.payload_digest
            );
        }
        let mut expected_offset = 0u64;
        let locations = self
            .packed
            .load_manifest_index(key, manifest.segments.len())?;
        for (position, (segment, location)) in manifest.segments.iter().zip(locations).enumerate() {
            if segment.index as usize != position || segment.offset != expected_offset {
                bail!("manifest {key} has invalid segment ordering");
            }
            self.validate_segment_ref(segment, location.as_ref())
                .with_context(|| format!("manifest {key} references a missing segment"))?;
            expected_offset = expected_offset
                .checked_add(segment.bytes)
                .context("manifest segment offsets overflow")?;
        }
        if expected_offset != manifest.total_bytes {
            bail!("manifest {key} does not tile its payload");
        }
        Ok(())
    }

    fn remove_dangling_prefix_links(&self) -> Result<u64> {
        // Files are about to be removed or rewritten in bulk.
        self.invalidate_usage();
        let root = self.root.join(PREFIX_INDEX_DIR);
        let mut removed = 0u64;
        for path in files_recursive(&root)? {
            let digest = match fs::read_to_string(&path) {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            if digest.is_empty() || !self.existing_manifest_path(&digest).is_file() {
                match fs::remove_file(&path) {
                    Ok(()) => removed += 1,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
        Ok(removed)
    }

    fn segment_path(&self, digest: &str) -> PathBuf {
        self.root.join(SEGMENT_DIR).join(format!("{digest}.seg"))
    }

    fn validate_segment_ref(
        &self,
        segment: &HandoffSegmentRef,
        location: Option<&packed::PackedSegmentLocation>,
    ) -> Result<()> {
        if let Some(location) = location {
            return self.packed.validate(location, segment.bytes);
        }
        let metadata = fs::metadata(self.segment_path(&segment.digest))?;
        if metadata.len() != segment.bytes {
            bail!("segment {} is truncated", segment.digest);
        }
        Ok(())
    }

    fn rebuild_packed_index(&self) -> Result<()> {
        let mut entries = Vec::new();
        for key in self.list_manifests()? {
            let Ok(manifest) = self.load_manifest(&key) else {
                continue;
            };
            let locations = self
                .packed
                .load_manifest_index(&key, manifest.segments.len())?;
            entries.extend(manifest.segments.into_iter().zip(locations).filter_map(
                |(segment, location)| location.map(|location| (segment.digest, location)),
            ));
        }
        self.packed.rebuild(entries);
        Ok(())
    }

    fn manifest_path(&self, payload_digest: &str) -> PathBuf {
        self.root
            .join(MANIFEST_DIR)
            .join(format!("{payload_digest}.manifest"))
    }

    /// Where this key is actually stored: the current container, or the legacy
    /// JSON file when only that exists. Writers always use [`Self::manifest_path`].
    fn existing_manifest_path(&self, payload_digest: &str) -> PathBuf {
        let current = self.manifest_path(payload_digest);
        if current.is_file() {
            return current;
        }
        let legacy = self
            .root
            .join(MANIFEST_DIR)
            .join(format!("{payload_digest}.json"));
        if legacy.is_file() { legacy } else { current }
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
        let bytes = payload_digest.as_bytes();
        let replaced_bytes =
            fs::metadata(&path)
                .map(|metadata| metadata.len())
                .or_else(|error| {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        Ok(0)
                    } else {
                        Err(error)
                    }
                })?;
        let growth_bytes = (bytes.len() as u64).saturating_sub(replaced_bytes);
        let reservation = match self.reserve_write(bytes.len() as u64, growth_bytes)? {
            Ok(reservation) => reservation,
            Err(refusal) => bail!("cannot store prefix link: {}", refusal.reason()),
        };
        let parent = path.parent().context("prefix entry has no parent")?;
        fsinfo::create_dir_all_without_links(parent)?;
        fsinfo::restrict_to_owner(parent, 0o700)?;
        // Replacement can shrink or grow the entry, so rescan rather than
        // trying to update the cached total from a racy pre-write stat.
        self.invalidate_usage();
        write_atomically(&path, bytes)?;
        fsinfo::restrict_to_owner(&path, 0o600)?;
        drop(reservation);
        self.enforce_budget()?;
        Ok(())
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
        // Legacy entries live at `<digest>.json`; the resolver keeps them
        // readable instead of turning a live prefix into a permanent miss.
        let manifest_path = self.existing_manifest_path(&payload_digest);
        match fs::read(&manifest_path) {
            Ok(bytes) => {
                let manifest = match decode_manifest(&payload_digest, &bytes) {
                    Ok(manifest) => manifest,
                    Err(_) => {
                        self.quarantine(&manifest_path)?;
                        self.forget_footprints(std::slice::from_ref(&payload_digest));
                        let _ = fs::remove_file(&path);
                        self.invalidate_usage();
                        return Ok(None);
                    }
                };
                // A hit is a use. Recording it here is what makes eviction
                // least-recently-*used* rather than least-recently-written.
                self.touch_manifest(&payload_digest);
                Ok(Some(manifest))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // The manifest was evicted after the link was written; only
                // this definite absence prunes the link.
                let _ = fs::remove_file(&path);
                self.invalidate_usage();
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
        // Establish the GC hold before observing or publishing the file. This
        // closes the rename-to-hold window where a concurrent clear could see
        // an unreferenced segment and remove it before its manifest commits.
        let hold = self.hold_segment(&digest);
        let path = self.segment_path(&digest);
        if path.exists() {
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
        if let Err(error) =
            write_atomically(&path, bytes).and_then(|()| fsinfo::restrict_to_owner(&path, 0o600))
        {
            // A failed cleanup or a permissions failure may leave physical
            // bytes behind. Force the next admission to reconcile with disk.
            self.invalidate_usage();
            return Err(error);
        }
        self.add_usage_bytes(bytes.len() as u64);
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

    /// Store one spill's logical segments in one immutable physical pack.
    pub fn try_put_segments<'store>(
        &'store self,
        segments: &[&[u8]],
    ) -> Result<Result<Vec<StoredSegment<'store>>, WriteRefusal>> {
        let digests = segments
            .iter()
            .map(|bytes| segment_digest(bytes))
            .collect::<Vec<_>>();
        let holds = digests
            .iter()
            .map(|digest| self.hold_segment(digest))
            .collect::<Vec<_>>();
        let inputs = digests
            .iter()
            .zip(segments)
            .map(|(digest, bytes)| (digest.as_str(), *bytes))
            .collect::<Vec<_>>();
        let estimated = self.packed.estimated_new_bytes(&inputs);
        let reservation = match self.reserve(estimated)? {
            Ok(reservation) => reservation,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let (packed, new_bytes) = match self.packed.write_batch(&inputs) {
            Ok(result) => result,
            Err(error) => {
                self.invalidate_usage();
                return Err(error);
            }
        };
        self.add_usage_bytes(new_bytes);
        drop(reservation);
        Ok(Ok(digests
            .into_iter()
            .zip(holds)
            .zip(packed)
            .zip(segments)
            .map(|(((digest, hold), packed), bytes)| StoredSegment {
                digest,
                put: SegmentPut {
                    new: packed.new,
                    bytes: bytes.len() as u64,
                },
                _hold: hold,
            })
            .collect()))
    }

    pub fn has_segment(&self, digest: &str) -> bool {
        self.segment_path(digest).exists() || self.packed.location(digest).is_some()
    }

    /// Read one segment, verifying its content digest.
    pub fn read_segment(&self, digest: &str) -> Result<Vec<u8>> {
        if let Some(location) = self.packed.location(digest) {
            let requests = [PackedReadRequest {
                digest,
                bytes: location.bytes,
                location: &location,
                output_offset: 0,
            }];
            let len = usize::try_from(location.bytes).context("segment exceeds usize")?;
            let mut bytes = Vec::with_capacity(len);
            let mut accounting = packed::ReadAccounting::default();
            return match self
                .packed
                .append_many(&requests, &mut bytes, None, &mut accounting)
            {
                Ok(()) => Ok(bytes),
                Err(failure) => {
                    let path = self.packed.pack_path(&failure.pack_digest);
                    let _ = self.quarantine(&path);
                    Err(failure.error)
                }
            };
        }
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
        // Files are about to be removed or rewritten in bulk.
        self.invalidate_usage();
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
        match self.try_commit(manifest)? {
            Ok(()) => Ok(()),
            Err(refusal) => bail!(
                "cannot commit manifest {}: {}",
                manifest.payload_digest,
                refusal.reason()
            ),
        }
    }

    /// As [`Self::commit`], preserving an admission refusal as structured
    /// state for the manager's effective-state contract.
    pub fn try_commit(&self, manifest: &HandoffManifest) -> Result<Result<(), WriteRefusal>> {
        // The manifest and its prefix index are new files on disk.
        self.invalidate_usage();
        if manifest.payload_digest.is_empty() {
            bail!("manifest has no payload digest");
        }
        validate_manifest_compatibility(manifest)?;
        let mut expected_offset = 0u64;
        let locations = manifest
            .segments
            .iter()
            .map(|segment| self.packed.location(&segment.digest))
            .collect::<Vec<_>>();
        for (position, (segment, location)) in manifest.segments.iter().zip(&locations).enumerate()
        {
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
            self.validate_segment_ref(segment, location.as_ref())
                .with_context(|| {
                    format!(
                        "manifest references missing segment {} ({})",
                        segment.index, segment.digest
                    )
                })?;
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
        if manifest.key_kind == KeyKind::Alias {
            bail!("an alias key cannot be committed as an L3 manifest");
        }
        if manifest.key_kind == KeyKind::CompositionRoot {
            let segmentation_id = manifest
                .segmentation_id
                .as_deref()
                .context("composition-root manifest has no segmentation identity")?;
            let root = composition_root(segmentation_id, &manifest.segments)?;
            if root != manifest.payload_digest {
                bail!(
                    "manifest key {} is not the composition root of its segments ({root})",
                    manifest.payload_digest
                );
            }
        }
        let limits = self.limits();
        if limits.budget_bytes > 0 && manifest.total_bytes > limits.budget_bytes {
            return Ok(Err(WriteRefusal::SkippedOversize));
        }
        // Compact, not pretty: a 64-row window turns a long prefix into
        // thousands of segment refs, and eviction parses every manifest to
        // build its reference map. Indentation is pure cost on a file nobody
        // reads by hand.
        let serialized =
            manifest_codec::encode_manifest(manifest).context("failed to encode manifest")?;
        let segment_digests = manifest
            .segments
            .iter()
            .map(|segment| segment.digest.clone())
            .collect::<Vec<_>>();
        let packed_index = self
            .packed
            .encode_manifest_index(&manifest.payload_digest, &segment_digests)?;
        // The manifest is what makes the segments loadable, so it is pinned
        // while it lands: eviction triggered by its own admission check must
        // not remove the entry being committed.
        let _pin = self.pin(&manifest.payload_digest);
        let manifest_path = self.manifest_path(&manifest.payload_digest);
        let replaced_bytes = fs::metadata(&manifest_path)
            .map(|metadata| metadata.len())
            .or_else(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Ok(0)
                } else {
                    Err(error)
                }
            })?;
        let growth_bytes = (serialized.len() as u64).saturating_sub(replaced_bytes);
        let packed_index_path = self.packed.index_path(&manifest.payload_digest);
        let replaced_index_bytes = fs::metadata(&packed_index_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let index_bytes = packed_index.as_ref().map_or(0, |bytes| bytes.len() as u64);
        let write_bytes = (serialized.len() as u64).saturating_add(index_bytes);
        let total_growth =
            growth_bytes.saturating_add(index_bytes.saturating_sub(replaced_index_bytes));
        match self.reserve_write(write_bytes, total_growth)? {
            Ok(_reservation) => {
                if let Some(index) = &packed_index {
                    self.packed
                        .publish_manifest_index(&manifest.payload_digest, index)?;
                } else {
                    self.packed
                        .remove_manifest_index(&manifest.payload_digest)?;
                }
                write_atomically(&manifest_path, &serialized)?;
                fsinfo::restrict_to_owner(&manifest_path, 0o600)?;
                self.record_footprint(manifest, &locations, serialized.len() as u64);
            }
            Err(refusal) => return Ok(Err(refusal)),
        }
        self.enforce_budget()?;
        Ok(Ok(()))
    }

    /// Record that an entry was used, so eviction can order by last use.
    /// Best-effort: a store on a read-only path still serves reads, and
    /// failing a restore because recency could not be recorded would trade a
    /// working cache for a bookkeeping detail.
    pub fn touch_manifest(&self, payload_digest: &str) {
        let _ = fsinfo::touch(&self.existing_manifest_path(payload_digest));
    }

    pub fn load_manifest(&self, payload_digest: &str) -> Result<HandoffManifest> {
        let bytes = fs::read(self.existing_manifest_path(payload_digest))
            .with_context(|| format!("failed to read manifest {payload_digest}"))?;
        decode_manifest(payload_digest, &bytes)
    }

    /// Newest manifest key, optionally restricted to wire-digest entries.
    ///
    /// A peer that asked for JSON can only verify a wire-digest key: handing
    /// it a composition-root entry would transcode a key its own admission
    /// re-hash rejects. The filter is on `key_kind`, not the file extension,
    /// because a legacy-shaped file is not the property that matters.
    pub fn newest_manifest(&self, wire_key_only: bool) -> Result<Option<String>> {
        for key in self.list_manifests()? {
            if !wire_key_only {
                return Ok(Some(key));
            }
            if self
                .load_manifest(&key)
                .is_ok_and(|manifest| manifest.key_kind == KeyKind::WireBlake3)
            {
                return Ok(Some(key));
            }
        }
        Ok(None)
    }

    /// Manifest keys, newest first by modification time.
    pub fn list_manifests(&self) -> Result<Vec<String>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(self.root.join(MANIFEST_DIR))? {
            let entry = entry?;
            let path = entry.path();
            // `.manifest` is the current container; `.json` entries are legacy
            // v2-v4 manifests kept readable for one release.
            if path
                .extension()
                .is_none_or(|extension| extension != "manifest" && extension != "json")
            {
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
        self.assemble_traced(manifest).map(|(payload, _)| payload)
    }

    /// As [`Self::assemble`], returning what the restore cost.
    ///
    /// Restore latency tracks physical reads and packs touched, not bytes, so
    /// these counters are the mechanism behind a TTFT number: a payload
    /// assembled from many small spill packs shows up here as a high
    /// `physical_reads` with a low `max_read_nanos`.
    pub fn assemble_traced(&self, manifest: &HandoffManifest) -> Result<(Vec<u8>, AssembleStats)> {
        let started = std::time::Instant::now();
        validate_manifest_compatibility(manifest)?;
        let total = usize::try_from(manifest.total_bytes).context("payload exceeds usize")?;
        let mut payload = Vec::with_capacity(total);
        // A composition-root manifest is verified from segment metadata, not
        // by re-hashing the payload. Wire-digest manifests keep the old path.
        let mut payload_hasher =
            (manifest.key_kind == KeyKind::WireBlake3).then(blake3::Hasher::new);
        let locations = self
            .packed
            .load_manifest_index(&manifest.payload_digest, manifest.segments.len())?;
        let mut expected_offset = 0u64;
        for segment in &manifest.segments {
            if segment.offset != expected_offset {
                bail!(
                    "segment {} offset {} does not match assembled length {expected_offset}",
                    segment.index,
                    segment.offset,
                );
            }
            expected_offset = expected_offset
                .checked_add(segment.bytes)
                .context("assembled payload size overflows")?;
        }
        if expected_offset != manifest.total_bytes {
            bail!(
                "assembled {expected_offset} bytes but manifest records {}",
                manifest.total_bytes
            );
        }
        let mut stats = AssembleStats {
            segments: manifest.segments.len() as u64,
            bytes: manifest.total_bytes,
            ..AssembleStats::default()
        };
        let mut packs = std::collections::HashSet::new();
        for location in locations.iter().flatten() {
            packs.insert(location.pack_digest.as_str());
        }
        stats.packs = packs.len() as u64;
        let mut packed_requests = Vec::new();
        for (segment, location) in manifest.segments.iter().zip(&locations) {
            if let Some(location) = location.as_ref() {
                packed_requests.push(PackedReadRequest {
                    digest: &segment.digest,
                    bytes: segment.bytes,
                    location,
                    output_offset: segment.offset,
                });
            } else {
                self.append_packed(
                    &packed_requests,
                    &mut payload,
                    &mut payload_hasher.as_mut(),
                    &mut stats,
                )?;
                packed_requests.clear();
                let bytes = self.read_segment(&segment.digest)?;
                if let Some(hasher) = payload_hasher.as_mut() {
                    hasher.update(&bytes);
                }
                payload.extend_from_slice(&bytes);
                stats.loose_segment_reads = stats.loose_segment_reads.saturating_add(1);
            }
        }
        self.append_packed(
            &packed_requests,
            &mut payload,
            &mut payload_hasher.as_mut(),
            &mut stats,
        )?;
        if payload.len() != total {
            bail!(
                "assembled {} bytes but manifest records {total}",
                payload.len()
            );
        }
        match payload_hasher {
            Some(hasher) => {
                if hasher.finalize().to_hex().as_str() != manifest.payload_digest {
                    bail!("assembled payload failed manifest digest verification");
                }
            }
            None if manifest.key_kind == KeyKind::Alias => {
                bail!("an alias manifest cannot be assembled from this store");
            }
            None => {
                let segmentation_id = manifest
                    .segmentation_id
                    .as_deref()
                    .context("composition-root manifest has no segmentation identity")?;
                let root = composition_root(segmentation_id, &manifest.segments)?;
                if root != manifest.payload_digest {
                    bail!("assembled payload failed composition-root verification");
                }
            }
        }
        if std::env::var("SKIPPY_L3_VERIFY_WIRE").is_ok_and(|value| value != "0" && value != "off")
        {
            verify_segments_at_offsets(&payload, &manifest.segments)?;
        }
        stats.total_nanos = started.elapsed().as_nanos() as u64;
        Ok((payload, stats))
    }

    fn append_packed(
        &self,
        requests: &[PackedReadRequest<'_>],
        payload: &mut Vec<u8>,
        payload_hasher: &mut Option<&mut blake3::Hasher>,
        stats: &mut AssembleStats,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let mut accounting = packed::ReadAccounting::default();
        let result = self.packed.append_many(
            requests,
            payload,
            payload_hasher.as_deref_mut(),
            &mut accounting,
        );
        stats.packs_opened = stats.packs_opened.saturating_add(accounting.packs_opened);
        stats.physical_reads = stats
            .physical_reads
            .saturating_add(accounting.physical_reads);
        stats.bytes_read = stats.bytes_read.saturating_add(accounting.bytes_read);
        stats.read_nanos = stats.read_nanos.saturating_add(accounting.read_nanos);
        stats.max_read_nanos = stats.max_read_nanos.max(accounting.max_read_nanos);
        stats.verify_nanos = stats.verify_nanos.saturating_add(accounting.verify_nanos);
        stats.hash_nanos = stats.hash_nanos.saturating_add(accounting.hash_nanos);
        stats.hashed_bytes = stats.hashed_bytes.saturating_add(accounting.hashed_bytes);
        match result {
            Ok(()) => Ok(()),
            Err(failure) => {
                let path = self.packed.pack_path(&failure.pack_digest);
                let _ = self.quarantine(&path);
                Err(failure.error)
            }
        }
    }

    pub fn segment_footprint_bytes(&self) -> Result<u64> {
        Ok(directory_bytes(&self.root.join(SEGMENT_DIR))?
            .saturating_add(self.packed.footprint_bytes()?))
    }

    /// Every byte the store manages: committed segments, the manifests that
    /// make them loadable, the prefix index that finds them, and capacity
    /// reserved by writes in flight.
    ///
    /// The budget is a physical cap on the cache root, so it has to count the
    /// bookkeeping too. Counting segments alone lets a store with many small
    /// entries sit well over its stated cap in manifests and index files.
    pub fn managed_usage_bytes(&self) -> Result<u64> {
        let on_disk = if self.usage_valid.load(Ordering::Acquire) {
            self.usage_bytes.load(Ordering::Acquire)
        } else {
            self.rescan_usage_bytes()?
        };
        Ok(on_disk.saturating_add(self.reserved_inflight.load(Ordering::Acquire)))
    }

    /// Stat every managed file and adopt the result as the running total.
    fn rescan_usage_bytes(&self) -> Result<u64> {
        let mut total = directory_bytes(&self.root.join(SEGMENT_DIR))?;
        total = total.saturating_add(self.packed.footprint_bytes()?);
        total = total.saturating_add(self.packed.index_footprint_bytes()?);
        total = total.saturating_add(directory_bytes(&self.root.join(MANIFEST_DIR))?);
        total = total.saturating_add(directory_bytes_recursive(
            &self.root.join(PREFIX_INDEX_DIR),
        )?);
        total = total.saturating_add(directory_bytes_recursive(&self.root.join(QUARANTINE_DIR))?);
        self.usage_bytes.store(total, Ordering::Release);
        self.usage_valid.store(true, Ordering::Release);
        Ok(total)
    }

    /// Drop the cached total. Every mutation that is not a plain segment put
    /// goes through here, so the next read rescans rather than trusting a
    /// figure that a bulk removal or rewrite may have invalidated.
    fn invalidate_usage(&self) {
        self.usage_valid.store(false, Ordering::Release);
    }

    /// Account for bytes a segment put just added, keeping the hot path free
    /// of a rescan. A no-op while the total is already known to be stale.
    fn add_usage_bytes(&self, bytes: u64) {
        if self.usage_valid.load(Ordering::Acquire) {
            self.usage_bytes.fetch_add(bytes, Ordering::AcqRel);
        }
    }

    /// Reserve capacity for `bytes` about to be written.
    ///
    /// Reservation happens before any temporary payload bytes exist on disk,
    /// so the cap and the free-space reserve hold even while several writers
    /// are mid-write. The returned guard releases the reservation on drop,
    /// including on the error paths.
    pub fn reserve(&self, bytes: u64) -> Result<Result<Reservation<'_>, WriteRefusal>> {
        self.reserve_write(bytes, bytes)
    }

    /// Reserve free space for the temporary write and budget capacity only
    /// for its projected net growth. Replacements need room for a complete
    /// temporary file, but charging their full size against the cache cap
    /// would evict healthy entries even when the final file has the same size.
    fn reserve_write(
        &self,
        write_bytes: u64,
        growth_bytes: u64,
    ) -> Result<Result<Reservation<'_>, WriteRefusal>> {
        let _admission = self
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let limits = self.limits();
        if limits.budget_bytes > 0 && growth_bytes > limits.budget_bytes {
            return Ok(Err(WriteRefusal::SkippedOversize));
        }
        let available = fsinfo::available_bytes(&self.root)?;
        if available.saturating_sub(write_bytes) < limits.minimum_free_bytes {
            return Ok(Err(WriteRefusal::ReadOnlyLowSpace));
        }
        if limits.budget_bytes > 0 {
            let used = self.managed_usage_bytes()?;
            if used.saturating_add(growth_bytes) > limits.budget_bytes {
                // Make room from inactive entries before refusing: a full
                // cache is the normal steady state, not an error. Clear to the
                // low-water mark, or further when this write alone needs more.
                let target = self
                    .low_water_bytes()
                    .min(limits.budget_bytes.saturating_sub(growth_bytes));
                self.enforce_budget_to(target)?;
                // Never refuse on a cached figure: rescan so an incremental
                // total that has drifted low cannot turn a writable cache into
                // a reported InsufficientSpace.
                let used = self
                    .rescan_usage_bytes()?
                    .saturating_add(self.reserved_inflight.load(Ordering::Acquire));
                if used.saturating_add(growth_bytes) > limits.budget_bytes {
                    return Ok(Err(WriteRefusal::InsufficientSpace));
                }
            }
        }
        self.reserved_inflight
            .fetch_add(growth_bytes, Ordering::AcqRel);
        Ok(Ok(Reservation {
            store: self,
            bytes: growth_bytes,
        }))
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

    pub(crate) fn is_pinned(&self, payload_digest: &str) -> bool {
        self.pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(payload_digest)
            .is_some_and(|count| *count > 0)
    }

    /// Remove the requested inactive manifests and collect physical objects
    /// that become unreferenced. Missing and pinned keys are skipped. This is
    /// the commit side of an external eviction policy; the store's own LRU
    /// reservation remains the hard-budget fallback.
    pub(crate) fn evict_manifest_keys(&self, keys: &[String]) -> Result<Vec<String>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        self.invalidate_usage();
        let mut removed = Vec::new();
        for key in keys {
            if self.is_pinned(key) {
                continue;
            }
            let path = self.existing_manifest_path(key);
            match fs::remove_file(&path) {
                Ok(()) => {
                    self.packed.remove_manifest_index(key)?;
                    self.evicted_manifests.fetch_add(1, Ordering::Relaxed);
                    removed.push(key.clone());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to evict selected manifest {key}"));
                }
            }
        }
        if !removed.is_empty() {
            self.forget_footprints(&removed);
            self.remove_dangling_prefix_links()?;
            self.collect_unreferenced_segments()?;
        }
        Ok(removed)
    }

    /// What the store holds right now, for the status contract.
    pub fn usage(&self) -> Result<StoreUsage> {
        let limits = self.limits();
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
            budget_bytes: limits.budget_bytes,
            used_bytes: self.managed_usage_bytes()?,
            reserved_inflight_bytes: self.reserved_inflight.load(Ordering::Acquire),
            filesystem_available_bytes: fsinfo::available_bytes(&self.root)?,
            minimum_free_bytes: limits.minimum_free_bytes,
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
        let limits = self.limits();
        if limits.budget_bytes == 0 {
            return Ok(0);
        }
        // Over the cap, evict below it: the pass is expensive enough that
        // paying it once per headroom refill beats paying it per commit.
        if self.managed_usage_bytes()? <= limits.budget_bytes {
            return Ok(0);
        }
        self.enforce_budget_to(self.low_water_bytes())
    }

    /// The level eviction drops to once it runs.
    fn low_water_bytes(&self) -> u64 {
        self.limits()
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
        self.enforce_budget_to_model(target_bytes, None)
    }

    fn enforce_budget_to_model(
        &self,
        target_bytes: u64,
        model_identity: Option<&str>,
    ) -> Result<u64> {
        let usage_before = self.managed_usage_bytes()?;
        if usage_before <= target_bytes {
            return Ok(0);
        }
        // Least-recently-used last, so eviction pops from the back.
        let keys = self.list_manifests()?;
        let mut entries = Vec::with_capacity(keys.len());
        let mut reference_counts: std::collections::HashMap<String, (usize, u64)> =
            std::collections::HashMap::new();
        for key in &keys {
            let footprint = self.manifest_footprint(key)?;
            for (pack, (count, bytes)) in &footprint.packs {
                let entry = reference_counts
                    .entry(format!("pack:{pack}"))
                    .or_insert((0, *bytes));
                entry.0 += count;
            }
            for (digest, (count, bytes)) in &footprint.loose {
                let entry = reference_counts
                    .entry(digest.clone())
                    .or_insert((0, *bytes));
                entry.0 += count;
            }
            entries.push((key.clone(), footprint));
        }
        let mut freeable = usage_before;
        let mut evicted_keys = Vec::new();
        while freeable > target_bytes {
            let Some(position) = entries.iter().rposition(|(key, footprint)| {
                !self.is_pinned(key)
                    && model_identity.is_none_or(|identity| footprint.model_identity == identity)
            }) else {
                // Everything left is in use. Refusing the write is correct;
                // tearing state out from under a live operation is not.
                break;
            };
            let (key, footprint) = entries.remove(position);
            self.invalidate_usage();
            fs::remove_file(self.existing_manifest_path(&key))
                .with_context(|| format!("failed to evict manifest {key}"))?;
            let index_bytes = self.packed.remove_manifest_index(&key)?;
            self.evicted_manifests.fetch_add(1, Ordering::Relaxed);
            freeable = freeable
                .saturating_sub(footprint.manifest_bytes)
                .saturating_sub(index_bytes);
            for (pack, (count, bytes)) in &footprint.packs {
                let object = format!("pack:{pack}");
                if let Some(entry) = reference_counts.get_mut(&object) {
                    entry.0 = entry.0.saturating_sub(*count);
                    if entry.0 == 0 {
                        freeable = freeable.saturating_sub(*bytes);
                    }
                }
            }
            for (digest, (count, bytes)) in &footprint.loose {
                if let Some(entry) = reference_counts.get_mut(digest) {
                    entry.0 = entry.0.saturating_sub(*count);
                    if entry.0 == 0 {
                        freeable = freeable.saturating_sub(*bytes);
                    }
                }
            }
            evicted_keys.push(key);
        }
        self.forget_footprints(&evicted_keys);
        if evicted_keys.is_empty() {
            return Ok(0);
        }
        self.remove_dangling_prefix_links()?;
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

    pub(crate) fn prune_model_to(&self, model_identity: &str, target_bytes: u64) -> Result<u64> {
        self.enforce_budget_to_model(target_bytes, Some(model_identity))
    }

    /// Remove every manifest that is not pinned, then collect the segments
    /// that became unreferenced. Returns bytes freed.
    ///
    /// Requests in flight keep serving: clearing removes stored state, and a
    /// miss falls back to cold prefill.
    pub fn clear(&self) -> Result<u64> {
        self.clear_model_inner(None)
    }

    pub(crate) fn clear_model(&self, model_identity: &str) -> Result<u64> {
        self.clear_model_inner(Some(model_identity))
    }

    fn clear_model_inner(&self, model_identity: Option<&str>) -> Result<u64> {
        // Files are about to be removed or rewritten in bulk.
        self.invalidate_usage();
        let before = self.managed_usage_bytes()?;
        let mut removed = Vec::new();
        for key in self.list_manifests()? {
            if self.is_pinned(&key) {
                continue;
            }
            if let Some(model_identity) = model_identity {
                let manifest = self.load_manifest(&key).with_context(|| {
                    format!("failed to inspect manifest {key} while clearing model state")
                })?;
                if manifest.model_identity != model_identity {
                    continue;
                }
            }
            // Legacy entries live at `<digest>.json`; clearing must reach them
            // too, or `clear` would report a freed store that still holds them.
            let path = self.existing_manifest_path(&key);
            match fs::remove_file(&path) {
                Ok(()) => removed.push(key.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| format!("failed to clear manifest {key}"));
                }
            }
            self.packed.remove_manifest_index(&key)?;
        }
        self.forget_footprints(&removed);
        self.remove_dangling_prefix_links()?;
        self.collect_unreferenced_segments()?;
        let after = self.managed_usage_bytes()?;
        Ok(before.saturating_sub(after))
    }

    fn hold_segment(&self, digest: &str) -> SegmentHold<'_> {
        let mut holds = self
            .inflight_segments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *holds.entry(digest.to_string()).or_insert(0) += 1;
        SegmentHold {
            store: self,
            digest: digest.to_string(),
        }
    }

    /// Remove segments referenced by no manifest. Returns bytes freed.
    pub fn collect_unreferenced_segments(&self) -> Result<u64> {
        // Files are about to be removed or rewritten in bulk.
        self.invalidate_usage();
        // Reference sets come from the footprints recorded at commit, so GC no
        // longer reparses every manifest's segment list to find them.
        let mut referenced_segments = std::collections::HashSet::new();
        let mut referenced_packs = std::collections::HashSet::new();
        let manifest_keys = self.list_manifests()?;
        for key in &manifest_keys {
            let footprint = self.manifest_footprint(key)?;
            for digest in footprint.loose.keys() {
                referenced_segments.insert(digest.clone());
            }
            for pack in footprint.packs.keys() {
                referenced_packs.insert(pack.clone());
            }
        }
        let mut freed = 0u64;
        let held = self
            .inflight_segments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        for entry in fs::read_dir(self.root.join(SEGMENT_DIR))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "seg") {
                continue;
            }
            let Some(stem) = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
            else {
                continue;
            };
            if !referenced_segments.contains(&stem) && !held.contains(&stem) {
                freed = freed.saturating_add(entry.metadata()?.len());
                fs::remove_file(&path)
                    .with_context(|| format!("failed to collect segment {stem}"))?;
            }
        }
        let manifests = manifest_keys
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        freed = freed.saturating_add(self.packed.remove_orphan_indexes(&manifests)?);
        freed = freed.saturating_add(self.packed.remove_orphan_packs(&referenced_packs, || {
            self.inflight_segments
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .keys()
                .cloned()
                .collect()
        })?);
        Ok(freed)
    }
}

fn files_recursive(directory: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_file() {
            files.push(entry.path());
        } else if file_type.is_dir() {
            files.extend(files_recursive(&entry.path())?);
        }
    }
    Ok(files)
}

fn remove_temporary_files(directory: &Path) -> Result<u64> {
    let mut removed = 0u64;
    for path in files_recursive(directory)? {
        let is_temporary = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".tmp-"));
        if !is_temporary {
            continue;
        }
        match fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to remove stale temp {}", path.display()));
            }
        }
    }
    Ok(removed)
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
    let (temp_path, mut temp_file) = tempfile_in(directory)?;
    let write_result = temp_file
        .write_all(bytes)
        .with_context(|| format!("failed to write {}", temp_path.display()))
        .and_then(|()| {
            temp_file
                .sync_all()
                .with_context(|| format!("failed to sync {}", temp_path.display()))
        });
    drop(temp_file);
    let publish_result = write_result.and_then(|()| fsinfo::replace_file(&temp_path, path));
    if let Err(error) = publish_result {
        return match fs::remove_file(&temp_path) {
            Ok(()) => Err(error),
            Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup) => Err(error.context(format!(
                "also failed to remove temporary file {}: {cleanup}",
                temp_path.display()
            ))),
        };
    }
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
mod tests;
