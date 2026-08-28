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
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const SEGMENT_DIR: &str = "segments";
const MANIFEST_DIR: &str = "manifests";
const PREFIX_INDEX_DIR: &str = "prefixes";
const MANIFEST_VERSION: u32 = 1;

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

#[derive(Debug)]
pub struct HandoffSegmentStore {
    root: PathBuf,
    budget_bytes: u64,
}

pub fn segment_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

impl HandoffSegmentStore {
    /// Open (creating if needed) a store rooted at `root`. `budget_bytes`
    /// caps the on-disk segment footprint enforced at commit; 0 disables
    /// eviction.
    pub fn open(root: impl Into<PathBuf>, budget_bytes: u64) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join(SEGMENT_DIR))
            .with_context(|| format!("failed to create segment dir under {}", root.display()))?;
        fs::create_dir_all(root.join(MANIFEST_DIR))
            .with_context(|| format!("failed to create manifest dir under {}", root.display()))?;
        fs::create_dir_all(root.join(PREFIX_INDEX_DIR))
            .with_context(|| format!("failed to create prefix index under {}", root.display()))?;
        Ok(Self { root, budget_bytes })
    }

    fn segment_path(&self, digest: &str) -> PathBuf {
        self.root.join(SEGMENT_DIR).join(format!("{digest}.seg"))
    }

    fn manifest_path(&self, payload_digest: &str) -> PathBuf {
        self.root
            .join(MANIFEST_DIR)
            .join(format!("{payload_digest}.json"))
    }

    fn prefix_path(&self, prefix_key: &str) -> PathBuf {
        // Prefix keys are `blake3:<hex>` identity hashes; strip the scheme
        // so the filename stays plain hex.
        let key = prefix_key.strip_prefix("blake3:").unwrap_or(prefix_key);
        self.root.join(PREFIX_INDEX_DIR).join(format!("{key}.key"))
    }

    /// Bind a prefix identity to a committed manifest so radix-style lookups
    /// can find state by prefix rather than payload digest.
    pub fn link_prefix(&self, prefix_key: &str, payload_digest: &str) -> Result<()> {
        write_atomically(&self.prefix_path(prefix_key), payload_digest.as_bytes())
    }

    /// The manifest a prefix identity points at, pruning dangling links to
    /// evicted manifests.
    pub fn manifest_for_prefix(&self, prefix_key: &str) -> Result<Option<HandoffManifest>> {
        let path = self.prefix_path(prefix_key);
        let Ok(bytes) = fs::read(&path) else {
            return Ok(None);
        };
        let payload_digest = String::from_utf8(bytes).context("malformed prefix link")?;
        match self.load_manifest(&payload_digest) {
            Ok(manifest) => Ok(Some(manifest)),
            Err(_) => {
                // The manifest was evicted after the link was written.
                let _ = fs::remove_file(&path);
                Ok(None)
            }
        }
    }

    /// Content-addressed, idempotent put. Concurrent writers of the same
    /// bytes race benignly: both write temp files, both rename onto the same
    /// final path.
    pub fn put_segment(&self, bytes: &[u8]) -> Result<(String, SegmentPut)> {
        let digest = segment_digest(bytes);
        let path = self.segment_path(&digest);
        if path.exists() {
            return Ok((
                digest,
                SegmentPut {
                    new: false,
                    bytes: bytes.len() as u64,
                },
            ));
        }
        write_atomically(&path, bytes)?;
        Ok((
            digest,
            SegmentPut {
                new: true,
                bytes: bytes.len() as u64,
            },
        ))
    }

    pub fn has_segment(&self, digest: &str) -> bool {
        self.segment_path(digest).exists()
    }

    /// Read one segment, verifying its content digest.
    pub fn read_segment(&self, digest: &str) -> Result<Vec<u8>> {
        let path = self.segment_path(digest);
        let bytes = fs::read(&path).with_context(|| format!("failed to read segment {digest}"))?;
        if segment_digest(&bytes) != digest {
            bail!("segment {digest} failed digest verification on read");
        }
        Ok(bytes)
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
        let serialized =
            serde_json::to_vec_pretty(manifest).context("failed to serialize manifest")?;
        write_atomically(&self.manifest_path(&manifest.payload_digest), &serialized)?;
        if self.budget_bytes > 0 {
            self.enforce_budget()?;
        }
        Ok(())
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
        let mut total = 0u64;
        for entry in fs::read_dir(self.root.join(SEGMENT_DIR))? {
            total = total.saturating_add(entry?.metadata()?.len());
        }
        Ok(total)
    }

    /// Evict oldest manifests (never the newest) and collect unreferenced
    /// segments until the segment footprint fits the budget. Returns bytes
    /// freed.
    pub fn enforce_budget(&self) -> Result<u64> {
        if self.budget_bytes == 0 {
            return Ok(0);
        }
        let mut freed = 0u64;
        loop {
            if self.segment_footprint_bytes()? <= self.budget_bytes {
                break;
            }
            let manifests = self.list_manifests()?;
            if manifests.len() <= 1 {
                // Never evict the newest manifest: the state just committed
                // must stay loadable even when it alone exceeds the budget.
                break;
            }
            let oldest = manifests.last().expect("len checked above").clone();
            fs::remove_file(self.manifest_path(&oldest))
                .with_context(|| format!("failed to evict manifest {oldest}"))?;
            freed = freed.saturating_add(self.collect_unreferenced_segments()?);
        }
        Ok(freed)
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
            if !referenced.contains(&stem) {
                freed = freed.saturating_add(entry.metadata()?.len());
                fs::remove_file(&path)
                    .with_context(|| format!("failed to collect segment {stem}"))?;
            }
        }
        Ok(freed)
    }
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

    fn manifest_for(
        store: &HandoffSegmentStore,
        payload: &[u8],
        segment_bytes: usize,
    ) -> HandoffManifest {
        let mut manifest = HandoffManifest::new("blake3:test".to_string(), "full-state".into());
        for (index, chunk) in payload.chunks(segment_bytes).enumerate() {
            let (digest, _) = store.put_segment(chunk).expect("put segment");
            manifest.segments.push(HandoffSegmentRef {
                index: index as u32,
                offset: (index * segment_bytes) as u64,
                bytes: chunk.len() as u64,
                digest,
                meta_json: None,
            });
        }
        manifest.total_bytes = payload.len() as u64;
        manifest.payload_digest = segment_digest(payload);
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
        let manifest = manifest_for(&store, &payload, 4096);
        store.commit(&manifest).expect("commit");
        let loaded = store
            .load_manifest(&manifest.payload_digest)
            .expect("load manifest");
        assert_eq!(store.assemble(&loaded).expect("assemble"), payload);
    }

    #[test]
    fn puts_are_idempotent_and_deduplicated() {
        let root = temp_root("idempotent");
        let store = store(&root, 0);
        let (first_digest, first) = store.put_segment(b"same bytes").expect("first put");
        let (second_digest, second) = store.put_segment(b"same bytes").expect("second put");
        assert_eq!(first_digest, second_digest);
        assert!(first.new);
        assert!(!second.new);
        assert_eq!(store.segment_footprint_bytes().expect("footprint"), 10);
    }

    #[test]
    fn commit_rejects_missing_segments_and_bad_tiling() {
        let root = temp_root("completeness");
        let store = store(&root, 0);
        let payload = vec![7u8; 10_000];
        let mut manifest = manifest_for(&store, &payload, 4096);

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
        let manifest = manifest_for(&store, &payload, 4096);
        store.commit(&manifest).expect("commit");

        let victim = store.segment_path(&manifest.segments[0].digest);
        let mut bytes = fs::read(&victim).expect("read segment file");
        bytes[0] ^= 0xFF;
        fs::write(&victim, bytes).expect("corrupt segment file");

        assert!(store.assemble(&manifest).is_err());
    }

    #[test]
    fn budget_evicts_oldest_manifest_but_never_the_newest() {
        let root = temp_root("budget");
        // Budget fits one payload but not two.
        let store = store(&root, 12_000);
        let old_payload = vec![1u8; 8_000];
        let new_payload = vec![2u8; 8_000];
        let old_manifest = manifest_for(&store, &old_payload, 4096);
        store.commit(&old_manifest).expect("commit old");
        // Ensure a later mtime for the second manifest.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let new_manifest = manifest_for(&store, &new_payload, 4096);
        store.commit(&new_manifest).expect("commit new");

        let manifests = store.list_manifests().expect("list");
        assert_eq!(manifests, vec![new_manifest.payload_digest.clone()]);
        assert!(store.assemble(&new_manifest).is_ok());
        assert!(store.segment_footprint_bytes().expect("footprint") <= 12_000);
    }

    #[test]
    fn unreferenced_segments_are_collected() {
        let root = temp_root("gc");
        let store = store(&root, 0);
        store.put_segment(b"orphan bytes").expect("orphan put");
        let payload = vec![9u8; 4096];
        let manifest = manifest_for(&store, &payload, 4096);
        store.commit(&manifest).expect("commit");

        let freed = store.collect_unreferenced_segments().expect("collect");
        assert_eq!(freed, 12);
        assert!(store.assemble(&manifest).is_ok());
    }
}
